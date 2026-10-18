"""FastAPI service implementing Jev's contract.

    POST /v1/systemone   evaluate a state against named questions
    GET  /v1/models      list the model ids and aliases this server answers for
    GET  /health         liveness (outside Jev's contract)

Environment:

    JEZ_MODELS         comma list of ``id`` or ``id=checkpoint`` (default ``jez-small-1.0.0``)
    JEZ_BACKEND        auto | torch | mlx | fake (default auto)
    JEZ_API_KEYS       comma list of accepted bearer tokens; JEZ_NO_AUTH=1 or --no-auth skips the check
    JEZ_MAX_QUEUE      requests allowed in flight before 429 (default 32)
    JEZ_MAX_BATCH      probes per forward pass (default 16)
    JEZ_CALIBRATION    path pattern with ``{model}`` (default ``releases/{model}/calibration.json``)
    JEZ_HOST, JEZ_PORT (default 127.0.0.1:8009)
"""
from __future__ import annotations

import argparse
import hmac
import json
import logging
import os
import threading
import time
import zlib
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, Response

from . import __version__
from . import models as registry
from .calibration import Calibration
from .decider import Decider
from .engine import load_backend
from .schema import JezError, parse_json_body, validate_request

log = logging.getLogger("jez")

MAX_BODY = 32 * 1024 * 1024  # 32 MiB, compressed or not
ZSTD_MAX_WINDOW = 8 * 1024 * 1024


class Settings:
    def __init__(self) -> None:
        self.models = os.environ.get("JEZ_MODELS", "jez-small-1.0.0")
        self.backend = os.environ.get("JEZ_BACKEND", "auto")
        self.api_keys = [k.strip() for k in os.environ.get("JEZ_API_KEYS", "").split(",") if k.strip()]
        self.no_auth = os.environ.get("JEZ_NO_AUTH", "") in ("1", "true", "yes")
        self.max_queue = int(os.environ.get("JEZ_MAX_QUEUE", "32"))
        self.max_batch = int(os.environ.get("JEZ_MAX_BATCH", "16"))
        self.calibration = os.environ.get("JEZ_CALIBRATION", "releases/{model}/calibration.json")
        self.debias = os.environ.get("JEZ_DEBIAS", "1") not in ("0", "false", "no")


class Service:
    """Loaded models plus the shared compute lock and queue counter."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.deciders: dict[str, Decider] = {}
        self.lock = threading.Lock()
        self.in_flight = 0
        self.count_lock = threading.Lock()
        if not settings.no_auth and not settings.api_keys:
            raise RuntimeError("set JEZ_API_KEYS to a comma-separated list of tokens, or start with --no-auth / JEZ_NO_AUTH=1")

    def load(self) -> None:
        for model_id, checkpoint in registry.parse_model_env(self.settings.models):
            backend = load_backend(checkpoint, self.settings.backend, max_batch=self.settings.max_batch)
            calib = Calibration.load(self.settings.calibration.format(model=model_id))
            if calib.model and calib.model not in (model_id, checkpoint):
                log.warning("calibration for %s was fitted for %s; using T=1", model_id, calib.model)
                calib = Calibration()
            if hasattr(backend, "warmup"):
                backend.warmup()
            self.deciders[model_id] = Decider(backend, model_id, calibration=calib, debias=self.settings.debias)
            log.info("serving %s from %s on %s", model_id, checkpoint, backend.name)

    def authorized(self, request: Request) -> bool:
        if self.settings.no_auth:
            return True
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return False
        token = token.strip()
        return any(hmac.compare_digest(token, k) for k in self.settings.api_keys)

    def models_body(self) -> dict[str, Any]:
        rows = []
        for model_id in self.deciders:
            spec = registry.MODELS[model_id]
            rows.append({"name": model_id, "description": spec.description, "release_date": spec.release_date})
            for alias in registry.aliases_for(model_id):
                rows.append({"name": alias, "description": f"Alias of {model_id}. {spec.description}",
                             "release_date": spec.release_date})
        return {"models": rows}


# ----------------------------------------------------------------------------- body handling

async def read_body(request: Request) -> bytes:
    length = request.headers.get("content-length")
    if length is not None:
        try:
            if int(length) > MAX_BODY:
                raise JezError(413, "request_too_large", f"the body is larger than {MAX_BODY} bytes")
        except ValueError:
            raise JezError(400, "invalid_request", "content-length is not an integer")
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > MAX_BODY:
            raise JezError(413, "request_too_large", f"the body is larger than {MAX_BODY} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


def _gunzip(data: bytes) -> bytes:
    d = zlib.decompressobj(wbits=31)
    try:
        out = d.decompress(data, MAX_BODY + 1)
    except zlib.error as e:
        raise JezError(400, "invalid_encoding", f"the gzip body is corrupt: {e}")
    if len(out) > MAX_BODY or d.unconsumed_tail:
        raise JezError(413, "request_too_large", f"the decompressed body is larger than {MAX_BODY} bytes")
    if not d.eof:
        raise JezError(400, "invalid_encoding", "the gzip body is truncated")
    return out


def _unzstd(data: bytes) -> bytes:
    """Decompress one zstd frame with the window capped at 8 MiB and the output capped at MAX_BODY.

    Only ``decompressobj`` enforces ``max_window_size`` in python-zstandard, so the input is fed to it in
    small chunks: a 1 KiB chunk can expand to at most 32 MiB, which keeps every step bounded.
    """
    import zstandard

    try:
        dobj = zstandard.ZstdDecompressor(max_window_size=ZSTD_MAX_WINDOW).decompressobj()
        out = bytearray()
        for i in range(0, len(data), 1024):
            out += dobj.decompress(data[i:i + 1024])
            if len(out) > MAX_BODY:
                raise JezError(413, "request_too_large", f"the decompressed body is larger than {MAX_BODY} bytes")
        if not dobj.eof:
            raise JezError(400, "invalid_encoding", "the zstd body is truncated")
    except zstandard.ZstdError as e:
        msg = str(e)
        if "memory" in msg.lower() or "window" in msg.lower():
            raise JezError(400, "invalid_encoding", f"the zstd window exceeds {ZSTD_MAX_WINDOW} bytes: {msg}")
        raise JezError(400, "invalid_encoding", f"the zstd body is corrupt: {msg}")
    return bytes(out)


def decode_body(request: Request, body: bytes) -> bytes:
    encoding = request.headers.get("content-encoding", "").strip().lower()
    if not encoding or encoding == "identity":
        return body
    layers = [p.strip() for p in encoding.split(",") if p.strip()]
    if len(layers) != 1:
        raise JezError(415, "unsupported_encoding", f"content-encoding {encoding!r} stacks {len(layers)} encodings; send one layer")
    layer = layers[0]
    if layer in ("gzip", "x-gzip"):
        return _gunzip(body)
    if layer == "zstd":
        return _unzstd(body)
    raise JezError(415, "unsupported_encoding", f"content-encoding {layer!r} is not supported; use gzip, x-gzip or zstd")


def _json_bytes(data: Any) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def error_response(err: JezError) -> Response:
    return Response(content=_json_bytes(err.body()), status_code=err.status, media_type="application/json",
                    headers=err.headers)


# ----------------------------------------------------------------------------- app

def create_app(settings: Settings | None = None, service: Service | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        svc = service or Service(settings)
        if not svc.deciders:
            svc.load()
        app.state.service = svc
        log.info("ready: %s", ", ".join(svc.deciders))
        yield

    app = FastAPI(title="jez", version=__version__, lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(JezError)
    async def _jez_error(_: Request, err: JezError) -> Response:
        return error_response(err)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> Response:
        if _is_oom(exc):
            log.error("out of memory: %s", exc)
            return error_response(JezError(529, "overloaded", "the model ran out of memory; retry after a short delay",
                                           headers={"retry-after": "2"}))
        log.exception("unhandled error")
        return error_response(JezError(500, "internal_error", f"{type(exc).__name__}: {exc}"))

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> Response:
        svc: Service = app.state.service
        if not svc.authorized(request):
            raise JezError(401, "authentication_error", "missing or invalid bearer token")
        body = decode_body(request, await read_body(request))
        req = validate_request(parse_json_body(body))
        decider = svc.deciders.get(req.model_id)
        if decider is None:
            raise JezError(529, "model_not_loaded", f"{req.model_id} is not loaded on this server", field="model",
                           headers={"retry-after": "5"})
        with svc.count_lock:
            if svc.in_flight >= svc.settings.max_queue:
                raise JezError(429, "rate_limited", f"more than {svc.settings.max_queue} requests are queued",
                               headers={"retry-after": "1"})
            svc.in_flight += 1
        t0 = time.perf_counter()
        try:
            def run():
                with svc.lock:
                    return decider.decide(req)

            decision = await run_in_threadpool(run)
        except JezError:
            raise
        except Exception as exc:  # noqa: BLE001 - every failure must become a Jev-shaped error
            if _is_oom(exc):
                log.error("out of memory: %s", exc)
                _release_cuda_cache()
                raise JezError(529, "overloaded", "the model ran out of memory; retry after a short delay",
                               headers={"retry-after": "2"})
            log.exception("decision failed")
            raise JezError(500, "internal_error", f"{type(exc).__name__}: {exc}")
        finally:
            with svc.count_lock:
                svc.in_flight -= 1
        latency_ms = (time.perf_counter() - t0) * 1000
        spec = registry.MODELS[req.model_id]
        cost = decision.input_tokens / 1e6 * spec.cost_per_mtok_usd
        headers = {"Jez-Latency-Ms": f"{latency_ms:.1f}", "Jez-Cost-Usd": f"{cost:.8f}",
                   "Jez-Request-Encoding": request.headers.get("content-encoding", "identity").strip().lower() or "identity"}
        log.info("systemone model=%s questions=%d probes=%d tokens=%d %.0fms", req.model_id, len(req.questions),
                 decision.probes, decision.input_tokens, latency_ms)
        return Response(content=_json_bytes(decision.body()), media_type="application/json", headers=headers)

    @app.get("/v1/models")
    async def models(request: Request) -> Response:
        svc: Service = app.state.service
        if not svc.authorized(request):
            raise JezError(401, "authentication_error", "missing or invalid bearer token")
        return Response(content=_json_bytes(svc.models_body()), media_type="application/json")

    @app.get("/health")
    async def health() -> JSONResponse:
        svc: Service = app.state.service
        return JSONResponse({
            "status": "ok", "version": __version__,
            "models": {mid: {"checkpoint": d.backend.checkpoint, "backend": d.backend.name, "device": d.backend.device,
                             "temperature": d.calibration.temperature} for mid, d in svc.deciders.items()},
            "in_flight": svc.in_flight,
        })

    return app


def _is_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    return "OutOfMemory" in name or "out of memory" in str(exc).lower() or isinstance(exc, MemoryError)


def _release_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


app = create_app()


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description="Serve Jez behind Jev's HTTP contract.")
    ap.add_argument("--host", default=os.environ.get("JEZ_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("JEZ_PORT", "8009")))
    ap.add_argument("--models", help="comma list of id or id=checkpoint (JEZ_MODELS)")
    ap.add_argument("--backend", choices=["auto", "torch", "mlx", "fake"], help="JEZ_BACKEND")
    ap.add_argument("--no-auth", action="store_true", help="skip the bearer-token check (local work only)")
    ap.add_argument("--max-queue", type=int, help="JEZ_MAX_QUEUE")
    args = ap.parse_args()
    if args.models:
        os.environ["JEZ_MODELS"] = args.models
    if args.backend:
        os.environ["JEZ_BACKEND"] = args.backend
    if args.no_auth:
        os.environ["JEZ_NO_AUTH"] = "1"
    if args.max_queue is not None:
        os.environ["JEZ_MAX_QUEUE"] = str(args.max_queue)
    uvicorn.run(create_app(Settings()), host=args.host, port=args.port, workers=1, log_level="info")


if __name__ == "__main__":
    main()
