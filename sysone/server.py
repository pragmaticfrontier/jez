"""FastAPI service: POST /v1/decide, GET /health, GET / (playground)."""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse

from . import __version__
from .calibration import Calibration
from .decider import Decider
from .engine import DEFAULT_MODEL, Scorer
from .schema import DecideRequest, DecideResponse

log = logging.getLogger("sysone")

_lock = threading.Lock()


def build_decider() -> Decider:
    model_id = os.environ.get("SYSONE_MODEL", DEFAULT_MODEL)
    device = os.environ.get("SYSONE_DEVICE") or None
    calib_path = os.environ.get("SYSONE_CALIBRATION", "calibration.json")
    calibration = Calibration.load(calib_path)
    if calibration.model and calibration.model != model_id:
        log.warning("calibration file %s was fitted for %s, not %s; using T=1", calib_path, calibration.model, model_id)
        calibration = Calibration()
    scorer = Scorer(model_id, device=device)
    scorer.warmup()
    return Decider(
        scorer,
        calibration=calibration,
        max_state_tokens=int(os.environ.get("SYSONE_MAX_STATE_TOKENS", "3000")),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app.state.decider = build_decider()
    log.info("ready: model=%s temperature=%s", app.state.decider.scorer.model_id, app.state.decider.calibration.temperature)
    yield


app = FastAPI(
    title="sysone",
    version=__version__,
    description="System-One style decision model: typed questions in, calibrated probabilities out.",
    lifespan=lifespan,
)


@app.post("/v1/decide", response_model=DecideResponse, response_model_exclude_none=True)
async def decide(req: DecideRequest) -> DecideResponse:
    decider: Decider = app.state.decider

    def run() -> DecideResponse:
        with _lock:  # one model, one GPU: serialize forward passes
            return decider.decide(req)

    try:
        return await run_in_threadpool(run)
    except ValueError as e:  # e.g. a label that is not a single token
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/health")
async def health() -> JSONResponse:
    d: Decider = app.state.decider
    return JSONResponse({
        "status": "ok",
        "model": d.scorer.model_id,
        "device": d.scorer.device,
        "dtype": str(d.scorer.dtype).replace("torch.", ""),
        "temperature": d.calibration.temperature,
        "version": __version__,
    })


@app.get("/v1/schema")
async def schema() -> JSONResponse:
    return JSONResponse({"request": DecideRequest.model_json_schema(), "response": DecideResponse.model_json_schema()})


@app.get("/", response_class=HTMLResponse)
async def playground() -> str:
    return PLAYGROUND_HTML


PLAYGROUND_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>sysone playground</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1a1d23;--muted:#6b7280;--line:#e3e6ea;--accent:#2f6fed;--bar:#dfe7fb;--ok:#15803d;--warn:#b45309}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a21;--ink:#e6e8ec;--muted:#9aa3b2;--line:#2a2f3a;--accent:#6b9bff;--bar:#243253;--ok:#4ade80;--warn:#fbbf24}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line);display:flex;gap:14px;align-items:baseline;flex-wrap:wrap}
header h1{font-size:16px;margin:0;font-weight:650}header .meta{color:var(--muted);font-size:12.5px}
main{display:grid;grid-template-columns:minmax(320px,1fr) minmax(320px,1fr);gap:18px;padding:18px 20px;max-width:1300px;margin:0 auto}
@media (max-width:860px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
label{display:block;font-weight:600;font-size:12.5px;margin:10px 0 4px;color:var(--muted);text-transform:uppercase;letter-spacing:.03em}
textarea{width:100%;border:1px solid var(--line);border-radius:8px;padding:9px 10px;font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:transparent;color:var(--ink);resize:vertical}
.row{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-top:12px}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:9px 16px;font-weight:600;cursor:pointer}button:disabled{opacity:.6;cursor:wait}
input[type=number]{width:70px;border:1px solid var(--line);border-radius:6px;padding:5px 7px;background:transparent;color:var(--ink)}
.ans{border-top:1px solid var(--line);padding:12px 0}.ans:first-child{border-top:0;padding-top:0}
.ans .head{display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap}
.ans .name{font-weight:650}.ans .type{color:var(--muted);font-size:12px;margin-left:6px}
.ans .pick{font-weight:650;color:var(--accent)}.conf{font-size:12.5px;color:var(--muted)}
.bars{margin-top:8px;display:grid;grid-template-columns:max-content 1fr max-content;gap:4px 10px;align-items:center;font-size:12.5px}
.bar{height:10px;background:var(--bar);border-radius:5px;overflow:hidden}.bar i{display:block;height:100%;background:var(--accent)}
.bar.win i{background:var(--ok)}
.foot{color:var(--muted);font-size:12.5px;margin-top:12px;display:flex;gap:14px;flex-wrap:wrap}
.err{color:#dc2626;white-space:pre-wrap;font-family:ui-monospace,monospace;font-size:12.5px}
.hint{color:var(--muted);font-size:12.5px;margin-top:6px}
code{font-family:ui-monospace,monospace;font-size:12px;background:var(--bar);padding:1px 5px;border-radius:4px}
</style></head><body>
<header><h1>sysone · decision playground</h1><span class="meta" id="meta">loading model info…</span></header>
<main>
<section class="card">
  <label for="state">State</label>
  <textarea id="state" rows="6">My card was charged twice for the monthly subscription. This is the third time this year and I'm done with you people. Fix it today or I'm cancelling.</textarea>
  <label for="questions">Questions (JSON)</label>
  <textarea id="questions" rows="22">{
  "routing": {
    "type": "choice",
    "instructions": "Which department should handle this ticket?",
    "criteria": {
      "billing": "Charges, invoices, and refunds",
      "technical": "Software bugs and service outages",
      "account": "Logins, passwords, and security"
    }
  },
  "is_urgent": {
    "type": "bool",
    "instructions": "Does the customer demand same-day action?"
  },
  "frustration": {
    "type": "score",
    "instructions": "How frustrated is the customer?",
    "min": 1, "max": 5,
    "criteria": {"1": "calm and polite", "3": "annoyed", "5": "furious, threatening to leave"}
  }
}</textarea>
  <div class="row">
    <button id="go">Decide</button>
    <label style="margin:0;text-transform:none;letter-spacing:0"><input type="checkbox" id="debias"> debias (rotate options)</label>
    <label style="margin:0;text-transform:none;letter-spacing:0">temperature <input type="number" id="temp" step="0.1" min="0.1" placeholder="auto"></label>
  </div>
  <p class="hint">Types: <code>choice</code> (criteria: option → description), <code>bool</code>, <code>score</code> (min, max, optional rubric). State may also be a JSON object.</p>
</section>
<section class="card" id="out"><p class="hint">Answers appear here. Nothing is generated: each question is one read of the model's next-token logits over its allowed answers.</p></section>
</main>
<script>
const $=id=>document.getElementById(id);
fetch('/health').then(r=>r.json()).then(h=>{$('meta').textContent=`${h.model} · ${h.device} ${h.dtype} · T=${JSON.stringify(h.temperature)}`}).catch(()=>{$('meta').textContent='server not ready'});
function pct(p){return (100*p).toFixed(1)+'%'}
function render(res){
  const out=$('out'); out.innerHTML='';
  for(const [name,a] of Object.entries(res.answers)){
    const div=document.createElement('div'); div.className='ans';
    let pick, probs;
    if(a.type==='choice'){pick=a.choice; probs=a.probabilities}
    else if(a.type==='bool'){pick=String(a.value); probs={true:a.probability,false:1-a.probability}}
    else {pick=`${a.score} (expected ${a.expected})`; probs=a.probabilities}
    const flag=a.confidence>=0.9?'':` <span class="conf" style="color:var(--warn)">· low confidence → escalate</span>`;
    let html=`<div class="head"><div><span class="name">${name}</span><span class="type">${a.type}</span></div><div><span class="pick">${pick}</span> <span class="conf">confidence ${pct(a.confidence)}</span>${flag}</div></div><div class="bars">`;
    const win=Object.entries(probs).sort((x,y)=>y[1]-x[1])[0][0];
    for(const [k,p] of Object.entries(probs)){html+=`<span>${k}</span><div class="bar ${k===win?'win':''}"><i style="width:${Math.max(0.5,100*p)}%"></i></div><span>${pct(p)}</span>`}
    div.innerHTML=html+'</div>'; out.appendChild(div);
  }
  const u=res.usage; const f=document.createElement('div'); f.className='foot';
  f.innerHTML=`<span><b>${res.latency_ms} ms</b> server latency</span><span>${u.input_tokens} input tokens</span><span>${u.output_tokens} output tokens</span><span>${u.probes} probes in ${u.forward_passes} forward pass${u.forward_passes>1?'es':''}</span><span>${res.model}</span>`;
  out.appendChild(f);
}
$('go').onclick=async()=>{
  const btn=$('go'); btn.disabled=true;
  try{
    let state=$('state').value; try{const j=JSON.parse(state); if(typeof j==='object'&&j!==null) state=j}catch(e){}
    const questions=JSON.parse($('questions').value);
    const options={debias:$('debias').checked}; if($('temp').value) options.temperature=parseFloat($('temp').value);
    const r=await fetch('/v1/decide',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({state,questions,options})});
    const body=await r.json();
    if(!r.ok){$('out').innerHTML=`<div class="err">HTTP ${r.status}\n${JSON.stringify(body,null,2)}</div>`;return}
    render(body);
  }catch(e){$('out').innerHTML=`<div class="err">${e}</div>`}
  finally{btn.disabled=false}
};
</script></body></html>
"""


def main() -> None:
    import uvicorn

    uvicorn.run(
        "sysone.server:app",
        host=os.environ.get("SYSONE_HOST", "127.0.0.1"),
        port=int(os.environ.get("SYSONE_PORT", "8008")),
        workers=1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
