#!/usr/bin/env python3
"""Run a GPU job under hard memory limits and a thermal governor so it cannot take the machine down.

On a DGX Spark (GB10) the GPU shares the 128 GB of system memory. A process that
over-allocates on the GPU is not charged to its cgroup and is not OOM-killed; the
machine hangs instead. Three independent limits close that hole:

1. cgroup cap on CPU-side memory      systemd-run --scope -p MemoryMax=... (kernel enforced)
2. PyTorch allocator cap on the GPU   JEZ_CUDA_MEMORY_FRACTION -> torch.cuda.set_per_process_memory_fraction,
                                      applied by jez.engine.torch_backend; over-allocation raises OutOfMemoryError
3. a watchdog on MemAvailable         this script kills the whole process group when free memory
                                      drops below --floor-gb, whatever allocated it

The same machine powers itself off under sustained GPU load (twice on 2026-10-16, both within minutes of a
model heating the SoC to 97°C with the GPU two degrees from its slowdown point and seven from its shutdown
point). Capping the clocks needs root, so a fourth limit works from user space:

4. a thermal governor                 the job is paused (SIGSTOP on its process group) when the GPU is within
                                      --min-tlimit degrees of its slowdown point, at or above --max-gpu-temp,
                                      or any SoC thermal zone is at or above --max-soc-temp; it resumes
                                      (SIGCONT) once all three have cooled by --resume-margin. Paused time
                                      does not count against --timeout; a pause longer than --max-pause
                                      kills the job (exit 125).

Usage:
    tools/gpu_guard.py [--cpu-cap-gb 32] [--cuda-fraction 0.5] [--floor-gb 24] [--timeout 3600]
                       [--max-gpu-temp 84] [--max-soc-temp 90] [--min-tlimit 6] -- cmd args...

Exit codes: the child's, or 137 when the memory watchdog killed it, 124 on timeout, 125 on a thermal kill.
Telemetry (power, temperatures, clocks, MemAvailable, paused flag, fan state) is appended to an fsync'd CSV every
--telemetry-period seconds so the moment of a silent freeze can be read back after a power cycle.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass


def mem_available_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    raise RuntimeError("MemAvailable not found")


def mem_total_gb() -> float:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1e6
    raise RuntimeError("MemTotal not found")


def _num(text: str) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Sample:
    seq: int
    gpu_temp: float | None      # temperature.gpu, °C
    tlimit: float | None        # temperature.gpu.tlimit: degrees of margin left before the GPU slows down
    soc_temp: float | None      # hottest /sys/class/thermal zone, °C
    power: float | None


def should_pause(s: Sample, max_gpu_temp: float, max_soc_temp: float, min_tlimit: float) -> bool:
    return ((s.tlimit is not None and s.tlimit <= min_tlimit)
            or (s.gpu_temp is not None and s.gpu_temp >= max_gpu_temp)
            or (s.soc_temp is not None and s.soc_temp >= max_soc_temp))


def can_resume(s: Sample, max_gpu_temp: float, max_soc_temp: float, min_tlimit: float, margin: float) -> bool:
    return ((s.tlimit is None or s.tlimit >= min_tlimit + margin)
            and (s.gpu_temp is None or s.gpu_temp <= max_gpu_temp - margin)
            and (s.soc_temp is None or s.soc_temp <= max_soc_temp - margin))


def soc_temp_max(zones: list[str]) -> float | None:
    temps = []
    for z in zones:
        try:
            with open(z) as f:
                temps.append(int(f.read().strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    return max(temps) if temps else None


class BlackBox(threading.Thread):
    """Sample the GPU and the SoC every ``period`` seconds, keep the latest sample, and append it to a CSV, fsync'd.

    The DGX Spark can freeze or power off silently under sustained GPU load, leaving nothing in the journal.
    This recorder leaves the last samples on disk so the moment of death can be read back after a power cycle.
    """

    FIELDS = ("timestamp,power.draw,temperature.gpu,temperature.gpu.tlimit,clocks.sm,clocks.mem,utilization.gpu,"
              "clocks_event_reasons.active")

    def __init__(self, path: str, period: float = 2.0) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.period = period
        self.stop = threading.Event()
        self.paused = False
        self.sample: Sample | None = None
        self.last = ""
        self.zones = sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp"))
        self.fans = sorted(glob.glob("/sys/class/hwmon/hwmon*/fan*_input"))  # ACPI fan state, read-only from user space

    def run(self) -> None:
        smi = shutil.which("nvidia-smi")
        seq = 0
        with open(self.path, "a", buffering=1) as f:
            f.write("time," + self.FIELDS + ",soc_temp_max,mem_available_gb,paused,fan\n")
            while not self.stop.is_set():
                gpu = ""
                if smi:
                    try:
                        gpu = subprocess.run([smi, f"--query-gpu={self.FIELDS}", "--format=csv,noheader,nounits"],
                                             capture_output=True, text=True, timeout=5).stdout.strip().replace(", ", ",")
                    except Exception:  # noqa: BLE001
                        gpu = "nvidia-smi-failed"
                parts = gpu.split(",")
                soc = soc_temp_max(self.zones)
                seq += 1
                self.sample = Sample(seq, _num(parts[2]) if len(parts) > 2 else None, _num(parts[3]) if len(parts) > 3 else None,
                                     soc, _num(parts[1]) if len(parts) > 1 else None)
                fan = ""
                for path in self.fans:
                    try:
                        with open(path) as fh:
                            fan = fh.read().strip()
                        break
                    except OSError:
                        continue
                self.last = (f"{dt.datetime.now().isoformat(timespec='seconds')},{gpu},"
                             f"{'' if soc is None else f'{soc:.1f}'},{mem_available_gb():.1f},{int(self.paused)},{fan}")
                f.write(self.last + "\n")
                f.flush()
                os.fsync(f.fileno())
                self.stop.wait(self.period)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cpu-cap-gb", type=float, default=32.0, help="cgroup MemoryMax for CPU-side memory")
    ap.add_argument("--cuda-fraction", type=float, default=0.5, help="share of device memory PyTorch may allocate")
    ap.add_argument("--floor-gb", type=float, default=24.0, help="kill the job when MemAvailable drops below this")
    ap.add_argument("--timeout", type=float, default=3600.0, help="limit on running (not paused) time in seconds")
    ap.add_argument("--poll", type=float, default=0.25, help="watchdog interval in seconds")
    ap.add_argument("--no-scope", action="store_true", help="do not wrap in a systemd scope (no cgroup cap)")
    ap.add_argument("--telemetry", default="runs/telemetry", help="directory for the black-box CSV (empty string disables)")
    ap.add_argument("--telemetry-period", type=float, default=2.0)
    ap.add_argument("--max-gpu-temp", type=float, default=84.0, help="pause the job at this GPU temperature (°C)")
    ap.add_argument("--max-soc-temp", type=float, default=90.0, help="pause the job when any SoC thermal zone reaches this (°C)")
    ap.add_argument("--min-tlimit", type=float, default=6.0, help="pause the job when the GPU is this close to its slowdown point (°C)")
    ap.add_argument("--resume-margin", type=float, default=6.0, help="resume once every reading has cooled this far past its limit")
    ap.add_argument("--max-pause", type=float, default=600.0, help="kill the job if one thermal pause lasts longer than this (s)")
    ap.add_argument("--no-thermal", action="store_true", help="disable the thermal governor")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd and args.cmd[0] == "--" else args.cmd
    if not cmd:
        ap.error("no command given")

    avail = mem_available_gb()
    if avail < args.floor_gb + 8:
        print(f"gpu_guard: only {avail:.1f} GB available; need at least {args.floor_gb + 8:.0f} GB to start", file=sys.stderr)
        return 3

    env = dict(os.environ)
    env.setdefault("JEZ_CUDA_MEMORY_FRACTION", str(args.cuda_fraction))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("JEZ_GPU_GUARD", "1")
    # Triton compiles a small C helper against the running Python's headers. The system Python here has no
    # python3-dev package, so the matching headers were unpacked into the venv (see README, "Fast kernels").
    include = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".venv", "include", "python3.12")
    if os.path.isdir(include):
        env["CPATH"] = include + (":" + env["CPATH"] if env.get("CPATH") else "")

    wrapper: list[str] = []
    if not args.no_scope and shutil.which("systemd-run"):
        wrapper = ["systemd-run", "--user", "--scope", "-q", "-p", f"MemoryMax={int(args.cpu_cap_gb)}G",
                   "-p", "MemorySwapMax=0", "--"]
    elif not args.no_scope:
        print("gpu_guard: systemd-run not found; running without a cgroup cap", file=sys.stderr)

    print(f"gpu_guard: MemAvailable {avail:.1f}/{mem_total_gb():.0f} GB | cpu cap {args.cpu_cap_gb:.0f} GB | "
          f"cuda fraction {args.cuda_fraction} | floor {args.floor_gb:.0f} GB | timeout {args.timeout:.0f}s | "
          f"thermal {'off' if args.no_thermal else f'gpu<{args.max_gpu_temp:.0f}°C soc<{args.max_soc_temp:.0f}°C tlimit>{args.min_tlimit:.0f}°C'}",
          file=sys.stderr)
    box = None
    if args.telemetry or not args.no_thermal:
        os.makedirs(args.telemetry or "runs/telemetry", exist_ok=True)
        box = BlackBox(os.path.join(args.telemetry or "runs/telemetry", dt.datetime.now().strftime("%Y%m%d-%H%M%S") + ".csv"),
                       args.telemetry_period)
        box.start()
        print(f"gpu_guard: telemetry -> {box.path}", file=sys.stderr)

    class Terminated(Exception):
        pass

    def on_term(signum, frame):  # noqa: ARG001
        raise Terminated()

    signal.signal(signal.SIGTERM, on_term)
    proc = subprocess.Popen(wrapper + cmd, env=env, start_new_session=True)
    start = time.monotonic()
    low = avail
    reason = None
    paused_at: float | None = None
    paused_total = 0.0
    pauses = 0
    seen = 0
    try:
        while proc.poll() is None:
            now_mono = time.monotonic()
            now = mem_available_gb()
            low = min(low, now)
            if now < args.floor_gb:
                reason = f"MemAvailable fell to {now:.1f} GB (floor {args.floor_gb:.0f} GB)"
                break
            s = box.sample if box is not None else None
            if not args.no_thermal and s is not None and s.seq != seen:
                seen = s.seq
                state = (f"gpu {s.gpu_temp:.0f}°C tlimit {s.tlimit:.0f}°C soc {s.soc_temp:.0f}°C {s.power:.0f} W"
                         if None not in (s.gpu_temp, s.tlimit, s.soc_temp, s.power) else str(s))
                if paused_at is None and should_pause(s, args.max_gpu_temp, args.max_soc_temp, args.min_tlimit):
                    os.killpg(proc.pid, signal.SIGSTOP)
                    paused_at = now_mono
                    pauses += 1
                    box.paused = True
                    print(f"gpu_guard: PAUSE #{pauses} ({state}) after {now_mono - start - paused_total:.0f}s of running", file=sys.stderr)
                elif paused_at is not None and can_resume(s, args.max_gpu_temp, args.max_soc_temp, args.min_tlimit, args.resume_margin):
                    os.killpg(proc.pid, signal.SIGCONT)
                    paused_total += now_mono - paused_at
                    paused_at = None
                    box.paused = False
                    wall = now_mono - start
                    print(f"gpu_guard: RESUME ({state}); paused {paused_total:.0f}s of {wall:.0f}s, duty {100 * (1 - paused_total / wall):.0f}%",
                          file=sys.stderr)
            if paused_at is not None and now_mono - paused_at > args.max_pause:
                reason = f"thermal pause longer than {args.max_pause:.0f}s ({s})"
                break
            running = now_mono - start - paused_total - (now_mono - paused_at if paused_at is not None else 0.0)
            if running > args.timeout:
                reason = f"timeout after {args.timeout:.0f}s of running time"
                break
            time.sleep(args.poll)
    except KeyboardInterrupt:
        reason = "interrupted"
    except Terminated:
        reason = "terminated"
    if box is not None:
        box.stop.set()
    wall = time.monotonic() - start
    if paused_at is not None:
        paused_total += wall - (paused_at - start)
    summary = f"lowest MemAvailable {low:.1f} GB; {pauses} thermal pauses, {paused_total:.0f}s paused of {wall:.0f}s"
    if reason is not None:
        print(f"gpu_guard: KILLING job: {reason}", file=sys.stderr)
        for sig in (signal.SIGCONT, signal.SIGKILL):
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass
        proc.wait(timeout=30)
        print(f"gpu_guard: {summary}", file=sys.stderr)
        return 124 if reason.startswith("timeout") else 125 if reason.startswith("thermal") else 137
    print(f"gpu_guard: job exited {proc.returncode} after {wall:.0f}s; {summary}", file=sys.stderr)
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
