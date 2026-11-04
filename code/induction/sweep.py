#!/usr/bin/env python3
"""
sweep.py — run the solver across several model configs on one GPU box and summarise
accuracy vs. cost, so we can pick the model that fits Kaggle's 12 hours without giving up accuracy.

For each config in sweep_configs.json:
  1. start `vllm serve <model>` (trying the listed parser names until one is accepted)
  2. wait until the server answers, run solver.py on the chosen split, stop the server
  3. score the submission; collect tokens, turns, truncations, verifier counts, wall time
Then write <out>/summary.md, summary.json and summary.png (accuracy vs projected Kaggle hours).

  python sweep.py --challenges ../data/arc-agi_evaluation_challenges.json \
                  --solutions  ../data/arc-agi_evaluation_solutions.json \
                  --split dev --out runs/sweep1 [--only qwen36_think,gptoss20b_medium] [--limit 10]
                  [--nvarc nvarc_dev_submission.json] [--l4-tps 1500] [--budget-hours 4]

Resumable: a config whose <out>/<label>/submission.json already exists is not re-run (use --force).
Offline test (no GPU): add "server_cmd": ["python", "fake_server.py", "--port", "{port}"] to a config.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ helpers
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def server_ready(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def tail(path, n=40):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except FileNotFoundError:
        return ""


def stop(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(30):
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def score(sub, solutions, ids):
    solved = set()
    tot = 0
    for tid in ids:
        for i, truth in enumerate(solutions[tid]):
            tot += 1
            a = sub.get(tid, [])
            if i < len(a) and truth in (a[i].get("attempt_1"), a[i].get("attempt_2")):
                solved.add((tid, i))
    return solved, tot


# ------------------------------------------------------------------ server
def launch_server(cfg, port, run_dir, a):
    """Start vLLM (or a custom server_cmd). Returns the process or None. Tries parser fallbacks."""
    logf = os.path.join(run_dir, "server.log")
    if cfg.get("server_cmd"):
        cmd = [x.replace("{port}", str(port)) for x in cfg["server_cmd"]]
        return start_and_wait(cmd, logf, port, a.server_timeout, cfg)
    tool_parsers = list(cfg.get("tool_parser") or ["hermes"])
    reasoning_parsers = list(cfg.get("reasoning_parser") or [None])
    extra = list(cfg.get("vllm_extra", []))
    ti = ri = 0
    for _ in range(len(tool_parsers) + len(reasoning_parsers) + len(extra)):
        cmd = ["vllm", "serve", cfg["model"], "--port", str(port), "--served-model-name", cfg["label"],
               "--max-model-len", str(a.max_model_len), "--gpu-memory-utilization", str(a.gpu_mem),
               "--enable-auto-tool-choice", "--tool-call-parser", tool_parsers[ti]]
        if reasoning_parsers[ri]:
            cmd += ["--reasoning-parser", reasoning_parsers[ri]]
            plugin = (cfg.get("reasoning_parser_plugin") or {}).get(reasoning_parsers[ri])
            if plugin:                                  # parser shipped as a file inside the model repo
                try:
                    from huggingface_hub import hf_hub_download
                    cmd += ["--reasoning-parser-plugin", hf_hub_download(cfg["model"], plugin)]
                except Exception as e:
                    log(f"   could not fetch parser plugin {plugin}: {e}")
        if a.tp > 1:
            cmd += ["--tensor-parallel-size", str(a.tp)]
        cmd += extra
        proc = start_and_wait(cmd, logf, port, a.server_timeout, cfg)
        if proc is not None:
            return proc
        err = tail(logf, 60)
        m = re.search(r"unrecognized arguments: (--[\w-]+)", err)
        if m and m.group(1) in extra:                   # optional flag this vLLM version lacks: drop it, retry
            i = extra.index(m.group(1))
            drop = 2 if i + 1 < len(extra) and not extra[i + 1].startswith("--") else 1
            log(f"   this vLLM has no {m.group(1)}; retrying without it")
            del extra[i:i + drop]
            continue
        if "invalid choice" in err and "reasoning-parser" in err and ri + 1 < len(reasoning_parsers):
            ri += 1
            log(f"   reasoning parser not accepted; trying '{reasoning_parsers[ri]}'")
            continue
        if "invalid choice" in err and "tool-call-parser" in err and ti + 1 < len(tool_parsers):
            ti += 1
            log(f"   tool parser not accepted; trying '{tool_parsers[ti]}'")
            continue
        return None
    return None


def start_and_wait(cmd, logf, port, timeout, cfg):
    log(f"   starting: {' '.join(shlex.quote(c) for c in cmd)}")
    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    with open(logf, "a") as lf:
        lf.write(f"\n===== {time.ctime()} {' '.join(cmd)}\n")
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    t0 = time.time()
    last = 0
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            log(f"   server exited with code {proc.returncode}. Last log lines:\n{tail(logf, 25)}")
            return None
        if server_ready(port):
            log(f"   server ready after {time.time() - t0:.0f}s")
            json.dump({"startup_s": round(time.time() - t0), "cmd": cmd},
                      open(os.path.join(os.path.dirname(logf), "startup.json"), "w"))
            return proc
        if time.time() - last > 60:
            last = time.time()
            log(f"   waiting for server ({(time.time() - t0) / 60:.0f} min) ...")
        time.sleep(5)
    log("   server did not become ready in time; stopping it")
    stop(proc)
    return None


# ------------------------------------------------------------------ one config
def run_config(cfg, a, port):
    run_dir = os.path.join(a.out, cfg["label"])
    os.makedirs(run_dir, exist_ok=True)
    sub_path = os.path.join(run_dir, "submission.json")
    rep_path = os.path.join(run_dir, "report.json")
    if os.path.exists(sub_path) and os.path.exists(rep_path) and not a.force:
        log(f">> {cfg['label']}: already done, skipping (use --force to re-run)")
        return True
    if not port_free(port):
        log(f"!! port {port} is busy; is an old server still running?  (pkill -f 'vllm serve')")
        return False
    proc = launch_server(cfg, port, run_dir, a)
    if proc is None:
        log(f"!! {cfg['label']}: could not start the model server; see {run_dir}/server.log")
        return False
    cmd = [sys.executable, os.path.join(HERE, "solver.py"),
           "--challenges", a.challenges, "--split", a.split,
           "--base-url", f"http://127.0.0.1:{port}/v1", "--model", cfg["label"],
           "--samples", str(a.samples), "--rounds", str(a.rounds),
           "--parallel-tasks", str(a.parallel_tasks), "--parallel-per-task", str(a.parallel_per_task),
           "--out", run_dir, "--seed", str(a.seed)]
    if a.solutions:
        cmd += ["--solutions", a.solutions]
    if a.limit:
        cmd += ["--limit", str(a.limit)]
    if a.tasks:
        cmd += ["--tasks", a.tasks]
    if a.time_budget:
        cmd += ["--time-budget", str(a.time_budget)]
    cmd += cfg.get("solver", [])
    log(f">> {cfg['label']}: running solver: {' '.join(shlex.quote(c) for c in cmd[1:])}")
    t0 = time.time()
    with open(os.path.join(run_dir, "solver.log"), "w") as lf:
        rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT)
    log(f">> {cfg['label']}: solver finished rc={rc} in {(time.time() - t0) / 60:.1f} min")
    stop(proc)
    for _ in range(20):
        if port_free(port):
            break
        time.sleep(1)
    return rc == 0 and os.path.exists(sub_path)


# ------------------------------------------------------------------ summary
def summarise(cfgs, a):
    ch = json.load(open(a.challenges))
    sol = json.load(open(a.solutions)) if a.solutions else None
    rows = []
    nvarc_solved = None
    ids_all = None
    for cfg in cfgs:
        run_dir = os.path.join(a.out, cfg["label"])
        try:
            sub = json.load(open(os.path.join(run_dir, "submission.json")))
            rep = json.load(open(os.path.join(run_dir, "report.json")))
        except FileNotFoundError:
            rows.append({"label": cfg["label"], "family": cfg.get("family", ""), "status": "failed"})
            continue
        tasks = rep["tasks"]
        ids = [t for t in tasks if "error" not in tasks[t]]
        ids_all = ids_all or ids
        r = {"label": cfg["label"], "family": cfg.get("family", ""), "model": cfg.get("model", ""),
             "status": "ok", "n_tasks": len(ids),
             "thinking": "off" if "off" in cfg.get("solver", []) else "on"}
        if sol:
            solved, tot = score(sub, sol, ids)
            r.update({"solved": len(solved), "total": tot, "accuracy": len(solved) / max(tot, 1),
                      "solved_set": sorted(f"{t}:{i}" for t, i in solved)})
            if a.nvarc:
                if nvarc_solved is None:
                    nvarc_solved, _ = score(json.load(open(a.nvarc)), sol, ids)
                r.update({"only_ours": len(solved - nvarc_solved), "only_nvarc": len(nvarc_solved - solved),
                          "both": len(solved & nvarc_solved),
                          "union_accuracy": len(solved | nvarc_solved) / max(tot, 1)})
        agg = lambda k: sum(tasks[t].get(k, 0) for t in ids)
        r.update({"tokens": agg("tokens"), "prompt_tokens": agg("prompt_tokens"), "turns": agg("turns"),
                  "truncated_turns": agg("truncated_turns"), "n_candidates": agg("n_candidates"),
                  "n_pass_train": agg("n_pass_train"), "n_trusted": agg("n_trusted"),
                  "n_hygiene_rejected": agg("n_hygiene_rejected"), "n_heldout_ok": agg("n_heldout_ok"),
                  "elapsed_s": rep.get("elapsed_s", 0)})
        try:
            r["startup_s"] = json.load(open(os.path.join(run_dir, "startup.json")))["startup_s"]
        except (FileNotFoundError, KeyError, ValueError):
            r["startup_s"] = 0
        r["tokens_per_task"] = r["tokens"] / max(len(ids), 1)
        r["gen_tps_here"] = r["tokens"] / max(r["elapsed_s"], 1)
        r["kaggle_hours_at_l4_tps"] = r["tokens_per_task"] * 240 / max(a.l4_tps, 1) / 3600
        r["trunc_rate"] = r["truncated_turns"] / max(r["turns"], 1)
        rows.append(r)

    json.dump({"config": vars(a), "rows": rows}, open(os.path.join(a.out, "summary.json"), "w"), indent=1)

    ok = [r for r in rows if r["status"] == "ok"]
    lines = [f"# Model sweep — {a.split} split, {a.samples} samples x {a.rounds} rounds",
             "", f"Projected Kaggle hours assume **{a.l4_tps:,} generated tokens/s on 4xL4** "
                 f"(measure with bench_speed.py on an L4 box and re-run with --l4-tps). "
                 f"Solver budget line: {a.budget_hours} h.", ""]
    hdr = "| config | model | acc (pass@2) | solved | only ours vs NVARC | tok/task | trunc. turns | trusted / pass-train / rejected | gen tok/s here | proj. Kaggle h | server start |"
    lines += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for r in ok:
        acc = f"{r.get('accuracy', 0):.1%}" if "accuracy" in r else "n/a"
        only = f"{r['only_ours']} / +{r['only_nvarc']} ({r['union_accuracy']:.1%} union)" if "only_ours" in r else "—"
        lines.append(f"| {r['label']} | {r['model']} | {acc} | {r.get('solved', '?')}/{r.get('total', '?')} | {only} | "
                     f"{r['tokens_per_task']:,.0f} | {r['trunc_rate']:.0%} | "
                     f"{r['n_trusted']} / {r['n_pass_train']} / {r['n_hygiene_rejected']} | "
                     f"{r['gen_tps_here']:,.0f} | {r['kaggle_hours_at_l4_tps']:.1f} | {r['startup_s'] / 60:.1f} min |")
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['label']} | failed — see {a.out}/{r['label']}/server.log | | | | | | | | | |")
    if sol and len(ok) > 1:
        union = set()
        for r in ok:
            union |= set(r["solved_set"])
        lines += ["", f"Any config solves {len(union)}/{ok[0]['total']} test outputs "
                      f"({len(union) / max(ok[0]['total'], 1):.1%}) — the diversity available to an ensemble."]
    lines += ["", "Reading the columns: *trusted* = programs that pass all demos AND the hygiene checks; "
                  "*rejected* = pass the demos but were caught by hygiene (the false positives the verifier removes); "
                  "*trunc. turns* = turns cut off by the token cap (high => raise --max-tokens or lower thinking); "
                  "*server start* = model load + kernel compile on this box (warm cache after the first run; "
                  "the cold start counts against Kaggle's 12 h unless the compile cache ships with the model)."]
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    if sol and ok:
        try:
            chart(ok, a)
        except Exception as e:  # matplotlib missing etc. — the table is the primary output
            log(f"chart skipped: {type(e).__name__}: {e}")


def chart(rows, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # validated categorical palette (slots 1-3 pass all-pairs checks), one hue per model family
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    families = []
    for r in rows:
        if r["family"] not in families:
            families.append(r["family"])
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    xs = [r["kaggle_hours_at_l4_tps"] for r in rows]
    ys = [r["accuracy"] * 100 for r in rows]
    xr = max(max(xs) * 1.1, a.budget_hours * 1.1, 0.1)
    yr = max(max(ys) * 1.1, 1)
    placed = []
    for r, x, y in sorted(zip(rows, xs, ys), key=lambda t: (t[1], -t[2])):
        c = palette[families.index(r["family"]) % len(palette)]
        marker = "o" if r["thinking"] == "on" else "s"
        ax.scatter(x, y, s=90, color=c, marker=marker, edgecolors="#fcfcfb", linewidths=2, zorder=3)
        dy = 6
        while any(abs(x - px) / xr < 0.12 and abs((y + dy / 6) - py) / yr < 0.05 for px, py in placed):
            dy -= 14                                    # stack labels that would collide
        placed.append((x, y + dy / 6))
        ax.annotate(r["label"], (x, y), xytext=(8, dy), textcoords="offset points", fontsize=9, color="#0b0b0b")
    ax.axvline(a.budget_hours, color="#52514e", linestyle="--", linewidth=1, zorder=1)
    ax.text(a.budget_hours, ax.get_ylim()[1], f" solver budget {a.budget_hours} h", color="#52514e",
            fontsize=8, va="top")
    for i, f in enumerate(families):
        ax.scatter([], [], color=palette[i % len(palette)], label=f, s=60)
    ax.scatter([], [], color="#52514e", marker="o", label="thinking on", s=60)
    ax.scatter([], [], color="#52514e", marker="s", label="thinking off", s=60)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlabel(f"projected Kaggle hours for 240 tasks (assumes {a.l4_tps:,} tok/s on 4xL4)", color="#52514e")
    ax.set_ylabel(f"pass@2 accuracy on {a.split} split (%)", color="#52514e")
    ax.set_title("Zero-shot solver: accuracy vs. cost by model", color="#0b0b0b", loc="left")
    ax.grid(color="#e6e5e1", linewidth=0.8, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "summary.png"), dpi=130)
    log(f"chart written to {a.out}/summary.png")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default=os.path.join(HERE, "sweep_configs.json"))
    ap.add_argument("--only", help="comma-separated labels to run (default: all)")
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--solutions")
    ap.add_argument("--split", default="dev", choices=["dev", "lockbox", "all"])
    ap.add_argument("--tasks", help="passed to solver.py (comma-separated ids or a file)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--parallel-tasks", type=int, default=8)
    ap.add_argument("--parallel-per-task", type=int, default=8)
    ap.add_argument("--time-budget", type=float, default=0, help="seconds per config (0 = none)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size for vLLM (4 on a 4xL4 box)")
    ap.add_argument("--max-model-len", type=int, default=65536)
    ap.add_argument("--gpu-mem", type=float, default=0.92)
    ap.add_argument("--server-timeout", type=float, default=3600, help="seconds to wait for vLLM (downloads included)")
    ap.add_argument("--nvarc", help="NVARC submission.json on the same split, for complementarity")
    ap.add_argument("--l4-tps", type=float, default=1500, help="assumed generated tokens/s on 4xL4 for the projection")
    ap.add_argument("--budget-hours", type=float, default=4.0, help="hours the solver may use inside Kaggle's 12")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--summary-only", action="store_true", help="just rebuild summary from existing runs")
    ap.add_argument("--out", default="runs/sweep")
    a = ap.parse_args()

    cfgs = json.load(open(a.configs))["configs"]
    if a.only:
        want = set(a.only.split(","))
        cfgs = [c for c in cfgs if c["label"] in want]
    os.makedirs(a.out, exist_ok=True)
    if not a.summary_only:
        log(f"sweep of {len(cfgs)} configs -> {a.out}")
        for cfg in cfgs:
            try:
                ok = run_config(cfg, a, a.port)
            except KeyboardInterrupt:
                log("interrupted")
                subprocess.call(["pkill", "-f", "vllm serve"])
                break
            except Exception as e:
                log(f"!! {cfg['label']}: {type(e).__name__}: {e}")
                ok = False
            log(f">> {cfg['label']}: {'done' if ok else 'FAILED'}")
    summarise(cfgs, a)


if __name__ == "__main__":
    main()
