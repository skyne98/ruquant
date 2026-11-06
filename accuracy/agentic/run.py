#!/usr/bin/env python3
"""Agentic coding runs: opencode drives a llama-server model through the tasks in tasks/, sandboxed (sandbox.sh).

    python3 run.py LABEL [--tasks a,b] [--reps N] [--server-log PATH] [--validate]

Each run gets tasks/<task>/repo as a fresh git work tree and the prompts in tasks/<task>/prompts/ as consecutive
turns of one opencode session. After each turn of a multi-turn task, and after the last turn, the hidden tests
(tasks/<task>/hidden overlaid on a copy of the work tree) and the agent's own test suite run without network.
Results go to accuracy/results/agentic/LABEL/<task>-r<k>/: the event streams, the exported session (reasoning
included), the final diff, the test output, the slice of the llama-server log, and meta.json.

--validate checks the harness instead: the reference solutions (tasks/<task>/ref) must pass the hidden tests and the
untouched repos must fail them.
"""
import argparse
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TASKS = HERE / "tasks"
RESULTS = HERE.parent / "results" / "agentic"
SCRATCH = Path(os.environ.get("AGENTIC_SCRATCH", "/tmp/agentic-runs"))
MODEL = "llamacpp/qwen3.8-27b-w4a4"
TURN_TIMEOUT = int(os.environ.get("TURN_TIMEOUT", 2400))
GIT = ["git", "-c", "user.name=agentic", "-c", "user.email=agentic@localhost", "-c", "commit.gpgsign=false"]


def sandbox(run: Path, cmd: list[str], *, nonet=False, cwd="work", timeout=None, stdout=None, stderr=None):
    env = dict(os.environ, RUN=str(run), NONET="1" if nonet else "0", SANDBOX_CWD=cwd)
    return subprocess.run([str(HERE / "sandbox.sh"), *cmd], env=env, timeout=timeout, stdout=stdout, stderr=stderr,
                          text=True)


def fresh_run_dir(path: Path, repo: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    for d in ["home", "tmp", "xdg/config/opencode", "xdg/data", "xdg/cache", "xdg/state"]:
        (path / d).mkdir(parents=True)
    cfg = (HERE / "opencode.json").read_text()
    if os.environ.get("OPENCODE_URL"):  # e.g. through log_proxy.py
        cfg = cfg.replace("http://127.0.0.1:8099/v1", os.environ["OPENCODE_URL"])
    (path / "xdg/config/opencode/opencode.json").write_text(cfg)
    shutil.copytree(repo, path / "work", symlinks=True)
    subprocess.run([*GIT, "init", "-q"], cwd=path / "work", check=True)
    subprocess.run([*GIT, "add", "-A"], cwd=path / "work", check=True)
    subprocess.run([*GIT, "commit", "-qm", "initial"], cwd=path / "work", check=True)


def parse_tests(out: str) -> dict:
    """Passed / failed counts from pytest, cargo test or node --test output."""
    passed = failed = errors = 0
    for line in out.splitlines():  # pytest summary lines (one per pytest call)
        if re.search(r"\d+ (passed|failed|errors?)\b.* in [\d.]+s", line) and not line.startswith("test result"):
            passed += sum(int(x) for x in re.findall(r"(\d+) passed", line))
            failed += sum(int(x) for x in re.findall(r"(\d+) failed", line))
            errors += sum(int(x) for x in re.findall(r"(\d+) errors?\b", line))
    for p, f in re.findall(r"test result: \w+\. (\d+) passed; (\d+) failed", out):  # cargo
        passed += int(p)
        failed += int(f)
    if re.search(r"error(\[E\d+\])?: could not compile|error: no test target", out):
        errors += 1
    if (p := re.search(r"^(?:#|ℹ) pass (\d+)", out, re.M)) and (f := re.search(r"^(?:#|ℹ) fail (\d+)", out, re.M)):  # node
        passed, failed = int(p.group(1)), int(f.group(1))
    return {"passed": passed, "failed": failed, "errors": errors}


def run_tests(run: Path, task: Path, cmd: str, overlay: bool, name: str, out_dir: Path) -> dict:
    """Runs cmd on a copy of the work tree (with the hidden files overlaid if asked), without network."""
    check = run / "check"
    if check.exists():
        shutil.rmtree(check)
    shutil.copytree(run / "work", check, symlinks=True, ignore=shutil.ignore_patterns(".git"))
    if overlay:
        shutil.copytree(task / "hidden", check, dirs_exist_ok=True)
    t = time.time()
    try:
        r = sandbox(run, ["bash", "-c", cmd], nonet=True, cwd="check", timeout=900, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT)
        out, rc = r.stdout, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or "") + "\n[timeout]", -1
    (out_dir / f"{name}.txt").write_text(f"$ {cmd}\n{out}\n[exit {rc}]\n")
    res = parse_tests(out)
    res.update(rc=rc, ok=rc == 0 and res["failed"] == 0 and res["errors"] == 0 and res["passed"] > 0,
               seconds=round(time.time() - t, 1))
    return res


def hidden_cmd(spec: dict, stage: int | None) -> str:
    cmd = spec["hidden"]
    if "{stage_files}" in cmd:
        cmd = cmd.replace("{stage_files}", " ".join(spec["stage_files"].format(k=k) for k in range(1, stage + 1)))
    return cmd


def log_size(path: Path | None) -> int:
    return path.stat().st_size if path and path.exists() else 0


def run_one(label: str, task: Path, rep: int, server_log: Path | None) -> dict:
    spec = json.loads((task / "task.json").read_text())
    prompts = sorted((task / "prompts").glob("*.md"), key=lambda p: int(p.stem))
    name = f"{task.name}-r{rep}"
    run = SCRATCH / label / name
    out = RESULTS / label / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    fresh_run_dir(run, task / "repo")
    meta = {"label": label, "task": task.name, "rep": rep, "model": MODEL, "turns": [], "started": time.time()}
    log0 = log_size(server_log)
    session = None
    for k, prompt in enumerate(prompts, 1):
        cmd = ["opencode2", "run", "--standalone", "--auto", "--format", "json", "--thinking", "-m", MODEL,
               "--title", name]
        if session:
            cmd += ["--session", session]
        cmd.append(prompt.read_text())
        t = time.time()
        status = "ok"
        with open(out / f"events-{k}.jsonl", "w") as ev, open(out / f"opencode-{k}.err", "w") as err:
            try:
                r = sandbox(run, cmd, timeout=TURN_TIMEOUT, stdout=ev, stderr=err)
                if r.returncode != 0:
                    status = f"exit {r.returncode}"
            except subprocess.TimeoutExpired:
                status = "timeout"
        turn = {"turn": k, "status": status, "seconds": round(time.time() - t, 1)}
        for line in open(out / f"events-{k}.jsonl"):
            try:
                session = session or json.loads(line).get("sessionID")
            except json.JSONDecodeError:
                pass
        if len(prompts) > 1:
            turn["hidden"] = run_tests(run, task, hidden_cmd(spec, k), True, f"hidden-turn{k}", out)
        meta["turns"].append(turn)
        print(f"  {name} turn {k}: {status}, {turn['seconds']:.0f} s"
              + (f", hidden {turn['hidden']['passed']}/{turn['hidden']['passed'] + turn['hidden']['failed']}"
                 if "hidden" in turn else ""), flush=True)
    meta["session"] = session
    meta["seconds"] = round(time.time() - meta["started"], 1)
    if session:
        with open(out / "session.json", "w") as f:
            sandbox(run, ["opencode2", "session", "export", "--standalone", session], stdout=f,
                    stderr=subprocess.DEVNULL, timeout=300)
    subprocess.run([*GIT, "add", "-A"], cwd=run / "work", check=True)
    diff = subprocess.run([*GIT, "diff", "--cached", "--stat", "-p", "--", ".", ":!target", ":!Cargo.lock"],
                          cwd=run / "work", capture_output=True, text=True).stdout
    (out / "final.diff").write_text(diff)
    meta["hidden"] = run_tests(run, task, hidden_cmd(spec, len(prompts)), True, "hidden", out)
    meta["suite"] = run_tests(run, task, spec["suite"], False, "suite", out)
    if server_log:
        with open(server_log, "rb") as f:
            f.seek(log0)
            (out / "server.log").write_bytes(f.read())
    for p in out.glob("*.jsonl"):
        with open(p, "rb") as src, gzip.open(f"{p}.gz", "wb") as dst:
            shutil.copyfileobj(src, dst)
        p.unlink()
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    h = meta["hidden"]
    print(f"{name}: hidden {h['passed']}/{h['passed'] + h['failed']}{' errors' if h['errors'] else ''} "
          f"{'PASS' if h['ok'] else 'FAIL'}, suite {'ok' if meta['suite']['ok'] else 'FAIL'}, {meta['seconds']:.0f} s",
          flush=True)
    return meta


def validate(task: Path) -> bool:
    spec = json.loads((task / "task.json").read_text())
    n = len(list((task / "prompts").glob("*.md")))
    ok = True
    for variant in ["ref", "repo"]:
        run = SCRATCH / "validate" / f"{task.name}-{variant}"
        fresh_run_dir(run, task / "repo")
        if variant == "ref":
            shutil.copytree(task / "ref", run / "work", dirs_exist_ok=True)
        out = run / "out"
        out.mkdir()
        res = run_tests(run, task, hidden_cmd(spec, n), True, "hidden", out)
        good = res["ok"] if variant == "ref" else not res["ok"]
        ok &= good
        print(f"{task.name:18} {variant:4} hidden {res['passed']} passed {res['failed']} failed {res['errors']} errors "
              f"-> {'as expected' if good else 'WRONG'}  ({out / 'hidden.txt'})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--rep-start", type=int, default=1)
    ap.add_argument("--server-log", type=Path)
    ap.add_argument("--validate", action="store_true")
    a = ap.parse_args()
    tasks = sorted(p for p in TASKS.iterdir() if (p / "task.json").exists())
    if a.tasks != "all":
        tasks = [TASKS / t for t in a.tasks.split(",")]
    if a.validate:
        sys.exit(0 if all([validate(t) for t in tasks]) else 1)
    for rep in range(a.rep_start, a.rep_start + a.reps):
        for task in tasks:
            run_one(a.label, task, rep, a.server_log)


if __name__ == "__main__":
    main()
