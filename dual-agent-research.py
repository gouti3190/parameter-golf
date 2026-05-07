#!/usr/bin/env python3
"""
dual-agent-research.py  —  parameter-golf edition

Architecture: PLANNERS think, CODER implements, GPU validates.

Two planning agents never write code — they propose ideas as text.
A separate coder agent implements each plan via SEARCH/REPLACE edits.
If the coder's edits fail syntax/runtime, it retries with the error.
If both attempts fail, the previous working version is kept.
Next cycle, the planner sees "your code crashed: <error>" and adjusts.

Pipeline (pipelined so thinking overlaps GPU):
  Preflight → confirms val_bpb is captured (30s)
  Pre-start → Planner 1 proposes + Coder implements
  Cycle 1-3 → GPU test A1 ∥ (Planner 2 proposes + Coder implements)
              GPU test A2 ∥ (Planner 1 proposes + Coder implements)
  Last cycle → GPU test A2 only
  Discussion → planners debate (text), 2 rounds
  Synthesis  → Coder implements agreed plan → 10-min final run
  Log        → RUN_LOG.md in synthesis branch only
"""

import anthropic
import asyncio
import argparse
import ast
import subprocess
import re
import sys
import os
from pathlib import Path
from datetime import datetime

# ── ANSI ──────────────────────────────────────────────────────────────────────
R = "\x1b[0m"; B = "\x1b[1m"; DIM = "\x1b[2m"
BLUE = "\x1b[94m"; YELLOW = "\x1b[93m"; GREEN = "\x1b[92m"
RED = "\x1b[91m"; CYAN = "\x1b[96m"; GRAY = "\x1b[90m"

def c(col, s): return f"{col}{s}{R}"
def bl(s): return c(BLUE, s)
def yl(s): return c(YELLOW, s)
def gr(s): return c(GREEN, s)
def rd(s): return c(RED, s)
def dim(s): return c(DIM, s)
def bold(s): return c(B, s)
def ansi_strip(s): return re.sub(r'\x1b\[[0-9;]*m', '', s)

TARGET_FILE = "train_gpt.py"
FINAL_SECS = 600
MODEL = "claude-sonnet-4-5"

# ── Logger ────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self):
        self.lines, self.start = [], datetime.now()
    def __call__(self, *args, **kw):
        text = " ".join(str(a) for a in args)
        print(text, **kw); self.lines.append(ansi_strip(text))
    def section(self, title):
        self(f"\n{'─'*64}\n  {title}\n{'─'*64}")
    def gpu_result(self, label, r):
        bpb = f"{r['val_bpb']:.4f}" if r.get("val_bpb") else "N/A"
        mb  = f"{r['model_mb']:.1f}MB" if r.get("model_mb") else "?MB"
        ok  = "PASS" if r.get("ok") else "FAIL"
        self(f"  {label}  val_bpb={bpb}  size={mb}  [{ok}]")
    def flush_to(self, path, summary):
        elapsed = datetime.now() - self.start
        header = f"# RUN_LOG.md\n\n**Generated:** {self.start:%Y-%m-%d %H:%M}  \n**Duration:** {str(elapsed).split('.')[0]}  \n**Config:** {summary}\n\n---\n\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + "\n".join(self.lines), encoding="utf-8")

log = Logger()

# ── Git ───────────────────────────────────────────────────────────────────────
def git(cmd, cwd=None):
    r = subprocess.run(f"git {cmd}", shell=True, cwd=str(cwd or Path.cwd()),
                       capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    return r.stdout.strip()

def git_try(cmd, cwd=None):
    try: return git(cmd, cwd)
    except: return None

# ── GPU test (runs in thread so event loop stays free) ────────────────────────
def _gpu_sync(file, nproc, gpu_time, cwd, run_id, out_dir, repo_root):
    data   = str(Path(repo_root) / "data/datasets/fineweb10B_sp1024/")
    tok    = str(Path(repo_root) / "data/tokenizers/fineweb_1024_bpe.model")
    env = {**os.environ, "RUN_ID": run_id, "DATA_PATH": data,
           "TOKENIZER_PATH": tok, "VOCAB_SIZE": "1024",
           "MAX_WALLCLOCK_SECONDS": str(gpu_time), "VAL_LOSS_EVERY": "0"}
    cmd = f"torchrun --standalone --nproc_per_node={nproc} {file.name}"
    try:
        r = subprocess.run(cmd, shell=True, cwd=str(cwd), env=env,
                           capture_output=True, text=True, timeout=gpu_time + 120)
        out = (r.stdout + r.stderr).strip()
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / f"{run_id}.log").write_text(out, encoding="utf-8")
        bpb  = re.search(r"val_bpb[:\s=]+([0-9]+\.[0-9]+)", out)
        loss = re.search(r"val_loss[:\s=]+([0-9]+\.[0-9]+)", out)
        size = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*MB", out)
        # grab last 30 lines for error context
        tail = "\n".join(out.split("\n")[-30:])
        return {"ok": r.returncode == 0 and bpb is not None,
                "val_bpb": float(bpb.group(1)) if bpb else None,
                "val_loss": float(loss.group(1)) if loss else None,
                "model_mb": float(size.group(1)) if size else None,
                "tail": tail, "run_id": run_id}
    except subprocess.TimeoutExpired:
        return {"ok": False, "val_bpb": None, "val_loss": None,
                "model_mb": None, "tail": "timeout", "run_id": run_id}
    except Exception as e:
        return {"ok": False, "val_bpb": None, "val_loss": None,
                "model_mb": None, "tail": str(e), "run_id": run_id}

async def gpu_test(file, nproc, gpu_time, cwd, run_id, out_dir, repo_root):
    return await asyncio.to_thread(_gpu_sync, file, nproc, gpu_time, cwd, run_id, out_dir, repo_root)

# ── Preflight ─────────────────────────────────────────────────────────────────
def preflight(nproc, out_dir, repo_root):
    SECS = 30
    data = str(Path(repo_root) / "data/datasets/fineweb10B_sp1024/")
    tok  = str(Path(repo_root) / "data/tokenizers/fineweb_1024_bpe.model")
    env = {**os.environ, "RUN_ID": "preflight", "DATA_PATH": data,
           "TOKENIZER_PATH": tok, "VOCAB_SIZE": "1024",
           "MAX_WALLCLOCK_SECONDS": str(SECS), "VAL_LOSS_EVERY": "0"}
    cmd = f"torchrun --standalone --nproc_per_node={nproc} {TARGET_FILE}"
    log(f"  Running: {dim(cmd)}  ({SECS}s)")
    try:
        r = subprocess.run(cmd, shell=True, env=env, cwd=str(repo_root),
                           capture_output=True, text=True, timeout=SECS + 120)
        out = (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        log(rd("  ✗ Timed out")); sys.exit(1)
    except Exception as e:
        log(rd(f"  ✗ {e}")); sys.exit(1)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "preflight.log").write_text(out, encoding="utf-8")
    bpb = re.search(r"val_bpb[:\s=]+([0-9]+\.[0-9]+)", out)
    if bpb and r.returncode == 0:
        log(f"  {gr('✓')} val_bpb={bold(bpb.group(1))} — preflight passed"); return
    if not bpb: log(rd("  ✗ val_bpb not found in output"))
    for line in out.split("\n")[-10:]: log(f"    {line}")
    sys.exit(1)

# ── SEARCH/REPLACE engine ────────────────────────────────────────────────────
EDIT_FORMAT = """
OUTPUT FORMAT — use this exact format for every change:

<<<SEARCH
exact lines copied from the current file (include 3-5 lines of context)
===REPLACE
replacement lines
>>>

RULES:
- SEARCH text must be EXACT — copy-paste from the file, preserve indentation
- Multiple <<<SEARCH...>>> blocks allowed
- 2-4 changes max
- No prose, no markdown — only edit blocks
""".strip()

def apply_edits(code, edits_text):
    pattern = r'<<<SEARCH\n(.*?)\n===REPLACE\n(.*?)\n>>>'
    matches = re.findall(pattern, edits_text, re.DOTALL)
    if not matches:
        log(f"  {c(YELLOW, '⚠')} No edit blocks found")
        return code, 0
    applied = 0
    for old, new in matches:
        old, new = old.rstrip(), new.rstrip()
        if old in code:
            code = code.replace(old, new, 1); applied += 1
        else:
            first = old.split("\n")[0][:60]
            log(f"  {c(YELLOW, '⚠')} No match: {dim(first)}...")
    log(f"  Applied {applied}/{len(matches)} edits")
    return code, applied

def check_syntax(code):
    try: ast.parse(code); return True, ""
    except SyntaxError as e: return False, f"SyntaxError line {e.lineno}: {e.msg}"

# ── API helpers ───────────────────────────────────────────────────────────────
async def call_agent(client, system, prompt, max_tokens=4096):
    full = ""
    async with client.messages.stream(
        model=MODEL, max_tokens=max_tokens, system=system,
        messages=[{"role": "user", "content": prompt}],
    ) as s:
        async for text in s.text_stream:
            full += text
    return full

# ── File map for context ─────────────────────────────────────────────────────
def file_map(code):
    lines = code.split("\n")
    entries = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if any(s.startswith(k) for k in ("import ", "from ", "class ", "def ")):
            entries.append(f"L{i:4d}  {s[:80]}")
        elif "=" in s and not s.startswith("#") and i < 200:
            if not s.startswith(("if ", "for ", "while ", "return ")):
                entries.append(f"L{i:4d}  {s[:80]}")
    return f"{TARGET_FILE} ({len(lines)} lines)\n" + "\n".join(entries[:100])

def bpb_str(r): return f"{r['val_bpb']:.4f}" if r.get("val_bpb") else "N/A"
def load_instr(p):
    if Path(p).exists(): return Path(p).read_text(encoding="utf-8")
    log(f"  {c(YELLOW, '⚠')} {p} not found"); return ""
def save(p, t): Path(p).parent.mkdir(parents=True, exist_ok=True); Path(p).write_text(t, encoding="utf-8")
def diff_summary(base, mod):
    bl, ml = base.split("\n"), mod.split("\n")
    d = []
    for i, (b, m) in enumerate(zip(bl, ml)):
        if b != m:
            d.append(f"  L{i+1}: {b.strip()[:70]}  →  {m.strip()[:70]}")
    if len(ml) > len(bl): d.append(f"  +{len(ml)-len(bl)} lines added")
    return "\n".join(d[:40]) if d else "(no changes)"

# ── Planner prompt (text only, never code) ────────────────────────────────────
def planner_system(focus):
    return (
        f"You are an expert ML researcher focused on: {focus}.\n"
        f"You NEVER write code. You propose specific, actionable changes.\n"
        f"For each change, specify: the exact function/class/variable name, "
        f"what the current value is, what to change it to, and why it helps val_bpb.\n"
        f"Limit to 2-4 changes per cycle. Be precise enough that a coder "
        f"can implement without asking questions."
    )

def planner_prompt(code, instructions, cycle, prev_result=None, prev_plan=None):
    fm = file_map(code)
    feedback = ""
    if prev_result:
        if prev_result.get("ok"):
            feedback = f"\nYour previous cycle achieved val_bpb={bpb_str(prev_result)}. Improve on this.\n"
        else:
            feedback = (
                f"\nYour previous cycle FAILED. The GPU test crashed with:\n"
                f"{prev_result.get('tail', 'unknown error')}\n"
                f"Your previous plan was:\n{prev_plan}\n"
                f"Adjust your plan to avoid this failure.\n"
            )
    return (
        f"Goal: improve train_gpt.py for lowest val_bpb under 16MB compressed.\n"
        f"Cycle {cycle}.\n{feedback}\n"
        f"YOUR FOCUS:\n{instructions}\n\n"
        f"FILE STRUCTURE:\n{fm}\n\n"
        f"Propose 2-4 specific changes. For each:\n"
        f"1. What to change (exact variable/function name + current value)\n"
        f"2. New value or modification\n"
        f"3. Expected impact on val_bpb\n"
        f"4. Risk assessment\n\n"
        f"Be extremely specific. A coder will implement your plan exactly."
    )

# ── Coder prompt (implements a plan via SEARCH/REPLACE) ───────────────────────
CODER_SYSTEM = (
    "You are a precise code editor. You receive a plan and a Python file.\n"
    "You implement EXACTLY what the plan says using SEARCH/REPLACE blocks.\n"
    "You NEVER add your own ideas — only implement the plan.\n"
    "The SEARCH text must be EXACT copy-paste from the file."
)

def coder_prompt(code, plan):
    return (
        f"Implement this plan in train_gpt.py:\n\n"
        f"PLAN:\n{plan}\n\n"
        f"FILE:\n{code}\n\n"
        f"{EDIT_FORMAT}\n"
    )

# ── Plan + implement + validate (one agent's full cycle) ──────────────────────
async def plan_and_implement(client, current_code, base_code, focus, instructions,
                              cycle, prev_result, prev_plan, label):
    """
    1. Planner proposes changes (text)
    2. Coder implements them (SEARCH/REPLACE)
    3. Syntax check — if fail, coder retries once with the error
    Returns (new_code, plan_text)
    """
    # Step 1: Plan
    log(f"  {dim('⟳')} {label} planning...")
    plan = await call_agent(client, planner_system(focus),
                            planner_prompt(current_code, instructions, cycle,
                                          prev_result, prev_plan))
    log(f"  {gr('✓')} {label} plan ready")
    log(f"  {dim(plan[:200])}...")

    # Step 2: Coder implements
    log(f"  {dim('⟳')} Coder implementing {label} plan...")
    edits_raw = await call_agent(client, CODER_SYSTEM,
                                  coder_prompt(current_code, plan), max_tokens=3000)
    new_code, n_applied = apply_edits(current_code, edits_raw)

    if n_applied == 0:
        log(f"  {c(YELLOW, '⚠')} No edits applied — keeping previous version")
        return current_code, plan

    # Step 3: Syntax check
    ok, err = check_syntax(new_code)
    if ok:
        log(f"  {gr('✓')} Coder done — {n_applied} edits, syntax OK")
        return new_code, plan

    # Step 4: Retry coder once with error
    log(f"  {c(YELLOW, '⚠')} {err} — coder retrying...")
    retry_prompt = (
        f"Your edits produced: {err}\n\n"
        f"Your previous output:\n{edits_raw}\n\n"
        f"Fix the edits. The SEARCH text must be EXACT from the file.\n"
        f"FILE:\n{current_code}\n\n"
        f"PLAN (implement this):\n{plan}\n\n"
        f"{EDIT_FORMAT}\n"
    )
    edits_raw2 = await call_agent(client, CODER_SYSTEM, retry_prompt, max_tokens=3000)
    new_code2, n2 = apply_edits(current_code, edits_raw2)
    ok2, err2 = check_syntax(new_code2)
    if ok2 and n2 > 0:
        log(f"  {gr('✓')} Coder retry succeeded — {n2} edits")
        return new_code2, plan

    log(f"  {c(YELLOW, '⚠')} Coder retry failed — keeping previous version")
    return current_code, plan

# ── Discussion ────────────────────────────────────────────────────────────────
def disc_system(num, my_focus, other_focus, instr):
    return (
        f"You are Agent {num} reviewing train_gpt.py improvements.\n"
        f"Your focus: {my_focus}. Other agent: {other_focus}.\n\n"
        f"Rules: reference specific changes, argue from val_bpb impact,\n"
        f"under 250 words, no code, end with:\n"
        f"'My proposal: keep [X from mine], keep [Y from theirs]'\n\n{instr}"
    )

async def run_discussion(client, a1_plans, a2_plans, a1_results, a2_results,
                         a1_focus, a2_focus, n_turns, a1_instr, a2_instr,
                         base_code, a1_code, a2_code):
    transcript = []
    s1 = disc_system(1, a1_focus, a2_focus, a1_instr)
    s2 = disc_system(2, a2_focus, a1_focus, a2_instr)

    a1_diff = diff_summary(base_code, a1_code)
    a2_diff = diff_summary(base_code, a2_code)
    best_a1 = min(a1_results, key=lambda r: r.get("val_bpb") or 999)
    best_a2 = min(a2_results, key=lambda r: r.get("val_bpb") or 999)

    a1_hist, a2_hist = [], []

    for turn in range(n_turns):
        if turn == 0:
            a1_ctx = (
                f"Your changes (Agent 1, {a1_focus}):\n{a1_diff}\n"
                f"Your plans:\n" + "\n".join(f"Cycle {i+1}: {p[:200]}" for i, p in enumerate(a1_plans)) +
                f"\nBest val_bpb: {bpb_str(best_a1)}\n\n"
                f"Agent 2's changes ({a2_focus}) — FIRST TIME seeing:\n{a2_diff}\n"
                f"Their best: {bpb_str(best_a2)}\n\n"
                f"What should go into the final 10-minute run?"
            )
        else:
            a1_ctx = f"Agent 2 said:\n{transcript[-1]['text']}\n\nRefine your proposal."

        log.section(f"DISCUSSION Turn {turn+1}/{n_turns} · Agent 1")
        a1_reply = await call_agent(client, s1, a1_ctx, max_tokens=500)
        a1_hist.append(a1_reply)
        transcript.append({"agent": 1, "turn": turn+1, "text": a1_reply})
        log(f"  {bl('A1:')} {a1_reply[:150]}...")

        if turn == 0:
            a2_ctx = (
                f"Your changes (Agent 2, {a2_focus}):\n{a2_diff}\n"
                f"Your plans:\n" + "\n".join(f"Cycle {i+1}: {p[:200]}" for i, p in enumerate(a2_plans)) +
                f"\nBest val_bpb: {bpb_str(best_a2)}\n\n"
                f"Agent 1's changes ({a1_focus}) — FIRST TIME seeing:\n{a1_diff}\n"
                f"Their best: {bpb_str(best_a1)}\n\n"
                f"Agent 1 said:\n{a1_reply}\n\nRespond."
            )
        else:
            a2_ctx = f"Agent 1 said:\n{a1_reply}\n\nRefine your proposal."

        log.section(f"DISCUSSION Turn {turn+1}/{n_turns} · Agent 2")
        a2_reply = await call_agent(client, s2, a2_ctx, max_tokens=500)
        a2_hist.append(a2_reply)
        transcript.append({"agent": 2, "turn": turn+1, "text": a2_reply})
        log(f"  {yl('A2:')} {a2_reply[:150]}...")

    md = [f"# Discussion ({datetime.now():%Y-%m-%d %H:%M})\n"]
    for e in transcript:
        md.append(f"## Agent {e['agent']} · Turn {e['turn']}\n\n{e['text']}\n\n---\n")
    return transcript, "\n".join(md)

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="Dual-agent parameter-golf")
    parser.add_argument("--nproc",    type=int, default=1)
    parser.add_argument("--gpu-time", type=int, default=150)
    parser.add_argument("--cycles",   type=int, default=3)
    parser.add_argument("--discuss",  type=int, default=2)
    parser.add_argument("--out",      default=".agents")
    parser.add_argument("--a1",       default="AGENT_1_INSTRUCTIONS.md")
    parser.add_argument("--a2",       default="AGENT_2_INSTRUCTIONS.md")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if git_try("rev-parse --show-toplevel") is None:
        print(rd("Not a git repo")); sys.exit(1)
    if not Path(TARGET_FILE).exists():
        print(rd(f"{TARGET_FILE} not found")); sys.exit(1)

    repo_root   = Path(git("rev-parse --show-toplevel"))
    base_branch = git("rev-parse --abbrev-ref HEAD")
    base_code   = Path(TARGET_FILE).read_text(encoding="utf-8")
    a1_instr    = load_instr(args.a1)
    a2_instr    = load_instr(args.a2)
    a1_focus    = "Architecture & Model Efficiency"
    a2_focus    = "Quantization & Compression"
    client      = anthropic.AsyncAnthropic()
    cfg = f"nproc={args.nproc} gpu={args.gpu_time}s cycles={args.cycles} discuss={args.discuss}"

    log(f"""
{bold(CYAN)}╔══════════════════════════════════════════════════════════════╗{R}
{bold(CYAN)}║{R}  {bold(CYAN)}Dual Agent Research  ·  parameter-golf{R}                      {bold(CYAN)}║{R}
{bold(CYAN)}╚══════════════════════════════════════════════════════════════╝{R}
  {dim("target:")}    {bold(TARGET_FILE)} ({len(base_code.split(chr(10)))} lines)
  {dim("method:")}    {bold("Planners think → Coder implements → GPU validates")}
  {bl("Agent 1")}  →  {a1_focus}  →  branch: {bold("agent-1")}
  {yl("Agent 2")}  →  {a2_focus}  →  branch: {bold("agent-2")}
  {dim("pipeline:")}  {args.cycles} cycles · {args.gpu_time}s tests · {args.discuss} discussion turns
  {dim("final:")}     {FINAL_SECS}s → synthesis + RUN_LOG.md
""")

    # ── Git ───────────────────────────────────────────────────────────────────
    log.section("GIT SETUP")
    wt1 = (out_dir / "worktree-1").resolve()
    wt2 = (out_dir / "worktree-2").resolve()
    for wt in [wt1, wt2]: git_try(f'worktree remove --force "{wt}"')
    for b in ["agent-1", "agent-2", "synthesis"]: git_try(f"branch -D {b}")
    git("checkout -b agent-1"); git(f"checkout {base_branch}")
    git("checkout -b agent-2"); git(f"checkout {base_branch}")
    git(f'worktree add "{wt1}" agent-1'); git(f'worktree add "{wt2}" agent-2')
    f1, f2 = wt1 / TARGET_FILE, wt2 / TARGET_FILE
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    log(f"  {gr('✓')} agent-1 → {dim(str(wt1))}")
    log(f"  {gr('✓')} agent-2 → {dim(str(wt2))}")

    # ── Preflight ─────────────────────────────────────────────────────────────
    log.section("PREFLIGHT")
    preflight(args.nproc, out_dir, repo_root)

    # ── State ─────────────────────────────────────────────────────────────────
    a1_code, a2_code = base_code, base_code
    a1_results, a2_results = [], []
    a1_plans, a2_plans = [], []

    # ── Pre-start: Agent 1 plans + coder implements ───────────────────────────
    log.section("PRE-START · Agent 1 plans + coder implements")
    a1_code, plan = await plan_and_implement(
        client, a1_code, base_code, a1_focus, a1_instr,
        cycle=1, prev_result=None, prev_plan=None, label=bl("Agent 1"),
    )
    a1_plans.append(plan)

    # ── Pipeline ──────────────────────────────────────────────────────────────
    for cycle in range(1, args.cycles + 1):

        # Step A: A1 GPU ∥ A2 plans+implements
        log.section(f"CYCLE {cycle}/{args.cycles} · A1 GPU ({args.gpu_time}s) ∥ A2 plans")
        f1.write_text(a1_code, encoding="utf-8")

        prev_a2 = a2_results[-1] if a2_results else None
        prev_a2_plan = a2_plans[-1] if a2_plans else None

        a1_result, (a2_code_new, a2_plan) = await asyncio.gather(
            gpu_test(f1, args.nproc, args.gpu_time, wt1, f"a1_c{cycle}", out_dir, repo_root),
            plan_and_implement(
                client, a2_code, base_code, a2_focus, a2_instr,
                cycle=cycle, prev_result=prev_a2, prev_plan=prev_a2_plan,
                label=yl("Agent 2"),
            ),
        )
        a1_results.append(a1_result)
        a2_code = a2_code_new
        a2_plans.append(a2_plan)
        log.gpu_result(f"Agent 1 Cycle {cycle}", a1_result)
        git(f'add "{TARGET_FILE}"', wt1)
        git(f'commit --allow-empty -m "A1 C{cycle} [{bpb_str(a1_result)}]"', wt1)

        # Step B: A2 GPU ∥ A1 plans+implements (skip A1 on last cycle)
        if cycle < args.cycles:
            log.section(f"CYCLE {cycle}/{args.cycles} · A2 GPU ({args.gpu_time}s) ∥ A1 plans")
            f2.write_text(a2_code, encoding="utf-8")

            a2_result, (a1_code_new, a1_plan) = await asyncio.gather(
                gpu_test(f2, args.nproc, args.gpu_time, wt2, f"a2_c{cycle}", out_dir, repo_root),
                plan_and_implement(
                    client, a1_code, base_code, a1_focus, a1_instr,
                    cycle=cycle+1, prev_result=a1_result, prev_plan=a1_plans[-1],
                    label=bl("Agent 1"),
                ),
            )
            a1_code = a1_code_new
            a1_plans.append(a1_plan)
        else:
            log.section(f"CYCLE {cycle}/{args.cycles} · A2 GPU ({args.gpu_time}s) — final")
            f2.write_text(a2_code, encoding="utf-8")
            a2_result = await gpu_test(f2, args.nproc, args.gpu_time, wt2,
                                       f"a2_c{cycle}", out_dir, repo_root)

        a2_results.append(a2_result)
        log.gpu_result(f"Agent 2 Cycle {cycle}", a2_result)
        git(f'add "{TARGET_FILE}"', wt2)
        git(f'commit --allow-empty -m "A2 C{cycle} [{bpb_str(a2_result)}]"', wt2)

    # ── Summary ───────────────────────────────────────────────────────────────
    log.section("CYCLE SUMMARY")
    for i, (r1, r2) in enumerate(zip(a1_results, a2_results), 1):
        log(f"  Cycle {i}   A1: {bpb_str(r1)}   A2: {bpb_str(r2)}")
    best_a1 = min(a1_results, key=lambda r: r.get("val_bpb") or 999)
    best_a2 = min(a2_results, key=lambda r: r.get("val_bpb") or 999)
    log(f"  Best A1: {bpb_str(best_a1)}   Best A2: {bpb_str(best_a2)}")

    # ── Discussion ────────────────────────────────────────────────────────────
    log.section("DISCUSSION · Agents see each other for the first time")
    transcript, disc_md = await run_discussion(
        client, a1_plans, a2_plans, a1_results, a2_results,
        a1_focus, a2_focus, args.discuss, a1_instr, a2_instr,
        base_code, a1_code, a2_code,
    )
    save(out_dir / "discussion.md", disc_md)

    # ── Synthesis ─────────────────────────────────────────────────────────────
    log.section("SYNTHESIS · Coder implements agreed changes")
    disc_text = "\n\n".join(f"Agent {e['agent']}:\n{e['text']}" for e in transcript)

    # Build a synthesis plan from the discussion
    synth_plan_prompt = (
        f"Based on this discussion, write a precise implementation plan:\n\n"
        f"{disc_text}\n\n"
        f"Agent 1's changes:\n{diff_summary(base_code, a1_code)}\n"
        f"Agent 2's changes:\n{diff_summary(base_code, a2_code)}\n\n"
        f"List the exact changes to apply to the ORIGINAL base file.\n"
        f"For each: variable/function name, old value, new value, reason.\n"
        f"This runs for 10 FULL MINUTES — pick the best from both agents."
    )
    synth_plan = await call_agent(client,
        "You are a senior ML engineer. Write a precise implementation plan. No code.",
        synth_plan_prompt, max_tokens=1000)
    log(f"  Synthesis plan: {dim(synth_plan[:200])}...")

    # Coder implements the synthesis plan on base_code
    log(f"  {dim('⟳')} Coder implementing synthesis plan...")
    synth_edits = await call_agent(client, CODER_SYSTEM,
                                    coder_prompt(base_code, synth_plan), max_tokens=3000)
    merged, n = apply_edits(base_code, synth_edits)
    ok, err = check_syntax(merged)
    if not ok or n == 0:
        # Retry
        log(f"  {c(YELLOW, '⚠')} {err or 'no edits'} — retrying...")
        retry = await call_agent(client, CODER_SYSTEM,
            f"Error: {err}\nPlan: {synth_plan}\nFile:\n{base_code}\n\n{EDIT_FORMAT}", max_tokens=3000)
        merged, n2 = apply_edits(base_code, retry)
        ok2, _ = check_syntax(merged)
        if not ok2 or n2 == 0:
            # Fall back to best agent
            if (best_a1.get("val_bpb") or 999) <= (best_a2.get("val_bpb") or 999):
                merged = a1_code
                log(f"  {c(YELLOW, '⚠')} Using Agent 1's code (best: {bpb_str(best_a1)})")
            else:
                merged = a2_code
                log(f"  {c(YELLOW, '⚠')} Using Agent 2's code (best: {bpb_str(best_a2)})")
        else:
            log(f"  {gr('✓')} Coder retry OK — {n2} edits")
    else:
        log(f"  {gr('✓')} Synthesis done — {n} edits applied")

    git("checkout -b synthesis")
    (repo_root / TARGET_FILE).write_text(merged, encoding="utf-8")

    # ── Final 10-min run ──────────────────────────────────────────────────────
    log.section(f"FINAL RUN · {FINAL_SECS}s ({FINAL_SECS//60} min)")
    data = str(repo_root / "data/datasets/fineweb10B_sp1024/")
    tok  = str(repo_root / "data/tokenizers/fineweb_1024_bpe.model")
    fenv = {**os.environ, "RUN_ID": "synthesis_final", "DATA_PATH": data,
            "TOKENIZER_PATH": tok, "VOCAB_SIZE": "1024",
            "MAX_WALLCLOCK_SECONDS": str(FINAL_SECS), "VAL_LOSS_EVERY": "200"}
    cmd = f"torchrun --standalone --nproc_per_node={args.nproc} {TARGET_FILE}"
    log(f"  $ {dim(cmd)}")
    try:
        r = subprocess.run(cmd, shell=True, cwd=str(repo_root), env=fenv,
                           capture_output=True, text=True, timeout=FINAL_SECS + 120)
        fout = (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired: fout = "timeout"
    except Exception as e: fout = str(e)

    (out_dir / "synthesis_final.log").write_text(fout, encoding="utf-8")
    fbpb = re.search(r"val_bpb[:\s=]+([0-9]+\.[0-9]+)", fout)
    fsize = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*MB", fout)
    final = {"ok": fbpb is not None,
             "val_bpb": float(fbpb.group(1)) if fbpb else None,
             "model_mb": float(fsize.group(1)) if fsize else None}
    log.gpu_result("FINAL (10 min)", final)

    # ── Commit ────────────────────────────────────────────────────────────────
    log_path = repo_root / "RUN_LOG.md"
    log.flush_to(log_path, cfg)
    git(f'add "{TARGET_FILE}" "RUN_LOG.md"')
    git(f'commit --allow-empty -m "Synthesis [{bpb_str(final)}]"')
    log(f"  {gr('✓')} Committed to synthesis")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    git_try(f'worktree remove --force "{wt1}"')
    git_try(f'worktree remove --force "{wt2}"')

    # ── Done ──────────────────────────────────────────────────────────────────
    log.section("DONE")
    all_bpb = [(r.get("val_bpb"), f"a1-c{i+1}") for i, r in enumerate(a1_results) if r.get("val_bpb")]
    all_bpb += [(r.get("val_bpb"), f"a2-c{i+1}") for i, r in enumerate(a2_results) if r.get("val_bpb")]
    if final.get("val_bpb"): all_bpb.append((final["val_bpb"], "synthesis"))
    if all_bpb:
        best, name = min(all_bpb, key=lambda x: x[0])
        log(f"\n  {gr('Best:')} {bold(f'{best:.4f}')} from {bold(name)}")
    log(f"""
  Branches:
    {bl("agent-1")}    {a1_focus}
    {yl("agent-2")}    {a2_focus}
    {gr("synthesis")}  merged + RUN_LOG.md  ← you are here

  git diff agent-1 agent-2 -- {TARGET_FILE}
  git diff {base_branch} synthesis -- {TARGET_FILE}
  cat RUN_LOG.md
""")

if __name__ == "__main__":
    asyncio.run(main())