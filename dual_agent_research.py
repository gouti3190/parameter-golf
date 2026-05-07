#!/usr/bin/env python3
"""
dual-agent-research.py  —  parameter-golf edition

Two Claude agents independently modify train_gpt.py on separate git branches,
each optimising for a different aspect of the same goal (lowest val_bpb).

Flow:
  Round 1    parallel  Agent 1: Code+Summary  |  Agent 2: Plan+Code   (GPU test A1)
  Round 2    parallel  roles swap             |                        (GPU test A2)
  Discussion sequential multi-turn debate (agents see each other for first time)
  Synthesis  merges code + discussion → commits to `synthesis` branch
  QA         final GPU test, writes .agents/report.md

Usage:
  python dual-agent-research.py [options]

Options:
  --nproc     GPUs for torchrun            (default: 1)
  --gpu-time  Seconds each GPU test runs   (default: 150 = 2.5 min)
  --timer     Seconds per coding round     (default: 120)
  --discuss   Discussion turns             (default: 2)
  --out       Output directory             (default: .agents)
  --a1        Agent 1 instructions file    (default: AGENT_1_INSTRUCTIONS.md)
  --a2        Agent 2 instructions file    (default: AGENT_2_INSTRUCTIONS.md)

Requirements:
  pip install anthropic
  export ANTHROPIC_API_KEY=sk-ant-...
  git 2.5+  (for git worktree)
  torchrun + data already downloaded (see repo README)
"""

import anthropic
import asyncio
import argparse
import subprocess
import re
import sys
import os
from pathlib import Path
from datetime import datetime

# ── ANSI colours ──────────────────────────────────────────────────────────────
R      = "\x1b[0m";  B      = "\x1b[1m";  DIM    = "\x1b[2m"
BLUE   = "\x1b[94m"; YELLOW = "\x1b[93m"; GREEN  = "\x1b[92m"
RED    = "\x1b[91m"; CYAN   = "\x1b[96m"; GRAY   = "\x1b[90m"

def c(col, s):  return f"{col}{s}{R}"
def bl(s):      return c(BLUE,   s)
def yl(s):      return c(YELLOW, s)
def gr(s):      return c(GREEN,  s)
def rd(s):      return c(RED,    s)
def dim(s):     return c(DIM,    s)
def bold(s):    return c(B,      s)

# ── Target file (locked to train_gpt.py) ─────────────────────────────────────
TARGET_FILE = "train_gpt.py"

# ── Git helpers ───────────────────────────────────────────────────────────────
def git(cmd: str, cwd: Path = None) -> str:
    r = subprocess.run(f"git {cmd}", shell=True, cwd=str(cwd or Path.cwd()),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    return r.stdout.strip()

def git_try(cmd: str, cwd: Path = None) -> str | None:
    try:    return git(cmd, cwd)
    except: return None

# ── GPU test via torchrun ─────────────────────────────────────────────────────
def run_gpu_test(file: Path, nproc: int, gpu_time: int,
                 cwd: Path, run_id: str, out_dir: Path) -> dict:
    env = {
        **os.environ,
        "RUN_ID":                run_id,
        "DATA_PATH":             "./data/datasets/fineweb10B_sp1024/",
        "TOKENIZER_PATH":        "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE":            "1024",
        "MAX_WALLCLOCK_SECONDS": str(gpu_time),
        "VAL_LOSS_EVERY":        "0",
    }
    cmd = f"torchrun --standalone --nproc_per_node={nproc} {file.name}"
    print(f"\n  {dim('$')} {dim(cmd)}")
    print(f"  {dim('running for')} {gpu_time}s {dim('on')} {nproc} GPU(s)...\n")
    try:
        r = subprocess.run(cmd, shell=True, cwd=str(cwd), env=env,
                           capture_output=True, text=True, timeout=gpu_time + 60)
        out = (r.stdout + r.stderr).strip()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{run_id}.log").write_text(out, encoding="utf-8")
        bpb   = re.search(r"val_bpb[:\s=]+([0-9]+\.[0-9]+)", out)
        loss  = re.search(r"val_loss[:\s=]+([0-9]+\.[0-9]+)", out)
        size  = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*MB", out)
        return {
            "ok":       r.returncode == 0 and bpb is not None,
            "val_bpb":  float(bpb.group(1))  if bpb  else None,
            "val_loss": float(loss.group(1)) if loss else None,
            "model_mb": float(size.group(1)) if size else None,
            "log":      out[:3000],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "val_bpb": None, "val_loss": None, "model_mb": None, "log": "timeout"}
    except Exception as e:
        return {"ok": False, "val_bpb": None, "val_loss": None, "model_mb": None, "log": str(e)}

def fmt_bpb(r):
    if r.get("val_bpb"):
        col = GREEN if r["val_bpb"] < 1.20 else YELLOW
        return f"{col}{r['val_bpb']:.4f} bpb{R}"
    return rd("no val_bpb")

def fmt_mb(r):
    if r.get("model_mb"):
        col = GREEN if r["model_mb"] < 16.0 else RED
        return f"{col}{r['model_mb']:.1f} MB{R}"
    return dim("? MB")

def bpb_str(r): return f"{r['val_bpb']:.4f}" if r.get("val_bpb") else "N/A"

# ── Misc helpers ──────────────────────────────────────────────────────────────
def strip_fences(text: str) -> str:
    m = re.search(r"```(?:\w*\n)?([\s\S]*?)```", text)
    return (m.group(1) if m else text).strip()

def save(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def section(title: str, color: str = B):
    bar = "─" * 64
    print(f"\n{color}{bar}{R}\n{color}  {title}{R}\n{color}{bar}{R}")

def load_instructions(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    print(f"  {c(YELLOW, '⚠')}  {path} not found — using built-in rules only")
    return ""

# ── System prompts ────────────────────────────────────────────────────────────
def make_coding_system(role: str, include_summary: bool, instructions: str) -> str:
    summary_rule = (
        "At the END of the file add:\n"
        "# AGENT SUMMARY:\n# - Changed: <each change>\n"
        "# - Rationale: <why it helps val_bpb>\n# - Risk: <what could go wrong>"
        if include_summary else
        "Do NOT add any summary or test block. Pure implementation only."
    )
    return (
        f"You are an expert ML engineer in the OpenAI Parameter Golf challenge.\n"
        f"Goal: lowest val_bpb (bits per byte) in a model that fits in 16MB compressed.\n\n"
        f"Role this round: {role}\n\n"
        f"OUTPUT RULES:\n"
        f"1. Output ONLY raw Python code. No markdown, no backticks, no prose.\n"
        f"2. File runs via: torchrun --standalone --nproc_per_node=N train_gpt.py\n"
        f"3. Respect env vars: RUN_ID, DATA_PATH, TOKENIZER_PATH, VOCAB_SIZE, MAX_WALLCLOCK_SECONDS.\n"
        f"4. Do NOT remove the val_bpb / val_loss print lines at the end.\n"
        f"5. Comment every change with [A1] or [A2] tag and val_bpb rationale.\n"
        f"6. {summary_rule}\n"
        f"7. You are isolated — zero knowledge of the other agent.\n\n"
        f"--- YOUR INSTRUCTIONS ---\n{instructions}"
    )

def make_discussion_system(agent_num: int, my_focus: str,
                           other_focus: str, instructions: str) -> str:
    return (
        f"You are Agent {agent_num} in a technical discussion about train_gpt.py.\n"
        f"Your focus: {my_focus}. Other agent's focus: {other_focus}.\n"
        f"You are seeing the other agent's code for the FIRST TIME.\n\n"
        f"Rules:\n"
        f"- Reference specific function names, hyperparameter values\n"
        f"- Argue from your focus — be specific about val_bpb impact\n"
        f"- Acknowledge what they got right; challenge what's weak\n"
        f"- Under 300 words. No code — discussion only.\n"
        f"- End every reply with: 'My proposal: keep [X from mine], keep [Y from theirs]'\n\n"
        f"--- YOUR INSTRUCTIONS (Discussion section applies) ---\n{instructions}"
    )

def make_synthesis_system(a1_focus: str, a2_focus: str, n_turns: int) -> str:
    return (
        f"You are a Synthesis engineer for the OpenAI Parameter Golf challenge.\n"
        f"Agent 1 focused on: {a1_focus}. Agent 2 focused on: {a2_focus}.\n"
        f"They held a {n_turns * 2}-message discussion and made concrete proposals.\n\n"
        f"OUTPUT RULES:\n"
        f"1. Output ONLY raw Python code. No markdown, no backticks, no prose.\n"
        f"2. File runs via: torchrun --standalone --nproc_per_node=N train_gpt.py\n"
        f"3. Implement what the agents agreed on in their discussion.\n"
        f"4. Attribute every merged change:\n"
        f"   # [Agent 1] <change> — serves {a1_focus}\n"
        f"   # [Agent 2] <change> — serves {a2_focus}\n"
        f"   # [Merged]  <compromise>\n"
        f"   # [Discussion] <idea from the debate>\n"
        f"5. Do NOT remove val_bpb / val_loss print lines.\n"
        f"6. Must be valid Python that runs without errors."
    )

# ── Parallel streaming ────────────────────────────────────────────────────────
async def stream_agent(client, system, messages, on_token) -> str:
    full = ""
    async with client.messages.stream(
        model="claude-sonnet-4-5", max_tokens=4000,
        system=system, messages=messages,
    ) as s:
        async for text in s.text_stream:
            full += text; on_token(full)
    return full

async def run_parallel_coding(client, a1_cfg, a2_cfg, timer_secs):
    """Fires both agent API streams and the timer simultaneously via asyncio.gather."""
    chars = [0, 0]; done = [False, False]; t_val = [timer_secs]
    sys.stdout.write("\n\n")

    def draw():
        def bar(n):
            f = min(20, round(min(1.0, n / 4000) * 20))
            return GRAY + "█" * f + DIM + "░" * (20 - f) + R
        t = t_val[0]
        ts = f"{t // 60}:{t % 60:02d}" if t > 0 else dim("done")
        sys.stdout.write(
            f"\x1b[2A"
            f"\r  {bl('Agent 1')} {bar(chars[0])} {chars[0]:>5} chars  {gr('✓') if done[0] else dim('…')}\n"
            f"\r  {yl('Agent 2')} {bar(chars[1])} {chars[1]:>5} chars  {gr('✓') if done[1] else dim('…')}"
            f"   {dim('timer:')} {ts}   \n"
        )
        sys.stdout.flush()

    draw()

    async def timer():
        while t_val[0] > 0:
            await asyncio.sleep(1); t_val[0] -= 1; draw()

    results = await asyncio.gather(
        stream_agent(client, a1_cfg["system"], a1_cfg["messages"],
                     lambda f: [chars.__setitem__(0, len(f)), draw()]),
        stream_agent(client, a2_cfg["system"], a2_cfg["messages"],
                     lambda f: [chars.__setitem__(1, len(f)), draw()]),
        timer(),
    )
    done[0] = done[1] = True; draw(); sys.stdout.write("\n")
    return strip_fences(results[0]), strip_fences(results[1])

async def stream_one(client, system, messages, max_tokens=4000) -> str:
    full = ""
    async with client.messages.stream(
        model="claude-sonnet-4-5", max_tokens=max_tokens,
        system=system, messages=messages,
    ) as s:
        async for text in s.text_stream:
            sys.stdout.write(text); sys.stdout.flush(); full += text
    sys.stdout.write("\n")
    return full

# ── Discussion ────────────────────────────────────────────────────────────────
async def run_discussion(client, a1r1, a1r2, a2r1, a2r2,
                         run1a1, run2a2, a1_focus, a2_focus,
                         n_turns, a1_instr, a2_instr):
    transcript = []
    a1_sys = make_discussion_system(1, a1_focus, a2_focus, a1_instr)
    a2_sys = make_discussion_system(2, a2_focus, a1_focus, a2_instr)
    a1_hist: list[dict] = []
    a2_hist: list[dict] = []

    for turn in range(n_turns):
        # Agent 1 speaks
        if turn == 0:
            a1_hist.append({"role": "user", "content": (
                f"Your code (Agent 1):\n\n"
                f"=== Round 1 (Code+Summary, {a1_focus}) ===\n{a1r1}\n"
                f"GPU: val_bpb={bpb_str(run1a1)}\n\n"
                f"=== Round 2 (Plan+Code, {a1_focus}) ===\n{a1r2}\n\n"
                f"Agent 2's code (FIRST TIME seeing this):\n\n"
                f"=== Agent 2 Round 1 (Plan+Code, {a2_focus}) ===\n{a2r1}\n\n"
                f"=== Agent 2 Round 2 (Code+Summary, {a2_focus}) ===\n{a2r2}\n"
                f"Agent 2 GPU: val_bpb={bpb_str(run2a2)}\n\n"
                f"Open the discussion. What do you see?"
            )})
        else:
            a1_hist.append({"role": "user", "content":
                f"Agent 2 said:\n\n{transcript[-1]['text']}\n\nRespond and refine your proposal."})

        section(f"DISCUSSION  Turn {turn+1}/{n_turns}  ·  {bl('Agent 1')} speaking", BLUE + B)
        a1_reply = await stream_one(client, a1_sys, a1_hist, max_tokens=600)
        a1_hist.append({"role": "assistant", "content": a1_reply})
        transcript.append({"agent": 1, "turn": turn+1, "focus": a1_focus, "text": a1_reply})

        # Agent 2 responds
        if turn == 0:
            a2_hist.append({"role": "user", "content": (
                f"Your code (Agent 2):\n\n"
                f"=== Round 1 (Plan+Code, {a2_focus}) ===\n{a2r1}\n\n"
                f"=== Round 2 (Code+Summary, {a2_focus}) ===\n{a2r2}\n"
                f"GPU: val_bpb={bpb_str(run2a2)}\n\n"
                f"Agent 1's code (FIRST TIME seeing this):\n\n"
                f"=== Agent 1 Round 1 (Code+Summary, {a1_focus}) ===\n{a1r1}\n"
                f"Agent 1 GPU: val_bpb={bpb_str(run1a1)}\n\n"
                f"=== Agent 1 Round 2 (Plan+Code, {a1_focus}) ===\n{a1r2}\n\n"
                f"Agent 1 opened:\n\n{a1_reply}\n\nRespond."
            )})
        else:
            a2_hist.append({"role": "user", "content":
                f"Agent 1 said:\n\n{a1_reply}\n\nRespond and refine your proposal."})

        section(f"DISCUSSION  Turn {turn+1}/{n_turns}  ·  {yl('Agent 2')} speaking", YELLOW + B)
        a2_reply = await stream_one(client, a2_sys, a2_hist, max_tokens=600)
        a2_hist.append({"role": "assistant", "content": a2_reply})
        transcript.append({"agent": 2, "turn": turn+1, "focus": a2_focus, "text": a2_reply})

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# Discussion Transcript\n\n_Generated {ts}_\n"]
    for e in transcript:
        lines.append(f"## Agent {e['agent']} · Turn {e['turn']} · {e['focus']}\n\n{e['text']}\n\n---\n")
    return transcript, "\n".join(lines)

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(
        prog="dual-agent-research",
        description="Two Claude agents improve train_gpt.py in parallel on separate git branches.",
    )
    parser.add_argument("--nproc",    type=int, default=1,
                        help="GPUs for torchrun (default: 1)")
    parser.add_argument("--gpu-time", type=int, default=150,
                        help="GPU test seconds per run (default: 150 = 2.5 min)")
    parser.add_argument("--timer",    type=int, default=120,
                        help="Coding round timer in seconds (default: 120)")
    parser.add_argument("--discuss",  type=int, default=2,
                        help="Discussion turns (default: 2)")
    parser.add_argument("--out",      default=".agents",
                        help="Output directory (default: .agents)")
    parser.add_argument("--a1",  default="AGENT_1_INSTRUCTIONS.md",
                        help="Agent 1 instructions file (default: AGENT_1_INSTRUCTIONS.md)")
    parser.add_argument("--a2",  default="AGENT_2_INSTRUCTIONS.md",
                        help="Agent 2 instructions file (default: AGENT_2_INSTRUCTIONS.md)")
    args = parser.parse_args()

    out_dir = Path(args.out)

    # ── Validate ──────────────────────────────────────────────────────────────
    if git_try("rev-parse --show-toplevel") is None:
        print(rd("Error: not a git repository.")); sys.exit(1)

    if not Path(TARGET_FILE).exists():
        print(rd(f"Error: {TARGET_FILE} not found. Run from the parameter-golf repo root."))
        sys.exit(1)

    if not Path("./data/datasets/fineweb10B_sp1024").exists():
        print(c(YELLOW, "Warning: ./data/datasets/fineweb10B_sp1024/ not found."))
        print(dim("  Run: python3 data/cached_challenge_fineweb.py --variant sp1024"))

    repo_root   = Path(git("rev-parse --show-toplevel"))
    base_branch = git("rev-parse --abbrev-ref HEAD")
    base_code   = Path(TARGET_FILE).read_text(encoding="utf-8")
    a1_instr    = load_instructions(Path(args.a1))
    a2_instr    = load_instructions(Path(args.a2))
    a1_focus    = "Architecture & Model Efficiency"
    a2_focus    = "Quantization & Compression"

    client = anthropic.AsyncAnthropic()   # reads ANTHROPIC_API_KEY from env

    # ── Banner ────────────────────────────────────────────────────────────────
    print(f"""
{bold(CYAN)}╔══════════════════════════════════════════════════════════════╗{R}
{bold(CYAN)}║{R}  {bold(CYAN)}Dual Agent Research  ·  parameter-golf edition{R}              {bold(CYAN)}║{R}
{bold(CYAN)}║{R}  {dim("claude-sonnet-4-5  ·  target: train_gpt.py")}         {bold(CYAN)}║{R}
{bold(CYAN)}╚══════════════════════════════════════════════════════════════╝{R}
  {dim("target:")}      {bold(TARGET_FILE)}  (only file agents touch)
  {dim("metric:")}      {bold("val_bpb")}  (lower is better · 16MB model limit)
  {bl("Agent 1")}    →  {a1_focus}   →  branch: {bold("agent-1")}
  {yl("Agent 2")}    →  {a2_focus}   →  branch: {bold("agent-2")}
  {dim("GPU test:")}    {args.gpu_time}s per test · {args.nproc} GPU(s) via torchrun
  {dim("discussion:")}  {args.discuss} turns × 2 agents = {args.discuss * 2} messages
  {dim("base:")}        {base_branch}
""")

    # ── Git setup ─────────────────────────────────────────────────────────────
    section("GIT SETUP  ·  Creating branches and worktrees")

    wt1 = (out_dir / "worktree-1").resolve()
    wt2 = (out_dir / "worktree-2").resolve()

    for wt in [wt1, wt2]:
        git_try(f'worktree remove --force "{wt}"')
    for b in ["agent-1", "agent-2", "synthesis"]:
        git_try(f"branch -D {b}")

    git("checkout -b agent-1"); git(f"checkout {base_branch}")
    git("checkout -b agent-2"); git(f"checkout {base_branch}")
    git(f'worktree add "{wt1}" agent-1')
    git(f'worktree add "{wt2}" agent-2')

    f1 = wt1 / TARGET_FILE
    f2 = wt2 / TARGET_FILE
    print(f"  {gr('✓')}  {bold('agent-1')}  →  {dim(str(wt1))}")
    print(f"  {gr('✓')}  {bold('agent-2')}  →  {dim(str(wt2))}")

    def task_msg(own_prev=None):
        msg = f"Improve train_gpt.py to get the lowest possible val_bpb within 16MB.\n\nCurrent {TARGET_FILE}:\n{base_code}"
        if own_prev:
            msg += f"\n\nYour previous round's code — improve it, do not restart:\n{own_prev}"
        return msg

    # ── Round 1 ───────────────────────────────────────────────────────────────
    section("ROUND 1  ·  Agent 1: Code+Summary  |  Agent 2: Plan+Code", BLUE + B)
    print(f"  {dim('No cross-agent context. Each sees only the base file.')}\n")

    a1r1, a2r1 = await run_parallel_coding(
        client,
        {"system": make_coding_system("CODE then add SUMMARY comment", True, a1_instr),
         "messages": [{"role": "user", "content": task_msg()}]},
        {"system": make_coding_system("PLAN then CODE — no summary", False, a2_instr),
         "messages": [{"role": "user", "content": task_msg()}]},
        args.timer,
    )

    f1.parent.mkdir(parents=True, exist_ok=True)
    f1.write_text(a1r1, encoding="utf-8")
    section(f"GPU TEST  ·  Agent 1 Round 1  ({args.gpu_time}s)", BLUE)
    run1a1 = run_gpu_test(f1, args.nproc, args.gpu_time, wt1, "a1_r1", out_dir)
    print(f"  val_bpb: {fmt_bpb(run1a1)}   size: {fmt_mb(run1a1)}")
    git(f'add "{TARGET_FILE}"', wt1)
    git(f'commit -m "Agent 1 R1: Architecture [{bpb_str(run1a1)}]"', wt1)
    print(f"  {gr('✓')}  Committed to {bold('agent-1')}")

    f2.parent.mkdir(parents=True, exist_ok=True)
    f2.write_text(a2r1, encoding="utf-8")
    git(f'add "{TARGET_FILE}"', wt2)
    git(f'commit -m "Agent 2 R1: Compression/Plan"', wt2)
    print(f"  {gr('✓')}  Committed to {bold('agent-2')}")

    # ── Round 2 ───────────────────────────────────────────────────────────────
    section("ROUND 2  ·  Agent 1: Plan+Code  |  Agent 2: Code+Summary  (swapped)", YELLOW + B)
    print(f"  {dim('Each agent builds on its own Round 1. Still no cross-agent context.')}\n")

    a1r2, a2r2 = await run_parallel_coding(
        client,
        {"system": make_coding_system("PLAN then CODE — no summary", False, a1_instr),
         "messages": [{"role": "user", "content": task_msg(a1r1)}]},
        {"system": make_coding_system("CODE then add SUMMARY comment", True, a2_instr),
         "messages": [{"role": "user", "content": task_msg(a2r1)}]},
        args.timer,
    )

    f1.write_text(a1r2, encoding="utf-8")
    git(f'add "{TARGET_FILE}"', wt1)
    git(f'commit -m "Agent 1 R2: Architecture/Plan"', wt1)
    print(f"\n  {gr('✓')}  Committed to {bold('agent-1')}")

    f2.write_text(a2r2, encoding="utf-8")
    section(f"GPU TEST  ·  Agent 2 Round 2  ({args.gpu_time}s)", YELLOW)
    run2a2 = run_gpu_test(f2, args.nproc, args.gpu_time, wt2, "a2_r2", out_dir)
    print(f"  val_bpb: {fmt_bpb(run2a2)}   size: {fmt_mb(run2a2)}")
    git(f'add "{TARGET_FILE}"', wt2)
    git(f'commit -m "Agent 2 R2: Compression [{bpb_str(run2a2)}]"', wt2)
    print(f"  {gr('✓')}  Committed to {bold('agent-2')}")

    # ── Reveal ────────────────────────────────────────────────────────────────
    section("REVEAL  ·  Independent work done — entering discussion")
    print(f"  {bl('agent-1')}  2 commits   val_bpb: {fmt_bpb(run1a1)}   ({a1_focus})")
    print(f"  {yl('agent-2')}  2 commits   val_bpb: {fmt_bpb(run2a2)}   ({a2_focus})")
    print(f"\n  Agents see each other's code for the first time now.")

    # ── Discussion ────────────────────────────────────────────────────────────
    transcript, disc_md = await run_discussion(
        client, a1r1, a1r2, a2r1, a2r2,
        run1a1, run2a2, a1_focus, a2_focus,
        args.discuss, a1_instr, a2_instr,
    )
    save(out_dir / "discussion.md", disc_md)
    print(f"\n  {gr('✓')}  Discussion → {bold(str(out_dir / 'discussion.md'))}")

    # ── Synthesis ─────────────────────────────────────────────────────────────
    section("SYNTHESIS  ·  Implementing the agreed merge", GREEN + B)

    disc_summary = "\n".join(
        f"--- Agent {e['agent']} Turn {e['turn']} ({e['focus']}) ---\n{e['text']}"
        for e in transcript
    )
    synth_content = (
        f"=== Agent 1 R1 Code+Summary ({a1_focus}) ===\n{a1r1}\nGPU: val_bpb={bpb_str(run1a1)}\n\n"
        f"=== Agent 2 R1 Plan+Code ({a2_focus}) ===\n{a2r1}\n\n"
        f"=== Agent 1 R2 Plan+Code ({a1_focus}) ===\n{a1r2}\n\n"
        f"=== Agent 2 R2 Code+Summary ({a2_focus}) ===\n{a2r2}\nGPU: val_bpb={bpb_str(run2a2)}\n\n"
        f"=== Discussion ({args.discuss * 2} messages) ===\n{disc_summary}\n\n"
        f"Implement the merged solution the agents agreed on."
    )

    print()
    merged_raw = await stream_one(
        client,
        make_synthesis_system(a1_focus, a2_focus, args.discuss),
        [{"role": "user", "content": synth_content}],
        max_tokens=5000,
    )
    merged = strip_fences(merged_raw)

    git("checkout -b synthesis")
    merged_path = repo_root / TARGET_FILE
    merged_path.write_text(merged, encoding="utf-8")

    section(f"GPU TEST  ·  Synthesis  ({args.gpu_time}s)", GREEN)
    merged_run = run_gpu_test(merged_path, args.nproc, args.gpu_time,
                              repo_root, "synthesis", out_dir)
    print(f"  val_bpb: {fmt_bpb(merged_run)}   size: {fmt_mb(merged_run)}")
    git(f'add "{TARGET_FILE}"')
    git(f'commit -m "Synthesis: agent-1 + agent-2 [{bpb_str(merged_run)}]"')
    print(f"  {gr('✓')}  Committed to {bold('synthesis')}")

    # ── QA report ─────────────────────────────────────────────────────────────
    section("QA REPORT")
    def pf(r): return "PASS" if r.get("ok") else "FAIL"
    qa_sys = (
        "QA engineer for the OpenAI Parameter Golf challenge.\n"
        "Write a concise Markdown report.\n\n"
        "# QA Report\n\n"
        "## val_bpb Results\n"
        "| Branch    | val_bpb | Size | Result |\n"
        "|-----------|---------|------|--------|\n"
        f"| agent-1   | {bpb_str(run1a1)} | {run1a1.get('model_mb','?')} MB | {pf(run1a1)} |\n"
        f"| agent-2   | {bpb_str(run2a2)} | {run2a2.get('model_mb','?')} MB | {pf(run2a2)} |\n"
        f"| synthesis | {bpb_str(merged_run)} | {merged_run.get('model_mb','?')} MB | {pf(merged_run)} |\n\n"
        f"## {a1_focus} Contributions\n<what agent 1 brought>\n\n"
        f"## {a2_focus} Contributions\n<what agent 2 brought>\n\n"
        "## Key Discussion Outcomes\n<ideas that changed because of the debate>\n\n"
        "## Verdict\nPASS or FAIL — one sentence."
    )
    print()
    report = await stream_one(client, qa_sys, [{"role": "user", "content":
        f"Discussion:\n{disc_summary}\n\nMerged code (first 2000 chars):\n{merged[:2000]}\n\n"
        f"GPU log:\n{merged_run.get('log','')[:1000]}"}], max_tokens=1200)
    save(out_dir / "report.md", report)
    print(f"\n  {gr('✓')}  Report → {bold(str(out_dir / 'report.md'))}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    git_try(f'worktree remove --force "{wt1}"')
    git_try(f'worktree remove --force "{wt2}"')

    # ── Done ──────────────────────────────────────────────────────────────────
    section("DONE", GREEN + B)
    scores = [(r.get("val_bpb"), n) for r, n in [
        (run1a1, "agent-1"), (run2a2, "agent-2"), (merged_run, "synthesis")
    ] if r.get("val_bpb")]
    if scores:
        best, name = min(scores, key=lambda x: x[0])
        print(f"\n  {gr('Best val_bpb:')} {bold(f'{best:.4f}')} from {bold(name)}")

    print(f"""
  {dim("Branches:")}
    {bl("agent-1")}    {a1_focus}
    {yl("agent-2")}    {a2_focus}
    {gr("synthesis")}  Merged result   ← you are here

  {dim("Inspect:")}
    git log --oneline agent-1
    git log --oneline agent-2
    git diff agent-1 agent-2 -- {TARGET_FILE}
    git diff {base_branch} synthesis -- {TARGET_FILE}

  {dim("Output files:")}
    {args.out}/discussion.md   ← agent debate transcript
    {args.out}/report.md       ← QA report with val_bpb table
    {args.out}/a1_r1.log       ← Agent 1 GPU log
    {args.out}/a2_r2.log       ← Agent 2 GPU log
    {args.out}/synthesis.log   ← Synthesis GPU log
""")

if __name__ == "__main__":
    asyncio.run(main())