#!/usr/bin/env python3
"""
dual-agent-research.py  —  parameter-golf edition

Pipelined flow so thinking never waits for GPU and GPU never waits for thinking:

  Pre-start : Agent 1 thinks  (cold start)
  Cycle 1   : Agent 1 GPU (150s)  ∥  Agent 2 thinks
              Agent 2 GPU (150s)  ∥  Agent 1 thinks
  Cycle 2   : Agent 1 GPU (150s)  ∥  Agent 2 thinks
              Agent 2 GPU (150s)  ∥  Agent 1 thinks
  Cycle 3   : Agent 1 GPU (150s)  ∥  Agent 2 thinks
              Agent 2 GPU (150s)
  Discussion: multi-turn debate — agents see each other for the first time
  Synthesis : best merged code → 10-minute final GPU run
  Log       : full run log written to synthesis branch only

Branches created:
  agent-1    3 commits  (one per cycle)
  agent-2    3 commits  (one per cycle)
  synthesis  merged result + RUN_LOG.md

Usage:
  python dual-agent-research.py [options]

Options:
  --nproc     GPUs for torchrun            (default: 1)
  --gpu-time  Seconds per cycle GPU test   (default: 150)
  --cycles    How many cycles to run       (default: 3)
  --discuss   Discussion turns             (default: 2)
  --out       Output directory             (default: .agents)
  --a1        Agent 1 instructions file    (default: AGENT_1_INSTRUCTIONS.md)
  --a2        Agent 2 instructions file    (default: AGENT_2_INSTRUCTIONS.md)

Requirements:
  pip install anthropic
  export ANTHROPIC_API_KEY=sk-ant-...
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
def ansi_strip(s): return re.sub(r'\x1b\[[0-9;]*m', '', s)

TARGET_FILE  = "train_gpt.py"
FINAL_SECS   = 600   # 10 minutes for the synthesis run

# ── Logger ────────────────────────────────────────────────────────────────────
class Logger:
    """Prints to terminal and accumulates a clean log for the synthesis branch."""

    def __init__(self):
        self.lines: list[str] = []
        self.start = datetime.now()

    def __call__(self, *args, **kwargs):
        text = " ".join(str(a) for a in args)
        print(text, **kwargs)
        self.lines.append(ansi_strip(text))

    def section(self, title: str):
        bar = "─" * 64
        self(f"\n{bar}\n  {title}\n{bar}")

    def code_block(self, label: str, code: str, max_lines: int = 60):
        self(f"\n### {label}\n```python")
        for line in code.split("\n")[:max_lines]:
            self(line)
        if code.count("\n") > max_lines:
            self(f"... ({code.count(chr(10)) - max_lines} more lines)")
        self("```")

    def gpu_result(self, label: str, r: dict):
        bpb  = f"{r['val_bpb']:.4f}" if r.get("val_bpb") else "N/A"
        mb   = f"{r['model_mb']:.1f} MB" if r.get("model_mb") else "? MB"
        ok   = "PASS" if r.get("ok") else "FAIL"
        self(f"  {label}  val_bpb={bpb}  size={mb}  [{ok}]")

    def flush_to(self, path: Path, args_summary: str):
        elapsed = datetime.now() - self.start
        header = (
            f"# RUN_LOG.md\n\n"
            f"**Generated:** {self.start.strftime('%Y-%m-%d %H:%M')}  \n"
            f"**Duration:** {str(elapsed).split('.')[0]}  \n"
            f"**Config:** {args_summary}  \n\n"
            f"---\n\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + "\n".join(self.lines), encoding="utf-8")

log = Logger()

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

# ── GPU test ──────────────────────────────────────────────────────────────────
def _run_gpu_test_sync(file: Path, nproc: int, gpu_time: int,
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
    try:
        r = subprocess.run(cmd, shell=True, cwd=str(cwd), env=env,
                           capture_output=True, text=True, timeout=gpu_time + 60)
        out = (r.stdout + r.stderr).strip()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{run_id}.log").write_text(out, encoding="utf-8")
        bpb  = re.search(r"val_bpb[:\s=]+([0-9]+\.[0-9]+)", out)
        loss = re.search(r"val_loss[:\s=]+([0-9]+\.[0-9]+)", out)
        size = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*MB", out)
        return {
            "ok":       r.returncode == 0 and bpb is not None,
            "val_bpb":  float(bpb.group(1))  if bpb  else None,
            "val_loss": float(loss.group(1)) if loss else None,
            "model_mb": float(size.group(1)) if size else None,
            "log":      out[:4000],
            "run_id":   run_id,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "val_bpb": None, "val_loss": None,
                "model_mb": None, "log": "timeout", "run_id": run_id}
    except Exception as e:
        return {"ok": False, "val_bpb": None, "val_loss": None,
                "model_mb": None, "log": str(e), "run_id": run_id}

async def run_gpu_test(file: Path, nproc: int, gpu_time: int,
                       cwd: Path, run_id: str, out_dir: Path) -> dict:
    """
    Async wrapper so the GPU subprocess runs in a thread pool
    and the event loop stays free for agent API calls to run concurrently.
    """
    return await asyncio.to_thread(
        _run_gpu_test_sync, file, nproc, gpu_time, cwd, run_id, out_dir)

# ── Preflight check ───────────────────────────────────────────────────────────
def preflight_check(nproc: int, out_dir: Path) -> bool:
    """
    Run train_gpt.py for 30 seconds on the ORIGINAL file.
    Print the full raw output so the user can see exactly what torchrun produces.
    Confirm val_bpb is captured by the regex before any GPU time is spent on agents.
    Exits the script if val_bpb is not found, showing what to fix.
    """
    PREFLIGHT_SECS = 30
    env = {
        **os.environ,
        "RUN_ID":                "preflight",
        "DATA_PATH":             "./data/datasets/fineweb10B_sp1024/",
        "TOKENIZER_PATH":        "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE":            "1024",
        "MAX_WALLCLOCK_SECONDS": str(PREFLIGHT_SECS),
        "VAL_LOSS_EVERY":        "0",
    }
    cmd = f"torchrun --standalone --nproc_per_node={nproc} {TARGET_FILE}"

    log(f"\n  Running preflight: {dim(cmd)}")
    log(f"  {dim(f'Duration: {PREFLIGHT_SECS}s  —  uses original {TARGET_FILE}')}\n")

    try:
        r = subprocess.run(cmd, shell=True, env=env,
                           capture_output=True, text=True,
                           timeout=PREFLIGHT_SECS + 30)
        out = (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        log(rd("  ✗  Preflight timed out — torchrun may not be installed"))
        return False
    except Exception as e:
        log(rd(f"  ✗  Preflight failed: {e}"))
        return False

    # Show the last 60 lines of raw output so user can inspect it
    lines = out.split("\n")
    log(f"\n  {'─'*60}")
    log(f"  RAW OUTPUT  (last {min(60, len(lines))} lines):")
    log(f"  {'─'*60}")
    for line in lines[-60:]:
        log(f"  {line}")
    log(f"  {'─'*60}\n")

    # Save for reference
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "preflight.log").write_text(out, encoding="utf-8")

    # Check every metric the script parses
    bpb   = re.search(r"val_bpb[:\s=]+([0-9]+\.[0-9]+)", out)
    loss  = re.search(r"val_loss[:\s=]+([0-9]+\.[0-9]+)", out)
    size  = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*MB", out)

    log(f"  {'─'*60}")
    log(f"  PREFLIGHT RESULTS:")
    log(f"  {'─'*60}")

    all_ok = True
    if bpb:
        log(f"  {gr('✓')}  val_bpb   found  →  {bold(bpb.group(1))}")
    else:
        log(f"  {rd('✗')}  val_bpb   NOT FOUND in output")
        log(f"      The script looks for: val_bpb  followed by : = or space + a number")
        log(f"      Check preflight.log and find the actual line, then update the regex.")
        all_ok = False

    if loss:
        log(f"  {gr('✓')}  val_loss  found  →  {bold(loss.group(1))}")
    else:
        log(f"  {c(YELLOW, '⚠')}  val_loss  not found  (non-fatal)")

    if size:
        log(f"  {gr('✓')}  model_mb  found  →  {bold(size.group(1))} MB")
    else:
        log(f"  {c(YELLOW, '⚠')}  model_mb  not found  (non-fatal)")

    if r.returncode != 0:
        log(f"  {rd('✗')}  torchrun exit code {r.returncode}  —  training crashed")
        log(f"      Full output saved to {str(out_dir / 'preflight.log')}")
        all_ok = False

    log(f"  {'─'*60}\n")

    if not all_ok:
        log(rd("  Preflight failed. Fix the issues above before running the full pipeline."))
        log(dim(f"  Full output saved to: {str(out_dir / 'preflight.log')}"))
        sys.exit(1)

    log(f"  {gr('✓')}  Preflight passed — val_bpb will be captured correctly.\n")
    return True

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_python(text: str) -> str:
    # Robustly extract Python from model output — handles three cases:
    # 1. Fenced in ```python ... ``` blocks
    # 2. Prose before code — find where imports/docstring start
    # 3. Already clean Python — return as-is
    # Case 1: fenced code block
    m = re.search(r"```(?:python)?\n([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()

    # Case 2: prose before code — find where Python actually starts
    # train_gpt.py always starts with a docstring or imports
    for marker in (r"^from __future__", r"^\"\"\"", r"^import ", r"^#"):
        m = re.search(marker, text, re.MULTILINE)
        if m:
            return text[m.start():].strip()

    # Case 3: just return as-is
    return text.strip()

def save(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def bpb_str(r: dict) -> str:
    return f"{r['val_bpb']:.4f}" if r.get("val_bpb") else "N/A"

def load_instructions(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    log(f"  {c(YELLOW, '⚠')}  {path} not found")
    return ""

def build_file_map(code: str) -> str:
    """Compact line-number index of the file so agents know where things are."""
    lines = code.split("\n")
    entries = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if any(s.startswith(k) for k in ("import ", "from ")):
            entries.append(f"L{i:4d}  {s[:80]}")
        elif any(s.startswith(k) for k in ("class ", "def ", "async def ")):
            entries.append(f"L{i:4d}  {s[:80]}")
        elif "=" in s and not s.startswith("#") and i < 200:
            if not s.startswith(("if ", "for ", "while ", "return ")):
                entries.append(f"L{i:4d}  {s[:80]}")
    return f"{TARGET_FILE} ({len(lines)} lines total)\n\n" + "\n".join(entries[:120])

# ── Prompts ───────────────────────────────────────────────────────────────────
def coding_system() -> str:
    return (
        "You are an expert ML engineer and researcher.\n"
        "You write clean, correct Python that runs first time.\n"
        "You make surgical improvements — never rewrite working code.\n"
        "Output only raw Python source code. No markdown fences, no explanation."
    )

def coding_prompt(base_code: str, instructions: str,
                  own_prev: str = None, cycle: int = 1) -> str:
    file_map = build_file_map(base_code)
    context = (
        f"This is cycle {cycle}. Your previous cycle's code is below — improve it.\n\n"
        f"YOUR PREVIOUS CODE:\n{own_prev}\n\n"
        f"ORIGINAL FILE (for structural reference only):\n{base_code}"
        if own_prev else
        f"This is cycle 1 — your first pass at the file.\n\n"
        f"CURRENT FILE:\n{base_code}"
    )
    return (
        f"You are improving train_gpt.py for the OpenAI Parameter Golf challenge.\n"
        f"Goal: lowest val_bpb (bits per byte) in a model compressed to under 16MB.\n\n"
        f"YOUR FOCUS FOR THIS CYCLE:\n{instructions}\n\n"
        f"FILE MAP (line numbers):\n{file_map}\n\n"
        f"{context}\n\n"
        f"INSTRUCTIONS:\n"
        f"- Output the COMPLETE train_gpt.py with your changes applied\n"
        f"- Make 2-4 targeted changes — do not restructure or rewrite the file\n"
        f"- Mark every changed line with: # [CHANGED] reason\n"
        f"- Do not remove val_bpb / val_loss print lines\n"
        f"- Raw Python only — no markdown, no explanation outside the file\n"
    )

def discussion_system(agent_num: int, my_focus: str,
                      other_focus: str, instructions: str) -> str:
    return (
        f"You are Agent {agent_num} in a technical peer review of train_gpt.py.\n"
        f"Your focus: {my_focus}. Other agent's focus: {other_focus}.\n\n"
        f"Rules:\n"
        f"- Reference specific function names, hyperparameter values, line numbers\n"
        f"- Argue from your focus — explain val_bpb impact concretely\n"
        f"- Acknowledge what they got right; push back on what's weak\n"
        f"- Under 250 words. No code.\n"
        f"- End with: 'My proposal: keep [X from mine], keep [Y from theirs]'\n\n"
        f"{instructions}"
    )

def synthesis_system(a1_focus: str, a2_focus: str) -> str:
    return (
        f"You are a senior ML engineer merging two train_gpt.py versions.\n"
        f"Agent 1 focused on: {a1_focus}.\n"
        f"Agent 2 focused on: {a2_focus}.\n\n"
        f"You write clean, correct Python that runs first time.\n"
        f"Implement exactly what both agents agreed on in the discussion.\n"
        f"Mark every merged change: # [Merged] reason\n"
        f"Output only raw Python. No markdown fences."
    )

# ── Streaming ─────────────────────────────────────────────────────────────────
async def think(client, system: str, prompt: str,
                label: str, max_tokens: int = 8096) -> str:
    """
    Single agent thinking call. Streams to terminal with a label prefix.
    Non-blocking — can run concurrently with a GPU test via asyncio.gather.
    """
    log(f"  {dim('⟳')}  {label} thinking...")
    full = ""
    async with client.messages.stream(
        model="claude-sonnet-4-5", max_tokens=max_tokens,
        system=system, messages=[{"role": "user", "content": prompt}],
    ) as s:
        async for text in s.text_stream:
            full += text
    log(f"  {gr('✓')}  {label} done")
    return extract_python(full)

async def stream_one(client, system: str, user: str,
                     max_tokens: int = 4096, silent: bool = False) -> str:
    full = ""
    async with client.messages.stream(
        model="claude-sonnet-4-5", max_tokens=max_tokens,
        system=system, messages=[{"role": "user", "content": user}],
    ) as s:
        async for text in s.text_stream:
            full += text
    if not silent:
        log(full)
    return full

# ── Discussion ────────────────────────────────────────────────────────────────
async def run_discussion(client, a1_codes, a2_codes, a1_results, a2_results,
                         a1_focus, a2_focus, n_turns, a1_instr, a2_instr) -> tuple[list, str]:
    """
    Multi-turn sequential debate.
    Agents see each other's code for the first time here.
    """
    transcript = []
    a1_sys = discussion_system(1, a1_focus, a2_focus, a1_instr)
    a2_sys = discussion_system(2, a2_focus, a1_focus, a2_instr)
    a1_hist: list[dict] = []
    a2_hist: list[dict] = []

    # Build compact history summaries
    a1_summary = "\n\n".join(
        f"Cycle {i+1} (val_bpb={bpb_str(r)}):\n{c[:300]}..."
        for i, (c, r) in enumerate(zip(a1_codes, a1_results))
    )
    a2_summary = "\n\n".join(
        f"Cycle {i+1} (val_bpb={bpb_str(r)}):\n{c[:300]}..."
        for i, (c, r) in enumerate(zip(a2_codes, a2_results))
    )

    for turn in range(n_turns):
        # Agent 1 speaks
        if turn == 0:
            a1_hist.append({"role": "user", "content": (
                f"Your code across {len(a1_codes)} cycles (Agent 1, focus: {a1_focus}):\n{a1_summary}\n\n"
                f"Agent 2's code (FIRST TIME seeing this, focus: {a2_focus}):\n{a2_summary}\n\n"
                f"Best val_bpb you achieved: {bpb_str(min(a1_results, key=lambda r: r.get('val_bpb') or 999))}\n"
                f"Best val_bpb they achieved: {bpb_str(min(a2_results, key=lambda r: r.get('val_bpb') or 999))}\n\n"
                f"Open the discussion. What should go into the final 10-minute run?"
            )})
        else:
            a1_hist.append({"role": "user", "content":
                f"Agent 2 said:\n{transcript[-1]['text']}\n\nRespond and refine your proposal."})

        log.section(f"DISCUSSION  Turn {turn+1}/{n_turns}  ·  Agent 1 speaking")
        a1_reply = await stream_one(client, a1_sys,
                                    a1_hist[-1]["content"], max_tokens=500, silent=True)
        a1_hist.append({"role": "assistant", "content": a1_reply})
        transcript.append({"agent": 1, "turn": turn+1, "text": a1_reply})
        log(a1_reply)

        # Agent 2 responds
        if turn == 0:
            a2_hist.append({"role": "user", "content": (
                f"Your code across {len(a2_codes)} cycles (Agent 2, focus: {a2_focus}):\n{a2_summary}\n\n"
                f"Agent 1's code (FIRST TIME seeing this, focus: {a1_focus}):\n{a1_summary}\n\n"
                f"Best val_bpb you achieved: {bpb_str(min(a2_results, key=lambda r: r.get('val_bpb') or 999))}\n"
                f"Best val_bpb they achieved: {bpb_str(min(a1_results, key=lambda r: r.get('val_bpb') or 999))}\n\n"
                f"Agent 1 opened:\n{a1_reply}\n\nRespond."
            )})
        else:
            a2_hist.append({"role": "user", "content":
                f"Agent 1 said:\n{a1_reply}\n\nRespond and refine your proposal."})

        log.section(f"DISCUSSION  Turn {turn+1}/{n_turns}  ·  Agent 2 speaking")
        a2_reply = await stream_one(client, a2_sys,
                                    a2_hist[-1]["content"], max_tokens=500, silent=True)
        a2_hist.append({"role": "assistant", "content": a2_reply})
        transcript.append({"agent": 2, "turn": turn+1, "text": a2_reply})
        log(a2_reply)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    md_lines = [f"# Discussion Transcript\n_Generated {ts}_\n"]
    for e in transcript:
        md_lines.append(f"## Agent {e['agent']} · Turn {e['turn']}\n\n{e['text']}\n\n---\n")
    return transcript, "\n".join(md_lines)

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(
        description="Pipelined dual-agent parameter-golf optimizer.")
    parser.add_argument("--nproc",    type=int, default=1,
                        help="GPUs for torchrun (default: 1)")
    parser.add_argument("--gpu-time", type=int, default=150,
                        help="GPU seconds per cycle test (default: 150)")
    parser.add_argument("--cycles",   type=int, default=3,
                        help="Pipeline cycles per agent (default: 3)")
    parser.add_argument("--discuss",  type=int, default=2,
                        help="Discussion turns (default: 2)")
    parser.add_argument("--out",      default=".agents",
                        help="Output directory (default: .agents)")
    parser.add_argument("--a1",       default="AGENT_1_INSTRUCTIONS.md")
    parser.add_argument("--a2",       default="AGENT_2_INSTRUCTIONS.md")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if git_try("rev-parse --show-toplevel") is None:
        print(rd("Error: not a git repository.")); sys.exit(1)
    if not Path(TARGET_FILE).exists():
        print(rd(f"Error: {TARGET_FILE} not found.")); sys.exit(1)

    repo_root   = Path(git("rev-parse --show-toplevel"))
    base_branch = git("rev-parse --abbrev-ref HEAD")
    base_code   = Path(TARGET_FILE).read_text(encoding="utf-8")
    a1_instr    = load_instructions(Path(args.a1))
    a2_instr    = load_instructions(Path(args.a2))
    a1_focus    = "Architecture & Model Efficiency"
    a2_focus    = "Quantization & Compression"
    client      = anthropic.AsyncAnthropic()

    args_summary = (
        f"nproc={args.nproc}  gpu-time={args.gpu_time}s  "
        f"cycles={args.cycles}  discuss={args.discuss}  final={FINAL_SECS}s"
    )

    log(f"""
{bold(CYAN)}╔══════════════════════════════════════════════════════════════╗{R}
{bold(CYAN)}║{R}  {bold(CYAN)}Dual Agent Research  ·  parameter-golf edition{R}              {bold(CYAN)}║{R}
{bold(CYAN)}╚══════════════════════════════════════════════════════════════╝{R}
  {dim("target:")}    {bold(TARGET_FILE)}
  {bl("Agent 1")}  →  {a1_focus}  →  branch: {bold("agent-1")}
  {yl("Agent 2")}  →  {a2_focus}  →  branch: {bold("agent-2")}
  {dim("pipeline:")}  {args.cycles} cycles  ·  think overlaps GPU  ·  {args.gpu_time}s per test
  {dim("discuss:")}   {args.discuss} turns × 2 agents
  {dim("final:")}     {FINAL_SECS}s GPU run → synthesis branch
  {dim("log:")}       RUN_LOG.md committed to synthesis branch only
""")

    # ── Git setup ─────────────────────────────────────────────────────────────
    log.section("GIT SETUP")
    wt1 = (out_dir / "worktree-1").resolve()
    wt2 = (out_dir / "worktree-2").resolve()
    for wt in [wt1, wt2]: git_try(f'worktree remove --force "{wt}"')
    for b in ["agent-1", "agent-2", "synthesis"]: git_try(f"branch -D {b}")
    git("checkout -b agent-1"); git(f"checkout {base_branch}")
    git("checkout -b agent-2"); git(f"checkout {base_branch}")
    git(f'worktree add "{wt1}" agent-1')
    git(f'worktree add "{wt2}" agent-2')
    f1, f2 = wt1 / TARGET_FILE, wt2 / TARGET_FILE
    f1.parent.mkdir(parents=True, exist_ok=True)
    f2.parent.mkdir(parents=True, exist_ok=True)
    log(f"  {gr('✓')}  agent-1  →  {dim(str(wt1))}")
    log(f"  {gr('✓')}  agent-2  →  {dim(str(wt2))}")

    # ── Preflight check ───────────────────────────────────────────────────────
    # Run train_gpt.py for 30s on the original file.
    # Confirms val_bpb is captured before any agent or GPU time is spent.
    log.section("PREFLIGHT CHECK  ·  30s test run on original train_gpt.py")
    preflight_check(args.nproc, out_dir)

    # ── Storage across cycles ─────────────────────────────────────────────────
    a1_codes:   list[str]  = []
    a2_codes:   list[str]  = []
    a1_results: list[dict] = []
    a2_results: list[dict] = []

    # ── PRE-START: Agent 1 cold-start think ───────────────────────────────────
    log.section("PRE-START  ·  Agent 1 cold-start think")
    a1_code = await think(
        client, coding_system(),
        coding_prompt(base_code, a1_instr, cycle=1),
        label=bl("Agent 1"), max_tokens=8096,
    )
    a1_codes.append(a1_code)

    # ── PIPELINE CYCLES ───────────────────────────────────────────────────────
    for cycle in range(1, args.cycles + 1):
        log.section(f"CYCLE {cycle}/{args.cycles}  ·  Pipeline step A")
        log(f"  {bl('Agent 1')} GPU test ({args.gpu_time}s)  ∥  {yl('Agent 2')} thinking")

        # Write Agent 1's code to its worktree
        f1.write_text(a1_codes[-1], encoding="utf-8")

        # Run Agent 1 GPU test AND Agent 2 thinks — at the same time
        a1_result, a2_code = await asyncio.gather(
            run_gpu_test(f1, args.nproc, args.gpu_time, wt1,
                         f"a1_c{cycle}", out_dir),
            think(
                client, coding_system(),
                coding_prompt(
                    base_code, a2_instr,
                    own_prev=a2_codes[-1] if a2_codes else None,
                    cycle=cycle,
                ),
                label=yl("Agent 2"),
            ),
        )

        a1_results.append(a1_result)
        a2_codes.append(a2_code)
        log.gpu_result(f"Agent 1 Cycle {cycle}", a1_result)

        # Commit Agent 1's result
        git(f'add "{TARGET_FILE}"', wt1)
        git(f'commit -m "Agent 1 Cycle {cycle} [{bpb_str(a1_result)}]"', wt1)
        log(f"  {gr('✓')}  Committed to {bold('agent-1')}  (Cycle {cycle})")

        log.section(f"CYCLE {cycle}/{args.cycles}  ·  Pipeline step B")

        # On the last cycle Agent 1 doesn't need to think (discussion follows)
        if cycle < args.cycles:
            log(f"  {yl('Agent 2')} GPU test ({args.gpu_time}s)  ∥  {bl('Agent 1')} thinking")
            f2.write_text(a2_codes[-1], encoding="utf-8")

            a2_result, a1_code_next = await asyncio.gather(
                run_gpu_test(f2, args.nproc, args.gpu_time, wt2,
                             f"a2_c{cycle}", out_dir),
                think(
                    client, coding_system(),
                    coding_prompt(
                        base_code, a1_instr,
                        own_prev=a1_codes[-1],
                        cycle=cycle + 1,
                    ),
                    label=bl("Agent 1"),
                ),
            )
            a1_codes.append(a1_code_next)
        else:
            # Last cycle — just run Agent 2's GPU test, no thinking needed
            log(f"  {yl('Agent 2')} GPU test ({args.gpu_time}s)  (final cycle — no thinking needed)")
            f2.write_text(a2_codes[-1], encoding="utf-8")
            a2_result = await run_gpu_test(
                f2, args.nproc, args.gpu_time, wt2, f"a2_c{cycle}", out_dir)

        a2_results.append(a2_result)
        log.gpu_result(f"Agent 2 Cycle {cycle}", a2_result)

        git(f'add "{TARGET_FILE}"', wt2)
        git(f'commit -m "Agent 2 Cycle {cycle} [{bpb_str(a2_result)}]"', wt2)
        log(f"  {gr('✓')}  Committed to {bold('agent-2')}  (Cycle {cycle})")

    # ── Cycle summary ─────────────────────────────────────────────────────────
    log.section("CYCLE SUMMARY")
    for i, (r1, r2) in enumerate(zip(a1_results, a2_results), 1):
        log(f"  Cycle {i}   Agent 1: val_bpb={bpb_str(r1)}   Agent 2: val_bpb={bpb_str(r2)}")

    best_a1 = min(a1_results, key=lambda r: r.get("val_bpb") or 999)
    best_a2 = min(a2_results, key=lambda r: r.get("val_bpb") or 999)
    log(f"\n  Best Agent 1: {bpb_str(best_a1)}   Best Agent 2: {bpb_str(best_a2)}")

    # ── Discussion ────────────────────────────────────────────────────────────
    log.section("REVEAL  ·  Agents see each other's work for the first time")
    log(f"  {args.discuss} turns × 2 agents  ·  goal: agree on the best 10-minute run\n")

    transcript, disc_md = await run_discussion(
        client, a1_codes, a2_codes, a1_results, a2_results,
        a1_focus, a2_focus, args.discuss, a1_instr, a2_instr,
    )
    save(out_dir / "discussion.md", disc_md)

    # ── Synthesis ─────────────────────────────────────────────────────────────
    log.section("SYNTHESIS  ·  Building the best version for the 10-minute run")

    disc_text = "\n\n".join(
        f"Agent {e['agent']} Turn {e['turn']}:\n{e['text']}" for e in transcript)

    # Give synthesis the best code from each agent (last cycle = most refined)
    synth_prompt = (
        f"You are merging two optimized versions of train_gpt.py.\n\n"
        f"Agent 1 ({a1_focus}) — best result val_bpb={bpb_str(best_a1)}\n"
        f"Agent 1 final code:\n{a1_codes[-1]}\n\n"
        f"Agent 2 ({a2_focus}) — best result val_bpb={bpb_str(best_a2)}\n"
        f"Agent 2 final code:\n{a2_codes[-1]}\n\n"
        f"Discussion and agreed proposals:\n{disc_text}\n\n"
        f"This merged version will run for the FULL 10 MINUTES — make it count.\n"
        f"Implement exactly what both agents agreed on.\n"
        f"Output the complete train_gpt.py — raw Python only."
    )

    print()
    merged_raw = await stream_one(
        client, synthesis_system(a1_focus, a2_focus), synth_prompt)
    merged = extract_python(merged_raw)

    # ── Switch to synthesis branch ────────────────────────────────────────────
    git("checkout -b synthesis")
    merged_path = repo_root / TARGET_FILE
    merged_path.write_text(merged, encoding="utf-8")

    # ── Final 10-minute GPU run ───────────────────────────────────────────────
    log.section(f"FINAL GPU RUN  ·  {FINAL_SECS}s ({FINAL_SECS // 60} minutes)  ·  synthesis branch")

    final_env = {
        **os.environ,
        "RUN_ID":                "synthesis_final",
        "DATA_PATH":             "./data/datasets/fineweb10B_sp1024/",
        "TOKENIZER_PATH":        "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE":            "1024",
        "MAX_WALLCLOCK_SECONDS": str(FINAL_SECS),
        "VAL_LOSS_EVERY":        "200",
    }
    cmd = f"torchrun --standalone --nproc_per_node={args.nproc} {TARGET_FILE}"
    log(f"  $ {cmd}")
    log(f"  Running for {FINAL_SECS}s ({FINAL_SECS // 60} min)...\n")
    try:
        r = subprocess.run(cmd, shell=True, cwd=str(repo_root), env=final_env,
                           capture_output=True, text=True, timeout=FINAL_SECS + 120)
        final_out = (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        final_out = "timeout"
    except Exception as e:
        final_out = str(e)

    (out_dir / "synthesis_final.log").write_text(final_out, encoding="utf-8")
    final_bpb  = re.search(r"val_bpb[:\s=]+([0-9]+\.[0-9]+)", final_out)
    final_size = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*MB", final_out)
    final_result = {
        "val_bpb":  float(final_bpb.group(1))  if final_bpb  else None,
        "model_mb": float(final_size.group(1)) if final_size else None,
    }
    log.gpu_result("FINAL (synthesis, 10 min)", final_result)

    # ── Commit everything to synthesis branch ─────────────────────────────────
    log.section("COMMITTING  ·  code + log → synthesis branch")

    # Write the log now so it gets committed
    log_path = repo_root / "RUN_LOG.md"
    log.flush_to(log_path, args_summary)

    git(f'add "{TARGET_FILE}" "RUN_LOG.md"')
    git(f'commit -m "Synthesis [{bpb_str(final_result)}] — full run log included"')
    log(f"  {gr('✓')}  Committed to {bold('synthesis')}  (train_gpt.py + RUN_LOG.md)")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    git_try(f'worktree remove --force "{wt1}"')
    git_try(f'worktree remove --force "{wt2}"')

    # ── Done ──────────────────────────────────────────────────────────────────
    log.section("DONE")

    all_bpb = [(r.get("val_bpb"), f"agent-1 c{i+1}") for i, r in enumerate(a1_results) if r.get("val_bpb")]
    all_bpb += [(r.get("val_bpb"), f"agent-2 c{i+1}") for i, r in enumerate(a2_results) if r.get("val_bpb")]
    if final_result.get("val_bpb"):
        all_bpb.append((final_result["val_bpb"], "synthesis (10 min)"))
    if all_bpb:
        best, name = min(all_bpb, key=lambda x: x[0])
        log(f"\n  {gr('Best val_bpb:')} {bold(f'{best:.4f}')} from {bold(name)}")

    log(f"""
  Branches:
    {bl("agent-1")}    {args.cycles} commits  ({a1_focus})
    {yl("agent-2")}    {args.cycles} commits  ({a2_focus})
    {gr("synthesis")}  merged result + RUN_LOG.md  ← you are here

  View the log:
    git checkout synthesis
    cat RUN_LOG.md

  Diff the agents:
    git diff agent-1 agent-2 -- {TARGET_FILE}
    git diff {base_branch} synthesis -- {TARGET_FILE}
""")

if __name__ == "__main__":
    asyncio.run(main())