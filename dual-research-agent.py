#!/usr/bin/env python3
"""
dual-agent-research.py  —  parameter-golf edition

Agents output SEARCH/REPLACE edits instead of the full file.
This avoids truncation on large files and produces targeted changes.

Pipeline:
  Pre-start : Agent 1 thinks (cold start)
  Cycle 1-3 : Agent 1 GPU ∥ Agent 2 thinks → Agent 2 GPU ∥ Agent 1 thinks
  Last cycle: Agent 2 GPU only (no Agent 1 think)
  Discussion: 2-turn debate
  Synthesis : merge → new branch → 10-min final run
  Log       : RUN_LOG.md in synthesis branch only
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
def bl(s):  return c(BLUE, s)
def yl(s):  return c(YELLOW, s)
def gr(s):  return c(GREEN, s)
def rd(s):  return c(RED, s)
def dim(s): return c(DIM, s)
def bold(s): return c(B, s)
def ansi_strip(s): return re.sub(r'\x1b\[[0-9;]*m', '', s)

TARGET_FILE = "train_gpt.py"
FINAL_SECS = 600

# ── Logger ────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self):
        self.lines: list[str] = []
        self.start = datetime.now()

    def __call__(self, *args, **kwargs):
        text = " ".join(str(a) for a in args)
        print(text, **kwargs)
        self.lines.append(ansi_strip(text))

    def section(self, title):
        self(f"\n{'─'*64}\n  {title}\n{'─'*64}")

    def gpu_result(self, label, r):
        bpb = f"{r['val_bpb']:.4f}" if r.get("val_bpb") else "N/A"
        mb  = f"{r['model_mb']:.1f} MB" if r.get("model_mb") else "? MB"
        ok  = "PASS" if r.get("ok") else "FAIL"
        self(f"  {label}  val_bpb={bpb}  size={mb}  [{ok}]")

    def flush_to(self, path, args_summary):
        elapsed = datetime.now() - self.start
        header = (
            f"# RUN_LOG.md\n\n"
            f"**Generated:** {self.start.strftime('%Y-%m-%d %H:%M')}  \n"
            f"**Duration:** {str(elapsed).split('.')[0]}  \n"
            f"**Config:** {args_summary}  \n\n---\n\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + "\n".join(self.lines), encoding="utf-8")

log = Logger()

# ── Git ───────────────────────────────────────────────────────────────────────
def git(cmd, cwd=None):
    r = subprocess.run(f"git {cmd}", shell=True, cwd=str(cwd or Path.cwd()),
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or r.stdout.strip())
    return r.stdout.strip()

def git_try(cmd, cwd=None):
    try: return git(cmd, cwd)
    except: return None

# ── GPU test ──────────────────────────────────────────────────────────────────
def _gpu_test_sync(file, nproc, gpu_time, cwd, run_id, out_dir):
    env = {
        **os.environ,
        "RUN_ID": run_id,
        "DATA_PATH": "./data/datasets/fineweb10B_sp1024/",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
        "MAX_WALLCLOCK_SECONDS": str(gpu_time),
        "VAL_LOSS_EVERY": "0",
    }
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
        return {
            "ok":       r.returncode == 0 and bpb is not None,
            "val_bpb":  float(bpb.group(1))  if bpb  else None,
            "val_loss": float(loss.group(1)) if loss else None,
            "model_mb": float(size.group(1)) if size else None,
            "log": out[:4000], "run_id": run_id,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "val_bpb": None, "val_loss": None,
                "model_mb": None, "log": "timeout", "run_id": run_id}
    except Exception as e:
        return {"ok": False, "val_bpb": None, "val_loss": None,
                "model_mb": None, "log": str(e), "run_id": run_id}

async def gpu_test(file, nproc, gpu_time, cwd, run_id, out_dir):
    return await asyncio.to_thread(_gpu_test_sync, file, nproc, gpu_time, cwd, run_id, out_dir)

# ── Preflight ─────────────────────────────────────────────────────────────────
def preflight_check(nproc, out_dir, repo_root):
    SECS = 30
    env = {
        **os.environ,
        "RUN_ID": "preflight",
        "DATA_PATH": "./data/datasets/fineweb10B_sp1024/",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
        "MAX_WALLCLOCK_SECONDS": str(SECS),
        "VAL_LOSS_EVERY": "0",
    }
    cmd = f"torchrun --standalone --nproc_per_node={nproc} {TARGET_FILE}"
    log(f"  Running: {dim(cmd)}  ({SECS}s)")
    try:
        r = subprocess.run(cmd, shell=True, env=env, cwd=str(repo_root),
                           capture_output=True, text=True, timeout=SECS + 120)
        out = (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        log(rd("  ✗  Timed out")); sys.exit(1)
    except Exception as e:
        log(rd(f"  ✗  Failed: {e}")); sys.exit(1)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "preflight.log").write_text(out, encoding="utf-8")

    bpb = re.search(r"val_bpb[:\s=]+([0-9]+\.[0-9]+)", out)
    if bpb and r.returncode == 0:
        log(f"  {gr('✓')}  val_bpb={bold(bpb.group(1))}  — preflight passed")
        return True
    if not bpb:
        log(rd("  ✗  val_bpb NOT FOUND in output"))
        lines = out.split("\n")
        for line in lines[-15:]:
            log(f"    {line}")
    if r.returncode != 0:
        log(rd(f"  ✗  torchrun exit code {r.returncode}"))
    log(dim(f"  Full log: {out_dir}/preflight.log"))
    sys.exit(1)

# ── Search/Replace edit engine ────────────────────────────────────────────────

def apply_edits(code: str, edits_text: str) -> str:
    """
    Parse SEARCH/REPLACE blocks and apply them to code.

    Expected format from the agent:
        <<<SEARCH
        exact lines to find in the file
        ===REPLACE
        new lines to put there
        >>>

    Returns the modified code. If a search block isn't found, it's skipped
    with a warning. If no valid edits are parsed, returns code unchanged.
    """
    pattern = r'<<<SEARCH\n(.*?)\n===REPLACE\n(.*?)\n>>>'
    matches = re.findall(pattern, edits_text, re.DOTALL)

    if not matches:
        log(f"  {c(YELLOW, '⚠')}  No SEARCH/REPLACE blocks found in agent output")
        return code

    applied = 0
    skipped = 0
    for old, new in matches:
        old = old.rstrip()
        new = new.rstrip()
        if old in code:
            code = code.replace(old, new, 1)
            applied += 1
        else:
            # Try with stripped whitespace matching
            old_stripped = "\n".join(l.rstrip() for l in old.split("\n"))
            code_stripped_check = "\n".join(l.rstrip() for l in code.split("\n"))
            if old_stripped in code_stripped_check:
                # Find position in original and replace
                idx = code_stripped_check.find(old_stripped)
                # Count how many chars to replace in original
                orig_lines = code.split("\n")
                stripped_lines = code_stripped_check.split("\n")
                # Simpler: just do the replacement on stripped then it's fine
                code = code_stripped_check.replace(old_stripped, new, 1)
                applied += 1
            else:
                skipped += 1
                first_line = old.split("\n")[0][:60]
                log(f"  {c(YELLOW, '⚠')}  Could not find: {dim(first_line)}...")

    log(f"  Applied {applied} edit(s), skipped {skipped}")
    return code


def safe_apply_edits(code: str, edits_text: str, base_code: str) -> str:
    """Apply edits, then syntax-check. Fall back to base_code if broken."""
    result = apply_edits(code, edits_text)
    try:
        ast.parse(result)
        return result
    except SyntaxError as e:
        log(f"  {c(YELLOW, '⚠')}  Edits produced SyntaxError at line {e.lineno} — using previous version")
        return code

# ── Helpers ───────────────────────────────────────────────────────────────────
def save(path, content):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(content, encoding="utf-8")

def bpb_str(r):
    return f"{r['val_bpb']:.4f}" if r.get("val_bpb") else "N/A"

def load_instructions(path):
    p = Path(path)
    if p.exists():
        return p.read_text(encoding="utf-8")
    log(f"  {c(YELLOW, '⚠')}  {path} not found")
    return ""

def build_file_map(code):
    lines = code.split("\n")
    entries = []
    for i, line in enumerate(lines, 1):
        s = line.strip()
        if any(s.startswith(k) for k in ("import ", "from ", "class ", "def ", "async def ")):
            entries.append(f"L{i:4d}  {s[:80]}")
        elif "=" in s and not s.startswith("#") and i < 200:
            if not s.startswith(("if ", "for ", "while ", "return ")):
                entries.append(f"L{i:4d}  {s[:80]}")
    return f"{TARGET_FILE} ({len(lines)} lines)\n" + "\n".join(entries[:100])

# ── Prompts ───────────────────────────────────────────────────────────────────

EDIT_FORMAT_INSTRUCTIONS = """
OUTPUT FORMAT — you MUST use this exact format for every change:

<<<SEARCH
exact lines from the current file to find (copy-paste, not paraphrased)
===REPLACE
new lines to put there
>>>

RULES:
- Each SEARCH block must be an EXACT match of existing lines in the file
- You can have multiple <<<SEARCH...>>>  blocks
- Make 2-4 targeted changes only
- Do NOT output the full file — only the SEARCH/REPLACE blocks
- No markdown fences, no explanation outside the blocks
- SEARCH text must include enough context (3-5 lines) to be unique in the file
""".strip()

def coding_system():
    return (
        "You are an expert ML engineer. You make surgical, targeted improvements.\n"
        "You output ONLY search/replace edit blocks — never the full file.\n"
        "Your edits must use exact text from the file for the SEARCH portion."
    )

def coding_prompt(current_code, instructions, cycle=1, prev_result=None):
    file_map = build_file_map(current_code)

    result_context = ""
    if prev_result and prev_result.get("val_bpb"):
        result_context = (
            f"\nYour previous cycle's GPU result: val_bpb={bpb_str(prev_result)}\n"
            f"Try to improve on this.\n"
        )

    return (
        f"You are improving train_gpt.py for the OpenAI Parameter Golf challenge.\n"
        f"Goal: lowest val_bpb (bits per byte) in a model under 16MB compressed.\n"
        f"This is cycle {cycle}.\n"
        f"{result_context}\n"
        f"YOUR FOCUS:\n{instructions}\n\n"
        f"FILE MAP:\n{file_map}\n\n"
        f"CURRENT FILE:\n{current_code}\n\n"
        f"{EDIT_FORMAT_INSTRUCTIONS}\n"
    )

def synthesis_prompt(base_code, a1_code, a2_code, a1_focus, a2_focus,
                     best_a1, best_a2, disc_text):
    return (
        f"You are merging improvements from two agents into train_gpt.py.\n\n"
        f"Agent 1 ({a1_focus}) — best val_bpb={bpb_str(best_a1)}\n"
        f"Agent 2 ({a2_focus}) — best val_bpb={bpb_str(best_a2)}\n\n"
        f"Their discussion and agreed proposals:\n{disc_text}\n\n"
        f"ORIGINAL BASE FILE:\n{base_code}\n\n"
        f"Agent 1's final version (key differences):\n"
        f"{show_diff_summary(base_code, a1_code)}\n\n"
        f"Agent 2's final version (key differences):\n"
        f"{show_diff_summary(base_code, a2_code)}\n\n"
        f"Apply the agreed changes to the ORIGINAL BASE FILE.\n"
        f"This merged version runs for 10 FULL MINUTES.\n\n"
        f"{EDIT_FORMAT_INSTRUCTIONS}\n"
    )

def show_diff_summary(base, modified):
    """Show lines that differ between base and modified (compact)."""
    base_lines = base.split("\n")
    mod_lines = modified.split("\n")
    diffs = []
    for i, (b, m) in enumerate(zip(base_lines, mod_lines)):
        if b != m:
            diffs.append(f"  L{i+1} OLD: {b.strip()[:80]}")
            diffs.append(f"  L{i+1} NEW: {m.strip()[:80]}")
    if len(mod_lines) > len(base_lines):
        diffs.append(f"  +{len(mod_lines) - len(base_lines)} new lines added at end")
    return "\n".join(diffs[:60]) if diffs else "(no differences)"

def discussion_system(agent_num, my_focus, other_focus, instructions):
    return (
        f"You are Agent {agent_num} reviewing train_gpt.py changes.\n"
        f"Your focus: {my_focus}. Other agent: {other_focus}.\n\n"
        f"Rules:\n"
        f"- Reference specific function names, hyperparameters, line numbers\n"
        f"- Argue from your focus — explain val_bpb impact\n"
        f"- Under 250 words. No code.\n"
        f"- End with: 'My proposal: keep [X from mine], keep [Y from theirs]'\n\n"
        f"{instructions}"
    )

# ── API calls ─────────────────────────────────────────────────────────────────

async def think(client, system, prompt, label, current_code, base_code,
                max_tokens=4096):
    """
    Agent thinks → outputs SEARCH/REPLACE blocks → applied to current_code.
    Returns the modified code (or current_code unchanged if edits fail).
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

    result = safe_apply_edits(current_code, full, base_code)
    return result

async def stream_one(client, system, user, max_tokens=4096, silent=False):
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
                         a1_focus, a2_focus, n_turns, a1_instr, a2_instr,
                         base_code):
    transcript = []
    a1_sys = discussion_system(1, a1_focus, a2_focus, a1_instr)
    a2_sys = discussion_system(2, a2_focus, a1_focus, a2_instr)
    a1_hist, a2_hist = [], []

    # Show agents compact diffs instead of full code
    a1_diff = show_diff_summary(base_code, a1_codes[-1])
    a2_diff = show_diff_summary(base_code, a2_codes[-1])

    best_a1 = min(a1_results, key=lambda r: r.get("val_bpb") or 999)
    best_a2 = min(a2_results, key=lambda r: r.get("val_bpb") or 999)

    for turn in range(n_turns):
        if turn == 0:
            a1_hist.append({"role": "user", "content": (
                f"Your changes (Agent 1, {a1_focus}):\n{a1_diff}\n"
                f"Best val_bpb: {bpb_str(best_a1)}\n\n"
                f"Agent 2's changes ({a2_focus}) — FIRST TIME seeing:\n{a2_diff}\n"
                f"Their best val_bpb: {bpb_str(best_a2)}\n\n"
                f"What should go into the final 10-minute run?"
            )})
        else:
            a1_hist.append({"role": "user", "content":
                f"Agent 2 said:\n{transcript[-1]['text']}\n\nRefine your proposal."})

        log.section(f"DISCUSSION Turn {turn+1}/{n_turns} · Agent 1")
        a1_reply = await stream_one(client, a1_sys, a1_hist[-1]["content"],
                                    max_tokens=500, silent=True)
        a1_hist.append({"role": "assistant", "content": a1_reply})
        transcript.append({"agent": 1, "turn": turn+1, "text": a1_reply})
        log(f"  {bl('A1:')} {a1_reply[:150]}...")

        if turn == 0:
            a2_hist.append({"role": "user", "content": (
                f"Your changes (Agent 2, {a2_focus}):\n{a2_diff}\n"
                f"Best val_bpb: {bpb_str(best_a2)}\n\n"
                f"Agent 1's changes ({a1_focus}) — FIRST TIME seeing:\n{a1_diff}\n"
                f"Their best val_bpb: {bpb_str(best_a1)}\n\n"
                f"Agent 1 said:\n{a1_reply}\n\nRespond."
            )})
        else:
            a2_hist.append({"role": "user", "content":
                f"Agent 1 said:\n{a1_reply}\n\nRefine your proposal."})

        log.section(f"DISCUSSION Turn {turn+1}/{n_turns} · Agent 2")
        a2_reply = await stream_one(client, a2_sys, a2_hist[-1]["content"],
                                    max_tokens=500, silent=True)
        a2_hist.append({"role": "assistant", "content": a2_reply})
        transcript.append({"agent": 2, "turn": turn+1, "text": a2_reply})
        log(f"  {yl('A2:')} {a2_reply[:150]}...")

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    md = [f"# Discussion Transcript\n_Generated {ts}_\n"]
    for e in transcript:
        md.append(f"## Agent {e['agent']} · Turn {e['turn']}\n\n{e['text']}\n\n---\n")
    return transcript, "\n".join(md)

# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Dual-agent parameter-golf optimizer")
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
        print(rd("Error: not a git repo")); sys.exit(1)
    if not Path(TARGET_FILE).exists():
        print(rd(f"Error: {TARGET_FILE} not found")); sys.exit(1)

    repo_root   = Path(git("rev-parse --show-toplevel"))
    base_branch = git("rev-parse --abbrev-ref HEAD")
    base_code   = Path(TARGET_FILE).read_text(encoding="utf-8")
    a1_instr    = load_instructions(args.a1)
    a2_instr    = load_instructions(args.a2)
    a1_focus    = "Architecture & Model Efficiency"
    a2_focus    = "Quantization & Compression"
    client      = anthropic.AsyncAnthropic()

    args_summary = f"nproc={args.nproc} gpu-time={args.gpu_time}s cycles={args.cycles} discuss={args.discuss}"

    log(f"""
{bold(CYAN)}╔══════════════════════════════════════════════════════════════╗{R}
{bold(CYAN)}║{R}  {bold(CYAN)}Dual Agent Research  ·  parameter-golf{R}                      {bold(CYAN)}║{R}
{bold(CYAN)}╚══════════════════════════════════════════════════════════════╝{R}
  {dim("target:")}    {bold(TARGET_FILE)} ({len(base_code.split(chr(10)))} lines)
  {dim("method:")}    {bold("SEARCH/REPLACE edits")} (no full-file output)
  {bl("Agent 1")}  →  {a1_focus}  →  branch: {bold("agent-1")}
  {yl("Agent 2")}  →  {a2_focus}  →  branch: {bold("agent-2")}
  {dim("pipeline:")}  {args.cycles} cycles · {args.gpu_time}s per test · {args.discuss} discussion turns
  {dim("final:")}     {FINAL_SECS}s → synthesis branch + RUN_LOG.md
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
    log(f"  {gr('✓')} agent-1 → {dim(str(wt1))}")
    log(f"  {gr('✓')} agent-2 → {dim(str(wt2))}")

    # ── Preflight ─────────────────────────────────────────────────────────────
    log.section("PREFLIGHT")
    preflight_check(args.nproc, out_dir, repo_root)

    # ── State ─────────────────────────────────────────────────────────────────
    a1_code = base_code   # Agent 1's current version of the file
    a2_code = base_code   # Agent 2's current version of the file
    a1_results, a2_results = [], []
    a1_codes, a2_codes = [base_code], [base_code]  # history for discussion

    # ── Pre-start: Agent 1 cold-start think ───────────────────────────────────
    log.section("PRE-START · Agent 1 thinks")
    a1_code = await think(
        client, coding_system(),
        coding_prompt(a1_code, a1_instr, cycle=1),
        label=bl("Agent 1"),
        current_code=a1_code, base_code=base_code,
    )
    a1_codes.append(a1_code)

    # ── Pipeline cycles ───────────────────────────────────────────────────────
    for cycle in range(1, args.cycles + 1):

        # Step A: Agent 1 GPU ∥ Agent 2 thinks
        log.section(f"CYCLE {cycle}/{args.cycles} · A1 GPU ({args.gpu_time}s) ∥ A2 thinks")
        f1.write_text(a1_code, encoding="utf-8")

        prev_a2_result = a2_results[-1] if a2_results else None
        a1_result, a2_code_new = await asyncio.gather(
            gpu_test(f1, args.nproc, args.gpu_time, wt1, f"a1_c{cycle}", out_dir),
            think(
                client, coding_system(),
                coding_prompt(a2_code, a2_instr, cycle=cycle, prev_result=prev_a2_result),
                label=yl("Agent 2"),
                current_code=a2_code, base_code=base_code,
            ),
        )
        a1_results.append(a1_result)
        a2_code = a2_code_new
        a2_codes.append(a2_code)
        log.gpu_result(f"Agent 1 Cycle {cycle}", a1_result)

        git(f'add "{TARGET_FILE}"', wt1)
        git(f'commit -m "Agent 1 Cycle {cycle} [{bpb_str(a1_result)}]"', wt1)
        log(f"  {gr('✓')} Committed to agent-1")

        # Step B: Agent 2 GPU ∥ Agent 1 thinks (skip think on last cycle)
        if cycle < args.cycles:
            log.section(f"CYCLE {cycle}/{args.cycles} · A2 GPU ({args.gpu_time}s) ∥ A1 thinks")
            f2.write_text(a2_code, encoding="utf-8")

            a2_result, a1_code_new = await asyncio.gather(
                gpu_test(f2, args.nproc, args.gpu_time, wt2, f"a2_c{cycle}", out_dir),
                think(
                    client, coding_system(),
                    coding_prompt(a1_code, a1_instr, cycle=cycle+1, prev_result=a1_result),
                    label=bl("Agent 1"),
                    current_code=a1_code, base_code=base_code,
                ),
            )
            a1_code = a1_code_new
            a1_codes.append(a1_code)
        else:
            log.section(f"CYCLE {cycle}/{args.cycles} · A2 GPU ({args.gpu_time}s) — final")
            f2.write_text(a2_code, encoding="utf-8")
            a2_result = await gpu_test(f2, args.nproc, args.gpu_time, wt2,
                                       f"a2_c{cycle}", out_dir)

        a2_results.append(a2_result)
        log.gpu_result(f"Agent 2 Cycle {cycle}", a2_result)
        git(f'add "{TARGET_FILE}"', wt2)
        git(f'commit -m "Agent 2 Cycle {cycle} [{bpb_str(a2_result)}]"', wt2)
        log(f"  {gr('✓')} Committed to agent-2")

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
        client, a1_codes, a2_codes, a1_results, a2_results,
        a1_focus, a2_focus, args.discuss, a1_instr, a2_instr, base_code,
    )
    save(out_dir / "discussion.md", disc_md)

    # ── Synthesis ─────────────────────────────────────────────────────────────
    log.section("SYNTHESIS · Merging agreed changes")
    disc_text = "\n\n".join(f"Agent {e['agent']} Turn {e['turn']}:\n{e['text']}" for e in transcript)

    synth_raw = await stream_one(
        client, coding_system(),
        synthesis_prompt(base_code, a1_code, a2_code, a1_focus, a2_focus,
                         best_a1, best_a2, disc_text),
        silent=True,
    )
    merged = safe_apply_edits(base_code, synth_raw, base_code)

    git("checkout -b synthesis")
    merged_path = repo_root / TARGET_FILE
    merged_path.write_text(merged, encoding="utf-8")

    # ── Final 10-min run ──────────────────────────────────────────────────────
    log.section(f"FINAL RUN · {FINAL_SECS}s ({FINAL_SECS//60} min) · synthesis branch")
    final_env = {
        **os.environ,
        "RUN_ID": "synthesis_final",
        "DATA_PATH": "./data/datasets/fineweb10B_sp1024/",
        "TOKENIZER_PATH": "./data/tokenizers/fineweb_1024_bpe.model",
        "VOCAB_SIZE": "1024",
        "MAX_WALLCLOCK_SECONDS": str(FINAL_SECS),
        "VAL_LOSS_EVERY": "200",
    }
    cmd = f"torchrun --standalone --nproc_per_node={args.nproc} {TARGET_FILE}"
    log(f"  $ {dim(cmd)}")
    try:
        r = subprocess.run(cmd, shell=True, cwd=str(repo_root), env=final_env,
                           capture_output=True, text=True, timeout=FINAL_SECS + 120)
        final_out = (r.stdout + r.stderr).strip()
    except subprocess.TimeoutExpired:
        final_out = "timeout"
    except Exception as e:
        final_out = str(e)

    (out_dir / "synthesis_final.log").write_text(final_out, encoding="utf-8")
    fbpb = re.search(r"val_bpb[:\s=]+([0-9]+\.[0-9]+)", final_out)
    fsize = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*MB", final_out)
    final_result = {
        "ok": fbpb is not None,
        "val_bpb": float(fbpb.group(1)) if fbpb else None,
        "model_mb": float(fsize.group(1)) if fsize else None,
    }
    log.gpu_result("FINAL (10 min)", final_result)

    # ── Commit to synthesis branch ────────────────────────────────────────────
    log_path = repo_root / "RUN_LOG.md"
    log.flush_to(log_path, args_summary)
    git(f'add "{TARGET_FILE}" "RUN_LOG.md"')
    git(f'commit -m "Synthesis [{bpb_str(final_result)}]"')
    log(f"  {gr('✓')} Committed to synthesis (train_gpt.py + RUN_LOG.md)")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    git_try(f'worktree remove --force "{wt1}"')
    git_try(f'worktree remove --force "{wt2}"')

    # ── Done ──────────────────────────────────────────────────────────────────
    log.section("DONE")
    all_bpb = [(r.get("val_bpb"), f"a1-c{i+1}") for i, r in enumerate(a1_results) if r.get("val_bpb")]
    all_bpb += [(r.get("val_bpb"), f"a2-c{i+1}") for i, r in enumerate(a2_results) if r.get("val_bpb")]
    if final_result.get("val_bpb"):
        all_bpb.append((final_result["val_bpb"], "synthesis"))
    if all_bpb:
        best, name = min(all_bpb, key=lambda x: x[0])
        log(f"\n  {gr('Best:')} {bold(f'{best:.4f}')} from {bold(name)}")

    log(f"""
  Branches:
    {bl("agent-1")}    {a1_focus}
    {yl("agent-2")}    {a2_focus}
    {gr("synthesis")}  merged + RUN_LOG.md  ← you are here

  Inspect:
    git diff agent-1 agent-2 -- {TARGET_FILE}
    git diff {base_branch} synthesis -- {TARGET_FILE}
    cat RUN_LOG.md
""")

if __name__ == "__main__":
    asyncio.run(main())