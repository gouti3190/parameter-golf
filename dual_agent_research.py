#!/usr/bin/env python3
"""
dual_agent_research.py

Two Claude agents edit the same file in parallel on separate git branches,
then meet in a structured discussion before a synthesis agent merges their work.

Flow:
  Round 1    → parallel  (Agent 1: Code+Test   | Agent 2: Plan+Code)
  Round 2    → parallel  (Agent 1: Plan+Code   | Agent 2: Code+Test)  [swapped]
  Discussion → sequential multi-turn debate (agents see each other for the first time)
  Synthesis  → reads all code + discussion transcript → merges → synthesis branch
  QA         → runs tests, writes report.md

Usage:
    python dual_agent_research.py --file <path> --task "<description>" [options]

Options:
    --m1, --metric1     Agent 1 metric      (default: "Performance & Speed")
    --m2, --metric2     Agent 2 metric      (default: "Readability & Maintainability")
    --timer             Seconds per round   (default: 120)
    --discuss           Discussion turns    (default: 2)
    --out               Output directory    (default: .agents)
    --instructions      Instructions file   (default: AGENT_INSTRUCTIONS.md)

Requirements:
    pip install anthropic
    ANTHROPIC_API_KEY in environment
    git 2.5+
"""

import anthropic
import asyncio
import argparse
import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime

# ── ANSI colours ──────────────────────────────────────────────────────────────
R      = "\x1b[0m"
B      = "\x1b[1m"
DIM    = "\x1b[2m"
BLUE   = "\x1b[94m"
YELLOW = "\x1b[93m"
GREEN  = "\x1b[92m"
RED    = "\x1b[91m"
CYAN   = "\x1b[96m"
GRAY   = "\x1b[90m"

def c(col, s):  return f"{col}{s}{R}"
def bl(s):      return c(BLUE,   s)
def yl(s):      return c(YELLOW, s)
def gr(s):      return c(GREEN,  s)
def rd(s):      return c(RED,    s)
def dim(s):     return c(DIM,    s)
def bold(s):    return c(B,      s)

# ── Git helpers ───────────────────────────────────────────────────────────────
def git(cmd: str, cwd: Path = None) -> str:
    result = subprocess.run(
        f"git {cmd}", shell=True,
        cwd=str(cwd or Path.cwd()),
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout.strip()

def git_try(cmd: str, cwd: Path = None) -> str | None:
    try:
        return git(cmd, cwd)
    except Exception:
        return None

# ── File runners ──────────────────────────────────────────────────────────────
RUNNERS = {
    ".py":  "python3",
    ".js":  "node",
    ".mjs": "node",
    ".ts":  "npx tsx",
    ".rb":  "ruby",
    ".go":  "go run",
    ".sh":  "bash",
    ".php": "php",
}

def get_runner(file: Path) -> str | None:
    return RUNNERS.get(file.suffix.lower())

def run_file(file: Path, runner: str, cwd: Path = None) -> dict:
    try:
        result = subprocess.run(
            f"{runner} {file.name}", shell=True,
            cwd=str(cwd or file.parent),
            capture_output=True, text=True, timeout=30,
        )
        out = (result.stdout + result.stderr).strip()
        return {"ok": result.returncode == 0, "out": out}
    except subprocess.TimeoutExpired:
        return {"ok": False, "out": "timeout after 30s"}
    except Exception as e:
        return {"ok": False, "out": str(e)}

# ── Misc helpers ──────────────────────────────────────────────────────────────
def strip_fences(text: str) -> str:
    m = re.search(r"```(?:\w*\n)?([\s\S]*?)```", text)
    return (m.group(1) if m else text).strip()

def save(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

def section(title: str, color: str = B):
    bar = "─" * 62
    print(f"\n{color}{bar}{R}")
    print(f"{color}  {title}{R}")
    print(f"{color}{bar}{R}")

def print_run(label: str, result: dict):
    icon = gr("✓") if result["ok"] else rd("✗")
    print(f"\n  {icon} {bold(label)}")
    lines = [l for l in result["out"].split("\n") if l.strip()]
    for line in lines[:25]:
        col = GREEN if line.startswith("PASS") else RED if line.startswith("FAIL") else GRAY
        print(f"    {col}{line}{R}")
    if len(lines) > 25:
        print(dim(f"    … {len(lines) - 25} more lines"))

# ── Coding round system prompt ────────────────────────────────────────────────
def make_coding_system(agent_num: int, metric: str, role: str,
                        include_tests: bool, lang: str, instructions: str) -> str:
    test_rule = (
        "4. INCLUDE a self-contained test block at the END that runs automatically.\n"
        "   Each test prints:  PASS: <name>  or  FAIL: <name> — <reason>"
        if include_tests else
        "4. NO tests, NO asserts, NO test scaffolding. Pure implementation only."
    )
    prompt = (
        f"You are Agent {agent_num}. Optimization metric: \"{metric}\". Role: {role}.\n\n"
        f"OUTPUT RULES — obey exactly:\n"
        f"1. Output ONLY raw {lang} source code. No markdown, no backticks, no prose.\n"
        f"2. Comment every key design decision and reference \"{metric}\" explicitly.\n"
        f"3. Your output is written directly to a {lang} file and executed — must run without errors.\n"
        f"{test_rule}\n"
        f"5. You are completely isolated. Zero knowledge of any other agent.\n"
        f"6. Every decision must serve: {metric}."
    )
    if instructions.strip():
        prompt += f"\n\n--- AGENT INSTRUCTIONS ---\n{instructions}"
    return prompt

# ── Discussion system prompt ───────────────────────────────────────────────────
def make_discussion_system(agent_num: int, metric: str, other_metric: str,
                            instructions: str) -> str:
    prompt = (
        f"You are Agent {agent_num}, optimizing for \"{metric}\".\n"
        f"You have just completed two rounds of independent coding.\n"
        f"You are now in a technical discussion with Agent {3 - agent_num} (metric: \"{other_metric}\").\n"
        f"This is the FIRST TIME you see their code.\n\n"
        f"Discussion rules:\n"
        f"- Be direct and specific — reference actual functions, patterns, or lines\n"
        f"- Argue from \"{metric}\" — explain concretely why your choices serve it\n"
        f"- Acknowledge genuinely better ideas — do not be defensive\n"
        f"- Challenge weak choices with a clear technical reason\n"
        f"- End every reply with: \"My proposal: keep [X] from mine, keep [Y] from theirs\"\n"
        f"- Keep each reply under 250 words\n"
        f"- Do NOT write any code — this is a discussion, not a coding round"
    )
    if instructions.strip():
        prompt += f"\n\n--- AGENT INSTRUCTIONS (Discussion section applies) ---\n{instructions}"
    return prompt

# ── Parallel streaming ────────────────────────────────────────────────────────
async def stream_agent(client, system: str, messages: list, on_token) -> str:
    full = ""
    async with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            full += text
            on_token(full)
    return full

async def run_parallel(client, a1_cfg: dict, a2_cfg: dict, timer_secs: int):
    """
    Fire both agent API streams simultaneously alongside a countdown timer.
    asyncio.gather runs all three coroutines concurrently — neither agent waits for the other.
    """
    chars  = [0, 0]
    done   = [False, False]
    t_val  = [timer_secs]

    sys.stdout.write("\n\n")

    def draw():
        def bar(n):
            f = min(18, round(min(1.0, n / 2000) * 18))
            return GRAY + "█" * f + DIM + "░" * (18 - f) + R
        t = t_val[0]
        t_str = f"{t // 60}:{t % 60:02d}" if t > 0 else dim("done")
        sys.stdout.write(
            f"\x1b[2A"
            f"\r  {bl('Agent 1')} {bar(chars[0])} {chars[0]:>5} chars  {gr('✓') if done[0] else dim('…')}\n"
            f"\r  {yl('Agent 2')} {bar(chars[1])} {chars[1]:>5} chars  {gr('✓') if done[1] else dim('…')}"
            f"   {dim('timer:')} {t_str}   \n"
        )
        sys.stdout.flush()

    draw()

    def on_a1(full): chars[0] = len(full); draw()
    def on_a2(full): chars[1] = len(full); draw()

    async def timer():
        while t_val[0] > 0:
            await asyncio.sleep(1)
            t_val[0] -= 1
            draw()

    results = await asyncio.gather(
        stream_agent(client, a1_cfg["system"], a1_cfg["messages"], on_a1),
        stream_agent(client, a2_cfg["system"], a2_cfg["messages"], on_a2),
        timer(),
    )
    done[0] = done[1] = True
    draw()
    sys.stdout.write("\n")

    return strip_fences(results[0]), strip_fences(results[1])

# ── Single streaming call ─────────────────────────────────────────────────────
async def stream_one(client, system: str, messages: list) -> str:
    full = ""
    async with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=3000,
        system=system,
        messages=messages,
    ) as stream:
        async for text in stream.text_stream:
            sys.stdout.write(text)
            sys.stdout.flush()
            full += text
    sys.stdout.write("\n")
    return full

# ── Discussion phase ──────────────────────────────────────────────────────────
async def run_discussion(
    client,
    a1r1: str, a1r2: str,
    a2r1: str, a2r2: str,
    run1a1: dict, run2a2: dict,
    m1: str, m2: str,
    lang: str,
    turns: int,
    instructions: str,
) -> tuple[list[dict], str]:
    """
    Multi-turn sequential discussion between Agent 1 and Agent 2.

    Each turn: Agent 1 speaks → Agent 2 responds.
    Agents see each other's code for the first time.
    Returns (transcript list, markdown string).
    """
    transcript = []

    a1_sys = make_discussion_system(1, m1, m2, instructions)
    a2_sys = make_discussion_system(2, m2, m1, instructions)

    # Conversation history maintained separately per agent
    # (each agent only sees its own conversation thread, not the other's internal monologue)
    a1_history: list[dict] = []
    a2_history: list[dict] = []

    for turn in range(turns):
        # ── Agent 1 speaks ────────────────────────────────────────────────────
        if turn == 0:
            # First time seeing Agent 2's code
            a1_opening_ctx = (
                f"Your code across both rounds:\n\n"
                f"=== Your Round 1 (Code+Test) ===\n{a1r1}\n\n"
                f"=== Your Round 2 (Plan+Code) ===\n{a1r2}\n\n"
                + (f"Your test results: {run1a1['out']}\n\n" if run1a1["out"] not in ("", "not executed") else "")
                + f"Agent 2's code (you are seeing this for the first time):\n\n"
                f"=== Agent 2 Round 1 (Plan+Code) ===\n{a2r1}\n\n"
                f"=== Agent 2 Round 2 (Code+Test) ===\n{a2r2}\n\n"
                + (f"Agent 2 test results: {run2a2['out']}\n\n" if run2a2["out"] not in ("", "not executed") else "")
                + "Open the technical discussion. What do you notice about their approach vs yours?"
            )
            a1_history.append({"role": "user", "content": a1_opening_ctx})
        else:
            last_a2 = transcript[-1]["text"]
            a1_history.append({
                "role": "user",
                "content": f"Agent 2 said:\n\n{last_a2}\n\nRespond and refine your proposal.",
            })

        section(f"DISCUSSION  Turn {turn + 1}  ·  Agent 1 speaking", BLUE + B)
        a1_reply = await stream_one(client, a1_sys, a1_history)
        a1_history.append({"role": "assistant", "content": a1_reply})
        transcript.append({"agent": 1, "turn": turn + 1, "metric": m1, "text": a1_reply})

        # ── Agent 2 responds ──────────────────────────────────────────────────
        if turn == 0:
            a2_opening_ctx = (
                f"Your code across both rounds:\n\n"
                f"=== Your Round 1 (Plan+Code) ===\n{a2r1}\n\n"
                f"=== Your Round 2 (Code+Test) ===\n{a2r2}\n\n"
                + (f"Your test results: {run2a2['out']}\n\n" if run2a2["out"] not in ("", "not executed") else "")
                + f"Agent 1's code (you are seeing this for the first time):\n\n"
                f"=== Agent 1 Round 1 (Code+Test) ===\n{a1r1}\n\n"
                f"=== Agent 1 Round 2 (Plan+Code) ===\n{a1r2}\n\n"
                + (f"Agent 1 test results: {run1a1['out']}\n\n" if run1a1["out"] not in ("", "not executed") else "")
                + f"Agent 1 just opened the discussion:\n\n{a1_reply}\n\n"
                "Respond to their points. What do you see in their approach?"
            )
            a2_history.append({"role": "user", "content": a2_opening_ctx})
        else:
            a2_history.append({
                "role": "user",
                "content": f"Agent 1 said:\n\n{a1_reply}\n\nRespond and refine your proposal.",
            })

        section(f"DISCUSSION  Turn {turn + 1}  ·  Agent 2 speaking", YELLOW + B)
        a2_reply = await stream_one(client, a2_sys, a2_history)
        a2_history.append({"role": "assistant", "content": a2_reply})
        transcript.append({"agent": 2, "turn": turn + 1, "metric": m2, "text": a2_reply})

    # ── Build markdown transcript ─────────────────────────────────────────────
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# Discussion Transcript\n\n_Generated {ts}_\n"]
    for entry in transcript:
        label = f"Agent {entry['agent']} · Turn {entry['turn']} · metric: {entry['metric']}"
        lines.append(f"## {label}\n\n{entry['text']}\n\n---\n")

    return transcript, "\n".join(lines)

# ── Main ──────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(
        prog="dual_agent_research",
        description="Two Claude agents edit a file in parallel on separate git branches, then discuss.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python dual_agent_research.py --file src/sort.py --task "implement quicksort"

  python dual_agent_research.py \\
    --file api/cache.py \\
    --task "build an LRU cache" \\
    --m1 "Performance & Speed" \\
    --m2 "Readability & Maintainability" \\
    --timer 90 --discuss 3
        """,
    )
    parser.add_argument("--file",  "-f", required=True,  help="File for agents to edit")
    parser.add_argument("--task",  "-t", required=True,  help="What to build or improve")
    parser.add_argument("--m1",  "--metric1", default="Performance & Speed",
                        help="Agent 1 metric")
    parser.add_argument("--m2",  "--metric2", default="Readability & Maintainability",
                        help="Agent 2 metric")
    parser.add_argument("--timer",   type=int, default=120, help="Seconds per round (default: 120)")
    parser.add_argument("--discuss", type=int, default=2,   help="Discussion turns (default: 2)")
    parser.add_argument("--out",     default=".agents",     help="Output directory (default: .agents)")
    parser.add_argument("--instructions", default="AGENT_INSTRUCTIONS.md",
                        help="Instructions file (default: AGENT_INSTRUCTIONS.md)")
    args = parser.parse_args()

    file_path   = Path(args.file)
    out_dir     = Path(args.out)
    instr_path  = Path(args.instructions)

    # ── Validate ──────────────────────────────────────────────────────────────
    if git_try("rev-parse --show-toplevel") is None:
        print(rd("Error: not a git repository. Run `git init` first."))
        sys.exit(1)

    if not file_path.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text("", encoding="utf-8")
        print(dim(f"  Created empty file: {file_path}"))

    repo_root    = Path(git("rev-parse --show-toplevel"))
    base_branch  = git("rev-parse --abbrev-ref HEAD")
    rel_file     = file_path.resolve().relative_to(repo_root)
    ext          = file_path.suffix
    lang         = ext.lstrip(".") or "text"
    runner       = get_runner(file_path)
    base         = file_path.read_text(encoding="utf-8")
    instructions = instr_path.read_text(encoding="utf-8") if instr_path.exists() else ""

    client = anthropic.AsyncAnthropic()

    # ── Banner ────────────────────────────────────────────────────────────────
    instr_note = str(instr_path) if instr_path.exists() else dim(f"{instr_path} (not found)")
    print(f"""
{bold(CYAN)}╔══════════════════════════════════════════════════════════╗{R}
{bold(CYAN)}║{R}  {bold(CYAN)}Dual Agent Research System{R}  {dim("claude-sonnet-4-20250514")}      {bold(CYAN)}║{R}
{bold(CYAN)}╚══════════════════════════════════════════════════════════╝{R}
  {dim("file:")}          {bold(str(file_path))}
  {dim("task:")}          {args.task}
  {bl("Agent 1")}       →  {args.m1}   →  branch: {bold("agent-1")}
  {yl("Agent 2")}       →  {args.m2}   →  branch: {bold("agent-2")}
  {dim("timer:")}         {args.timer}s per round · 2 rounds
  {dim("discussion:")}    {args.discuss} turns × 2 agents = {args.discuss * 2} messages
  {dim("base branch:")}   {base_branch}
  {dim("instructions:")}  {instr_note}
""")
    if not runner:
        print(f"  {c(YELLOW, '⚠')}  Unknown file type — code saved but not executed\n")

    # ── Git: branches and worktrees ───────────────────────────────────────────
    section("GIT SETUP  ·  Creating branches and worktrees")

    wt1 = (out_dir / "worktree-1").resolve()
    wt2 = (out_dir / "worktree-2").resolve()

    git_try(f'worktree remove --force "{wt1}"')
    git_try(f'worktree remove --force "{wt2}"')
    for branch in ["agent-1", "agent-2", "synthesis"]:
        git_try(f"branch -D {branch}")

    git("checkout -b agent-1")
    git(f"checkout {base_branch}")
    git("checkout -b agent-2")
    git(f"checkout {base_branch}")
    git(f'worktree add "{wt1}" agent-1')
    git(f'worktree add "{wt2}" agent-2')

    f1 = wt1 / rel_file
    f2 = wt2 / rel_file

    print(f"  {gr('✓')}  {bold('agent-1')} branch  →  {dim(str(wt1))}")
    print(f"  {gr('✓')}  {bold('agent-2')} branch  →  {dim(str(wt2))}")

    def task_msg(own_prev: str = None) -> str:
        msg = f"Task: {args.task}\n\nCurrent file ({rel_file}):\n{base or '(empty)'}"
        if own_prev:
            msg += f"\n\nYour previous round output — build on this:\n{own_prev}"
        return msg

    # ── ROUND 1 ──────────────────────────────────────────────────────────────
    section("ROUND 1  ·  Agent 1: Code+Test  |  Agent 2: Plan+Code", BLUE + B)
    print(f"  {dim('No cross-agent context. Each agent sees only the original file.')}\n")

    a1r1, a2r1 = await run_parallel(
        client,
        {
            "system":   make_coding_system(1, args.m1, "CODE then TEST", True, lang, instructions),
            "messages": [{"role": "user", "content": task_msg()}],
        },
        {
            "system":   make_coding_system(2, args.m2, "PLAN then CODE — no testing", False, lang, instructions),
            "messages": [{"role": "user", "content": task_msg()}],
        },
        args.timer,
    )

    save(f1, a1r1)
    run1a1 = {"ok": False, "out": "not executed"}
    if runner:
        run1a1 = run_file(f1, runner, wt1)
        print_run("Agent 1 · Round 1 tests", run1a1)
        save(out_dir / "agent1_r1.log", run1a1["out"])
    git(f'add "{rel_file}"', wt1)
    git(f'commit -m "Agent 1 R1: Code+Test [{args.m1}] — {"PASS" if run1a1["ok"] else "FAIL"}"', wt1)
    print(f"\n  {gr('✓')}  Committed to {bold('agent-1')}")

    save(f2, a2r1)
    git(f'add "{rel_file}"', wt2)
    git(f'commit -m "Agent 2 R1: Plan+Code [{args.m2}]"', wt2)
    print(f"  {gr('✓')}  Committed to {bold('agent-2')}")

    # ── ROUND 2 ──────────────────────────────────────────────────────────────
    section("ROUND 2  ·  Agent 1: Plan+Code  |  Agent 2: Code+Test  (swapped)", YELLOW + B)
    print(f"  {dim('Each agent builds on its own Round 1. Still isolated from each other.')}\n")

    a1r2, a2r2 = await run_parallel(
        client,
        {
            "system":   make_coding_system(1, args.m1, "PLAN then CODE — no testing", False, lang, instructions),
            "messages": [{"role": "user", "content": task_msg(a1r1)}],
        },
        {
            "system":   make_coding_system(2, args.m2, "CODE then TEST", True, lang, instructions),
            "messages": [{"role": "user", "content": task_msg(a2r1)}],
        },
        args.timer,
    )

    save(f1, a1r2)
    git(f'add "{rel_file}"', wt1)
    git(f'commit -m "Agent 1 R2: Plan+Code [{args.m1}]"', wt1)
    print(f"\n  {gr('✓')}  Committed to {bold('agent-1')}")

    save(f2, a2r2)
    run2a2 = {"ok": False, "out": "not executed"}
    if runner:
        run2a2 = run_file(f2, runner, wt2)
        print_run("Agent 2 · Round 2 tests", run2a2)
        save(out_dir / "agent2_r2.log", run2a2["out"])
    git(f'add "{rel_file}"', wt2)
    git(f'commit -m "Agent 2 R2: Code+Test [{args.m2}] — {"PASS" if run2a2["ok"] else "FAIL"}"', wt2)
    print(f"  {gr('✓')}  Committed to {bold('agent-2')}")

    # ── REVEAL ────────────────────────────────────────────────────────────────
    section("REVEAL  ·  Independent work complete — entering discussion")
    print(f"  {bl('agent-1')}  2 commits  {gr('tests ✓') if run1a1['ok'] else rd('tests ✗')}  ({args.m1})")
    print(f"  {yl('agent-2')}  2 commits  {gr('tests ✓') if run2a2['ok'] else rd('tests ✗')}  ({args.m2})")
    print(f"\n  Agents will now see each other's code for the first time.")
    print(f"  {dim(str(args.discuss) + ' turns × 2 agents = ' + str(args.discuss * 2) + ' messages total')}")

    # ── DISCUSSION ────────────────────────────────────────────────────────────
    transcript, discussion_md = await run_discussion(
        client,
        a1r1, a1r2, a2r1, a2r2,
        run1a1, run2a2,
        args.m1, args.m2,
        lang, args.discuss, instructions,
    )
    save(out_dir / "discussion.md", discussion_md)
    print(f"\n  {gr('✓')}  Discussion saved to {bold(str(out_dir / 'discussion.md'))}")

    # ── SYNTHESIS ─────────────────────────────────────────────────────────────
    section("SYNTHESIS  ·  Implementing what the agents agreed on", GREEN + B)

    # Build a readable discussion summary for the synthesis prompt
    disc_summary = "\n".join(
        f"--- Agent {e['agent']} (Turn {e['turn']}, metric: {e['metric']}) ---\n{e['text']}"
        for e in transcript
    )

    synth_system = (
        f"You are a Synthesis expert implementing the agreed-upon merge.\n"
        f"Two agents worked independently then held a {args.discuss}-turn discussion.\n"
        f"Agent 1 metric: \"{args.m1}\".  Agent 2 metric: \"{args.m2}\".\n\n"
        f"OUTPUT RULES:\n"
        f"1. Output ONLY raw {lang} code. No markdown fences, no prose.\n"
        f"2. Attribute every key decision:\n"
        f"   # [Agent 1] <reason — serves {args.m1}>\n"
        f"   # [Agent 2] <reason — serves {args.m2}>\n"
        f"   # [Merged]  <compromise rationale>\n"
        f"   # [Discussion] <idea that emerged from the debate>\n"
        f"3. Follow the proposals the agents made in the discussion — they agreed on this.\n"
        f"4. Include a test block at the end (PASS/FAIL per test, runs automatically).\n"
        f"5. File must execute without errors."
    )

    r1_log = f"\nTest output:\n{run1a1['out']}\n" if run1a1["out"] not in ("", "not executed") else ""
    r2_log = f"\nTest output:\n{run2a2['out']}\n" if run2a2["out"] not in ("", "not executed") else ""

    synth_content = (
        f"Task: {args.task}\n\n"
        f"=== Agent 1 · R1 · Code+Test · metric: {args.m1} ===\n{a1r1}{r1_log}\n"
        f"=== Agent 2 · R1 · Plan+Code · metric: {args.m2} ===\n{a2r1}\n\n"
        f"=== Agent 1 · R2 · Plan+Code · metric: {args.m1} ===\n{a1r2}\n\n"
        f"=== Agent 2 · R2 · Code+Test · metric: {args.m2} ===\n{a2r2}{r2_log}\n\n"
        f"=== Discussion Transcript ({args.discuss * 2} messages) ===\n{disc_summary}\n\n"
        f"Implement the merged solution the agents discussed and agreed on."
    )

    print()
    merged_raw = await stream_one(client, synth_system, [{"role": "user", "content": synth_content}])
    merged = strip_fences(merged_raw)

    git("checkout -b synthesis")
    merged_path = repo_root / rel_file
    save(merged_path, merged)

    merged_run = {"ok": False, "out": "not executed"}
    if runner:
        merged_run = run_file(merged_path, runner, repo_root)
        print_run("Synthesis tests", merged_run)
        save(out_dir / "merged.log", merged_run["out"])

    git(f'add "{rel_file}"')
    git(f'commit -m "Synthesis: agent-1 + agent-2 (discussed {args.discuss * 2} msgs) — {"PASS" if merged_run["ok"] else "FAIL"}"')
    print(f"  {gr('✓')}  Committed to {bold('synthesis')}")

    # ── QA REPORT ─────────────────────────────────────────────────────────────
    section("QA REPORT")

    def pf(r): return "PASS" if r["ok"] else "FAIL"

    qa_system = (
        f"QA expert. Write a concise technical report in Markdown.\n\n"
        f"# QA Report\n\n"
        f"## Test Results\n<pass/fail summary>\n\n"
        f"## {args.m1}\nScore: X/10\n<2–3 sentence analysis>\n\n"
        f"## {args.m2}\nScore: X/10\n<2–3 sentence analysis>\n\n"
        f"## What Agent 1 Brought\n<specific contributions>\n\n"
        f"## What Agent 2 Brought\n<specific contributions>\n\n"
        f"## Key Discussion Outcomes\n<what changed because of the debate — ideas neither agent had alone>\n\n"
        f"## Trade-offs\n<what was balanced or sacrificed>\n\n"
        f"## Branch Summary\n"
        f"| Branch    | Commits | Tests       |\n"
        f"|-----------|---------|-------------|\n"
        f"| agent-1   | 2       | {pf(run1a1)} |\n"
        f"| agent-2   | 2       | {pf(run2a2)} |\n"
        f"| synthesis | 1       | {pf(merged_run)} |\n\n"
        f"## Verdict\nPASS or FAIL — one sentence."
    )

    print()
    report = await stream_one(client, qa_system, [{
        "role": "user",
        "content": (
            f"Task: {args.task}\n\n"
            f"Discussion transcript:\n{disc_summary}\n\n"
            f"Merged code:\n{merged}\n\n"
            f"Test output:\n{merged_run['out'] or '(not run)'}"
        ),
    }])
    save(out_dir / "report.md", report)
    print(f"\n  {gr('✓')}  Report saved to {bold(str(out_dir / 'report.md'))}")

    # ── Cleanup worktrees ─────────────────────────────────────────────────────
    git_try(f'worktree remove --force "{wt1}"')
    git_try(f'worktree remove --force "{wt2}"')

    # ── Done ──────────────────────────────────────────────────────────────────
    section("DONE", GREEN + B)
    print(f"""
  {dim("Branches:")}
    {bl("agent-1")}    Agent 1 · {args.m1}
    {yl("agent-2")}    Agent 2 · {args.m2}
    {gr("synthesis")}  Merged result   ← currently checked out

  {dim("Inspect:")}
    git log --oneline agent-1
    git log --oneline agent-2
    git diff agent-1 agent-2 -- {rel_file}
    git diff {base_branch} synthesis -- {rel_file}

  {dim("Output files:")}
    {str(out_dir)}/discussion.md   ← full agent debate transcript
    {str(out_dir)}/report.md       ← QA report with scores
    {str(out_dir)}/agent1_r1.log   ← Agent 1 test run
    {str(out_dir)}/agent2_r2.log   ← Agent 2 test run
    {str(out_dir)}/merged.log      ← synthesis test run
""")

if __name__ == "__main__":
    asyncio.run(main())