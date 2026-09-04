# Agent Skills distribution channel

These instruction-only packages teach Claude Code, Cursor, and Codex to drive deep-research-agent through the same five job-boundary verbs as the MCP channel: `submit`, `poll`, `approve`, `redirect`, and `cancel`.

The repository-root `skills/` tree is the distribution source, not an automatically discovered harness directory. Copy the matching package into a supported discovery location. The installed folder must remain named `deep-research-agent` because that name matches the `SKILL.md` frontmatter.

## Claude Code

Project installation, from this repository root:

```bash
mkdir -p .claude/skills/deep-research-agent
cp skills/claude-code/deep-research-agent/SKILL.md \
  .claude/skills/deep-research-agent/SKILL.md
```

For a user-wide installation, copy the same file to `~/.claude/skills/deep-research-agent/SKILL.md` instead. Start or restart Claude Code if the new top-level skills directory was absent when the session started. Invoke the skill as `/deep-research-agent`, or ask Claude Code to submit or control a deep-research-agent run.

## Cursor

Project installation, from this repository root:

```bash
mkdir -p .cursor/skills/deep-research-agent
cp skills/cursor/deep-research-agent/SKILL.md \
  .cursor/skills/deep-research-agent/SKILL.md
```

For a user-wide installation, copy the same file to `~/.cursor/skills/deep-research-agent/SKILL.md` instead. Cursor also supports `.agents/skills/`, but `.cursor/skills/` makes this harness-specific package explicit. Invoke it from Agent chat as `/deep-research-agent`, or ask Cursor to submit or control a deep-research-agent run.

## Codex

Project installation, from this repository root:

```bash
mkdir -p .agents/skills/deep-research-agent
cp skills/codex/deep-research-agent/SKILL.md \
  .agents/skills/deep-research-agent/SKILL.md
```

For a user-wide installation, copy the same file to `~/.agents/skills/deep-research-agent/SKILL.md` instead. Codex detects skill changes automatically; restart it if the skill does not appear. Use `/skills` or mention `$deep-research-agent`, or ask Codex to submit or control a deep-research-agent run.

## Usage contract

Each package contains the complete command shapes. The synchronous path is the documented repository entrypoint:

```bash
python -m scripts.run_research "Your research question"
```

The asynchronous path uses the existing API origin (locally, `http://localhost:8000`) and preserves the returned `run_id` across poll and control operations.

**NEVER bypass the job boundary.** Harnesses and harness-authored scripts must not import or invoke the graph, agents, graph nodes, pipeline stages, or database transitions directly. If the CLI or HTTP API is unavailable, report the missing prerequisite instead of creating another execution path.

`tests/test_skills.py` parses `server.py` route decorators and every distributed `SKILL.md`. It fails if the server routes, skill endpoint tables, skill verb sets, or pinned MCP job-boundary verb set drift apart.
