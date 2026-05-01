# TBot Project Rules

This file is the operating contract for the TBot → AI Trading System project.
Both developer and AI assistant follow these rules without exception.

---

## 1. Chunked execution
Each phase is broken into small chunks (~30–150 LOC, one logical concept).
Build one chunk → stop → test → fix if needed → move on.

## 2. No batching
Never write multiple chunks in one response. The "let me also add…" instinct is forbidden.

## 3. Verification gate per chunk
Every chunk ends with a concrete command to run and the expected output.
The user runs it and reports back before anything new is built.

## 4. Teaching mode is on
Before writing code, explain *why* — what the piece does, why this approach,
what was considered and rejected. This project is a learning opportunity.

## 5. User drives the pace
Never advance to the next chunk without an explicit "ok next" (or equivalent) from the user.

## 6. Decisions surface
At every fork (library choice, schema field, risk parameter) — stop and ask.
Even when the plan already has a recommendation, the user makes the final call.

## 7. Context hygiene
When context exceeds 50%, run `/compact` before starting the next chunk.
After compaction, re-read this file + the plan file + `docs/PROGRESS.md` to re-ground.

## 8. Progress log
Each completed chunk is appended to `docs/PROGRESS.md`:
- Chunk name
- Files touched
- Verification command run
- User's sign-off note

## 9. No silent fixes
If verification fails, don't quietly patch. Show the error, state the hypothesis, ask before changing.

## 10. The plan is the map; rules are the road code
Master plan: `~/.claude/plans/spicy-gliding-hejlsberg.md`
These rules govern *how* we travel toward it.

## 11. Push to GitHub on every phase completion
After all chunks in a phase are signed off, commit and push to GitHub before starting the next phase.
The commit must include updated `docs/PROGRESS.md` with all chunk sign-offs for that phase.

## 12. Verification scripts, not inline commands
Never use `python -c "..."` for multi-line verification code.
Always write a script file under `scripts/test_<feature>.py` and tell the user to run it.
Reason: multi-line `-c` commands cause IndentationError when the shell wraps lines.
Single truly-one-liner commands are the only exception.

---

*Progress and phase status are tracked in `docs/PROGRESS.md`.*
