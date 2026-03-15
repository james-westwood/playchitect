#!/usr/bin/env bash
# ralph-loop.sh — Multi-agent AFK loop.
#
# !! DO NOT MODIFY THIS FILE — it is maintained by the human owner (James). !!
#
# Each task:
#   1. Assign Claude, Gemini, or opencode as CODER (and REVIEWER unless --skip-review)
#   2. Create a feature branch
#   3. CODER implements in atomic commits (src → tests → tracking)
#   4. Push branch, open PR on GitHub
#   5. REVIEWER reads the diff via `gh pr diff`, posts a review comment
#   6. Auto-merge → delete branch → pull main
#
# Stops when: all ralph-owned tasks done | next task is human-owned | iteration cap
#
# Usage:
#   ./ralph-loop.sh                              # up to 10 iterations, random coder/reviewer
#   ./ralph-loop.sh --max 50                    # up to 50 iterations
#   ./ralph-loop.sh --skip-review               # skip AI review, auto-merge immediately
#   ./ralph-loop.sh --claude-only               # Claude codes and reviews (no Gemini/opencode)
#   ./ralph-loop.sh --gemini-only               # Gemini codes and reviews (no Claude/opencode)
#   ./ralph-loop.sh --opencode-only             # opencode codes and reviews (no Claude/Gemini)
#   ./ralph-loop.sh --opencode-model google/gemini-2.0-flash  # override opencode model
#   ./ralph-loop.sh --claude-only --skip-review # Claude only, no review step
#   ./ralph-loop.sh --resume                    # resume stale branch if one exists for current task
#
# Requirements:
#   - claude CLI (with --dangerously-skip-permissions support)
#   - gemini CLI  (Google Gemini CLI, uses --yolo to auto-approve tool use)
#   - opencode CLI  (optional; used when --opencode-only or randomly assigned)
#   - gh CLI authenticated (gh auth login)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAX_ITERATIONS=10
SKIP_REVIEW=false
MODEL_MODE="random"  # random | claude | gemini | opencode
RESUME=false
OPENCODE_MODEL="opencode/claude-sonnet-4-6"  # override with --opencode-model

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max)            MAX_ITERATIONS="$2"; shift 2 ;;
    --skip-review)    SKIP_REVIEW=true; shift ;;
    --claude-only)    MODEL_MODE="claude"; shift ;;
    --gemini-only)    MODEL_MODE="gemini"; shift ;;
    --opencode-only)  MODEL_MODE="opencode"; shift ;;
    --opencode-model) OPENCODE_MODEL="$2"; shift 2 ;;
    --resume)         RESUME=true; shift ;;
    *) echo "Unknown flag: $1" >&2; exit 1 ;;
  esac
done

LOG_FILE="$SCRIPT_DIR/ralph-loop.log"
MAIN_BRANCH="main"

# ── Helpers ──────────────────────────────────────────────────────────────────

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

die() {
  log "FATAL: $*"
  exit 1
}

# Gemini model to use — gemini-2.5-pro has a generous free tier and stable capacity
GEMINI_MODEL="gemini-2.5-pro"

# Coding agent — needs full file-system tool access
# Each agent falls back to Claude on failure (rate limits, no capacity, etc.)
run_coder() {
  local agent="$1" prompt="$2"
  if [[ "$agent" == "claude" ]]; then
    if env -u CLAUDECODE claude --dangerously-skip-permissions --print "$prompt"; then
      return 0
    else
      log "  Claude coder failed — falling back to Gemini"
      gemini -m "$GEMINI_MODEL" --yolo -p "$prompt"
    fi
  elif [[ "$agent" == "gemini" ]]; then
    if gemini -m "$GEMINI_MODEL" --yolo -p "$prompt"; then
      return 0
    else
      log "  Gemini coder failed — falling back to Claude"
      env -u CLAUDECODE claude --dangerously-skip-permissions --print "$prompt"
    fi
  else
    # opencode — tools are auto-approved in run mode, no extra flags needed
    if opencode run --model "$OPENCODE_MODEL" "$prompt"; then
      return 0
    else
      log "  opencode coder failed — falling back to Claude"
      env -u CLAUDECODE claude --dangerously-skip-permissions --print "$prompt"
    fi
  fi
}

# Reviewing agent — reads a diff and returns text; no file-system writes needed
# Each agent falls back to Claude on failure
run_reviewer() {
  local agent="$1" prompt="$2"
  if [[ "$agent" == "claude" ]]; then
    if env -u CLAUDECODE claude --print "$prompt"; then
      return 0
    else
      log "  Claude reviewer failed — falling back to Gemini"
      gemini -m "$GEMINI_MODEL" -p "$prompt"
    fi
  elif [[ "$agent" == "gemini" ]]; then
    if gemini -m "$GEMINI_MODEL" -p "$prompt"; then
      return 0
    else
      log "  Gemini reviewer failed — falling back to Claude"
      env -u CLAUDECODE claude --print "$prompt"
    fi
  else
    # opencode — auto-approved tools; model only needs to analyse the diff text
    if opencode run --model "$OPENCODE_MODEL" "$prompt"; then
      return 0
    else
      log "  opencode reviewer failed — falling back to Claude"
      env -u CLAUDECODE claude --print "$prompt"
    fi
  fi
}

# Predicate: task is actionable (incomplete, ralph-owned, not blocked)
# Stop if no actionable tasks remain (exit 0 = tasks remain, exit 1 = all done)
check_complete() {
  python3 -c "
import json, sys
with open('prd.json') as f: prd = json.load(f)
actionable = [t for t in prd['tasks'] if not t.get('completed') and t.get('owner') != 'human' and not t.get('blocked')]
sys.exit(0 if actionable else 1)
" 2>/dev/null
}

# Prints task label and exits 0 if the next actionable task is human-owned; exits 1 otherwise
check_next_is_human() {
  python3 -c "
import json, sys
with open('prd.json') as f: prd = json.load(f)
incomplete = [t for t in prd['tasks'] if not t.get('completed') and not t.get('blocked')]
if incomplete and incomplete[0].get('owner') == 'human':
    t = incomplete[0]
    print(f'[{t[\"id\"]}] {t[\"title\"]} ({t[\"epic\"]})')
    sys.exit(0)
sys.exit(1)
" 2>/dev/null
}

count_remaining() {
  python3 -c "
import json
with open('prd.json') as f: prd = json.load(f)
print(len([t for t in prd['tasks'] if not t.get('completed') and t.get('owner') != 'human' and not t.get('blocked')]))
" 2>/dev/null || echo "?"
}

# Read a single field from the next actionable ralph-owned task
next_task_field() {
  local field="$1"
  python3 -c "
import json
with open('prd.json') as f: prd = json.load(f)
t = [x for x in prd['tasks'] if not x.get('completed') and x.get('owner') != 'human' and not x.get('blocked')][0]
print(t['$field'])
" 2>/dev/null
}

# ── Preflight ─────────────────────────────────────────────────────────────────

command -v gh     >/dev/null 2>&1 || die "'gh' not found — install GitHub CLI: https://cli.github.com"
command -v gemini >/dev/null 2>&1 || die "'gemini' not found — install Google Gemini CLI"
[[ "$MODEL_MODE" == "opencode" ]] && { command -v opencode >/dev/null 2>&1 || die "'opencode' not found — install opencode CLI"; }
gh auth status    >/dev/null 2>&1 || die "Not authenticated with gh — run: gh auth login"

REVIEW_MODE_LABEL=$( [[ "$SKIP_REVIEW" == "true" ]] && echo "auto-merge (no review)" || echo "AI review" )
MODEL_LABEL=$( case "$MODEL_MODE" in
  claude)   echo "Claude only" ;;
  gemini)   echo "Gemini only" ;;
  opencode) echo "opencode ($OPENCODE_MODEL)" ;;
  *)        echo "Claude ↔ Gemini ↔ opencode (random)" ;;
esac )
RESUME_LABEL=$( [[ "$RESUME" == "true" ]] && echo "yes (resume stale branches)" || echo "no (fresh branches only)" )
echo "================================================================"
echo "  Ralph Loop — playchitect Multi-Agent AFK Mode"
echo "  Agents:   $MODEL_LABEL"
echo "  Workflow: branch → atomic commits → PR → $REVIEW_MODE_LABEL → merge"
echo "  Resume:   $RESUME_LABEL"
echo "  Max iterations: $MAX_ITERATIONS"
echo "  Log: $LOG_FILE"
echo "================================================================"
echo ""

log "Starting Ralph loop. Max iterations: $MAX_ITERATIONS"

ITERATION=0

while true; do

  # ── Stop conditions ────────────────────────────────────────────────────────

  if HUMAN_TASK=$(check_next_is_human 2>/dev/null); then
    log "Reached human-owned task: $HUMAN_TASK. Handing over."
    echo ""
    echo "================================================================"
    echo "  YOUR TURN"
    echo "  Next task is yours to implement: $HUMAN_TASK"
    echo "  Mark it complete in prd.json, then re-run ralph."
    echo "================================================================"
    break
  fi

  if ! check_complete; then
    log "All ralph-owned tasks complete."
    echo ""
    echo "================================================================"
    echo "  ALL RALPH TASKS COMPLETE"
    echo "================================================================"
    break
  fi

  if [[ $ITERATION -ge $MAX_ITERATIONS ]]; then
    log "Iteration cap ($MAX_ITERATIONS). $(count_remaining) tasks remaining."
    echo ""
    echo "================================================================"
    echo "  ITERATION CAP ($MAX_ITERATIONS) — run with --max N to continue"
    echo "  Tasks remaining: $(count_remaining)"
    echo "================================================================"
    break
  fi

  ITERATION=$((ITERATION + 1))

  # ── Task details ───────────────────────────────────────────────────────────

  TASK_ID=$(next_task_field id)
  TASK_TITLE=$(next_task_field title)
  TASK_DESC=$(next_task_field description)
  TASK_AC=$(next_task_field acceptance_criteria)
  TASK_EPIC=$(next_task_field epic)
  BRANCH="ralph/task-${TASK_ID}-${TASK_TITLE}"
  TODAY=$(date +%Y-%m-%d)

  # Assign coder/reviewer based on model mode
  case "$MODEL_MODE" in
    claude)   CODER="claude";    REVIEWER="claude" ;;
    gemini)   CODER="gemini";    REVIEWER="gemini" ;;
    opencode) CODER="opencode";  REVIEWER="opencode" ;;
    *)
      # 3-way random: Claude codes+Gemini reviews, Gemini codes+Claude reviews,
      # or opencode codes+Claude reviews (opencode review is a bonus cross-check)
      case $(( RANDOM % 3 )) in
        0) CODER="claude";   REVIEWER="gemini" ;;
        1) CODER="gemini";   REVIEWER="claude" ;;
        2) CODER="opencode"; REVIEWER="claude" ;;
      esac
      ;;
  esac

  echo ""
  echo "--- Iteration $ITERATION / $MAX_ITERATIONS  |  $(count_remaining) remaining ---"
  echo "  Task:     [$TASK_ID] $TASK_TITLE"
  echo "  Epic:     $TASK_EPIC"
  echo "  Branch:   $BRANCH"
  echo "  Coder:    $CODER  |  Reviewer: $REVIEWER"
  log "Iteration $ITERATION: [$TASK_ID] $TASK_TITLE | coder=$CODER reviewer=$REVIEWER branch=$BRANCH"

  # ── Branch setup ───────────────────────────────────────────────────────────

  BRANCH_EXISTS=false
  PR_EXISTS=""

  if git show-ref --verify --quiet "refs/heads/$BRANCH" || git show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
    BRANCH_EXISTS=true
    PR_EXISTS=$(gh pr list --head "$BRANCH" --json number -q '.[0].number' 2>/dev/null || true)
  fi

  if [[ "$BRANCH_EXISTS" == "true" && "$RESUME" == "true" ]]; then
    log "  Resuming stale branch: $BRANCH"
    git checkout "$BRANCH"
    git pull --ff-only origin "$BRANCH" 2>/dev/null || true

    if [[ -n "$PR_EXISTS" ]]; then
      # PR already open — skip coding, go straight to merge
      log "  PR #$PR_EXISTS already open — skipping coding step, merging directly."
      PR_NUMBER="$PR_EXISTS"
      PR_URL="$(gh pr view "$PR_NUMBER" --json url -q '.url')"
      log "  PR: $PR_URL"

      if [[ "$SKIP_REVIEW" != "true" ]]; then
        log "  Fetching diff for review ($REVIEWER)..."
        PR_DIFF=""
        for _retry in 1 2 3 4 5; do
          PR_DIFF=$(gh pr diff "$PR_NUMBER" 2>/dev/null) && break
          log "  gh pr diff returned nothing (attempt $_retry/5) — waiting 10s..."
          sleep 10
        done
        REVIEW_PROMPT="You are the code reviewer for a pull request in the playchitect project.

The code was written by $CODER. You are $REVIEWER.

PR: [$TASK_ID] $TASK_TITLE
Acceptance criteria: $TASK_AC

Diff:
---
$PR_DIFF
---

Write a concise code review covering:
1. Correctness — does the implementation satisfy the acceptance criteria?
2. Code quality — readability, naming, structure
3. Test quality — are tests meaningful and sufficient?
4. Any bugs, edge cases, or concerns

Be constructive and specific. End your review with exactly one of:
- **APPROVED** — code is good to merge as-is
- **CHANGES REQUESTED: {brief reason}** — if there are real issues

Output only the review text. It will be posted as a GitHub PR comment."

        log "  Running reviewer ($REVIEWER)..."
        REVIEW_TEXT=$(run_reviewer "$REVIEWER" "$REVIEW_PROMPT" 2>&1 | tee -a "$LOG_FILE")
        log "  Posting review comment..."
        gh pr comment "$PR_NUMBER" --body "$(cat <<EOF
## Code Review by \`$REVIEWER\`

$REVIEW_TEXT

---
*Implemented by \`$CODER\` · Reviewed by \`$REVIEWER\`*
EOF
)"
      else
        gh pr comment "$PR_NUMBER" --body "*Review skipped — merged automatically via \`--skip-review\`.*"
      fi

      log "  Waiting for CI to pass on PR #$PR_NUMBER..."
      for _wait in $(seq 1 60); do
        sleep 30
        CI_STATUS=$(gh pr checks "$PR_NUMBER" --json state -q '.[].state' 2>/dev/null | sort -u | tr '\n' ' ')
        if echo "$CI_STATUS" | grep -q "FAILURE\|ERROR"; then
          log "  CI FAILED on PR #$PR_NUMBER — stopping."; exit 1
        fi
        if echo "$CI_STATUS" | grep -qv "PENDING\|IN_PROGRESS\|QUEUED\|WAITING\|EXPECTED" && [[ -n "$CI_STATUS" ]]; then
          log "  CI passed (check ${_wait}/60, ci=$CI_STATUS)"; break
        fi
      done
      log "  Merging PR #$PR_NUMBER..."
      gh pr merge "$PR_NUMBER" --merge --delete-branch
      git checkout "$MAIN_BRANCH"
      git pull --ff-only origin "$MAIN_BRANCH"
      log "Iteration $ITERATION complete (resumed): [$TASK_ID] $TASK_TITLE | $PR_URL"
      sleep 2
      continue
    else
      log "  Branch exists but no PR — resuming coding on existing branch."
    fi
  else
    # Fresh start
    git checkout "$MAIN_BRANCH"
    git pull --ff-only origin "$MAIN_BRANCH" 2>/dev/null || true
    git checkout -b "$BRANCH"
  fi

  # ── Coding step ────────────────────────────────────────────────────────────

  log "  Running coder ($CODER)..."

  RESUME_NOTE=""
  if [[ "$BRANCH_EXISTS" == "true" && "$RESUME" == "true" ]]; then
    RESUME_NOTE="
IMPORTANT: This branch already has partial work from a previous run that was interrupted.
Run \`git log --oneline\` to see what commits exist. Do NOT redo work that is already committed.
Pick up from where the previous run left off."
  fi

  CODER_PROMPT="You are the CODER implementing task [$TASK_ID] $TASK_TITLE for the playchitect project. Another AI will review your work — write clean, production-quality code.
$RESUME_NOTE

Read CLAUDE.md for project conventions.

Epic: $TASK_EPIC
Description: $TASK_DESC
Acceptance criteria: $TASK_AC

Implementation steps:
1. Write source files under playchitect/ (core logic) or tests/ (tests)
2. Run tests and fix any failures: uv run pytest tests/ -v
3. Run pre-commit and fix all failures: uv run pre-commit run --all-files
4. Make ATOMIC commits — do not squash everything into one commit:
   - Commit A (source):   git add playchitect/ && git commit -m '[$TASK_ID] $TASK_TITLE: implement'
   - Commit B (tests):    git add tests/ && git commit -m '[$TASK_ID] $TASK_TITLE: add tests'
   - Commit C (tracking): set \"completed\": true in prd.json for task $TASK_ID,
                          append to progress.txt: [$TODAY] [$TASK_ID] $TASK_TITLE: {one-line summary}
                          git add prd.json progress.txt && git commit -m '[$TASK_ID] $TASK_TITLE: mark complete'

Do NOT push. Do NOT create a PR. The orchestrator handles that.

Rules:
- Never implement any task with \"owner\": \"human\"
- Use uv for all Python commands (uv run pytest, uv sync, etc.)
- Source under playchitect/, tests under tests/
- Follow all conventions in CLAUDE.md"

  run_coder "$CODER" "$CODER_PROMPT" 2>&1 | tee -a "$LOG_FILE"

  # ── Push and open PR ───────────────────────────────────────────────────────

  log "  Pushing $BRANCH..."
  git push -u origin "$BRANCH"

  log "  Creating PR..."
  PR_URL=$(gh pr create \
    --title "[$TASK_ID] $TASK_TITLE" \
    --body "$(cat <<EOF
## [$TASK_ID] $TASK_TITLE

**Epic:** $TASK_EPIC
**Coder:** \`$CODER\` | **Reviewer:** \`$REVIEWER\`

### Description
$TASK_DESC

### Acceptance Criteria
$TASK_AC

---
*Ralph Loop — multi-agent AI pair programming*
EOF
)" \
    --base "$MAIN_BRANCH" \
    --head "$BRANCH")

  PR_NUMBER=$(echo "$PR_URL" | grep -oE '[0-9]+$')
  log "  PR #$PR_NUMBER: $PR_URL"

  # ── Review step ────────────────────────────────────────────────────────────

  if [[ "$SKIP_REVIEW" == "true" ]]; then
    log "  Skipping review (--skip-review). Auto-merging."
    gh pr comment "$PR_NUMBER" --body "*Review skipped — merged automatically via \`--skip-review\`.*"
  else
    log "  Fetching diff for review ($REVIEWER)..."
    PR_DIFF=""
    for _retry in 1 2 3 4 5; do
      PR_DIFF=$(gh pr diff "$PR_NUMBER" 2>/dev/null) && break
      log "  gh pr diff returned nothing (attempt $_retry/5) — waiting 10s..."
      sleep 10
    done

    REVIEW_PROMPT="You are the code reviewer for a pull request in the playchitect project.

The code was written by $CODER. You are $REVIEWER.

PR: [$TASK_ID] $TASK_TITLE
Acceptance criteria: $TASK_AC

Diff:
---
$PR_DIFF
---

Write a concise code review covering:
1. Correctness — does the implementation satisfy the acceptance criteria?
2. Code quality — readability, naming, structure
3. Test quality — are tests meaningful and sufficient?
4. Any bugs, edge cases, or concerns

Be constructive and specific. End your review with exactly one of:
- **APPROVED** — code is good to merge as-is
- **CHANGES REQUESTED: {brief reason}** — if there are real issues

Output only the review text. It will be posted as a GitHub PR comment."

    log "  Running reviewer ($REVIEWER)..."
    REVIEW_TEXT=$(run_reviewer "$REVIEWER" "$REVIEW_PROMPT" 2>&1 | tee -a "$LOG_FILE")

    log "  Posting review comment..."
    gh pr comment "$PR_NUMBER" --body "$(cat <<EOF
## Code Review by \`$REVIEWER\`

$REVIEW_TEXT

---
*Implemented by \`$CODER\` · Reviewed by \`$REVIEWER\`*
EOF
)"
  fi

  # ── Merge ──────────────────────────────────────────────────────────────────

  log "  Waiting for CI to pass on PR #$PR_NUMBER..."
  for _wait in $(seq 1 60); do
    sleep 30
    CI_STATUS=$(gh pr checks "$PR_NUMBER" --json state -q '.[].state' 2>/dev/null | sort -u | tr '\n' ' ')
    if echo "$CI_STATUS" | grep -q "FAILURE\|ERROR"; then
      log "  CI FAILED on PR #$PR_NUMBER — stopping. Fix the failure and re-run ralph."
      exit 1
    fi
    if echo "$CI_STATUS" | grep -qv "PENDING\|IN_PROGRESS\|QUEUED\|WAITING\|EXPECTED" && [[ -n "$CI_STATUS" ]]; then
      log "  CI passed (check ${_wait}/60, ci=$CI_STATUS)"
      break
    fi
    log "  Still waiting... (check ${_wait}/60, ci=$CI_STATUS)"
  done
  log "  Merging PR #$PR_NUMBER..."
  gh pr merge "$PR_NUMBER" --merge --delete-branch

  git checkout "$MAIN_BRANCH"
  git pull --ff-only origin "$MAIN_BRANCH"

  log "Iteration $ITERATION complete: [$TASK_ID] $TASK_TITLE | $PR_URL"

  sleep 2

done

log "Loop finished. $ITERATION iterations. $(count_remaining) tasks remaining."
echo ""
echo "Summary: $ITERATION iterations run, $(count_remaining) tasks remaining."
