#!/usr/bin/env bash
# ralph-once.sh — Run one Ralph loop iteration (branch → atomic commits → PR → review → merge).
#
# Usage:
#   ./ralph-once.sh               # run next ralph-owned task
#   ./ralph-once.sh --task TASK-03  # force a specific task ID
#
# Requirements:
#   - opencode CLI, gh CLI (authenticated)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FORCED_TASK_ID="${2:-}"
MAIN_BRANCH="main"
TODAY=$(date +%Y-%m-%d)

# ── Helpers ───────────────────────────────────────────────────────────────────

CODER_MODEL="opencode/nemotron-3-super-free"
REVIEWER_MODEL="opencode/big-pickle"

run_coder() {
  local agent="$1" prompt="$2"
  if opencode run -m "$CODER_MODEL" --dangerously-skip-permissions "$prompt"; then
    return 0
  else
    echo "  $CODER_MODEL coder failed — falling back to $REVIEWER_MODEL" >&2
    opencode run -m "$REVIEWER_MODEL" --dangerously-skip-permissions "$prompt"
  fi
}

run_reviewer() {
  local agent="$1" prompt="$2"
  if opencode run -m "$REVIEWER_MODEL" "$prompt"; then
    return 0
  else
    echo "  $REVIEWER_MODEL reviewer failed — falling back to $CODER_MODEL" >&2
    opencode run -m "$CODER_MODEL" "$prompt"
  fi
}

next_task_field() {
  local field="$1" filter="${2:-}"
  python3 -c "
import json
with open('prd.json') as f: prd = json.load(f)
if '$filter':
    t = next(x for x in prd['tasks'] if x['id'] == '$filter')
else:
    t = [x for x in prd['tasks'] if not x['completed'] and x.get('owner') != 'human'][0]
print(t['$field'])
" 2>/dev/null
}

# ── Preflight ─────────────────────────────────────────────────────────────────

command -v gh       >/dev/null 2>&1 || { echo "ERROR: 'gh' not found"; exit 1; }
command -v opencode >/dev/null 2>&1 || { echo "ERROR: 'opencode' not found"; exit 1; }
gh auth status      >/dev/null 2>&1 || { echo "ERROR: not authenticated with gh — run: gh auth login"; exit 1; }

# ── Resolve task ──────────────────────────────────────────────────────────────

echo "=== Ralph — Single Iteration (playchitect) ==="
echo ""

TASK_ID=$(next_task_field id "$FORCED_TASK_ID")
TASK_TITLE=$(next_task_field title "$FORCED_TASK_ID")
TASK_DESC=$(next_task_field description "$FORCED_TASK_ID")
TASK_AC=$(next_task_field acceptance_criteria "$FORCED_TASK_ID")
TASK_EPIC=$(next_task_field epic "$FORCED_TASK_ID")

TASK_NOTE=$(python3 -c "
import json
with open('prd.json') as f: prd = json.load(f)
if '$FORCED_TASK_ID':
    t = next(x for x in prd['tasks'] if x['id'] == '$FORCED_TASK_ID')
else:
    t = [x for x in prd['tasks'] if not x['completed'] and x.get('owner') != 'human'][0]
print(t.get('note', ''))
" 2>/dev/null || true)

TASK_FILES=$(python3 -c "
import json
with open('prd.json') as f: prd = json.load(f)
if '$FORCED_TASK_ID':
    t = next(x for x in prd['tasks'] if x['id'] == '$FORCED_TASK_ID')
else:
    t = [x for x in prd['tasks'] if not x['completed'] and x.get('owner') != 'human'][0]
files = t.get('files', [])
print('\n'.join(files))
" 2>/dev/null || true)

# Check if human-owned (only relevant when auto-picking)
if [[ -z "$FORCED_TASK_ID" ]]; then
  OWNER=$(python3 -c "
import json
with open('prd.json') as f: prd = json.load(f)
incomplete = [t for t in prd['tasks'] if not t['completed']]
if incomplete: print(incomplete[0].get('owner', ''))
" 2>/dev/null)
  if [[ "$OWNER" == "human" ]]; then
    echo "Next task [$TASK_ID] $TASK_TITLE is YOURS to implement."
    echo "Mark it complete in prd.json, then re-run ralph."
    exit 0
  fi
fi

# ── Create GitHub issue if not already linked ──────────────────────────────────

CLOSES_LINE=""
if [[ "$TASK_NOTE" =~ \#([0-9]+) ]]; then
  # Existing issue number in note field — reuse it
  CLOSES_LINE="Closes #${BASH_REMATCH[1]}"
  echo "  Linked issue: #${BASH_REMATCH[1]} (from prd.json note)"
else
  # No issue yet — create one now
  echo "--- Creating GitHub issue for [$TASK_ID] ---"
  ISSUE_URL=$(gh issue create \
    --title "[$TASK_ID] $TASK_TITLE" \
    --body "$(printf '%s\n\n**Acceptance criteria:**\n%s' "$TASK_DESC" "$TASK_AC")" \
    --label "type-bug" 2>/dev/null || \
  gh issue create \
    --title "[$TASK_ID] $TASK_TITLE" \
    --body "$(printf '%s\n\n**Acceptance criteria:**\n%s' "$TASK_DESC" "$TASK_AC")" 2>/dev/null || true)
  ISSUE_NUMBER=$(echo "$ISSUE_URL" | grep -oE '[0-9]+$' || true)
  if [[ -n "$ISSUE_NUMBER" ]]; then
    CLOSES_LINE="Closes #${ISSUE_NUMBER}"
    # Write issue number back to prd.json so future runs don't recreate it
    python3 -c "
import json
with open('prd.json') as f: prd = json.load(f)
for t in prd['tasks']:
    if t['id'] == '$TASK_ID':
        t['note'] = 'GitHub issue #$ISSUE_NUMBER'
        break
with open('prd.json', 'w') as f:
    json.dump(prd, f, indent=2)
    f.write('\n')
"
    echo "  Created GitHub issue #${ISSUE_NUMBER}: $ISSUE_URL"
  else
    echo "  Warning: could not create GitHub issue — continuing without Closes link"
  fi
fi

BRANCH="ralph/task-${TASK_ID}-${TASK_TITLE}"

# Derive display names from model IDs (strip opencode/ prefix)
CODER="${CODER_MODEL#opencode/}"
REVIEWER="${REVIEWER_MODEL#opencode/}"

echo ""
echo "  Task:     [$TASK_ID] $TASK_TITLE"
echo "  Epic:     $TASK_EPIC"
echo "  Branch:   $BRANCH"
echo "  Coder:    $CODER  |  Reviewer: $REVIEWER"
echo ""

# ── Branch setup ──────────────────────────────────────────────────────────────

git checkout "$MAIN_BRANCH"
git pull --ff-only origin "$MAIN_BRANCH" 2>/dev/null || true
git checkout -b "$BRANCH"

# ── Coding step ───────────────────────────────────────────────────────────────

echo "--- Coding ($CODER) ---"

FILES_HINT=""
if [[ -n "$TASK_FILES" ]]; then
  FILES_HINT="
Files likely relevant to this task (read these first):
$TASK_FILES"
fi

CODER_PROMPT="You are the CODER implementing task [$TASK_ID] $TASK_TITLE for the playchitect project. Another AI will review your work — write clean, production-quality code.

Read CLAUDE.md for project conventions.

Epic: $TASK_EPIC
Description: $TASK_DESC
Acceptance criteria: $TASK_AC
${FILES_HINT}

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
- Use uv for all Python commands
- Follow all conventions in CLAUDE.md"

run_coder "$CODER" "$CODER_PROMPT"

# ── Push and open PR ──────────────────────────────────────────────────────────

echo ""
echo "--- Pushing and creating PR ---"
git push -u origin "$BRANCH"

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
${CLOSES_LINE:+
$CLOSES_LINE}
---
*Ralph Loop — multi-agent AI pair programming*
EOF
)" \
  --base "$MAIN_BRANCH" \
  --head "$BRANCH")

PR_NUMBER=$(echo "$PR_URL" | grep -oE '[0-9]+$')
echo "PR #$PR_NUMBER: $PR_URL"

# ── Review step ───────────────────────────────────────────────────────────────

echo ""
echo "--- Reviewing ($REVIEWER) ---"
# Wait for GitHub to propagate the PR before fetching the diff
sleep 5
PR_DIFF=$(gh pr diff "$PR_NUMBER")

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

REVIEW_TEXT=$(run_reviewer "$REVIEWER" "$REVIEW_PROMPT")

gh pr comment "$PR_NUMBER" --body "$(cat <<EOF
## Code Review by \`$REVIEWER\`

$REVIEW_TEXT

---
*Implemented by \`$CODER\` · Reviewed by \`$REVIEWER\`*
EOF
)"

# ── Enable auto-merge (CI must pass before GitHub merges) ─────────────────────

echo ""
echo "--- Enabling auto-merge on PR #$PR_NUMBER (squash, waits for CI) ---"
gh pr merge "$PR_NUMBER" --auto --squash --delete-branch

echo ""
echo "=== Ralph done — PR open, auto-merge enabled ==="
echo "    Task:  [$TASK_ID] $TASK_TITLE"
echo "    PR:    $PR_URL"
echo ""
echo "Delivery manager: monitor CI, verify ACs, then PR will self-merge when green."
