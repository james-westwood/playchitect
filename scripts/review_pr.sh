#!/usr/bin/env bash
# review_pr.sh — Run Gemini code review on the current feature branch.
#
# Usage:
#   ./scripts/review_pr.sh                        # Compare current branch → main
#   ./scripts/review_pr.sh develop                # Compare current branch → develop
#   ./scripts/review_pr.sh main gemini-2.0-flash  # Use a specific model
#
# Gemini must be installed: npm install -g @google/gemini-cli
# gh CLI must be installed and authenticated for GitHub commenting.
# Review instructions live in GEMINI.md at the repo root.

set -euo pipefail

BASE="${1:-main}"
MODEL="${2:-gemini-2.0-flash}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
GEMINI_INSTRUCTIONS="$REPO_ROOT/GEMINI.md"

# ── Sanity checks ──────────────────────────────────────────────────────────────

if ! command -v gemini &>/dev/null; then
    echo "ERROR: gemini CLI not found. Install with: npm install -g @google/gemini-cli"
    exit 1
fi

if ! command -v gh &>/dev/null; then
    echo "ERROR: gh CLI not found. Install from https://cli.github.com"
    exit 1
fi

if [ ! -f "$GEMINI_INSTRUCTIONS" ]; then
    echo "ERROR: GEMINI.md not found at $GEMINI_INSTRUCTIONS"
    exit 1
fi

BRANCH=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)

if [ "$BRANCH" = "$BASE" ]; then
    echo "ERROR: You are on '$BASE'. Switch to a feature branch before reviewing."
    exit 1
fi

# ── Find open PR for this branch ───────────────────────────────────────────────

PR_NUMBER=$(gh pr list --head "$BRANCH" --json number --jq '.[0].number' 2>/dev/null || true)

if [ -z "$PR_NUMBER" ] || [ "$PR_NUMBER" = "null" ]; then
    echo "WARNING: No open PR found for branch '$BRANCH'. Review will print locally only."
    POST_TO_GITHUB=false
else
    POST_TO_GITHUB=true
    echo "Found PR #$PR_NUMBER for branch '$BRANCH'."
fi

# ── Gather PR context ──────────────────────────────────────────────────────────

MERGE_BASE=$(git -C "$REPO_ROOT" merge-base "$BASE" "$BRANCH")
PR_DIFF=$(git -C "$REPO_ROOT" diff "$MERGE_BASE"..."$BRANCH")
COMMIT_LOG=$(git -C "$REPO_ROOT" log "$MERGE_BASE".."$BRANCH" --oneline)
CHANGED_FILES=$(git -C "$REPO_ROOT" diff --name-only "$MERGE_BASE"..."$BRANCH")

if [ -z "$PR_DIFF" ]; then
    echo "No changes found between '$BASE' and '$BRANCH'."
    exit 0
fi

# ── Print header ───────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         Gemini PR Review — Playchitect                   ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf  "║  Branch : %-46s ║\n" "$BRANCH → $BASE"
if [ "$POST_TO_GITHUB" = true ]; then
printf  "║  PR     : %-46s ║\n" "#$PR_NUMBER"
fi
printf  "║  Model  : %-46s ║\n" "$MODEL"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "Changed files:"
echo "$CHANGED_FILES" | sed 's/^/  • /'
echo ""
echo "Commits:"
echo "$COMMIT_LOG" | sed 's/^/  • /'
echo ""
echo "Sending diff to Gemini for review..."
echo "────────────────────────────────────────────────────────────"
echo ""

# ── Invoke Gemini and capture output ───────────────────────────────────────────

REVIEW_OUTPUT=$(
    {
        echo "Branch under review: $BRANCH (target: $BASE)"
        echo ""
        echo "## Commits"
        echo "$COMMIT_LOG"
        echo ""
        echo "## Changed Files"
        echo "$CHANGED_FILES"
        echo ""
        echo "## Git Diff"
        echo "$PR_DIFF"
    } | gemini --model "$MODEL" --prompt "$(cat "$GEMINI_INSTRUCTIONS")"
)

# Print review to terminal
echo "$REVIEW_OUTPUT"
echo ""

# ── Post review as GitHub PR comment ───────────────────────────────────────────

if [ "$POST_TO_GITHUB" = true ]; then
    echo "────────────────────────────────────────────────────────────"
    echo "Posting review to PR #$PR_NUMBER on GitHub..."

    COMMENT_BODY="$(printf '### Gemini Code Review\n\n%s' "$REVIEW_OUTPUT")"

    gh pr comment "$PR_NUMBER" --body "$COMMENT_BODY"

    echo "Review posted: $(gh pr view "$PR_NUMBER" --json url --jq '.url')"
fi
