#!/usr/bin/env bash
# review_pr.sh — Run Gemini code review on the current feature branch.
#
# Usage:
#   ./scripts/review_pr.sh              # Compare current branch → main
#   ./scripts/review_pr.sh develop      # Compare current branch → develop
#
# Gemini must be installed: npm install -g @google/gemini-cli
# Review instructions live in GEMINI.md at the repo root.

set -euo pipefail

BASE="${1:-main}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
GEMINI_INSTRUCTIONS="$REPO_ROOT/GEMINI.md"

# ── Sanity checks ──────────────────────────────────────────────────────────────

if ! command -v gemini &>/dev/null; then
    echo "ERROR: gemini CLI not found. Install with: npm install -g @google/gemini-cli"
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

# ── Invoke Gemini ──────────────────────────────────────────────────────────────
# Pipe the diff as stdin; GEMINI.md content becomes the -p prompt.

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
} | gemini --prompt "$(cat "$GEMINI_INSTRUCTIONS")"
