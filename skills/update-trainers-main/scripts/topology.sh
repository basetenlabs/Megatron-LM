#!/usr/bin/env bash
# Report and verify the trainers-main topology invariant:
#   trainers-main == <nvidia dev commits> + <linear b10 commits on top>
#
# Usage: topology.sh [--fetch]
#   --fetch   run `git fetch upstream dev` and `git fetch origin` first
#
# Exit codes: 0 = invariants hold, 1 = violation found.
set -euo pipefail

BRANCH="origin/trainers-main"
UPSTREAM="upstream/dev"

if [[ "${1:-}" == "--fetch" ]]; then
  git fetch upstream dev --quiet
  git fetch origin --quiet
fi

git rev-parse --verify -q "$BRANCH" >/dev/null || { echo "FATAL: $BRANCH not found"; exit 1; }
git rev-parse --verify -q "$UPSTREAM" >/dev/null || { echo "FATAL: $UPSTREAM not found (git remote add upstream git@github.com:NVIDIA/Megatron-LM.git)"; exit 1; }

BASE=$(git merge-base "$BRANCH" "$UPSTREAM")
VIOLATIONS=0

echo "== NVIDIA base =="
git log --oneline -1 "$BASE"
echo
echo "== b10 commits on top (oldest first) =="
git log --oneline --reverse "$BASE..$BRANCH"
echo
echo "== upstream distance =="
echo "$(git rev-list --count "$BASE..$UPSTREAM") commits on $UPSTREAM beyond the base"
echo "upstream tip: $(git log --oneline -1 "$UPSTREAM")"
echo

# Invariant 1: the b10 segment is linear (no merge commits).
MERGES=$(git rev-list --min-parents=2 "$BASE..$BRANCH")
if [[ -n "$MERGES" ]]; then
  echo "VIOLATION: merge commit(s) in the b10 segment:"
  git log --oneline --no-walk $MERGES
  VIOLATIONS=1
fi

# Invariant 2: local trainers-main (if checked out anywhere) matches origin.
if git rev-parse --verify -q trainers-main >/dev/null; then
  if [[ "$(git rev-parse trainers-main)" != "$(git rev-parse "$BRANCH")" ]]; then
    echo "VIOLATION: local trainers-main ($(git rev-parse --short trainers-main)) != $BRANCH ($(git rev-parse --short "$BRANCH"))"
    echo "  Never commit locally to trainers-main. Reconcile before doing anything else."
    VIOLATIONS=1
  fi
fi

# Invariant 3: every b10 commit carries a Signed-off-by trailer.
for c in $(git rev-list "$BASE..$BRANCH"); do
  if [[ -z "$(git log -1 --format='%(trailers:key=Signed-off-by,valueonly)' "$c")" ]]; then
    echo "WARNING: $(git log --oneline -1 "$c") has no Signed-off-by trailer"
  fi
done

if [[ "$VIOLATIONS" -eq 0 ]]; then
  echo "OK: topology invariants hold."
else
  echo "FAILED: fix violations before updating the branch."
fi
exit "$VIOLATIONS"
