#!/usr/bin/env bash
# Audit a rebuilt trainers-main candidate branch before it may replace trainers-main.
#
# Usage: audit_next.sh <OLD_TIP> <NEXT> <NEW_BASE>
#   OLD_TIP   the trainers-main tip being replaced (e.g. origin/trainers-main
#             or backup/trainers-main-<date>)
#   NEXT      the candidate branch (e.g. trainers-main-next)
#   NEW_BASE  the NVIDIA commit NEXT is supposed to sit on (e.g. upstream/dev
#             or an explicit SHA)
#
# Checks:
#   1. NEXT is exactly NEW_BASE + linear commits (no merges, correct base).
#   2. Intent diff: files NEXT adds over NEW_BASE, mapped per b10 commit.
#   3. Preservation: every line OLD_TIP added over ITS OWN nvidia base still
#      exists in NEXT. Reported drops must each be justified by a specific
#      upstream commit that subsumed them (document in the audit record).
#   4. All Python files changed by the b10 segment byte-compile.
#
# Exit codes: 0 = clean, 1 = structural violation, 2 = dropped-line report
# needs human/agent review (not necessarily wrong — verify subsumption).
set -euo pipefail

OLD_TIP=${1:?usage: audit_next.sh OLD_TIP NEXT NEW_BASE}
NEXT=${2:?}
NEW_BASE=${3:?}
UPSTREAM="upstream/dev"

fail=0

echo "== 1. structure =="
actual_base=$(git merge-base "$NEXT" "$UPSTREAM")
if [[ "$actual_base" != "$(git rev-parse "$NEW_BASE")" ]]; then
  echo "VIOLATION: $NEXT's nvidia base is $(git log --oneline -1 "$actual_base"),"
  echo "           expected $(git log --oneline -1 "$NEW_BASE")"
  fail=1
fi
merges=$(git rev-list --min-parents=2 "$NEW_BASE..$NEXT" || true)
if [[ -n "$merges" ]]; then
  echo "VIOLATION: merge commit(s) on $NEXT:"; git log --oneline --no-walk $merges
  fail=1
fi
[[ "$fail" -eq 0 ]] && echo "OK: $NEXT = $(git rev-parse --short "$NEW_BASE") + $(git rev-list --count "$NEW_BASE..$NEXT") linear commits"
echo

echo "== 2. intent diff (what $NEXT adds over $NEW_BASE) =="
git diff "$NEW_BASE" "$NEXT" --stat | tail -3
for c in $(git rev-list --reverse "$NEW_BASE..$NEXT"); do
  echo "--- $(git log --oneline -1 "$c")"
  git show --stat --format= "$c" | sed 's/^/    /' | tail -5
done
echo

echo "== 3. preservation audit (b10 lines on $OLD_TIP that vanished) =="
OLD_BASE=$(git merge-base "$OLD_TIP" "$UPSTREAM")
dropped_files=0
for f in $(git diff --name-only "$OLD_BASE" "$OLD_TIP" | sort -u); do
  [[ "$f" == "uv.lock" ]] && continue   # machine-generated; never line-audited
  git cat-file -e "$OLD_TIP:$f" 2>/dev/null || continue
  missing=$(comm -23 \
      <(git show "$OLD_TIP:$f" 2>/dev/null | sort -u) \
      <(git show "$OLD_BASE:$f" 2>/dev/null | sort -u) \
    | comm -23 - <(git show "$NEXT:$f" 2>/dev/null | sort -u) \
    | grep -E '[[:alnum:]_]' || true)
  if [[ -n "$missing" ]]; then
    echo "=== $f ==="; printf '%s\n' "$missing"
    dropped_files=$((dropped_files + 1))
  fi
done
if [[ "$dropped_files" -eq 0 ]]; then
  echo "OK: no b10 lines dropped."
else
  echo
  echo "REVIEW REQUIRED: $dropped_files file(s) with dropped b10 lines above."
  echo "Each drop is acceptable ONLY if a specific upstream commit subsumed it;"
  echo "name that commit SHA in the audit record. Otherwise restore the line."
fi
echo

echo "== 4. compile check =="
py_files=$(git diff --name-only "$NEW_BASE" "$NEXT" -- '*.py' || true)
if [[ -n "$py_files" ]]; then
  # shellcheck disable=SC2086
  python3 -m py_compile $py_files && echo "OK: $(echo "$py_files" | wc -l | tr -d ' ') changed .py files compile"
else
  echo "no python changes"
fi

if [[ "$fail" -ne 0 ]]; then exit 1; fi
if [[ "$dropped_files" -ne 0 ]]; then exit 2; fi
echo
echo "AUDIT CLEAN"
