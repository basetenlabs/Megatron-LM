---
name: update-trainers-main
description: Safely update the basetenlabs trainers-main branch to a newer NVIDIA dev base (latest upstream/dev or a specified commit/PR) while preserving its invariant of a linear history of NVIDIA commits plus clearly-separated b10 commits on top. Use when asked to "update trainers-main", "rebase trainers-main onto dev", "pull PR #XXXX into trainers-main", "bump the nvidia base", or when trainers-main has diverged, gained a merge commit, or otherwise violates linearity. Also use to audit trainers-main's topology or to verify a trainers-main-next candidate before landing.
---

# Update trainers-main

## The invariant

`trainers-main` is the b10 fork's **main branch**. Treat it with main-branch
protections. Its history MUST always be:

```
<b10 commits, linear, newest on top>     ← ours, clearly separated
<NVIDIA dev commits>                     ← verbatim upstream history
```

- The b10 segment is everything above `git merge-base origin/trainers-main upstream/dev`.
- No merge commits anywhere in the b10 segment. Updates move the NVIDIA base
  forward and replay the b10 segment on top — never merge.
- Each b10 commit is one feature/fix with a descriptive subject and a
  `Signed-off-by` trailer, so the two segments stay visually distinguishable
  in `git log`.

## Hard safety rules

1. **Never commit, merge, or `git pull` directly on `trainers-main`.** All
   work happens on a `trainers-main-next` candidate branch.
2. **Before any ref change, snapshot the current tip on the remote:**
   `git push origin origin/trainers-main:refs/heads/backup/trainers-main-<YYYYMMDD>`
3. **Never force-push `trainers-main` without explicit human approval**, and
   then only with `--force-with-lease=trainers-main:<expected-old-sha>`.
4. **Never hand-edit or textually merge `uv.lock`** — take one side wholesale
   if it is provably consistent with the merged `pyproject.toml`, otherwise
   regenerate with `uv lock` inside the CI container (see the
   `mcore-build-and-dependency` skill).
5. A history rewrite orphans every branch, worktree, agent, and runbook based
   on the old SHAs. Identify and notify consumers before landing.

## Procedure

### Phase 0 — Inspect

```bash
skills/update-trainers-main/scripts/topology.sh --fetch
```

Prints the current NVIDIA base, the b10 segment, and upstream distance, and
fails if the invariant is already broken (merge commits, local/origin drift).
Fix violations before proceeding.

Pick the target base: `upstream/dev` for "latest", or a specific SHA. If the
request is "include PR #XXXX", resolve its merge commit
(`gh pr view XXXX --repo NVIDIA/Megatron-LM --json mergeCommit`) and verify it
is an ancestor of the chosen base (`git merge-base --is-ancestor <sha> <base>`).

### Phase 1 — Rebuild on a candidate branch

```bash
git push origin origin/trainers-main:refs/heads/backup/trainers-main-$(date +%Y%m%d)
git checkout -b trainers-main-next <TARGET_BASE>
git cherry-pick -x -s <each b10 commit, oldest first>
```

For every b10 commit, first check whether upstream **subsumed** it:

- `git cherry <TARGET_BASE> origin/trainers-main <old-base>` — commits marked
  `-` are patch-identical upstream: drop them.
- A `+` commit can still be *semantically* subsumed (upstream reworked the
  same code). Diff what the commit changed against what upstream changed in
  the same files since the old base. Drop only when upstream demonstrably
  covers the intent (e.g. upstream's lock already pins the revision a
  pyproject edit asked for); record the subsuming upstream SHA. When both
  changes are complementary (fix different instances of the same bug class),
  keep the commit and verify the merged result contains both.

Conflict resolution follows the `nightly-sync` skill's discipline: combine
both sides, never blanket-take one version, watch for squash-merge-chain
traps. For `pyproject.toml`/`uv.lock` conflicts, resolve the TOML by hand,
then apply rule 4 above.

### Phase 2 — Audit (mandatory before pushing)

```bash
skills/update-trainers-main/scripts/audit_next.sh \
  origin/backup/trainers-main-<YYYYMMDD> trainers-main-next <TARGET_BASE>
```

Four checks: (1) candidate = target base + linear commits; (2) intent diff —
every changed file must map to a b10 commit, nothing stray; (3) preservation —
every line the old tip added over its own base must survive, and each reported
drop must be justified by a named subsuming upstream SHA (exit 2 = review the
report); (4) changed Python files compile.

Additionally run the repo's format/lint on touched files and, for changes near
DSv4/attention/CP code, a smoke training run before landing.

### Phase 3 — Land (human-gated)

```bash
git push -u origin trainers-main-next
```

**Stop here by default.** Present the audit results and the landing command;
the swap itself requires explicit approval:

```bash
git push origin trainers-main-next:trainers-main \
  --force-with-lease=trainers-main:<current-origin-tip-sha>
```

After landing: update local `trainers-main` (`git branch -f` + checkout),
notify consumers of the new SHAs, keep the backup branch for at least a week,
and delete `trainers-main-next`.

## Recovering a broken trainers-main

If `topology.sh` reports violations (merge commit, diverged local, twins of
b10 commits with different SHAs on both sides): do NOT try to merge your way
out. Audit what the stray local/merge state contains vs `origin/trainers-main`
at the **tree level** (`git diff <origin-tip> <stray-tip> --stat`) — stray
work is often already upstream under a different SHA or obsolete. Salvage
genuinely unique changes as cherry-picks onto a candidate branch via the
normal procedure, then reset the broken ref to `origin/trainers-main`.
