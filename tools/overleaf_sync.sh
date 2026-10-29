#!/usr/bin/env bash
# Sync paper/ with the Overleaf project (remote "overleaf", which holds only paper/'s contents).
#
#   git overleaf-pull          bring Overleaf edits into paper/ (then review and commit them)
#   git overleaf-push ["msg"]  send the committed paper/ to Overleaf
#
# refs/sync/overleaf remembers the Overleaf commit paper/ was last in sync with. Pull applies only
# what changed on Overleaf since then, so local edits survive; push refuses if Overleaf has edits
# you haven't pulled, so nobody's work gets overwritten. Push sends HEAD:paper (committed files
# only), so stray untracked files in paper/ never reach Overleaf.
set -euo pipefail

MARK=refs/sync/overleaf
die() { echo "error: $*" >&2; exit 1; }

cd "$(git rev-parse --show-toplevel)"
git fetch -q overleaf
remote=$(git rev-parse overleaf/main)
base=$(git rev-parse -q --verify "$MARK" || true)
[ -n "$base" ] || die "no sync marker. If paper/ currently matches Overleaf, run: git update-ref $MARK overleaf/main"

case "${1:-}" in
pull)
    if [ "$base" = "$remote" ]; then echo "Already up to date with Overleaf."; exit 0; fi
    echo "Overleaf changes since last sync:"
    git diff --stat "$base" "$remote"
    if git diff --binary "$base" "$remote" | git apply --3way --directory=paper; then
        git update-ref "$MARK" "$remote"
        echo "Pulled. Review with 'git diff --cached -- paper', then commit."
    elif [ -n "$(git diff --name-only --diff-filter=U)" ]; then
        git update-ref "$MARK" "$remote"
        echo "Pulled with conflicts. Fix the files listed as 'both modified' in 'git status', then commit."
    else
        die "could not apply Overleaf changes. Commit or stash your paper/ edits, then pull again."
    fi
    ;;
push)
    [ "$base" = "$remote" ] || die "Overleaf has edits you haven't pulled. Run: git overleaf-pull"
    if [ -n "$(git status --porcelain -- paper)" ]; then
        git status -s -- paper
        die "paper/ has uncommitted changes (above). Commit them first; push only sends committed files."
    fi
    tree=$(git rev-parse HEAD:paper)
    if [ "$tree" = "$(git rev-parse "$remote^{tree}")" ]; then echo "Overleaf already up to date."; exit 0; fi
    msg=${2:-$(git log -1 --format=%s -- paper)}
    echo "Pushing to Overleaf: $msg"
    git diff --stat "$remote" "$tree"
    c=$(git commit-tree "$tree" -p "$remote" -m "$msg")
    git push -q overleaf "$c:refs/heads/main"
    git update-ref "$MARK" "$c"
    echo "Done."
    ;;
*)
    die "usage: $0 pull | push [message]"
    ;;
esac
