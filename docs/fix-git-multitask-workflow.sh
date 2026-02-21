#!/usr/bin/env bash
# fix-git-multitask-workflow.sh
#
# Reorganizes branches so that feature work lives on dedicated feature branches
# instead of main. Then pushes the feature branches to your fork (origin).
#
# This script assumes:
#   - 'origin' points to YOUR fork (e.g., github.com/drgmo/STAMP)
#   - You may optionally have 'upstream' pointing to the lab's original repo
#
# Usage:
#   chmod +x docs/fix-git-multitask-workflow.sh
#   ./docs/fix-git-multitask-workflow.sh

set -euo pipefail

echo "=== Step 1: Ensure we are on master ==="
git checkout master

# --- Step 2: Commit any uncommitted feature work ---
if [ -n "$(git status --porcelain)" ]; then
    echo "=== Step 2: Committing uncommitted changes ==="
    git add -A
    git commit -m "feat: add multitask training branch support"
else
    echo "=== Step 2: No uncommitted changes, skipping ==="
fi

# --- Step 3: Create feature branches (if they don't already exist) ---
echo "=== Step 3: Creating feature branches ==="
if git show-ref --verify --quiet refs/heads/feature/multitask; then
    echo "  feature/multitask already exists, skipping"
else
    git branch feature/multitask
    echo "  Created feature/multitask"
fi

if git show-ref --verify --quiet refs/heads/feature/attention-export; then
    echo "  feature/attention-export already exists, skipping"
else
    git branch feature/attention-export feature/multitask
    echo "  Created feature/attention-export"
fi

# --- Step 4: Reset master to match origin/main ---
echo "=== Step 4: Resetting master to origin/main ==="
git fetch origin main
git reset --hard origin/main

# --- Step 5: Push feature branches to your fork ---
echo "=== Step 5: Pushing feature branches to origin (your fork) ==="
git push -u origin feature/multitask
git push -u origin feature/attention-export

echo ""
echo "=== Done! Verification ==="
echo ""
echo "master (should match origin/main):"
git log --oneline master -3
echo ""
echo "feature/multitask:"
git log --oneline feature/multitask -3
echo ""
echo "feature/attention-export:"
git log --oneline feature/attention-export -3
echo ""
echo "Remote branches on origin:"
git branch -r | grep -E "feature/(multitask|attention-export)" || echo "  (none yet — push may have failed)"
