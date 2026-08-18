#!/bin/bash
# Offsite backup of everything worth keeping from Scholar before access ends.
# Re-runnable (incremental). Two targets:
#   1) Google Drive via rclone (remote name "gdrive"; one-time: ~/bin/rclone config)
#   2) Home machine via rsync PULL (run the printed command FROM the home machine)
#
# Tiers:
#   MUST   repo (29G incl. results row-level + frozen npz artifacts), bd clone,
#          ~/.claude (memory + sessions), coursework, dotfile picks
#   SKIP   hf-cache (re-download from HF), conda-envs (environment.lock.yml),
#          scratch tmp
#   NEVER TO CLOUD  ~/.ssh, .env (secrets: home copy only, or re-issue)
set -uo pipefail
RCLONE=~/bin/rclone
DEST=${1:-gdrive:scholar-backup}

echo "== manifest + checksums (results npz/npy/jsonl over 10MB) =="
find /scratch/scholar/skiron/phantom-or-real/results -size +10M \
  \( -name '*.npz' -o -name '*.npy' -o -name '*.jsonl' \) -print0 |
  xargs -0 sha256sum > /scratch/scholar/skiron/phantom-or-real/results/BACKUP_SHA256SUMS.txt
wc -l /scratch/scholar/skiron/phantom-or-real/results/BACKUP_SHA256SUMS.txt

echo "== rclone sync: repo (with .git), beyond-deduction, home picks =="
$RCLONE sync /scratch/scholar/skiron/phantom-or-real "$DEST/phantom-or-real" \
  --transfers 8 --checkers 16 --drive-chunk-size 128M --fast-list -P \
  --exclude '.git/**' # .git goes as a bundle below (Drive hates many small files)
git -C /scratch/scholar/skiron/phantom-or-real bundle create /tmp/phantom-or-real.gitbundle --all
$RCLONE copyto /tmp/phantom-or-real.gitbundle "$DEST/phantom-or-real.gitbundle" -P
$RCLONE sync /scratch/scholar/skiron/beyond-deduction "$DEST/beyond-deduction" \
  --exclude '.git/**' --transfers 8 -P
$RCLONE sync /home/skiron/.claude "$DEST/home/dot-claude" --transfers 8 -P
for d in cs587 cs408 cs373 R nltk_data; do
  [ -d "/home/skiron/$d" ] && $RCLONE sync "/home/skiron/$d" "$DEST/home/$d" --transfers 8 -P
done

echo "== done. Verify with: $RCLONE check /scratch/scholar/skiron/phantom-or-real $DEST/phantom-or-real --one-way --exclude '.git/**' =="
echo
echo "HOME-MACHINE PULL (run from your home computer; repeat any time to re-sync):"
echo "  rsync -avP --exclude hf-cache --exclude conda-envs --exclude tmp \\"
echo "      skiron@scholar.rcac.purdue.edu:/scratch/scholar/skiron/ ~/scholar-backup/scratch/"
echo "  rsync -avP skiron@scholar.rcac.purdue.edu:/home/skiron/ ~/scholar-backup/home/"
