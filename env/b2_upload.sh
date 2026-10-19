#!/usr/bin/env bash
# Upload everything in env/transfer_list.txt (plus Claude memory) to Backblaze B2 with rclone.
#   archives/scratch.tar.gz  the small-file trees: your scratch (results, splits, offsets, ES run
#                            roots; 160k files) and logs/, with their autonlab paths minus the '/'
#   archives/claude.tar.gz   ~/.claude/CLAUDE.md and this project's memory folder
#   archives/SHA256SUMS      checksums of the two archives
#   files/<autonlab path>    big binaries as plain objects: WOMD val shards 0-9, the compact failure
#                            set, policy checkpoints, world models, discriminator, failure list
# Re-run any time (e.g. just before the outage): files/ is incremental, archives are rebuilt.
# The download side is env/MIGRATION.md section 1.
#
# usage: bash env/b2_upload.sh [DEST]      default DEST=b2:proteina/waymax_rs
#   REUSE_ARCHIVES=1  upload the archives already staged in $STAGE instead of rebuilding them
# The proteina bucket also holds the CPSea project (cpsea/): never `rclone sync`/`cleanup` it.
set -euo pipefail
DEST="${1:-b2:proteina/waymax_rs}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
STAGE="${STAGE:-/var/tmp/$USER/waymax_b2}"   # local disk on lov4, not ZFS
NPROC="$(nproc)"
mkdir -p "$STAGE"
echo "dest=$DEST stage=$STAGE host=$(hostname) nproc=$NPROC repo=$REPO @ $(git -C "$REPO" rev-parse --short HEAD)"

ARCHIVE_TREES=(zfsauton/scratch/yixiz/waymax_rs zfsauton/scratch/yixiz/ScenarioMaxWaymoFailures
               zfsauton2/home/yixiz/waymax_rs/logs)
COMPACT=zfsauton/scratch/yixiz/waymax_rs/womd_compact   # binary, inside the scratch tree: sent as files

# --- plain-file list: the transfer list minus the archived trees, directories expanded
: > "$STAGE/files.txt"
grep -v '^#' "$REPO/env/transfer_list.txt" | grep . | sed 's|^/||' | while read -r p; do
  for a in "${ARCHIVE_TREES[@]}"; do [[ "$p" == "$a" ]] && continue 2; done
  if [[ -d "/$p" ]]; then (cd / && find -P "$p" -type f); else echo "$p"; fi
done >> "$STAGE/files.txt"
(cd / && find -P "$COMPACT" -type f) >> "$STAGE/files.txt"
echo "plain files: $(wc -l < "$STAGE/files.txt")"

# --- archives (tar exit 1 = "file changed as we read it": the ES jobs are still writing)
if [[ "${REUSE_ARCHIVES:-0}" == 1 && -s "$STAGE/scratch.tar.gz" && -s "$STAGE/claude.tar.gz" ]]; then
  echo "reusing staged archives:"; (cd "$STAGE" && sha256sum -c SHA256SUMS)
else
rc=0
tar -C / --use-compress-program="pigz -p $NPROC" --warning=no-file-changed --warning=no-file-removed \
    --exclude=.jax_compilation_cache --exclude="$COMPACT" \
    -cf "$STAGE/scratch.tar.gz.tmp" "${ARCHIVE_TREES[@]}" || rc=$?
(( rc <= 1 )) || { echo "tar failed ($rc)" >&2; exit "$rc"; }
mv "$STAGE/scratch.tar.gz.tmp" "$STAGE/scratch.tar.gz"
tar -C "$HOME/.claude" --use-compress-program="pigz -p $NPROC" -cf "$STAGE/claude.tar.gz" \
    CLAUDE.md projects/-zfsauton2-home-yixiz-waymax-rs/memory
(cd "$STAGE" && sha256sum scratch.tar.gz claude.tar.gz > SHA256SUMS && cat SHA256SUMS && ls -l scratch.tar.gz claude.tar.gz)
fi

# --- upload, then verify against what is now in the bucket
ARCH=(--include scratch.tar.gz --include claude.tar.gz --include SHA256SUMS)
rclone copy "$STAGE" "$DEST/archives" "${ARCH[@]}" --stats 30s --stats-one-line
rclone copy / "$DEST/files" --files-from "$STAGE/files.txt" --transfers 16 --stats 30s --stats-one-line
rclone check "$STAGE" "$DEST/archives" "${ARCH[@]}" --one-way
rclone check / "$DEST/files" --files-from "$STAGE/files.txt" --one-way --fast-list
echo "b2_upload: DONE -> $DEST"
