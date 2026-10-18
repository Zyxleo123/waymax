#!/usr/bin/env bash
# Point the code and the copied inputs at a new cluster. Assumes the trees listed in
# env/transfer_list.txt were copied under ONE root with their full autonlab paths kept
# (rsync --files-from / -R), so the rewrite is a pure prefix:
#   /zfsauton/...  ->  $NEWROOT/zfsauton/...      /zfsauton2/...  ->  $NEWROOT/zfsauton2/...
#
# usage: bash env/relocate_paths.sh NEWROOT [--dry-run] [CHECKOUT...]
#   CHECKOUT   git checkouts whose tracked text files are rewritten (default: this one)
#   --dry-run  list what would change, touch nothing
# Optional env:
#   GPU_PARTITION / CPU_PARTITION  rewrite '#SBATCH --partition=general' / '=cpu' lines
#   DROP_QOS=1                     delete '#SBATCH --qos=' lines (autonlab QOS names)
#   DROP_NODE_PINS=1               drop autonlab node pins: '#SBATCH --exclude/--nodelist=gpuNN'
#                                  lines, literal '--exclude=gpuNN' flags, '${X-gpuNN}' defaults
#   GPU_GRES=gpu:<type>:1          rewrite '#SBATCH --gres=gpu:1' and '--gres=gpu:<type>:1' lines
#
# Also rewrites the small input files under the copied scratch (split JSONs, subset
# manifests, run env files) and re-points absolute symlinks (the ES subsets link into
# WOMD). Vendored byte-identical trees are skipped (es_baseline has a sha256 manifest).
# Idempotent: a path already under NEWROOT is never rewritten twice.
set -euo pipefail
NEWROOT="${1:?usage: relocate_paths.sh NEWROOT [--dry-run] [CHECKOUT...]}"; shift
NEWROOT="${NEWROOT%/}"
[[ "$NEWROOT" == /* ]] || { echo "NEWROOT must be absolute" >&2; exit 1; }
DRY=0; if [[ "${1:-}" == --dry-run ]]; then DRY=1; shift; fi
(( $# )) || set -- "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
export NEWROOT
# (?<![\w./]) = the path starts here, so "$NEWROOT/zfsauton/..." does not match again.
PAT='(?<![\w./])/zfsauton(2?)/'

rewrite() {  # files...
  local f n
  for f in "$@"; do
    n=$(perl -ne "\$n += () = m#$PAT#g; END { print \$n + 0 }" "$f")
    (( n > 0 )) || continue
    if (( DRY )); then echo "  $n  $f"; else perl -pi -e "s#$PAT#\$ENV{NEWROOT}/zfsauton\$1/#g" "$f"; fi
  done
}

for co in "$@"; do
  echo "== checkout $co"
  # env/ holds this script, the transfer list and the guide: source paths, never rewritten.
  mapfile -t files < <(git -C "$co" grep -lIE '/zfsauton2?/' -- . ':!env' ':!es_baseline' \
                         ':!rl/diffusion_ft/wm_vendor' ':!multiFuture_GaussianWorldModel_ishan' | sed "s|^|$co/|")
  echo "   ${#files[@]} files reference /zfsauton*"
  if (( ${#files[@]} )); then rewrite "${files[@]}"; fi
  if [[ -n "${GPU_PARTITION:-}${CPU_PARTITION:-}" || "${DROP_QOS:-0}" == 1 ]]; then
    mapfile -t sb < <(git -C "$co" grep -lE '^#SBATCH --(partition|qos)=' -- '*.sbatch' '*.sh' ':!env' ':!es_baseline' | sed "s|^|$co/|")
    echo "   ${#sb[@]} batch scripts with #SBATCH partition/qos lines"
    if (( ! DRY )) && (( ${#sb[@]} )); then
      [[ -n "${GPU_PARTITION:-}" ]] && sed -i "s|^#SBATCH --partition=general\$|#SBATCH --partition=$GPU_PARTITION|" "${sb[@]}"
      [[ -n "${CPU_PARTITION:-}" ]] && sed -i "s|^#SBATCH --partition=cpu\$|#SBATCH --partition=$CPU_PARTITION|" "${sb[@]}"
      [[ "${DROP_QOS:-0}" == 1 ]] && sed -i '/^#SBATCH --qos=/d' "${sb[@]}"
    fi
  fi
  if [[ "${DROP_NODE_PINS:-0}" == 1 ]]; then
    mapfile -t pins < <(git -C "$co" grep -lE -- '--(exclude|nodelist)=gpu[0-9]|-gpu[0-9][0-9,]*\}' -- '*.sbatch' '*.sh' \
                          ':!env' ':!es_baseline' ':!legacy' | sed "s|^|$co/|")
    echo "   ${#pins[@]} scripts with autonlab node pins"
    if (( ! DRY )) && (( ${#pins[@]} )); then
      perl -ni -e 'next if /^#SBATCH --(exclude|nodelist)=gpu[0-9]/;
                   s/(\$\{[A-Z_]+-)gpu[0-9][0-9,]*\}/$1}/g; s/--(exclude|nodelist)=gpu[0-9][0-9,]* ?//g; print' "${pins[@]}"
    fi
  fi
  if [[ -n "${GPU_GRES:-}" ]]; then
    mapfile -t gr < <(git -C "$co" grep -lE '^#SBATCH --gres=gpu:([a-z0-9_]+:)?1$' -- '*.sbatch' '*.sh' ':!env' ':!es_baseline' \
                        ':!legacy' | sed "s|^|$co/|")
    echo "   ${#gr[@]} batch scripts with a 1-GPU #SBATCH --gres line"
    if (( ! DRY )) && (( ${#gr[@]} )); then
      sed -i -E "s|^#SBATCH --gres=gpu:([a-z0-9_]+:)?1\$|#SBATCH --gres=$GPU_GRES|" "${gr[@]}"
    fi
  fi
done

Y="$NEWROOT/zfsauton/scratch/yixiz/waymax_rs"
if [[ -d "$Y" ]]; then
  echo "== copied inputs under $Y"
  mapfile -t inputs < <(find -P "$Y/diffusion_ft/hard_womd" "$Y/es_posthoc" -maxdepth 3 -type f -name '*.json' -size -20M 2>/dev/null
                        find -P "$Y" -type f \( -name env -o -name '*.env' \) -size -2048k 2>/dev/null)
  if (( ${#inputs[@]} )); then rewrite "${inputs[@]}"; fi
  echo "== absolute symlinks under $NEWROOT/zfsauton/scratch"
  n=0
  while IFS= read -r l; do
    t=$(readlink "$l"); n=$((n + 1))
    (( DRY )) || ln -sfn "$NEWROOT$t" "$l"
  done < <(find -P "$NEWROOT/zfsauton/scratch" -type l \( -lname '/zfsauton/*' -o -lname '/zfsauton2/*' \) 2>/dev/null)
  echo "   $n symlinks to re-point"
fi
if (( DRY )); then echo "(dry run: nothing changed)"; fi
