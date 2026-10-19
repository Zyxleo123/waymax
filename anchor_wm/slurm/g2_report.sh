#!/bin/bash
#SBATCH --job-name=awm_g2rep
#SBATCH --partition=cpu
#SBATCH --qos=qos_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=0-12:00:00
#SBATCH --output=/zfsauton/scratch/ishanj/anchor_wm/logs/g2_report_%j.out
#SBATCH --error=/zfsauton/scratch/ishanj/anchor_wm/logs/g2_report_%j.err
#
# Refreshes the G2 closed-loop report (anchor_wm/g2_report.py) every INTERVAL seconds until the run
# has all EXPECTED scenes (then writes DONE and exits). ONCE=1 writes one report and exits.
# PUSH_BRANCH=<branch>: also commit the report to that GitHub branch (under PUSH_PATH) every
# PUSH_EVERY refreshes and at the end, from a throwaway shallow clone (the cluster checkout is untouched).
#
#   RUN=<closed_loop run name> PUSH_BRANCH=ishan sbatch anchor_wm/slurm/g2_report.sh
set -euo pipefail
set +u   # conda's activate.d scripts read unset variables
source /zfsauton/scratch/sbellad/miniconda3/bin/activate /zfsauton/scratch/sbellad/conda_envs/waymax_rs
set -u
export PYTHONPATH=/zfsauton/scratch/ishanj/waymax_rs
RUN_DIR="${RUN_ROOT:-/zfsauton/scratch/ishanj/anchor_wm/closed_loop}/${RUN}"
OUT_DIR="${OUT_DIR:-/zfsauton/scratch/ishanj/anchor_wm/results/${RUN}}"
PUSH_BRANCH="${PUSH_BRANCH:-}"
PUSH_PATH="${PUSH_PATH:-results/anchor_wm/fail1500_g2_ft_rule_hdg}"
PUSH_EVERY="${PUSH_EVERY:-6}"
REMOTE="${REMOTE:-git@github.com:mini-hong/waymax_rs.git}"
export GIT_SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=30"

push_report() {
  local done_n msg w attempt
  done_n=$(python -c "import json; d=json.load(open('${OUT_DIR}/report.json')); print('%d scenes, target %s (%s)' % (d['scenes_done'], d.get('target', d['expected']), 'final' if d['complete'] else 'partial'))")
  msg="[auto] ${PUSH_TITLE:-fail1500 G2 report}: ${done_n}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  for attempt in 1 2 3; do   # a fresh clone per attempt, so a moved branch just means re-applying
    w=$(mktemp -d)
    if timeout 300 git clone -q --depth 1 --branch "${PUSH_BRANCH}" "${REMOTE}" "${w}/repo"; then
      mkdir -p "${w}/repo/${PUSH_PATH}"
      cp "${OUT_DIR}/report.md" "${OUT_DIR}/report.json" "${OUT_DIR}/per_scene.csv" "${w}/repo/${PUSH_PATH}/"
      git -C "${w}/repo" add -f "${PUSH_PATH}"
      if git -C "${w}/repo" diff --cached --quiet; then echo "push: report unchanged"; rm -rf "${w}"; return 0; fi
      git -C "${w}/repo" -c user.name="Ishan Jain" -c user.email="ishanjain@Ishans-MacBook-Air-9.local" commit -q -m "${msg}"
      if timeout 300 git -C "${w}/repo" push -q origin "HEAD:${PUSH_BRANCH}"; then
        echo "push: ${done_n} -> ${PUSH_BRANCH} ($(git -C "${w}/repo" rev-parse --short HEAD))"; rm -rf "${w}"; return 0
      fi
    fi
    rm -rf "${w}"; echo "push: attempt ${attempt} failed"; sleep 30
  done
  return 1
}

i=0
while :; do
  python -m anchor_wm.g2_report --run_dir "${RUN_DIR}" --out_dir "${OUT_DIR}" --expected "${EXPECTED:-1500}" \
    --label "${LABEL:-ft_rule_hdg scorer}" ${REPORT_ARGS:-}
  finished=0; [[ "${ONCE:-0}" == "1" || -f "${OUT_DIR}/DONE" ]] && finished=1
  if [[ -n "${PUSH_BRANCH}" ]] && (( finished || i % PUSH_EVERY == 0 )); then
    push_report || echo "push: giving up for now"
  fi
  (( finished )) && break
  i=$((i + 1))
  sleep "${INTERVAL:-600}"
done
