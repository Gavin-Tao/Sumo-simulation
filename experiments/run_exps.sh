#!/bin/bash
# Launch multiple experiments, each in its own tmux session (detached).
# Run from anywhere:
#   cd /home/xiaowen/sumo-rl/experiments && bash run_exps.sh
#
# Each experiment gets its own session named exp86, exp87, etc.
# Attach with: tmux attach -t exp86

declare -A SCRIPT=(
  [86]="traincolight.py"
  [87]="trainorico.py"
  [90]="trainorico.py"
  [93]="trainorico.py"
  [95]="traincolight.py"
  [96]="trainorico.py"
)

declare -A CONFIG=(
  [86]="exp86_colight_1x3_priority_51_WEstr_only.yaml"
  [87]="exp87_colightorig_1x3_priority_51_WEstr_only.yaml"
  [90]="exp90_colightorig_1x3_priority_51_WEstr_only_tuned.yaml"
  [93]="exp93_colightorig_1x3_priority_51_WEstr_only_noisy.yaml"
  [95]="exp95_colight_1x3_priority_51_WEstr_only_noisy.yaml"
  [96]="exp96_colightorig_1x3_priority_51_WEstr_only_noisy.yaml"
)

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${DIR}/tmux"

# Detect conda environment: path-based (~/.../sumo_rl) takes priority over named env
if conda info --envs 2>/dev/null | awk '{print $NF}' | grep -qE "/sumo_rl$"; then
  CONDA_ENV=$(conda info --envs 2>/dev/null | awk '{print $NF}' | grep -E "/sumo_rl$" | head -1)
elif conda info --envs 2>/dev/null | awk '{print $1}' | grep -qx "sumo_rl"; then
  CONDA_ENV="sumo_rl"
else
  CONDA_ENV=""
fi

if [ -n "$CONDA_ENV" ]; then
  ACTIVATE="conda activate ${CONDA_ENV} &&"
  echo "Using conda env: ${CONDA_ENV}"
else
  ACTIVATE=""
  echo "No sumo_rl conda env found, using current Python"
fi

for EXP in 86 87 90 93 95 96; do
  SCRIPT_FILE="${SCRIPT[$EXP]}"
  CONFIG_FILE="${CONFIG[$EXP]}"
  CMD="${ACTIVATE} cd ${DIR} && python ${SCRIPT_FILE} --config ./configs/${CONFIG_FILE} 2>&1 | tee -a tmux/exp${EXP}_output_\$(date +%Y%m%d_%H%M%S).txt"
  tmux new-session -d -s "exp${EXP}" -c "${DIR}"
  tmux send-keys -t "exp${EXP}" "$CMD" Enter
  echo "Launched exp${EXP}"
done

echo "Done. Check with: tmux ls"
