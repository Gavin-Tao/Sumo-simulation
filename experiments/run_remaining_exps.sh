#!/bin/bash
# Run experiments from exp85-96 that were not included in run_exps.sh, plus exp99.
# Each experiment gets its own tmux session.
# Attach with: tmux attach -t exp85
#
# Experiments:
#   85  — CoeffDQN orig
#   88  — CoLightDQN tuned
#   89  — CoeffDQN tuned
#   91  — CoLightDQN noisy (tuned base)
#   92  — CoeffDQN noisy (tuned base)
#   94  — CoeffDQN noisy (orig base)
#   99  — CoeffDQN noisy + tuned, runs=3 (best CoeffDQN, statistical comparison)

declare -A SCRIPT=(
  [85]="traincoeff.py"
  [88]="traincolight.py"
  [89]="traincoeff.py"
  [91]="traincolight.py"
  [92]="traincoeff.py"
  [94]="traincoeff.py"
  [99]="traincoeff.py"
)

declare -A CONFIG=(
  [85]="exp85_coeff_1x3_priority_51_WEstr_only.yaml"
  [88]="exp88_colight_1x3_priority_51_WEstr_only_tuned.yaml"
  [89]="exp89_coeff_1x3_priority_51_WEstr_only_tuned.yaml"
  [91]="exp91_colight_1x3_priority_51_WEstr_only_noisy.yaml"
  [92]="exp92_coeff_1x3_priority_51_WEstr_only_noisy.yaml"
  [94]="exp94_coeff_1x3_priority_51_WEstr_only_noisy.yaml"
  [99]="exp99_coeff_1x3_priority_51_WEstr_only_noisy_tuned.yaml"
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

for EXP in 85 88 89 91 92 94 99; do
  SCRIPT_FILE="${SCRIPT[$EXP]}"
  CONFIG_FILE="${CONFIG[$EXP]}"
  CMD="${ACTIVATE} cd ${DIR} && python ${SCRIPT_FILE} --config ./configs/${CONFIG_FILE} 2>&1 | tee -a tmux/exp${EXP}_output_\$(date +%Y%m%d_%H%M%S).txt"
  tmux new-session -d -s "exp${EXP}" -c "${DIR}"
  tmux send-keys -t "exp${EXP}" "$CMD" Enter
  echo "Launched exp${EXP}"
done

echo "Done. Check with: tmux ls"
