#!/usr/bin/env bash
# Пересчёт шагов 03 → 05 с общей компонентой ТОЛЬКО по 20 RDO (COMPONENT=rdo) для нескольких моделей
# подряд на ОДНОЙ GPU, в отдельной tmux-сессии; лог в logs/. Нужны готовые шаги 00–02.
# Пишет рядом с исходными: component_rdo/, energy_rdo/, ablation_rdo/, reports/*_rdo.md — их не трогает.
# 03 и 04 — CPU (секунды), 05 — GPU (~10–20 мин на модель). Падение одной модели не останавливает следующие.
#
#   GPU=0 bash run/component_rdo.sh                                          # все наши модели
#   GPU=0 CONFIGS="configs/qwen2.5-14b.yaml" bash run/component_rdo.sh       # свой список
#   tmux attach -t lb_comp_rdo                                               # посмотреть (выйти: Ctrl-b d)
#
# Env: GPU (обязателен), CONFIGS, SESSION (lb_comp_rdo).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GPU:?укажи GPU=<индекс>}"
CONFIGS="${CONFIGS:-configs/llama3-8b.yaml configs/qwen2.5-1.5b.yaml configs/qwen2.5-7b.yaml configs/qwen2.5-14b.yaml}"

if [[ "${INNER:-}" == 1 ]]; then
  export COMPONENT=rdo CUDA_VISIBLE_DEVICES="$GPU"
  for c in $CONFIGS; do
    [[ -f "$c" ]] || { echo "нет файла $c — пропускаю" >&2; continue; }
    for s in 03 04 05; do
      echo "================ $(basename "$c" .yaml): шаг $s (COMPONENT=rdo) — $(date '+%F %T') — GPU $GPU ================"
      CONFIG="$c" bash run.sh "$s" || { echo "!!! $c: шаг $s упал — модель пропущена" >&2; break; }
    done
  done
  echo "================ готово — $(date '+%F %T') ================"
  exit 0
fi

SESSION="${SESSION:-lb_comp_rdo}"
LOG="logs/component_rdo_$(date +%Y%m%d-%H%M%S).log"
mkdir -p logs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "сессия $SESSION уже есть — tmux attach -t $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "INNER=1 GPU=$GPU CONFIGS='$CONFIGS' bash run/component_rdo.sh 2>&1 | tee $LOG; echo '[конец, нажми Enter]'; read"
echo "сессия: $SESSION  GPU: $GPU  конфиги: $CONFIGS  лог: $LOG"
