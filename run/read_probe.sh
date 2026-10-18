#!/usr/bin/env bash
# Шаг 06 (read против write по слоям) для нескольких моделей подряд на ОДНОЙ GPU, в отдельной
# tmux-сессии; лог в logs/. Только forward, ~5–15 мин на модель. Нужны шаги 00–02 и 03 с COMPONENT=rdo
# (если component_rdo нет — 03 досчитывается здесь же, это CPU). Падение одной модели не останавливает следующие.
#
#   GPU=0 bash run/read_probe.sh                                         # все семь моделей
#   GPU=0 CONFIGS="configs/llama3.1-8b.yaml" bash run/read_probe.sh      # свой список
#   tmux attach -t lb_read_probe                                         # посмотреть (выйти: Ctrl-b d)
#
# Env: GPU (обязателен), CONFIGS, SESSION (lb_read_probe).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GPU:?укажи GPU=<индекс>}"
CONFIGS="${CONFIGS:-configs/llama3.2-1b.yaml configs/llama3.2-3b.yaml configs/llama3-8b.yaml configs/llama3.1-8b.yaml configs/qwen2.5-1.5b.yaml configs/qwen2.5-7b.yaml configs/qwen2.5-14b.yaml}"

if [[ "${INNER:-}" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
  for c in $CONFIGS; do
    [[ -f "$c" ]] || { echo "нет файла $c — пропускаю" >&2; continue; }
    alias_="$(sed -n 's/^model: *\([^ #]*\).*/\1/p' "$c" | sed 's#.*/##')"
    if [[ ! -f "results/$alias_/component_rdo/component.pt" ]]; then
      echo "================ $(basename "$c" .yaml): шаг 03 (COMPONENT=rdo) — $(date '+%F %T') ================"
      COMPONENT=rdo CONFIG="$c" bash run.sh 03 || { echo "!!! $c: шаг 03 упал — модель пропущена" >&2; continue; }
    fi
    echo "================ $(basename "$c" .yaml): шаг 06 — $(date '+%F %T') — GPU $GPU ================"
    CONFIG="$c" bash run.sh 06 || echo "!!! $c: шаг 06 упал" >&2
  done
  echo "================ готово — $(date '+%F %T') ================"
  exit 0
fi

SESSION="${SESSION:-lb_read_probe}"
LOG="logs/read_probe_$(date +%Y%m%d-%H%M%S).log"
mkdir -p logs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "сессия $SESSION уже есть — tmux attach -t $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "INNER=1 GPU=$GPU CONFIGS='$CONFIGS' bash run/read_probe.sh 2>&1 | tee $LOG; echo '[конец, нажми Enter]'; read"
echo "сессия: $SESSION  GPU: $GPU  конфиги: $CONFIGS  лог: $LOG"
