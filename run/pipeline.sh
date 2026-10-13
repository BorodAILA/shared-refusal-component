#!/usr/bin/env bash
# Весь пайплайн 00 → 05 для одного конфига в отдельной tmux-сессии; лог в logs/.
# Обёртка над launch/run_all.sh: передаёт CONFIG в tmux явно (launch/start.sh его не пробрасывает).
#
#   GPU=0 CONFIG=configs/qwen2.5-7b.yaml bash run/pipeline.sh            # старт
#   GPU=0 CONFIG=configs/qwen2.5-7b.yaml START=02 bash run/pipeline.sh   # продолжить с шага 02
#   tmux attach -t lb_src_qwen2_5-7b                                     # посмотреть (выйти: Ctrl-b d)
#   tail -f logs/qwen2.5-7b_<время>.log                                  # или по логу
#
# Env: GPU (обязателен), CONFIG (обязателен), START (00), SESSION (lb_src_<имя конфига>).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GPU:?укажи GPU=<индекс>}"
: "${CONFIG:?укажи CONFIG=configs/<модель>.yaml}"
[[ -f "$CONFIG" ]] || { echo "нет файла $CONFIG" >&2; exit 1; }
NAME="$(basename "$CONFIG" .yaml)"
SESSION="${SESSION:-lb_src_${NAME//./_}}"   # tmux не допускает точку в имени сессии
START="${START:-00}"
LOG="logs/${NAME}_$(date +%Y%m%d-%H%M%S).log"
mkdir -p logs

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "сессия $SESSION уже есть — tmux attach -t $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "CONFIG=$CONFIG GPU=$GPU START=$START bash launch/run_all.sh 2>&1 | tee $LOG; echo '[конец, нажми Enter]'; read"
echo "сессия: $SESSION  GPU: $GPU  config: $CONFIG  лог: $LOG"
