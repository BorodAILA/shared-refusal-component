#!/usr/bin/env bash
# Весь пайплайн 00 → 05 на Llama-3-8B-Instruct в отдельной tmux-сессии; лог в logs/.
# Обёртка над launch/run_all.sh: передаёт CONFIG в tmux явно (launch/start.sh его не пробрасывает).
#
#   GPU=1 bash run/llama3-8b.sh               # старт
#   GPU=1 START=02 bash run/llama3-8b.sh      # продолжить с шага 02
#   tmux attach -t lb_src_llama3              # посмотреть (выйти: Ctrl-b d)
#   tail -f logs/llama3-8b_<время>.log        # или по логу
#
# Env: GPU (обязателен), START (00), CONFIG (configs/llama3-8b.yaml), SESSION (lb_src_llama3).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GPU:?укажи GPU=<индекс>}"
CONFIG="${CONFIG:-configs/llama3-8b.yaml}"
SESSION="${SESSION:-lb_src_llama3}"
START="${START:-00}"
LOG="logs/llama3-8b_$(date +%Y%m%d-%H%M%S).log"
mkdir -p logs

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "сессия $SESSION уже есть — tmux attach -t $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "CONFIG=$CONFIG GPU=$GPU START=$START bash launch/run_all.sh 2>&1 | tee $LOG; echo '[конец, нажми Enter]'; read"
echo "сессия: $SESSION  GPU: $GPU  config: $CONFIG  лог: $LOG"
