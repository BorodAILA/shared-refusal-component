#!/usr/bin/env bash
# Запускает run_all.sh в отдельной tmux-сессии; лог дублируется в файл.
#
#   GPU=1 bash launch/start.sh            # старт
#   tmux attach -t fz-src                 # посмотреть (выйти: Ctrl-b d)
#   tail -f launch/logs/<время>.log       # или по логу
#
# Env: GPU (обязателен), START (00), SESSION (fz-src).
set -euo pipefail
cd "$(dirname "$0")/.."

: "${GPU:?укажи GPU=<индекс>}"
SESSION="${SESSION:-fz-src}"
START="${START:-00}"
LOG="launch/logs/$(date +%Y%m%d-%H%M%S).log"
mkdir -p launch/logs

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "сессия $SESSION уже есть — tmux attach -t $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "GPU=$GPU START=$START bash launch/run_all.sh 2>&1 | tee $LOG; echo '[конец, нажми Enter]'; read"
echo "сессия: $SESSION  GPU: $GPU  лог: $LOG"
