#!/usr/bin/env bash
# Скачивает веса моделей в общий HF-кэш в отдельной tmux-сессии; лог в logs/. GPU не нужен.
# Берутся только config/tokenizer/safetensors — без .bin/.pth/оригинальных чекпоинтов.
# Уже скачанные файлы пропускаются, так что прерванную закачку можно просто перезапустить.
#
#   bash run/download.sh                                          # Qwen2.5-1.5B и 14B-Instruct
#   MODELS="Qwen/Qwen2.5-0.5B-Instruct" bash run/download.sh      # свой список
#   tmux attach -t lb_dl                                          # посмотреть (выйти: Ctrl-b d)
#
# Env: MODELS (через пробел), HF_HOME (~/.cache/huggingface), SESSION (lb_dl).
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."

MODELS="${MODELS:-Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-14B-Instruct}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"   # xet на этом диске зависает; обычный HTTP докачивает .incomplete

if [[ "${INNER:-}" == 1 ]]; then
  df -h "$HF_HOME" | tail -1
  for m in $MODELS; do
    echo "================ $m — $(date '+%F %T') ================"
    uv run python -c "
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*'])
print('готово:', p)" "$m"
  done
  df -h "$HF_HOME" | tail -1
  echo "================ всё скачано — $(date '+%F %T') ================"
  exit 0
fi

SESSION="${SESSION:-lb_dl}"
LOG="logs/download_$(date +%Y%m%d-%H%M%S).log"
mkdir -p logs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "сессия $SESSION уже есть — tmux attach -t $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "INNER=1 MODELS='$MODELS' HF_HOME='$HF_HOME' HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET bash run/download.sh 2>&1 | tee $LOG; echo '[конец, нажми Enter]'; read"
echo "сессия: $SESSION  модели: $MODELS  лог: $LOG"
