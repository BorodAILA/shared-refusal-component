#!/usr/bin/env bash
# Для каждого конфига по очереди: скачать веса его модели → прогнать пайплайн 00 → 05 на ОДНОЙ GPU.
# Всё в одной tmux-сессии; общий лог logs/fetch_and_run_<время>.log, у каждого прогона ещё свой
# logs/<имя конфига>_<время>.log. Падение одной модели не останавливает следующие.
#
#   GPU=0 bash run/fetch_and_run.sh                                         # Qwen2.5-1.5B, затем 14B
#   GPU=0 CONFIGS="configs/qwen2.5-14b.yaml" bash run/fetch_and_run.sh      # свой список
#   tmux attach -t lb_fetch_run                                             # посмотреть (выйти: Ctrl-b d)
#
# Сколько скачалось: ls -la ~/.cache/huggingface/hub/models--<орг>--<модель>/blobs/*.incomplete
#
# Env: GPU (обязателен), CONFIGS, HF_HOME (~/.cache/huggingface), SESSION (lb_fetch_run).
set -euo pipefail
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/.."

: "${GPU:?укажи GPU=<индекс>}"
CONFIGS="${CONFIGS:-configs/qwen2.5-1.5b.yaml configs/qwen2.5-14b.yaml}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"   # xet на этом диске зависает; обычный HTTP докачивает .incomplete

if [[ "${INNER:-}" == 1 ]]; then
  for c in $CONFIGS; do
    [[ -f "$c" ]] || { echo "нет файла $c — пропускаю" >&2; continue; }
    model="$(sed -n 's/^model: *\([^ #]*\).*/\1/p' "$c")"
    name="$(basename "$c" .yaml)"
    echo "================ $name: закачка $model — $(date '+%F %T') ================"
    if ! uv run python -c "
import sys
from huggingface_hub import snapshot_download
print('готово:', snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'tokenizer*']))" "$model"; then
      echo "!!! $name: закачка упала — прогон пропущен" >&2; continue
    fi
    df -h "$HF_HOME" | tail -1
    echo "================ $name: прогон на GPU $GPU — $(date '+%F %T') ================"
    CONFIG="$c" GPU="$GPU" bash launch/run_all.sh 2>&1 | tee "logs/${name}_$(date +%Y%m%d-%H%M%S).log" \
      || echo "!!! $name: прогон упал — смотри его лог; продолжить: START=<шаг> GPU=$GPU CONFIG=$c bash run/pipeline.sh" >&2
  done
  echo "================ всё — $(date '+%F %T') ================"
  exit 0
fi

SESSION="${SESSION:-lb_fetch_run}"
LOG="logs/fetch_and_run_$(date +%Y%m%d-%H%M%S).log"
mkdir -p logs
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "сессия $SESSION уже есть — tmux attach -t $SESSION" >&2
  exit 1
fi
tmux new-session -d -s "$SESSION" -c "$PWD" \
  "INNER=1 GPU=$GPU CONFIGS='$CONFIGS' HF_HOME='$HF_HOME' HF_HUB_DISABLE_XET=$HF_HUB_DISABLE_XET bash run/fetch_and_run.sh 2>&1 | tee $LOG; echo '[конец, нажми Enter]'; read"
echo "сессия: $SESSION  GPU: $GPU  конфиги: $CONFIGS  лог: $LOG"
