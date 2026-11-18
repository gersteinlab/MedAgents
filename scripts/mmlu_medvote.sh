#!/bin/bash
# MedVote no MMLU médico (6 matérias, 4 alternativas), com as configurações fixadas no MedQA. Retomável.
#   bash scripts/mmlu_medvote.sh validation   # ensaio: 123 questões
#   bash scripts/mmlu_medvote.sh test         # 1.089 questões
# Sistemas: voto (11 nano + 10 4o-mini), MedVote atual, fichas sem e com o peso do voto, etapa única com 3 amostras.
cd "$(dirname "$0")/.."
SPLIT=${1:-validation}
export PYTHONIOENCODING=utf-8
PY=./.venv/Scripts/python.exe
NANO=gpt-5.4-nano
mkdir -p logs
step() { name=$1; shift; $PY medvote.py "$@" --dataset mmlu --split $SPLIT >> logs/mmlu-$SPLIT-$name.log 2>&1
         echo "=== $(date +%H:%M:%S) $name exit $?"; grep -a "falharam\|Traceback\|fazer\|faltam" logs/mmlu-$SPLIT-$name.log | tail -2; }
for pass in 1 2; do  # a segunda passada refaz só as questões que falharam
  step vote  vote --limit 0 --models $NANO,gpt-4o-mini --samples 11,10 --workers 16
  step diff  diff --models $NANO --variants books_cases --workers 12
  step cards ddx  --models $NANO --tag=-2c --workers 12
  step prior ddx  --models $NANO --prior --tag=-prior --workers 12
  step rag   rag  --limit 0 --models $NANO --per_model 3 --workers 16
done
$PY analyze_mmlu.py --split $SPLIT > logs/mmlu-$SPLIT.md 2>&1; echo "=== análise exit $?"
cat logs/mmlu-$SPLIT.md
