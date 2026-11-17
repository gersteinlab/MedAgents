#!/bin/bash
# Confirmação nas 972 questões da validação fora do piloto (nunca usadas), com as configurações fixadas antes:
#   MedVote atual, MedVote com fichas (2 candidatas), etapa única com 3 amostras e o voto sozinho.
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8 MEDVOTE_OUT=outputs/medvote/confirmacao
PY=./.venv/Scripts/python.exe
NANO=gpt-5.4-nano
step() { name=$1; shift; $PY medvote.py "$@" > logs/conf-$name.log 2>&1; echo "=== $(date +%H:%M:%S) $name exit $?"; grep -a "falharam\|Traceback\|fazer\|faltam" logs/conf-$name.log | tail -2; }
step vote vote --split validation --rest_of 300 --models $NANO,gpt-4o-mini --samples 11,10 --workers 16
step vote2 vote --split validation --rest_of 300 --models $NANO,gpt-4o-mini --samples 11,10 --workers 4
step diff diff --split validation --models $NANO --variants books_cases --workers 12
step ddx ddx --split validation --models $NANO --tag=-2c --workers 12
step rag rag --split validation --rest_of 300 --models $NANO --per_model 3 --workers 16
for s in diff ddx rag; do step ${s}2 $( [ $s = rag ] && echo "rag --split validation --rest_of 300 --models $NANO --per_model 3" || ([ $s = diff ] && echo "diff --split validation --models $NANO --variants books_cases" || echo "ddx --split validation --models $NANO --tag=-2c") ) --workers 4; done
$PY analyze_ddx.py --name confirmacao --third none > logs/confirmacao_ddx.md 2>&1; echo "=== análise exit $?"
echo "=== CONFIRMAÇÃO PRONTA"
