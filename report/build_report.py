"""Gera report/relatorio.html a partir das saídas em outputs/MedQA/.

  .\.venv\Scripts\python.exe report\build_report.py

Os números e gráficos vêm de analyze.py (mesmas funções, mesmos critérios). Os textos de
interpretação ficam em report/narrativa.json, escrito à mão depois de ver os resultados.
Os casos citados são lidos dos registros originais pelos ids listados na narrativa.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import analyze  # noqa: E402

MODEL_ORDER = ['gpt-4o-mini', 'gpt-5.4-nano']
METHOD_ORDER = ['base_direct', 'base_cot', 'base_cot-cot1000', 'base_sc', 'syn_verif']


def clean(x):
    """NaN não existe em JSON; vira null."""
    if isinstance(x, float) and x != x:
        return None
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def model_block(methods):
    acc = {a['key']: a for a in analyze.accuracy_table(methods)}
    out = {
        'accuracy': [acc[k] for k in METHOD_ORDER if k in acc],
        'mcnemar': analyze.mcnemar_vs_ma(methods),
        'by_rounds': analyze.by_rounds(methods),
        'sc_vs_rounds': analyze.sc_agreement_vs_rounds(methods),
        'cascade': analyze.cascade(methods),
        'sc_curve': analyze.sc_curve(methods['base_sc']) if 'base_sc' in methods else [],
    }
    if analyze.MA in methods:
        out['round_dynamics'] = analyze.round_dynamics(methods[analyze.MA])
        out['votes'] = analyze.votes_detail(methods[analyze.MA])
    return out


def case_record(runs, model, qid):
    """Dados de um caso: votos por rodada, resposta de cada versão do relatório e dos outros métodos."""
    m = runs[model]
    ma = m[analyze.MA][qid]
    return {
        'model': model, 'qid': qid, 'step': ma.get('step', ''),
        'options': ma['options'], 'gold': ma['gold_answer'], 'medagents': ma['pred_answer'],
        'round_answers': ma.get('round_answers', []),
        'votes': [{d: v for d, v in vh.items()} for vh in ma['vote_history']],
        'revision_history': ma.get('revision_history', []),
        'others': {k: m[k][qid]['pred_answer'] for k in METHOD_ORDER if k in m and k != analyze.MA and qid in m[k]},
        'sc_answers': m['base_sc'][qid]['sc_answers'] if 'base_sc' in m and qid in m['base_sc'] else [],
    }


def main():
    runs = analyze.load_runs('./outputs/MedQA')
    narr_path = os.path.join(HERE, 'narrativa.json')
    narr = json.load(open(narr_path, encoding='utf-8')) if os.path.exists(narr_path) else {}
    data = {
        'models': {m: model_block(runs[m]) for m in MODEL_ORDER if m in runs},
        'labels': {k: analyze.label(k) for k in METHOD_ORDER},
        'cases': [case_record(runs, c['model'], c['qid']) for c in narr.get('cases', [])],
    }
    html = open(os.path.join(HERE, 'template.html'), encoding='utf-8').read()
    html = html.replace('/*__DATA__*/null', json.dumps(clean(data), ensure_ascii=False))
    html = html.replace('/*__TEXT__*/null', json.dumps(narr, ensure_ascii=False))
    out = os.path.join(HERE, 'relatorio.html')
    open(out, 'w', encoding='utf-8').write(html)
    print(f'Gerado {out} ({len(html) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
