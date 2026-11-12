"""Resultados do MedVote no teste, com a configuração fixada na validação (usado por build_medvote.py)."""
import json
import os
import re

import analyze
import analyze_medvote as am

NORM = lambda s: re.sub(r'\W+', ' ', s).strip().lower()  # noqa: E731


def _load(split, name):
    path = os.path.join(am.OUT_DIR, f'{split}-{name}.jsonl')
    return {r['qid']: r for r in map(json.loads, open(path, encoding='utf-8'))} if os.path.exists(path) else {}


def fixed_config(votes, diffs, cfg):
    """Decisão e custo da configuração fixada, questão a questão (primeiras amostras, sem sorteio)."""
    dec, cost, esc = {}, {}, set()
    for q, v in votes.items():
        sm = v['samples'][am.NANO][:cfg['nano']] + v['samples'][am.MINI][:cfg['mini']]
        a = am.majority(sm)
        c = am.stage1_cost(v, cfg['nano'], cfg['mini'])
        if am.agreement(sm) < cfg['threshold'] and q in diffs:
            esc.add(q)
            d = diffs[q]['variants'][cfg['variant']]
            a = am.majority([x for m in cfg['voters'] for x in d['answers'][m]])
            c += am.usd(diffs[q]['planning_usage']) + am.usd({m: d['usage'][m] for m in cfg['voters']})
        dec[q], cost[q] = a, c
    return dec, cost, esc


def medagents_preds(votes):
    """Previsões das partes anteriores (MedAgents e Self-Consistency), casadas pelo texto da questão."""
    by_text = {NORM(v['question'] if 'question' in v else ''): q for q, v in votes.items()}
    out = {}
    for model in (am.NANO, am.MINI):
        for method, label in (('syn_verif', 'MedAgents'), ('base_sc', 'Self-Consistency (parte 2)')):
            path = os.path.join('outputs', 'MedQA', model, f'{model}-{method}.jsonl')
            preds = {}
            for r in map(json.loads, open(path, encoding='utf-8')):
                q = by_text.get(NORM(r['question']))
                if q is not None:
                    preds[q] = r['pred_answer']
            out[(label, model)] = preds
    return out


def results(cfg):
    votes, diffs = _load('test', 'vote'), _load('test', 'diff')
    if not votes:
        return None
    import retrieval
    texts = {q['qid']: q['question'] for q in retrieval.load_split('test')}
    for q in votes:
        votes[q].setdefault('question', texts[q])
    qids = sorted(votes)
    N = len(qids)
    gold = {q: votes[q]['gold'] for q in qids}
    dec, cost, esc = fixed_config(votes, diffs, cfg)
    ok = {q: dec[q] == gold[q] for q in qids}

    rows = []

    def add(label, model, pred, usd_per_q=None):
        k = sum(pred.get(q) == gold[q] for q in qids)
        other = {q: pred.get(q) == gold[q] for q in qids}
        x = sum(ok[q] and not other[q] for q in qids)
        y = sum(other[q] and not ok[q] for q in qids)
        rows.append({'label': label, 'model': model, 'k': k, 'n': N, 'fixed': x, 'broke': y,
                     'p': analyze.mcnemar_exact(x, y), 'usd': usd_per_q})

    sc = {m: {q: am.majority(votes[q]['samples'][m]) for q in qids} for m in (am.NANO, am.MINI)}
    add('Self-Consistency, 21 amostras', am.NANO, sc[am.NANO], sum(am.stage1_cost(votes[q], 21, 0) for q in qids) / N)
    add('Self-Consistency, 21 amostras', am.MINI, sc[am.MINI], sum(am.stage1_cost(votes[q], 0, 21) for q in qids) / N)
    s1 = {q: am.majority(votes[q]['samples'][am.NANO][:cfg['nano']] + votes[q]['samples'][am.MINI][:cfg['mini']])
          for q in qids}
    add('Etapa 1 sozinha, 11 + 10 amostras', 'misto', s1, sum(am.stage1_cost(votes[q], cfg['nano'], cfg['mini']) for q in qids) / N)
    prior = json.load(open(os.path.join('analysis', 'results.json'), encoding='utf-8'))
    prior_usd = {(a['label'], m): a['usd_per_question'] for m in prior for a in prior[m]['accuracy']}
    for (label, model), preds in medagents_preds(votes).items():
        add(label, model, preds, prior_usd.get(('MedAgents' if label == 'MedAgents' else 'Self-Consistency', model)))

    # quase-cópias no treino
    ci = retrieval.CaseIndex()
    near = {q for q in qids if ci.nearest_jaccard(texts[q]) >= 0.6}
    clean = [q for q in qids if q not in near]

    total = (sum(am.usd(r['usage']) for r in votes.values()) +
             sum(am.usd(r['planning_usage']) + sum(am.usd(v['usage']) for v in r['variants'].values())
                 for r in diffs.values()))
    dropped = sum(len(r.get('dropped_duplicates', r.get('dropped', [])) or []) for r in diffs.values())
    return {'n': N, 'k': sum(ok.values()), 'ci': analyze.wilson(sum(ok.values()), N), 'usd': sum(cost.values()) / N,
            'escalated': len(esc), 'rows': rows, 'near': len(near),
            'clean_k': sum(ok[q] for q in clean), 'clean_n': len(clean),
            'clean_sc': sum(sc[am.NANO][q] == gold[q] for q in clean),
            'total_usd': total, 'dropped': dropped, 'n_diff': len(diffs)}
