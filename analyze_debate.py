"""Análise do persona_debate (debate.py): topologias, rodadas, custo e comparação com SC e MedAgents.

  .\.venv\Scripts\python.exe analyze_debate.py

Lê outputs/MedQA/<pasta>/*-persona_debate*.jsonl. A pasta 'hetero' guarda os arranjos com
modelos alternados entre os agentes. Grava analysis/debate_results.json.
"""
import glob
import json
import os
import random
from collections import Counter

import analyze  # mesmos preços, Wilson e McNemar

ROOT = './outputs/MedQA'
TOPO_LABEL = {'full': 'Completa', 'ring': 'Anel', 'ring_conf': 'Anel + Conformist'}


def load_debates():
    out = {}
    for path in sorted(glob.glob(os.path.join(ROOT, '*', '*-persona_debate*.jsonl'))):
        recs = {}
        for line in open(path, encoding='utf-8'):
            if line.strip():
                r = json.loads(line)
                recs[r['qid']] = r
        name = os.path.basename(path)[:-len('.jsonl')]
        arr = name.split('-persona_debate', 1)
        tag = arr[1].lstrip('-')
        key = tag.replace('hetero-', 'hetero: ') if tag else arr[0]
        out[key] = recs
    return out


def usd(usage_by_model):
    total = 0.0
    for m, u in usage_by_model.items():
        p = analyze.PRICES[m]
        total += (u.get('prompt_tokens', 0) * p[0] + u.get('completion_tokens', 0) * p[1]) / 1e6
    return total


def add(*parts):
    tot = {}
    for part in parts:
        for m, u in part.items():
            t = tot.setdefault(m, {})
            for k, v in u.items():
                t[k] = t.get(k, 0) + v
    return tot


def variants(recs):
    """chave -> {qid: resposta}, custo médio em US$ e chamadas médias por questão."""
    any_r = next(iter(recs.values()))
    T = any_r['debate_rounds']
    topos = list(any_r['debates'])
    out = {}
    out['vote'] = {
        'label': f"Voto dos {any_r['n_agents']} especialistas, sem debate",
        'answers': {q: r['vote_answer'] for q, r in recs.items()},
        'usd': sum(usd(r['usage_by_part']['shared']) for r in recs.values()) / len(recs),
        'calls': sum(sum(u['calls'] for u in r['usage_by_part']['shared'].values()) for r in recs.values()) / len(recs),
    }
    for topo in topos:
        for t in range(1, T + 1):
            key = f'{topo}@{t}'
            calls = []
            for r in recs.values():
                shared = sum(u['calls'] for u in r['usage_by_part']['shared'].values())
                calls.append(shared + sum(r['debates'][topo]['calls_per_round'][:t]))
            out[key] = {
                'label': f"{TOPO_LABEL[topo]}, {t} rodada{'s' if t > 1 else ''} de debate",
                'answers': {q: r['debates'][topo]['final_by_round'][t] for q, r in recs.items()},
                # custo em US$ só é exato para a última rodada (tokens não são separados por rodada)
                'usd': (sum(usd(add(r['usage_by_part']['shared'], r['usage_by_part'][topo])) for r in recs.values())
                        / len(recs)) if t == T else None,
                'calls': sum(calls) / len(calls),
            }
    return out, topos, T


def mean_agent_acc(recs, topo):
    """Acurácia média de um agente em cada rodada (Choi et al.: debate ~ martingale)."""
    T = next(iter(recs.values()))['debate_rounds']
    res = []
    for t in range(T + 1):
        hits = tot = 0
        for r in recs.values():
            for a in r['debates'][topo]['answers'][t]:
                hits += a == r['gold_answer']; tot += 1
        res.append(hits / tot)
    return res


def references(arr_key, qids):
    """SC e MedAgents nas mesmas questões, do(s) modelo(s) usado(s) no arranjo."""
    models = ['gpt-4o-mini', 'gpt-5.4-nano'] if arr_key.startswith('hetero') else [arr_key]
    runs = analyze.load_runs(ROOT, models)
    ref = {}
    for m in models:
        for k in ('base_sc', 'syn_verif', 'base_cot-cot1000'):
            if k in runs.get(m, {}) and all(q in runs[m][k] for q in qids):
                ref[f'{analyze.label(k)} ({m})'] = ({q: runs[m][k][q]['pred_answer'] for q in qids},
                                                     analyze.usd_per_question({q: runs[m][k][q] for q in qids}))
    return ref


def main():
    data = load_debates()
    results = {}
    for arr, recs in data.items():
        qids = sorted(recs)
        gold = {q: recs[q]['gold_answer'] for q in qids}
        n = len(qids)
        V, topos, T = variants(recs)
        refs = references(arr, qids)
        print(f"\n## {arr} — {n} questões\n")
        print('| Variação | Acurácia | IC 95% | Chamadas/q | US$/q | vs voto (consertou/estragou) |')
        print('|---|---|---|---|---|---|')
        rows = []
        vote_ok = {q: V['vote']['answers'][q] == gold[q] for q in qids}
        for key, v in V.items():
            ok = {q: v['answers'][q] == gold[q] for q in qids}
            k = sum(ok.values()); lo, hi = analyze.wilson(k, n)
            fixed = sum(ok[q] and not vote_ok[q] for q in qids); broke = sum(vote_ok[q] and not ok[q] for q in qids)
            usd_txt = f"{v['usd']:.4f}" if v['usd'] is not None else '—'
            print(f"| {v['label']} | {k / n:.1%} | {lo:.1%}–{hi:.1%} | {v['calls']:.1f} | {usd_txt} | "
                  f"{'' if key == 'vote' else f'{fixed}/{broke}'} |")
            row = {'key': key, 'label': v['label'], 'correct': k, 'n': n, 'acc': k / n, 'ci': [lo, hi],
                   'calls': v['calls'], 'usd': v['usd'], 'fixed_vs_vote': fixed, 'broke_vs_vote': broke, 'vs': {}}
            for rname, (ans, rusd) in refs.items():
                rok = {q: ans[q] == gold[q] for q in qids}
                a, b = sum(ok[q] and not rok[q] for q in qids), sum(rok[q] and not ok[q] for q in qids)
                row['vs'][rname] = {'only_this': a, 'only_ref': b, 'p': analyze.mcnemar_exact(a, b)}
            rows.append(row)
        print('\nReferências nas mesmas questões: ' + '; '.join(
            f"{name} {sum(ans[q] == gold[q] for q in qids) / n:.1%}"
            f"{f' (US$ {u:.4f}/q)' if u is not None else ''}" for name, (ans, u) in refs.items()))
        for row in rows:
            if row['vs']:
                print(f"  {row['label']}: " + '; '.join(
                    f"vs {k}: +{v['only_this']}/-{v['only_ref']} p={v['p']:.3g}" for k, v in row['vs'].items()))
        traj = {topo: mean_agent_acc(recs, topo) for topo in topos}
        print('\nAcurácia média de um agente por rodada (0 = antes do debate): ' + '; '.join(
            f"{TOPO_LABEL[t]}: " + ' → '.join(f'{x:.1%}' for x in v) for t, v in traj.items()))
        unanimous = sum(len(set(a for a in r['debates'][topos[0]]['answers'][0] if a)) == 1 for r in recs.values())
        print(f"Rodada 0 unânime em {unanimous} de {n} questões.")
        results[arr] = {'n': n, 'rows': rows, 'agent_acc_by_round': traj, 'round0_unanimous': unanimous,
                        'references': {k: {'acc': sum(a[q] == gold[q] for q in qids) / n, 'usd': u}
                                       for k, (a, u) in refs.items()}}
    os.makedirs('analysis', exist_ok=True)
    json.dump(results, open('analysis/debate_results.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('\nResultados gravados em analysis/debate_results.json')


if __name__ == '__main__':
    main()
