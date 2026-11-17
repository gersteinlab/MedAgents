"""Ablação da arquitetura do MedVote: as duas etapas são necessárias?

  python analyze_ablacao.py --split test

Compara, nas mesmas questões, o MedVote com a configuração fixada na validação contra:
  - etapa 1 sozinha (só o voto);
  - etapa única (medvote.py rag): busca e resposta direta, sem voto e sem candidatas, com 1, 3, 5 e 7 amostras;
  - duas etapas sem candidatas: nas questões divididas, a resposta da etapa única no lugar do diferencial;
  - MedVote sem portão: etapa 2 também nas questões não unânimes com concordância >= 90% (diff --tag=-semportao).
Grava analysis/ablacao_<split>.json.
"""
import argparse
import json
import os
import random
import sys

import analyze
import analyze_medvote as am

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report'))
import medvote_test  # noqa: E402

CONFIG = {'nano': 11, 'mini': 10, 'threshold': 0.9, 'variant': 'books_cases', 'voters': (am.NANO,)}
DRAWS = 50


def load(split, name):
    path = os.path.join(am.OUT_DIR, f'{split}-{name}.jsonl')
    return {r['qid']: r for r in map(json.loads, open(path, encoding='utf-8'))} if os.path.exists(path) else {}


def compare(ok, ref):
    x = sum(ok[q] and not ref[q] for q in ref)
    y = sum(ref[q] and not ok[q] for q in ref)
    return x, y, analyze.mcnemar_exact(x, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='test')
    args = p.parse_args()
    votes, diffs = load(args.split, 'vote'), load(args.split, 'diff')
    rag, nogate = load(args.split, 'rag'), load(args.split, 'diff-semportao')
    qids = sorted(q for q in votes if q in rag)
    N = len(qids)
    gold = {q: votes[q]['gold'] for q in qids}
    votes = {q: votes[q] for q in qids}

    dec, cost, esc = medvote_test.fixed_config(votes, diffs, CONFIG)
    mv_ok = {q: dec[q] == gold[q] for q in qids}
    rows = []

    def add(name, ok, usd, note=''):
        x, y, p_ = compare(ok, mv_ok)
        k = sum(ok.values()) if isinstance(ok, dict) else ok
        d = (x - y) / N
        se = ((x + y) - (x - y) ** 2 / N) ** 0.5 / N  # diferença pareada de proporções
        rows.append({'name': name, 'k': k, 'n': N, 'acc': k / N, 'usd': usd, 'vs_mv_gain': x, 'vs_mv_loss': y,
                     'p': p_, 'diff': d, 'diff_ci': [d - 1.96 * se, d + 1.96 * se], 'note': note})

    add('MedVote (duas etapas, configuração fixada)', mv_ok, sum(cost.values()) / N)

    s1 = {q: am.majority(votes[q]['samples'][am.NANO][:11] + votes[q]['samples'][am.MINI][:10]) for q in qids}
    add('Só a etapa 1 (voto, 11 nano + 10 4o-mini)', {q: s1[q] == gold[q] for q in qids},
        sum(am.stage1_cost(votes[q], 11, 10) for q in qids) / N)

    # etapa única com n amostras: primeiras n (determinístico) e média de sorteios para n < 7
    def rag_cost(q, n):
        r = rag[q]
        u = r['usage'][am.NANO]
        tot = r['per_model'] if 'per_model' in r else len(r['answers'][am.NANO])
        part = {am.NANO: {'prompt_tokens': u['prompt_tokens'],  # um prompt por chamada (n <= 8 numa chamada)
                          'completion_tokens': u['completion_tokens'] * n / tot}}
        return am.usd(r['planning_usage']) + am.usd(part)

    single = {}
    for n in (1, 3, 5, 7):
        ans = {q: am.majority(rag[q]['answers'][am.NANO][:n]) for q in qids}
        ok = {q: ans[q] == gold[q] for q in qids}
        accs = []
        if n < 7:
            for d in range(DRAWS):
                rng = random.Random(d)
                accs.append(sum(am.majority(rng.sample(rag[q]['answers'][am.NANO], n)) == gold[q] for q in qids) / N)
        note = f'média de {DRAWS} sorteios: {sum(accs) / len(accs):.1%}' if accs else ''
        add(f'Etapa única, {n} amostra{"s" if n > 1 else ""} do nano', ok, sum(rag_cost(q, n) for q in qids) / N, note)
        single[n] = ans

    # duas etapas, mas a etapa 2 sem candidatas: resposta da etapa única (3 amostras) nas questões escaladas
    hyb = {q: (single[3][q] if q in esc else s1[q]) for q in qids}
    add('Duas etapas, etapa 2 sem candidatas (etapa única nas divididas)', {q: hyb[q] == gold[q] for q in qids},
        sum(am.stage1_cost(votes[q], 11, 10) + (rag_cost(q, 3) if q in esc else 0) for q in qids) / N)

    # sem portão: etapa 2 também nas não unânimes com concordância >= 90%
    ng_dec, ng_cost = dict(dec), dict(cost)
    extra = 0
    for q in qids:
        if q in esc or q not in nogate:
            continue
        d = nogate[q]['variants']['books_cases']
        ng_dec[q] = am.majority(d['answers'][am.NANO])
        ng_cost[q] += am.usd(nogate[q]['planning_usage']) + am.usd({am.NANO: d['usage'][am.NANO]})
        extra += 1
    add('MedVote sem portão (etapa 2 em todas as não unânimes)', {q: ng_dec[q] == gold[q] for q in qids},
        sum(ng_cost.values()) / N, f'{len(esc) + extra} questões na etapa 2')

    # onde cada arquitetura ganha: questões por faixa de concordância da etapa 1
    bands = {'unânimes': [], 'concordância >= 90%, não unânimes': [], 'divididas (etapa 2)': []}
    for q in qids:
        sm = votes[q]['samples'][am.NANO][:11] + votes[q]['samples'][am.MINI][:10]
        if q in esc:
            bands['divididas (etapa 2)'].append(q)
        elif len({a for a in sm if a}) <= 1:
            bands['unânimes'].append(q)
        else:
            bands['concordância >= 90%, não unânimes'].append(q)
    band_rows = []
    for name, qs in bands.items():
        if not qs:
            continue
        band_rows.append({'band': name, 'n': len(qs),
                          'medvote': sum(mv_ok[q] for q in qs), 'stage1': sum(s1[q] == gold[q] for q in qs),
                          'single7': sum(single[7][q] == gold[q] for q in qs),
                          'nogate': sum(ng_dec[q] == gold[q] for q in qs)})

    res = {'split': args.split, 'n': N, 'escalated': len(esc), 'nogate_extra': extra, 'rows': rows, 'bands': band_rows,
           'rag_dropped': sum(len(r['dropped_duplicates']) for q, r in rag.items() if q in votes),
           'total_usd': sum(am.usd(r['planning_usage']) + am.usd(r['usage']) for r in rag.values()) +
           sum(am.usd(r['planning_usage']) + am.usd(r['variants']['books_cases']['usage']) for r in nogate.values())}

    print(f'## Ablação, {args.split}: {N} questões ({len(esc)} na etapa 2 do MedVote)\n')
    print('| Arquitetura | Acertos | Acurácia | US$ / 1.000 q | vs MedVote (+/−) | p | Nota |\n|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['name']} | {r['k']} | {r['acc']:.1%} | {r['usd'] * 1000:.2f} | +{r['vs_mv_gain']} / −{r['vs_mv_loss']} | "
              f"{r['p']:.3f} | {r['note']} |")
    print('\n| Faixa | Questões | MedVote | Só etapa 1 | Etapa única (7) | Sem portão |\n|---|---|---|---|---|---|')
    for b in band_rows:
        print(f"| {b['band']} | {b['n']} | {b['medvote']} | {b['stage1']} | {b['single7']} | {b['nogate']} |")
    print(f"\nCusto desta ablação: US$ {res['total_usd']:.2f}")
    os.makedirs('analysis', exist_ok=True)
    json.dump(res, open(f'analysis/ablacao_{args.split}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'Gravado analysis/ablacao_{args.split}.json')


if __name__ == '__main__':
    main()
