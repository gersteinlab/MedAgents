"""Análise do MedVote (medvote.py). Tudo é simulado a partir das amostras gravadas.

  python analyze_medvote.py [--split validation]

Grava analysis/medvote_<split>.json. As escolhas de configuração devem ser feitas na validação;
o teste é rodado uma vez só, com a configuração escolhida.
"""
import argparse
import json
import os
import random
from collections import Counter

import analyze  # preços, Wilson, McNemar

OUT_DIR = os.environ.get('MEDVOTE_OUT', os.path.join('outputs', 'medvote'))
NANO, MINI = 'gpt-5.4-nano', 'gpt-4o-mini'
MIXES = [(21, 0), (0, 21), (14, 7), (11, 10), (7, 14), (21, 21)]  # (nano, 4o-mini)
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
DRAWS = 50


def majority(answers):
    valid = [a for a in answers if a]
    if not valid:
        return ''
    c = Counter(valid)
    best = max(c.values())
    return next(a for a in valid if c[a] == best)


def agreement(answers):
    valid = [a for a in answers if a]
    return Counter(valid).most_common(1)[0][1] / len(valid) if valid else 0.0


def usd(usage):
    return sum((u.get('prompt_tokens', 0) * analyze.PRICES[m][0] + u.get('completion_tokens', 0) * analyze.PRICES[m][1]) / 1e6
               for m, u in usage.items())


def mix_label(a, b):
    if b == 0:
        return f'{a} amostras do nano'
    if a == 0:
        return f'{b} amostras do 4o-mini'
    return f'{a} do nano + {b} do 4o-mini'


def draw(rec, a, b, rng):
    s = rec['samples']
    return rng.sample(s[NANO], a) + rng.sample(s[MINI], b)


def stage1_cost(rec, a, b):
    """Custo proporcional às amostras usadas (o prompt é cobrado uma vez por lote de 8; aproximação)."""
    u = rec['usage']
    full = {NANO: len(rec['samples'][NANO]), MINI: len(rec['samples'][MINI])}
    part = {NANO: {k: v * a / full[NANO] for k, v in u[NANO].items()},
            MINI: {k: v * b / full[MINI] for k, v in u[MINI].items()}}
    return usd(part)


def mcnemar(ok_a, ok_b):
    x = sum(ok_a[q] and not ok_b[q] for q in ok_a)
    y = sum(ok_b[q] and not ok_a[q] for q in ok_a)
    return x, y, analyze.mcnemar_exact(x, y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='validation')
    args = p.parse_args()
    votes = {r['qid']: r for r in map(json.loads, open(os.path.join(OUT_DIR, f'{args.split}-vote.jsonl'), encoding='utf-8'))}
    dpath = os.path.join(OUT_DIR, f'{args.split}-diff.jsonl')
    diffs = {r['qid']: r for r in map(json.loads, open(dpath, encoding='utf-8'))} if os.path.exists(dpath) else {}
    qids = sorted(votes)
    N = len(qids)
    gold = {q: votes[q]['gold'] for q in qids}
    res = {'split': args.split, 'n': N}

    # ---------------- etapa 1: misturas de modelos
    print(f'## {args.split}: {N} questões\n\n### Etapa 1: voto\n')
    print('| Amostras | Acurácia | IC 95% | US$ / 1.000 q |\n|---|---|---|---|')
    s1 = []
    for a, b in MIXES:
        accs = []
        for d in range(DRAWS if 0 < a < 21 or 0 < b < 21 else 1):
            rng = random.Random(d)
            accs.append(sum(majority(draw(votes[q], a, b, rng)) == gold[q] for q in qids) / N)
        acc = sum(accs) / len(accs)
        cost = sum(stage1_cost(votes[q], a, b) for q in qids) / N
        lo, hi = analyze.wilson(round(acc * N), N)
        s1.append({'nano': a, 'mini': b, 'label': mix_label(a, b), 'acc': acc, 'ci': [lo, hi], 'usd': cost})
        print(f'| {mix_label(a, b)} | {acc:.1%} | {lo:.1%}–{hi:.1%} | {cost * 1000:.2f} |')
    res['stage1'] = s1

    # ---------------- etapa 2: variantes nas questões divididas
    if not diffs:
        json.dump(res, open(f'analysis/medvote_{args.split}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        return
    dq = sorted(diffs)
    pooled = {q: majority(votes[q]['samples'][NANO] + votes[q]['samples'][MINI]) for q in qids}
    cover = sum(gold[q] in diffs[q]['candidates'] for q in dq)
    print(f'\n### Etapa 2: {len(dq)} questões divididas (concordância das 42 amostras < 90%)\n')
    print(f'Gabarito entre as duas candidatas: {cover} de {len(dq)} ({cover / len(dq):.0%}). '
          f'Voto das 42 amostras nessas questões: {sum(pooled[q] == gold[q] for q in dq)} acertos.\n')
    print('| Variante | Quem vota | Acertos | vs voto da etapa 1 (consertou / estragou) | p | US$ / questão dividida |')
    print('|---|---|---|---|---|---|')
    base_ok = {q: pooled[q] == gold[q] for q in dq}
    s2 = []
    variants = list(next(iter(diffs.values()))['variants'])
    for v in variants:
        for who, models in [('3 nano + 3 4o-mini', (NANO, MINI)), ('3 nano', (NANO,)), ('3 4o-mini', (MINI,))]:
            dec = {q: majority([a for m in models for a in diffs[q]['variants'][v]['answers'][m]]) for q in dq}
            ok = {q: dec[q] == gold[q] for q in dq}
            x, y, p_ = mcnemar(ok, base_ok)
            cost = sum(usd(diffs[q]['planning_usage']) * (v != 'none') +
                       usd({m: diffs[q]['variants'][v]['usage'][m] for m in models}) for q in dq) / len(dq)
            s2.append({'variant': v, 'who': who, 'correct': sum(ok.values()), 'n': len(dq), 'fixed': x, 'broke': y,
                       'p': p_, 'usd': cost})
            print(f'| {v} | {who} | {sum(ok.values())} | +{x} / −{y} | {p_:.2f} | {cost:.4f} |')
    res['stage2'] = {'n': len(dq), 'gold_in_candidates': cover, 'pooled_correct': sum(base_ok.values()), 'rows': s2}

    # ---------------- sistema completo: etapa 1 (mistura) + etapa 2 abaixo do limiar
    print('\n### Sistema completo (etapa 1 + etapa 2 abaixo do limiar; votantes da etapa 2: 3 nano + 3 4o-mini)\n')
    print('| Etapa 1 | Limiar | Variante | Questões na etapa 2 | Acurácia | US$ / 1.000 q |\n|---|---|---|---|---|---|')
    full = []
    for a, b in MIXES:
        for v in variants:
            for t in THRESHOLDS:
                accs, esc = [], []
                for d in range(DRAWS if 0 < a < 21 or 0 < b < 21 else 1):
                    rng = random.Random(d)
                    hit = n_esc = 0
                    for q in qids:
                        sm = draw(votes[q], a, b, rng)
                        ans = majority(sm)
                        if agreement(sm) < t and q in diffs:
                            ans = majority([x for m in (NANO, MINI) for x in diffs[q]['variants'][v]['answers'][m]])
                            n_esc += 1
                        hit += ans == gold[q]
                    accs.append(hit / N)
                    esc.append(n_esc)
                n_e = sum(esc) / len(esc)
                c2 = sum(usd(diffs[q]['planning_usage']) * (v != 'none') + usd(diffs[q]['variants'][v]['usage'])
                         for q in dq) / len(dq)
                cost = next(r['usd'] for r in s1 if (r['nano'], r['mini']) == (a, b)) + c2 * n_e / N
                full.append({'nano': a, 'mini': b, 'variant': v, 'threshold': t, 'escalated': n_e,
                             'acc': sum(accs) / len(accs), 'usd': cost})
    for r in sorted(full, key=lambda r: -r['acc'])[:12]:
        print(f"| {mix_label(r['nano'], r['mini'])} | {r['threshold']:.0%} | {r['variant']} | {r['escalated']:.0f} | "
              f"{r['acc']:.1%} | {r['usd'] * 1000:.2f} |")
    res['full'] = full
    os.makedirs('analysis', exist_ok=True)
    json.dump(res, open(f'analysis/medvote_{args.split}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\nGravado analysis/medvote_{args.split}.json')


if __name__ == '__main__':
    main()
