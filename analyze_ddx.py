"""MedVote com fichas de diferenças (medvote.py ddx) contra o MedVote atual.

  python analyze_ddx.py                                   # ajuste: piloto de 300 questões da validação
  MEDVOTE_OUT=outputs/medvote/confirmacao python analyze_ddx.py --name confirmacao --third none

Mesmo portão nos dois sistemas (voto de 11 amostras do nano + 10 do 4o-mini, concordância < 90%); muda só a etapa 2.
--third escolhe a regra da 3ª candidata (none ou a fatia mínima, ex. 0.15); sem --third, o ajuste testa todas.
Grava analysis/ddx_<name>.json.
"""
import argparse
import json
import os
import sys

import analyze
import analyze_medvote as am

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report'))
import medvote_test  # noqa: E402

CONFIG = {'nano': 11, 'mini': 10, 'threshold': 0.9, 'variant': 'books_cases', 'voters': (am.NANO,)}
THIRDS = [None, 0.1, 0.15, 0.2, 0.25]


def load(name):
    path = os.path.join(am.OUT_DIR, f'validation-{name}.jsonl')
    return {r['qid']: r for r in map(json.loads, open(path, encoding='utf-8'))} if os.path.exists(path) else {}


def ddx_cost(r):
    return am.usd(r['planning_usage']) + am.usd(r['card_usage']) + am.usd(r['usage'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', default='ajuste')
    p.add_argument('--third', default=None, help="'none' ou fatia mínima; sem valor, testa todas (ajuste)")
    args = p.parse_args()
    votes, diffs = load('vote'), load('diff')
    d2, d3, rag = load('ddx-2c'), load('ddx-3c'), load('rag')
    qids = sorted(votes)
    N = len(qids)
    gold = {q: votes[q]['gold'] for q in qids}

    dec, cost, esc = medvote_test.fixed_config(votes, diffs, CONFIG)
    mv_ok = {q: dec[q] == gold[q] for q in qids}
    s1 = {q: am.majority(votes[q]['samples'][am.NANO][:11] + votes[q]['samples'][am.MINI][:10]) for q in qids}
    s1_cost = {q: am.stage1_cost(votes[q], 11, 10) for q in qids}
    missing = [q for q in esc if q not in d2]
    if missing:
        print(f'Atenção: {len(missing)} questões escaladas sem registro de fichas; ficam com a resposta do voto.')

    def fichas(third):
        out, c = {}, {}
        for q in qids:
            out[q], c[q] = s1[q], s1_cost[q]
            if q not in esc or q not in d2:
                continue
            r = d2[q]
            if third is not None and q in d3 and d3[q]['shares'].get(d3[q]['candidates'][2], 0) >= third:
                r = d3[q]
            out[q] = am.majority(r['answers'][am.NANO])
            c[q] += ddx_cost(r)
        return out, c

    rows = []

    def add(name, ans, c, key):
        ok = {q: ans[q] == gold[q] for q in qids}
        x = sum(ok[q] and not mv_ok[q] for q in qids)
        y = sum(mv_ok[q] and not ok[q] for q in qids)
        d = (x - y) / N
        se = ((x + y) - (x - y) ** 2 / N) ** 0.5 / N
        k_esc = sum(ok[q] for q in esc)
        rows.append({'key': key, 'name': name, 'k': sum(ok.values()), 'n': N, 'acc': sum(ok.values()) / N,
                     'ci': analyze.wilson(sum(ok.values()), N), 'usd': sum(c.values()) / N,
                     'gain': x, 'loss': y, 'p': analyze.mcnemar_exact(x, y), 'diff': d,
                     'diff_ci': [d - 1.96 * se, d + 1.96 * se], 'escalated_correct': k_esc})

    add('Só o voto (etapa 1)', s1, s1_cost, 'vote')
    add('MedVote atual (diferencial com livros e questões resolvidas)', dec, cost, 'medvote')
    thirds = THIRDS if args.third is None else [None if args.third == 'none' else float(args.third)]
    for t in thirds:
        a, c = fichas(t)
        lab = 'MedVote com fichas, 2 candidatas' if t is None else f'MedVote com fichas, 3ª candidata com ≥ {t:.0%} dos votos'
        add(lab, a, c, 'ddx' if t is None else f'ddx3_{t}')
    if rag:
        ra = {q: am.majority(rag[q]['answers'][am.NANO][:3]) for q in qids}
        def rag3_cost(r):  # um prompt por chamada; saída proporcional a 3 amostras
            u = r['usage'][am.NANO]
            n = len(r['answers'][am.NANO])
            return am.usd(r['planning_usage']) + am.usd({am.NANO: {'prompt_tokens': u['prompt_tokens'],
                                                                   'completion_tokens': u['completion_tokens'] * 3 / n}})
        rc = {q: rag3_cost(rag[q]) for q in qids}
        add('Etapa única com evidência, 3 amostras', ra, rc, 'single3')

    gold_in = sum(gold[q] in d2[q]['candidates'] for q in esc if q in d2)
    res = {'name': args.name, 'n': N, 'escalated': len(esc), 'gold_in_candidates': gold_in, 'rows': rows,
           'run_usd': sum(ddx_cost(r) for r in list(d2.values()) + list(d3.values())),
           'extra_candidate_questions': len(d3)}
    print(f'## {args.name}: {N} questões, {len(esc)} na etapa 2 (gabarito entre as 2 candidatas em {gold_in})\n')
    print('| Sistema | Acertos | Acurácia | Acertos na etapa 2 | US$ / 1.000 q | vs MedVote atual | IC 95% (pontos) | p |')
    print('|---|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['name']} | {r['k']} | {r['acc']:.1%} | {r['escalated_correct']} de {len(esc)} | {r['usd'] * 1000:.2f} | "
              f"+{r['gain']} / −{r['loss']} | {r['diff_ci'][0] * 100:+.1f} a {r['diff_ci'][1] * 100:+.1f} | {r['p']:.3f} |")
    print(f"\nCusto das fichas: US$ {res['run_usd']:.2f} ({len(d3)} questões com 3ª candidata)")
    os.makedirs('analysis', exist_ok=True)
    json.dump(res, open(f'analysis/ddx_{args.name}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
