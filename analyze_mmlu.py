"""MedVote no MMLU médico (6 matérias, 4 alternativas), com as configurações fixadas no MedQA.

  python analyze_mmlu.py --split test          # ou --split validation (ensaio)

Comparação principal, fixada antes de rodar: MedVote com fichas e o peso do voto (ddx --prior) contra o MedVote atual.
Secundárias: o voto sozinho, as fichas sem o peso do voto e a etapa única com evidência e 3 amostras.
Mesmo portão em todos (voto de 11 amostras do nano + 10 do 4o-mini, concordância < 90%).
Grava analysis/mmlu_<split>.json.
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import analyze
import analyze_medvote as am
import retrieval

am.OUT_DIR = os.environ.get('MEDVOTE_OUT', os.path.join('outputs', 'medvote', 'mmlu'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report'))
import medvote_test  # noqa: E402

CONFIG = {'nano': 11, 'mini': 10, 'threshold': 0.9, 'variant': 'books_cases', 'voters': (am.NANO,)}
SUBJECT = {'anatomy': 'Anatomia', 'clinical_knowledge': 'Conhecimento clínico', 'college_biology': 'Biologia (graduação)',
           'college_medicine': 'Medicina (graduação)', 'medical_genetics': 'Genética médica',
           'professional_medicine': 'Medicina profissional'}


def load(split, name):
    path = os.path.join(am.OUT_DIR, f'{split}-{name}.jsonl')
    return {r['qid']: r for r in map(json.loads, open(path, encoding='utf-8'))} if os.path.exists(path) else {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', default='test')
    args = p.parse_args()
    votes, diffs = load(args.split, 'vote'), load(args.split, 'diff')
    cards, prior, rag = load(args.split, 'ddx-2c'), load(args.split, 'ddx-prior'), load(args.split, 'rag')
    subj = {q['qid']: q['step'] for q in retrieval.load_split(args.split, 'mmlu')}
    qids = sorted(votes)
    N = len(qids)
    gold = {q: votes[q]['gold'] for q in qids}

    dec, cost, esc = medvote_test.fixed_config(votes, diffs, CONFIG)
    mv_ok = {q: dec[q] == gold[q] for q in qids}
    s1 = {q: am.majority(votes[q]['samples'][am.NANO][:11] + votes[q]['samples'][am.MINI][:10]) for q in qids}
    s1_cost = {q: am.stage1_cost(votes[q], 11, 10) for q in qids}
    for name, d in (('diff', diffs), ('ddx-2c', cards), ('ddx-prior', prior)):
        miss = [q for q in esc if q not in d]
        if miss:
            print(f'Atenção: {len(miss)} questões escaladas sem registro em {name}; ficam com a resposta do voto.')

    def stage2(records):
        out, c = {}, {}
        for q in qids:
            out[q], c[q] = s1[q], s1_cost[q]
            if q in esc and q in records:
                r = records[q]
                out[q] = am.majority(r['answers'][am.NANO])
                c[q] += am.usd(r['planning_usage']) + am.usd(r['card_usage']) + am.usd(r['usage'])
        return out, c

    def rag3_cost(r):  # um prompt por chamada; saída proporcional a 3 amostras
        u = r['usage'][am.NANO]
        n = len(r['answers'][am.NANO])
        return am.usd(r['planning_usage']) + am.usd({am.NANO: {'prompt_tokens': u['prompt_tokens'],
                                                               'completion_tokens': u['completion_tokens'] * 3 / n}})

    systems = [('medvote', 'MedVote atual', dec, cost),
               ('prior', 'MedVote com fichas e o peso do voto', *stage2(prior)),
               ('cards', 'MedVote com fichas (sem o peso do voto)', *stage2(cards)),
               ('vote', 'Só o voto (etapa 1)', s1, s1_cost)]
    if rag:
        systems.append(('single3', 'Etapa única com evidência, 3 amostras',
                        {q: am.majority(rag[q]['answers'][am.NANO][:3]) if q in rag else '' for q in qids},
                        {q: rag3_cost(rag[q]) if q in rag else 0 for q in qids}))

    rows, by_subject = [], defaultdict(dict)
    for key, name, ans, c in systems:
        ok = {q: ans[q] == gold[q] for q in qids}
        x = sum(ok[q] and not mv_ok[q] for q in qids)
        y = sum(mv_ok[q] and not ok[q] for q in qids)
        d = (x - y) / N
        se = ((x + y) - (x - y) ** 2 / N) ** 0.5 / N
        k = sum(ok.values())
        switched = sum(ans[q] != s1[q] for q in esc)
        rows.append({'key': key, 'name': name, 'k': k, 'n': N, 'acc': k / N, 'ci': analyze.wilson(k, N),
                     'usd': sum(c.values()) / N, 'gain': x, 'loss': y, 'p': analyze.mcnemar_exact(x, y), 'diff': d,
                     'diff_ci': [d - 1.96 * se, d + 1.96 * se], 'escalated_correct': sum(ok[q] for q in esc),
                     'switched_from_vote': switched})
        for s in SUBJECT:
            qs = [q for q in qids if subj[q] == s]
            by_subject[s][key] = sum(ok[q] for q in qs) / len(qs) if qs else None
            by_subject[s]['n'] = len(qs)

    gold_in = sum(gold[q] in diffs[q]['candidates'] for q in esc if q in diffs)
    res = {'split': args.split, 'n': N, 'escalated': len(esc), 'gold_in_candidates': gold_in, 'rows': rows,
           'subjects': [{'subject': s, 'label': SUBJECT[s], **by_subject[s]} for s in SUBJECT],
           'run_usd': sum(sum(c.values()) for _, _, _, c in systems[:1]) + sum(
               am.usd(r['planning_usage']) + am.usd(r['card_usage']) + am.usd(r['usage'])
               for d in (cards, prior) for r in d.values()) + sum(
               am.usd(r['planning_usage']) + am.usd(r['usage']) for r in rag.values())}
    print(f'## MMLU médico, {args.split}: {N} questões, {len(esc)} na etapa 2 '
          f'(gabarito entre as 2 candidatas em {gold_in})\n')
    print('| Sistema | Acertos | Acurácia | Etapa 2 | Trocas do voto | US$ / 1.000 q | vs MedVote | IC 95% (pontos) | p |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['name']} | {r['k']} | {r['acc']:.1%} | {r['escalated_correct']} de {len(esc)} | "
              f"{r['switched_from_vote']} | {r['usd'] * 1000:.2f} | +{r['gain']} / −{r['loss']} | "
              f"{r['diff_ci'][0] * 100:+.1f} a {r['diff_ci'][1] * 100:+.1f} | {r['p']:.3f} |")
    print('\n| Matéria | n | ' + ' | '.join(k for k, *_ in systems) + ' |')
    print('|---|---|' + '---|' * len(systems))
    for s in SUBJECT:
        b = by_subject[s]
        print(f"| {SUBJECT[s]} | {b['n']} | " + ' | '.join(f"{b[k]:.1%}" for k, *_ in systems) + ' |')
    print(f"\nCusto medido das chamadas: US$ {res['run_usd']:.2f}")
    os.makedirs('analysis', exist_ok=True)
    json.dump(res, open(f'analysis/mmlu_{args.split}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
