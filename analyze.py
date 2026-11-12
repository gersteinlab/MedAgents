"""Análise dos resultados em outputs/MedQA/<modelo>/*.jsonl.

  .\.venv\Scripts\python.exe analyze.py                 # todos os modelos
  .\.venv\Scripts\python.exe analyze.py --models gpt-4o-mini

Imprime as tabelas em Markdown e grava analysis/results.json (usado para montar o relatório).
Acerto = letra prevista igual ao gabarito (comparação estrita).
"""
import argparse
import glob
import json
import math
import os
import random
from collections import Counter, defaultdict

N_TOTAL = 1273

# rótulo de exibição por arquivo "<metodo>[-tag]"
METHOD_LABELS = {
    'base_direct': 'Resposta direta',
    'base_cot': 'CoT · 300 tokens',
    'base_cot-cot1000': 'CoT · 1.000 tokens',
    'base_sc': 'Self-Consistency',
    'syn_only': 'MedAgents sem votação',
    'syn_verif': 'MedAgents',
    'syn_verif-fix': 'MedAgents corrigido',
}
MA = 'syn_verif'
REVIEWER_DOMAIN = 'general medicine (external reviewer)'

# US$ por 1M tokens (entrada, saída), só para estimar custo. Ignora o desconto de entrada em cache,
# então tende a superestimar um pouco. gpt-4o-mini e gpt-5.4-nano conferidos em
# developers.openai.com/api/docs/pricing (09/11/2026); Claude conforme a tabela da Anthropic.
PRICES = {
    'gpt-4o-mini': (0.15, 0.60), 'gpt-5.4-nano': (0.20, 1.25),
    'gpt-4.1-mini': (0.40, 1.60), 'gpt-4o': (2.50, 10.00),
    'claude-opus-5-5': (4.00, 20.00), 'claude-sonnet-5-5': (2.00, 10.00), 'claude-haiku-5-5': (0.10, 0.50),
}


def label(key):
    if key.startswith('syn_verif_review'):
        return 'MedAgents + revisor ' + key[len('syn_verif_review'):].lstrip('-')
    return METHOD_LABELS.get(key, key)


# ---------------------------------------------------------------- estatística
def wilson(k, n, z=1.96):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b, c):
    """p bilateral exato (binomial com p=0,5 sobre os pares discordantes)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pct(x):
    return f"{100 * x:.1f}%" if x == x else "—"


# ---------------------------------------------------------------- carga
def load_runs(root, models=None):
    runs = defaultdict(dict)  # modelo -> chave do método -> {qid: registro}
    for path in sorted(glob.glob(os.path.join(root, '*', '*.jsonl'))):
        model = os.path.basename(os.path.dirname(path))
        if models and model not in models:
            continue
        name = os.path.basename(path)[:-len('.jsonl')]
        # arquivo: <modelo>-<metodo>[-tag]; o modelo pode ter hífens, então casamos pelo método
        key = None
        for m in ['base_direct', 'base_cot', 'base_sc', 'syn_only', 'syn_verif_review', 'syn_verif', 'anal_only']:
            marker = f'-{m}'
            if marker in name:
                key = m + name.split(marker, 1)[1]
                break
        if key is None:
            continue
        recs = {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    recs[r['qid']] = r
        runs[model][key] = recs
    return runs


def method_calls(r):
    """Chamadas do método em si, sem as de instrumentação (round_answers)."""
    return r.get('usage', {}).get('calls', 0) - r.get('probe_usage', {}).get('calls', 0)


def method_tokens(r, kind):
    return r.get('usage', {}).get(kind, 0) - r.get('probe_usage', {}).get(kind, 0)


def usd_per_question(recs):
    """Custo médio estimado por questão (modelo da equipe + revisor). None se faltar preço ou tokens."""
    total = 0.0
    for r in recs.values():
        price = PRICES.get(r.get('model', ''))
        if not r.get('usage') or price is None:
            return None
        total += (method_tokens(r, 'prompt_tokens') * price[0] + method_tokens(r, 'completion_tokens') * price[1]) / 1e6
        ru = r.get('reviewer_usage')
        if ru:
            rp = PRICES.get(r.get('reviewer_model', ''))
            if rp is None:
                return None
            total += (ru.get('input_tokens', 0) * rp[0] + ru.get('output_tokens', 0) * rp[1]) / 1e6
    return total / len(recs) if recs else None


def correct(r):
    return r['pred_answer'] == r['gold_answer']


# ---------------------------------------------------------------- blocos de análise
def accuracy_table(methods):
    rows = []
    for key, recs in methods.items():
        n = len(recs)
        k = sum(correct(r) for r in recs.values())
        by_step = defaultdict(lambda: [0, 0])
        for r in recs.values():
            by_step[r.get('step', '')][0] += correct(r)
            by_step[r.get('step', '')][1] += 1
        calls = [method_calls(r) for r in recs.values()]
        samples = [(r.get('sc_samples', 0) or method_calls(r)) + r.get('reviewer_usage', {}).get('calls', 0)
                   for r in recs.values()]
        rtok = [r.get('reviewer_usage', {}).get('input_tokens', 0) + r.get('reviewer_usage', {}).get('output_tokens', 0)
                for r in recs.values()]
        ptok = [method_tokens(r, 'prompt_tokens') for r in recs.values()]
        ctok = [method_tokens(r, 'completion_tokens') for r in recs.values()]
        lo, hi = wilson(k, n)
        rows.append({
            'key': key, 'label': label(key), 'n': n, 'correct': k, 'acc': k / n if n else float('nan'),
            'ci': [lo, hi],
            'step1': by_step['step1'][0] / by_step['step1'][1] if by_step['step1'][1] else float('nan'),
            'step23': by_step['step2&3'][0] / by_step['step2&3'][1] if by_step['step2&3'][1] else float('nan'),
            'mean_requests': sum(calls) / n if n else 0,
            'mean_generations': sum(samples) / n if n else 0,
            'mean_prompt_tokens': sum(ptok) / n if n else 0,
            'mean_completion_tokens': sum(ctok) / n if n else 0,
            'empty_answers': sum(1 for r in recs.values() if not r['pred_answer']),
            'mean_reviewer_tokens': sum(rtok) / n if n else 0,
            'usd_per_question': usd_per_question(recs),
        })
    return rows


def mcnemar_vs_ma(methods):
    if MA not in methods:
        return []
    ma = methods[MA]
    out = []
    for key, recs in methods.items():
        if key == MA:
            continue
        common = sorted(set(ma) & set(recs))
        b = sum(1 for q in common if correct(ma[q]) and not correct(recs[q]))  # só MedAgents
        c = sum(1 for q in common if not correct(ma[q]) and correct(recs[q]))  # só o outro
        out.append({'key': key, 'label': label(key), 'n': len(common),
                    'only_ma': b, 'only_other': c, 'p': mcnemar_exact(b, c)})
    return out


def sc_curve(recs, reps=200, seed=0):
    """Acurácia esperada do voto majoritário com k amostras sorteadas sem reposição
    entre as K gravadas (média sobre `reps` sorteios por questão)."""
    from utils import majority_vote  # mesma regra de desempate do experimento
    rng = random.Random(seed)
    K = min(len(r['sc_answers']) for r in recs.values())
    curve = []
    for k in range(1, K + 1):
        if k == K:
            acc = sum(majority_vote(r['sc_answers'][:K]) == r['gold_answer'] for r in recs.values()) / len(recs)
        else:
            tot = 0.0
            for r in recs.values():
                hits = 0
                for _ in range(reps):
                    hits += majority_vote(rng.sample(r['sc_answers'][:K], k)) == r['gold_answer']
                tot += hits / reps
            acc = tot / len(recs)
        curve.append({'k': k, 'acc': acc})
    return curve


def sc_agreement(r):
    v = [a for a in r['sc_answers'] if a]
    if not v:
        return 0.0
    return Counter(v).most_common(1)[0][1] / len(r['sc_answers'])


def n_rounds(r):
    return len(r['vote_history'])


def by_rounds(methods):
    """Cada método avaliado dentro dos grupos definidos pelo número de rodadas do MedAgents."""
    if MA not in methods:
        return []
    ma = methods[MA]
    groups = defaultdict(list)
    for q, r in ma.items():
        groups[n_rounds(r)].append(q)
    out = []
    for g in sorted(groups):
        qs = groups[g]
        row = {'rounds': g, 'n': len(qs), 'methods': {}}
        for key, recs in methods.items():
            have = [q for q in qs if q in recs]
            if have:
                row['methods'][key] = sum(correct(recs[q]) for q in have) / len(have)
        if 'base_sc' in methods:
            have = [q for q in qs if q in methods['base_sc']]
            if have:
                row['sc_mean_agreement'] = sum(sc_agreement(methods['base_sc'][q]) for q in have) / len(have)
                row['sc_unanimous_share'] = sum(sc_agreement(methods['base_sc'][q]) == 1.0 for q in have) / len(have)
        # rodada 1: quantos especialistas votaram "não"
        nos = [sum(v == 'no' for d, v in ma[q]['vote_history'][0].items() if d != REVIEWER_DOMAIN)
               for q in qs if ma[q]['vote_history']]
        row['mean_no_votes_round1'] = sum(nos) / len(nos) if nos else 0
        out.append(row)
    return out


def sc_agreement_vs_rounds(methods, bins=((0, .5), (.5, .7), (.7, .9), (.9, 1.0001))):
    """Os dois sinais de incerteza (dispersão do SC e discordância dos agentes) apontam
    para as mesmas questões? Linhas: faixa de concordância do SC; colunas: rodadas."""
    if MA not in methods or 'base_sc' not in methods:
        return []
    ma, sc = methods[MA], methods['base_sc']
    out = []
    for lo, hi in bins:
        qs = [q for q in ma if q in sc and lo <= sc_agreement(sc[q]) < hi]
        if not qs:
            continue
        rc = Counter(n_rounds(ma[q]) for q in qs)
        out.append({
            'bin': [lo, min(hi, 1.0)], 'n': len(qs),
            'rounds_dist': {str(k): rc[k] for k in sorted(rc)},
            'acc_sc': sum(correct(sc[q]) for q in qs) / len(qs),
            'acc_ma': sum(correct(ma[q]) for q in qs) / len(qs),
        })
    return out


def round_dynamics(ma):
    """Usa round_answers: a resposta que cada versão do relatório levaria a escolher."""
    recs = [r for r in ma.values() if r.get('round_answers')]
    if not recs:
        return None
    trans = defaultdict(Counter)  # (v -> v+1) -> tipo de transição
    for r in recs:
        ra, gold = r['round_answers'], r['gold_answer']
        for v in range(len(ra) - 1):
            a, b = ra[v], ra[v + 1]
            if a == b:
                t = 'mantém certa' if a == gold else 'mantém errada'
            elif b == gold:
                t = 'errada → certa'
            elif a == gold:
                t = 'certa → errada'
            else:
                t = 'errada → outra errada'
            trans[f'v{v}→v{v + 1}'][t] += 1
    initial_acc = sum(r['round_answers'][0] == r['gold_answer'] for r in recs) / len(recs)
    final_acc = sum(correct(r) for r in recs) / len(recs)
    revised = [r for r in recs if len(r['round_answers']) > 1]
    changed = [r for r in revised if r['round_answers'][0] != r['round_answers'][-1]]
    return {
        'n': len(recs),
        'acc_initial_report': initial_acc,
        'acc_final': final_acc,
        'n_revised': len(revised),
        'n_answer_changed': len(changed),
        'changed_to_correct': sum(correct(r) for r in changed),
        'changed_to_wrong': sum(r['round_answers'][0] == r['gold_answer'] for r in changed),
        'transitions': {k: dict(v) for k, v in sorted(trans.items())},
    }


def votes_detail(ma):
    """Votos "não" por rodada e acurácia pelo número de "não" na primeira votação."""
    per_round = defaultdict(list)
    by_no1 = defaultdict(lambda: [0, 0])
    dissent = defaultdict(lambda: [0, 0])  # especialidade -> [não, votos]
    for r in ma.values():
        for i, votes in enumerate(r['vote_history']):
            votes = {d: v for d, v in votes.items() if d != REVIEWER_DOMAIN}
            per_round[i + 1].append(sum(v == 'no' for v in votes.values()))
            for dom, v in votes.items():
                dissent[dom.strip()][0] += v == 'no'
                dissent[dom.strip()][1] += 1
        if r['vote_history']:
            k = sum(v == 'no' for d, v in r['vote_history'][0].items() if d != REVIEWER_DOMAIN)
            by_no1[k][0] += correct(r)
            by_no1[k][1] += 1
    top = sorted(((d, a / b, b) for d, (a, b) in dissent.items() if b >= 30), key=lambda x: -x[1])[:10]
    return {
        'mean_no_per_round': {k: sum(v) / len(v) for k, v in sorted(per_round.items())},
        'questions_per_round': {k: len(v) for k, v in sorted(per_round.items())},
        'acc_by_no_votes_round1': {k: {'n': b, 'acc': a / b} for k, (a, b) in sorted(by_no1.items())},
        'top_dissenting_specialties': [{'domain': d, 'no_rate': x, 'votes': n} for d, x, n in top],
    }


def reviewer_detail(recs, ma=None):
    """O que o revisor fez, rodada a rodada, e o efeito dele sobre o MedAgents sem revisor."""
    rs = [r for r in recs.values() if r.get('reviewer_history')]
    if not rs:
        return None
    no_by_round = defaultdict(lambda: [0, 0])
    for r in rs:
        for i, rv in enumerate(r['reviewer_history']):
            no_by_round[i + 1][0] += rv['vote'] == 'no'
            no_by_round[i + 1][1] += 1
    first = [r['reviewer_history'][0] for r in rs]
    alone = [(rv['answer'], r) for rv, r in zip(first, rs) if rv['answer']]
    out = {
        'n': len(rs),
        'no_rate_by_round': {k: {'no': x, 'votes': y, 'rate': x / y} for k, (x, y) in sorted(no_by_round.items())},
        'failed': sum(rv['raw'] == 'ERROR.' for r in rs for rv in r['reviewer_history']),
        'refused': sum(rv['raw'] == 'REFUSED.' for r in rs for rv in r['reviewer_history']),
        'served_by_fallback': sum(r.get('reviewer_usage', {}).get('served_by_fallback', 0) for r in rs),
        'claude_alone_acc': sum(ans == r['gold_answer'] for ans, r in alone) / len(alone) if alone else float('nan'),
        'claude_alone_n': len(alone),
        'final_agrees_with_claude': sum(r['pred_answer'] == ans for ans, r in alone) / len(alone) if alone else float('nan'),
        'acc_when_round1_no': None, 'acc_when_round1_yes': None,
    }
    for vote in ('no', 'yes'):
        sub = [r for rv, r in zip(first, rs) if rv['vote'] == vote]
        if sub:
            out[f'acc_when_round1_{vote}'] = {'n': len(sub), 'acc': sum(correct(r) for r in sub) / len(sub)}
    if ma:
        common = [r for r in rs if r['qid'] in ma]
        if common:
            out['vs_medagents'] = {
                'n': len(common),
                'corrected': sum(correct(r) and not correct(ma[r['qid']]) for r in common),
                'broken': sum(correct(ma[r['qid']]) and not correct(r) for r in common),
                'acc_medagents_same_questions': sum(correct(ma[r['qid']]) for r in common) / len(common),
                'acc_with_reviewer': sum(correct(r) for r in common) / len(common),
                'rounds_without': dict(sorted(Counter(n_rounds(ma[r['qid']]) for r in common).items())),
                'rounds_with': dict(sorted(Counter(n_rounds(r) for r in common).items())),
            }
    return out


def cascade(methods, thresholds=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    """Estratégia híbrida: responde com o SC; só chama o MedAgents quando a concordância
    das amostras fica abaixo do limiar. Custo medido em gerações por questão."""
    if MA not in methods or 'base_sc' not in methods:
        return []
    ma, sc = methods[MA], methods['base_sc']
    qs = sorted(set(ma) & set(sc))
    sc_cost = sum(sc[q].get('sc_samples', len(sc[q]['sc_answers'])) for q in qs) / len(qs)
    out = []
    for t in thresholds:
        esc = [q for q in qs if sc_agreement(sc[q]) < t]
        acc = sum(correct(ma[q]) if q in esc else correct(sc[q]) for q in qs) / len(qs)
        extra = sum(method_calls(ma[q]) for q in esc) / len(qs)
        out.append({'threshold': t, 'escalated': len(esc), 'acc': acc, 'generations': sc_cost + extra})
    return out


# ---------------------------------------------------------------- saída
def md_table(header, rows):
    s = '| ' + ' | '.join(header) + ' |\n|' + '|'.join('---' for _ in header) + '|\n'
    for r in rows:
        s += '| ' + ' | '.join(str(x) for x in r) + ' |\n'
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='./outputs/MedQA')
    ap.add_argument('--models', nargs='*')
    ap.add_argument('--out', default='./analysis/results.json')
    args = ap.parse_args()

    runs = load_runs(args.root, args.models)
    if not runs:
        print(f'Nenhum resultado em {args.root}/<modelo>/*.jsonl')
        return
    results = {}
    for model, methods in runs.items():
        res = {'accuracy': accuracy_table(methods), 'mcnemar': mcnemar_vs_ma(methods),
               'by_rounds': by_rounds(methods), 'sc_vs_rounds': sc_agreement_vs_rounds(methods),
               'cascade': cascade(methods)}
        if 'base_sc' in methods:
            res['sc_curve'] = sc_curve(methods['base_sc'])
        if MA in methods:
            res['round_dynamics'] = round_dynamics(methods[MA])
            res['votes'] = votes_detail(methods[MA])
        res['review'] = {}
        for key, recs in methods.items():
            if key.startswith('syn_verif_review'):
                res['review'][key] = {'reviewer': reviewer_detail(recs, methods.get(MA)),
                                      'round_dynamics': round_dynamics(recs), 'votes': votes_detail(recs)}
        results[model] = res

        print(f'\n## {model}\n')
        incompletos = [f"{a['label']} ({a['n']})" for a in res['accuracy'] if a['n'] != N_TOTAL]
        if incompletos:
            print(f'Atenção, execuções incompletas: {", ".join(incompletos)}\n')
        print(md_table(['Método', 'Acertos', 'Acurácia', 'IC 95%', 'Step 1', 'Step 2&3', 'Gerações/q',
                        'Tokens/q equipe', 'Tokens/q revisor', 'US$/q (estim.)'],
              [[a['label'], f"{a['correct']}/{a['n']}", pct(a['acc']), f"{pct(a['ci'][0])}–{pct(a['ci'][1])}",
                pct(a['step1']), pct(a['step23']), f"{a['mean_generations']:.1f}",
                f"{a['mean_prompt_tokens'] + a['mean_completion_tokens']:,.0f}",
                f"{a['mean_reviewer_tokens']:,.0f}" if a['mean_reviewer_tokens'] else '—',
                f"{a['usd_per_question']:.4f}" if a['usd_per_question'] is not None else '—'] for a in res['accuracy']]))
        if res['mcnemar']:
            print(md_table(['Contra', 'Só MedAgents', 'Só o outro', 'p (McNemar)'],
                  [[m['label'], m['only_ma'], m['only_other'], f"{m['p']:.3g}"] for m in res['mcnemar']]))
        if res.get('sc_curve'):
            c = res['sc_curve']
            marks = [x for x in c if x['k'] in (1, 3, 5, 7, 11, 15, 21) or x['k'] == len(c)]
            print('Self-Consistency por número de amostras: ' +
                  ', '.join(f"k={x['k']}: {pct(x['acc'])}" for x in marks) + '\n')
        if res['by_rounds']:
            keys = [k for k in methods]
            print(md_table(['Rodadas do MedAgents', 'Questões'] + [label(k) for k in keys] + ['Concordância média do SC', '"Não" na 1ª votação'],
                  [[r['rounds'], r['n']] + [pct(r['methods'].get(k, float('nan'))) for k in keys] +
                   [pct(r.get('sc_mean_agreement', float('nan'))), f"{r['mean_no_votes_round1']:.2f}"] for r in res['by_rounds']]))
        if res['sc_vs_rounds']:
            print(md_table(['Concordância do SC', 'Questões', 'Rodadas (1/2/3)', 'Acurácia SC', 'Acurácia MedAgents'],
                  [[f"{pct(b['bin'][0])}–{pct(b['bin'][1])}", b['n'], '/'.join(str(b['rounds_dist'].get(str(i), 0)) for i in (1, 2, 3)),
                    pct(b['acc_sc']), pct(b['acc_ma'])] for b in res['sc_vs_rounds']]))
        rd = res.get('round_dynamics')
        if rd:
            print(f"Relatório inicial (antes da votação): {pct(rd['acc_initial_report'])} | final: {pct(rd['acc_final'])} | "
                  f"revisados: {rd['n_revised']} | resposta mudou: {rd['n_answer_changed']} "
                  f"(para certa: {rd['changed_to_correct']}, de certa para errada: {rd['changed_to_wrong']})")
            for k, v in rd['transitions'].items():
                print(f"  {k}: {v}")
        for key, rv in res['review'].items():
            d = rv['reviewer']
            if not d:
                continue
            print(f"\n### {label(key)}\n")
            print('Votos NÃO do revisor por rodada: ' + ', '.join(
                f"rodada {k}: {x['no']}/{x['votes']} ({pct(x['rate'])})" for k, x in d['no_rate_by_round'].items()))
            print(f"Falhas: {d['failed']} | recusas: {d['refused']} | respondidas por fallback: {d['served_by_fallback']}")
            print(f"Resposta do Claude sozinho (1ª revisão): {pct(d['claude_alone_acc'])} em {d['claude_alone_n']} questões | "
                  f"resposta final igual à do Claude: {pct(d['final_agrees_with_claude'])}")
            for vote in ('no', 'yes'):
                x = d[f'acc_when_round1_{vote}']
                if x:
                    print(f"  Claude votou {'NÃO' if vote == 'no' else 'SIM'} na 1ª rodada: {x['n']} questões, acurácia final {pct(x['acc'])}")
            vs = d.get('vs_medagents')
            if vs:
                print(f"Contra o MedAgents sem revisor nas mesmas {vs['n']} questões: {pct(vs['acc_medagents_same_questions'])} → "
                      f"{pct(vs['acc_with_reviewer'])} | corrigiu {vs['corrected']}, estragou {vs['broken']}")
                print(f"  Rodadas sem revisor: {vs['rounds_without']} | com revisor: {vs['rounds_with']}")
            if rv['round_dynamics']:
                for k, v in rv['round_dynamics']['transitions'].items():
                    print(f"  {k}: {v}")
        if res['cascade']:
            print('\n' + md_table(['Limiar de concordância do SC', 'Escaladas ao MedAgents', 'Acurácia', 'Gerações/q'],
                  [[pct(c['threshold']), c['escalated'], pct(c['acc']), f"{c['generations']:.1f}"] for c in res['cascade']]))

    if len(results) > 1:
        print('\n## Entre modelos\n')
        keys = sorted({a['key'] for r in results.values() for a in r['accuracy']}, key=lambda k: list(METHOD_LABELS).index(k) if k in METHOD_LABELS else 99)
        print(md_table(['Modelo'] + [label(k) for k in keys] + ['% aprovado na 1ª rodada'],
              [[m] + [next((pct(a['acc']) for a in r['accuracy'] if a['key'] == k), '—') for k in keys] +
               [next((pct(b['n'] / sum(x['n'] for x in r['by_rounds'])) for b in r['by_rounds'] if b['rounds'] == 1), '—')]
               for m, r in results.items()]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f'\nResultados gravados em {args.out}')


if __name__ == '__main__':
    main()
