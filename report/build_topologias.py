"""Gera report/topologias.html: proposta de topologias para o MedAgents, testes sem custo e piloto.

  .\.venv\Scripts\python.exe analyze_debate.py
  .\.venv\Scripts\python.exe report\build_topologias.py
"""
import json
import math
import os
import random
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import analyze  # noqa: E402

MODELS = ['gpt-4o-mini', 'gpt-5.4-nano']


def load(m, k):
    return {r['qid']: r for r in map(json.loads, open(f'outputs/MedQA/{m}/{m}-{k}.jsonl', encoding='utf-8'))}


def mv(xs):
    v = [a for a in xs if a]
    if not v:
        return ''
    c = Counter(v); b = max(c.values())
    return next(a for a in v if c[a] == b)


def pct(x, d=1):
    return f"{x * 100:.{d}f}".replace('.', ',') + '%'


def num(x, d=4):
    return f"{x:.{d}f}".replace('.', ',')


def pfmt(p):
    return 'p < 0,001' if p < 0.001 else 'p = ' + f"{p:.2f}".replace('.', ',')


# ------------------------------------------------------------------ testes sem custo
def offline():
    out = {}
    for m in MODELS:
        ma, sc = load(m, 'syn_verif'), load(m, 'base_sc')
        N = len(ma)
        ok = lambda a, q: a == ma[q]['gold_answer']
        base = {q: ok(r['pred_answer'], q) for q, r in ma.items()}
        V = [
            ('MedAgents original', base, 'Até 3 votações; se ainda houver “não”, a revisão é aplicada mesmo assim.'),
            ('Relatório inicial, sem votação', {q: ok(r['round_answers'][0], q) for q, r in ma.items()},
             'Decide direto pelo relatório sintetizado.'),
            ('Para no último relatório votado', {q: ok(r['round_answers'][len(r['vote_history']) - 1], q) for q, r in ma.items()},
             'Descarta a revisão que nenhum especialista votou.'),
            ('No máximo 1 rodada', {q: ok(r['round_answers'][min(1, len(r['round_answers']) - 1)], q) for q, r in ma.items()},
             'Uma votação e uma revisão.'),
        ]
        scd = {q: ok(sc[q]['pred_answer'], q) for q in sc}
        rows = []
        for name, v, desc in V:
            a = sum(v[q] and not base[q] for q in v); b = sum(base[q] and not v[q] for q in v)
            rows.append({'name': name, 'desc': desc, 'acc': sum(v.values()) / N, 'plus': a, 'minus': b,
                         'p': analyze.mcnemar_exact(a, b) if name != 'MedAgents original' else None})
        out[m] = {'rows': rows, 'sc': sum(scd.values()) / N,
                  'unvoted': sum(len(r['round_answers']) > len(r['vote_history']) for r in ma.values())}
    a, b = load('gpt-4o-mini', 'base_sc'), load('gpt-5.4-nano', 'base_sc')
    g = {q: a[q]['gold_answer'] for q in a}
    N = len(a)
    acc = lambda f: sum(f(q) == g[q] for q in a) / N
    het = [('21 amostras do gpt-4o-mini', acc(lambda q: mv(a[q]['sc_answers'])), 21, 0),
           ('21 amostras do gpt-5.4-nano', acc(lambda q: mv(b[q]['sc_answers'])), 0, 21)]
    for ka, kb in [(14, 7), (10, 11), (7, 14)]:
        r = []
        for s in range(50):
            rr = random.Random(s)
            r.append(acc(lambda q: mv(rr.sample(a[q]['sc_answers'], ka) + rr.sample(b[q]['sc_answers'], kb))))
        het.append((f'{ka} do gpt-4o-mini + {kb} do gpt-5.4-nano', sum(r) / len(r), ka, kb))
    # custo por amostra de cada modelo (média medida no SC completo)
    cost_a = analyze.usd_per_question(a) / 21
    cost_b = analyze.usd_per_question(b) / 21
    het = [(n_, ac, ka * cost_a + kb * cost_b) for n_, ac, ka, kb in het]
    cats = Counter((mv(a[q]['sc_answers']) == g[q], mv(b[q]['sc_answers']) == g[q]) for q in a)
    return out, het, cats



# ------------------------------------------------------------------ diagnóstico do corte
def coverage(recs, qs):
    """Fração das análises de alternativas que citam o início do texto de cada alternativa."""
    hit = Counter(); tot = 0
    for q in qs:
        r = recs[q]
        for txt in r['option_analyses'].values():
            tot += 1
            for L, o in r['options'].items():
                if ' '.join(o.lower().split()[:3]) in txt.lower():
                    hit[L] += 1
    return {L: hit[L] / tot for L in 'ABCDE'}


def diagnosis():
    out = {}
    for m in MODELS:
        ma, sc = load(m, 'syn_verif'), load(m, 'base_sc')
        qs = sorted(ma)
        by = {}
        for L in 'ABCDE':
            qL = [q for q in qs if ma[q]['gold_answer'] == L]
            by[L] = (sum(ma[q]['pred_answer'] == L for q in qL) / len(qL),
                     sum(sc[q]['pred_answer'] == L for q in qL) / len(qL), len(qL))
        d = {'cov': coverage(ma, qs), 'by': by}
        path = f'outputs/MedQA/{m}/{m}-syn_verif-fix.jsonl'
        if os.path.exists(path):
            fx = load(m, 'syn_verif-fix'); fq = sorted(fx)
            sub = lambda dd: {q: dd[q] for q in fq}
            acc = lambda dd: sum(dd[q]['pred_answer'] == dd[q]['gold_answer'] for q in fq)
            d['pilot'] = {'n': len(fq), 'cov_orig': coverage(ma, fq), 'cov_fix': coverage(fx, fq),
                          'acc_orig': acc(ma), 'acc_fix': acc(fx), 'acc_sc': acc(sc),
                          'usd_orig': analyze.usd_per_question(sub(ma)), 'usd_fix': analyze.usd_per_question(fx),
                          'calls_orig': sum(analyze.method_calls(ma[q]) for q in fq) / len(fq),
                          'calls_fix': sum(analyze.method_calls(fx[q]) for q in fq) / len(fq)}
        out[m] = d
    return out


# ------------------------------------------------------------------ diagrama
def graph_svg(kind, title, sub):
    W, H, R, cx, cy = 220, 200, 70, 110, 96
    pts = [(cx + R * math.sin(2 * math.pi * i / 7), cy - R * math.cos(2 * math.pi * i / 7)) for i in range(7)]
    lines = []
    if kind == 'star':
        lines = [((cx, cy), p) for p in pts]
    elif kind == 'full':
        lines = [(pts[i], pts[j]) for i in range(7) for j in range(i + 1, 7)]
    else:
        lines = [(pts[i], pts[(i + 1) % 7]) for i in range(7)]
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}: {sub}">']
    for (x1, y1), (x2, y2) in lines:
        s.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" class="edge"/>')
    if kind == 'star':
        s.append(f'<rect x="{cx - 17}" y="{cy - 12}" width="34" height="24" rx="3" class="hub"/>')
        s.append(f'<text x="{cx}" y="{cy + 4}" text-anchor="middle" class="hub-t">rel.</text>')
    for x, y in pts:
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" class="node {kind}"/>')
    s.append('</svg>')
    return f'<figure class="graph"><div class="graph-svg">{"".join(s)}</div><figcaption><b>{title}</b><span>{sub}</span></figcaption></figure>'


# ------------------------------------------------------------------ página
def main():
    off, het, cats = offline()
    dg = diagnosis()
    cov_rows = ''
    for m in MODELS:
        d = dg[m]
        cov_rows += (f'<tr><td class="m" rowspan="3">{m}</td><td>Análises de alternativas que tratam da letra</td>'
                     + ''.join(f'<td class="n">{pct(d["cov"][L], 0)}</td>' for L in 'ABCDE') + '</tr>')
        cov_rows += ('<tr class="is-mc"><td>Acurácia do MedAgents quando é o gabarito</td>'
                     + ''.join(f'<td class="n">{pct(d["by"][L][0], 0)}</td>' for L in 'ABCDE') + '</tr>')
        cov_rows += ('<tr class="is-sc"><td>Acurácia do Self-Consistency quando é o gabarito</td>'
                     + ''.join(f'<td class="n">{pct(d["by"][L][1], 0)}</td>' for L in 'ABCDE') + '</tr>')
    pil_rows = ''
    for m in MODELS:
        pl = dg[m].get('pilot')
        if not pl:
            continue
        pil_rows += (f'<tr><td class="m">{m}</td>'
                     f'<td class="n">{pct(pl["cov_orig"]["D"], 0)} → {pct(pl["cov_fix"]["D"], 0)}</td>'
                     f'<td class="n">{pct(pl["cov_orig"]["E"], 0)} → {pct(pl["cov_fix"]["E"], 0)}</td>'
                     f'<td class="n">{pl["acc_orig"]} → {pl["acc_fix"]}<small>SC: {pl["acc_sc"]}</small></td>'
                     f'<td class="n">{num(pl["calls_orig"], 1)} → {num(pl["calls_fix"], 1)}</td>'
                     f'<td class="n">US$ {num(pl["usd_orig"] * 1000, 2)} → {num(pl["usd_fix"] * 1000, 2)}'
                     f'<small>+{(pl["usd_fix"] / pl["usd_orig"] - 1) * 100:.0f}%</small></td></tr>')
    n_fix = next(dg[m]['pilot']['n'] for m in MODELS if 'pilot' in dg[m])
    est_fix = sum(dg[m]['pilot']['usd_fix'] * 1273 for m in MODELS if 'pilot' in dg[m])
    d4 = dg['gpt-4o-mini']
    covD, covE = pct(d4['cov']['D'], 0), pct(d4['cov']['E'], 0)
    est_fix_txt = num(est_fix, 0)
    deb = json.load(open('analysis/debate_results.json', encoding='utf-8'))
    css = open(os.path.join(HERE, 'topologias.css'), encoding='utf-8').read()

    # tabela sem custo
    off_rows = ''
    for m in MODELS:
        o = off[m]
        for i, r in enumerate(o['rows']):
            delta = '' if r['p'] is None else f"+{r['plus']} / −{r['minus']}"
            pv = '' if r['p'] is None else pfmt(r['p'])
            cls = ' class="is-mc"' if i == 0 else ''
            first = f'<td rowspan="{len(o["rows"]) + 1}" class="m">{m}</td>' if i == 0 else ''
            off_rows += f'<tr{cls}>{first}<td>{r["name"]}<small>{r["desc"]}</small></td><td class="n">{pct(r["acc"])}</td><td class="n">{delta}</td><td class="n">{pv}</td></tr>'
        off_rows += f'<tr class="is-sc"><td>Self-Consistency, 21 amostras<small>Referência.</small></td><td class="n">{pct(o["sc"])}</td><td class="n"></td><td class="n"></td></tr>'

    best_het = max(h[1] for h in het)
    het_rows = ''.join(
        f'<tr{" class=hl" if a_ == best_het else ""}><td>{n_}</td><td class="n">{pct(a_)}</td><td class="n">US$ {num(c_ * 1000, 2)}</td></tr>'
        for n_, a_, c_ in het)

    # piloto: custo
    arr_names = {'gpt-4o-mini': 'gpt-4o-mini', 'gpt-5.4-nano': 'gpt-5.4-nano',
                 'hetero: gpt-4o-mini+gpt-5.4-nano': '4o-mini + nano alternados'}
    order = ['gpt-4o-mini', 'gpt-5.4-nano', 'hetero: gpt-4o-mini+gpt-5.4-nano']
    show = [('vote', 'Voto dos 7 especialistas, sem debate'), ('full@2', 'Completa, 2 rodadas'),
            ('ring@2', 'Anel, 2 rodadas'), ('ring_conf@2', 'Anel + Conformist, 2 rodadas')]
    cost_head = ''.join(f'<th class="n">{arr_names[a]}</th>' for a in order if a in deb)
    cost_rows = ''
    for key, lab in show:
        cells = ''
        for a in order:
            if a not in deb:
                continue
            row = next(r for r in deb[a]['rows'] if r['key'] == key)
            full = next(r for r in deb[a]['rows'] if r['key'] == 'full@2')
            red = '' if key in ('vote', 'full@2') else f'<small>{"−" if row["usd"] < full["usd"] else "+"}{abs(1 - row["usd"] / full["usd"]) * 100:.0f}% vs completa</small>'
            cells += f'<td class="n">US$ {num(row["usd"] * 1000, 2)}<small>{num(row["calls"], 1)} chamadas</small>{red}</td>'
        cost_rows += f'<tr><td>{lab}</td>{cells}</tr>'
    refs = deb['gpt-4o-mini']['references'] if 'gpt-4o-mini' in deb else {}
    ref_cells_sc = ref_cells_ma = ''
    for a in order:
        if a not in deb:
            continue
        rf = deb[a]['references']
        sc = [v for k, v in rf.items() if k.startswith('Self-Consistency')]
        ma = [v for k, v in rf.items() if k.startswith('MedAgents')]
        ref_cells_sc += '<td class="n">' + ' / '.join(f'US$ {num(v["usd"] * 1000, 2)}' for v in sc) + '</td>'
        ref_cells_ma += '<td class="n">' + ' / '.join(f'US$ {num(v["usd"] * 1000, 2)}' for v in ma) + '</td>'
    cost_rows += f'<tr class="is-sc"><td>Self-Consistency (referência)</td>{ref_cells_sc}</tr>'
    cost_rows += f'<tr class="is-mc"><td>MedAgents original (referência)</td>{ref_cells_ma}</tr>'

    # piloto: acurácia e trajetória
    acc_rows = ''
    for key, lab in show:
        cells = ''
        for a in order:
            if a not in deb:
                continue
            row = next(r for r in deb[a]['rows'] if r['key'] == key)
            cells += f'<td class="n">{row["correct"]} de {row["n"]}</td>'
        acc_rows += f'<tr><td>{lab}</td>{cells}</tr>'
    traj_rows = ''
    lab_t = {'full': 'Completa', 'ring': 'Anel', 'ring_conf': 'Anel + Conformist'}
    for a in order:
        if a not in deb:
            continue
        for t, v in deb[a]['agent_acc_by_round'].items():
            traj_rows += f'<tr><td>{arr_names[a]}</td><td>{lab_t[t]}</td>' + ''.join(f'<td class="n">{pct(x)}</td>' for x in v) + '</tr>'
    unan = ', '.join(f"{deb[a]['round0_unanimous']} ({arr_names[a]})" for a in order if a in deb)
    n_pilot = deb[order[0]]['n']

    full_est = {}
    for a in order:
        if a in deb:
            parts = {r['key']: r['usd'] for r in deb[a]['rows']}
            full_est[a] = (parts['full@2'] + parts['ring@2'] + parts['ring_conf@2'] - 2 * parts['vote']) * 1273
    est_total = sum(full_est.values())

    o4, on = off['gpt-4o-mini'], off['gpt-5.4-nano']
    stop4 = o4['rows'][2]; stopn = on['rows'][2]

    html = f"""<title>Topologias do MedAgents</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@500;600;700;800&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{css}</style>
<div class="page">
<header>
  <span class="label">MedQA · terceira parte · proposta e piloto</span>
  <h1>Topologias do MedAgents</h1>
  <p class="lede">Cinco artigos sobre sistemas multiagentes sugerem mudar quem conversa com quem: menos conexões (Li et al.), modelos diferentes entre os agentes (Zhang et al.) e regras que protegem a resposta da maioria (Choi et al.). Esta página mostra como essas ideias se encaixam no MedAgents, o que já deu para medir sem gastar nada e o que um piloto de {n_pilot} questões revelou sobre custo.</p>
  <p class="verdict">Nenhuma execução completa foi feita ainda. O achado mais concreto não é de topologia: o código original corta as análises dos especialistas antes das alternativas D e E, justamente onde o MedAgents perde do Self-Consistency, e a correção já passou num piloto. Com as saídas que já existem, mudar só as regras de parada do MedAgents rende no máximo 0,6 ponto. O piloto mostrou que o debate em anel custa cerca de um quarto menos que o completo, e que o anel com a regra Conformist custa menos que o próprio Self-Consistency. Se isso vem com a mesma acurácia, só a execução completa responde.</p>
  <p class="prev">Partes anteriores: <a href="https://claude.ai/artifact/2bcRN8TtUwtG4CGsGp6LB9">MedAgents no MedQA</a> · <a href="https://claude.ai/artifact/Gegp5JWH5TBL6Z18FU3F5e">MedAgents contra Self-Consistency</a> · <a href="https://claude.ai/artifact/JZfAaMwbtudi2CwiZvh5nu">Pacote de reprodução</a></p>
</header>

<section aria-labelledby="h-papers">
  <div class="section-head"><span class="label">Leitura</span><h2 id="h-papers">O que cada artigo sugere para cá</h2></div>
  <div class="table-wrap"><table class="papers">
    <thead><tr><th>Artigo</th><th>Achado principal</th><th>O que virou aqui</th></tr></thead>
    <tbody>
      <tr><td><b>Li et al., 2024</b><small>Sparse communication topology</small></td><td>Agentes que leem só os vizinhos (anel) empatam ou superam o debate em que todos leem todos, com 40% a 50% menos tokens de entrada. Debates esparsos demoram mais a convergir, o que dá mais rodadas úteis.</td><td>Topologia em anel, comparada à completa a partir das mesmas respostas iniciais (a técnica de redução de variância do artigo).</td></tr>
      <tr><td><b>Choi et al., 2025</b><small>Debate or Vote</small></td><td>O voto majoritário explica quase todo o ganho do debate. Em média, a crença de cada agente não melhora com as rodadas (um martingale). Regras que protegem a resposta da maioria, como a Conformist, ajudam.</td><td>O voto dos especialistas antes do debate entra como baseline; a regra Conformist entra como variação; a acurácia média por agente é medida a cada rodada.</td></tr>
      <tr><td><b>Zhang et al., 2025</b><small>If MAD is the answer, what is the question?</small></td><td>Cinco métodos de debate raramente superam CoT e Self-Consistency. O que mais ajudou foi misturar modelos diferentes entre os agentes (Heter-MAD).</td><td>Arranjo heterogêneo, com gpt-4o-mini e gpt-5.4-nano alternados; e um Self-Consistency misto, testado sem custo.</td></tr>
      <tr><td><b>Tang et al., 2024</b><small>MedAgents</small></td><td>O próprio artigo atribui 77% dos erros a conhecimento médico ausente ou mal recuperado, e só 8% ao raciocínio.</td><td>A base de tudo: mesmos especialistas, mesmos prompts de escolha de especialidade.</td></tr>
      <tr><td><b>Zhao et al., 2025; Sun et al., 2025</b><small>MedRAG; DyG-RAG</small></td><td>Recuperação apoiada em grafos de conhecimento: diferenças diagnósticas entre doenças parecidas (MedRAG) e eventos ordenados no tempo (DyG-RAG).</td><td>Não implementado. Ataca a causa de erro mais comum do MedAgents, mas exige montar uma base de conhecimento médica, o que é outro projeto.</td></tr>
    </tbody>
  </table></div>
</section>

<section aria-labelledby="h-graph">
  <div class="section-head"><span class="label">Comunicação</span><h2 id="h-graph">O MedAgents já é um grafo, em forma de estrela</h2>
  <div class="prose"><p>No código original, os sete especialistas nunca leem uns aos outros. Cada um escreve uma análise, um sintetizador junta tudo num relatório, e os especialistas só respondem “sim” ou “não” a esse relatório. Toda a informação passa pelo centro. No <i>Debate or Vote</i>, a versão centralizada foi a pior das três topologias testadas, nos dois modelos.</p>
  <p>Para testar topologias de verdade, os especialistas precisam debater diretamente. O método novo mantém os especialistas que o MedAgents escolhe e troca o relatório central por um debate, em duas formas.</p></div></div>
  <div class="graphs">
    {graph_svg('star', 'Estrela', 'MedAgents original: tudo passa pelo relatório central')}
    {graph_svg('full', 'Completa', 'Cada especialista lê os outros 6 (Du et al.)')}
    {graph_svg('ring', 'Anel', 'Cada especialista lê só os 2 vizinhos (Li et al.)')}
  </div>
</section>

<section aria-labelledby="h-free">
  <div class="section-head"><span class="label">Sem custo</span><h2 id="h-free">O que dá para testar com as saídas que já existem</h2>
  <div class="prose"><p>Como a execução anterior registrou a resposta de cada versão do relatório, dá para simular mudanças nas regras de parada do MedAgents sem nenhuma chamada nova. A mudança mais defensável é parar no último relatório votado: no código original, quando ainda há um “não” na terceira votação, a revisão é aplicada e decide a resposta sem que ninguém a vote. Isso aconteceu em {o4['unvoted']} questões no gpt-4o-mini e {on['unvoted']} no gpt-5.4-nano.</p>
  <p>O ganho é pequeno e não se repete: +{stop4['plus']}/−{stop4['minus']} questões no gpt-4o-mini ({pfmt(stop4['p'])}), +{stopn['plus']}/−{stopn['minus']} no gpt-5.4-nano. Nenhuma variação chega ao Self-Consistency. Ajustar as regras do MedAgents não muda o quadro.</p></div></div>
  <figure class="figure">
    <div class="figure-title">Variações das regras de parada do MedAgents, nas 1.273 questões</div>
    <div class="table-wrap"><table>
      <thead><tr><th>Modelo</th><th>Variação</th><th class="n">Acurácia</th><th class="n">vs original</th><th class="n">McNemar</th></tr></thead>
      <tbody>{off_rows}</tbody>
    </table></div>
    <p class="figure-note">“vs original”: questões que a variação acerta e o original erra / o contrário. Simulado a partir das respostas já gravadas de cada versão do relatório.</p>
  </figure>
  <figure class="figure">
    <div class="figure-title">Self-Consistency misturando os dois modelos (a ideia do Heter-MAD, sem debate)</div>
    <div class="table-wrap"><table>
      <thead><tr><th>Amostras no voto</th><th class="n">Acurácia</th><th class="n">US$ por 1.000 q</th></tr></thead>
      <tbody>{het_rows}</tbody>
    </table></div>
    <p class="figure-note">Misturas: média de 50 sorteios entre as 21 amostras gravadas de cada modelo. Custo: preço médio medido por amostra de cada modelo. O melhor arranjo foi escolhido olhando os mesmos dados, então o ganho é otimista. Os dois Self-Consistency acertam juntos {cats[(True, True)]} questões, erram juntos {cats[(False, False)]}; só o nano acerta {cats[(False, True)]}, só o 4o-mini {cats[(True, False)]}.</p>
  </figure>
</section>

<section aria-labelledby="h-cut">
  <div class="section-head"><span class="label">Diagnóstico</span><h2 id="h-cut">O MedAgents corta as próprias análises</h2>
  <div class="prose"><p>Antes de mudar quem conversa com quem, há um defeito mais simples. O código original limita cada análise de especialista a 300 tokens, e quase todas batem nesse limite; a última frase incompleta é apagada sem aviso. A análise das alternativas percorre A, B, C… em ordem, então o corte cai justamente no fim: no gpt-4o-mini, só {covD} das análises chegam a tratar da alternativa D e {covE} da E.</p>
  <p>O efeito aparece onde se espera. Quando o gabarito é A, B ou C, o MedAgents empata com o Self-Consistency ou o supera; quando é D ou E, perde de 5 a 6 pontos no gpt-4o-mini. Os dois modelos escolhem D e E menos vezes do que o gabarito pede. Há ainda um bug do código original: o Key Knowledge que o sintetizador escreve nunca entra no relatório votado (0 de 1.273 relatórios).</p></div></div>
  <figure class="figure">
    <div class="figure-title">Quanto cada alternativa é analisada, e quanto o MedAgents acerta quando ela é a correta</div>
    <div class="table-wrap"><table>
      <thead><tr><th>Modelo</th><th></th><th class="n">A</th><th class="n">B</th><th class="n">C</th><th class="n">D</th><th class="n">E</th></tr></thead>
      <tbody>{cov_rows}</tbody>
    </table></div>
    <p class="figure-note">Cobertura: análises das alternativas que citam o início do texto de cada uma, nas 1.273 questões. O gpt-5.4-nano escreve menos e é cortado menos, mas mostra o mesmo padrão.</p>
  </figure>
  <div class="prose"><p>As correções entraram no <code>run.py</code> como opções desligadas por padrão, para a reprodução original continuar igual: <code>--analysis_max_tokens 1000</code> e <code>--advice_max_tokens 1000</code> tiram os cortes, <code>--fix_key_knowledge</code> põe o Key Knowledge no relatório e <code>--stop_at_voted</code> decide pelo último relatório votado, sem pedir conselhos que não seriam usados. Um piloto de {n_fix} questões por modelo confirmou que a correção funciona; a acurácia, com essa amostra, ainda não diz nada.</p></div>
  <figure class="figure">
    <div class="figure-title">Piloto do MedAgents corrigido nas {n_fix} primeiras questões (original → corrigido)</div>
    <div class="table-wrap"><table>
      <thead><tr><th>Modelo</th><th class="n">Análises que tratam da D</th><th class="n">Tratam da E</th><th class="n">Acertos</th><th class="n">Chamadas / q</th><th class="n">US$ por 1.000 q</th></tr></thead>
      <tbody>{pil_rows}</tbody>
    </table></div>
    <p class="figure-note">As análises ficam mais longas, então cada questão custa mais, mas há menos chamadas: mais questões fecham consenso na primeira votação. Rodar as 1.273 questões nos dois modelos custaria cerca de US$ {est_fix_txt}.</p>
  </figure>
</section>

<section aria-labelledby="h-method">
  <div class="section-head"><span class="label">Método novo</span><h2 id="h-method">Especialistas que debatem diretamente</h2>
  <div class="prose"><p>Implementado em <code>debate.py</code>, como o método <code>persona_debate</code> do <code>run.py</code>. Para cada questão:</p></div></div>
  <ol class="steps">
    <li><b>Escolha dos especialistas.</b> Os mesmos dois prompts do MedAgents: 5 especialidades pela questão, 2 pelas alternativas. São 7 agentes, cada um com o papel de especialista que o MedAgents usa.</li>
    <li><b>Rodada 0.</b> Cada especialista responde sozinho, com o mesmo prompt de raciocínio do CoT e do Self-Consistency, temperatura 0,7 e até 1.000 tokens. O voto majoritário dessa rodada é o baseline “voto dos especialistas”.</li>
    <li><b>Duas rodadas de debate</b>, com o modelo de prompt do <i>Debate or Vote</i>: o agente lê as respostas que a topologia permite, relê a própria e responde de novo. As três variações partem da mesma rodada 0:
      <ul><li><b>Completa:</b> lê os outros 6.</li><li><b>Anel:</b> lê só os 2 vizinhos.</li><li><b>Anel + Conformist:</b> quem concordou com a maioria na rodada anterior mantém a resposta, sem nova chamada.</li></ul></li>
    <li><b>Resposta final.</b> Voto majoritário dos 7 agentes depois de cada rodada.</li>
  </ol>
  <div class="prose"><p>Com <code>--agent_models gpt-4o-mini,gpt-5.4-nano</code>, os agentes alternam entre os dois modelos. Cada registro guarda as respostas de todos os agentes em todas as rodadas e o custo de cada parte separado, então uma única execução avalia as três topologias, as duas rodadas e o voto sem debate.</p></div>
</section>

<section aria-labelledby="h-pilot">
  <div class="section-head"><span class="label">Piloto · {n_pilot} questões</span><h2 id="h-pilot">Quanto cada topologia custa</h2>
  <div class="prose"><p>O piloto rodou as {n_pilot} primeiras questões do MedQA nos três arranjos e custou US$ 0,84. O custo por questão já é confiável com essa amostra; a acurácia ainda não.</p>
  <p>O anel gastou de 24% a 28% menos que a topologia completa, perto dos 30% a 40% de Li et al. O anel com Conformist saiu mais barato que o Self-Consistency, porque a rodada 0 foi unânime em {unan} das {n_pilot} questões, e nesses casos ninguém debate.</p></div></div>
  <figure class="figure">
    <div class="figure-title">Custo medido por questão (US$ por 1.000 questões)</div>
    <div class="table-wrap"><table>
      <thead><tr><th>Variação</th>{cost_head}</tr></thead>
      <tbody>{cost_rows}</tbody>
    </table></div>
    <p class="figure-note">Inclui a escolha dos especialistas e a rodada 0, que todas as variações usam. Referências: Self-Consistency e MedAgents nas mesmas {n_pilot} questões; no arranjo misto aparecem os dois modelos (4o-mini / nano).</p>
  </figure>
  <div class="multiples">
    <figure class="figure">
      <div class="figure-title">Acertos no piloto</div>
      <div class="table-wrap"><table>
        <thead><tr><th>Variação</th>{cost_head}</tr></thead>
        <tbody>{acc_rows}</tbody>
      </table></div>
      <p class="figure-note">Com {n_pilot} questões, uma questão vale 5 pontos e o intervalo de confiança passa de ±15 pontos. Nada aqui é diferença real ainda.</p>
    </figure>
    <figure class="figure">
      <div class="figure-title">Acurácia média de um agente, rodada a rodada</div>
      <div class="table-wrap"><table>
        <thead><tr><th>Arranjo</th><th>Topologia</th><th class="n">R0</th><th class="n">R1</th><th class="n">R2</th></tr></thead>
        <tbody>{traj_rows}</tbody>
      </table></div>
      <p class="figure-note">É o teste do <i>Debate or Vote</i>: se o debate for um martingale, a média fica parada. No piloto ela subiu um pouco em quase todos os arranjos, mais no misto; com {n_pilot} questões isso ainda é indício, não conclusão.</p>
    </figure>
  </div>
</section>

<section aria-labelledby="h-next">
  <div class="section-head"><span class="label">Próximo passo</span><h2 id="h-next">A execução completa</h2>
  <div class="prose"><p>Rodar as três topologias juntas nas 1.273 questões custa, pelo piloto, cerca de {'; '.join(f'US$ {num(v, 0)} ({arr_names[a]})' for a, v in full_est.items())}, ou US$ {num(est_total, 0)} no total. As três execuções podem rodar ao mesmo tempo; a mais lenta, a do gpt-4o-mini, leva umas três horas com 24 questões em paralelo. As questões do piloto já contam.</p>
  <p>Com a execução completa, a pergunta passa a ter resposta: alguma topologia de debate entre especialistas supera o Self-Consistency, e o anel com Conformist mantém a acurácia gastando menos?</p></div></div>
<pre><code>.\\.venv\\Scripts\\python.exe run.py --method persona_debate --model_name gpt-4o-mini --cot_max_tokens 1000 --workers 24 --output_files_folder outputs\\MedQA\\gpt-4o-mini
.\\.venv\\Scripts\\python.exe run.py --method persona_debate --model_name gpt-5.4-nano --cot_max_tokens 1000 --workers 24 --output_files_folder outputs\\MedQA\\gpt-5.4-nano
.\\.venv\\Scripts\\python.exe run.py --method persona_debate --model_name gpt-4o-mini --agent_models gpt-4o-mini,gpt-5.4-nano --cot_max_tokens 1000 --workers 24 --output_files_folder outputs\\MedQA\\hetero
.\\.venv\\Scripts\\python.exe analyze_debate.py</code></pre>
</section>

<footer>Dados: MedQA (USMLE), conjunto de teste, 1.273 questões · Artigos: Li et al. (EMNLP Findings 2024), Choi et al. (NeurIPS 2025), Zhang et al. (arXiv 2502.08788), Tang et al. (ACL Findings 2024), Zhao et al. (arXiv 2502.04413), Sun et al. (arXiv 2507.13396) · Novembro de 2026</footer>
</div>
"""
    out = os.path.join(HERE, 'topologias.html')
    open(out, 'w', encoding='utf-8').write(html)
    print(f'Gerado {out} ({len(html) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
