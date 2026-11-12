"""Gera report/medvote.html (proposta + piloto do MedVote) a partir de outputs/medvote/.

  .\.venv\Scripts\python.exe analyze_medvote.py
  .\.venv\Scripts\python.exe report\build_medvote.py
"""
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import analyze  # noqa: E402
import analyze_medvote as am  # noqa: E402
sys.path.insert(0, HERE)
import medvote_test  # noqa: E402

SPLIT = 'validation'
CONFIG = {'nano': 11, 'mini': 10, 'threshold': 0.9, 'variant': 'books_cases', 'voters': (am.NANO,)}
CASE_QID = 552


def pct(x, d=1):
    return f'{x * 100:.{d}f}'.replace('.', ',') + '%'


def num(x, d=2):
    return f'{x:.{d}f}'.replace('.', ',')


def pfmt(p):
    return 'p < 0,001' if p < 0.001 else 'p = ' + f'{p:.2f}'.replace('.', ',')


def load(name):
    return {r['qid']: r for r in map(json.loads, open(os.path.join('outputs', 'medvote', f'{SPLIT}-{name}.jsonl'),
                                                         encoding='utf-8'))}


def main():
    votes, diffs = load('vote'), load('diff')
    res = json.load(open(f'analysis/medvote_{SPLIT}.json', encoding='utf-8'))
    qids = sorted(votes)
    N = len(qids)
    gold = {q: votes[q]['gold'] for q in qids}

    # configuração fixada: primeiras amostras de cada modelo, sem sorteio
    sc_ok, s1_ok, sys_ok, esc, cost = {}, {}, {}, 0, 0.0
    for q in qids:
        v = votes[q]
        sc_ok[q] = am.majority(v['samples'][am.NANO]) == gold[q]
        sm = v['samples'][am.NANO][:CONFIG['nano']] + v['samples'][am.MINI][:CONFIG['mini']]
        a = am.majority(sm)
        s1_ok[q] = a == gold[q]
        cost += am.stage1_cost(v, CONFIG['nano'], CONFIG['mini'])
        if am.agreement(sm) < CONFIG['threshold'] and q in diffs:
            esc += 1
            d = diffs[q]['variants'][CONFIG['variant']]
            a = am.majority([x for m in CONFIG['voters'] for x in d['answers'][m]])
            cost += am.usd(diffs[q]['planning_usage']) + am.usd({m: d['usage'][m] for m in CONFIG['voters']})
        sys_ok[q] = a == gold[q]
    vlabel_ = {'none': 'Sem evidência', 'books': 'Livros-texto', 'books_cases': 'Livros + questões resolvidas'}
    sc_cost = sum(am.stage1_cost(votes[q], 21, 0) for q in qids) / N
    k_sc, k_s1, k_sys = sum(sc_ok.values()), sum(s1_ok.values()), sum(sys_ok.values())
    x1 = sum(sys_ok[q] and not sc_ok[q] for q in qids)
    y1 = sum(sc_ok[q] and not sys_ok[q] for q in qids)
    p1 = analyze.mcnemar_exact(x1, y1)
    total_cost = (sum(am.usd(r['usage']) for r in votes.values()) +
                  sum(am.usd(r['planning_usage']) + sum(am.usd(v['usage']) for v in r['variants'].values())
                      for r in diffs.values()))
    lo_sys, hi_sys = analyze.wilson(k_sys, N)

    # ---------------- teste (configuração fixada, rodado uma vez)
    T = medvote_test.results(CONFIG)
    tres = json.load(open('analysis/medvote_test.json', encoding='utf-8'))
    TN, TK = T['n'], T['k']
    tr = {(r['label'], r['model']): r for r in T['rows']}
    sc_t = tr[('Self-Consistency (parte 2)', am.NANO)]
    ma_t = tr[('MedAgents', am.NANO)]
    s1_t = tr[('Etapa 1 sozinha, 11 + 10 amostras', 'misto')]
    mlab = {am.NANO: 'gpt-5.4-nano', am.MINI: 'gpt-4o-mini', 'misto': 'nano + 4o-mini'}
    trows = (f'<tr class="is-mc"><td>MedVote</td><td>nano + 4o-mini</td><td class="n">{TK}</td>'
             f'<td class="n pct">{pct(TK / TN)}</td><td class="n">US$ {num(T["usd"] * 1000)}</td><td class="n">—</td><td class="n">—</td></tr>')
    for key in [('Etapa 1 sozinha, 11 + 10 amostras', 'misto'), ('Self-Consistency (parte 2)', am.NANO),
                ('MedAgents', am.NANO), ('Self-Consistency (parte 2)', am.MINI), ('MedAgents', am.MINI)]:
        r = tr[key]
        name = {'Self-Consistency (parte 2)': 'Self-Consistency, 21 amostras', 'MedAgents': 'MedAgents'}.get(r['label'], 'Etapa 1 sozinha')
        cls = ' class="is-sc"' if r['label'].startswith('Self') and r['model'] == am.NANO else ''
        trows += (f'<tr{cls}><td>{name}</td><td>{mlab[r["model"]]}</td><td class="n">{r["k"]}</td>'
                  f'<td class="n pct">{pct(r["k"] / TN)}</td><td class="n">US$ {num(r["usd"] * 1000)}</td>'
                  f'<td class="n">+{r["fixed"]} / −{r["broke"]}</td><td class="n">{pfmt(r["p"])}</td></tr>')
    s2t = tres['stage2']
    s2t_rows = ''
    for r in s2t['rows']:
        hl = ' class="hl"' if (r['variant'], r['who']) == ('books_cases', '3 nano') else ''
        s2t_rows += (f'<tr{hl}><td>{vlabel_[r["variant"]]}</td><td>{r["who"]}</td><td class="n">{r["correct"]} de {r["n"]}</td>'
                     f'<td class="n">+{r["fixed"]} / −{r["broke"]}</td><td class="n">{pfmt(r["p"])}</td></tr>')
    s1_val = {(r['nano'], r['mini']): r['acc'] for r in res['stage1']}
    s1_tst = {(r['nano'], r['mini']): r['acc'] for r in tres['stage1']}
    vt_rows = ''.join(f'<tr><td>{lab}</td><td class="n">{pct(v)}</td><td class="n">{pct(t)}</td></tr>' for lab, v, t in [
        ('Self-Consistency, 21 amostras do nano', s1_val[(21, 0)], s1_tst[(21, 0)]),
        ('Self-Consistency, 21 amostras do 4o-mini', s1_val[(0, 21)], s1_tst[(0, 21)]),
        ('Etapa 1: 11 do nano + 10 do 4o-mini', s1_val[(11, 10)], s1_tst[(11, 10)]),
        ('MedVote (configuração fixada)', k_sys / N, TK / TN)])
    tlo, thi = T["ci"]
    TNs = f"{TN:,}".replace(",", ".")

    # placar
    board = [('Self-Consistency, 21 amostras do nano', k_sc, sc_cost, 'is-sc'),
             ('Etapa 1 sozinha: 11 do nano + 10 do 4o-mini', k_s1, sum(am.stage1_cost(votes[q], 11, 10) for q in qids) / N, ''),
             ('MedVote: etapa 1 + diferencial com livros e casos', k_sys, cost / N, 'is-mc')]
    board_rows = ''.join(
        f'<tr class="{c}"><td>{n}</td><td class="n">{k} de {N}</td><td class="n pct">{pct(k / N)}</td>'
        f'<td class="n">US$ {num(u * 1000)}</td></tr>' for n, k, u, c in board)

    s1_rows = ''.join(f'<tr{" class=hl" if (r["nano"], r["mini"]) == (11, 10) else ""}><td>{r["label"]}</td>'
                      f'<td class="n">{pct(r["acc"])}</td><td class="n">{pct(r["ci"][0])} a {pct(r["ci"][1])}</td>'
                      f'<td class="n">US$ {num(r["usd"] * 1000)}</td></tr>' for r in res['stage1'])
    vlabel = vlabel_
    _unused = {'none': 'Sem evidência', 'books': 'Livros-texto', 'books_cases': 'Livros + questões resolvidas'}
    s2 = res['stage2']
    s2_rows = ''
    for r in s2['rows']:
        hl = ' class="hl"' if (r['variant'], r['who']) == ('books_cases', '3 nano') else ''
        s2_rows += (f'<tr{hl}><td>{vlabel[r["variant"]]}</td><td>{r["who"]}</td><td class="n">{r["correct"]} de {r["n"]}</td>'
                    f'<td class="n">+{r["fixed"]} / −{r["broke"]}</td><td class="n">{pfmt(r["p"])}</td></tr>')

    # robustez: mesma etapa 1, todas as variantes, limiares e votantes (com sorteios)
    def sim(v, t, models):
        accs = []
        for d in range(30):
            rng = random.Random(d)
            hit = 0
            for q in qids:
                sm = am.draw(votes[q], 11, 10, rng)
                a = am.majority(sm)
                if am.agreement(sm) < t and q in diffs:
                    a = am.majority([x for m in models for x in diffs[q]['variants'][v]['answers'][m]])
                hit += a == gold[q]
            accs.append(hit / N)
        return sum(accs) / len(accs)
    rob_rows = ''
    for models, lab in [((am.NANO, am.MINI), '3 nano + 3 4o-mini'), ((am.NANO,), '3 nano')]:
        for v in ('none', 'books', 'books_cases'):
            rob_rows += f'<tr><td>{vlabel[v]}</td><td>{lab}</td>' + ''.join(
                f'<td class="n">{pct(sim(v, t, models))}</td>' for t in (0.7, 0.8, 0.9)) + '</tr>'

    case = diffs[CASE_QID]
    cv = votes[CASE_QID]
    case_none = case['variants']['none']['answers']
    case_bc = case['variants']['books_cases']['answers']

    css = open(os.path.join(HERE, 'topologias.css'), encoding='utf-8').read()
    css += """
.flow { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem; }
@media (max-width: 860px) { .flow { grid-template-columns: 1fr; } }
.stage { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 1.25rem; display: flex; flex-direction: column; gap: 0.6rem; min-width: 0; }
.stage.gate { border-color: var(--other); }
.stage .label { color: var(--ink-2); }
.stage p { font-size: 0.9375rem; line-height: 1.5; }
.stage .src { font-family: var(--f-data); font-size: 0.75rem; color: var(--muted); }
.board td.pct { font-family: var(--f-data); font-size: var(--step-1); font-weight: 600; }
.case { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 1.5rem; display: flex; flex-direction: column; gap: 1rem; max-width: var(--col); }
.case ul { margin: 0; padding-left: 1.1rem; display: flex; flex-direction: column; gap: 0.35rem; }
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.chip { font-family: var(--f-data); font-size: 0.8125rem; padding: 0.15rem 0.5rem; border-radius: 3px; border: 1px solid var(--rule); }
.chip.ok { background: var(--accent-soft); border-color: var(--accent); }
.chip.bad { background: var(--no-soft); border-color: var(--no); }
blockquote { margin: 0; border-left: 3px solid var(--rule); padding-left: 1rem; color: var(--ink-2); font-size: 0.9375rem; }
"""

    def chips(ans):
        return ''.join(f'<span class="chip {"ok" if a == case["gold"] else "bad"}">{a or "?"}</span>' for a in ans)

    html = f"""<title>MedVote</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@500;600;700;800&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{css}</style>
<div class="page">
<header>
  <span class="label">MedQA · quarta parte · sistema próprio · validação e teste</span>
  <h1>MedVote</h1>
  <p class="lede">Um sistema multiagentes montado a partir do que os artigos e as três partes anteriores mostraram: o voto faz quase todo o trabalho, modelos diferentes se complementam e o que falta é conhecimento. Ele vota primeiro e só chama agentes com livros-texto nas questões em que o voto hesita.</p>
  <p class="verdict">Nas {TNs} questões de teste do MedQA, rodadas uma vez só com a configuração escolhida na validação, o MedVote acertou {pct(TK / TN)}. O Self-Consistency do gpt-5.4-nano acertou {pct(sc_t['k'] / TN)}: o MedVote acertou {sc_t['fixed']} questões que ele errou e errou {sc_t['broke']} que ele acertou ({pfmt(sc_t['p'])}). O MedAgents do mesmo modelo acertou {pct(ma_t['k'] / TN)}. O MedVote custou {pct(T['usd'] / sc_t['usd'] - 1, 0)} a mais que o Self-Consistency e {pct(1 - T['usd'] / ma_t['usd'], 0)} a menos que o MedAgents.</p>
  <p class="prev">Partes anteriores: <a href="https://claude.ai/artifact/2bcRN8TtUwtG4CGsGp6LB9">MedAgents no MedQA</a> · <a href="https://claude.ai/artifact/Gegp5JWH5TBL6Z18FU3F5e">MedAgents contra Self-Consistency</a> · <a href="https://claude.ai/artifact/V49Ajx37SBxt2mife5kh3m">Topologias do MedAgents</a> · <a href="https://claude.ai/artifact/JZfAaMwbtudi2CwiZvh5nu">Pacote de reprodução</a></p>
</header>

<section aria-labelledby="h-test">
  <div class="section-head"><span class="label">Resultado no teste</span><h2 id="h-test">{TNs} questões, configuração fixada antes de rodar</h2>
  <div class="prose"><p>Mesmas questões das partes anteriores, pareadas uma a uma. A coluna “MedVote consertou / estragou” conta as questões que o MedVote acertou e o outro sistema errou, e o contrário.</p></div></div>
  <div class="table-wrap board"><table>
    <thead><tr><th>Sistema</th><th>Modelo</th><th class="n">Acertos</th><th class="n">Acurácia</th><th class="n">US$ por 1.000 q</th><th class="n">MedVote consertou / estragou</th><th class="n">McNemar</th></tr></thead>
    <tbody>{trows}</tbody>
  </table></div>
  <p class="figure-note">Intervalo de 95% do MedVote: {pct(tlo)} a {pct(thi)}. {T['escalated']} questões chegaram à etapa 2. Self-Consistency e MedAgents: previsões e custos medidos nas partes 2 e 3. As 21 amostras novas de cada modelo gravadas pelo MedVote deram {pct(s1_tst[(21, 0)])} (nano) e {pct(s1_tst[(0, 21)])} (4o-mini), praticamente iguais. Sem as {T['near']} questões do teste que têm quase-cópia no treino, o MedVote acerta {pct(T['clean_k'] / T['clean_n'])} e o Self-Consistency do nano, {pct(T['clean_sc'] / T['clean_n'])}.</p>
</section>

<section aria-labelledby="h-design">
  <div class="section-head"><span class="label">Desenho</span><h2 id="h-design">Votar barato, consultar só onde há dúvida</h2>
  <div class="prose"><p>Cada peça vem de um achado concreto, dos artigos ou das partes anteriores.</p></div></div>
  <div class="flow">
    <div class="stage"><span class="label">Etapa 1 · todas as questões</span><h3>Voto heterogêneo</h3>
      <p>21 raciocínios independentes, misturando gpt-5.4-nano e gpt-4o-mini, temperatura 0,7. A letra mais votada é a resposta.</p>
      <span class="src">Choi et al.: o voto explica quase todo o ganho do debate. Zhang et al. e parte 3: modelos diferentes erram em questões diferentes.</span></div>
    <div class="stage gate"><span class="label">Portão</span><h3>Concordância abaixo de 90%?</h3>
      <p>Se 90% ou mais dos votos foram para a mesma letra, a resposta sai da etapa 1. No teste, {pct(1 - T["escalated"] / TN, 0)} das questões pararam aqui.</p>
      <span class="src">Parte 2: a concordância do Self-Consistency aponta as questões difíceis. Survey (MDAgents): gastar mais só onde é preciso.</span></div>
    <div class="stage"><span class="label">Etapa 2 · questões divididas</span><h3>Diagnóstico diferencial com evidência</h3>
      <p>As duas letras mais votadas viram candidatas. Um agente extrai os achados e escreve três buscas; o BM25 traz trechos dos 18 livros do MedQA e três questões resolvidas parecidas do treino. Três agentes comparam as candidatas com a evidência e votam.</p>
      <span class="src">MedAgents: 77% dos erros são de conhecimento. MedRAG: decompor as manifestações e buscar as diferenças entre diagnósticos parecidos.</span></div>
  </div>
</section>

<section aria-labelledby="h-held">
  <div class="section-head"><span class="label">Validação contra teste</span><h2 id="h-held">O que se manteve e o que não</h2>
  <div class="prose"><p>O MedVote caiu de {pct(k_sys / N)} na validação para {pct(TK / TN)} no teste. Parte da queda era esperada: a configuração foi escolhida olhando as questões da validação, e com 300 questões o intervalo é largo.</p>
  <p><b>Não se manteve:</b> o ganho de misturar os modelos no voto. Na validação, metade de cada modelo subia cerca de 3 pontos; no teste, a mistura empata com o nano sozinho ({pct(s1_tst[(11, 10)])} contra {pct(s1_tst[(21, 0)])}). O 4o-mini é bem mais fraco no teste, e o achado do piloto parece ter sido acaso.</p>
  <p><b>Manteve-se:</b> a etapa 2. Sobre a resposta da etapa 1, ela conserta {s1_t['fixed']} questões e estraga {s1_t['broke']}, cerca de 5 pontos, como na validação. Todo o ganho do MedVote no teste vem daqui.</p></div></div>
  <figure class="figure"><div class="figure-title">Acurácia na validação ({N} questões) e no teste ({TNs})</div>
    <div class="table-wrap"><table><thead><tr><th>Sistema</th><th class="n">Validação</th><th class="n">Teste</th></tr></thead><tbody>{vt_rows}</tbody></table></div>
    <p class="figure-note">Nas misturas, média de 50 sorteios das amostras; no MedVote, as primeiras amostras de cada modelo, sem sorteio.</p></figure>
</section>

<section aria-labelledby="h-s2t">
  <div class="section-head"><span class="label">Etapa 2 no teste</span><h2 id="h-s2t">Os livros consertam; o 4o-mini continua sem aproveitar</h2>
  <div class="prose"><p>No teste, {s2t['n']} questões ficaram abaixo de 90% de concordância nas 42 amostras. O voto da etapa 1 acertou {s2t['pooled_correct']} delas; a resposta certa estava entre as duas candidatas em {s2t['gold_in_candidates']} ({pct(s2t['gold_in_candidates'] / s2t['n'], 0)}).</p>
  <p>O padrão da validação se repete. Com os livros, o nano conserta mais de duas questões para cada uma que estraga. O 4o-mini, com a mesma evidência, conserta e estraga quase o mesmo número de questões. Duas diferenças em relação à validação: restringir às duas candidatas já ajuda sem evidência, e as questões resolvidas do treino não acrescentam nada aos livros.</p></div></div>
  <figure class="figure"><div class="figure-title">Acertos nas questões divididas do teste, comparados ao voto da etapa 1</div>
    <div class="table-wrap"><table><thead><tr><th>Evidência</th><th>Quem vota</th><th class="n">Acertos</th><th class="n">Consertou / estragou</th><th class="n">McNemar</th></tr></thead><tbody>{s2t_rows}</tbody></table></div>
    <p class="figure-note">Todas as variantes foram gravadas para esta comparação, mas a configuração do placar foi fixada antes: livros e questões resolvidas, votada por 3 amostras do nano. A busca de questões resolvidas descartou {T['dropped']} quase-cópias do treino.</p></figure>
</section>

<section aria-labelledby="h-score">
  <div class="section-head"><span class="label">Validação · piloto</span><h2 id="h-score">Como a configuração foi escolhida: {N} questões sorteadas da validação</h2></div>
  <div class="table-wrap board"><table>
    <thead><tr><th>Sistema</th><th class="n">Acertos</th><th class="n">Acurácia</th><th class="n">US$ por 1.000 q</th></tr></thead>
    <tbody>{board_rows}</tbody>
  </table></div>
  <p class="figure-note">Intervalo de 95% do MedVote: {pct(lo_sys)} a {pct(hi_sys)}. Configuração fixada: 11 amostras do nano e 10 do 4o-mini; abaixo de 90% de concordância, diferencial com livros e questões resolvidas, votado por 3 amostras do nano ({esc} questões chegaram à etapa 2). Usa as primeiras amostras gravadas de cada modelo, sem sorteio.</p>
</section>

<section aria-labelledby="h-s1">
  <div class="section-head"><span class="label">Validação · etapa 1</span><h2 id="h-s1">Na validação, misturar modelos parecia render</h2>
  <div class="prose"><p>Com 21 amostras, os dois modelos sozinhos ficam em torno de 80%. Metade de cada um subia para cerca de 83%, pelo mesmo preço. Esse ganho não se repetiu no teste.</p></div></div>
  <figure class="figure"><div class="figure-title">Voto majoritário na etapa 1</div>
    <div class="table-wrap"><table><thead><tr><th>Amostras no voto</th><th class="n">Acurácia</th><th class="n">IC 95%</th><th class="n">US$ por 1.000 q</th></tr></thead><tbody>{s1_rows}</tbody></table></div>
    <p class="figure-note">Misturas: média de 50 sorteios entre as 21 amostras gravadas de cada modelo.</p></figure>
</section>

<section aria-labelledby="h-s2">
  <div class="section-head"><span class="label">Validação · etapa 2</span><h2 id="h-s2">A evidência dos livros é o que conserta</h2>
  <div class="prose"><p>Nas {s2['n']} questões em que as 42 amostras da etapa 1 se dividiram, o voto acertou {s2['pooled_correct']}. A resposta certa estava entre as duas candidatas em {s2['gold_in_candidates']} delas ({pct(s2['gold_in_candidates'] / s2['n'], 0)}), o teto da etapa 2.</p>
  <p>Só restringir a escolha a duas candidatas, sem evidência, muda pouco. Com os trechos dos livros, o nano conserta 20 questões e estraga 7; com livros e questões resolvidas, conserta 22 e estraga 4. O gpt-4o-mini, com a mesma evidência, estraga mais do que conserta em todas as variantes: ele tende a manter a resposta que já tinha. Por isso, na configuração fixada, só o nano vota na etapa 2.</p></div></div>
  <figure class="figure"><div class="figure-title">Acertos nas questões divididas, comparados ao voto da etapa 1</div>
    <div class="table-wrap"><table><thead><tr><th>Evidência</th><th>Quem vota</th><th class="n">Acertos</th><th class="n">Consertou / estragou</th><th class="n">McNemar</th></tr></thead><tbody>{s2_rows}</tbody></table></div>
    <p class="figure-note">Mesmas questões, mesmas candidatas e mesma busca para as três variantes; muda só o que entra no prompt.</p></figure>
</section>

<section aria-labelledby="h-rob">
  <div class="section-head"><span class="label">Validação · robustez</span><h2 id="h-rob">O ganho não depende de um ajuste fino</h2>
  <div class="prose"><p>Escolher a melhor combinação olhando os mesmos dados infla o resultado. A tabela mostra que todas as combinações com evidência ficam acima de 85%, para qualquer limiar entre 70% e 90%, enquanto o Self-Consistency do nano fica em {pct(k_sc / N)}.</p></div></div>
  <figure class="figure"><div class="figure-title">Sistema completo com a etapa 1 de 11 + 10 amostras, por limiar da etapa 2</div>
    <div class="table-wrap"><table><thead><tr><th>Evidência</th><th>Quem vota na etapa 2</th><th class="n">70%</th><th class="n">80%</th><th class="n">90%</th></tr></thead><tbody>{rob_rows}</tbody></table></div>
    <p class="figure-note">Média de 30 sorteios das amostras da etapa 1. A etapa 2 só foi rodada nas questões em que as 42 amostras ficaram abaixo de 90%; fora delas, vale a resposta da etapa 1.</p></figure>
</section>

<section aria-labelledby="h-case">
  <div class="section-head"><span class="label">Caso real da validação</span><h2 id="h-case">Discinesia ciliar ou fibrose cística</h2></div>
  <div class="case">
    <p class="stem">Homem de 31 anos com tosse crônica, sinusite recorrente desde a infância e estertores nos dois pulmões. Ele e a esposa tentam engravidar há 5 anos sem sucesso. Qual estrutura molecular tem o defeito mais provável? Candidatas da etapa 1: <b>(C) proteína CFTR</b> e <b>(D) braço de dineína</b>. Gabarito: D.</p>
    <ul>
      <li><b>Etapa 1:</b> só {cv['samples'][am.NANO].count('D')} das 21 amostras do nano e {cv['samples'][am.MINI].count('D')} das 21 do 4o-mini escolheram D. O voto foi para C.</li>
      <li><b>Buscas escritas pelo agente:</b> “dynein arm defect primary ciliary dyskinesia infertility situs inversus chronic sinusitis bronchiectasis”, entre outras.</li>
      <li><b>Trechos recuperados:</b> Nelson (síndrome de Kartagener; homens inférteis por espermatozoides imóveis), Ross (cílios e flagelos compartilham o arranjo 9+2), First Aid (dineína axonemal) e Novak (CFTR e ausência congênita de ducto deferente).</li>
    </ul>
    <div><span class="label">Sem evidência</span><div class="chips">{chips(case_none[am.NANO])}<span class="chip">nano</span> {chips(case_none[am.MINI])}<span class="chip">4o-mini</span></div></div>
    <div><span class="label">Com livros e questões resolvidas</span><div class="chips">{chips(case_bc[am.NANO])}<span class="chip">nano</span> {chips(case_bc[am.MINI])}<span class="chip">4o-mini</span></div></div>
    <blockquote>“Fibrose cística causa infertilidade masculina por ausência bilateral congênita do ducto deferente, […] mas a sinusite recorrente desde a infância e os sintomas ciliares apontam para discinesia ciliar primária: o mesmo axonema 9+2 faz os espermatozoides imóveis.” — raciocínio do nano com a evidência (resumo traduzido)</blockquote>
    <p>A evidência fez o nano trocar a resposta nas três amostras. O 4o-mini recebeu os mesmos trechos e manteve C nas três.</p>
  </div>
</section>

<section aria-labelledby="h-care">
  <div class="section-head"><span class="label">Cuidados</span><h2 id="h-care">O que foi verificado e o que falta</h2></div>
  <div class="prose"><ul>
    <li><b>Vazamento pelo treino:</b> 25 questões da validação e 40 do teste têm uma quase-cópia no treino (Jaccard ≥ 0,6). A busca de questões resolvidas descarta essas quase-cópias; foram descartadas 6 no piloto e {T['dropped']} no teste. Sem as {T['near']} questões do teste com quase-cópia, o MedVote fica em {pct(T['clean_k'] / T['clean_n'])}.</li>
    <li><b>Vazamento pelos livros:</b> o trecho recuperado mais parecido com qualquer questão do piloto tem Jaccard de 0,19. Os livros são o corpus oficial que acompanha o MedQA.</li>
    <li><b>Seleção na validação:</b> mistura, limiar, variante e votantes foram escolhidos nas {N} questões da validação. O teste foi rodado uma vez, com essa configuração fixada; nenhuma escolha foi refeita depois de ver o teste.</li>
    <li><b>O que o teste sugere, sem ter sido testado:</b> como a mistura não ajudou e as questões resolvidas não acrescentaram, uma versão mais simples (só o nano na etapa 1 e só os livros na etapa 2) deve render o mesmo. Confirmar isso exigiria outro conjunto de questões, porque o teste já foi usado.</li>
    <li><b>Modelos pequenos:</b> tudo foi feito com gpt-5.4-nano sem raciocínio e gpt-4o-mini. O ganho com modelos maiores não foi medido.</li>
  </ul></div>
</section>

<section aria-labelledby="h-next">
  <div class="section-head"><span class="label">Custo e reprodução</span><h2 id="h-next">Quanto custou e como repetir</h2>
  <div class="prose"><p>O piloto na validação custou US$ {num(total_cost)} e o teste, US$ {num(T['total_usd'])}, gravando tudo: 21 amostras de cada modelo e as três variantes da etapa 2 nos dois modelos. Só a configuração fixada teria custado US$ {num(T['usd'] * TN)} no teste.</p></div></div>
<pre><code>.\\.venv\\Scripts\\python.exe medvote.py vote --split test --limit 0 --workers 16
.\\.venv\\Scripts\\python.exe medvote.py diff --split test --limit 0 --workers 12
.\\.venv\\Scripts\\python.exe analyze_medvote.py --split test</code></pre>
</section>

<footer>Dados: MedQA (Jin et al., 2021), teste, validação e treino do conjunto original em inglês com 5 alternativas; livros-texto do MedQA em trechos (MedRAG/textbooks, Xiong et al., 2024) · Recuperação: BM25 (bm25s) · Modelos: gpt-5.4-nano sem raciocínio e gpt-4o-mini · Código: medvote.py, retrieval.py, analyze_medvote.py, report/medvote_test.py · Novembro de 2026</footer>
</div>
"""
    out = os.path.join(HERE, 'medvote.html')
    open(out, 'w', encoding='utf-8').write(html)
    print(f'Gerado {out} ({len(html) / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
