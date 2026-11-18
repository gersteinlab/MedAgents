"""MedVote: voto heterogêneo + diagnóstico diferencial com recuperação só nas questões divididas.

Desenho (ver report/medvote.html):
  Etapa 1 (vote)  Amostras independentes de CoT de dois modelos (Self-Consistency heterogêneo).
                  Choi et al. 2025: o voto explica quase todo o ganho dos sistemas multiagentes.
                  Zhang et al. 2025: misturar modelos ajuda.
  Etapa 2 (diff)  Só onde o voto se divide. As duas alternativas mais votadas viram candidatas;
                  um agente extrai os achados e escreve buscas (decomposição de manifestações, MedRAG);
                  recuperamos trechos de livros-texto e questões resolvidas parecidas; agentes dos dois
                  modelos comparam as candidatas com a evidência (diferenças diagnósticas, MedRAG) e votam.
                  Três variantes, para separar efeitos: sem evidência / livros / livros + casos.

Tudo é gravado (respostas de cada amostra, trechos recuperados, custo por parte), para que limiares,
misturas e variantes sejam escolhidos depois, no conjunto de validação, sem chamadas novas.

  python medvote.py vote --split validation --limit 300
  python medvote.py diff --split validation --limit 300

Ablação da arquitetura em duas etapas:
  python medvote.py rag  --split test --limit 0 --models gpt-5.4-nano --per_model 7
      etapa única: busca e resposta direta entre as 5 alternativas, sem voto prévio nem candidatas
  python medvote.py diff --split test --limit 0 --models gpt-5.4-nano --variants books_cases \
      --min_agreement 0.9 --trigger 1.01 --tag=-semportao
      etapa 2 também nas questões que o portão deixava de fora (concordância >= 90%, não unânimes)

Etapa 2 com fichas de diferenças (MedRAG, Zhao et al.):
  python medvote.py ddx --split validation --models gpt-5.4-nano --tag=-2c
      busca separada para cada candidata; um agente monta, só com os livros e a pergunta final (sem o caso),
      a ficha de cada candidata: o que ela exige, o que a confirma, o que a descarta e o que a separa das outras;
      três agentes conferem os achados do caso contra as fichas e votam
  --max_candidates 3 --third_min 0.1: inclui a 3ª mais votada quando ela tem pelo menos 10% dos votos

Conjunto de confirmação (as questões da validação fora do piloto de 300), em pasta própria:
  MEDVOTE_OUT=outputs/medvote/confirmacao python medvote.py vote --split validation --rest_of 300 --samples 11,10

Fichas com o peso do voto (--prior): o agente sabe quantos votos cada candidata teve e só troca a majoritária
diante de um achado do caso que contradiga a ficha dela.
  python medvote.py ddx --split test --models gpt-5.4-nano --prior --tag=-prior

MMLU médico (6 matérias, 4 alternativas; saídas em outputs/medvote/mmlu): --dataset mmlu. Ver scripts/mmlu_medvote.sh.
"""
import argparse
import json
import os
import random
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import tqdm

from api_utils import api_handler, FATAL_ERRORS
from data_utils import cleansing_final_output
from prompt_generator import get_cot_prompt
from run import load_dotenv

OUT_DIR = os.environ.get('MEDVOTE_OUT', os.path.join('outputs', 'medvote'))
DATASET = 'medqa'
VARIANTS = ('none', 'books', 'books_cases')


def majority(answers):
    valid = [a for a in answers if a]
    if not valid:
        return ''
    c = Counter(valid)
    best = max(c.values())
    return next(a for a in valid if c[a] == best)


def ranked(answers):
    c = Counter(a for a in answers if a)
    return [a for a, _ in c.most_common()]


_ANS = re.compile(r'Answer\s*[:：]\s*\**\s*\(?([A-E])\b')


def parse_answer(text):
    """Última ocorrência de 'Answer: X'; se não houver, o mesmo extrator do CoT (data_utils)."""
    if not text or text == 'ERROR.':
        return ''
    m = _ANS.findall(text)
    return m[-1] if m else cleansing_final_output(text)[0]


def snapshot(handlers):
    return {m: h.usage() for m, h in handlers.items()}


def diff_usage(after, before):
    return {m: {k: after[m].get(k, 0) - before.get(m, {}).get(k, 0) for k in after[m]} for m in after}


def subset(split, limit, seed, rest_of=0):
    from retrieval import load_split
    qs = load_split(split, DATASET)
    if rest_of:  # questões fora do sorteio de rest_of (o piloto), para confirmar sem reusar dados
        used = {q['qid'] for q in subset(split, rest_of, seed)}
        return [q for q in qs if q['qid'] not in used]
    if limit and limit < len(qs):
        rng = random.Random(seed)
        qs = sorted(rng.sample(qs, limit), key=lambda q: q['qid'])
    return qs


def done_qids(path):
    if not os.path.exists(path):
        return set()
    return {json.loads(l)['qid'] for l in open(path, encoding='utf-8') if l.strip()}


def run_pool(items, fn, path, workers, desc):
    lock = threading.Lock()
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(fn, it) for it in items]
        for fut in tqdm.tqdm(as_completed(futs), total=len(futs), desc=desc):
            try:
                rec = fut.result()
            except FATAL_ERRORS:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as e:  # questão fica sem registro e é refeita na próxima execução
                failed += 1
                print(f'Falha: {e!r}')
                continue
            with lock, open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    if failed:
        print(f'{failed} questões falharam. Rode o mesmo comando de novo para completá-las.')


# ------------------------------------------------------------------ etapa 1
def cmd_vote(args, handlers):
    path = os.path.join(OUT_DIR, f'{args.split}-vote.jsonl')
    done = done_qids(path)
    todo = [q for q in subset(args.split, args.limit, args.seed, args.rest_of) if q['qid'] not in done]
    print(f'{path}: já feitas {len(done)}, faltam {len(todo)}')
    n_per = dict(zip(handlers, args.samples))

    def solve(q):
        for h in handlers.values():
            h.reset_usage()
        prompt = get_cot_prompt(q['question'], q['options'])
        samples, outputs = {}, {}
        for m, h in handlers.items():
            outs = h.get_samples(prompt, n=n_per[m], max_tokens=args.max_tokens, temperature=args.temperature)
            outputs[m] = outs
            samples[m] = [cleansing_final_output(o)[0] if o != 'ERROR.' else '' for o in outs]
        if not any(a for s in samples.values() for a in s):
            raise RuntimeError(f"questão {q['qid']}: nenhuma amostra válida")
        return {'qid': q['qid'], 'split': args.split, 'step': q['step'], 'gold': q['answer_idx'],
                'options': q['options'], 'samples': samples, 'outputs': outputs,
                'usage': snapshot(handlers), 'temperature': args.temperature, 'max_tokens': args.max_tokens}

    run_pool(todo, solve, path, args.workers, 'etapa 1')


# ------------------------------------------------------------------ etapa 2
QUERY_PROMPT = """You are helping a physician look up textbook passages to answer a USMLE question.

Question: {question}
Options: {options}
After a first round of independent answers, the two leading candidates are ({a}) {a_text} and ({b}) {b_text}.

1. List the key findings in the question (symptoms, signs, labs, history), as short phrases.
2. Write 3 short search queries (keywords, not sentences) for a medical textbook index. Together they should
   retrieve the facts that best tell the two candidates apart: what each candidate implies or requires, and how
   the key findings fit each one.

Reply only with JSON: {{"findings": ["..."], "queries": ["...", "...", "..."]}}"""

DIFF_PROMPT = """You are a physician answering a USMLE question. After a first round of independent answers, two options \
remain as the leading candidates: ({a}) {a_text} and ({b}) {b_text}.
{evidence}
Question: {question}
Options: {options}

Work through it:
1. List the key findings in the question.
2. For each candidate, state what you would expect if it were correct, then check those expectations against the \
findings: what supports it and what argues against it.{use_evidence}
3. Decide. Choose an option outside the two candidates only if the findings clearly point to it.

Output in exactly this format:
Thought: [your comparison]
Answer: [one letter, A-E]"""


def format_books(chunks):
    return '\n'.join(f"[{i + 1}] ({c['title'].split('_')[0]}) {c['content']}" for i, c in enumerate(chunks))


def format_cases(cases):
    out = []
    for i, c in enumerate(cases):
        opts = ' '.join(f'({k}) {v}' for k, v in c['options'].items())
        out.append(f"Case {i + 1}: {c['question']}\nOptions: {opts}\nCorrect answer: ({c['answer_idx']}) "
                   f"{c['options'][c['answer_idx']]}")
    return '\n\n'.join(out)


def retrieve_books(books, queries, k_books):
    """Trechos divididos entre as buscas, sem repetir trecho."""
    seen, chunks = set(), []
    per = max(1, k_books // len(queries))
    for s in queries:
        for c in books.search(s, k=per + 2):
            if c['id'] not in seen and len([x for x in chunks if x['query'] == s]) < per:
                seen.add(c['id'])
                chunks.append(dict(c, query=s))
    return chunks[:k_books]


def parse_plan(raw):
    try:
        plan = json.loads(raw[raw.find('{'):raw.rfind('}') + 1])
        queries = [s for s in plan.get('queries', []) if isinstance(s, str) and s.strip()][:3]
    except (ValueError, AttributeError):
        plan, queries = {}, []
    return (plan if isinstance(plan, dict) else {}), queries


def cmd_diff(args, handlers):
    from retrieval import TextbookIndex, CaseIndex
    votes = {json.loads(l)['qid']: json.loads(l) for l in open(os.path.join(OUT_DIR, f'{args.split}-vote.jsonl'),
                                                                 encoding='utf-8')}
    path = os.path.join(OUT_DIR, f'{args.split}-diff{args.tag}.jsonl')
    done = done_qids(path)
    todo = []
    for qid, v in sorted(votes.items()):
        pooled = [a for s in v['samples'].values() for a in s if a]
        top = [a for a in ranked(pooled) if a in v['options']]  # só letras que existem (MMLU: A-D)
        agree = Counter(pooled)[top[0]] / len(pooled)
        if args.min_agreement <= agree < args.trigger and len(top) >= 2 and qid not in done:
            todo.append((v, top[:2], agree))
    print(f'{path}: questões com concordância em [{args.min_agreement:.0%}, {args.trigger:.0%}) a fazer: {len(todo)}; '
          f'já feitas {len(done)}')
    if not todo:
        return
    books, cases = TextbookIndex(), CaseIndex()
    qsrc = {q['qid']: q for q in subset(args.split, 0, 0)}
    planner = handlers[args.planner]

    def solve(item):
        v, (a, b), agree = item
        q = qsrc[v['qid']]
        opts = q['options']
        for h in handlers.values():
            h.reset_usage()
        before = snapshot(handlers)
        # 1) achados e buscas (um agente, temperatura 0)
        raw = planner.get_output_multiagent('', QUERY_PROMPT.format(question=q['question'], options=opts, a=a, b=b,
                                                                    a_text=opts[a], b_text=opts[b]),
                                            max_tokens=300, temperature=0)
        try:
            plan = json.loads(raw[raw.find('{'):raw.rfind('}') + 1])
            queries = [s for s in plan.get('queries', []) if isinstance(s, str) and s.strip()][:3]
        except (ValueError, AttributeError):
            plan, queries = {}, []
        if not queries:  # plano falhou: busca pelas candidatas
            queries = [f'{opts[a]} {opts[b]}']
        # 2) recuperação: trechos por busca, sem repetir; casos parecidos sem quase-duplicatas
        chunks = retrieve_books(books, queries, args.k_books)
        similar, dropped = cases.search(q['question'], k=args.k_cases)
        planning = diff_usage(snapshot(handlers), before)
        # 3) comparação das candidatas, em três variantes
        ev = {'none': '',
              'books': f'\nTextbook passages that may help:\n{format_books(chunks)}\n',
              'books_cases': f'\nTextbook passages that may help:\n{format_books(chunks)}\n\n'
                             f'Solved practice questions similar to this one:\n{format_cases(similar)}\n'}
        use = {'none': '', 'books': ' Use the textbook passages where they apply.',
               'books_cases': ' Use the textbook passages and the solved questions where they apply.'}
        variants = {}
        for name in args.variants:
            prompt = DIFF_PROMPT.format(a=a, b=b, a_text=opts[a], b_text=opts[b], evidence=ev[name],
                                        question=q['question'], options=opts, use_evidence=use[name])
            vb = snapshot(handlers)
            outs, ans = {}, {}
            for m, h in handlers.items():
                o = h.get_samples(prompt, n=args.per_model, max_tokens=args.max_tokens, temperature=args.temperature)
                outs[m] = o
                ans[m] = [parse_answer(x) for x in o]
            variants[name] = {'answers': ans, 'outputs': outs, 'usage': diff_usage(snapshot(handlers), vb)}
        return {'qid': q['qid'], 'split': args.split, 'gold': q['answer_idx'], 'candidates': [a, b],
                'agreement': agree, 'findings': plan.get('findings', []) if isinstance(plan, dict) else [],
                'queries': queries,
                'books': [{'id': c['id'], 'query': c['query'], 'score': c['score']} for c in chunks],
                'cases': [{'train_qid': c['train_qid'], 'jaccard': c['jaccard']} for c in similar],
                'dropped_duplicates': dropped, 'planning_usage': planning, 'variants': variants}

    run_pool(todo, solve, path, args.workers, 'etapa 2')


# ------------------------------------------------------------------ ablação: etapa única
RAG_QUERY_PROMPT = """You are helping a physician look up textbook passages to answer a USMLE question.

Question: {question}
Options: {options}

1. List the key findings in the question (symptoms, signs, labs, history), as short phrases.
2. Write 3 short search queries (keywords, not sentences) for a medical textbook index. Together they should
   retrieve the facts that best tell the most plausible options apart: what each option implies or requires, and how
   the key findings fit each one.

Reply only with JSON: {{"findings": ["..."], "queries": ["...", "...", "..."]}}"""

RAG_PROMPT = """You are a physician answering a USMLE question.
{evidence}
Question: {question}
Options: {options}

Work through it:
1. List the key findings in the question.
2. For each plausible option, state what you would expect if it were correct, then check those expectations against \
the findings: what supports it and what argues against it. Use the textbook passages and the solved questions where \
they apply.
3. Decide.

Output in exactly this format:
Thought: [your comparison]
Answer: [one letter, A-E]"""


def cmd_rag(args, handlers):
    """Etapa única: o mesmo planejamento, a mesma busca e a mesma evidência da etapa 2 (livros + questões
    resolvidas), mas sem voto prévio, em todas as questões e com as 5 alternativas abertas."""
    from retrieval import TextbookIndex, CaseIndex
    path = os.path.join(OUT_DIR, f'{args.split}-rag{args.tag}.jsonl')
    done = done_qids(path)
    todo = [q for q in subset(args.split, args.limit, args.seed, args.rest_of) if q['qid'] not in done]
    print(f'{path}: já feitas {len(done)}, faltam {len(todo)}')
    if not todo:
        return
    books, cases = TextbookIndex(), CaseIndex()
    planner = handlers[args.planner]

    def solve(q):
        opts = q['options']
        for h in handlers.values():
            h.reset_usage()
        before = snapshot(handlers)
        raw = planner.get_output_multiagent('', RAG_QUERY_PROMPT.format(question=q['question'], options=opts),
                                            max_tokens=300, temperature=0)
        plan, queries = parse_plan(raw)
        if not queries:  # plano falhou: busca pelo enunciado
            queries = [q['question'][-300:]]
        chunks = retrieve_books(books, queries, args.k_books)
        similar, dropped = cases.search(q['question'], k=args.k_cases)
        planning = diff_usage(snapshot(handlers), before)
        evidence = (f'\nTextbook passages that may help:\n{format_books(chunks)}\n\n'
                    f'Solved practice questions similar to this one:\n{format_cases(similar)}\n')
        prompt = RAG_PROMPT.format(evidence=evidence, question=q['question'], options=opts)
        vb = snapshot(handlers)
        outs, ans = {}, {}
        for m, h in handlers.items():
            outs[m] = h.get_samples(prompt, n=args.per_model, max_tokens=args.max_tokens, temperature=args.temperature)
            ans[m] = [parse_answer(x) for x in outs[m]]
        if not any(a for v in ans.values() for a in v):
            raise RuntimeError(f"questão {q['qid']}: nenhuma amostra válida")
        return {'qid': q['qid'], 'split': args.split, 'gold': q['answer_idx'],
                'findings': plan.get('findings', []), 'queries': queries,
                'books': [{'id': c['id'], 'query': c['query'], 'score': c['score']} for c in chunks],
                'cases': [{'train_qid': c['train_qid'], 'jaccard': c['jaccard']} for c in similar],
                'dropped_duplicates': dropped, 'planning_usage': planning,
                'answers': ans, 'outputs': outs, 'usage': diff_usage(snapshot(handlers), vb)}

    run_pool(todo, solve, path, args.workers, 'etapa única')


# ------------------------------------------------------------------ etapa 2 com fichas de diferenças (MedRAG)
DDX_QUERY_PROMPT = """You are helping a physician look up textbook passages to answer a USMLE question.

Question: {question}
Options: {options}
After a first round of independent answers, the leading candidates are: {cands}.

1. List the key findings in the question (symptoms, signs, labs, history), as short phrases.
2. For EACH candidate, write 2 short search queries (keywords, not sentences) for a medical textbook index that
   retrieve what that candidate requires, implies or typically shows, and what sets it apart from the other candidates.

Reply only with JSON: {{"findings": ["..."], "queries": {{{qkeys}}}}}"""

CARD_PROMPT = """You are building reference cards from textbook passages. You will NOT see the patient case.

The exam question asks: "{ask}"
Candidate answers: {cands}

Textbook passages, grouped by the candidate they were retrieved for:
{passages}

For each candidate, write a short card with:
- Requires / typical: findings, settings or facts that must be present (or are typical) when this is the right answer.
- Rules it out: findings or facts that make this answer unlikely or wrong.
- Versus the others: the specific features that tell this candidate apart from each of the other candidates.
Base the cards on the passages and on well-established medical knowledge; cite passage numbers like [3] when used.
Be concise: at most 6 bullet points per candidate."""

DDX_PROMPT = """You are a physician answering a USMLE question. After a first round of independent answers, the leading \
candidates are: {cands}.{prior}

Reference cards for each candidate (built from textbooks, without seeing this case):
{card}

Textbook passages behind the cards:
{passages}

Question: {question}
Options: {options}

Work through it:
1. List the key findings in the question.
2. For each candidate, check the findings against its card: which required features are present or missing, and \
whether any rule-out feature is present.
3. Decide. Choose an option outside the candidates only if the findings clearly point to it.

Output in exactly this format:
Thought: [your comparison]
Answer: [one letter, A-E]"""


PRIOR_TEXT = """ In that first round, {shares}. The majority answer, ({top}), is the default: switch to \
another candidate only if something in the question contradicts its card (a rule-out feature is present, or a required \
feature is clearly absent) and fits the other candidate better. Cards are written for every candidate and can make a \
weak candidate sound as strong as the right one, so a supportive card alone is not a reason to switch."""


def last_sentence(text):
    parts = [x.strip() for x in re.split(r'(?<=[.?!])\s+', text.strip()) if x.strip()]
    return parts[-1] if parts else text[-200:]


def cmd_ddx(args, handlers):
    from retrieval import TextbookIndex
    votes = {json.loads(l)['qid']: json.loads(l) for l in open(os.path.join(OUT_DIR, f'{args.split}-vote.jsonl'),
                                                                 encoding='utf-8')}
    path = os.path.join(OUT_DIR, f'{args.split}-ddx{args.tag}.jsonl')
    done = done_qids(path)
    todo = []
    for qid, v in sorted(votes.items()):
        pooled = [a for s in v['samples'].values() for a in s if a]
        c = Counter(pooled)
        top = [a for a in ranked(pooled) if a in v['options']]  # só letras que existem (MMLU: A-D)
        agree = c[top[0]] / len(pooled)
        if not (args.min_agreement <= agree < args.trigger) or len(top) < 2 or qid in done:
            continue
        cands = top[:2]
        if args.max_candidates >= 3 and len(top) >= 3 and c[top[2]] / len(pooled) >= args.third_min:
            cands = top[:3]
        elif args.only_extra_candidate:
            continue
        todo.append((v, cands, agree, {a: c[a] / len(pooled) for a in top[:3]}))
    print(f'{path}: a fazer {len(todo)}; já feitas {len(done)}')
    if not todo:
        return
    books = TextbookIndex()
    qsrc = {q['qid']: q for q in subset(args.split, 0, 0)}
    planner = handlers[args.planner]

    def solve(item):
        v, cands, agree, shares = item
        q = qsrc[v['qid']]
        opts = q['options']
        clist = ', '.join(f'({a}) {opts[a]}' for a in cands)
        for h in handlers.values():
            h.reset_usage()
        before = snapshot(handlers)
        # 1) achados e buscas por candidata
        raw = planner.get_output_multiagent('', DDX_QUERY_PROMPT.format(
            question=q['question'], options=opts, cands=clist,
            qkeys=', '.join(f'"{a}": ["...", "..."]' for a in cands)), max_tokens=400, temperature=0)
        try:
            plan = json.loads(raw[raw.find('{'):raw.rfind('}') + 1])
            qmap = plan.get('queries', {}) if isinstance(plan, dict) else {}
        except (ValueError, AttributeError):
            plan, qmap = {}, {}
        queries = {}
        for a in cands:
            got = qmap.get(a) if isinstance(qmap, dict) else None
            got = [x for x in (got or []) if isinstance(x, str) and x.strip()][:2]
            queries[a] = got or [opts[a]]
        # 2) trechos por candidata, sem repetir entre candidatas
        seen, groups, n = set(), {}, 0
        for a in cands:
            groups[a] = []
            for ch in retrieve_books(books, queries[a], args.k_per_candidate):
                if ch['id'] not in seen:
                    seen.add(ch['id'])
                    n += 1
                    groups[a].append(dict(ch, n=n))
        passages = '\n'.join(f"For ({a}) {opts[a]}:\n" + '\n'.join(
            f"[{c['n']}] ({c['title'].split('_')[0]}) {c['content']}" for c in groups[a]) for a in cands)
        planning = diff_usage(snapshot(handlers), before)
        # 3) fichas, sem o caso
        cb = snapshot(handlers)
        card = planner.get_output_multiagent('', CARD_PROMPT.format(ask=last_sentence(q['question']), cands=clist,
                                                                    passages=passages),
                                             max_tokens=args.card_max_tokens, temperature=0)
        card_usage = diff_usage(snapshot(handlers), cb)
        # 4) conferência do caso contra as fichas
        prior = ''
        if args.prior:
            prior = PRIOR_TEXT.format(top=cands[0], shares=', '.join(
                f'({a}) got {shares[a]:.0%} of the answers' for a in cands))
        prompt = DDX_PROMPT.format(cands=clist, prior=prior, card=card, passages=passages, question=q['question'],
                                   options=opts)
        vb = snapshot(handlers)
        outs, ans = {}, {}
        for m, h in handlers.items():
            outs[m] = h.get_samples(prompt, n=args.per_model, max_tokens=args.max_tokens, temperature=args.temperature)
            ans[m] = [parse_answer(x) for x in outs[m]]
        if not any(a for x in ans.values() for a in x):
            raise RuntimeError(f"questão {q['qid']}: nenhuma amostra válida")
        return {'qid': q['qid'], 'split': args.split, 'gold': q['answer_idx'], 'candidates': cands, 'shares': shares,
                'agreement': agree, 'findings': plan.get('findings', []) if isinstance(plan, dict) else [],
                'queries': queries,
                'books': {a: [{'id': c['id'], 'query': c['query'], 'score': c['score']} for c in groups[a]] for a in cands},
                'card': card, 'prior': prior, 'answers': ans, 'outputs': outs,
                'planning_usage': planning, 'card_usage': card_usage, 'usage': diff_usage(snapshot(handlers), vb)}

    run_pool(todo, solve, path, args.workers, 'etapa 2 (fichas)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['vote', 'diff', 'rag', 'ddx'])
    p.add_argument('--dataset', default='medqa', choices=['medqa', 'mmlu'])
    p.add_argument('--prior', action='store_true', help='ddx: informa os votos e mantém a majoritária salvo contradição')
    p.add_argument('--split', default='validation', choices=['train', 'validation', 'test'])
    p.add_argument('--limit', type=int, default=300, help='sorteia este número de questões (0 = todas)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--rest_of', type=int, default=0, help='usa as questões fora do sorteio deste tamanho (confirmação)')
    p.add_argument('--models', default='gpt-5.4-nano,gpt-4o-mini')
    p.add_argument('--samples', default='21', help='etapa 1: amostras por modelo (um número, ou um por modelo: 11,10)')
    p.add_argument('--per_model', type=int, default=3, help='etapa 2: amostras por modelo e variante')
    p.add_argument('--variants', default=','.join(VARIANTS))
    p.add_argument('--trigger', type=float, default=0.9, help='etapa 2 roda abaixo desta concordância')
    p.add_argument('--min_agreement', type=float, default=0.0, help='etapa 2 roda a partir desta concordância')
    p.add_argument('--tag', default='', help='sufixo do arquivo de saída (ex.: -semportao)')
    p.add_argument('--k_books', type=int, default=6)
    p.add_argument('--k_cases', type=int, default=3)
    p.add_argument('--k_per_candidate', type=int, default=4, help='ddx: trechos por candidata')
    p.add_argument('--card_max_tokens', type=int, default=700, help='ddx: limite das fichas')
    p.add_argument('--max_candidates', type=int, default=2, help='ddx: 2 ou 3 candidatas')
    p.add_argument('--third_min', type=float, default=0.1, help='ddx: fatia mínima de votos da 3ª candidata')
    p.add_argument('--only_extra_candidate', action='store_true', help='ddx: só as questões com 3ª candidata')
    p.add_argument('--planner', default='gpt-5.4-nano')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--max_tokens', type=int, default=1000)
    p.add_argument('--workers', type=int, default=10)
    args = p.parse_args()
    DATASET = args.dataset
    if DATASET == 'mmlu':
        OUT_DIR = os.environ.get('MEDVOTE_OUT', os.path.join('outputs', 'medvote', 'mmlu'))
        for _name in ('QUERY_PROMPT', 'DIFF_PROMPT', 'RAG_QUERY_PROMPT', 'RAG_PROMPT', 'DDX_QUERY_PROMPT', 'DDX_PROMPT'):
            globals()[_name] = globals()[_name].replace('USMLE question', 'medical exam question').replace('A-E', 'A-D')
    args.variants = [v for v in args.variants.split(',') if v]
    ns = [int(x) for x in args.samples.split(',')]
    n_models = len(args.models.split(','))
    args.samples = ns * n_models if len(ns) == 1 else ns
    assert len(args.samples) == n_models, '--samples: um número, ou um por modelo'
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
    os.makedirs(OUT_DIR, exist_ok=True)
    handlers = {m: api_handler(m) for m in args.models.split(',')}
    for h in handlers.values():
        h.preflight()
    t0 = time.time()
    {'vote': cmd_vote, 'diff': cmd_diff, 'rag': cmd_rag, 'ddx': cmd_ddx}[args.command](args, handlers)
    print(f'Tempo: {time.time() - t0:.0f}s')
