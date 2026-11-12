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


def subset(split, limit, seed):
    from retrieval import load_split
    qs = load_split(split)
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
    todo = [q for q in subset(args.split, args.limit, args.seed) if q['qid'] not in done]
    print(f'{path}: já feitas {len(done)}, faltam {len(todo)}')

    def solve(q):
        for h in handlers.values():
            h.reset_usage()
        prompt = get_cot_prompt(q['question'], q['options'])
        samples, outputs = {}, {}
        for m, h in handlers.items():
            outs = h.get_samples(prompt, n=args.samples, max_tokens=args.max_tokens, temperature=args.temperature)
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


def cmd_diff(args, handlers):
    from retrieval import TextbookIndex, CaseIndex
    votes = {json.loads(l)['qid']: json.loads(l) for l in open(os.path.join(OUT_DIR, f'{args.split}-vote.jsonl'),
                                                                 encoding='utf-8')}
    path = os.path.join(OUT_DIR, f'{args.split}-diff.jsonl')
    done = done_qids(path)
    todo = []
    for qid, v in sorted(votes.items()):
        pooled = [a for s in v['samples'].values() for a in s if a]
        top = ranked(pooled)
        agree = Counter(pooled)[top[0]] / len(pooled)
        if agree < args.trigger and len(top) >= 2 and qid not in done:
            todo.append((v, top[:2], agree))
    print(f'{path}: questões divididas (concordância < {args.trigger:.0%}) a fazer: {len(todo)}; já feitas {len(done)}')
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
        seen, chunks = set(), []
        per = max(1, args.k_books // len(queries))
        for s in queries:
            for c in books.search(s, k=per + 2):
                if c['id'] not in seen and len([x for x in chunks if x['query'] == s]) < per:
                    seen.add(c['id'])
                    chunks.append(dict(c, query=s))
        chunks = chunks[:args.k_books]
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


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['vote', 'diff'])
    p.add_argument('--split', default='validation', choices=['train', 'validation', 'test'])
    p.add_argument('--limit', type=int, default=300, help='sorteia este número de questões (0 = todas)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--models', default='gpt-5.4-nano,gpt-4o-mini')
    p.add_argument('--samples', type=int, default=21, help='etapa 1: amostras por modelo')
    p.add_argument('--per_model', type=int, default=3, help='etapa 2: amostras por modelo e variante')
    p.add_argument('--variants', default=','.join(VARIANTS))
    p.add_argument('--trigger', type=float, default=0.9, help='etapa 2 roda abaixo desta concordância')
    p.add_argument('--k_books', type=int, default=6)
    p.add_argument('--k_cases', type=int, default=3)
    p.add_argument('--planner', default='gpt-5.4-nano')
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--max_tokens', type=int, default=1000)
    p.add_argument('--workers', type=int, default=10)
    args = p.parse_args()
    args.variants = [v for v in args.variants.split(',') if v]
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
    os.makedirs(OUT_DIR, exist_ok=True)
    handlers = {m: api_handler(m) for m in args.models.split(',')}
    for h in handlers.values():
        h.preflight()
    t0 = time.time()
    (cmd_vote if args.command == 'vote' else cmd_diff)(args, handlers)
    print(f'Tempo: {time.time() - t0:.0f}s')
