"""Recuperação para o MedVote: livros-texto do MedQA e questões resolvidas do treino (BM25).

Dados (baixados uma vez para ext_data/, fora do git):
  ext_data/medqa/{train,validation,test}.parquet   MedQA original, 5 alternativas (bigbio/med_qa, med_qa_en_source)
  ext_data/textbooks/*.jsonl                        18 livros do MedQA em trechos (MedRAG/textbooks, Xiong et al. 2024)

Os índices são construídos na primeira chamada e guardados em ext_data/index/.
"""
import glob
import json
import os
import re
import threading

import bm25s
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.join(HERE, 'ext_data')
INDEX_DIR = os.path.join(EXT, 'index')
_lock = threading.Lock()


def load_split(split):
    """Questões de um conjunto do MedQA. qid = posição no conjunto (no teste, igual ao run.py)."""
    rows = pq.read_table(os.path.join(EXT, 'medqa', f'{split}.parquet')).to_pylist()
    out = []
    for i, r in enumerate(rows):
        opts = r['options']
        if isinstance(opts, list):  # parquet guarda o dict como lista de {key, value}
            opts = {o['key']: o['value'] for o in opts}
        out.append({'qid': i, 'question': r['question'], 'options': opts,
                    'answer_idx': r['answer_idx'], 'step': r.get('meta_info', '')})
    return out


def _tokens(texts):
    return bm25s.tokenize(texts, stopwords='en', show_progress=False)


_WORD = re.compile(r'\w\w+')


def token_set(text):
    return set(_WORD.findall(text.lower()))


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


class TextbookIndex:
    def __init__(self):
        self.chunks = []
        for path in sorted(glob.glob(os.path.join(EXT, 'textbooks', '*.jsonl'))):
            with open(path, encoding='utf-8') as f:
                for line in f:
                    c = json.loads(line)
                    self.chunks.append({'id': c['id'], 'title': c['title'], 'content': c['content']})
        path = os.path.join(INDEX_DIR, 'textbooks')
        if os.path.exists(os.path.join(path, 'params.index.json')):
            self.bm25 = bm25s.BM25.load(path, show_progress=False)
        else:
            self.bm25 = bm25s.BM25()
            self.bm25.index(_tokens([c['content'] for c in self.chunks]), show_progress=False)
            os.makedirs(path, exist_ok=True)
            self.bm25.save(path, show_progress=False)

    def search(self, query, k=6):
        with _lock:  # bm25s não garante uso concorrente
            idx, scores = self.bm25.retrieve(_tokens([query]), k=k, show_progress=False)
        return [dict(self.chunks[int(i)], score=float(s)) for i, s in zip(idx[0], scores[0])]


class CaseIndex:
    """Questões resolvidas do treino. Quase-duplicatas da questão consultada são descartadas,
    para que a recuperação não entregue a resposta de uma questão repetida."""

    def __init__(self, dup_threshold=0.6):
        self.cases = load_split('train')
        self.dup_threshold = dup_threshold
        self._sets = [token_set(c['question']) for c in self.cases]
        path = os.path.join(INDEX_DIR, 'train_questions')
        if os.path.exists(os.path.join(path, 'params.index.json')):
            self.bm25 = bm25s.BM25.load(path, show_progress=False)
        else:
            self.bm25 = bm25s.BM25()
            self.bm25.index(_tokens([c['question'] for c in self.cases]), show_progress=False)
            os.makedirs(path, exist_ok=True)
            self.bm25.save(path, show_progress=False)

    def search(self, question, k=3, pool=20):
        with _lock:
            idx, scores = self.bm25.retrieve(_tokens([question]), k=pool, show_progress=False)
        q = token_set(question)
        out, dropped = [], []
        for i, s in zip(idx[0], scores[0]):
            i = int(i)
            sim = jaccard(q, self._sets[i])
            if sim >= self.dup_threshold:
                dropped.append({'train_qid': i, 'jaccard': round(sim, 3)})
                continue
            out.append(dict(self.cases[i], train_qid=i, score=float(s), jaccard=round(sim, 3)))
            if len(out) == k:
                break
        return out, dropped

    def nearest_jaccard(self, question, pool=20):
        """Maior Jaccard com o treino (para medir sobreposição entre conjuntos)."""
        with _lock:
            idx, _ = self.bm25.retrieve(_tokens([question]), k=pool, show_progress=False)
        q = token_set(question)
        return max(jaccard(q, self._sets[int(i)]) for i in idx[0])
