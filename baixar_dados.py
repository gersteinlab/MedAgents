"""Baixa os dados externos usados pelo MedVote (retrieval.py, medvote.py) para ext_data/.

  python baixar_dados.py

  ext_data/medqa/{train,validation,test}.parquet   MedQA original em inglês, 5 alternativas
                                                    (bigbio/med_qa, configuração med_qa_en_source)
  ext_data/textbooks/*.jsonl                        os 18 livros-texto do MedQA em trechos
                                                    (MedRAG/textbooks, Xiong et al., 2024)

Cerca de 280 MB. Arquivos já baixados são pulados. Os índices BM25 são construídos depois,
na primeira chamada do retrieval.py, e guardados em ext_data/index/.
"""
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HF = 'https://huggingface.co/datasets'
EXT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ext_data')
BOOKS = ['Anatomy_Gray', 'Biochemistry_Lippincott', 'Cell_Biology_Alberts', 'First_Aid_Step1', 'First_Aid_Step2',
         'Gynecology_Novak', 'Histology_Ross', 'Immunology_Janeway', 'InternalMed_Harrison', 'Neurology_Adams',
         'Obstentrics_Williams', 'Pathology_Robbins', 'Pathoma_Husain', 'Pediatrics_Nelson', 'Pharmacology_Katzung',
         'Physiology_Levy', 'Psichiatry_DSM-5', 'Surgery_Schwartz']  # nomes como estão no repositório de origem

FILES = ([(f'{HF}/bigbio/med_qa/resolve/main/med_qa_en_source/{s}-00000-of-00001.parquet',
           os.path.join(EXT, 'medqa', f'{s}.parquet')) for s in ('train', 'validation', 'test')] +
         [(f'{HF}/MedRAG/textbooks/resolve/main/chunk/{b}.jsonl', os.path.join(EXT, 'textbooks', f'{b}.jsonl'))
          for b in BOOKS])


def fetch(item):
    url, path = item
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return f'já existe  {os.path.relpath(path, EXT)}'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.part'
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, 'wb') as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    os.replace(tmp, path)
    return f'baixado    {os.path.relpath(path, EXT)} ({os.path.getsize(path) / 1e6:.1f} MB)'


if __name__ == '__main__':
    with ThreadPoolExecutor(6) as ex:
        for msg in ex.map(fetch, FILES):
            print(msg)
