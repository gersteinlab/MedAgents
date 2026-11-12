"""Converte saídas no formato do repositório original (uma linha JSON por questão, em ordem,
sem qid) para o formato novo, para reaproveitar execuções já feitas.

  .\.venv\Scripts\python.exe import_legacy.py --input outputs\MedQA\final\gpt-4o-mini-syn_verif `
      --model_name gpt-4o-mini --method syn_verif --probe_rounds

--probe_rounds (só syn_verif): preenche round_answers chamando o prompt de decisão final em
cada versão anterior do relatório. Só gasta chamadas nas questões que tiveram revisão.
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor

from prompt_generator import get_final_answer_prompt_wsyn
from data_utils import cleansing_final_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--model_name', required=True)
    ap.add_argument('--method', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--start_pos', type=int, default=0, help='qid da primeira linha do arquivo')
    ap.add_argument('--dataset', default='./datasets/MedQA/test.jsonl')
    ap.add_argument('--root', default='./outputs/MedQA')
    ap.add_argument('--probe_rounds', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    data = [json.loads(l) for l in open(args.dataset, encoding='utf-8')]
    recs = [json.loads(l) for l in open(args.input, encoding='utf-8') if l.strip()]
    for i, r in enumerate(recs):
        qid = args.start_pos + i
        if r['gold_answer'] != data[qid]['answer_idx'] or data[qid]['question'][:40] not in r['question']:
            raise SystemExit(f'Linha {i} não corresponde à questão {qid} do dataset. Confira --start_pos.')
        r.update({'qid': qid, 'step': data[qid]['meta_info'], 'model': args.model_name, 'method': args.method})
        r.setdefault('round_answers', [])
        r.setdefault('usage', {})
        r.setdefault('probe_usage', {})

    if args.probe_rounds and args.method == 'syn_verif':
        from api_utils import api_handler
        from run import load_dotenv
        load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
        handler = api_handler(args.model_name)
        handler.preflight()

        def probe(r):
            handler.reset_usage()
            answers = []
            for report in r['syn_repo_history'][:-1]:
                out = handler.get_output_multiagent(system_role="", user_input=get_final_answer_prompt_wsyn(report),
                                                    max_tokens=2500, temperature=0)
                answers.append(cleansing_final_output(out)[0])
            r['round_answers'] = answers + [r['pred_answer']]
            r['probe_usage'] = handler.usage()
        with ThreadPoolExecutor(args.workers) as pool:
            list(pool.map(probe, recs))

    safe = args.model_name.replace('/', '_').replace(':', '_')
    out_dir = os.path.join(args.root, safe)
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{safe}-{args.method}{('-' + args.tag) if args.tag else ''}.jsonl")
    with open(out, 'w', encoding='utf-8') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'{len(recs)} registros gravados em {out}')
    if not any(r['usage'] for r in recs):
        print('Aviso: o formato antigo não registra chamadas/tokens; o custo dessa execução aparecerá como 0.')


if __name__ == '__main__':
    main()
