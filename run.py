from data_utils import MyDataset
from api_utils import api_handler, FATAL_ERRORS
try:
    import anthropic
    FATAL_REVIEWER_ERRORS = (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError)
except ImportError:
    FATAL_REVIEWER_ERRORS = ()
from string import punctuation
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os
import threading
import time
import tqdm
import json
from utils import *
from debate import persona_debate, TOPOLOGIES


def load_dotenv(path='.env'):
    """Lê chaves de um arquivo .env (CHAVE=valor por linha) sem sobrescrever o ambiente.
    Evita ter de reiniciar o terminal/VS Code depois de criar a chave."""
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def done_qids(path):
    """Ids já gravados, para retomar uma execução interrompida sem repetir chamadas."""
    if not os.path.exists(path):
        return set()
    seen = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    seen.add(json.loads(line)['qid'])
                except (json.JSONDecodeError, KeyError):
                    pass
    return seen


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='gpt-4o-mini',
                        help="apelido de api_utils.MODELS ou 'provedor:modelo'")
    parser.add_argument('--base_url', default=None, help='sobrescreve a URL do provedor')
    parser.add_argument('--dataset_name', default='MedQA')
    parser.add_argument('--dataset_dir', default='./datasets/MedQA/')
    parser.add_argument('--start_pos', type=int, default=0)
    parser.add_argument('--end_pos', type=int, default=-1)
    parser.add_argument('--output_files_folder', default='./outputs/MedQA')

    parser.add_argument('--method', type=str, default='syn_verif',
                        choices=['syn_verif', 'syn_verif_review', 'syn_only', 'anal_only', 'base_direct', 'base_cot', 'base_sc',
                                 'persona_debate'])
    parser.add_argument('--reviewer_model', default='claude-opus-5-5',
                        help='revisor do syn_verif_review (ex.: claude-haiku-5-5, claude-sonnet-5-5)')
    parser.add_argument('--reviewer_effort', default='high', choices=['low', 'medium', 'high', 'xhigh', 'max'])
    parser.add_argument('--max_attempt_vote', type=int, default=3)
    parser.add_argument('--cot_max_tokens', type=int, default=300,
                        help='300 = código original; 1000 evita respostas cortadas em modelos novos')
    parser.add_argument('--sc_samples', type=int, default=21,
                        help='amostras do Self-Consistency (21 ≈ chamadas do MedAgents por questão)')
    parser.add_argument('--sc_temperature', type=float, default=0.7)
    parser.add_argument('--no_probe_rounds', dest='probe_rounds', action='store_false',
                        help='não registrar a resposta de cada versão do relatório (syn_verif)')
    # correções do MedAgents (desligadas por padrão = código original)
    parser.add_argument('--analysis_max_tokens', type=int, default=300,
                        help='limite das análises dos especialistas (300 = original; corta as últimas alternativas)')
    parser.add_argument('--advice_max_tokens', type=int, default=500, help='limite dos conselhos de revisão (500 = original)')
    parser.add_argument('--fix_key_knowledge', action='store_true',
                        help='inclui o Key Knowledge no relatório (no original ele nunca entra)')
    parser.add_argument('--stop_at_voted', action='store_true',
                        help='não aplica a revisão depois da última votação; decide pelo último relatório votado')
    # persona_debate (debate.py): especialistas do MedAgents debatendo diretamente
    parser.add_argument('--n_agents', type=int, default=7)
    parser.add_argument('--debate_rounds', type=int, default=2)
    parser.add_argument('--topologies', default=','.join(TOPOLOGIES),
                        help='full (todos leem todos), ring (só os 2 vizinhos), ring_conf (anel + Conformist)')
    parser.add_argument('--agent_models', default='',
                        help='modelos alternados entre os agentes, ex.: gpt-4o-mini,gpt-5.4-nano (Heter-MAD)')
    parser.add_argument('--workers', type=int, default=8, help='questões processadas em paralelo')
    parser.add_argument('--tag', default='', help='sufixo do arquivo de saída (ex.: cot1000)')
    args = parser.parse_args()
    args.topologies = [t for t in args.topologies.split(',') if t]
    assert all(t in TOPOLOGIES for t in args.topologies), args.topologies
    print(args)

    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
    handler = api_handler(args.model_name, base_url=args.base_url)
    reviewer = None
    if args.method == 'syn_verif_review':
        from claude_reviewer import claude_reviewer
        reviewer = claude_reviewer(model=args.reviewer_model, effort=args.reviewer_effort)
        # cada combinação revisor/esforço vai para um arquivo próprio
        args.tag = '-'.join(x for x in [f"{args.reviewer_model}-{args.reviewer_effort}", args.tag] if x)
    handlers = [handler]
    if args.method == 'persona_debate' and args.agent_models:
        models = [m.strip() for m in args.agent_models.split(',') if m.strip()]
        assert models[0] == args.model_name, '--agent_models deve começar pelo --model_name'
        handlers = [handler] + [api_handler(m, base_url=args.base_url) for m in models[1:]]
        args.tag = '-'.join(x for x in ['hetero-' + '+'.join(models), args.tag] if x)
    for h in handlers:
        h.preflight()
    if reviewer:
        reviewer.preflight()
    dataobj = MyDataset('test', args, traindata_obj=None)

    end_pos = len(dataobj) if args.end_pos == -1 else args.end_pos
    test_range = range(args.start_pos, end_pos)

    os.makedirs(args.output_files_folder, exist_ok=True)
    safe_model = args.model_name.replace('/', '_').replace(':', '_')
    exact_output_file = os.path.join(args.output_files_folder,
                                     f"{safe_model}-{args.method}{('-' + args.tag) if args.tag else ''}.jsonl")
    skip = done_qids(exact_output_file)
    todo = [i for i in test_range if i not in skip]
    print(f"Saída: {exact_output_file} | já feitas: {len(skip)} | faltam: {len(todo)}")

    write_lock = threading.Lock()

    def solve(idx):
        raw_sample = dataobj.get_by_idx(idx)
        question = raw_sample['question'] if raw_sample['question'][-1] in punctuation else raw_sample['question'] + '?'
        options = raw_sample['options']
        gold_answer = raw_sample['answer_idx']
        if args.dataset_name == 'PubMedQA':
            question = raw_sample['context'] + ' ' + question
        for h in handlers:
            h.reset_usage()
        if reviewer:
            reviewer.reset_usage()
        t0 = time.time()
        if args.method == 'persona_debate':
            res = persona_debate(question, options, handlers, args)
            if res is None:
                raise RuntimeError(f"questão {idx}: todas as respostas da rodada 0 falharam na API")
            res.update({'question': question, 'options': options, 'gold_answer': gold_answer,
                        'pred_answer': res['vote_answer'], 'raw_output': '', 'sc_outputs': []})
            data_info = res
        else:
            data_info = fully_decode(idx, idx, question, options, gold_answer, handler, args, dataobj, reviewer)
        # Resposta final que falhou na API não vira "erro do método": a questão não é gravada
        # e é refeita quando o mesmo comando for rodado de novo.
        failed_final = (all(o == "ERROR." for o in data_info['sc_outputs']) if args.method == 'base_sc'
                        else data_info['raw_output'] == "ERROR.")
        if failed_final:
            raise RuntimeError(f"questão {idx}: a chamada final falhou na API")
        data_info.update({
            'qid': idx,
            'step': raw_sample.get('meta_info', ''),
            'model': args.model_name,
            'method': args.method,
            'usage': handler.usage(),
            'usage_by_model': {h.model: h.usage() for h in handlers},
            'seconds': round(time.time() - t0, 2),
        })
        if reviewer:
            data_info.update({'reviewer_model': args.reviewer_model, 'reviewer_effort': args.reviewer_effort,
                              'reviewer_usage': reviewer.usage()})
        if args.method == 'base_sc':
            data_info.update({'sc_samples': args.sc_samples, 'sc_temperature': args.sc_temperature})
        if args.method in ('syn_verif', 'syn_verif_review'):
            data_info.update({'analysis_max_tokens': args.analysis_max_tokens, 'advice_max_tokens': args.advice_max_tokens,
                              'fix_key_knowledge': args.fix_key_knowledge, 'stop_at_voted': args.stop_at_voted})
        if args.method in ('base_cot', 'base_sc'):
            data_info['cot_max_tokens'] = args.cot_max_tokens
        with write_lock:
            with open(exact_output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data_info, ensure_ascii=False) + '\n')
        return idx

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(solve, i) for i in todo]
        failed = 0
        for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc=f"{args.model_name} {args.method}"):
            try:
                fut.result()
            except FATAL_ERRORS + FATAL_REVIEWER_ERRORS:
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as e:  # a questão fica sem registro e é refeita na próxima execução
                failed += 1
                print(f"Falha em uma questão: {e!r}")
    if failed:
        print(f"{failed} questões falharam. Rode o mesmo comando de novo para completá-las.")
