"""Camada de API compatível com qualquer endpoint no formato OpenAI (Chat Completions).

Substitui a versão original (openai==0.27, Azure). O comportamento do método não muda:
mesmos prompts, mesmos limites de tokens e temperatura 0 nas chamadas do MedAgents.

Escolha do modelo (--model_name):
  * um apelido de MODELS abaixo (ex.: gpt-4o-mini, gpt-4.1-mini, qwen2.5-7b-local), ou
  * "provedor:id-do-modelo" (ex.: openrouter:meta-llama/llama-3.3-70b-instruct,
    ollama:qwen2.5:7b-instruct, openai:gpt-4o).

Chaves lidas de variáveis de ambiente (nunca do código): OPENAI_API_KEY, OPENROUTER_API_KEY,
GROQ_API_KEY, TOGETHER_API_KEY, DEEPSEEK_API_KEY. Ollama/vLLM locais não precisam de chave.
"""
import os
import random
import threading
import time

import openai
from openai import OpenAI

# Erros que nenhuma nova tentativa resolve (chave inválida, sem permissão, modelo inexistente ou
# descontinuado). Interrompem a execução inteira, em vez de virar "ERROR." em todas as questões.
FATAL_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError)

PROVIDERS = {
    'openai':     {'base_url': None,                                'key_env': 'OPENAI_API_KEY'},
    'openrouter': {'base_url': 'https://openrouter.ai/api/v1',      'key_env': 'OPENROUTER_API_KEY'},
    'groq':       {'base_url': 'https://api.groq.com/openai/v1',    'key_env': 'GROQ_API_KEY'},
    'together':   {'base_url': 'https://api.together.xyz/v1',       'key_env': 'TOGETHER_API_KEY'},
    'deepseek':   {'base_url': 'https://api.deepseek.com',          'key_env': 'DEEPSEEK_API_KEY'},
    'ollama':     {'base_url': 'http://localhost:11434/v1',         'key_env': None},
    'vllm':       {'base_url': 'http://localhost:8000/v1',          'key_env': None},
}

# apelido -> (provedor, id do modelo, é modelo de raciocínio?)
MODELS = {
    'gpt-4o-mini':       ('openai', 'gpt-4o-mini', False),
    'gpt-4o':            ('openai', 'gpt-4o', False),
    'gpt-4.1-nano':      ('openai', 'gpt-4.1-nano', False),
    'gpt-4.1-mini':      ('openai', 'gpt-4.1-mini', False),
    'gpt-4.1':           ('openai', 'gpt-4.1', False),
    'gpt-5-mini':        ('openai', 'gpt-5-mini', True),
    'gpt-5.4-nano':      ('openai', 'gpt-5.4-nano', False),
    'gpt-5.4-mini':      ('openai', 'gpt-5.4-mini', False),
    'llama-3.3-70b':     ('openrouter', 'meta-llama/llama-3.3-70b-instruct', False),
    'qwen2.5-72b':       ('openrouter', 'qwen/qwen-2.5-72b-instruct', False),
    'deepseek-v3':       ('deepseek', 'deepseek-chat', False),
    'llama-3.1-8b-local': ('ollama', 'llama3.1:8b-instruct-q8_0', False),
    'qwen2.5-7b-local':  ('ollama', 'qwen2.5:7b-instruct-q8_0', False),
    'qwen2.5-14b-local': ('ollama', 'qwen2.5:14b-instruct-q4_K_M', False),
}

# Modelos de raciocínio (o1/o3/gpt-5) não aceitam temperatura e gastam tokens "pensando"
# antes de responder; sem essa folga, limites como 30 tokens no voto devolvem texto vazio.
REASONING_TOKEN_BUDGET = int(os.environ.get('REASONING_TOKEN_BUDGET', '4096'))
MAX_ATTEMPTS = 5

# Modelos da família gpt-5.x que permitem desligar o raciocínio. Com reasoning_effort="none"
# eles aceitam temperatura e n como um modelo comum, então rodam os mesmos métodos que o
# gpt-4o-mini. Usam max_completion_tokens no lugar de max_tokens.
REASONING_OFF = {'gpt-5.4-nano', 'gpt-5.4-mini'}


def resolve_model(model_name, base_url=None):
    if model_name in MODELS:
        provider, model_id, reasoning = MODELS[model_name]
    elif ':' in model_name and model_name.split(':', 1)[0] in PROVIDERS:
        provider, model_id = model_name.split(':', 1)
        reasoning = model_id.startswith(('o1', 'o3', 'o4', 'gpt-5'))
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}. Use um apelido de MODELS "
                         f"ou 'provedor:modelo' com provedor em {list(PROVIDERS)}.")
    cfg = PROVIDERS[provider]
    url = base_url or os.environ.get(f'{provider.upper()}_BASE_URL') or cfg['base_url']
    key = os.environ.get(cfg['key_env']) if cfg['key_env'] else 'local'
    if not key:
        raise RuntimeError(f"Defina a variável de ambiente {cfg['key_env']} para usar {model_name}.")
    return provider, model_id, reasoning, url, key


class api_handler:
    """Mantém a interface usada por utils.py (get_output_multiagent) e conta chamadas/tokens
    por thread, para que cada questão saia com o próprio custo registrado."""

    def __init__(self, model_name, base_url=None, timeout=120):
        self.model = model_name
        self.provider, self.engine, self.reasoning, url, key = resolve_model(model_name, base_url)
        self.client = OpenAI(api_key=key, base_url=url, timeout=timeout, max_retries=0)
        self._local = threading.local()
        self._n_supported = self.provider in ('openai', 'vllm', 'together', 'deepseek')

    def preflight(self):
        """Chamada real mínima (até 16 tokens de resposta) antes de começar: um modelo descontinuado
        ainda aparece na lista de modelos, então só consultar a lista não basta."""
        self.reset_usage()
        response = self._create([{'role': 'user', 'content': 'Reply with OK.'}], max_tokens=16, temperature=0)
        if response is None:
            raise RuntimeError(f'O modelo {self.engine} não respondeu ao teste inicial.')

    # ---------- contadores por questão ----------
    def reset_usage(self):
        self._local.usage = {'calls': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'errors': 0}

    def usage(self):
        return dict(getattr(self._local, 'usage', {}))

    def _count(self, response=None, error=False):
        u = getattr(self._local, 'usage', None)
        if u is None:
            self.reset_usage()
            u = self._local.usage
        if error:
            u['errors'] += 1
            return
        u['calls'] += 1
        if response is not None and getattr(response, 'usage', None):
            u['prompt_tokens'] += response.usage.prompt_tokens or 0
            u['completion_tokens'] += response.usage.completion_tokens or 0

    # ---------- chamada base ----------
    def _create(self, messages, max_tokens, temperature, n=1, stop=None):
        kwargs = {'model': self.engine, 'messages': messages}
        if self.reasoning:
            kwargs['max_completion_tokens'] = max_tokens + REASONING_TOKEN_BUDGET
            effort = os.environ.get('REASONING_EFFORT')
            if effort:
                kwargs['reasoning_effort'] = effort
        elif self.engine in REASONING_OFF:
            kwargs.update(max_completion_tokens=max_tokens, temperature=temperature, top_p=1, reasoning_effort='none')
            if stop:
                kwargs['stop'] = stop
        else:
            kwargs.update(max_tokens=max_tokens, temperature=temperature, top_p=1)
            if stop:
                kwargs['stop'] = stop
        if n > 1:
            kwargs['n'] = n
        if self.provider == 'ollama':
            # contexto padrão do Ollama (2048) corta os relatórios longos da etapa de revisão
            kwargs['extra_body'] = {'options': {'num_ctx': int(os.environ.get('OLLAMA_NUM_CTX', '16384'))}}

        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self.client.chat.completions.create(**kwargs)
                self._count(response)
                return response
            except FATAL_ERRORS:
                raise
            except openai.BadRequestError as error:
                # pedido recusado (ex.: filtro de conteúdo nesta questão): repetir não muda nada
                self._count(error=True)
                print(f'[{self.engine}] pedido recusado: {error}')
                return None
            except Exception as error:  # rede, limite de taxa, 5xx
                self._count(error=True)
                wait = min(60, 2 ** attempt) + random.random()
                print(f'[{self.engine}] tentativa {attempt + 1}/{MAX_ATTEMPTS} falhou: {error}. '
                      f'Nova tentativa em {wait:.1f}s')
                time.sleep(wait)
        return None

    def get_output_multiagent(self, system_role, user_input, max_tokens, temperature=0,
                              frequency_penalty=0, presence_penalty=0, stop=None):
        messages = []
        if system_role:
            messages.append({'role': 'system', 'content': system_role})
        messages.append({'role': 'user', 'content': user_input})
        response = self._create(messages, max_tokens, temperature, stop=stop)
        if response is None or not response.choices or response.choices[0].message.content is None:
            return "ERROR."
        return response.choices[0].message.content

    def get_samples(self, user_input, n, max_tokens, temperature, system_role=""):
        """n amostras independentes do mesmo prompt (Self-Consistency).
        Usa o parâmetro n quando o provedor aceita (o prompt é cobrado uma vez só)."""
        messages = []
        if system_role:
            messages.append({'role': 'system', 'content': system_role})
        messages.append({'role': 'user', 'content': user_input})
        outputs = []
        if self._n_supported and not self.reasoning:
            remaining = n
            while remaining > 0:
                batch = min(remaining, 8)
                response = self._create(messages, max_tokens, temperature, n=batch)
                if response is None:
                    outputs.extend(["ERROR."] * batch)
                else:
                    got = [c.message.content or "ERROR." for c in response.choices]
                    outputs.extend(got + ["ERROR."] * (batch - len(got)))
                remaining -= batch
        else:
            for _ in range(n):
                outputs.append(self.get_output_multiagent(system_role, user_input, max_tokens, temperature))
        return outputs[:n]
