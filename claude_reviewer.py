"""Revisor externo do método syn_verif_review: o Claude lê o relatório sintetizado pela equipe
(gpt-4o-mini ou outro modelo) e vota SIM/NÃO junto com os especialistas, propondo correções.

Trazido do pacote da outra máquina, com dois ajustes:
  * contagem de tokens por questão (thread-local), porque o run.py processa questões em paralelo;
  * registro do modelo que de fato respondeu (o fallback de recusa pode trocar de modelo).

Lê ANTHROPIC_API_KEY do ambiente (ou do arquivo .env, carregado pelo run.py).
"""
import threading
import time

import anthropic

DEFAULT_REVIEWER_MODEL = "claude-opus-5-5"

# Modelos que aceitam o fallback de recusa no servidor (fallbacks="default"): se o filtro de
# segurança recusar uma pergunta médica por falso positivo, a API refaz a chamada em outro
# modelo em vez de devolver a recusa. O Haiku 5.5 não tem essa opção.
FALLBACK_MODELS = {"claude-fable-5-2", "claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}


class claude_reviewer:
    def __init__(self, model=DEFAULT_REVIEWER_MODEL, effort="high"):
        self.model = model
        self.effort = effort
        # max_retries: o SDK já repete 429, 5xx e falhas de conexão com espera exponencial
        self.client = anthropic.Anthropic(timeout=300, max_retries=4)
        self._local = threading.local()

    def preflight(self):
        """Consulta o modelo (gratuito) para falhar logo se a chave ou o nome estiverem errados."""
        self.client.models.retrieve(self.model)

    def reset_usage(self):
        self._local.usage = {'calls': 0, 'input_tokens': 0, 'output_tokens': 0,
                             'errors': 0, 'refusals': 0, 'served_by_fallback': 0}

    def usage(self):
        return dict(getattr(self._local, 'usage', {}))

    def _u(self):
        if not hasattr(self._local, 'usage'):
            self.reset_usage()
        return self._local.usage

    def get_output(self, system_role, user_input, max_tokens=16000):
        # Sem temperature: os modelos atuais do Claude rejeitam parâmetros de amostragem.
        # max_tokens alto porque o raciocínio (sempre ligado) conta no limite; a resposta é curta.
        params = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=system_role,
            messages=[{"role": "user", "content": user_input}],
            output_config={"effort": self.effort},
        )
        u = self._u()
        t0 = time.time()
        try:
            if self.model in FALLBACK_MODELS:
                response = self.client.beta.messages.create(
                    **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                response = self.client.messages.create(**params)
        except anthropic.BadRequestError as error:
            print(f"Revisor: requisição rejeitada: {error.message}")
            u['errors'] += 1
            return "ERROR."
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError):
            raise  # chave ou modelo errado: melhor parar tudo do que gravar centenas de abstenções
        except anthropic.APIStatusError as error:
            print(f"Revisor: erro da API ({error.status_code}): {error.message}")
            u['errors'] += 1
            return "ERROR."
        except anthropic.APIConnectionError as error:
            print(f"Revisor: erro de conexão: {error}")
            u['errors'] += 1
            return "ERROR."

        u['calls'] += 1
        u['input_tokens'] += response.usage.input_tokens
        u['output_tokens'] += response.usage.output_tokens  # inclui os tokens de raciocínio
        if response.model != self.model:
            u['served_by_fallback'] += 1
        print(f"Revisor ({response.model}): {time.time() - t0:.1f}s")

        if response.stop_reason == "refusal":
            u['refusals'] += 1
            return "REFUSED."
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        return text if text else "ERROR."
