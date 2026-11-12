# MedAgents no MedQA: reprodução, comparações e o MedVote

Fork do [MedAgents](https://github.com/gersteinlab/MedAgents) (Tang et al., ACL 2024 Findings). O README original dos autores está em [README_original.md](README_original.md).

Este fork faz quatro coisas com as 1.273 questões de teste do MedQA (USMLE, inglês, 5 alternativas):

1. **Reprodução do MedAgents** com o gpt-4o-mini e o gpt-5.4-nano, ao lado de quatro métodos de base: resposta direta, CoT com 300 e com 1.000 tokens, e Self-Consistency.
2. **Análise do MedAgents rodada a rodada**: em que rodada a resposta muda, quando a revisão melhora ou piora o relatório, e um diagnóstico de truncamento de saídas.
3. **Variações do MedAgents** inspiradas na literatura de sistemas multiagentes: debate direto entre os especialistas em topologia completa, em anel e em anel com a regra Conformist, e opções que corrigem o truncamento. Ambas foram rodadas só em piloto.
4. **MedVote**, um sistema multiagentes próprio. Primeiro, um voto entre amostras de raciocínio. Depois, só nas questões em que o voto se divide, um diagnóstico diferencial entre as duas alternativas mais votadas, apoiado em trechos dos livros-texto do MedQA recuperados por BM25. A configuração foi escolhida em 300 questões da validação e rodada uma única vez no teste.

## Resultados no teste

Mesmas 1.273 questões em todas as linhas. Os modelos são o gpt-5.4-nano (sem raciocínio, `reasoning_effort="none"`) e o gpt-4o-mini.

| Método | gpt-4o-mini | gpt-5.4-nano | US$ por 1.000 questões (nano) |
|---|---|---|---|
| Resposta direta | 70,5% | 61,1% | 0,06 |
| CoT, 1.000 tokens | 75,3% | 76,0% | 0,21 |
| Self-Consistency, 21 amostras | 76,0% | 79,7% | 3,50 |
| MedAgents | 74,1% | 78,9% | 7,42 |
| **MedVote** (os dois modelos) | | **84,4%** | 4,10 |

- **MedAgents contra Self-Consistency:** o MedAgents fica abaixo nos dois modelos, sem diferença significativa (McNemar exato: p = 0,06 no gpt-4o-mini e p = 0,48 no gpt-5.4-nano).
- **MedVote contra o Self-Consistency do nano:** o MedVote acerta 97 questões que ele erra e erra 37 que ele acerta (p < 0,001).
- **MedVote contra o MedAgents do nano:** +136 / −65 (p < 0,001).
- **Questões com quase-cópia no treino:** sem as 40 questões de teste que têm uma (Jaccard ≥ 0,6), o MedVote fica em 84,3% e o Self-Consistency do nano em 79,5%.

**O que explica o MedVote:**
- **O ganho vem do diagnóstico diferencial com os livros.** Essa etapa consertou 98 questões e estragou 35 em relação ao voto.
- **O gpt-4o-mini não aproveita a evidência**, por isso só o nano vota nessa etapa.
- **Misturar os dois modelos no voto não ajudou no teste**, embora tenha parecido ajudar na validação.

## Estrutura

**Código original, modificado.** Todas as mudanças são compatíveis: sem as opções novas, o comportamento é o do original.

| Arquivo | Mudança |
|---|---|
| `run.py` | Execução paralela e retomável; métodos de base (`base_direct`, `base_cot`, `base_sc`); revisor Claude (`syn_verif_review`); debate (`persona_debate`); opções de correção (`--analysis_max_tokens`, `--advice_max_tokens`, `--fix_key_knowledge`, `--stop_at_voted`); chaves lidas de `.env`. |
| `utils.py`, `data_utils.py` | Registro da resposta de cada rodada e do uso de tokens; Self-Consistency; correção opcional da leitura do relatório. |
| `prompt_generator.py` | Prompt do revisor externo. |
| `api_utils.py` | Tabela de modelos (OpenAI ou servidor local compatível), novas tentativas, contagem de tokens por questão e raciocínio desligado no gpt-5.4-nano. |

**Código novo.**

| Arquivo | Para quê |
|---|---|
| `analyze.py` | Tabelas de acurácia, intervalos de Wilson, McNemar, custo e análise por rodada; grava `analysis/results.json`. |
| `debate.py`, `analyze_debate.py` | Debate entre os especialistas com topologia configurável e a análise dele. |
| `claude_reviewer.py` | Claude como revisor externo do relatório do MedAgents (implementado, ainda não executado). |
| `import_legacy.py` | Converte saídas no formato do repositório original para o formato novo. |
| `retrieval.py` | BM25 sobre os livros-texto do MedQA e sobre as questões do treino, com filtro de quase-cópias. |
| `medvote.py`, `analyze_medvote.py` | MedVote: etapa de voto (`vote`), etapa de diferencial (`diff`) e análise de misturas, variantes e limiares. |
| `baixar_dados.py` | Baixa o MedQA original (treino, validação, teste) e os livros-texto para `ext_data/`. |
| `inference.ps1`, `run_experiments.ps1` | Atalhos em PowerShell para as execuções das partes 1 e 2. |
| `report/` | Scripts que geram as páginas HTML dos relatórios a partir das saídas. |
| `analysis/*.json` | Números de todas as tabelas, para conferir sem rodar nada. |

As saídas brutas (`outputs/`, cerca de 190 MB) e os dados baixados (`ext_data/`, cerca de 280 MB) não estão no repositório.

## Instalação

Python 3.10 ou mais recente.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   ·   Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab')"
```

Copie `.env.exemplo` para `.env` e preencha `OPENAI_API_KEY`. `ANTHROPIC_API_KEY` só é preciso para o revisor Claude. O `.env` está no `.gitignore`.

## Como reproduzir

### Partes 1 e 2: métodos de base e MedAgents

Saídas em `outputs/MedQA/<modelo>/`. Os comandos para o gpt-5.4-nano são os mesmos, com `--model_name gpt-5.4-nano`.

```bash
python run.py --model_name gpt-4o-mini --method syn_verif --workers 8
python run.py --model_name gpt-4o-mini --method base_sc --cot_max_tokens 1000 --sc_samples 21
python run.py --model_name gpt-4o-mini --method base_cot --cot_max_tokens 1000 --tag cot1000
python analyze.py
```

No Windows, `run_experiments.ps1` roda tudo em sequência. As execuções são retomáveis: se uma cair, rode de novo e ela continua de onde parou.

### Parte 3: debate e correções (pilotos)

```bash
# debate entre os 7 especialistas, nas topologias completa, anel e anel com Conformist
python run.py --method persona_debate --model_name gpt-4o-mini --cot_max_tokens 1000 --end_pos 20
# com modelos alternados entre os agentes
python run.py --method persona_debate --model_name gpt-4o-mini --agent_models gpt-4o-mini,gpt-5.4-nano \
    --output_files_folder ./outputs/MedQA/hetero --end_pos 20
python analyze_debate.py

# MedAgents com as correções de truncamento
python run.py --method syn_verif --model_name gpt-4o-mini --analysis_max_tokens 1000 --advice_max_tokens 1000 \
    --fix_key_knowledge --stop_at_voted --tag fix --end_pos 30
```

### Parte 4: MedVote

```bash
python baixar_dados.py
python medvote.py vote --split validation --limit 300 --workers 16
python medvote.py diff --split validation --limit 300 --workers 12
python analyze_medvote.py --split validation
# teste: rodado uma vez, com a configuração fixada na validação
python medvote.py vote --split test --limit 0 --workers 16
python medvote.py diff --split test --limit 0 --workers 12
python analyze_medvote.py --split test
```

**Configuração fixada:**
- **Voto:** 11 amostras do gpt-5.4-nano e 10 do gpt-4o-mini, com temperatura 0,7 e até 1.000 tokens.
- **Quando há diferencial:** quando menos de 90% dos votos concordam.
- **Evidência:** trechos dos livros-texto e questões resolvidas do treino.
- **Votação do diferencial:** 3 amostras do nano.

Os scripts gravam tudo: todas as amostras e as três variantes de evidência nos dois modelos. Assim, misturas, limiares e variantes podem ser comparados depois, sem novas chamadas. Por isso o teste custou US$ 12,54; só a configuração fixada custaria cerca de US$ 5.

### Páginas dos relatórios

```bash
python report/build_report.py      # MedAgents contra Self-Consistency
python report/build_topologias.py  # variações do MedAgents
python report/build_medvote.py     # MedVote
```

## Custos medidos

Preços por 1M de tokens (entrada / saída): gpt-4o-mini US$ 0,15 / 0,60; gpt-5.4-nano US$ 0,20 / 1,25.

| Execução | Custo |
|---|---|
| gpt-4o-mini, 5 métodos, 1.273 questões | US$ 9,69 |
| gpt-5.4-nano, 5 métodos, 1.273 questões | US$ 14,67 |
| MedVote, piloto na validação (300 questões) | US$ 2,84 |
| MedVote, teste (1.273 questões, tudo gravado) | US$ 12,54 |

## Ressalvas

- **Debate e correções do MedAgents:** só foram rodados em pilotos de 20 a 30 questões. Mostram diferenças de custo, mas não de acurácia.
- **Uso do teste:** o MedVote foi avaliado no teste uma única vez. Versões mais simples sugeridas pelo resultado (só o nano no voto, só os livros como evidência) não foram testadas, porque o teste já foi usado.
- **Modelos:** só modelos pequenos e sem raciocínio.

## Créditos

- **MedAgents:** Tang, Zou, Zhang, Zhao, Zhang, Cohan e Gerstein. *MedAgents: Large Language Models as Collaborators for Zero-shot Medical Reasoning*. ACL 2024 Findings. [arXiv:2311.10537](https://arxiv.org/abs/2311.10537). O repositório original não declara licença; este fork existe nos termos de fork do GitHub, e o código original pertence aos seus autores.
- **MedQA:** Jin et al., 2021.
- **Livros-texto em trechos:** MedRAG/textbooks, Xiong et al., 2024.
- **Ideias usadas no debate e no MedVote:**
  - Choi et al., *Debate or Vote*;
  - Li et al., topologias esparsas;
  - Zhang et al., Heter-MAD;
  - Zhao et al., MedRAG com grafo de conhecimento;
  - o survey *How far are we from Baymax*.
