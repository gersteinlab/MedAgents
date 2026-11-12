"""Debate entre os especialistas do MedAgents, com topologia de comunicação configurável.

No MedAgents original os especialistas nunca leem uns aos outros: escrevem análises, um
sintetizador junta tudo num relatório e eles só votam sim/não nesse relatório (topologia em
estrela, centralizada). Aqui os mesmos especialistas debatem diretamente, como em:

  - Du et al. (2023), Society of Minds: cada agente responde, depois revisa lendo os outros;
  - Li et al. (2024), sparse topology: cada agente lê só os vizinhos (anel) em vez de todos;
  - Choi et al. (2025), Debate or Vote: o voto majoritário da rodada 0 é o baseline que o
    debate precisa superar; a regra "Conformist" mantém a resposta de quem já concorda com
    a maioria (e, de quebra, poupa a chamada);
  - Zhang et al. (2025), Heter-MAD: agentes podem usar modelos diferentes.

Todas as topologias partem da MESMA rodada 0 (redução de variância de Li et al., seção 4.5),
então as diferenças entre elas vêm só da comunicação. Cada registro guarda as respostas de
todos os agentes em todas as rodadas, e o custo de cada parte separadamente.
"""
from collections import Counter

from prompt_generator import get_question_domains_prompt, get_options_domains_prompt, get_cot_prompt
from data_utils import cleansing_final_output

TOPOLOGIES = ('full', 'ring', 'ring_conf')


def majority(answers):
    valid = [a for a in answers if a]
    if not valid:
        return ""
    c = Counter(valid)
    best = max(c.values())
    return next(a for a in valid if c[a] == best)


def neighbors(i, n, topology):
    if topology == 'full':
        return [j for j in range(n) if j != i]
    # anel: os dois vizinhos (densidade 2/(n-1), a mais esparsa de Li et al.)
    return [(i - 1) % n, (i + 1) % n]


def persona_role(domain):
    # mesmo papel que o MedAgents dá aos especialistas na análise da questão
    return (f"You are a medical expert in the domain of {domain}. From your area of specialization, "
            f"you will scrutinize and diagnose the symptoms presented by patients in specific medical scenarios.")


def debate_prompt(question, options, own, others):
    # modelo de prompt de Choi et al. (apêndice B.1), com o formato de resposta do nosso CoT
    parts = ["These are the recent opinions from other agents:"]
    for o in others:
        parts.append(f"One of the agents' response:\n{o}")
    parts.append(f"This was your most recent opinion:\n{own}")
    parts.append("Use these opinions carefully as additional advice to revise your recent opinion "
                 "to give your final answer to the question:")
    parts.append(get_cot_prompt(question, options))
    return "\n\n".join(parts)


def parse_domains(raw, k, fallback="General Medicine"):
    if not raw or raw == "ERROR.":
        return [fallback] * k
    doms = [d.strip(" .'\"*") for d in raw.split(":")[-1].strip().split("|")]
    doms = [d for d in doms if d]
    return (doms + [fallback] * k)[:k]


def _snapshot(handlers):
    return {h.model: dict(h.usage()) for h in handlers}


def _diff(after, before):
    return {m: {k: after[m].get(k, 0) - before.get(m, {}).get(k, 0) for k in after[m]} for m in after}


def persona_debate(question, options, handlers, args):
    """handlers: um por modelo; o agente i usa handlers[i % len(handlers)].
    A escolha dos especialistas usa sempre o primeiro modelo."""
    n = args.n_agents
    main = handlers[0]
    snap0 = _snapshot(handlers)

    # 1) especialistas, com os mesmos prompts do MedAgents (5 pela questão + 2 pelas alternativas)
    role_q, p_q = get_question_domains_prompt(question)
    role_o, p_o = get_options_domains_prompt(question, options)
    q_dom = parse_domains(main.get_output_multiagent(system_role=role_q, user_input=p_q, max_tokens=50, temperature=0), 5)
    o_dom = parse_domains(main.get_output_multiagent(system_role=role_o, user_input=p_o, max_tokens=50, temperature=0), 2)
    agents = (q_dom + o_dom + ["General Medicine"] * n)[:n]
    agent_handlers = [handlers[i % len(handlers)] for i in range(n)]

    # 2) rodada 0: cada especialista responde sozinho (temperatura > 0, como no Self-Consistency)
    cot = get_cot_prompt(question, options)
    r0_out = [h.get_output_multiagent(system_role=persona_role(d), user_input=cot,
                                      max_tokens=args.cot_max_tokens, temperature=args.sc_temperature)
              for d, h in zip(agents, agent_handlers)]
    r0_ans = [cleansing_final_output(o)[0] if o != "ERROR." else "" for o in r0_out]
    if all(o == "ERROR." for o in r0_out):
        return None
    snap1 = _snapshot(handlers)

    # 3) debate, uma vez para cada topologia, sempre a partir da mesma rodada 0
    debates, usage_by_part = {}, {'shared': _diff(snap1, snap0)}
    for topo in args.topologies:
        before = _snapshot(handlers)
        outs, answers, calls = [r0_out], [r0_ans], []
        for _ in range(args.debate_rounds):
            prev_out, prev_ans = outs[-1], answers[-1]
            maj = majority(prev_ans)
            new_out, new_ans, made = [], [], 0
            for i in range(n):
                if topo == 'ring_conf' and prev_ans[i] and prev_ans[i] == maj:
                    new_out.append(prev_out[i]); new_ans.append(prev_ans[i])  # Conformist: mantém
                    continue
                others = [prev_out[j] if prev_out[j] != "ERROR." else "(no response)"
                          for j in neighbors(i, n, topo)]
                own = prev_out[i] if prev_out[i] != "ERROR." else "(no response)"
                o = agent_handlers[i].get_output_multiagent(
                    system_role=persona_role(agents[i]), user_input=debate_prompt(question, options, own, others),
                    max_tokens=args.cot_max_tokens, temperature=args.sc_temperature)
                made += 1
                if o == "ERROR.":  # falha de API: o agente mantém a resposta anterior
                    new_out.append(prev_out[i]); new_ans.append(prev_ans[i])
                else:
                    new_out.append(o); new_ans.append(cleansing_final_output(o)[0])
            outs.append(new_out); answers.append(new_ans); calls.append(made)
        debates[topo] = {'answers': answers, 'outputs': outs[1:], 'calls_per_round': calls,
                         'final_by_round': [majority(a) for a in answers]}
        usage_by_part[topo] = _diff(_snapshot(handlers), before)

    return {
        'agents': agents,
        'agent_models': [h.model for h in agent_handlers],
        'round0_outputs': r0_out,
        'vote_answer': majority(r0_ans),
        'debates': debates,
        'usage_by_part': usage_by_part,
        'n_agents': n, 'debate_rounds': args.debate_rounds,
        'temperature': args.sc_temperature, 'cot_max_tokens': args.cot_max_tokens,
    }
