
NUM_QD = 5
NUM_OD = 2

# question domains
def get_question_domains_prompt(question):
    question_domain_format = "Medical Field: " + " | ".join(["Field" + str(i) for i in range(NUM_QD)])
    question_classifier = "You are a medical expert who specializes in categorizing a specific medical scenario into specific areas of medicine."
    prompt_get_question_domain = f"You need to complete the following steps:" \
            f"1. Carefully read the medical scenario presented in the question: '''{question}'''. \n" \
            f"2. Based on the medical scenario in it, classify the question into five different subfields of medicine. \n" \
            f"3. You should output in exactly the same format as '''{question_domain_format}'''."
    return question_classifier, prompt_get_question_domain



def get_question_analysis_prompt(question, question_domain):
    question_analyzer = f"You are a medical expert in the domain of {question_domain}. " \
        f"From your area of specialization, you will scrutinize and diagnose the symptoms presented by patients in specific medical scenarios."
    prompt_get_question_analysis = f"Please meticulously examine the medical scenario outlined in this question: '''{question}'''." \
                        f"Drawing upon your medical expertise, interpret the condition being depicted. " \
                        f"Subsequently, identify and highlight the aspects of the issue that you find most alarming or noteworthy."

    return question_analyzer, prompt_get_question_analysis

def get_options_domains_prompt(question, options):
    options_domain_format =  "Medical Field: " + " | ".join(["Field" + str(i) for i in range(NUM_OD)])
    options_classifier = f"As a medical expert, you possess the ability to discern the two most relevant fields of expertise needed to address a multiple-choice question encapsulating a specific medical context."
    prompt_get_options_domain = f"You need to complete the following steps:" \
                f"1. Carefully read the medical scenario presented in the question: '''{question}'''." \
                f"2. The available options are: '''{options}'''. Strive to understand the fundamental connections between the question and the options." \
                f"3. Your core aim should be to categorize the options into two distinct subfields of medicine. " \
                f"You should output in exactly the same format as '''{options_domain_format}'''"
    return options_classifier, prompt_get_options_domain


def get_options_analysis_prompt(question, options, op_domain, question_analysis):
    option_analyzer = f"You are a medical expert specialized in the {op_domain} domain. " \
                f"You are adept at comprehending the nexus between questions and choices in multiple-choice exams and determining their validity. " \
                f"Your task, in particular, is to analyze individual options with your expert medical knowledge and evaluate their relevancy and correctness."

    prompt_get_options_analyses = f"Regarding the question: '''{question}''', we procured the analysis of five experts from diverse domains. \n"
    for _domain, _analysis in question_analysis.items():
        prompt_get_options_analyses += f"The evaluation from the {_domain} expert suggests: {_analysis} \n"
        prompt_get_options_analyses += f"The following are the options available: '''{options}'''." \
                    f"Reviewing the question's analysis from the expert team, you're required to fathom the connection between the options and the question from the perspective of your respective domain, " \
                    f"and scrutinize each option individually to assess whether it is plausible or should be eliminated based on reason and logic. "\
                    f"Pay close attention to discerning the disparities among the different options and rationalize their existence. " \
                    f"A handful of these options might seem right on the first glance but could potentially be misleading in reality."
    return option_analyzer, prompt_get_options_analyses


def get_final_answer_prompt_analonly(question, options, question_analyses, option_analyses):
    prompt = f"Question: {question} \nOptions: {options} \n" \
        f"Answer: Let's work this out in a step by step way to be sure we have the right answer. \n" \
        f"Step 1: Decode the question properly. We have a team of experts who have done a detailed analysis of this question. " \
        f"The team includes five experts from different medical domains related to the problem. \n"
    
    for _domain, _analysis in question_analyses.items():
        prompt += f"Insight from an expert in {_domain} suggests, {_analysis} \n"
    
    prompt += f"Step 2: Evaluate each presented option individually, based on both the specifics of the patient's scenario as well as your medical knowledge. " \
            f"Pay close attention to discerning the disparities among the different options. " \
            f"A handful of these options might seem right on the first glance but could potentially be misleading in reality. " \
            f"We have detailed analyses from experts across two domains. \n"
    
    for _domain, _analysis in option_analyses.items():
        prompt += f"Assessment from an expert in {_domain} suggests, {_analysis} \n"
    prompt += f"Step 3: Based on the understanding gathered from the above steps, select the optimal choice to answer the question. \n" \
        f"Points to note: \n" \
        f"1. The analyses provided should guide you towards the correct response. \n" \
        f"2. Any option containing incorrect information inherently cannot be the correct choice. \n" \
        f"3. Please respond only with the selected option's letter, like A, B, C, D, or E, using the following format: '''Option: [Selected Option's Letter]'''. " \
        f"Remember, it's the letter we need, not the full content of the option."

    return prompt

def get_final_answer_prompt_wsyn(syn_report):
    prompt = f"Here is a synthesized report: {syn_report} \n" \
        f"Based on the above report, select the optimal choice to answer the question. \n" \
        f"Points to note: \n" \
        f"1. The analyses provided should guide you towards the correct response. \n" \
        f"2. Any option containing incorrect information inherently cannot be the correct choice. \n" \
        f"3. Please respond only with the selected option's letter, like A, B, C, D, or E, using the following format: '''Option: [Selected Option's Letter]'''. " \
        f"Remember, it's the letter we need, not the full content of the option."
    return prompt


def get_direct_prompt(question, options):
    prompt = f"Question: {question} \n" \
        f"Options: {options} \n" \
        f"Please respond only with the selected option's letter, like A, B, C, D, or E, using the following format: '''Option: [Selected Option's Letter]'''."
    return prompt

def get_cot_prompt(question, options):
    cot_format = f"Thought: [the step-by-step thoughts] \n" \
                f"Answer: [Selected Option's Letter (like A, B, C, D, or E)] \n"
    prompt = f"Question: {question} \n" \
        f"Options: {options} \n" \
        f"Answer: Let's work this out in a step by step way to be sure we have the right answer. " \
        f"You should output in exactly the same format as '''{cot_format}'''"
    return prompt


def get_synthesized_report_prompt(question_analyses, option_analyses):
    synthesizer = "You are a medical decision maker who excels at summarizing and synthesizing based on multiple experts from various domain experts."

    syn_report_format = f"Key Knowledge: [extracted key knowledge] \n" \
                f"Total Analysis: [synthesized analysis] \n"
    prompt = f"Here are some reports from different medical domain experts.\n "
    prompt += f"You need to complete the following steps:" \
                f"1. Take careful and comprehensive consideration of the following reports." \
                f"2. Extract key knowledge from the following reports. " \
                f"3. Derive the comprehensive and summarized analysis based on the knowledge." \
                f"4. Your ultimate goal is to derive a refined and synthesized report based on the following reports." \
                f"You should output in exactly the same format as '''{syn_report_format}'''"
    prompt += question_analyses
    prompt += option_analyses
    
    return synthesizer, prompt


def get_consensus_prompt(domain, syn_report):
    voter = f"You are a medical expert specialized in the {domain} domain."
    cons_prompt = f"Here is a medical report: {syn_report} \n"\
        f"As a medical expert specialized in {domain}, please carefully read the report and decide whether your opinions are consistent with this report." \
        f"Please respond only with: [YES or NO]."
    return voter, cons_prompt


def get_consensus_opinion_prompt(domain, syn_report):
    opinion_prompt = f"Here is a medical report: {syn_report} \n"\
        f"As a medical expert specialized in {domain}, please make full use of your expertise to propose revisions to this report." \
        f"You should output in exactly the same format as '''Revisions: [proposed revision advice] '''"
    return opinion_prompt


#revision_prompt = get_revision_prompt(revision_advice)

def get_revision_prompt(syn_report, revision_advice):
    revision_prompt = f"Here is the original report: {syn_report}\n\n"
    for domain, advice in revision_advice.items():
        revision_prompt += f"Here is advice from a medical expert specialized in {domain}: {advice}.\n"
    revision_prompt += f"Based on the above advice, output the revised analysis in exactly the same format as '''Total Analysis: [revised analysis] '''"
    return revision_prompt