import openai
import time
import random
from wrapt_timeout_decorator import timeout

openai.api_type = "azure"
openai.api_base = ""
openai.api_version = ""
openai.api_key = ""


@timeout(200) # 200 seconds timeout
def generate_response_multiagent(engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, system_role, user_input):
    print("Generating response for engine: ", engine)
    start_time = time.time()
    response = openai.ChatCompletion.create(
                    engine=engine, # engine is the name of the deployment
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1, # top_p的意思是选择概率质量值之和达到top_p的概率分布采样结果
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    messages=[  
                        {"role": "system", "content": system_role},
                        {"role": "user", "content": user_input}
                    ],
                )
    end_time = time.time()
    print('Finish!')
    print("Time taken: ", end_time - start_time)

    return response

@timeout(10) # 10 seconds timeout
def generate_response(engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, input_text):
    print("Generating response for engine: ", engine)
    start_time = time.time()
    response = openai.ChatCompletion.create(
                    engine=engine, # engine is the name of the deployment
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=1, # top_p的意思是选择概率质量值之和达到top_p的概率分布采样结果
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                    stop=stop,
                    messages=[{"role": "user", "content": input_text}],
                )
    end_time = time.time()
    print('Finish!')
    print("Time taken: ", end_time - start_time)

    return response

@timeout(20) # 20 seconds timeout
def generate_response_ins(engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, input_text, suffix, echo):
    print("Generating response for engine: ", engine)
    start_time = time.time()
    response = openai.Completion.create(
                        engine=engine,
                        prompt=input_text,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=1,
                        suffix=suffix,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        stop=stop,
                        echo=echo,
                        logprobs=1,
                    )
    end_time = time.time()
    print('Finish!')
    print("Time taken: ", end_time - start_time)

    return response

class api_handler:
    def __init__(self, model):
        self.model = model

        if self.model == 'instructgpt':
            self.engine = 'text-davinci-002'
        elif self.model == 'instructgpt-gen':
            self.engine = 'text-davinci-002'
        elif self.model == 'newinstructgpt':
            self.engine = 'text-davinci-003'
        elif self.model == 'oldinstructgpt':
            self.engine = 'text-davinci-001'
        elif self.model == 'gpt3':
            self.engine = 'davinci'
        elif self.model == 'codex':
            self.engine = 'code-davinci-002'
        elif self.model == 'gpt3-edit':
            self.engine = 'text-davinci-edit-001'
        elif self.model == 'codex-edit':
            self.engine = 'code-davinci-edit-001'
        elif self.model == 'chatgpt':
            self.engine = 'gpt-35-turbo-16k'
        elif self.model == 'gpt4':
            self.engine = 'gpt-4'
        else:
            raise NotImplementedError

    def get_output_multiagent(self, system_role, user_input, max_tokens, temperature=0,
                    frequency_penalty=0, presence_penalty=0, stop=None):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = generate_response_multiagent(self.engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, system_role, user_input)
                if response.choices and response.choices[0].message and "content" in response.choices[0].message:
                    return response.choices[0].message["content"]
                else:
                    return "ERROR." 
            except (TimeoutError, openai.error.Timeout, Exception) as error:
                print(f'Attempt {attempt+1} of {max_attempts} failed with error: {error}')
                if attempt == max_attempts - 1:
                    return "ERROR."


    def get_output(self, input_text, max_tokens, temperature=0,
                   suffix=None, stop=None, do_tunc=False, echo=False, ban_pronoun=False,
                   frequency_penalty=0, presence_penalty=0, return_prob=False):
        try:
            response = generate_response(self.engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, input_text)
        except (TimeoutError, openai.error.Timeout, Exception):    
            print("Timeout")
            try:
                response = generate_response(self.engine, temperature, max_tokens, frequency_penalty, presence_penalty, stop, input_text)
            except (TimeoutError, openai.error.Timeout, Exception):
                print("Timeout occurred again. Exiting.")
                response = "ERROR."
                return response  # 直接返回空字符串
        if response.choices and response.choices[0].message and "content" in response.choices[0].message:
            x = response.choices[0].message["content"]
        else:
            print(response)
            x = "ERROR."  # 或者设置一个默认值 防止生成的response没有content造成问题
            return x


        if do_tunc: # do_tunc的意思是是否要截断 保证返回的值里没有换行符，Q:，Question:等
            y = x.strip() # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
            if '\n' in y:
                pos = y.find('\n') # 这里的意思是找到第一个换行符的位置
                y = y[:pos] # 这里的意思是把第一个换行符之前的内容保留
            if 'Q:' in y:
                pos = y.find('Q:')
                y = y[:pos]
            if 'Question:' in y:
                pos = y.find('Question:')
                y = y[:pos]
            assert not ('\n' in y)
            if not return_prob:
                return y

        if not return_prob:
            return x

        # pdb.set_trace()
        output_token_offset_real, output_token_tokens_real, output_token_probs_real = [], [], []
        return x, (output_token_offset_real, output_token_tokens_real, output_token_probs_real)

"""
(Pdb) x
' Academy Award because The Curious Case of Benjamin Button won three Academy Awards, which are given by the Academy of Motion Picture Arts and Sciences.'
(Pdb) output_token_offset_real
[0, 8, 14, 22, 26, 34, 39, 42, 51, 58, 62, 68, 76, 83, 84, 90, 94, 100, 103, 107, 115, 118, 125, 133, 138, 142, 151]
(Pdb) output_token_tokens_real
[' Academy', ' Award', ' because', ' The', ' Curious', ' Case', ' of', ' Benjamin', ' Button', ' won', ' three', ' Academy', ' Awards', ',', ' which', ' are', ' given', ' by', ' the', ' Academy', ' of', ' Motion', ' Picture', ' Arts', ' and', ' Sciences', '.']
(Pdb) output_token_probs_real
[-0.7266144, -0.68505085, -0.044669915, -0.00023392851, -0.0021017971, -2.1768952e-05, -1.1430258e-06, -6.827632e-08, -3.01145e-05, -1.2231317e-05, -0.07086051, -2.7967804e-05, -6.6619094e-07, -0.41155097, -0.0020535963, -0.0021325003, -0.6671403, -0.51776046, -0.00014945272, -0.41470888, -3.076318e-07, -3.583558e-05, -2.9311614e-06, -3.869565e-05, -1.1430258e-06, -9.606849e-06, -0.017712338]
"""

        # except Exception as e:
        #     if 'You exceeded your current quota, please check your plan and billing details.' in str(e):
        #         print("Exit because no quota")
        #         exit()
        #     time.sleep(2 * self.interval)
        #     return self.get_output(input_text, max_tokens, temperature=temperature,
        #            suffix=suffix, stop=stop, do_tunc=do_tunc, echo=echo, ban_pronoun=ban_pronoun,
        #            frequency_penalty=frequency_penalty, presence_penalty=presence_penalty, return_prob=return_prob)
