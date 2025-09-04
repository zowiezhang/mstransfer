import time
import random
from utils.rephraser import gen_chatgpt_outputs



# eval_prompt_template = '''
# <|im_start|>system
# You are a helpful assistant that ranks models based on the degree to which their responses align with the given preferences.
# <|im_end|>
# <|im_start|>user
# Which of the following responses answers the given question while better aligning with the specified preferences, without including unnecessary or irrelevant details?

# Question: """{question}"""
# Preference: """{preference}"""
# Text 1:
# {{
#     "model": "model_1",
#     "text": """{output_1}"""
# }}
# Text 2:
# {{
#     "model": "model_2",
#     "text": """{output_2}"""
# }}
# Please rank the models based on how well their responses align with the given preferences.
# Then only return an option in [Text 1, Text 2, Tie].
# Please provide the ranking that the majority of humans would give.
# <|im_end|>
# '''

# eval_prompt_template = '''
# <|im_start|>system
# You are a helpful assistant that ranks models based on the degree to which their responses align with the given question.
# <|im_end|>
# <|im_start|>user
# Which of the following responses answers the given question better?

# Question: """{question}"""

# Text 1:
# {{
#     "model": "model_1",
#     "text": """{output_1}"""
# }}
# Text 2:
# {{
#     "model": "model_2",
#     "text": """{output_2}"""
# }}
# Please rank the models based on how well their responses align with the given question.
# Then only return an option in [Text 1, Text 2, Tie].
# Please provide the ranking that the majority of humans would give.
# <|im_end|>
# '''

eval_prompt_template = '''
Which of the following responses answers the given question while better aligning with the specified preferences, without including unnecessary or irrelevant details?

Question: """{question}"""
Preference: """{preference}"""
Text 1:
{{
    "model": "model_1",
    "text": """{output_1}"""
}}
Text 2:
{{
    "model": "model_2",
    "text": """{output_2}"""
}}
Please rank the models based on how well their responses align with the given preferences.
Then only return an option in [Text 1, Text 2, Tie].
Please provide the ranking that the majority of humans would give.
'''

# eval_prompt_template = '''
# Which of the following responses answers the given question better?

# Question: """{question}"""

# Text 1:
# {{
#     "model": "model_1",
#     "text": """{output_1}"""
# }}
# Text 2:
# {{
#     "model": "model_2",
#     "text": """{output_2}"""
# }}
# Please rank the models based on how well their responses align with the given question.
# Then only return an option in [Text 1, Text 2, Tie].
# Please provide the ranking that the majority of humans would give.
# '''






# eval_prompt_template = '''
# Which of the following responses the requirement prompts better?

# Prompts: """{question}"""

# Text 1:
# {{
#     "model": "model_1",
#     "text": """{output_1}"""
# }}
# Text 2:
# {{
#     "model": "model_2",
#     "text": """{output_2}"""
# }}
# Please rank the models based on how well their responses satisfy the requirement prompts.
# Then only return an option in [Text 1, Text 2, Tie].
# Please provide the ranking that the majority of humans would give.
# '''







# def gpt_winner_evaluator(ques, pref, text1, text2):
#     '''
#     Function to evaluate the quality of the text based on the preference.
#     Return whether winner is text 1 based on the preference.
#     '''
#     # eliminate the sequence bias
#     results = []
#     result = 0
#     for seq_flag in [0, 1]:
#         if seq_flag:
#             eval_prompt = eval_prompt_template.format(question=ques, preference=pref, output_1=text1, output_2=text2)
#         else:
#             eval_prompt = eval_prompt_template.format(question=ques, preference=pref, output_1=text2, output_2=text1)
    
#         while True:
#             text = gen_chatgpt_outputs(eval_prompt, sysprompt = 'You are a helpful assistant that ranks models based on the degree to which their responses align with the given preferences.', max_token = 10, temperature = 0)
    
#             if text not in ['[Text 1]', '[Text 2]', 'Text 1.', 'Text 2.', 'Text 1', 'Text 2', 'Tie', 'Tie.', '[Tie]']:
#                 time.sleep(5)
#                 eval_prompt += '\n You can ONLY RESPONSE IN [Text 1, TEXT 2, Tie]'
#             else:
#                 if text in ['Tie', 'Tie.', '[Tie]']: 
#                     result = 0
#                     break
#                 if seq_flag:
#                     result = 1 if text in ['[Text 1]', 'Text 1', 'Text 1.'] else -1
#                 else:
#                     result = -1 if text in ['[Text 1]', 'Text 1', 'Text 1.'] else 1
#                 break
#         results.append(result)
#     sumr = sum(results)
#     if sumr == 0:
#         return 0
#     elif sumr > 0:
#         return 1
#     else:
#         return -1

def gpt_winner_evaluator(ques, pref, text1, text2):
    '''
    Function to evaluate the quality of the text based on the preference.
    Return whether winner is text 1 based on the preference.
    '''
    # eliminate the sequence bias
    seq_flag = random.choice([0, 1])
    if seq_flag:
        eval_prompt = eval_prompt_template.format(question=ques, preference=pref, output_1=text1, output_2=text2)
    else:
        eval_prompt = eval_prompt_template.format(question=ques, preference=pref, output_1=text2, output_2=text1)
    # if seq_flag:
    #     eval_prompt = eval_prompt_template.format(question=ques+ '\n' + pref, output_1=text1, output_2=text2)
    # else:
    #     eval_prompt = eval_prompt_template.format(question=ques + '\n' + pref, output_1=text2, output_2=text1)

    while True:
        text = gen_chatgpt_outputs(eval_prompt, sysprompt = 'You are a helpful assistant that ranks models based on the degree to which their responses align with the given preferences.', max_token = 5, temperature = 0) # sysprompt = 'You are a helpful assistant that ranks models based on the degree to which their responses align with the given preferences.',  'You are a helpful assistant that ranks models based on the degree to which their responses satisfy the given requirement prompts.'

        if text not in ['[Text 1]', '[Text 2]', 'Text 1.', 'Text 2.', 'Text 1', 'Text 2', 'Tie', 'Tie.', '[Tie]']:
            time.sleep(5)
            eval_prompt += '\n You can ONLY RESPONSE IN [Text 1, TEXT 2, Tie]'
        else:
            if text in ['Tie', 'Tie.', '[Tie]']: 
                return 0
            if seq_flag:
                return 1 if text in ['[Text 1]', 'Text 1', 'Text 1.'] else -1
            return -1 if text in ['[Text 1]', 'Text 1', 'Text 1.'] else 1




