import re
import os
import time
from openai import OpenAI
from openai import AzureOpenAI
from openai_key_info import OPENAI_API_KEY, OPENAI_BASE_URL, AZURE_API_KEY, AZURE_REGION, AZURE_API_BASE

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

get_rephrase_prompt = '''Please rephrase the following sentence in {} significantly different ways, with each sentence enclosed in []. The output should only contain the {} sentences:
{}'''

use_azure = True


def gen_chatgpt_outputs(prompt, sysprompt = "You are a helpful assistant.", max_token = 200, temperature = 0.0, top_p = 0.95, seed = 42):      
    while True:
        try:
            if use_azure:
                ENDPOINT = f"{AZURE_API_BASE}/{AZURE_REGION}"
                client = AzureOpenAI(api_key = AZURE_API_KEY, api_version = "2024-02-01", azure_endpoint = ENDPOINT)
                completion = client.chat.completions.create(
                                    model = "gpt-4o-2024-05-13",
                                    messages = [
                                        {"role": "system", "content": sysprompt},
                                        {"role": "user", "content": f"{prompt}"}
                                    ],
                                    max_tokens = max_token,
                                    temperature = temperature,
                                    top_p = top_p,
                                    seed = seed 
                                )
            else:
                completion = OpenAI().chat.completions.create(
                                    model = "gpt-4o-2024-05-13",
                                    messages = [
                                        {"role": "system", "content": sysprompt},
                                        {"role": "user", "content": f"{prompt}"}
                                    ],
                                    max_tokens = max_token,
                                    temperature = temperature,
                                    top_p = top_p,
                                    seed = seed 
                                )
            break
        except Exception as e:
            print(e)
            time.sleep(5)

    return completion.choices[0].message.content

def prompt_rephraser(prompt, num_rephrases):
    '''
    Using '[]' as the delimiter, rephrase the response in num_rephrases different ways. 
    Return in lists of rephrased prompts.
    '''
    if num_rephrases <= 1:
        return [prompt]

    num_rephrases -= 1
    rephrase_prompt = get_rephrase_prompt.format(num_rephrases, num_rephrases, prompt)

    text = gen_chatgpt_outputs(rephrase_prompt)
    # text = "[You ought to strive for a writing style reminiscent of Shakespeare.] [Aim to emulate the eloquence and style of Shakespeare in your writing.] [Endeavor to infuse your writing with the same artistry and flair as Shakespeare's works.]"
    matches = re.findall(r'\[(.*?)\]', text)
    matches.append(prompt)
    # print(matches)
    return matches

# print(prompt_rephraser('hello', 2))
