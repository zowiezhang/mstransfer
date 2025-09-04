import os
import time
import sys

from openai import OpenAI, AsyncOpenAI, OpenAIError
from dymat.logs import logger
from openai import AzureOpenAI
import asyncio 
from openai_keys import OPENAI_API_KEY, OPENAI_BASE_URL, AZURE_API_KEY, AZURE_REGION, AZURE_API_BASE, DEEPSEEK_API_KEY, DEEPSEEK_API_URL, V3_API_KEY, V3_API_URL


llm_type = 'gpt-4o-mini'
RATE_LIMIT = 5  # Max calls per second  
rate_limiter = asyncio.Semaphore(RATE_LIMIT)  

gpt4o_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)  
gpt35_client = AzureOpenAI(  
    api_key=AZURE_API_KEY,  
    api_version="2024-02-01",  
    azure_endpoint=f"{AZURE_API_BASE}/{AZURE_REGION}"  
)  
deepseek_client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL) 


async def call_models(sysprompt = "You are a helpful assistant.", prompt = '', max_token = 200, temperature = 0.0, top_p = 0.95, seed = 42, backoff_factor = 2, retries = 3):       
    
    async with rate_limiter: 
        for attempt in range(retries):  
            try:  
                full_response = ""  # 用于聚合完整响应  
                if llm_type == 'gpt-4o':   
                    stream = await asyncio.wait_for(  
                        gpt4o_client.chat.completions.create(  
                            model="gpt-4o-2024-11-20",  
                            messages=[  
                                {"role": "system", "content": sysprompt},  
                                {"role": "user", "content": f"{prompt}"}  
                            ],  
                            max_tokens=max_token,  
                            temperature=temperature,  
                            top_p=top_p,  
                            seed=seed,  
                            stream=True  # 启用流式  
                        ),  
                        timeout=60  
                    )  
                    async for chunk in stream:  
                        chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""  # extract the message
                        # log_llm_stream(chunk_message)
                        full_response += chunk_message   
                
                elif llm_type == 'gpt-3.5-turbo':   
                    stream = await asyncio.wait_for(  
                        gpt35_client.chat.completions.create(  
                            model="gpt-35-turbo-0125",  
                            messages=[  
                                {"role": "system", "content": sysprompt},  
                                {"role": "user", "content": f"{prompt}"}  
                            ],  
                            max_tokens=max_token,  
                            temperature=temperature,  
                            top_p=top_p,  
                            seed=seed,  
                            stream=True  # 启用流式  
                        ),  
                        timeout=60  
                    )  
                    async for chunk in stream:  
                        chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""  # extract the message
                        # log_llm_stream(chunk_message)
                        full_response += chunk_message  
                
                elif llm_type == 'gpt-4o-mini':    
                    stream = await asyncio.wait_for(  
                        gpt4o_client.chat.completions.create(  
                            model="gpt-4o-mini-2024-07-18",  
                            messages=[  
                                {"role": "system", "content": sysprompt},  
                                {"role": "user", "content": f"{prompt}"}  
                            ],  
                            max_tokens=max_token,  
                            temperature=temperature,  
                            top_p=top_p,  
                            seed=seed,  
                            stream=True  # 启用流式  
                        ),  
                        timeout=60  
                    )  
                    async for chunk in stream:  
                        chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""  # extract the message
                        # log_llm_stream(chunk_message)
                        full_response += chunk_message 
                
                elif llm_type == 'deepseek':  
                    stream = await asyncio.wait_for(  
                        deepseek_client.chat.completions.create(  
                            model="deepseek-r1-250528",  
                            messages=[  
                                {"role": "system", "content": sysprompt},  
                                {"role": "user", "content": f"{prompt}"}  
                            ],  
                            max_tokens=max_token,  
                            temperature=temperature,  
                            top_p=top_p,  
                            seed=seed,  
                            stream=True  # 启用流式  
                        ),  
                        timeout=60  
                    )  
                    async for chunk in stream:  
                        chunk_message = chunk.choices[0].delta.content or "" if chunk.choices else ""  # extract the message
                        # log_llm_stream(chunk_message)
                        full_response += chunk_message 
                
                else:  
                    raise ValueError(f"Unsupported llm_type: {llm_type}")  
                
                return full_response  # 返回聚合后的完整响应   
            
            except asyncio.TimeoutError:  
                logger.error(f"Attempt {attempt + 1} timed out after 60 seconds")  
                if attempt == retries - 1:  
                    raise  
                backoff = backoff_factor ** attempt  
                await asyncio.sleep(backoff)  
            except OpenAIError as e:  
                logger.error(f"Attempt {attempt + 1} failed with error: {e}")  
                if attempt == retries - 1:  
                    raise  
                backoff = backoff_factor ** attempt  
                await asyncio.sleep(backoff)  
            except Exception as e:  
                logger.error(f"Unexpected error: {e}") 
                if attempt == retries - 1:  
                    raise  
                backoff = backoff_factor ** attempt  
                await asyncio.sleep(backoff)  
                
        


