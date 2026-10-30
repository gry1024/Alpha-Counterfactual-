import backoff  # for exponential backoff
import openai
import json
import asyncio
from typing import Any
from retrying import retry

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIConnectionError))
def completions_with_backoff(client, **kwargs):
    return client.chat.completions.create(**kwargs)

async def dispatch_openai_chat_requests(
    client,
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop_words: list[str]
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
        stop_words: List of words to stop the model from generating.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        client.chat.completions.create(
            model=model,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop = stop_words
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


class OpenAIModel:
    def __init__(self, model_name, stop_words, max_new_tokens, request_max_tokens=None) -> None:
        self.model_name = model_name
        self.stop_words = stop_words
        self.max_new_tokens = max_new_tokens
        self.request_max_tokens = request_max_tokens

    @retry(stop_max_attempt_number=10, wait_fixed=2000)
    def chat_generate(self, client, system_prompt:str, user_prompt:str, temperature = 0.0):
        request_options = {}
        if self.request_max_tokens is not None:
            request_options['max_tokens'] = self.request_max_tokens
        response = completions_with_backoff(
                client,
                model = self.model_name,
                messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                temperature = temperature,
                top_p = 1,
                stop = self.stop_words,
                **request_options
        )
        generated_text = (response.choices[0].message.content or '').strip()
        finish_reason = response.choices[0].finish_reason
        if self.request_max_tokens is not None:
            if finish_reason == 'length':
                raise ValueError('Model output truncated; increase OPENAI_MAX_TOKENS')
            if not generated_text:
                raise ValueError('Model returned empty answer content; reasoning alone is not an answer')
            reasoning = getattr(response.choices[0].message, 'reasoning_content', None) or ''
            print(f'[llm] finish_reason={finish_reason}, content_chars={len(generated_text)}, '
                  f'reasoning_chars={len(reasoning)}', flush=True)
        return generated_text, finish_reason

    def generate(self, client, input_string, temperature = 0.0):
        return self.chat_generate(client, input_string, temperature)
    
    def batch_chat_generate(self, client, messages_list, temperature = 0.0):
        open_ai_messages_list = []
        system_prompt = "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."
        for message in messages_list:
            open_ai_messages_list.append(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": message}]
            )
        predictions = asyncio.run(
            dispatch_openai_chat_requests(
                    client, open_ai_messages_list, self.model_name, temperature, self.max_new_tokens, 1.0, self.stop_words
            )
        )
        finish_reason = [x.choices[0].finish_reason.strip() for x in predictions]
        return [x.choices[0].message.content.strip() for x in predictions]

    def batch_generate(self, client, messages_list, temperature = 0.0):
        return self.batch_chat_generate(client, messages_list, temperature)
