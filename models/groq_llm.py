import os
import time
import asyncio
from typing import Optional, Union
import json

from deepeval.models import DeepEvalBaseLLM
from groq import Groq, AsyncGroq, BadRequestError, RateLimitError
from pydantic import BaseModel
from config import EVAL_LLM_MODEL_NAME, GROQ_API_KEY, EVAL_MODELS


class GroqLlama3(DeepEvalBaseLLM):
    """
    DeepEval LLM wrapper using Groq's chat completions API.
    Uses AsyncGroq for a_generate to prevent blocking the event loop.

    Concurrency / throttling is handled externally:
      - In evaluate_rag.py via AsyncConfig(max_concurrent=..., throttle_value=...)
      - Via DEEPEVAL_DISABLE_TIMEOUTS=1 so rate-limit retries are never cancelled
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        retry_count: int = 5
    ):
        if model_name is None:
            model_name = EVAL_LLM_MODEL_NAME

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_count = retry_count

        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY environment variable is not set.")

        self.client = Groq(api_key=GROQ_API_KEY)
        self.async_client = AsyncGroq(api_key=GROQ_API_KEY)

    def get_model_name(self) -> str:
        return self.model_name

    def load_model(self):
        return self.client

    def _build_api_args(self, prompt: str, schema: Optional[BaseModel], max_tokens: int) -> dict:
        api_args = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
        }
        if schema is not None:
            api_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": schema.model_json_schema()
                }
            }
            api_args["messages"][0]["content"] += (
                "\n\nIMPORTANT: You must respond with a valid JSON object matching the schema."
            )
        return api_args

    def _parse_response(self, content: str, schema: Optional[BaseModel]):
        if schema is None:
            return content
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            return schema.model_validate(json.loads(content))
        return schema.model_validate(json.loads(content))

    # ------------------------------------------------------------------ #
    # Sync path                                                            #
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, schema: Optional[BaseModel] = None, **kwargs) -> Union[str, BaseModel]:
        """Synchronous generate with model fallback."""
        client = self.load_model()
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        api_args = self._build_api_args(prompt, schema, max_tokens)

        models_to_try = [self.model_name] + [m for m in EVAL_MODELS if m != self.model_name]

        for model in models_to_try:
            api_args["model"] = model
            print(f"🔄 (Sync) Generating with model: {model}")

            for attempt in range(self.retry_count):
                try:
                    resp = client.chat.completions.create(**api_args)
                    return self._parse_response(resp.choices[0].message.content, schema)
                except BadRequestError as e:
                    if "json_validate_failed" in str(e) or "json_object" in str(e):
                        print(f"⚠️ JSON validation error on {model} attempt {attempt+1}.")
                    else:
                        print(f"❌ Bad Request on {model}: {e}")
                except RateLimitError:
                    wait = 15 * (attempt + 1)
                    print(f"⏳ Rate limit on {model} attempt {attempt+1}. Sleeping {wait}s...")
                    time.sleep(wait)
                except Exception as e:
                    wait = min(2 ** attempt, 30)
                    print(f"❌ Error on {model} attempt {attempt+1}: {e}. Sleeping {wait}s...")
                    time.sleep(wait)

            print(f"⚠️ Exhausted retries for {model}. Switching to next model...")

        raise RuntimeError("All models exhausted. Failed to generate response.")

    # ------------------------------------------------------------------ #
    # Async path                                                           #
    # ------------------------------------------------------------------ #
    async def a_generate(self, prompt: str, schema: Optional[BaseModel] = None, **kwargs) -> Union[str, BaseModel]:
        """
        Async generate with model fallback.

        NOTE: No semaphore here — any awaitable lock inside DeepEval's
        asyncio.wait_for deadline will be cancelled, turning into a
        CancelledError → TimeoutError. Concurrency is controlled at the
        DeepEval level via AsyncConfig in evaluate_rag.py.
        """
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        api_args = self._build_api_args(prompt, schema, max_tokens)

        models_to_try = [self.model_name] + [m for m in EVAL_MODELS if m != self.model_name]

        for model in models_to_try:
            api_args["model"] = model

            for attempt in range(self.retry_count):
                try:
                    print(f"🔄 (Async) Generating with model: {model} (attempt {attempt+1})")
                    resp = await self.async_client.chat.completions.create(**api_args)
                    return self._parse_response(resp.choices[0].message.content, schema)

                except BadRequestError as e:
                    if "json_validate_failed" in str(e) or "json_object" in str(e):
                        print(f"⚠️ JSON validation error on {model} attempt {attempt+1}.")
                    else:
                        print(f"❌ Bad Request on {model}: {e}")
                except RateLimitError:
                    wait = 15 * (attempt + 1)
                    print(f"⏳ Rate limit on {model} attempt {attempt+1}. Sleeping {wait}s...")
                    await asyncio.sleep(wait)
                except Exception as e:
                    wait = min(2 ** attempt, 30)
                    print(f"❌ Error on {model} attempt {attempt+1}: {e}. Sleeping {wait}s...")
                    await asyncio.sleep(wait)

            print(f"⚠️ Exhausted retries for {model}. Switching to next model...")

        raise RuntimeError("All models exhausted. Failed to generate response.")