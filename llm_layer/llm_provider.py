"""
LLM provider abstraction layer.

Wraps OpenAI API with retry logic, timeout handling,
and structured output parsing.
"""

import json
import time
from typing import Any, Dict, List, Optional

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
config = Config()


class LLMProviderError(Exception):
    """Raised when the LLM provider returns an error."""
    pass


class LLMProvider:
    """
    Abstraction over LLM APIs (OpenAI, etc.).

    Design decisions:
    - Single provider abstraction: Easy to swap OpenAI → Anthropic → local.
    - Retry with exponential backoff: LLM APIs are rate-limited. 3 retries
      with 2/4/8s delays handle transient failures without overwhelming.
    - Temperature 0.1: Near-deterministic for financial assessments.
      Risk analysis must be reproducible, not creative.
    - Structured JSON output: Enforced via system prompt + json_object
      response format, with fallback parsing for older models.
    """

    def __init__(self) -> None:
        llm_config = config.get_section("llm_layer")
        self.provider = llm_config.get("provider", "openai")
        self.model = llm_config.get("model", "gpt-4o")
        self.temperature = llm_config.get("temperature", 0.1)
        self.max_tokens = llm_config.get("max_tokens", 1024)
        self.timeout = llm_config.get("timeout", 30)
        self.retry_attempts = llm_config.get("retry_attempts", 3)
        self.retry_delay = llm_config.get("retry_delay", 2)
        self._client = None

    def _get_client(self) -> Any:
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            import os
            try:
                from openai import OpenAI
                api_key = os.environ.get("OPENAI_API_KEY", "")
                if not api_key:
                    raise LLMProviderError(
                        "OPENAI_API_KEY not set in environment variables. "
                        "Copy .env.example to .env and add your key."
                    )
                self._client = OpenAI(
                    api_key=api_key,
                    timeout=self.timeout,
                )
                logger.info(f"OpenAI client initialized with model: {self.model}")
            except ImportError:
                raise LLMProviderError(
                    "openai package not installed. Run: pip install openai"
                )
        return self._client

    def generate(
        self,
        messages: List[Dict[str, str]],
        json_mode: bool = True,
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            json_mode: If True, request JSON-formatted response.

        Returns:
            Generated response string.

        Raises:
            LLMProviderError: If all retries are exhausted.
        """
        client = self._get_client()
        last_error: Optional[Exception] = None

        for attempt in range(1, self.retry_attempts + 1):
            try:
                kwargs: Dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }

                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                response = client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content

                logger.info(
                    f"LLM response generated: {len(content or '')} chars, "
                    f"model={self.model}, attempt={attempt}"
                )
                return content or ""

            except Exception as e:
                last_error = e
                logger.warning(
                    f"LLM call failed (attempt {attempt}/{self.retry_attempts}): {e}"
                )
                if attempt < self.retry_attempts:
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    time.sleep(delay)

        raise LLMProviderError(
            f"All {self.retry_attempts} LLM calls failed. Last error: {last_error}"
        )

    def generate_json(
        self,
        messages: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        """
        Generate a JSON response and parse it.

        Args:
            messages: Chat messages.

        Returns:
            Parsed JSON dictionary.

        Raises:
            LLMProviderError: If response is not valid JSON.
        """
        response = self.generate(messages, json_mode=True)

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            # Attempt to extract JSON from markdown code block
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            raise LLMProviderError(f"Invalid JSON response from LLM: {response[:500]}")

    def health_check(self) -> bool:
        """
        Test connectivity to the LLM provider.

        Returns:
            True if the provider is reachable.
        """
        try:
            response = self.generate(
                messages=[{"role": "user", "content": "Respond with: {\"status\": \"ok\"}"}],
                json_mode=True,
            )
            return "ok" in response.lower()
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False
