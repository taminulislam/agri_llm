"""
Google Gemini Flash API client with rate limiting and error handling.
"""

import time
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Please run: pip install google-generativeai")

from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class APIUsageStats:
    """Track API usage and costs"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class GeminiClient:
    """
    Google Gemini Flash API client with built-in rate limiting.

    Rate Limits (Gemini Flash free tier):
    - 15 requests per minute (RPM)
    - 1 million tokens per minute (TPM)
    - 1,500 requests per day (RPD)

    After free tier:
    - Input: $0.075 per 1M tokens
    - Output: $0.30 per 1M tokens
    """

    # Pricing (per 1M tokens)
    PRICE_INPUT_PER_1M = 0.075  # $0.075
    PRICE_OUTPUT_PER_1M = 0.30  # $0.30

    # Rate limits (free tier)
    FREE_RPM = 15
    FREE_TPM = 1_000_000
    FREE_RPD = 1_500

    def __init__(self,
                 api_key: str,
                 model_name: str = "gemini-2.5-flash",
                 rpm_limit: int = 15,
                 tpm_limit: int = 1_000_000,
                 temperature: float = 0.7):

        if not GENAI_AVAILABLE:
            raise ImportError("google-generativeai not installed")

        genai.configure(api_key=api_key)

        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

        # Rate limiting
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit

        # Generation config
        self.generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }

        # Usage tracking
        self.stats = APIUsageStats()

        # Rate limiting state
        self.request_timestamps = []
        self.tokens_used_this_minute = 0
        self.last_token_reset = time.time()

        # Logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def _check_rate_limits(self, estimated_tokens: int = 2000):
        """
        Check and enforce rate limits before making request.
        Sleeps if necessary to stay within limits.
        """
        current_time = time.time()

        # Clean up old request timestamps (older than 60 seconds)
        self.request_timestamps = [
            ts for ts in self.request_timestamps
            if current_time - ts < 60
        ]

        # Check RPM limit
        if len(self.request_timestamps) >= self.rpm_limit:
            # Calculate sleep time
            oldest_request = min(self.request_timestamps)
            sleep_time = 60 - (current_time - oldest_request)
            if sleep_time > 0:
                self.logger.info(f"Rate limit reached. Sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                # Clear timestamps after sleep
                self.request_timestamps = []

        # Check TPM limit
        if current_time - self.last_token_reset >= 60:
            # Reset token counter every minute
            self.tokens_used_this_minute = 0
            self.last_token_reset = current_time

        if self.tokens_used_this_minute + estimated_tokens > self.tpm_limit:
            # Wait until next minute
            sleep_time = 60 - (current_time - self.last_token_reset)
            if sleep_time > 0:
                self.logger.info(f"Token limit reached. Sleeping for {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self.tokens_used_this_minute = 0
                self.last_token_reset = time.time()

        # Record this request
        self.request_timestamps.append(time.time())

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def generate(self, prompt: str, estimated_tokens: int = 2000) -> Optional[str]:
        """
        Generate text using Gemini Flash with automatic retries.

        Args:
            prompt: Input prompt
            estimated_tokens: Estimated tokens for rate limiting

        Returns:
            Generated text or None if failed
        """
        # Check rate limits
        self._check_rate_limits(estimated_tokens)

        try:
            # Make API call
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )

            # Update stats
            self.stats.total_requests += 1
            self.stats.successful_requests += 1

            # Estimate token usage (Gemini doesn't always provide exact counts)
            input_tokens = len(prompt.split()) * 1.3  # Rough estimate
            output_tokens = len(response.text.split()) * 1.3

            self.stats.total_input_tokens += int(input_tokens)
            self.stats.total_output_tokens += int(output_tokens)
            self.tokens_used_this_minute += int(input_tokens + output_tokens)

            # Update cost estimate
            input_cost = (self.stats.total_input_tokens / 1_000_000) * self.PRICE_INPUT_PER_1M
            output_cost = (self.stats.total_output_tokens / 1_000_000) * self.PRICE_OUTPUT_PER_1M
            self.stats.estimated_cost_usd = input_cost + output_cost

            return response.text

        except Exception as e:
            self.stats.total_requests += 1
            self.stats.failed_requests += 1
            self.logger.error(f"API call failed: {e}")
            raise

    def generate_qa_batch(self, prompt: str) -> List[Dict]:
        """
        Generate Q&A batch and parse JSON response.

        Returns:
            List of Q&A pairs or empty list if failed
        """
        try:
            response_text = self.generate(prompt)

            if not response_text:
                return []

            # Try to extract JSON from response
            # Sometimes the model adds markdown code blocks
            response_text = response_text.strip()

            # Remove markdown code blocks if present
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]

            response_text = response_text.strip()

            # Parse JSON
            qa_pairs = json.loads(response_text)

            if not isinstance(qa_pairs, list):
                self.logger.error("Response is not a JSON array")
                return []

            return qa_pairs

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            self.logger.debug(f"Response text: {response_text[:500]}")
            return []
        except Exception as e:
            self.logger.error(f"Failed to generate Q&A batch: {e}")
            return []

    def get_usage_stats(self) -> Dict:
        """Get current usage statistics"""
        return {
            'total_requests': self.stats.total_requests,
            'successful_requests': self.stats.successful_requests,
            'failed_requests': self.stats.failed_requests,
            'total_input_tokens': self.stats.total_input_tokens,
            'total_output_tokens': self.stats.total_output_tokens,
            'estimated_cost_usd': round(self.stats.estimated_cost_usd, 4),
            'success_rate': round(
                self.stats.successful_requests / self.stats.total_requests * 100, 1
            ) if self.stats.total_requests > 0 else 0.0
        }
