"""
Google Gemini Flash API client with rate limiting and error handling.
"""

import re
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

    def _repair_json(self, text: str) -> str:
        """
        Attempt to repair common JSON issues from LLM responses.
        
        Common issues:
        - Unterminated strings (truncated response)
        - Missing closing brackets
        - Trailing commas
        """
        text = text.strip()
        
        # Remove trailing commas before ] or }
        text = re.sub(r',\s*]', ']', text)
        text = re.sub(r',\s*}', '}', text)
        
        # Try to fix unterminated strings by finding last complete JSON object
        # Look for the last complete object pattern: {...}
        # Count brackets to find where array should end
        bracket_count = 0
        brace_count = 0
        in_string = False
        escape_next = False
        last_complete_idx = -1
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
                
            if char == '[':
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if bracket_count == 0:
                    last_complete_idx = i + 1
            elif char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and bracket_count == 1:
                    # Found complete object inside array
                    last_complete_idx = i + 1
        
        # If we have unclosed brackets/braces, try to close them
        if bracket_count > 0 or brace_count > 0:
            # Find last complete object and truncate there
            if last_complete_idx > 0:
                text = text[:last_complete_idx]
                # Add missing closing brackets
                if not text.rstrip().endswith(']'):
                    text = text.rstrip().rstrip(',') + ']'
        
        return text
    
    def _extract_partial_qa_pairs(self, text: str) -> List[Dict]:
        """
        Extract valid Q&A pairs even from partially malformed JSON.
        Uses regex to find complete JSON objects.
        """
        qa_pairs = []
        
        # Pattern to match individual Q&A JSON objects
        # Matches: {"question": "...", "answer": "...", ...}
        pattern = r'\{\s*"question"\s*:\s*"[^"]*"\s*,\s*"answer"\s*:\s*"[^"]*"[^}]*\}'
        
        matches = re.findall(pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                obj = json.loads(match)
                if 'question' in obj and 'answer' in obj:
                    qa_pairs.append(obj)
            except json.JSONDecodeError:
                continue
        
        return qa_pairs

    def generate_qa_batch(self, prompt: str) -> List[Dict]:
        """
        Generate Q&A batch and parse JSON response with robust error recovery.

        Returns:
            List of Q&A pairs or empty list if failed
        """
        response_text = ""
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

            # Try direct parse first
            try:
                qa_pairs = json.loads(response_text)
                if isinstance(qa_pairs, list):
                    self.logger.info(f"Successfully parsed {len(qa_pairs)} Q&A pairs")
                    return qa_pairs
            except json.JSONDecodeError:
                pass  # Will try repair methods
            
            # Try JSON repair
            self.logger.warning("Direct JSON parse failed, attempting repair...")
            repaired_text = self._repair_json(response_text)
            try:
                qa_pairs = json.loads(repaired_text)
                if isinstance(qa_pairs, list):
                    self.logger.info(f"Repaired JSON: parsed {len(qa_pairs)} Q&A pairs")
                    return qa_pairs
            except json.JSONDecodeError:
                pass  # Will try extraction method
            
            # Last resort: extract individual Q&A pairs using regex
            self.logger.warning("JSON repair failed, extracting individual pairs...")
            qa_pairs = self._extract_partial_qa_pairs(response_text)
            if qa_pairs:
                self.logger.info(f"Extracted {len(qa_pairs)} Q&A pairs from partial response")
                return qa_pairs
            
            self.logger.error("All JSON parsing methods failed")
            self.logger.debug(f"Response text (first 500 chars): {response_text[:500]}")
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
