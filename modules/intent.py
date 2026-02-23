"""
modules/intent.py — Intent classification using the local LLM.

Classifies each user message into exactly one of four intents:
  - greeting
  - general_inquiry
  - account_information
  - product_inquiry

Uses temperature=0 and a tight stop sequence for deterministic output.
Falls back to 'general_inquiry' on any parse failure.
"""

import re
from modules.llm import generate
from config import INTENTS, INTENT_MAX_TOKENS, INTENT_TEMPERATURE


def _build_prompt(message: str) -> str:
    return f"""<|system|>
You are an intent classifier for a Globus Bank customer support chatbot.

Classify the user message into exactly one intent from this list:
- greeting: user is saying hello or starting a conversation
- general_inquiry: broad or unclear question not specifically about their account or a product
- account_information: question about their account balance, transactions, cards, account status, or account details
- product_inquiry: question about bank products (loans, savings, investments, debit cards, features, eligibility)

Reply with ONLY the intent name. No explanation. No punctuation. Just the single intent word or phrase.
<|end|>
<|user|>
Message: {message}
<|assistant|>
Intent:"""


def classify_intent(message: str) -> str:
    """
    Classify the intent of a user message.

    Returns one of: 'greeting', 'general_inquiry', 'account_information', 'product_inquiry'.
    Defaults to 'general_inquiry' on any failure.
    """
    try:
        raw = generate(
            prompt=_build_prompt(message),
            max_tokens=INTENT_MAX_TOKENS,
            temperature=INTENT_TEMPERATURE,
            stop=["\n", "<|end|>", "<|user|>"],
        )

        # Normalise: take first token, strip punctuation
        first_word = raw.strip().split()[0].lower().strip(".,!?\"'") if raw.strip() else ""

        if first_word in INTENTS:
            return first_word

        # Fuzzy fallback — handle outputs like "account_information." or partial matches
        for intent in INTENTS:
            if intent in raw.lower():
                return intent

        return "general_inquiry"

    except Exception as e:
        print(f"[intent] Classification error: {e}")
        return "general_inquiry"
