"""
modules/response.py — Response generation using the local LLM.

Responsibilities:
  - Enforce access control (anonymous users cannot receive account info)
  - Build the full prompt: system instructions + customer context + product docs + history
  - Call the LLM and return the assistant's reply

Access control is enforced HERE, not in the intent classifier.
This keeps each module single-purpose and independently testable.
"""

from modules.llm import generate
from modules.auth import format_customer_context
from config import LLM_MAX_TOKENS, LLM_TEMPERATURE, ANONYMOUS_RESTRICTION_MSG

# ── Prompt building blocks ────────────────────────────────────────────────────

_BASE_SYSTEM = """You are a helpful, professional customer support assistant for Globus Bank Nigeria.
You assist customers with enquiries about their accounts, transactions, cards, loan products, \
savings accounts, and investment products.
Be warm, accurate, and concise. Use Nigerian Naira (NGN) unless the account is a domiciliary account.
Never invent interest rates, fees, or product terms not provided to you. \
If you are unsure, advise the customer to visit a branch or call Globus Bank customer service. \
Always prioritise customer security and privacy. Do not ask for or reveal sensitive information unless necessary. \
Keep it all about Globus Bank — do not discuss other banks or financial institutions and their products."""

_CUSTOMER_BLOCK = """
The customer is authenticated. Use their full profile below to give accurate, personalised responses.
Do not reveal email addresses unless directly asked.

CUSTOMER PROFILE:
{context}
"""

_ANONYMOUS_BLOCK = f"""
This user is NOT authenticated — no valid account number was provided.
You may ONLY discuss general Globus Bank product information.
If they ask about account details, balances, transactions, or cards, respond with:
"{ANONYMOUS_RESTRICTION_MSG}"
"""

_DOCS_BLOCK = """
Use the following Globus Bank product documentation to answer accurately.
Do not invent product features, rates, or terms beyond what is documented here.

PRODUCT DOCUMENTATION:
{docs}
"""


def _build_prompt(
    message: str,
    customer: dict | None,
    queried_account: str | None,
    product_docs: str | None,
    history: list[dict],
) -> str:
    """
    Assemble the full prompt in llama.cpp chat format.
    Compatible with Phi-3, Llama-3.2 chat templates.
    """
    system = _BASE_SYSTEM

    if customer:
        system += _CUSTOMER_BLOCK.format(
            context=format_customer_context(customer, queried_account or "")
        )
    else:
        system += _ANONYMOUS_BLOCK

    if product_docs:
        system += _DOCS_BLOCK.format(docs=product_docs)

    prompt = f"<|system|>\n{system.strip()}\n<|end|>\n"

    for turn in history:
        tag = "<|user|>" if turn["role"] == "user" else "<|assistant|>"
        prompt += f"{tag}\n{turn['content']}\n<|end|>\n"

    prompt += f"<|user|>\n{message}\n<|end|>\n<|assistant|>\n"
    return prompt


# ── Public interface ──────────────────────────────────────────────────────────

def generate_response(
    message: str,
    intent: str,
    customer: dict | None,
    queried_account: str | None,
    product_docs: str | None,
    history: list[dict],
) -> str:
    """
    Generate a support response for the current conversation turn.

    Args:
        message:         Current user message.
        intent:          Classified intent string.
        customer:        Authenticated customer dict, or None if anonymous.
        queried_account: The account number the user entered (for context formatting).
        product_docs:    Retrieved product documentation, or None.
        history:         Prior conversation turns [{role, content}].

    Returns:
        Assistant reply as a plain string.
    """
    # Hard access control: block account queries for unauthenticated sessions
    if intent == "account_information" and customer is None:
        return ANONYMOUS_RESTRICTION_MSG

    prompt = _build_prompt(
        message=message,
        customer=customer,
        queried_account=queried_account,
        product_docs=product_docs,
        history=history,
    )

    try:
        return generate(
            prompt=prompt,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
        )
    except Exception as e:
        print(f"[response] Generation error: {e}")
        return "I'm having trouble responding right now. Please try again in a moment."
