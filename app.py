"""
app.py — Globus Bank Support Chatbot (Flask entry point).

Routes:
  GET  /                   → Chat UI
  POST /api/authenticate   → Validate account number, initialise session
  POST /api/chat           → Send a message, receive a response
  POST /api/reset          → Clear session (start over / sign out)

Server-side session state (stored in Flask session):
  customer         → full customer profile dict | None
  queried_account  → account number string entered by the user
  authenticated    → bool
  history          → list of {role, content} dicts
"""

from pathlib import Path
from flask import Flask, request, jsonify, session, render_template

from modules.auth import get_customer
from modules.intent import classify_intent
from modules.knowledge_base import get_relevant_info
from modules.response import generate_response
from config import (
    FLASK_HOST, FLASK_PORT, FLASK_DEBUG,
    SECRET_KEY, MAX_HISTORY_TURNS,
    ACCOUNT_NOT_FOUND_MSG,
)

_HERE = Path(__file__).parent
app = Flask(__name__, template_folder=str(_HERE / "templates"), static_folder=str(_HERE / "static"))
app.secret_key = SECRET_KEY

# Intents that trigger a knowledge base lookup
_KB_INTENTS = {"product_inquiry", "general_inquiry"}


def _trim_history(history: list[dict]) -> list[dict]:
    """Keep only the last MAX_HISTORY_TURNS turns in the session."""
    cap = MAX_HISTORY_TURNS * 2   # each turn = user msg + assistant msg
    return history[-cap:] if len(history) > cap else history


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/authenticate", methods=["POST"])
def authenticate():
    """
    Validate the supplied account number and initialise a session.

    Request body:  { "account_number": "100023489" }
    Response:      { "success": bool, "name": str|null, "message": str }
    """
    data           = request.get_json(silent=True) or {}
    account_number = data.get("account_number", "").strip()

    if not account_number:
        return jsonify({"success": False, "message": "Please enter an account number."}), 400

    customer = get_customer(account_number)

    # Initialise a fresh session regardless of outcome
    session["history"]          = []
    session["authenticated"]    = customer is not None
    session["customer"]         = customer
    session["queried_account"]  = account_number

    if customer:
        return jsonify({
            "success": True,
            "name":    customer["name"],
            "message": f"Welcome back, {customer['name']}! How can I help you today?",
        })
    else:
        return jsonify({
            "success": False,
            "name":    None,
            "message": ACCOUNT_NOT_FOUND_MSG,
        })


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Process a user message and return the assistant's response.

    Request body:  { "message": "What is my account balance?" }
    Response:      { "reply": str, "intent": str }
    """
    if "authenticated" not in session:
        return jsonify({"error": "Please enter your account number first."}), 403

    data    = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()

    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    customer        = session.get("customer")
    queried_account = session.get("queried_account")
    history         = session.get("history", [])

    # Step 1 — Classify intent
    intent = classify_intent(message)

    # Step 2 — Retrieve product docs if relevant
    product_docs = get_relevant_info(message) if intent in _KB_INTENTS else None

    # Step 3 — Generate response
    reply = generate_response(
        message=message,
        intent=intent,
        customer=customer,
        queried_account=queried_account,
        product_docs=product_docs,
        history=history,
    )

    # Step 4 — Persist updated history
    history.append({"role": "user",      "content": message})
    history.append({"role": "assistant", "content": reply})
    session["history"] = _trim_history(history)

    return jsonify({"reply": reply, "intent": intent})


@app.route("/api/reset", methods=["POST"])
def reset():
    """Clear the current session so the user can re-authenticate."""
    session.clear()
    return jsonify({"message": "Session cleared."})


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # LLM and knowledge base are loaded at import time (modules/llm.py,
    # modules/knowledge_base.py) so everything is warm before Flask starts.
    print(f"\n[app] Globus Bank Support Chatbot → http://{FLASK_HOST}:{FLASK_PORT}\n")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)