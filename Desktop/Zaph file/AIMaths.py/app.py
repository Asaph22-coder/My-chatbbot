from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from openai import OpenAI
import json
import os
import logging
from functools import wraps
from dotenv import load_dotenv
from pathlib import Path

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
load_dotenv(Path(__file__).parent / '.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=os.getenv('ALLOWED_ORIGINS', '*').split(','))

# ──────────────────────────────────────────────
# Constantes
# ──────────────────────────────────────────────
MAX_HISTORY = 10
MAX_MESSAGE_LENGTH = 2000
MODEL = 'google/gemini-2.0-flash-001'

LANG_INSTRUCTIONS = {
    'en': 'Step explanations must be in English, clear and educational.',
    'fr': 'Les explications des étapes doivent être en français, claires et pédagogiques.',
    'hi': 'चरण स्पष्टीकरण हिंदी में होने चाहिए, स्पष्ट और शैक्षणिक।',
    'te': 'దశల వివరణలు తెలుగులో ఉండాలి, స్పష్టంగా మరియు విద్యాపరంగా.',
}

# ──────────────────────────────────────────────
# Initialisation OpenRouter
# ──────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY manquante dans le fichier .env")

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def get_system_prompt(language: str = 'en') -> str:
    lang_instr = LANG_INSTRUCTIONS.get(language, LANG_INSTRUCTIONS['en'])
    return f"""You are MathMind, an expert and friendly math teacher.
You adapt your response type based on what the user is asking:

1. If the user asks to SOLVE a math problem, respond with:
{{
  "type": "solution",
  "title": "Solution for [problem]",
  "steps": [
    {{ "num": 1, "explanation": "short explanation", "formula": "formula or calculation" }},
    ...
  ],
  "final_answer": "final answer",
  "verification": "optional verification"
}}

2. If the user asks to EXPLAIN, UNDERSTAND, or LEARN a concept, theorem, or topic (e.g. "explain derivatives", "I don't understand X", "what is X"), respond like a patient professor:
{{
  "type": "explanation",
  "title": "title of the concept",
  "introduction": "simple and friendly introduction",
  "sections": [
    {{ "heading": "Definition", "content": "clear definition" }},
    {{ "heading": "Intuition", "content": "simple real-world analogy or intuition" }},
    {{ "heading": "Formula", "content": "the key formula if applicable" }},
    {{ "heading": "Example", "content": "a concrete worked example" }}
  ],
  "summary": "short summary to remember"
}}

3. For any other general question or conversation, respond with:
{{
  "type": "text",
  "content": "your answer here"
}}

Respond ONLY in valid JSON, no backticks or markdown. Be encouraging and pedagogical. {lang_instr}"""


def validate_history(history: list) -> list:
    if not isinstance(history, list):
        return []

    cleaned = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get('role')
        content = item.get('content')
        if role not in ('user', 'assistant') or not isinstance(content, str):
            continue
        cleaned.append({'role': role, 'content': content[:MAX_MESSAGE_LENGTH]})

    return cleaned[-MAX_HISTORY:]


def require_json(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 415
        return f(*args, **kwargs)
    return decorated


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('.', 'app.html')


@app.route('/index.html')
def chat():
    return send_from_directory('.', 'index.html')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/solve', methods=['POST'])
@require_json
def solve():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Corps JSON invalide'}), 400

    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'Le champ "message" est requis'}), 400
    if len(user_message) > MAX_MESSAGE_LENGTH:
        return jsonify({'error': f'Message trop long (max {MAX_MESSAGE_LENGTH} caractères)'}), 400

    language = data.get('language', 'en')
    if language not in LANG_INSTRUCTIONS:
        language = 'en'

    raw_history = data.get('history', [])
    history = validate_history(raw_history)

    messages = [
        {'role': 'system', 'content': get_system_prompt(language)},
        *history,
        {'role': 'user', 'content': user_message},
    ]

    try:
        response = openrouter_client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )
        raw_text = response.choices[0].message.content

    except Exception as e:
        err = str(e)
        if '401' in err or 'invalid_api_key' in err.lower():
            logger.error("Clé API invalide : %s", e)
            return jsonify({'error': 'Clé API invalide'}), 500
        if '429' in err or 'rate_limit' in err.lower():
            logger.warning("Quota OpenRouter dépassé")
            return jsonify({'error': 'Quota API dépassé, réessayez plus tard'}), 429
        logger.exception("Erreur inattendue lors de l'appel OpenRouter")
        return jsonify({'error': 'Erreur serveur interne'}), 500

    try:
        clean = raw_text.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(clean)
        if 'type' not in parsed:
            raise ValueError("Champ 'type' manquant")
        return jsonify(parsed)

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Réponse non-JSON : %s", e)
        return jsonify({'type': 'text', 'content': raw_text})


# ──────────────────────────────────────────────
# Gestionnaires d'erreurs globaux
# ──────────────────────────────────────────────
@app.errorhandler(404)
def not_found(_):
    return jsonify({'error': 'Route introuvable'}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({'error': 'Méthode HTTP non autorisée'}), 405


@app.errorhandler(500)
def internal_error(_):
    logger.exception("Erreur 500 non gérée")
    return jsonify({'error': 'Erreur serveur interne'}), 500


# ──────────────────────────────────────────────
# Lancement
# ──────────────────────────────────────────────
if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    app.run(debug=debug_mode, port=port)
