from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import google.generativeai as genai
import google.api_core.exceptions as google_exceptions
import json
import os
import logging
from functools import wraps
from dotenv import load_dotenv
from pathlib import Path


load_dotenv(Path(__file__).parent / '.env')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins=os.getenv('ALLOWED_ORIGINS', '*').split(','))

MAX_HISTORY = 10
MAX_MESSAGE_LENGTH = 2000

LANG_INSTRUCTIONS = {
    'en': 'Step explanations must be in English, clear and educational.',
    'fr': 'Les explications des étapes doivent être en français, claires et pédagogiques.',
    'hi': 'चरण स्पष्टीकरण हिंदी में होने चाहिए, स्पष्ट और शैक्षणिक।',
    'te': 'దశల వివరణలు తెలుగులో ఉండాలి, స్పష్టంగా మరియు విద్యాపరంగా.',
}


GOOGLE_API_KEY = os.getenv('GIzaSyCw31kzv-jzKP5nHxOeYfJR2PtuEOPOGE0')
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY manquante dans le fichier .env")

genai.configure(api_key=GOOGLE_API_KEY)


def get_system_prompt(language: str = 'en') -> str:
    lang_instr = LANG_INSTRUCTIONS.get(language, LANG_INSTRUCTIONS['en'])
    return f"""You are MathMind, an expert math assistant.
When the user enters a math problem, you must:
1. ALWAYS structure your response in JSON with this exact format:
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

If the user asks a general question (not a problem to solve), respond normally in JSON:
{{
  "type": "text",
  "content": "your answer here"
}}

Respond ONLY in valid JSON, no backticks or markdown. {lang_instr}"""


def validate_history(history: list) -> list:
    """Valide et assainit l'historique venant du client."""
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
    """Décorateur : vérifie que le Content-Type est application/json."""
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

    # ── Validation du message ──
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'Le champ "message" est requis'}), 400
    if len(user_message) > MAX_MESSAGE_LENGTH:
        return jsonify({'error': f'Message trop long (max {MAX_MESSAGE_LENGTH} caractères)'}), 400

    # ── Validation de la langue ──
    language = data.get('language', 'en')
    if language not in LANG_INSTRUCTIONS:
        language = 'en'

    # ── Validation de l'historique ──
    raw_history = data.get('history', [])
    history = validate_history(raw_history)

    # ── Construction de l'historique Gemini ──
    gemini_history = [
        {
            'role': 'model' if m['role'] == 'assistant' else 'user',
            'parts': [{'text': m['content']}]
        }
        for m in history
    ]

    # ── Appel Gemini avec gestion d'erreurs ──
    try:
        model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            system_instruction=get_system_prompt(language)
        )
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(user_message)
        raw_text = response.text

    except google_exceptions.InvalidArgument as e:
        logger.error("Clé API ou argument invalide : %s", e)
        return jsonify({'error': 'Clé API invalide ou requête mal formée'}), 500

    except google_exceptions.ResourceExhausted:
        logger.warning("Quota Gemini dépassé")
        return jsonify({'error': 'Quota API dépassé, réessayez plus tard'}), 429

    except google_exceptions.GoogleAPIError as e:
        logger.error("Erreur API Google : %s", e)
        return jsonify({'error': 'Service IA temporairement indisponible'}), 503

    except Exception as e:
        logger.exception("Erreur inattendue lors de l'appel Gemini")
        return jsonify({'error': 'Erreur serveur interne'}), 500

    # ── Parsing de la réponse ──
    try:
        clean = raw_text.replace('```json', '').replace('```', '').strip()
        parsed = json.loads(clean)

        # Vérification minimale du format retourné
        if 'type' not in parsed:
            raise ValueError("Champ 'type' manquant dans la réponse")

        return jsonify(parsed)

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Réponse Gemini non-JSON : %s", e)
        return jsonify({'type': 'text', 'content': raw_text})


# ──────────────────────────────────────────────
# Gestionnaires d'erreurs globaux
# ──────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Route introuvable'}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({'error': 'Méthode HTTP non autorisée'}), 405


@app.errorhandler(500)
def internal_error(e):
    logger.exception("Erreur 500 non gérée")
    return jsonify({'error': 'Erreur serveur interne'}), 500


# ──────────────────────────────────────────────
# Lancement
# ──────────────────────────────────────────────
if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    port = int(os.getenv('PORT', 5000))
    app.run(debug=debug_mode, port=port)
