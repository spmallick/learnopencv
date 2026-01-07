import requests
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

def probe_vision_capability(base_url, model):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "about:blank"},
                    {"type": "text", "text": "ping"},
                ],
            }
        ],
        "max_tokens": 1,
    }

    try:
        r = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=10,
        )

        if r.status_code == 200:
            return True

        error_msg = r.text.lower()

        if any(k in error_msg for k in [
            "image",
            "vision",
            "multimodal",
            "image_url",
            "image token",
        ]):
            return True

        return False

    except requests.RequestException:
        return False

def get_models(base_url):
    """Fetch available models from the /v1/models endpoint."""
    try:
        # Normalize base URL
        normalized_base = base_url.rstrip("/")
        if normalized_base.endswith("/v1"):
            normalized_base = normalized_base[:-3]
        
        r = requests.get(f"{normalized_base}/v1/models", headers={"ngrok-skip-browser-warning": "true"}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            models = data.get("data", [])
            return [m.get("id") or m.get("model") for m in models if m.get("id") or m.get("model")]
        return []
    except requests.RequestException:
        return []

if __name__ == "__main__":
    base_url = os.getenv("OPENAI_API_BASE", "")
    models = get_models(base_url)
    
    if not models:
        print("No models found at the API endpoint.")
    else:
        print(f"Found {len(models)} model(s):")
        for model in models:
            vision = probe_vision_capability(base_url, model)
            print(f"  - {model}: {'Vision-capable' if vision else 'Text-only'}")