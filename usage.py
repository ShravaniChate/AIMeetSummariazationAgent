import os, requests
from dotenv import load_dotenv
load_dotenv()
r = requests.post(
    "https://api.mistral.ai/v1/chat/completions",
    headers={"Authorization": f"Bearer {os.getenv('MISTRAL_API_KEY')}"},
    json={"model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 5},
)
print(r.status_code)
print(r.headers)
print(r.text)