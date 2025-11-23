#!/usr/bin/env python3
"""
Minimal test script for Cursor OpenRouter Proxy
"""

import requests
import json

import os
from dotenv import load_dotenv
load_dotenv()

# Proxy configuration
PROXY_URL = "http://localhost:9000"
API_KEY = os.getenv("OPENROUTER_API_KEY")  # The proxy will use its own OPENROUTER_API_KEY


def test_chat_completion():
    """Test a simple chat completion request"""

    url = f"{PROXY_URL}/v1/chat/completions"

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    payload = {
        "model": "gpt-4",  # The proxy will override this with OPENROUTER_MODEL
        "messages": [{"role": "user", "content": "Say hello!"}],
        "stream": False,
    }

    print("Sending request to proxy...")
    print(f"URL: {url}")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print("\nResponse:")
            print(json.dumps(result, indent=2))

            # Extract and print the actual message
            if "choices" in result and len(result["choices"]) > 0:
                message = result["choices"][0]["message"]["content"]
                print(f"\n✓ AI Response: {message}")
        else:
            print(f"\n✗ Error: {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"\n✗ Request failed: {e}")


if __name__ == "__main__":
    test_chat_completion()
