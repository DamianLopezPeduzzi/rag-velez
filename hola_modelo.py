import sys
import os
from dotenv import load_dotenv
from anthropic import Anthropic

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

client = Anthropic()

mensaje = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[
        {"role": "user", "content": "Resumime en una frase qué es Vélez Sársfield."}
    ],
)

print(mensaje.content[0].text)
