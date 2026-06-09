import sys
from dotenv import load_dotenv
from anthropic import Anthropic

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

client = Anthropic()

PROMPT_RAG = """Respondé la pregunta del usuario usando SOLO la siguiente información de contexto.
Si la respuesta no está en el contexto, decí "No tengo información suficiente para responder eso."
No inventes datos. Citá el contexto cuando sea posible.

Contexto:
{contexto}

Pregunta: {pregunta}"""

CONTEXTO_EJEMPLO = """
El Club Atlético Vélez Sársfield fue fundado el 1 de enero de 1910 en el barrio
de Liniers, Buenos Aires. En 1994, Vélez ganó la Copa Libertadores y la Copa
Intercontinental, venciendo al AC Milan en Tokio. Carlos Bianchi fue el director
técnico de ese equipo histórico.
"""


def preguntar_rag(pregunta, contexto):
    prompt = PROMPT_RAG.format(contexto=contexto, pregunta=pregunta)

    respuesta = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    return respuesta.content[0].text


# --- Test 1: pregunta que SÍ está en el contexto ---
print("=" * 50)
print("TEST 1: Pregunta con respuesta en el contexto")
print("=" * 50)
r1 = preguntar_rag("¿A quién venció Vélez en la Intercontinental?", CONTEXTO_EJEMPLO)
print(f"R: {r1}\n")

# --- Test 2: pregunta que NO está en el contexto ---
print("=" * 50)
print("TEST 2: Pregunta SIN respuesta en el contexto")
print("=" * 50)
r2 = preguntar_rag("¿Quién es el máximo goleador histórico de Vélez?", CONTEXTO_EJEMPLO)
print(f"R: {r2}\n")

# --- Test 3: pregunta que intenta hacer alucinar ---
print("=" * 50)
print("TEST 3: Pregunta tramposa (dato falso)")
print("=" * 50)
r3 = preguntar_rag("¿Es cierto que Vélez fue fundado en 1920?", CONTEXTO_EJEMPLO)
print(f"R: {r3}\n")
