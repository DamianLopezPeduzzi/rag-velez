"""Módulo 5: set de evaluación del RAG.

Sin medición no sabés si tus cambios mejoran o empeoran. Este script
corre 10 preguntas con respuesta conocida y usa a Claude como juez
para verificar si la respuesta del RAG contiene el dato correcto.

Para cada fallo, diagnosticá:
  - ¿Trajo chunks equivocados?  -> problema de RETRIEVAL (tocá chunking/k)
  - ¿Contexto bueno, mala resp? -> problema de GENERACIÓN (tocá el prompt)
"""

import sys
from dotenv import load_dotenv
from anthropic import Anthropic
from rag import preguntar_rag, recuperar_contexto

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

client = Anthropic()

# (pregunta, dato que la respuesta correcta debe contener)
SET_EVALUACION = [
    ("¿En qué año se fundó Vélez Sarsfield?", "1910"),
    ("¿En qué barrio de Buenos Aires juega Vélez?", "Liniers"),
    ("¿Quién fue José Amalfitani?", "presidente del club / dirigente"),
    ("¿Contra quién ganó Vélez la Copa Intercontinental 1994?", "Milan"),
    ("¿Quién era el técnico de Vélez cuando ganó la Libertadores 1994?", "Carlos Bianchi"),
    ("¿Cómo se llama el estadio de Vélez?", "José Amalfitani"),
    ("¿A quién homenajea el nombre del club?", "Dalmacio Vélez Sársfield, autor del Código Civil"),
    ("¿Qué copa internacional ganó Vélez en 1994 además de la Intercontinental?", "Copa Libertadores"),
    ("¿Qué torneo local ganó Vélez en el Clausura 1993?", "Primera División / Clausura 1993"),
    ("¿De qué color es la camiseta titular de Vélez?", "blanca con una V azul"),
]

PROMPT_JUEZ = """Sos un evaluador. Te doy una pregunta, la respuesta correcta esperada, y la respuesta de un sistema.
Respondé SOLO con "CORRECTA" si la respuesta del sistema contiene el dato esperado, o "INCORRECTA" si no.

Pregunta: {pregunta}
Dato esperado: {esperado}
Respuesta del sistema: {respuesta}"""


def juzgar(pregunta, esperado, respuesta):
    """Usa a Claude como juez: ¿la respuesta contiene el dato esperado?"""
    r = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": PROMPT_JUEZ.format(
                pregunta=pregunta, esperado=esperado, respuesta=respuesta
            ),
        }],
    )
    return "CORRECTA" in r.content[0].text.upper()


def main():
    aciertos = 0
    fallos = []

    for i, (pregunta, esperado) in enumerate(SET_EVALUACION, 1):
        respuesta = preguntar_rag(pregunta, mostrar_fuentes=False)
        correcta = juzgar(pregunta, esperado, respuesta)

        estado = "OK " if correcta else "FALLO"
        print(f"[{i:2d}/{len(SET_EVALUACION)}] {estado} - {pregunta}")
        print(f"          R: {respuesta[:120].strip()}...")

        if correcta:
            aciertos += 1
        else:
            fallos.append((pregunta, esperado, respuesta))

    print("\n" + "=" * 60)
    print(f"RESULTADO: {aciertos}/{len(SET_EVALUACION)} correctas")

    # Diagnóstico de fallos: mostramos qué chunks trajo el retrieval
    for pregunta, esperado, respuesta in fallos:
        print("\n" + "-" * 60)
        print(f"FALLO: {pregunta}")
        print(f"Esperado: {esperado}")
        print(f"Respuesta: {respuesta[:300]}")
        print("\nChunks recuperados (¿el dato estaba en el contexto?):")
        chunks, fuentes = recuperar_contexto(pregunta)
        for chunk, fuente in zip(chunks, fuentes):
            print(f"  [{fuente}] {chunk[:150]}...")
        print("\n>> Si el dato NO está en los chunks: problema de RETRIEVAL.")
        print(">> Si el dato SÍ está pero respondió mal: problema de GENERACIÓN.")


if __name__ == "__main__":
    main()
