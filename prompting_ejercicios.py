import sys
from dotenv import load_dotenv
from anthropic import Anthropic

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

client = Anthropic()

PARRAFO_VELEZ = """
El Club Atlético Vélez Sársfield fue fundado el 1 de enero de 1910 en el barrio
de Liniers, Buenos Aires. Su nombre homenajea a Dalmacio Vélez Sársfield, autor
del Código Civil argentino. En 1968, bajo la presidencia de José Amalfitani, el
club inauguró su estadio en Liniers, que hoy lleva el nombre del dirigente.
El momento más glorioso llegó en 1994, cuando Vélez ganó la Copa Libertadores
y la Copa Intercontinental, venciendo al AC Milan en Tokio. Carlos Bianchi fue
el director técnico de ese equipo histórico. En el ámbito local, Vélez acumula
más de 10 títulos de Primera División, incluyendo los Clausura 1993, 1996, 1998
y los Apertura 2005, 2009, 2011 y 2012.
"""


def llamar(prompt, etiqueta):
    print(f"\n{'='*60}")
    print(f">>> {etiqueta}")
    print(f"{'='*60}")
    print(f"Prompt: {prompt[:100]}...")
    print("-" * 40)

    respuesta = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    print(respuesta.content[0].text)


# --- EJERCICIO 1: Prompt vago vs. específico ---

print("\n\n*** EJERCICIO 1: VAGO vs. ESPECÍFICO ***")

llamar(
    "Contame de Vélez",
    "VAGO - sin instrucciones de formato ni foco",
)

llamar(
    f"A partir del siguiente texto, listá los títulos internacionales que ganó "
    f"Vélez Sársfield, indicando año y rival. Respondé en formato lista.\n\n"
    f"Texto: {PARRAFO_VELEZ}",
    "ESPECÍFICO - con contexto, tarea clara y formato pedido",
)


# --- EJERCICIO 2: Zero-shot vs. Few-shot ---

print("\n\n*** EJERCICIO 2: ZERO-SHOT vs. FEW-SHOT ***")

llamar(
    f"Extraé los eventos clave del siguiente texto en formato "
    f"'AÑO - EVENTO'.\n\nTexto: {PARRAFO_VELEZ}",
    "ZERO-SHOT - sin dar ejemplos",
)

llamar(
    f"Extraé los eventos clave del siguiente texto en formato "
    f"'AÑO - EVENTO'.\n\n"
    f"Ejemplo de formato esperado:\n"
    f"1905 - Fundación del Club Atlético Boca Juniors\n"
    f"1940 - Inauguración de La Bombonera\n\n"
    f"Texto: {PARRAFO_VELEZ}",
    "FEW-SHOT - con ejemplos del formato esperado",
)


# --- EJERCICIO 3: Chain-of-thought ---

print("\n\n*** EJERCICIO 3: CHAIN-OF-THOUGHT ***")

llamar(
    f"¿Cuántos años pasaron entre la fundación de Vélez y su Copa "
    f"Intercontinental?\n\nTexto: {PARRAFO_VELEZ}",
    "SIN chain-of-thought",
)

llamar(
    f"¿Cuántos años pasaron entre la fundación de Vélez y su Copa "
    f"Intercontinental? Pensá paso a paso: primero identificá cada fecha, "
    f"después hacé el cálculo.\n\nTexto: {PARRAFO_VELEZ}",
    "CON chain-of-thought - le pedimos que razone",
)

print("\n\n*** FIN DE LOS EJERCICIOS ***")
print("Observá las diferencias entre cada par de respuestas.")
