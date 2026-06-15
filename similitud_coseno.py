"""Módulo 3: similitud coseno calculada "a mano".

Genera embeddings de frases con el modelo local de ChromaDB
(all-MiniLM-L6-v2) y mide qué tan parecidas son con SciPy.

La intuición: un embedding convierte texto en un vector de números que
captura su SIGNIFICADO. Dos frases que dicen lo mismo con distintas
palabras tienen vectores parecidos -> similitud coseno alta.
"""

import sys
from scipy.spatial.distance import cosine
from chromadb.utils import embedding_functions

sys.stdout.reconfigure(encoding="utf-8")

# Mismo modelo de embeddings que vamos a usar para el RAG.
# IMPORTANTE: hay que usar el MISMO modelo para documentos y preguntas.
embedder = embedding_functions.DefaultEmbeddingFunction()


def similitud(frase_a, frase_b):
    """Similitud coseno entre dos frases. 1 = idénticas, 0 = nada que ver."""
    vec_a, vec_b = embedder([frase_a, frase_b])
    # scipy.cosine devuelve la DISTANCIA coseno; similitud = 1 - distancia
    return 1 - cosine(vec_a, vec_b)


# --- Frases PARECIDAS (dicen lo mismo con otras palabras) ---
parecida_1 = "Vélez ganó la Intercontinental en 1994"
parecida_2 = "El club se consagró campeón del mundo en 1994"

# --- Frases DISTINTAS (no tienen nada que ver) ---
distinta_1 = "Vélez ganó la Intercontinental en 1994"
distinta_2 = "La receta de la milanesa lleva pan rallado y huevo"

print("=" * 60)
print("FRASES PARECIDAS:")
print(f"  A: {parecida_1}")
print(f"  B: {parecida_2}")
print(f"  Similitud coseno: {similitud(parecida_1, parecida_2):.4f}")

print()
print("FRASES DISTINTAS:")
print(f"  A: {distinta_1}")
print(f"  B: {distinta_2}")
print(f"  Similitud coseno: {similitud(distinta_1, distinta_2):.4f}")

print()
print("Un embedding tiene", len(embedder([parecida_1])[0]), "dimensiones.")
print("Las frases parecidas deberían dar un número claramente más alto.")
