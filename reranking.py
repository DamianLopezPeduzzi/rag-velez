"""Módulo 6c: demostración del re-ranking.

El retrieval vectorial trae los chunks "más cercanos" en el espacio de
embeddings. Pero cercano no siempre es lo más útil. El re-ranking agrega
un segundo paso que re-ordena esos chunks por relevancia real.

Este script muestra, para una misma pregunta, qué chunks trae el retrieval
solo vs. cuáles quedan después de re-rankear. Así ves el efecto con tus ojos.
"""

import sys
from rag import recuperar_contexto

sys.stdout.reconfigure(encoding="utf-8")


def mostrar(titulo, chunks, fuentes):
    print(f"\n{titulo}")
    print("-" * 60)
    for i, (chunk, fuente) in enumerate(zip(chunks, fuentes), 1):
        print(f"{i}. [{fuente}] {chunk[:120].strip()}...")


def main():
    preguntas = [
        "¿Quién era el técnico de Vélez cuando ganó la Libertadores 1994?",
        "¿De qué color es la camiseta de Vélez?",
    ]

    for pregunta in preguntas:
        print("=" * 60)
        print(f"PREGUNTA: {pregunta}")

        # Sin re-ranking: orden tal cual lo devuelve ChromaDB
        sin, fuentes_sin = recuperar_contexto(pregunta, rerank=False)
        mostrar("SIN re-ranking (solo cercanía de embeddings):", sin, fuentes_sin)

        # Con re-ranking: recupera el doble y Claude elige los mejores
        con, fuentes_con = recuperar_contexto(pregunta, rerank=True)
        mostrar("CON re-ranking (Claude re-ordena por utilidad):", con, fuentes_con)
        print()


if __name__ == "__main__":
    main()
