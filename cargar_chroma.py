"""Módulo 3/4: parte el corpus en chunks y lo carga en ChromaDB.

El chunking es donde está el 80% de los problemas de calidad de un RAG:
- Chunks muy grandes -> traen ruido al contexto
- Chunks muy chicos -> pierden el contexto de lo que dicen

Acá usamos chunks por párrafos agrupados (~1000 caracteres con
solapamiento), un punto de partida razonable para experimentar.
"""

import os
import sys
import chromadb

sys.stdout.reconfigure(encoding="utf-8")

CARPETA_CORPUS = "corpus"
CARPETA_DB = "chroma_db"
NOMBRE_COLECCION = "velez"

TAMANO_CHUNK = 1000   # caracteres objetivo por chunk
SOLAPAMIENTO = 200    # caracteres que se repiten entre chunks consecutivos


def partir_en_chunks(texto, tamano=TAMANO_CHUNK, solapamiento=SOLAPAMIENTO):
    """Parte un texto en chunks por párrafos, agrupando hasta ~tamano."""
    parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]
    chunks = []
    actual = ""

    for parrafo in parrafos:
        if len(actual) + len(parrafo) <= tamano:
            actual = f"{actual}\n\n{parrafo}".strip()
        else:
            if actual:
                chunks.append(actual)
            # Arrancamos el nuevo chunk con el final del anterior (solapamiento)
            cola = actual[-solapamiento:] if actual else ""
            actual = f"{cola}\n\n{parrafo}".strip()

    if actual:
        chunks.append(actual)
    return chunks


def main():
    cliente = chromadb.PersistentClient(path=CARPETA_DB)

    # Si la colección ya existía, la borramos para cargar de cero
    try:
        cliente.delete_collection(NOMBRE_COLECCION)
        print(f"Colección '{NOMBRE_COLECCION}' anterior borrada.")
    except Exception:
        pass

    coleccion = cliente.create_collection(NOMBRE_COLECCION)

    total = 0
    for archivo in sorted(os.listdir(CARPETA_CORPUS)):
        if not archivo.endswith(".txt"):
            continue

        ruta = os.path.join(CARPETA_CORPUS, archivo)
        with open(ruta, encoding="utf-8") as f:
            texto = f.read()

        chunks = partir_en_chunks(texto)
        ids = [f"{archivo}-{i}" for i in range(len(chunks))]
        metadatas = [{"fuente": archivo} for _ in chunks]

        coleccion.add(documents=chunks, ids=ids, metadatas=metadatas)
        total += len(chunks)
        print(f"{archivo}: {len(chunks)} chunks cargados")

    print(f"\nTotal: {total} chunks en la colección '{NOMBRE_COLECCION}'")
    print(f"Base vectorial persistida en {CARPETA_DB}/")

    # Prueba rápida de búsqueda semántica
    print("\n--- Prueba de retrieval ---")
    resultado = coleccion.query(query_texts=["¿Quién fue José Amalfitani?"], n_results=2)
    for doc, meta in zip(resultado["documents"][0], resultado["metadatas"][0]):
        print(f"\n[{meta['fuente']}]")
        print(doc[:200] + "...")


if __name__ == "__main__":
    main()
