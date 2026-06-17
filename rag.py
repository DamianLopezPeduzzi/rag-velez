"""Módulo 4 (+ 6): el RAG completo, ensamblado a mano (sin framework).

El flujo completo:
  pregunta -> embedding de la pregunta -> retrieval de chunks similares
  -> (opcional) re-ranking de esos chunks -> armado del prompt con contexto
  -> generación con Claude (normal o por streaming)

Cada paso está separado para que se entienda qué hace cada pieza.
El Módulo 6 agrega capacidades sin romper lo anterior:
  - rerankear():                mejora la PRECISIÓN del retrieval (6c)
  - generar_respuesta_stream(): mejora la EXPERIENCIA (tokens en vivo) (6a)
  - historial:                  el modelo entiende preguntas de seguimiento (6b)
"""

import re
import sys
import chromadb
from dotenv import load_dotenv
from anthropic import Anthropic

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

client = Anthropic()
MODELO = "claude-sonnet-4-6"
K_CHUNKS = 4        # cuántos chunks usar finalmente en el prompt
POOL_RERANK = 30    # cuántos candidatos traer cuando se va a re-rankear

# El prompt-plantilla del Módulo 2 (+ reglas de dominio del Módulo 6):
# obliga a responder solo con el contexto y resuelve ambigüedades por defecto.
PROMPT_RAG = """Respondé la pregunta del usuario usando SOLO la siguiente información de contexto.

Reglas:
- Asumí que las preguntas se refieren al fútbol profesional del club, salvo que el usuario aclare otra cosa (básquet, vóley, inferiores, etc.).
- Si preguntan por una cantidad total (por ejemplo, cuántos títulos) y el contexto desglosa categorías (Primera División, copas nacionales, copas internacionales), sumá esas categorías y dá el total oficial, mostrando el desglose. No mezcles otros deportes ni divisiones inferiores salvo que lo pidan.
- Si la respuesta no está en el contexto, decí "No tengo información suficiente para responder eso."
- No inventes datos. Citá el contexto cuando sea posible.

Contexto:
{contexto}

Pregunta: {pregunta}"""

# Prompt del re-ranker: le pedimos a Claude que ordene los chunks por utilidad
PROMPT_RERANK = """Sos un re-ranker. Dada una PREGUNTA y una lista de FRAGMENTOS numerados,
devolvé SOLO los números de los {top} fragmentos más útiles para responder, ordenados
del más útil al menos útil, separados por comas. Ejemplo de respuesta: 2,0,5

PREGUNTA: {pregunta}

FRAGMENTOS:
{fragmentos}"""

# Prompt de reformulación: reescribe la pregunta como una consulta de búsqueda rica.
# Esto opera ANTES del retrieval, para acercar el embedding al chunk correcto.
PROMPT_REFORMULAR = """Reescribí la pregunta del usuario como una consulta de búsqueda rica
en palabras clave y sinónimos, para buscar en documentos sobre la historia del Club Atlético
Vélez Sársfield. Si hay conversación previa, usala para resolver referencias ("eso", "ahí").
Devolvé SOLO la consulta reescrita, en una línea, sin explicaciones ni comillas.

Conversación previa:
{historial}

Pregunta del usuario: {pregunta}"""

# Conexión a la base vectorial cargada por cargar_chroma.py
coleccion = chromadb.PersistentClient(path="chroma_db").get_collection("velez")


def rerankear(pregunta, chunks, fuentes, top=K_CHUNKS):
    """Módulo 6c: re-ordena los chunks por relevancia real usando Claude como juez.

    El retrieval vectorial trae lo "más cercano" en el espacio de embeddings,
    pero cercano != más útil. Este segundo paso afina ese orden.
    """
    fragmentos = "\n\n".join(f"[{i}] {c[:500]}" for i, c in enumerate(chunks))
    prompt = PROMPT_RERANK.format(top=top, pregunta=pregunta, fragmentos=fragmentos)

    r = client.messages.create(
        model=MODELO,
        max_tokens=30,
        messages=[{"role": "user", "content": prompt}],
    )
    # Parseamos los índices que devolvió (ej: "2,0,5" -> [2, 0, 5])
    indices = [int(n) for n in re.findall(r"\d+", r.content[0].text)]

    # Filtramos índices válidos y sin repetir, y nos quedamos con los top
    elegidos = []
    for i in indices:
        if 0 <= i < len(chunks) and i not in elegidos:
            elegidos.append(i)
    elegidos = elegidos[:top] or list(range(min(top, len(chunks))))  # fallback

    chunks_ord = [chunks[i] for i in elegidos]
    fuentes_ord = [fuentes[i] for i in elegidos]
    return chunks_ord, fuentes_ord


def reformular_query(pregunta, historial=None):
    """Módulo 6e: reescribe la pregunta del usuario como una consulta de búsqueda rica.

    Opera ANTES del retrieval. Resuelve el problema de los sinónimos: "¿cuántos
    títulos?" se expande a "palmarés campeonatos Primera División copas...", lo que
    acerca el embedding de la búsqueda al chunk correcto. Si hay historial, también
    resuelve referencias de seguimiento ("¿y en qué año fue eso?").
    """
    if historial:
        ultimos = historial[-4:]  # últimos ~2 turnos, recortados
        previa = "\n".join(f"{m['role']}: {m['content'][:200]}" for m in ultimos)
    else:
        previa = "(sin conversación previa)"

    prompt = PROMPT_REFORMULAR.format(historial=previa, pregunta=pregunta)
    r = client.messages.create(
        model=MODELO,
        max_tokens=80,
        messages=[{"role": "user", "content": prompt}],
    )
    return r.content[0].text.strip()


def recuperar_contexto(pregunta, k=K_CHUNKS, rerank=False):
    """Paso 1-2: embedding de la pregunta + retrieval de los k chunks más similares.

    Si rerank=True, recupera un pool grande de candidatos (POOL_RERANK) y los
    filtra con el re-ranker hasta quedarse con los k mejores. El pool grande es
    clave: para preguntas tipo "cuántos títulos" el chunk correcto a veces queda
    en posición ~15, fuera de un k chico, pero dentro del pool.
    """
    n = POOL_RERANK if rerank else k
    resultado = coleccion.query(query_texts=[pregunta], n_results=n)
    chunks = resultado["documents"][0]
    fuentes = [m["fuente"] for m in resultado["metadatas"][0]]

    if rerank:
        chunks, fuentes = rerankear(pregunta, chunks, fuentes, top=k)

    return chunks, fuentes


def _armar_mensajes(pregunta, chunks, historial=None):
    """Arma la lista de mensajes para la API: historial previo + turno actual con contexto.

    El historial guarda Q/A en texto plano (sin contexto), así no crece de más.
    Solo el turno actual lleva los chunks inyectados.
    """
    contexto = "\n\n---\n\n".join(chunks)
    prompt = PROMPT_RAG.format(contexto=contexto, pregunta=pregunta)
    mensajes = list(historial or [])
    mensajes.append({"role": "user", "content": prompt})
    return mensajes


def generar_respuesta(pregunta, chunks, historial=None):
    """Paso 3-4: arma el prompt con el contexto y genera la respuesta (de una)."""
    respuesta = client.messages.create(
        model=MODELO,
        max_tokens=512,
        messages=_armar_mensajes(pregunta, chunks, historial),
    )
    return respuesta.content[0].text


def generar_respuesta_stream(pregunta, chunks, historial=None):
    """Módulo 6a: igual que generar_respuesta pero emite los tokens a medida que llegan.

    Es un generador: en vez de 'return texto', hace 'yield trozo' muchas veces.
    Quien lo consume (la API web) reenvía cada trozo al browser en tiempo real.
    """
    with client.messages.stream(
        model=MODELO,
        max_tokens=512,
        messages=_armar_mensajes(pregunta, chunks, historial),
    ) as stream:
        for trozo in stream.text_stream:
            yield trozo


def preguntar_rag(pregunta, mostrar_fuentes=True, historial=None, rerank=False, reformular=False):
    """El flujo RAG completo para una pregunta (respuesta de una sola vez).

    Ojo: la BÚSQUEDA usa la query (reformulada o no), pero la GENERACIÓN siempre
    usa la pregunta original del usuario, así Claude responde lo que realmente preguntó.
    """
    query = reformular_query(pregunta, historial) if reformular else pregunta
    chunks, fuentes = recuperar_contexto(query, rerank=rerank)
    respuesta = generar_respuesta(pregunta, chunks, historial)
    if mostrar_fuentes:
        respuesta += f"\n  (fuentes: {', '.join(set(fuentes))})"
    return respuesta


def preguntar_rag_stream(pregunta, historial=None, rerank=False, reformular=False):
    """Módulo 6: flujo RAG completo en modo streaming, con historial opcional."""
    query = reformular_query(pregunta, historial) if reformular else pregunta
    chunks, _ = recuperar_contexto(query, rerank=rerank)
    yield from generar_respuesta_stream(pregunta, chunks, historial)


if __name__ == "__main__":
    preguntas = [
        "¿En qué año se fundó Vélez?",
        "¿Quién fue José Amalfitani?",
        "¿Qué pasó en 1994?",
        "¿Cuántos títulos de Primera División ganó Vélez?",
        # Esta NO está en el corpus: el RAG debe decir que no sabe
        "¿Cuál es la capital de Francia?",
    ]

    for p in preguntas:
        print("=" * 60)
        print(f"P: {p}")
        print(f"R: {preguntar_rag(p)}\n")
