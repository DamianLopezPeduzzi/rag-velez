"""Módulo 5 (+ 6): el RAG expuesto como API web con FastAPI.

Módulo 5 dejó un endpoint /preguntar que respondía de una sola vez.
Módulo 6 agrega:
  - GET /            -> sirve la interfaz de chat (index.html)
  - POST /chat       -> respuesta en STREAMING + historial de conversación

Para levantarla:
    .venv/Scripts/uvicorn api:app --reload

Y usarla:
    http://localhost:8000        (la interfaz de chat)
    http://localhost:8000/docs   (la doc interactiva automática)
"""

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from rag import preguntar_rag, preguntar_rag_stream

app = FastAPI(
    title="RAG Vélez Sársfield",
    description="Asistente que responde preguntas sobre la historia de Vélez",
)


# ---- Modelos de entrada/salida (Pydantic valida los tipos) ----

class Pregunta(BaseModel):
    texto: str


class Respuesta(BaseModel):
    pregunta: str
    respuesta: str


class Turno(BaseModel):
    role: str       # "user" o "assistant"
    content: str


class MensajeChat(BaseModel):
    texto: str
    historial: list[Turno] = []   # turnos previos de la conversación
    rerank: bool = False          # activar el re-ranking del Módulo 6c


# ---- Endpoints ----

@app.get("/")
def raiz():
    """Sirve la interfaz de chat (Módulo 6d)."""
    return FileResponse("index.html")


@app.post("/preguntar", response_model=Respuesta)
def preguntar(pregunta: Pregunta):
    """Módulo 5: respuesta completa de una sola vez (sin streaming ni historial)."""
    respuesta = preguntar_rag(pregunta.texto, mostrar_fuentes=False)
    return Respuesta(pregunta=pregunta.texto, respuesta=respuesta)


@app.post("/chat")
def chat(mensaje: MensajeChat):
    """Módulo 6: respuesta en streaming, recordando el historial de la conversación.

    Devuelve un StreamingResponse: los tokens viajan al browser a medida que
    Claude los genera, en vez de esperar a que termine toda la respuesta.
    """
    historial = [t.model_dump() for t in mensaje.historial]
    generador = preguntar_rag_stream(
        mensaje.texto, historial=historial, rerank=mensaje.rerank
    )
    return StreamingResponse(generador, media_type="text/plain; charset=utf-8")
