"""Módulo 4: el mismo RAG, reconstruido con LangChain.

Compará cada pieza con rag.py:
  - Chroma(...)                      <-> chromadb.PersistentClient + get_collection
  - vectorstore.as_retriever(k=3)    <-> recuperar_contexto()
  - ChatPromptTemplate               <-> PROMPT_RAG.format(...)
  - ChatAnthropic                    <-> client.messages.create(...)
  - la cadena (chain) con |          <-> preguntar_rag() que encadena los pasos

LangChain no hace magia: hace exactamente lo mismo que hicimos a mano,
pero con piezas intercambiables y menos código repetido.
"""

import sys
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from chromadb.utils import embedding_functions

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()


# Adaptador: LangChain espera una interfaz de embeddings propia,
# pero queremos usar el MISMO modelo con el que cargamos ChromaDB
class EmbeddingsChroma:
    def __init__(self):
        self._ef = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts):
        return [[float(x) for x in v] for v in self._ef(texts)]

    def embed_query(self, text):
        return [float(x) for x in self._ef([text])[0]]


# 1. Vectorstore: se conecta a la misma base que cargó cargar_chroma.py
vectorstore = Chroma(
    collection_name="velez",
    persist_directory="chroma_db",
    embedding_function=EmbeddingsChroma(),
)

# 2. Retriever: equivale a nuestra función recuperar_contexto()
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. Prompt: el mismo del Módulo 2, en formato LangChain
prompt = ChatPromptTemplate.from_template(
    """Respondé la pregunta del usuario usando SOLO la siguiente información de contexto.
Si la respuesta no está en el contexto, decí "No tengo información suficiente para responder eso."
No inventes datos.

Contexto:
{contexto}

Pregunta: {pregunta}"""
)

# 4. Modelo: equivale a client.messages.create()
llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=512)


def formatear_docs(docs):
    return "\n\n---\n\n".join(d.page_content for d in docs)


# 5. La cadena (LCEL): conecta retrieval -> prompt -> modelo -> texto
chain = (
    {"contexto": retriever | formatear_docs, "pregunta": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    preguntas = [
        "¿En qué año se fundó Vélez?",
        "¿Qué pasó en 1994?",
    ]
    for p in preguntas:
        print("=" * 60)
        print(f"P: {p}")
        print(f"R: {chain.invoke(p)}\n")
