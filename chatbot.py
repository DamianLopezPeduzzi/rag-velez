import sys
import time
from dotenv import load_dotenv
from anthropic import Anthropic, RateLimitError, APIConnectionError, APIStatusError

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

client = Anthropic()

SYSTEM_MESSAGE = (
    "Sos un asistente experto en la historia del Club Atlético Vélez Sársfield. "
    "Respondé en español rioplatense, de forma clara y concisa. "
    "Si no sabés algo, decilo honestamente."
)


def preguntar(mensaje, historial):
    historial.append({"role": "user", "content": mensaje})

    try:
        respuesta = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            system=SYSTEM_MESSAGE,
            messages=historial,
        )
        texto = respuesta.content[0].text
        historial.append({"role": "assistant", "content": texto})
        return texto, historial

    except RateLimitError:
        historial.pop()
        print("\n[Rate limit alcanzado, reintentando en 5 segundos...]")
        time.sleep(5)
        return preguntar(mensaje, historial)

    except APIConnectionError:
        historial.pop()
        return "[Error: no se pudo conectar con la API. Revisá tu conexión.]", historial

    except APIStatusError as e:
        historial.pop()
        return f"[Error de la API: {e.status_code} - {e.message}]", historial


def main():
    historial = []
    print("=== Chatbot Vélez Sársfield ===")
    print("Escribí tu pregunta (o 'salir' para terminar)\n")

    while True:
        pregunta = input("Vos: ").strip()
        if not pregunta:
            continue
        if pregunta.lower() in ("salir", "exit", "quit"):
            print("¡Hasta la próxima!")
            break

        respuesta, historial = preguntar(pregunta, historial)
        print(f"\nAsistente: {respuesta}\n")


if __name__ == "__main__":
    main()
