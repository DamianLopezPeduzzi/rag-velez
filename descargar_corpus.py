"""Módulo 3: descarga los artículos de Wikipedia sobre Vélez y arma el corpus.

Baja el texto plano de cada artículo, lo limpia (saca secciones de
referencias, líneas vacías repetidas) y lo guarda en corpus/*.txt.
"""

import re
import sys
import requests

sys.stdout.reconfigure(encoding="utf-8")

API_URL = "https://es.wikipedia.org/w/api.php"

# Wikipedia exige identificarse con un User-Agent descriptivo
HEADERS = {"User-Agent": "RAGVelezSarsfield/1.0 (proyecto educativo)"}

ARTICULOS = {
    "velez_principal.txt": "Club Atlético Vélez Sarsfield",
    "velez_historia.txt": "Historia del Club Atlético Vélez Sarsfield",
    "velez_palmares.txt": "Anexo:Palmarés del Club Atlético Vélez Sarsfield",
    "velez_estadio.txt": "Estadio José Amalfitani",
}

# Secciones que no aportan contenido (basura para el RAG)
SECCIONES_BASURA = [
    "Referencias", "Enlaces externos", "Véase también",
    "Bibliografía", "Notas",
]


def descargar_articulo(titulo):
    """Pide a la API de Wikipedia el texto plano de un artículo."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "format": "json",
        "titles": titulo,
        "redirects": 1,
    }
    respuesta = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    respuesta.raise_for_status()
    paginas = respuesta.json()["query"]["pages"]
    pagina = next(iter(paginas.values()))
    if "extract" not in pagina:
        return None
    return pagina["extract"]


def limpiar_texto(texto):
    """Limpieza del corpus: saca secciones basura y normaliza espacios."""
    # Cortar el texto cuando empieza una sección basura (== Referencias ==)
    for seccion in SECCIONES_BASURA:
        patron = rf"\n=+\s*{seccion}\s*=+.*$"
        texto = re.sub(patron, "", texto, flags=re.DOTALL)

    # Sacar los marcadores de sección (== Título ==) pero dejar el título
    texto = re.sub(r"=+\s*(.*?)\s*=+", r"\1", texto)

    # Colapsar 3+ saltos de línea en 2
    texto = re.sub(r"\n{3,}", "\n\n", texto)

    return texto.strip()


def main():
    for nombre_archivo, titulo in ARTICULOS.items():
        print(f"Descargando: {titulo}...")
        texto = descargar_articulo(titulo)

        if texto is None:
            print(f"  [!] No se encontró el artículo '{titulo}', se omite.")
            continue

        texto_limpio = limpiar_texto(texto)
        ruta = f"corpus/{nombre_archivo}"
        with open(ruta, "w", encoding="utf-8") as f:
            f.write(texto_limpio)
        print(f"  OK -> {ruta} ({len(texto_limpio)} caracteres)")

    print("\nCorpus listo en la carpeta corpus/")


if __name__ == "__main__":
    main()
