# Deploy del RAG Vélez en AWS (App Runner)

Guía para publicar el asistente en la nube y obtener una URL HTTPS pública.

Hay **dos caminos**. Elegí uno:

- **Camino A — Docker + ECR (recomendado):** construís una imagen, la subís al registro
  de AWS (ECR) y App Runner la corre. Es el estándar de la industria y lo que más suma
  para tu certificación MLA-C01.
- **Camino B — Sin Docker (más simple):** App Runner clona tu repo de GitHub y lo construye
  solo, usando `apprunner.yaml`. No necesitás instalar Docker.

> **Regla de oro (igual que en local):** la `ANTHROPIC_API_KEY` se configura como **variable
> de entorno del servicio** en AWS. NUNCA va en la imagen, en el repo ni en este archivo.

---

## Requisitos previos (para ambos caminos)

1. Una **cuenta de AWS** (https://aws.amazon.com/). Pedirá tarjeta; App Runner no es gratis (ver Costos).
2. **AWS CLI** instalada: https://aws.amazon.com/cli/
3. Configurá tus credenciales una vez:
   ```bash
   aws configure
   # te pide: Access Key, Secret Key, región (ej: us-east-1), formato (json)
   ```
   (Las Access Keys se crean en la consola: IAM → Users → tu usuario → Security credentials.)

---

## Camino A — Docker + ECR

### A0. Instalar Docker
- Docker Desktop: https://www.docker.com/products/docker-desktop/
- Verificá: `docker --version`

### A1. Probar la imagen LOCAL primero (gratis, antes de tocar AWS)
Desde la carpeta `rag-velez/`:
```bash
docker build -t rag-velez .
docker run -p 8080:8080 -e ANTHROPIC_API_KEY=TU_API_KEY rag-velez
```
Abrí http://localhost:8080 y probá una pregunta. Si funciona, seguí. Si no, lo arreglamos
antes de pagar nada.

### A2. Crear el repositorio en ECR
```bash
aws ecr create-repository --repository-name rag-velez --region us-east-1
```
Anotá el `repositoryUri` que devuelve (algo como `123456789.dkr.ecr.us-east-1.amazonaws.com/rag-velez`).

### A3. Login + subir la imagen
```bash
# Reemplazá 123456789 por tu Account ID y la región si cambia
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com

docker tag rag-velez:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/rag-velez:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/rag-velez:latest
```

### A4. Crear el servicio en App Runner (consola)
1. Consola AWS → **App Runner** → **Create service**.
2. Source: **Container registry** → **Amazon ECR** → elegí la imagen `rag-velez:latest`.
3. Deployment: **Manual** (o automático si querés que redeploye al pushear).
4. Service settings:
   - **Port: 8080**
   - **Environment variables** → agregá `ANTHROPIC_API_KEY` con tu key.
   - CPU/Memoria: **1 vCPU / 2 GB** (suficiente para los embeddings).
5. Create & deploy. En unos minutos te da una **URL pública HTTPS**. Listo.

---

## Camino B — Sin Docker (App Runner desde GitHub)

El archivo `apprunner.yaml` ya está en el repo.

1. Consola AWS → **App Runner** → **Create service**.
2. Source: **Source code repository** → conectá tu cuenta de GitHub → elegí el repo
   `rag-velez` y la rama `master`.
3. Deployment: automático (redeploya cuando pusheás).
4. Build: **Use a configuration file** (detecta `apprunner.yaml` solo).
5. Service settings:
   - **Port: 8080** (ya está en el yaml).
   - **Environment variables** → agregá `ANTHROPIC_API_KEY` con tu key.
   - CPU/Memoria: **1 vCPU / 2 GB**.
6. Create & deploy → URL pública HTTPS.

---

## Seguridad de la API key (mejor práctica)

Lo mínimo: cargarla como **environment variable** del servicio (pasos de arriba).

Lo ideal en producción: guardarla en **AWS Secrets Manager** y referenciarla desde App Runner,
así no queda visible en texto plano en la config del servicio.

---

## Costos (importante para no llevarte sorpresas)

- App Runner cobra por el contenedor aprovisionado + las requests. Un servicio chico siempre
  encendido ronda **~15–25 USD/mes**.
- Mientras aprendés: cuando no lo uses, **pausá o borrá el servicio** (App Runner → tu servicio
  → Pause/Delete) para no pagar de más.
- Aparte, cada pregunta consume tokens de tu cuenta de Anthropic (eso corre por separado).

---

## Cómo actualizar después de un cambio

- **Camino A:** `docker build` → `docker push` la nueva imagen → App Runner redeploya
  (automático si lo configuraste, o "Deploy" manual).
- **Camino B:** `git push` a `master` → App Runner redeploya solo.
