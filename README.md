# Automatic_chapterization_and_Question_Answering

# Video Learning Tools

**Autores:** Daniel Meyer, Eduard Mendez

Este repositorio contiene un proyecto que demuestra la utilización de técnicas avanzadas de procesamiento de lenguaje natural (NLP) para la transcripción automática de videos, segmentación de texto y respuesta a preguntas. Utiliza modelos de última generación como Whisper para la transcripción y otros modelos de lenguaje grande (LLM) para la segmentación semántica y generación de respuestas.

## Contenido del Repositorio

- **Transcripción de Video con Whisper:** Utiliza el modelo Whisper para transcribir videos de manera eficiente.
- **Segmentación Semántica de Texto:** Implementa técnicas de tokenización y segmentación para dividir el texto en secciones coherentes.
- **Generación de Títulos:** Genera títulos para las secciones segmentadas utilizando técnicas avanzadas de NLP.
- **Generación de Embeddings:** Utiliza modelos de transformers para generar embeddings de los segmentos de texto.
- **Almacenamiento Vectorial:** Implementa un almacén vectorial utilizando la librería Annoy para búsquedas eficientes.
- **Respuesta a Preguntas:** Construye un pipeline para responder preguntas basadas en los segmentos de texto más relevantes.

## Instalación

Primero, instala las dependencias necesarias:

```sh
pip install -U -q openai-whisper
pip install -U -q transformers bitsandbytes accelerate loralib
pip install -U langchain-community
```
Uso
Transcripción de Video

# Uso

## Transcripción de Video

```python
import whisper

video_path = "/path/to/video.mp4"
model = whisper.load_model("small.en")
result = model.transcribe(video_path)
transcription = result['text']
```
## Segmentación Semántica de Texto

```python
from nltk.tokenize import PunktSentenceTokenizer, TextTilingTokenizer

st = PunktSentenceTokenizer()
tt = TextTilingTokenizer(w=50)
segmented_sections = tt.tokenize(transcription)
```

## Generación de Embeddings
```python
from langchain.embeddings import HuggingFaceEmbeddings

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = [embeddings_model.embed(text) for text in segmented_sections]
```

## Almacenamiento Vectorial

```python
from langchain.vectorstores import Annoy

vector_store = Annoy.from_texts(segmented_sections, embeddings_model)
```

### Respuesta a Preguntas
```python
def get_qs(question):
    top3_chunks = [text.page_content for text in vector_store.similarity_search(question, k=3)]
    return top3_chunks

question = "¿De qué trata el video?"
relevant_segments = get_qs(question)

for segment in relevant_segments:
    print(segment)
```
### Resultados

Presentamos algunos ejemplos de preguntas y respuestas para demostrar el rendimiento de la solución:

Pregunta: ¿Quién es Yann LeCun?

Respuesta: Yann LeCun es un científico en el campo de la inteligencia artificial...
Conclusión

En nuestro estudio, introdujimos dos herramientas: capítulos automáticos y respuesta a preguntas, demostrando la eficacia de modelos de lenguaje grandes en la transcripción y análisis de videos.
### Contribución

Si deseas contribuir a este proyecto, por favor abre un issue o un pull request.
