# Knowledge Assistant 🤖📚

Sistema de análisis inteligente de documentación técnica usando RAG, LangGraph y múltiples agentes.

## 📋 Requisitos Previos

- Docker & Docker Compose
- Python 3.11+ (si ejecutas sin Docker)
- OpenAI API Key

## 🚀 Instalación Rápida

### 1. Clonar y configurar

```bash
git clone <tu-repo>
cd knowledge-assistant
```

### 2. Configurar variables de entorno

```bash
cp .env.example .env
# Edita .env y añade tu OPENAI_API_KEY
```

### 3. Levantar con Docker

```bash
docker-compose up --build
```

La aplicación estará disponible en: `http://localhost:7860`

## 📁 Estructura del Proyecto

```
knowledge-assistant/
├── src/
│   ├── ingestion/      # Carga y procesamiento de documentos
│   ├── rag/            # Sistema RAG (ChromaDB + LlamaIndex)
│   ├── agents/         # Agentes del workflow
│   ├── workflow/       # Orquestación con LangGraph
│   ├── prompts/        # Templates de prompts
│   └── ui/             # Interfaz Gradio
├── config/             # Configuraciones YAML/JSON
├── data/               # Documentos y base vectorial
├── airflow/            # Tareas programadas (opcional)
└── tests/              # Tests unitarios
```

## 🎯 Uso

### Agregar Documentos

1. Coloca tus PDFs/Markdown en `data/raw/`
2. Ejecuta el proceso de ingesta:

```bash
docker-compose exec app python src/ingestion/loader.py
```

### Hacer Consultas

Abre `http://localhost:7860` y empieza a preguntar sobre tus documentos.

## 🛠️ Desarrollo Local (sin Docker)

```bash
# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
python src/ui/app.py
```

## 📚 Tecnologías Utilizadas

- **LangChain**: Orquestación de LLMs y prompts
- **LangGraph**: Workflow con múltiples agentes
- **LlamaIndex**: Framework RAG
- **ChromaDB**: Base de datos vectorial
- **Gradio**: Interfaz de usuario
- **Docker**: Contenedorización

## 🔧 Configuración Avanzada

Edita `config/agents.yaml` para ajustar:
- Modelos de LLM
- Parámetros de RAG (top_k, threshold)
- Prompts de sistema de cada agente

## 📝 TODOs

- [ ] Configurar entorno ✅ (Estamos aquí)
- [ ] Implementar ingesta de documentos
- [ ] Crear sistema RAG básico
- [ ] Desarrollar agentes
- [ ] Integrar LangGraph
- [ ] Construir UI con Gradio
- [ ] Añadir Airflow para automatización

## 📄 Licencia

MIT