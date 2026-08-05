# Course Assistant (RAG System)

A Retrieval-Augmented Generation (RAG) web application built to help users interactively query course materials. The app leverages a local vector database for fast retrieval and an LLM to answer questions specifically based on the provided documents.

**[🚀 Try the Live Demo here!](https://courseragproject.streamlit.app/)**

## Features

- **Conversational Interface:** A modern chat interface built with Streamlit to ask questions easily.
- **Document Ingestion:** Supports automated loading and chunking of multiple document formats (PDF, DOCX, CSV, TXT).
- **Fast Local Search:** Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) and a local Chroma Vector Database for efficient text retrieval.
- **Accurate Answers:** Connects to Groq's fast Llama-3 API to provide accurate answers restricted to your course's context.
- **Source Verification:** Every answer includes expandables displaying the exact source files, pages, and snippets used to formulate the response.

## Project Structure

```
.
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (API keys)
├── data/
│   ├── raw_courses/        # Put your raw PDF, DOCX, CSV, or TXT documents here
│   └── vector_db/          # Automatically generated local vector database
└── src/
    ├── ingestion.py        # Logic for loading and chunking raw documents
    ├── embeddings.py       # Logic for generating embeddings and creating the vector DB
    └── retriever_llm.py    # LangChain setup for retrieving data and generating answers
```

## Running Locally

**1. Clone the repository and navigate to the project folder:**
```bash
cd course_rag_project
```

**2. Create a virtual environment and activate it (optional but recommended):**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Set up environment variables:**
Create a `.env` file in the `src` folder (or at the project root) and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

**5. Initialize the Vector Database & Run**
Generate the database locally and launch Streamlit:
```bash
python src/embeddings.py
streamlit run app.py
```
