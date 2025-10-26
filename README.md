# Local RAG Chatbot with Agentic Capabilities and Web Search.

This project provides a local Retrieval-Augmented Generation (RAG) chatbot with agentic capabilities, allowing it to answer questions based on your local PDF documents, perform web searches for information, or directly use a local Large Language Model (LLM). The application is built with Streamlit for an interactive user interface.

## Project Goal

The primary goal of this project is to create an accessible and user-friendly container where individuals with less technical expertise can easily manage their data, perform Retrieval-Augmented Generation (RAG) on their documents, and conduct internet searches, all with minimal or no coding required. It aims to democratize access to powerful AI capabilities for personal knowledge management and information retrieval

## Features

*   **Local LLM Integration:** Utilizes `ChatOllama` to connect with local LLMs (defaulting to `gemma:2b`).
*   **Retrieval-Augmented Generation (RAG):**
    *   Ingests PDF documents to create a knowledge base.
    *   Uses `HuggingFaceEmbeddings` for document embeddings.
    *   Leverages `ChromaDB` for persistent vector storage.
    *   Answers questions by retrieving relevant information from your ingested PDFs.
*   **Web Search Integration:**
    *   Performs web searches using DuckDuckGo (`ddgs`).
    *   Scrapes and extracts clean text from web pages using `readability-lxml` to provide context for answers.
*   **Agentic Workflow:**
    *   Employs `langchain.agents` to enable the LLM to intelligently decide whether to use the RAG system or perform a web search based on the user's query
*   **Multiple Context Modes:**
    *   **RAG Mode:** Answers questions using your ingested PDF documents.
    *   **Web Search Mode:** Fetches information from the web to answer questions.
    *   **Local LLM Text Mode:** Directly queries the local LLM for general knowledge.
*   **Streamlit User Interface:**
    *   An intuitive web interface for chatting with the bot.
    *   Allows easy uploading of PDF files for ingestion.
    *   Provides options to switch between different context sources.
*   **Environment Variable Configuration:** Easily configure your Ollama endpoint using a `.env` file.

## Setup

### Prerequisites

*   Python 3.8+
*   Ollama installed and running with the `gemma:2b` model pulled. You can pull the model using:
    ```bash
    ollama pull gemma:2b
    ```

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/locallm-1.git # Replace with your actual repo URL
    cd locallm-1
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows: `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure environment variables:**
    Create a `.env` file in the root directory of the project with your Ollama endpoint.
    ```
    OLLAMA_ENDPOINT=http://localhost:11434 # Or your Ollama server address
    ```
    (Refer to `.env.example` for an example.)

## Usage

### 1. Ingesting PDF Documents

You can ingest PDF documents in two ways:

#### a) From a `pdfs/` folder (CLI)

1.  Create a folder named `pdfs` in the root directory of the project.
2.  Place your PDF files inside the `pdfs/` folder.
3.  Run the ingestion script:
    ```bash
    python main.py --ingest-pdfs
    ```
    This will process all PDFs in the `pdfs/` folder and store their embeddings in `chroma_db/`.

#### b) Via Streamlit UI

1.  Start the Streamlit application (see next section).
2.  Use the "Upload a PDF file" option in the sidebar to upload and process individual PDF documents.

### 2. Running the Chatbot

1.  Start the Streamlit application:
    ```bash
    streamlit run main.py
    ```
2.  Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

### Interacting with the Chatbot

*   **Select Context Source:** In the sidebar, choose between "Web Search Mode", "Local LLM Text Mode", or "RAG Mode" based on how you want the chatbot to answer your questions.
*   **Ask Questions:** Type your questions in the chat input box.
*   **Upload PDFs:** Use the file uploader in the sidebar to add new documents to your knowledge base.

## Project Structure

```
.
├── .env.example           # Example environment variables
├── .gitignore             # Git ignore file
├── LICENSE                # Project license
├── main.py                # Main application logic (Streamlit app, RAG, Agent)
├── push.sh                # (Optional) Script for pushing changes
├── README.md              # This README file
├── requirements.txt       # Python dependencies
├── test_ollama.py         # Script to test Ollama connectivity
├── web_search.py          # Utility for web searching and scraping
├── __pycache__/           # Python cache directory
├── .git/                  # Git repository files.
└── chroma_db/             # Persistent directory for ChromaDB vector store
```

## Testing Ollama Connectivity

You can test if your Ollama setup is working correctly by running:

```bash
python test_ollama.py
```

This script will attempt to connect to your configured Ollama endpoint and invoke the `gemma:2b` model with a simple query.
