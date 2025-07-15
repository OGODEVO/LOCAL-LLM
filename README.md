# Local RAG Chatbot

This is a local AI chatbot built with Streamlit, LangChain, and Ollama (gemma:2b).

## Features

- **Local Ollama Integration**: Uses `gemma:2b` running locally.
- **Streamlit UI**: Interactive chat interface for easy use.
- **PDF Ingestion**: Upload PDFs via the Streamlit UI or ingest from a local `pdfs/` folder.
- **ChromaDB**: Persists vector store for efficient retrieval.
- **Offline First**: Designed to run completely offline.

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <your-github-repo-url>
    cd <your-repo-name>
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ollama Setup**:
    -   Ensure Ollama is running locally at `http://localhost:11434`.
    -   Download the `gemma:2b` model:
        ```bash
        ollama pull gemma:2b
        ```

5.  **Environment Variables**:
    Create a `.env` file in the project root based on `.env.example`:
    ```
    OLLAMA_ENDPOINT=http://localhost:11434
    GITHUB_REPO_URL=https://github.com/youruser/yourrepo.git
    ```

## Usage

### Running the Streamlit App

```bash
streamlit run main.py
```

Access the app in your browser at `http://localhost:8501`.

### Ingesting PDFs from a Local Folder

1.  Place your PDF files into the `pdfs/` directory in the project root.
2.  Run the ingestion script (or function within `main.py` if implemented as such):
    ```bash
    python main.py --ingest-pdfs
    ```
    *(Note: The exact command for ingestion might vary based on implementation. Refer to `main.py` for details.)*

### Pushing to GitHub

Use the provided `push.sh` script to add, commit, and push your changes to GitHub:

```bash
./push.sh
```

Make sure your `GITHUB_REPO_URL` is correctly set in the `.env` file.

## Project Structure

```
.env
.env.example
.gitignore
main.py
push.sh
README.md
requirements.txt
pdfs/             # Directory for local PDF ingestion
chroma_db/        # Persistent ChromaDB vector store
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
