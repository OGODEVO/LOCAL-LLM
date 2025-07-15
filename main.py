import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
# from langchain.vectorstores.utils import filter_complex_metadata
import os
import tempfile
from dotenv import load_dotenv
import argparse
import glob

print("Loading environment variables...")
load_dotenv()
print("Environment variables loaded.")

# Define a persistent directory for ChromaDB
CHROMA_DB_DIR = "chroma_db"

class Chatbot:

    def __init__(self):
        print("Initializing Chatbot...")
        ollama_endpoint = os.getenv("OLLAMA_ENDPOINT")
        print(f"Ollama endpoint: {ollama_endpoint}")

        print("Initializing ChatOllama...")
        self.model = ChatOllama(base_url=ollama_endpoint, model="gemma:2b")
        print("ChatOllama initialized.")

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.prompt = PromptTemplate.from_template(
            """
            <s>[INST] You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. [/INST] </s>
            [INST] Question: {question}
            Context: {context}
            Answer: [/INST]
            """
        )

        print("Initializing HuggingFaceEmbeddings...")
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        print("HuggingFaceEmbeddings initialized.")
        
        print("Initializing ChromaDB...")
        self.vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=self.embedding_function)
        print("ChromaDB initialized.")

                self.retriever = self.vector_store.as_retriever()
        print("Chatbot initialized successfully.")


    def ingest(self, pdf_file_path: str):
        try:
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            chunks = self.text_splitter.split_documents(docs)
            # chunks = filter_complex_metadata(chunks) # Re-enable if needed, was commented out in original
            self.vector_store.add_documents(documents=chunks)
            self.vector_store.persist()
            print(f"Successfully ingested {pdf_file_path}")
        except Exception as e:
            print(f"Error ingesting {pdf_file_path}: {e}")

    def ingest_pdfs_from_folder(self, folder_path: str):
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        if not pdf_files:
            print(f"No PDF files found in {folder_path}")
            return

        for pdf_file in pdf_files:
            print(f"Ingesting {pdf_file}...")
            self.ingest(pdf_file)
        print("PDF ingestion from folder complete.")


    @property
    def chain(self):
        return (
                        {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local RAG Chatbot")
    parser.add_argument("--ingest-pdfs", action="store_true", help="Ingest PDFs from the 'pdfs/' folder")
    args = parser.parse_args()

    if args.ingest_pdfs:
        print("Ingesting PDFs from 'pdfs/' folder...")
        chatbot_instance = Chatbot()
        chatbot_instance.ingest_pdfs_from_folder("pdfs/")
        print("PDF ingestion complete. Exiting.")
    else:
        print("Checking for chatbot in session state...")
        if "chatbot" not in st.session_state:
            print("Creating new chatbot instance...")
            st.session_state.chatbot = Chatbot()
            print("Chatbot instance created and stored in session state.")

        st.title("Local RAG Chatbot")

        with st.sidebar:
            st.header("Upload Documents")
            pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
            if st.button("Process"):
                if pdf_file:
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(pdf_file.getvalue())
                            tmp_file_path = tmp_file.name
                        
                        st.session_state.chatbot.ingest(tmp_file_path)
                        os.remove(tmp_file_path) # Clean up the temporary file
                        st.success("File processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing file: {e}")
                else:
                    st.error("Please upload a PDF file.")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                try:
                    chain = st.session_state.chatbot.chain
                    response = chain.invoke(prompt)
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
