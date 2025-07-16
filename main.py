import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
import os
import tempfile
from dotenv import load_dotenv
import argparse
import glob
import requests
import json
from langchain.tools import BaseTool, Tool
from typing import Type, Optional
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate as AgentPromptTemplate # Renamed to avoid conflict
from web_search import get_web_context # Import the new web search utility

print("Loading environment variables...")
load_dotenv()
print("Environment variables loaded.")

# Define a persistent directory for ChromaDB
CHROMA_DB_DIR = "chroma_db"



# --- Chatbot (RAG) Class ---
class Chatbot:
    def __init__(self):
        print("Initializing Chatbot (RAG system)...")
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
        print("Chatbot (RAG system) initialized successfully.")

    def ingest(self, pdf_file_path: str):
        try:
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            chunks = self.text_splitter.split_documents(docs)
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

    def get_rag_chain(self):
        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.model
            | StrOutputParser()
        )

    def web_chain(self, query: str) -> str:
        context = get_web_context(query)
        # Clean and truncate the text to about 3000 characters
        cleaned_context = " ".join(context.split()) # Remove extra whitespace
        truncated_context = cleaned_context[:3000] # Truncate
        
        # Format the prompt with the question and context
        full_prompt = self.prompt.format(question=query, context=truncated_context)
        
        # Pass the prompt to the LLM model and return the response
        return self.model.invoke(full_prompt)

# --- RAG Tool Definition (wraps Chatbot functionality) ---
class RAGToolInput(BaseModel):
    query: str = Field(description="The question to answer using the ingested documents.")

class RAGTool(BaseTool):
    name: str = "Document Search"
    description: str = (
        "A tool for answering questions based on previously ingested PDF documents. "
        "Use this tool when the user's question is likely to be answered by the content of the loaded documents. "
        "Input should be a clear and concise question related to the documents."
    )
    args_schema: Type[BaseModel] = RAGToolInput
    chatbot_instance: Chatbot = None # Will be set during initialization

    def _run(self, query: str) -> str:
        """Answers a question using the RAG system."""
        if not self.chatbot_instance:
            return "Error: RAG system not initialized."
        try:
            chain = self.chatbot_instance.get_rag_chain()
            response = chain.invoke(query)
            return response
        except Exception as e:
            return f"Error using RAG tool: {e}"

    async def _arun(self, query: str) -> str:
        """Asynchronous run is not implemented for this tool."""
        raise NotImplementedError("RAGTool does not support async operation yet.")

# --- Main execution logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local RAG Chatbot with Agent capabilities")
    parser.add_argument("--ingest-pdfs", action="store_true", help="Ingest PDFs from the 'pdfs/' folder")
    args = parser.parse_args()

    # Initialize the RAG chatbot instance once
    # This instance will be passed to the RAGTool
    rag_chatbot_instance = Chatbot()

    if args.ingest_pdfs:
        print("Ingesting PDFs from 'pdfs/' folder...")
        rag_chatbot_instance.ingest_pdfs_from_folder("pdfs/")
        print("PDF ingestion complete. Exiting.")
    else:
        print("Setting up agent for Streamlit UI...")
        if "agent_executor" not in st.session_state:
            print("Creating new agent executor instance...")
            
            # Initialize tools
            rag_tool_instance = RAGTool(chatbot_instance=rag_chatbot_instance)

            tools = [
                rag_tool_instance,
            ]

            # Define the agent prompt
            # This prompt guides the LLM on how to use the tools
            agent_prompt = AgentPromptTemplate.from_template("""
            You are a helpful AI assistant. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            If you can answer the question directly without using any tools, provide the Final Answer immediately.

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}
            """)

            # Initialize the LLM for the agent
            # Using the same Ollama model as your RAG system
            agent_llm = ChatOllama(base_url=os.getenv("OLLAMA_ENDPOINT"), model="gemma:2b")

            # Create the agent
            agent = create_react_agent(agent_llm, tools, agent_prompt)

            # Create the agent executor
            st.session_state.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
            print("Agent executor created and stored in session state.")

        st.title("Local RAG Chatbot with Web Search")

        with st.sidebar:
            st.header("Context Source")
            context_source = st.radio(
                "Select context source:",
                ("Web Search Mode", "Local LLM Text Mode", "RAG Mode"),
                index=2, # Default to RAG Mode
                key="context_source_radio"
            )

            st.header("Upload Documents")
            pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])
            if st.button("Process", key="process_pdf_button") and pdf_file:
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(pdf_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    rag_chatbot_instance.ingest(tmp_file_path) # Use the shared instance
                    os.remove(tmp_file_path) # Clean up the temporary file
                    st.success("File processed successfully!")
                except Exception as e:
                    st.error(f"Error processing file: {e}")
            elif st.button("Process", key="process_no_pdf_button") and not pdf_file:
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
                    if context_source == "Web Search Mode":
                        response_content = rag_chatbot_instance.web_chain(prompt).content
                    elif context_source == "Local LLM Text Mode":
                        response_content = rag_chatbot_instance.model.invoke(prompt).content
                    else: # RAG Mode
                        # Check if there are documents in ChromaDB
                        if rag_chatbot_instance.vector_store._collection.count() > 0:
                            response_content = st.session_state.agent_executor.invoke({"input": prompt})["output"]
                        else:
                            st.warning("No local documents found. Falling back to Web Search Mode.")
                            response_content = rag_chatbot_instance.web_chain(prompt)
                    
                    st.markdown(response_content)
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})