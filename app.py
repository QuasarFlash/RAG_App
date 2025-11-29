from time import time
import streamlit as st
import os
import shutil
import asyncio
from pathlib import Path
from glob import glob
import re

# Langchain & Milvus
# from langchain_community.document_loaders import SimpleDirectoryReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Docling
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions

# Setup Page Config
st.set_page_config(page_title="TAD LAB RAG Chatbot", layout="wide")

# Constants
UPLOAD_DIR = "./uploaded_docs"
DB_URI = "./vector_database.db"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

class ChatBackend:
    def __init__(self, collection_name="StreamlitChat"):
        self.collection_name = collection_name
        self.embedding_model = OllamaEmbeddings(
            model="nomic-embed-text",
            base_url=OLLAMA_BASE_URL
        )
        self.milvus_client = None
        self.vector_store = None
        self._init_vector_db()

    def _init_vector_db(self):
        """Initialize connection to Milvus Lite"""
        self.vector_store = Milvus(
            embedding_function=self.embedding_model,
            connection_args={"uri": DB_URI},
            collection_name=self.collection_name,
            auto_id=True,
            drop_old=False 
        )

    def process_and_index(self, directory_path):
        """Process files using Docling and Index into Milvus"""
        text_lines = []
        
        # Docling Configuration
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        ocr_options = EasyOcrOptions(force_full_page_ocr=True)
        pipeline_options.ocr_options = ocr_options

        converter = DocumentConverter(
            allowed_formats=[
                    InputFormat.PDF,
                    InputFormat.IMAGE,
                    InputFormat.HTML,
                    InputFormat.PPTX,
                    InputFormat.CSV,
                    InputFormat.MD,
                    InputFormat.ASCIIDOC,
                    InputFormat.JSON_DOCLING,
                ],
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )

        files = glob(os.path.join(directory_path, "*"))
        if not files:
            return "No files found to process."

        progress_text = "Converting documents via Docling..."
        my_bar = st.progress(0, text=progress_text)

        processed_docs = []
        for i, file_path in enumerate(files):
            try:
                # Convert
                result = converter.convert(source=file_path)
                md_content = result.document.export_to_markdown()
                
                # Split
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, chunk_overlap=200, add_start_index=True
                )
                splits = text_splitter.create_documents([md_content])
                processed_docs.extend(splits)
            except Exception as e:
                st.error(f"Error processing {file_path}: {e}")
            
            my_bar.progress((i + 1) / len(files), text=f"Processed {file_path}")

        if processed_docs:
            st.info(f"Indexing {len(processed_docs)} chunks into Milvus...")
            self.vector_store.add_documents(processed_docs)
            my_bar.empty()
            return f"Successfully indexed {len(files)} files."
        
        return "No content could be extracted."

    def query(self, question, model_name):
        """Perform RAG Query"""
        llm = OllamaLLM(model=model_name, base_url=OLLAMA_BASE_URL)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Dynamic Prompt based on model type
        if "deepseek" in model_name.lower():
             template = """
            User: {question}
            Context: {context}
            
            Assistant: <think>
            """
        else:
            template = """
            Use the following pieces of information enclosed in <context> tags to provide an answer to the question.
            <context>
            {context}
            </context>
            <question>
            {question}
            </question>
            """

        prompt = PromptTemplate.from_template(template)

        def format_docs(docs):
            return "\n\n".join([d.page_content for d in docs])

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        return rag_chain.invoke(question)

    def clear_database(self):
        # Milvus Lite simplified drop
        try:
            # Re-initialize with drop_old=True to clear
            self.vector_store = Milvus(
                embedding_function=self.embedding_model,
                connection_args={"uri": DB_URI},
                collection_name=self.collection_name,
                drop_old=True
            )
            return "Database cleared."
        except Exception as e:
            return f"Error clearing DB: {e}"
        
def robust_rmtree(path, max_retries=5, delay_seconds=1):
    for i in range(max_retries):
        try:
            shutil.rmtree(path)
            return
        except OSError as e:
            if e.errno == 16: # Device or resource busy
                yield (f"Device busy, retrying in {delay_seconds} seconds... (Attempt {i+1}/{max_retries})")
                time.sleep(delay_seconds)
            else:
                raise
    yield (f"Failed to remove {path} after {max_retries} attempts.")
# --- Streamlit Logic ---

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # Model Selector
    model_options = [
        # 'hf.co/bartowski/microsoft_Phi-4-mini-instruct-GGUF:Q4_K_M',
        # 'deepseek-r1:7b',
        'gemma3:latest' # changed from gemma3 for stability, change back if needed
    ]
    selected_model = st.selectbox("Select Ollama Model", model_options, index=0)
    
    st.divider()
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Documents (PDF, MD, TXT)", 
        accept_multiple_files=True,
        type=['pdf', 'md', 'txt']
    )
    
    if st.button("Process & Index Documents"):
        if uploaded_files:
            # Clear upload dir first
            st.warning(robust_rmtree(UPLOAD_DIR))
            # if os.path.exists(UPLOAD_DIR):
            #     shutil.rmtree(UPLOAD_DIR)
            # os.makedirs(UPLOAD_DIR)
            
            # Save files
            for uploaded_file in uploaded_files:
                with open(os.path.join(UPLOAD_DIR, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
            
            # Initialize backend and process
            backend = ChatBackend()
            status = backend.process_and_index(UPLOAD_DIR)
            st.success(status)
        else:
            st.warning("Please upload files first.")

    st.divider()
    if st.button("Clear Vector Database"):
        backend = ChatBackend()
        msg = backend.clear_database()
        st.warning(msg)

# Main Chat Interface
st.title("🤖 TAD LAB RAG Chatbot")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        backend = ChatBackend()
        with st.spinner("Thinking..."):
            try:
                response = backend.query(prompt, selected_model)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error generating response: {e}")