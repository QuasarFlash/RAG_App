import streamlit as st
import os
import shutil
import base64
import time
import concurrent.futures
from glob import glob
import ollama

# LangChain & Milvus
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Docling - Optimized Imports
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
# Backend optimization (make sure to install docling[pypdfium2] or standard docling)
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

# Page Config
st.set_page_config(page_title="TAD LAB Turbo RAG", layout="wide", page_icon="⚡")

# Constants
UPLOAD_DIR = "./uploaded_docs"
DB_URI = "./vector_database/vector_database.db"
OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ===========================
# 1. Helper Functions
# ===========================
def robust_rmtree(path, max_retries=5, delay=1):
    """Robustly remove a directory, handling Windows file locks."""
    for i in range(max_retries):
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            return True
        except OSError:
            time.sleep(delay)
    return False

# ===========================
# 2. Cached Resources (The Efficiency Core)
# ===========================
@st.cache_resource(show_spinner="Connecting to Knowledge Base...")
def get_vector_store():
    """
    Cache the database connection. This prevents reconnecting
    to Milvus on every user interaction.
    """
    embedding_model = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=OLLAMA_HOST,
    )
    
    return Milvus(
        embedding_function=embedding_model,
        connection_args={"uri": DB_URI},
        collection_name="StreamlitTurboChat",
        auto_id=True,
        drop_old=False
    )

@st.cache_resource(show_spinner="Loading Document Converter...")
def get_docling_converter(enable_ocr=False):
    """
    Cache Docling models. Toggling OCR off provides ~10x speedup for digital PDFs.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = enable_ocr
    pipeline_options.do_table_structure = enable_ocr 
    
    if enable_ocr:
        ocr_options = EasyOcrOptions(force_full_page_ocr=True)
        pipeline_options.ocr_options = ocr_options

    return DocumentConverter(
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
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend 
            )
        }
    )

# ===========================
# 3. Parallel Processing Backend (Text RAG)
# ===========================
class RAGBackend:
    def __init__(self):
        self.vector_store = get_vector_store()

    def process_single_file(self, file_path, enable_ocr):
        """Helper function to be run in parallel threads"""
        try:
            # Re-fetch converter (cached) to ensure thread access
            converter = get_docling_converter(enable_ocr)
            result = converter.convert(source=file_path)
            md_content = result.document.export_to_markdown()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.create_documents(
                [md_content], 
                metadatas=[{"source": os.path.basename(file_path)}]
            )
            return splits
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return []

    def process_and_index_parallel(self, directory_path, status_container, enable_ocr=False):
        """
        OPTIMIZATION: Uses ThreadPoolExecutor to process files concurrently.
        """
        files = glob(os.path.join(directory_path, "*"))
        if not files: return "No files found."

        all_splits = []
        total_files = len(files)
        progress_bar = status_container.progress(0)
        status_text = status_container.empty()

        # Adjust max_workers based on your CPU cores
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_to_file = {
                executor.submit(self.process_single_file, f, enable_ocr): f 
                for f in files
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_file):
                file_p = future_to_file[future]
                try:
                    splits = future.result()
                    all_splits.extend(splits)
                    completed += 1
                    # Update UI
                    progress = completed / total_files
                    progress_bar.progress(progress)
                    status_text.write(f"Processed {completed}/{total_files}: {os.path.basename(file_p)}")
                except Exception as e:
                    st.error(f"Failed to process {file_p}: {e}")

        if all_splits:
            status_text.write(f"Indexing {len(all_splits)} chunks into Vector DB...")
            self.vector_store.add_documents(all_splits)
            progress_bar.empty()
            status_text.empty()
            return f"Success: Indexed {len(all_splits)} chunks from {total_files} files."
        return "Failed: No valid content extracted."

    def query_text_rag(self, question, model_name):
        llm = OllamaLLM(
            model=model_name, 
            base_url=OLLAMA_HOST,
            keep_alive="-1m" # Keep model loaded in memory
        )
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})
        
        template = """Answer the question based ONLY on the following context:
        {context}
        
        Question: {question}
        """
        prompt = PromptTemplate.from_template(template)
        
        def format_docs(docs):
             return "\n\n".join([f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs])

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )
        return rag_chain.invoke(question)
    
    def clear_database(self):
        try:
            # Re-init with drop_old=True to clear
            embeddings = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=OLLAMA_HOST,
            )
            Milvus(
                embedding_function=embeddings,
                connection_args={"uri": DB_URI},
                collection_name="StreamlitTurboChat",
                drop_old=True
            )
            return "Database cleared."
        except Exception as e:
            return f"Error clearing DB: {e}"

# ===========================
# 4. Vision Backend (Images)
# ===========================
class VisionBackend:
    def __init__(self):
        self.client = ollama.Client(host=OLLAMA_HOST)

    def query_vision(self, prompt, image_file, model_name="gemma3:latest"):
        # Convert uploaded file to base64
        b64_img = base64.b64encode(image_file.getvalue()).decode('utf-8')
        
        # Use Chat API (Required for instruction-tuned vision models like Gemma/Llama 3.2)
        response = self.client.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [b64_img]
            }],
            keep_alive="1h"
        )
        return response['message']['content']

# ===========================
# 5. Main Streamlit Interface
# ===========================

if "messages" not in st.session_state: st.session_state.messages = []

# Initialize Backends
rag_backend = RAGBackend()
vision_backend = VisionBackend()

with st.sidebar:
    st.header("⚡ Turbo Configuration")
    
    # 1. Mode Selector (New!)
    chat_mode = st.radio("Select Capability", ["🤖 Text RAG (PDFs)", "👁️ Vision Chat (Images)"])
    st.divider()
    
    # --- Mode A: Text RAG ---
    if chat_mode == "🤖 Text RAG (PDFs)":
        text_model = st.selectbox("Text Model", ['gemma3:latest'], index=0)
        
        st.subheader("Knowledge Base")
        # OCR Toggle (Efficiency Boost)
        use_ocr = st.checkbox("Enable OCR (Slower)", value=False, help="Uncheck for digital PDFs to speed up processing by 10x")
        
        uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['pdf','md','txt'])
        status_area = st.empty()
        
        if st.button("Process & Index"):
            if uploaded_files:
                # Robust clean
                robust_rmtree(UPLOAD_DIR)
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                
                # Save
                for uf in uploaded_files:
                    with open(os.path.join(UPLOAD_DIR, uf.name), "wb") as f: f.write(uf.getbuffer())
                
                # Process Parallel
                msg = rag_backend.process_and_index_parallel(UPLOAD_DIR, status_area, enable_ocr=use_ocr)
                st.success(msg)
        
        if st.button("Clear Database"):
            st.warning(rag_backend.clear_database())

    # --- Mode B: Vision Chat ---
    elif chat_mode == "👁️ Vision Chat (Images)":
        vision_model = st.selectbox("Vision Model", ['gemma3:latest'], index=0)
        
        st.subheader("Image Input")
        uploaded_image = st.file_uploader("Upload Image", type=['png','jpg','jpeg'], key="vision_up")
        if uploaded_image:
            st.image(uploaded_image, caption="Analysis Target", use_container_width=True)

# Main Chat Loop
st.title("🤖 TAD LAB Turbo RAG")

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # Show image if present in history
        if "image_data" in msg:
            st.image(msg["image_data"], width=250)
        st.markdown(msg["content"])

# Handle Input
if prompt := st.chat_input("Ask a question..."):
    
    # Logic for Vision Mode
    if chat_mode == "👁️ Vision Chat (Images)" and uploaded_image:
        # User Bubble
        st.session_state.messages.append({"role": "user", "content": prompt, "image_data": uploaded_image.getvalue()})
        with st.chat_message("user"):
            st.image(uploaded_image, width=250)
            st.markdown(prompt)
        
        # Assistant Bubble
        with st.chat_message("assistant"):
            with st.spinner("Analyzing image..."):
                try:
                    resp = vision_backend.query_vision(prompt, uploaded_image, vision_model)
                    st.markdown(resp)
                    st.session_state.messages.append({"role": "assistant", "content": resp})
                except Exception as e:
                    st.error(f"Vision Error: {e}")

    # Logic for Text RAG Mode
    elif chat_mode == "🤖 Text RAG (PDFs)":
        # User Bubble
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Assistant Bubble
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = rag_backend.query_text_rag(prompt, text_model)
                    # Clean <think> tags if using Deepseek
                    clean_resp = resp.split("</think>")[-1].strip() if "</think>" in resp else resp
                    st.markdown(clean_resp)
                    st.session_state.messages.append({"role": "assistant", "content": clean_resp})
                except Exception as e:
                    st.error(f"RAG Error: {e}")
    
    else:
        st.warning("Please upload an image for Vision Chat, or switch to Text RAG.")