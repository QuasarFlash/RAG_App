import uuid
import streamlit as st
import os
import shutil
import base64
import time
import concurrent.futures
from glob import glob
import ollama
from db_handler import ChatDatabase
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
IMAGES_DIR = "./chat_images"
# Ensure directories exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)


# ===========================
# Helper Functions
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
# Cached Resources (The Efficiency Core)
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
# Parallel Processing Backend (Text RAG)
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
# Vision Backend (Images)
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
# Main Streamlit Interface
# ===========================

if "messages" not in st.session_state: st.session_state.messages = []

# Initialize Backends
rag_backend = RAGBackend()
vision_backend = VisionBackend()
chatHistory_backend = ChatDatabase(db_path="./chat_history/chat_history.db")
# ===========================
# Session State Logic
# ===========================

# Ensure current session state
if "current_session_id" not in st.session_state:
    sessions = chatHistory_backend.get_all_sessions()
    if sessions:
        st.session_state.current_session_id = sessions[0]["id"]
    else:
        st.session_state.current_session_id = chatHistory_backend.create_session()

# ===========================
# Sidebar UI
# ===========================
with st.sidebar:
    st.header("🗂️ Manage Chats")
    
    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):
        new_id = chatHistory_backend.create_session()
        st.session_state.current_session_id = new_id
        st.rerun()

    # History Dropdown
    all_sessions = chatHistory_backend.get_all_sessions()
    session_titles = [s["title"] for s in all_sessions]
    session_ids = [s["id"] for s in all_sessions]
    # 2. Search Bar (NEW)
    search_query = st.text_input("🔍 Search History", placeholder="Type to find chats...")

    # 3. Logic: Get filtered or all sessions
    if search_query:
        # DB Search
        available_sessions = chatHistory_backend.search_sessions(search_query)
    else:
        # Default Load
        available_sessions = chatHistory_backend.get_all_sessions()

    # 4. Handle Empty Search Results
    if not available_sessions:
        st.caption("No matching chats found.")
        # If no results, we keep the current ID if possible, or don't allow switching
        session_titles = []
        session_ids = []
    else:
        session_titles = [s["title"] for s in available_sessions]
        session_ids = [s["id"] for s in available_sessions]

    # 5. History Dropdown (Smart Selection)
    if session_ids:
        # Try to keep the current selection active in the dropdown if it exists in the search results
        try:
            curr_idx = session_ids.index(st.session_state.current_session_id)
        except ValueError:
            curr_idx = 0 # Default to top result if current chat is not in search results

        selected_idx = st.selectbox(
            "History", 
            range(len(session_titles)), 
            format_func=lambda x: session_titles[x], 
            index=curr_idx, 
            key="history_select"
        )
        
        # Update State
        st.session_state.current_session_id = session_ids[selected_idx]
    # try:
    #     curr_idx = session_ids.index(st.session_state.current_session_id)
    # except ValueError:
    #     curr_idx = 0

    # if session_ids:
    #     selected_idx = st.selectbox("History", range(len(session_titles)), 
    #                                 format_func=lambda x: session_titles[x], 
    #                                 index=curr_idx, key="history_select")
    #     st.session_state.current_session_id = session_ids[selected_idx]
    
    # Delete Button
    if st.button("🗑️ Delete Current Chat"):
        if st.session_state.current_session_id:
            chatHistory_backend.delete_session(st.session_state.current_session_id)
            # Reset to the most recent chat available
            remaining = chatHistory_backend.get_all_sessions()
            if remaining:
                st.session_state.current_session_id = remaining[0]["id"]
            else:
                st.session_state.current_session_id = chatHistory_backend.create_session()
            st.rerun()

    st.divider()
    st.header("⚙️ Settings")
    chat_mode = st.radio("Mode", ["🤖 Text RAG", "👁️ Vision Chat"])
    
    if chat_mode == "🤖 Text RAG":
        text_model = st.selectbox("Model", ['gemma3:latest'], index=0)
        use_ocr = st.checkbox("Enable OCR", value=False)
        uploaded_files = st.file_uploader("Docs", accept_multiple_files=True, type=['pdf','md','txt','csv','png','jpg','jpeg'])
        status_area = st.empty()
        
        if st.button("Index Docs"):
            if uploaded_files:
                robust_rmtree(UPLOAD_DIR)
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                for uf in uploaded_files:
                    with open(os.path.join(UPLOAD_DIR, uf.name), "wb") as f: f.write(uf.getbuffer())
                st.success(rag_backend.process_and_index_parallel(UPLOAD_DIR, status_area, use_ocr))
        
        if st.button("Clear Vector DB"): st.warning(rag_backend.clear_database())

    elif chat_mode == "👁️ Vision Chat":
        vision_model = st.selectbox("Vision Model", ['gemma3:latest'], index=0)
        uploaded_image = st.file_uploader("Image", type=['png','jpg','jpeg'], key="v_upload")
        if uploaded_image: st.image(uploaded_image, width=200)

# ===========================
# Main Chat Loop
# ===========================

# Load messages for the CURRENT session from SQLite
current_messages = chatHistory_backend.get_messages(st.session_state.current_session_id)

st.title("⚡ Turbo Multimodal RAG")

for msg in current_messages:
    with st.chat_message(msg["role"]):
        # Load image from DISK if exists
        if msg["image_path"] and os.path.exists(msg["image_path"]):
            st.image(msg["image_path"], width=300)
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    
    # Vision Logic
    if chat_mode == "👁️ Vision Chat" and uploaded_image:
        # 1. Save Image to Disk
        img_filename = f"{uuid.uuid4()}.png"
        img_path = os.path.join(IMAGES_DIR, img_filename)
        with open(img_path, "wb") as f: f.write(uploaded_image.getbuffer())
        
        # 2. Save User Msg to DB
        chatHistory_backend.add_message(st.session_state.current_session_id, "user", prompt, img_path)
        
        # 3. Show immediately (optimistic UI)
        with st.chat_message("user"):
            st.image(uploaded_image, width=250)
            st.markdown(prompt)

        # 4. Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                try:
                    resp = vision_backend.query_vision(prompt, uploaded_image, vision_model)
                    st.markdown(resp)
                    chatHistory_backend.add_message(st.session_state.current_session_id, "assistant", resp)
                    st.rerun()
                except Exception as e: st.error(f"Error: {e}")

    # Text Logic
    elif chat_mode == "🤖 Text RAG":
        chatHistory_backend.add_message(st.session_state.current_session_id, "user", prompt)
        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = rag_backend.query_text_rag(prompt, text_model)
                    clean_resp = resp.split("</think>")[-1].strip() if "</think>" in resp else resp
                    st.markdown(clean_resp)
                    chatHistory_backend.add_message(st.session_state.current_session_id, "assistant", clean_resp)
                    st.rerun()
                except Exception as e: st.error(f"Error: {e}")
    
    else:
        st.warning("Please upload an image for Vision Chat.")