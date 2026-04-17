"""
RAG AI Tutor — Streamlit Frontend
Run with:
  cd frontend && streamlit run app.py
"""
import requests
import streamlit as st
from pathlib import Path
API_BASE = "http://localhost:8000"
st.set_page_config(
  page_title=" RAG AI Tutor",
  page_icon="",
  layout="wide",
  initial_sidebar_state="expanded",
)
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  .stApp {
    font-family: 'Inter', sans-serif;
  }
  .main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    color: white;
    box-shadow: 0 8px 32px rgba(102, 126, 234, 0.25);
  }
  .main-header h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.5px;
  }
  .main-header p {
    margin: 0.5rem 0 0;
    opacity: 0.9;
    font-size: 1.05rem;
    font-weight: 300;
  }
  .user-msg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem 1.25rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.95rem;
    box-shadow: 0 2px 12px rgba(102, 126, 234, 0.2);
  }
  .bot-msg {
    background: #f0f2f6;
    color: #1a1a2e;
    padding: 1.25rem 1.5rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.5rem 0;
    max-width: 85%;
    font-size: 0.95rem;
    line-height: 1.7;
    border: 1px solid #e0e3eb;
  }
  .image-card {
    background: white;
    border: 1px solid #e0e3eb;
    border-radius: 12px;
    padding: 1rem;
    margin-top: 0.75rem;
    max-width: 300px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .image-card img {
    border-radius: 8px;
    width: 100%;
  }
  .image-title {
    font-weight: 600;
    font-size: 0.9rem;
    margin-top: 0.5rem;
    color: #4a4a6a;
  }
  .status-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
  }
  .badge-success {
    background: #d4edda;
    color: #155724;
  }
  .badge-info {
    background: #d1ecf1;
    color: #0c5460;
  }
  .debug-chunk {
    background: #f8f9fa;
    border-left: 3px solid #667eea;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #4a4a6a;
  }
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
  }
  section[data-testid="stSidebar"] .stMarkdown {
    color: #e0e0e0;
  }
  #MainMenu {visibility: hidden;}
  footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
if "topic_id" not in st.session_state:
  st.session_state.topic_id = None
if "pdf_filename" not in st.session_state:
  st.session_state.pdf_filename = None
if "chat_history" not in st.session_state:
  st.session_state.chat_history = []
if "upload_info" not in st.session_state:
  st.session_state.upload_info = None
with st.sidebar:
  st.markdown("## Document Upload")
  st.markdown("---")
  uploaded_file = st.file_uploader(
    "Upload a PDF to get started",
    type=["pdf"],
    help="Upload a PDF document to create an AI tutor for its content.",
  )
  if uploaded_file is not None:
    if st.button(" Process PDF", use_container_width=True, type="primary"):
      with st.spinner(" Extracting text & building index…"):
        try:
          files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
          resp = requests.post(f"{API_BASE}/upload", files=files, timeout=120)
          resp.raise_for_status()
          data = resp.json()
          st.session_state.topic_id = data["topicId"]
          st.session_state.pdf_filename = uploaded_file.name
          st.session_state.upload_info = data
          st.session_state.chat_history = []
          st.success(" PDF processed — you can now ask questions")
        except requests.exceptions.ConnectionError:
          st.error(" Cannot reach the backend. Make sure it's running on port 8000.")
        except Exception:
          st.error(" Something went wrong, try again.")
  st.markdown("---")
  if st.session_state.topic_id:
    st.markdown(f'<span class="status-badge badge-success"> Connected</span>', unsafe_allow_html=True)
    st.markdown(f"**Topic ID:** `{st.session_state.topic_id}`")
    if st.session_state.pdf_filename:
      st.markdown(f"**File:** `{st.session_state.pdf_filename}`")
    if st.session_state.upload_info:
      st.markdown(f"**Chunks:** {st.session_state.upload_info.get('chunksCreated', 'N/A')}")
  else:
    st.markdown(f'<span class="status-badge badge-info"> No document loaded</span>', unsafe_allow_html=True)
  st.markdown("---")
  debug_mode = st.checkbox(" Debug mode", help="Show retrieved chunks alongside answers")
  if st.session_state.topic_id:
    if st.button(" Upload New PDF", use_container_width=True):
      st.session_state.topic_id = None
      st.session_state.pdf_filename = None
      st.session_state.upload_info = None
      st.session_state.chat_history = []
      st.rerun()
  if st.session_state.chat_history:
    if st.button(" Clear chat", use_container_width=True):
      st.session_state.chat_history = []
      st.rerun()
  st.markdown("---")
  st.markdown(
    "Built with using\n"
    "**FastAPI** · **Streamlit** · **FAISS** · **sentence-transformers**"
  )
st.markdown(
  '<div class="main-header">'
  "<h1> RAG AI Tutor</h1>"
  "<p>Upload a PDF and ask questions — get AI-powered answers with relevant diagrams</p>"
  "</div>",
  unsafe_allow_html=True,
)
if not st.session_state.topic_id:
  col1, col2, col3 = st.columns(3)
  with col1:
    st.markdown("### Step 1")
    st.markdown("Upload a PDF document using the sidebar")
  with col2:
    st.markdown("### Step 2")
    st.markdown("Ask questions about the document content")
  with col3:
    st.markdown("### Step 3")
    st.markdown("Get answers with relevant diagrams & images")
  st.stop()
for entry in st.session_state.chat_history:
  st.markdown(f'<div class="user-msg"> {entry["query"]}</div>', unsafe_allow_html=True)
  st.markdown(f'<div class="bot-msg"> {entry["answer"]}</div>', unsafe_allow_html=True)
  if entry.get("image"):
    img = entry["image"]
    image_url = f"{API_BASE}/{img['filename']}"
    st.markdown(
      f'<div class="response-image">\n'
      f'  <img src="{image_url}" alt="{img["title"]}" style="max-width:300px; margin-top:10px;" />\n'
      f'  <p style="font-size:12px; color:gray;">{img["title"]}</p>\n'
      f'</div>',
      unsafe_allow_html=True,
    )
  if entry.get("sources"):
    with st.expander(" Sources", expanded=False):
      for i, chunk in enumerate(entry["sources"], 1):
        st.markdown(
          f'<div class="debug-chunk"><strong>Chunk {i}:</strong> {chunk[:500]}…</div>',
          unsafe_allow_html=True,
        )
query = st.chat_input("Ask a question about your document…", disabled=not st.session_state.topic_id)
if query:
  st.markdown(f'<div class="user-msg"> {query}</div>', unsafe_allow_html=True)
  with st.spinner(" Thinking…"):
    try:
      payload = {
        "topicId": st.session_state.topic_id,
        "query": query,
      }
      params = {"debug": "true"} if debug_mode else {}
      resp = requests.post(f"{API_BASE}/chat", json=payload, params=params, timeout=60)
      resp.raise_for_status()
      data = resp.json()
      st.session_state.chat_history.append({
        "query": query,
        "answer": data["answer"],
        "image": data.get("image"),
        "sources": data.get("sources"),
      })
      st.rerun()
    except requests.exceptions.ConnectionError:
      st.error(" Cannot reach the backend. Make sure it's running on port 8000.")
    except Exception:
      st.error(" Something went wrong, try again.")