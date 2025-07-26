from dotenv import load_dotenv
import os

load_dotenv()


import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import pinecone
from groq import Groq
import os
import time


st.set_page_config(
    page_title="Loan Approval Assistant", 
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #f7f7f8;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* User messages */
    [data-testid="stChatMessageContainer"]:has(.user-message) {
        background-color: #e3f2fd;
    }
    
    /* Headers */
    h1 {
        color: #1976d2;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        padding-bottom: 1rem;
        border-bottom: 3px solid #e3f2fd;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f5f7fa;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1976d2;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


col1, col2 = st.columns([3, 1])
with col1:
    st.title("🏦 Loan Approval Assistant")
    st.markdown("**Powered by RAG Technology** • Get instant answers about loan approvals")
with col2:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)


@st.cache_resource
def init_connections():
    with st.spinner("🔌 Connecting to services..."):
        pc = pinecone.init(api_key=st.secrets["PINECONE_API_KEY"])
        index = pc.Index("loan-rag-index")
        groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return index, groq_client, embedding_model


try:
    index, groq_client, embedding_model = init_connections()
    connection_status = True
except Exception as e:
    st.error(f"❌ Connection Error: {str(e)}")
    connection_status = False

# Sidebar
with st.sidebar:
    st.markdown("## 📊 About This Assistant")
    st.info(
        "This AI assistant uses Retrieval-Augmented Generation (RAG) to answer "
        "questions about loan approvals based on real application data."
    )
    
    st.markdown("### 🎯 Example Questions")
    example_questions = [
        "What's the approval rate for married people?",
        "How does income affect loan approval?",
        "What role does credit history play?",
        "Show me approved loans for self-employed",
        "What's the average loan amount approved?"
    ]
    
    for i, question in enumerate(example_questions):
        if st.button(f"📌 {question}", key=f"example_{i}"):
            st.session_state.sample_question = question
    
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    if connection_status:
        st.success("✅ All systems operational")
    else:
        st.error("❌ Connection failed")
    
    st.markdown("---")
    st.markdown("### 📈 Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", "614", "📄")
    with col2:
        st.metric("Avg Response", "< 2s", "⚡")


def query_rag(question, top_k=5):
    question_embedding = embedding_model.encode([question])
    search_results = index.query(
        vector=question_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    contexts = []
    sources = []
    for match in search_results['matches']:
        contexts.append(match['metadata']['text'])
        sources.append(match['id'])
    return contexts, sources

def get_answer(question):
    if not connection_status:
        return "❌ Cannot process queries - connection error. Please check API keys.", []
    
    contexts, sources = query_rag(question)
    
    if not contexts:
        return "🔍 I couldn't find relevant loan data for your question. Try asking about approvals, income requirements, or credit history.", []
    
    context_text = "\n\n".join(contexts)
    prompt = f"""You are a helpful loan advisor. Answer based ONLY on this loan data:

{context_text}

Question: {question}

Provide a clear, concise answer with specific examples from the data when possible."""
    
    try:
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a professional loan advisor. Be helpful and specific."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content, sources
    except Exception as e:
        return f"❌ Error generating response: {str(e)}", []

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Welcome message
    welcome_msg = """👋 Welcome! I'm your AI Loan Approval Assistant. 
    
I can help you understand loan approval patterns, requirements, and statistics based on our historical data. 
    
Feel free to ask me questions like:
- What factors influence loan approval?
- What's the typical income for approved loans?
- How important is credit history?

How can I assist you today?"""
    st.session_state.messages.append({"role": "assistant", "content": welcome_msg})


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📎 View sources"):
                st.write(f"Based on {len(message['sources'])} relevant records")

if "sample_question" in st.session_state:
    question = st.session_state.sample_question
    del st.session_state.sample_question
    
    
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching loan database..."):
            answer, sources = get_answer(question)
            st.write(answer)
            if sources:
                with st.expander("📎 View sources"):
                    st.write(f"Based on {len(sources)} relevant records")
    
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    st.rerun()


if question := st.chat_input("Ask me about loan approvals...", disabled=not connection_status):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching loan database..."):
            answer, sources = get_answer(question)
            st.write(answer)
            if sources:
                with st.expander("📎 View sources"):
                    st.write(f"Based on {len(sources)} relevant records")
    
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

st.markdown("---")
st.markdown(
    "<center><p style='color: #666; font-size: 0.9rem;'>Built with Streamlit • RAG powered by Pinecone & Groq</p></center>", 
    unsafe_allow_html=True
)
