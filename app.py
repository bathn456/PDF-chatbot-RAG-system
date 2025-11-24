import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq

# --- 1. SETUP API KEY ---
# Load environment variables (for local testing)
load_dotenv()

# Get the key from the system (works on Cloud and Local)
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY not found! If you are running locally, check .env. If on cloud, check Secrets.")
    st.stop()

# --- 2. APP CONFIGURATION (BRANDING) ---
st.set_page_config(
    page_title="Batuhan Yılmaz | AI Architect", 
    layout="wide"
)

st.title("🤖 Chat with PDFs - Built by Batuhan Yılmaz")
st.caption("Powered by Groq Llama 3 & Open Source Embeddings")

# --- 3. HELPER FUNCTION ---
def parse_groq_stream(stream):
    for chunk in stream:
        if chunk.choices:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("Batuhan's PDF Engine")
    uploaded_file = st.file_uploader("Upload a document", type="pdf")
    
    if uploaded_file and "vector_store" not in st.session_state:
        st.write("Processing data...")
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)
        st.session_state.vector_store = vector_store
        st.success("Analysis Complete!")

# --- 5. CHAT LOGIC ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask Batuhan's AI a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "vector_store" in st.session_state:
        results = st.session_state.vector_store.similarity_search(prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in results])
    else:
        context = "No PDF uploaded."

    client = Groq(api_key=api_key)
    
    full_prompt = f"""
    You are a helpful AI assistant built by Batuhan Yılmaz.
    The following text is the content of a PDF document uploaded by the user.
    Answer the user's question based ONLY on this content.
    
    PDF Content:
    {context}
    
    Question: 
    {prompt}
    """
    
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": full_prompt}],
            stream=True
        )
        response = st.write_stream(parse_groq_stream(stream))
        
    st.session_state.messages.append({"role": "assistant", "content": response})
