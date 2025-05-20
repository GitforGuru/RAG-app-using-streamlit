import streamlit as st
from typing import List
import PyPDF2
import docx
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import google.generativeai as genai

# Set your Gemini API key (replace with your actual key)
genai.configure(api_key="AIzaSyAS_aAklwH82GdSUZxB1BdgEgPk8bdF-SE")


st.title('RAG Chat with Uploaded Documents')
uploaded_files = st.sidebar.file_uploader(
    'Upload multiple documents (PDF, DOCX, TXT)',
    accept_multiple_files=True,
    type=['pdf', 'docx', 'txt']
)

st.header('Chat Interface')
user_input = st.text_input('Ask a question about your documents:')
submit = st.button("Submit")

def extract_text_from_pdf(file) -> str:
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + '\n'
    return text

def extract_text_from_docx(file) -> str:
    doc = docx.Document(file)
    text = '\n'.join([para.text for para in doc.paragraphs])
    return text

def extract_text_from_txt(file) -> str:
    return file.read().decode('utf-8')

def parse_documents(files: List) -> List[str]:
    texts = []
    for file in files:
        if file.type == 'application/pdf':
            text = extract_text_from_pdf(file)
        elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            text = extract_text_from_docx(file)
        elif file.type == 'text/plain':
            text = extract_text_from_txt(file)
        else:
            text = ''
        texts.append(text)
    return texts

# Chunk documents for better retrieval (optional, for large docs)
def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        print(i)
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

# Build FAISS index
def build_faiss_index(documents):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    all_chunks = []
    chunk_map = []
    for doc_id, doc in enumerate(documents):
        chunks = chunk_text(doc)
        all_chunks.extend(chunks)
        chunk_map.extend([doc_id]*len(chunks))
    embeddings = embedder.encode(all_chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index, all_chunks, embedder

# Retrieve relevant chunks
def retrieve(query, index, all_chunks, embedder, top_k=3):
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec), top_k)
    return [all_chunks[i] for i in I[0]]

def generate_response_gemini(query, retrieved_docs):
    context = '\n'.join(retrieved_docs)
    prompt = f"Answer the question based on the following context:\n{context}\nQuestion: {query}\nAnswer:"
    model = genai.GenerativeModel('models/gemini-1.5-flash-002')
    response = model.generate_content(prompt)
    return response.text

if uploaded_files:
    documents = parse_documents(uploaded_files)
    index, all_chunks, embedder = build_faiss_index(documents)

    if submit and user_input:
        retrieved_docs = retrieve(user_input, index, all_chunks, embedder)
        response = generate_response_gemini(user_input, retrieved_docs)
        st.write('**Response:**', response)