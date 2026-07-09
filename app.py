import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Set up page config
st.set_page_config(page_title="AI PDF Chatbot", layout="wide")

# Retrieve API Key from Streamlit Secrets
api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("Please add your GOOGLE_API_KEY to Streamlit Secrets.")
    st.stop()

st.sidebar.title("1. Upload Document")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])
process_button = st.sidebar.button("Process & Index PDF")

# Initialize session states
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process PDF locally on the cloud server
if uploaded_file and process_button:
    with st.sidebar.status("Processing PDF...", expanded=True) as status:
        # Save temporary file
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
        
        # Load and split
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
       # Create Embeddings and Vector Store
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=api_key)
        st.session_state.vector_store = FAISS.from_documents(splits, embeddings)
        
        # Clean up temp file
        os.remove(temp_path)
        status.update(label="PDF Indexed Successfully!", state="complete")

# UI Layout
st.title("2. Chat with your PDF")
st.caption("Upload a document on the left panel and ask questions instantly.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if user_question := st.chat_input("Ask something about the uploaded document..."):
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    if st.session_state.vector_store is None:
        with st.chat_message("assistant"):
            st.markdown("Please upload and process a PDF document first.")
    else:
        with st.chat_message("assistant"):
            # Setup RAG Chain
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)
            
            system_prompt = (
                "You are an expert assistant. Use the following pieces of retrieved context to answer "
                "the question. If you don't know the answer, say 'I cannot find the answer in the provided document.'\n\n"
                "Context:\n{context}"
            )
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{input}"),
            ])
            
            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            
            # Generate response
            response = rag_chain.invoke({"input": user_question})
            answer = response["answer"]
            
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
