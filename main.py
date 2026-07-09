import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(title="AI PDF Chatbot via RAG")

# Temporary storage paths
UPLOAD_DIR = "./uploaded_pdfs"
FAISS_INDEX_DIR = "./faiss_index"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Update both models to their correct production versions
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", google_api_key=api_key)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, google_api_key=api_key)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Endpoint to upload a PDF, process it, and save to FAISS vector store."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    try:
        # 1. Save the uploaded file locally
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # 2. Load and parse the PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # 3. Split the text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        
        # 4. Create embeddings and save to FAISS index locally
        db = FAISS.from_documents(final_documents, embeddings)
        db.save_local(FAISS_INDEX_DIR)
        
        return {"message": f"Successfully processed {file.filename} and updated vector store."}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask")
async def ask_question(question: str = Form(...)):
    """Endpoint to query the processed PDF using RAG."""
    # Check if the vector database index exists
    if not os.path.exists(FAISS_INDEX_DIR):
        raise HTTPException(status_code=400, detail="No PDF has been uploaded or indexed yet. Please upload a PDF first.")
    
    try:
        # 1. Load the FAISS index
        db = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_kwargs={"k": 4})  # Retrieve top 4 chunks
        
        # 2. Define the system prompt blueprint
        system_prompt = (
            "You are an expert assistant. Answer the user's question using exclusively the provided context below. "
            "If you do not know the answer or if it's not in the context, say exactly "
            "'I cannot find the answer in the provided document.' Do not make things up.\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # 3. Create the RAG chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        # 4. Invoke the chain
        response = rag_chain.invoke({"input": question})
        
        return {
            "question": question,
            "answer": response["answer"]
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)