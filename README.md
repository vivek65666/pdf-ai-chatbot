# 📄 AI PDF Chatbot (RAG System)

A fully functional Retrieval-Augmented Generation (RAG) chatbot application. This project enables users to upload any PDF document (like a resume, textbook, or manual) and have an interactive, context-aware chat session with it.

🔗 **Live Demo:** [Click here to try the app live!](https://pdf-ai-chatbot-nr6oxfrrgbvfjynbgsctlf.streamlit.app/)

Built with a fast, cloud-optimized structure and a clean, reactive frontend user interface.

## 🚀 Features

* **Dynamic PDF Upload & Processing:** Uses LangChain to dynamically parse and split documents into optimized text chunks.
* **Local Vector Store:** Utilizes FAISS to store mathematical embeddings natively, ensuring quick retrieval without complex external database infrastructure.
* **Modern LLM Integration:** Integrated with Google's advanced `gemini-2.5-flash` for high-performance context processing and text generation.
* **Interactive UI:** A conversational interface built entirely with Streamlit featuring clean chat history formatting.

## 🛠️ Tech Stack

* **Frontend & Processing:** Streamlit, Python
* **AI/RAG Framework:** LangChain, LangChain-Classic
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **LLM & Embeddings:** Google Gemini API, HugfulFace Transformers (`all-MiniLM-L6-v2`)

---

## ⚙️ Setup and Installation

### 1. Clone the repository
```bash
git clone [https://github.com/vivek65666/pdf-ai-chatbot.git](https://github.com/vivek65666/pdf-ai-chatbot.git)
cd pdf-ai-chatbot
