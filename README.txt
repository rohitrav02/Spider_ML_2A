# RAG System for Research Paper Q&A

This project implements a question-answering system using Retrieval-Augmented Generation (RAG). It processes five machine learning research papers, stores them in a vector database, and uses LangChain to answer both custom and benchmark questions through a simple web interface.

---

# What This Project Does

- Parses and cleans research papers in PDF format
- Splits and embeds the documents using Hugging Face sentence transformers
- Stores the vector embeddings in a FAISS database
- Uses LangChain with OpenAI to answer questions based on retrieved content
- Displays results in a clean Streamlit app
- Includes benchmark questions with logs and retrieval scores

---

# Papers Used

- Attention Is All You Need
- BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- GPT-3: Language Models are Few-Shot Learners
- Contrastive Language-Image Pretraining with Knowledge Graphs (CLIP)
- LLaMA: Open and Efficient Foundation Language Models

---

# How It Works

1. PDF Parsing
   PDF files are parsed using `PyPDF2`, and the extracted text is cleaned to remove unnecessary formatting and symbols.

2. Chunking and Embedding
   Documents are split into overlapping text chunks using LangChain’s `RecursiveCharacterTextSplitter`. Chunks are embedded using the `all-MiniLM-L6-v2` model from `sentence-transformers`.

3. Vector Store
   FAISS is used to store and index the document embeddings. The vector database is persisted to disk to avoid reprocessing every time.

4. Question Answering
   A custom prompt template is used with LangChain’s `RetrievalQA` chain. Relevant document chunks are retrieved, and OpenAI’s LLM generates an answer.

5. Web Interface 
   A simple Streamlit interface allows users to ask their own questions or test the benchmark set. Retrieval logs, preview text, and similarity scores are displayed for transparency.

---


```

Install the required Python libraries:
```bash
pip install -r requirements.txt
bash
streamlit run app.py
```

---


```
.
app.py               # Streamlit frontend
rag.py               # Core RAG logic
papers/              # Folder to store input PDF files
vector_db/           # Saved FAISS vector store
.env                 # Contains your OpenAI API key
requirements.txt     # List of dependencies
```


# Requirements

- Python 3.8+
- LangChain
- OpenAI
- FAISS
- sentence-transformers
- PyPDF2
- Streamlit
- dotenv
- NumPy
- Plotly
- Pickle

---

# Benchmark Questions

These are evaluated internally and also available from the UI:

1. What is the main innovation introduced in the "Attention is All You Need" paper?
2. How does BERT differ from traditional left-to-right language models?
3. Describe the few-shot learning capability of GPT-3 with an example.
4. What is the loss function used in CLIP and why is it effective?
5. What approach does LLaMA take to reduce computational cost during training?

---

This project uses:
- LangChain for document processing and RAG pipeline
- FAISS for vector indexing
- Hugging Face sentence-transformers
- OpenAI API for question answering
- Streamlit for the frontend
