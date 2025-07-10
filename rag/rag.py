#importing all necessary libraries
import os
import pickle
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
import re
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS #stores vector embeddings and allows fast retrieval.
from langchain.docstore.document import Document
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv() # to load my open ai api key

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env!")
llm = OpenAI(openai_api_key=api_key, temperature=0.1, max_tokens=1000)



class RAGSystem:
    def __init__(self, papers_folder: str = "papers"): #sees inside paper folder to find the pdfs
        self.papers_folder = papers_folder
        self.documents = []
        self.vector_store = None #FAISS DB storing embeddings
        self.retriever = None
        self.qa_chain = None
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800, #breaks long documents into chunks of 800 characters, with 400-character overlap (to have conntext) to avoid cutting important content mid-way
            chunk_overlap=400,
            separators=["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
            length_function=len,
        )

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        text = "" #used to accumulate all the cleaned text from the pdf
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            cleaned_page = self.clean_page_text(page_text) #to remove visuals from the page
                            if cleaned_page.strip():
                                text += cleaned_page + "\n\n" #to remove line breaks
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                        continue
                        
            cleaned_text = self.clean_text(text)
            logger.info(f"Extracted {len(cleaned_text)} characters from {pdf_path}")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        text = re.sub(r'ﬁ', 'fi', text)
        text = re.sub(r'ﬂ', 'fl', text)

        text = re.sub(r'<\w+>', '', text)

        text = re.sub(r'^\s*([-_.=+|]{2,}\s*)+$', '', text, flags=re.MULTILINE)

        lines = text.splitlines()
        cleaned_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                cleaned_lines.append("")  # preserve paragraph spacing
                continue

            alphanum_count = len(re.findall(r'[a-zA-Z0-9]', line))
            total_count = len(line)
            is_jumbled_letters = re.search(r'^([a-zA-Z]\s){3,}$', line)

            if total_count > 0 and ((alphanum_count / total_count < 0.5 and total_count < 50) or is_jumbled_letters):
                continue

            cleaned_lines.append(line)

        # Join lines into paragraphs (fix line breaks inside sentences)
        merged_text = ""
        prev_line = ""
        for line in cleaned_lines:
            if line == "":
                merged_text += "\n\n"
                prev_line = ""
                continue

            if prev_line and not prev_line.endswith(('.', '!', '?', ':')):
                # likely a continuation of the same sentence
                merged_text += " " + line
            else:
                merged_text += line if merged_text.endswith('\n\n') or not merged_text else "\n" + line
            prev_line = line

        # Citation and formatting cleanup
        merged_text = re.sub(r'\[(\d+(?:,\s*\d+)*)\]', '', merged_text)
        merged_text = re.sub(r'\b[A-Z][a-z]+ et al\.?[\s\[\(]*(\d{4}[a-z]?)[\s\]\)]*', '', merged_text)
        merged_text = re.sub(r'\([A-Z][a-z]+,?\s*\d{4}[a-z]?\)', '', merged_text)
        merged_text = re.sub(r'\([A-Z][a-z]+\s+\d{4}[a-z]?\)', '', merged_text)
        merged_text = re.sub(r'\b[A-Z][a-z]+\s+\(\d{4}[a-z]?\)', '', merged_text)

        # Remove in-line references to figures, equations, tables
        merged_text = re.sub(r'\(\s*(Section|Sec\.?|Figure|Fig\.?|Table|Tab\.?|Equation|Eq\.?)\s+\d+(\.\d+)*\s*\)', '', merged_text)

        # Cleanup punctuation spacing
        merged_text = re.sub(r'\s+([,.!?;:])', r'\1', merged_text)
        merged_text = re.sub(r'[ \t]+', ' ', merged_text)
        merged_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', merged_text)

        return merged_text.strip()


    def clean_page_text(self, page_text: str) -> str:
        page_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', page_text)
        page_text = re.sub(r'[ \t]+', ' ', page_text)
        page_text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', page_text)
        page_text = re.sub(r'^\s*\d+\s*$', '', page_text, flags=re.MULTILINE)
        page_text = re.sub(r'^.*?(?:©|Copyright).*?$', '', page_text, flags=re.MULTILINE)
        page_text = re.sub(r'^.*?(?:Proceedings|Conference|Journal).*?$', '', page_text, flags=re.MULTILINE)
        page_text = re.sub(r'\b[A-Z][a-z]+\s+et\s+al\.?\s*\(\s*\d{4}[a-z]?\s*\)', '', page_text)
        page_text = re.sub(r'\(\s*Section\s+\d+(?:\.\d+)*\s*\)', '', page_text)
        return page_text.strip()


    def load_documents(self):
        papers_path = Path(self.papers_folder)
        if not papers_path.exists():
            logger.error(f"Papers folder '{self.papers_folder}' does not exist!")
            return
        
        pdf_files = list(papers_path.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            text = self.extract_text_from_pdf(str(pdf_file))
            
            if text.strip() and len(text) > 100:
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": pdf_file.name, 
                        "path": str(pdf_file),
                        "length": len(text)
                    }
                )
                self.documents.append(doc)
                logger.info(f"Added document: {pdf_file.name} ({len(text)} chars)")
        
        logger.info(f"Loaded {len(self.documents)} documents")

    def create_vector_store(self):
        if not self.documents:
            logger.error("No documents loaded!")
            return
        
        logger.info("Creating vector store with RecursiveCharacterTextSplitter...")
        
        all_chunks = self.text_splitter.split_documents(self.documents)
        
        logger.info(f"Total chunks created: {len(all_chunks)}")
        
        if all_chunks:
            self.vector_store = FAISS.from_documents(all_chunks, self.embeddings)
            logger.info(f"Created vector store with {len(all_chunks)} chunks")
        else:
            logger.error("No valid chunks created!")

    def save_vector_store(self):
        try:
            self.vector_store.save_local("vector_db")
            with open("vector_db/metadata.pkl", "wb") as f:
                pickle.dump({
                    "num_documents": len(self.documents),
                    "document_sources": [doc.metadata['source'] for doc in self.documents]
                }, f)
            logger.info("Vector store and metadata saved")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def load_vector_store(self):
        try:
            self.vector_store = FAISS.load_local("vector_db", self.embeddings, allow_dangerous_deserialization=True)
            logger.info("Vector store loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise e

    def setup_qa_system(self):
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5, "fetch_k": 15}
        )
        
        prompt_template = """You are an AI assistant specializing in academic research papers. Use the following context to provide detailed, accurate answers based on the research content.

Context: {context}

Question: {question}

Instructions:
- Provide a comprehensive answer based on the research context
- Use specific details, methods, and findings from the papers when possible
- If the context doesn't contain enough information, clearly state what information is missing
- Maintain academic rigor and accuracy
- Structure your answer clearly with main points and supporting details
- IMPORTANT:Don't cite or mention specific sections or figures from the paper

Answer:"""
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        llm = OpenAI(temperature=0.1, max_tokens=1000)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )

    def calculate_similarity(self, text1: str, text2: str) -> float:
        try:
            embedding1 = self.embeddings.embed_query(text1)
            embedding2 = self.embeddings.embed_query(text2)
            
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            return float(similarity)
        except:
            return 0.0

    def query(self, question: str) -> Dict[str, Any]:
        logger.info(f"Processing question: {question}")
        
        docs = self.retriever.invoke(question)
        
        retrieval_info = []
        for i, doc in enumerate(docs):
            similarity = self.calculate_similarity(question, doc.page_content)
            
            preview = doc.page_content[:400]
            if len(doc.page_content) > 400:
                preview += "..."
            
            retrieval_info.append({
                "chunk_id": i + 1,
                "source": doc.metadata.get("source", "Unknown"),
                "section": doc.metadata.get("section", "Unknown"),
                "chunk_type": doc.metadata.get("chunk_type", "standard"),
                "similarity_score": similarity,
                "preview": preview,
                "chunk_length": len(doc.page_content)
            })
            
            logger.info(f"Retrieved chunk {i+1} from {doc.metadata.get('source')} "
                       f"(section: {doc.metadata.get('section', 'Unknown')}, "
                       f"similarity: {similarity:.4f})")
        
        try:
            result = self.qa_chain.invoke({"query": question})
            answer = result["result"]
        except Exception as e:
            logger.error(f"QA chain failed: {e}")
            context = "\n\n".join([doc.page_content for doc in docs[:3]])
            answer = f"Based on the retrieved context:\n\n{context[:1000]}..."
        
        return {
            "question": question,
            "answer": answer,
            "retrieval_info": retrieval_info,
            "source_documents": docs
        }

    def initialize(self):
        try:
            self.load_vector_store()
            logger.info("Loaded existing vector store")
        except:
            logger.info("Creating new vector store...")
            self.load_documents()
            if not self.documents:
                raise Exception("No documents loaded! Please check your papers folder.")
            self.create_vector_store()
            if self.vector_store is None:
                raise Exception("Failed to create vector store!")
            self.save_vector_store()
        
        self.setup_qa_system()
        logger.info("RAG system ready!")

BENCHMARK_QUESTIONS = [
    "What is the main innovation introduced in the 'Attention is All You Need' paper?",
    "How does BERT differ from traditional left-to-right language models?",
    "Describe the few-shot learning capability of GPT-3 with an example.",
    "What is the loss function used in CLIP and why is it effective?",
    "What approach does LLaMA take to reduce computational cost during training?"
]

if __name__ == "__main__":
    rag = RAGSystem()
    rag.initialize()
    
    for question in BENCHMARK_QUESTIONS:
        print(f"\n{'='*50}")
        print(f"Question: {question}")
        print(f"{'='*50}")
        
        result = rag.query(question)
        for idx, (info, doc) in enumerate(zip(result['retrieval_info'], result['source_documents']), start=1):
            print(f"\n{'-'*40}")
            print(f"Chunk {idx} from {info['source']} "
                f"(section: {info['section']}, "
                f"sim={info['similarity_score']:.4f}):\n")
            print(doc.page_content)    # or info['preview'] if you only want the snippet
        print(f"\n{'='*40}\nAnswer:\n{result['answer']}\n")
