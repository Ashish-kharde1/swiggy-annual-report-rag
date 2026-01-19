import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Check for API Key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found. Please set it in the .env file.")

# Configuration
PDF_PATH = "data/swiggy_annual_report.pdf"
FAISS_INDEX_PATH = "faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def ingest_data():
    """
    Ingests PDF data, chunks it, generates embeddings, and saves to FAISS index.
    """
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    print("Loading PDF...")
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages.")

    print("Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks.")

    print("Generating embeddings and creating vector store...")
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)
    
    # Process in small batches
    batch_size = 10
    vectorstore = None
    
    import time
    import sys
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        print(f"Processing batch {i}/{len(texts)}...")
        sys.stdout.flush()
        try:
            if vectorstore is None:
                vectorstore = FAISS.from_documents(batch, embeddings)
            else:
                vectorstore.add_documents(batch)
        except Exception as e:
            print(f"Error on batch {i}: {e}")
            raise

    print(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
    if vectorstore:
        vectorstore.save_local(FAISS_INDEX_PATH)
    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_data()
