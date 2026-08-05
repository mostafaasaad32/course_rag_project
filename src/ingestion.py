import os
import glob
from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    TextLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw_courses')

def load_documents():
    """
    Reads all files from the directory and selects the appropriate loader based on the file extension.
    """
    documents = []
    
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} not found.")
        return documents

    files = glob.glob(os.path.join(DATA_DIR, "*"))
    
    for file_path in files:
        extension = file_path.split('.')[-1].lower()
        loader = None
        
        # Select the appropriate loader
        if extension == 'pdf':
            loader = PyPDFLoader(file_path)
        elif extension == 'docx':
            loader = Docx2txtLoader(file_path)
        elif extension == 'csv':
            loader = CSVLoader(file_path)
        elif extension == 'txt':
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            print(f"Skipping file {file_path} - Unsupported extension.")
            continue
            
        if loader:
            print(f"Loading: {file_path}...")
            # Load the file and add it to the documents list
            documents.extend(loader.load())
            
    return documents

def split_documents(documents):
    """
    Split documents into smaller chunks to facilitate searching (Chunking).
    """
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Number of characters in each chunk
        chunk_overlap=200, # Number of overlapping characters to maintain context
        separators=["\n\n", "\n", ".", " ", ""] # Priority order of separators
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

# This section is for testing the file independently
if __name__ == "__main__":
    print("Starting ingestion process...")
    docs = load_documents()
    print(f"Loaded {len(docs)} pages/documents.")
    
    if docs:
        chunks = split_documents(docs)
        print(f"Split documents into {len(chunks)} chunks.")
        print("\nExample of the first chunk's data:")
        print("-" * 30)
        print(f"Content:\n{chunks[0].page_content[:200]}...")
        print(f"\nSource (Metadata):\n{chunks[0].metadata}")