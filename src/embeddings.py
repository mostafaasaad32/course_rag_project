import os
# Update: Import the tool from its new independent library
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Import the functions we wrote in the previous file
from ingestion import load_documents, split_documents

# Path to the vector database directory
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_db')

def create_vector_db():
    print("1. Reading and splitting documents...")
    docs = load_documents()
    chunks = split_documents(docs)
    
    print("2. Initializing HuggingFace Embeddings model (free and local)...")
    # Load a free, lightweight, and fast model that runs locally
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"3. Converting {len(chunks)} chunks into vectors and saving them to ChromaDB...")
    print("This process might take some time as it runs on your local machine...")
    
    # Create the vector database and save it to disk
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    
    print(f"\nSuccessfully finished! The database has been saved in the path: {VECTOR_DB_DIR}")
    return vector_db

if __name__ == "__main__":
    create_vector_db()