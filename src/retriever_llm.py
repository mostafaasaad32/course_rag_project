import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (Make sure GROQ_API_KEY is in your .env file)
load_dotenv()

# Path to the vector database directory
VECTOR_DB_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'vector_db')

def ask_course_assistant(question):
    # 1. Load Embeddings and Database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)
    
    # 2. Setup the Retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    
    # 3. Retrieve the relevant documents for the specific question
    retrieved_docs = retriever.invoke(question)
    
    # Combine the text from the retrieved documents into one big string
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # 4. Setup the LLM (Updated to the active Llama 3.1 model on Groq)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    
    # 5. Setup the Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful teaching assistant. Answer the student's question based ONLY on the following context. If you don't know the answer based on the context, say 'I don't know'.\n\nContext:\n{context}"),
        ("human", "{question}")
    ])
    
    # 6. Build the modern pipeline (Prompt -> LLM -> Text Output)
    chain = prompt | llm | StrOutputParser()
    
    # 7. Generate the answer
    answer = chain.invoke({
        "context": context_text,
        "question": question
    })
    
    return answer, retrieved_docs

if __name__ == "__main__":
    print("Initializing Modern RAG System with Groq...\n")
    
    test_question = "What is a data pipeline?"
    print(f"Question: {test_question}\n")
    
    try:
        final_answer, sources = ask_course_assistant(test_question)
        
        print("--- Answer ---")
        print(final_answer)
        
        print("\n--- Sources Used ---")
        for doc in sources:
            source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page', 'N/A')
            print(f"- File: {source_file} (Page {page})")
            
    except Exception as e:
        print(f"\nError: {e}")