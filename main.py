"""
Example of using the FiQA dataset with our RAG implementation
"""

import os
import requests
import json
from rag_implementation import RAGSystem
from langchain.schema import Document

def download_fiqa_sample():
    """
    Download a sample of the FiQA dataset from HuggingFace
    """
    print("Downloading a sample of the FiQA dataset...")
    
    # Create a directory for the dataset
    os.makedirs("fiqa_data", exist_ok=True)
    
    # We'll use a small sample of the corpus for demonstration
    url = "https://huggingface.co/datasets/explodinggradients/fiqa/raw/main/corpus/part-000.json"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Each line is a JSON object, so we'll parse them line by line
        data = []
        for line in response.text.strip().split("\n"):
            data.append(json.loads(line))
            
            # For demo purposes, let's limit to first 50 entries
            if len(data) >= 50:
                break
        
        # Save the data to a file
        with open("fiqa_data/sample.json", "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        print(f"Downloaded {len(data)} FiQA dataset samples to fiqa_data/sample.json")
        return data
    
    except Exception as e:
        print(f"Error downloading FiQA dataset: {e}")
        return []

def convert_fiqa_to_documents(fiqa_data):
    """
    Convert FiQA data to LangChain Document format
    """
    documents = []
    
    for item in fiqa_data:
        # Create a document from the text content
        content = item.get("body", "")
        
        # Extract metadata
        metadata = {
            "id": item.get("id", ""),
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "topics": item.get("topics", []),
            "tags": item.get("tags", [])
        }
        
        # Create the document
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    print(f"Converted {len(documents)} FiQA items to Document format")
    return documents

def main():
    """
    Main function to demonstrate RAG with the FiQA dataset
    """
    # Download the FiQA dataset sample
    fiqa_data = download_fiqa_sample()
    
    if not fiqa_data:
        print("Could not download FiQA dataset. Exiting.")
        return
    
    # Convert to Documents
    documents = convert_fiqa_to_documents(fiqa_data)
    
    # Initialize the RAG system with Milvus Lite
    print("Initializing RAG system with Milvus Lite...")
    rag = RAGSystem(milvus_db_path="./fiqa_milvus_lite.db")
    
    # Set up components
    print("Setting up RAG components...")
    rag.setup_embedding_model()
    rag.setup_chat_model()
    
    # Set up the vector store
    print("Setting up vector store with FiQA documents...")
    rag.setup_vectorstore(documents)
    
    # Set up the RAG chain
    print("Setting up RAG chain...")
    rag.setup_rag_chain()
    
    # Example financial queries for the FiQA dataset
    example_queries = [
        "What are people saying about Samsung?",
        "Tell me about investment opportunities mentioned in the documents.",
        "What financial regulations are discussed in the corpus?",
        "Summarize opinions about tech companies."
    ]
    
    # Run example queries
    print("\n=== Example Queries ===")
    for query in example_queries:
        print(f"\nQuery: {query}")
        try:
            result = rag.query(query)
            print(f"Answer: {result['answer']}")
            print(f"Retrieved {len(result['context'])} documents")
        except Exception as e:
            print(f"Error processing query: {e}")
    
    # Interactive query loop
    print("\n=== RAG Query System for FiQA Dataset ===")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your question about financial topics: ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query.strip():
            continue
            
        try:
            result = rag.query(query)
            print("\nAnswer:")
            print(result['answer'])
            
            # Optionally show retrieved documents
            show_docs = input("Show retrieved document chunks? (y/n): ")
            if show_docs.lower() == 'y':
                for i, doc in enumerate(result['context'], 1):
                    print(f"\nChunk {i}:")
                    print(f"Content: {doc.page_content[:200]}...")
                    print(f"Metadata: {doc.metadata}")
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    main()