"""
Main script to run a RAG system with FiQA dataset using NVIDIA embeddings and Milvus Lite
"""

import os
import logging
from rag_implementation import RAGSystem
from langchain.schema import Document

# Set logging level
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_fiqa_dataset():
    """
    Load the FiQA dataset using the Hugging Face datasets library
    """
    print("Loading the FiQA dataset from Hugging Face...")
    
    try:
        from datasets import load_dataset
        
        # Load the FiQA corpus dataset
        dataset = load_dataset("explodinggradients/fiqa", "corpus")
        print(dataset)
        
        # For demonstration purposes, let's use a small subset
        # Start with just 10 samples to test the system
        sample_data = dataset["corpus"].select(range(10))
        
        print(f"Loaded {len(sample_data)} samples from the FiQA corpus dataset")
        return sample_data
    
    except Exception as e:
        print(f"Error loading FiQA dataset: {e}")
        return None

def convert_fiqa_to_documents(fiqa_data):
    """
    Convert FiQA data to LangChain Document format
    """
    documents = []
    
    for item in fiqa_data:
        # The FiQA corpus dataset has a field called 'doc' that contains the text
        content = item["doc"]
        
        # Create metadata
        metadata = {
            "source": "fiqa_corpus",
            "index": len(documents)
        }
        
        # Create the document
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)
    
    print(f"Converted {len(documents)} FiQA items to Document format")
    return documents

def main():
    """
    Main function to run the NVIDIA RAG with FiQA dataset and Milvus Lite
    """
    # Test the NVIDIA embedding service first
    print("\n=== Testing NVIDIA Embedding Service ===")
    rag = NVIDIARAGSystem(
        embedding_base_url="http://localhost:8089",
        embedding_model_name="nvidia/nv-embedqa-e5-v5",
        llm_base_url="http://localhost:8088",
        llm_model_name="meta/llama-3.1-nemotron-70b-instruct:latest"
    )
    
    # Test embedding service
    test_result = rag.test_embedding_service("Testing the NVIDIA embedding service.")
    if test_result is None:
        print("Error: NVIDIA embedding service test failed. Please check configuration and service availability.")
        return
    else:
        print(f"Success! NVIDIA embedding service is working. Vector dimension: {len(test_result)}")
    
    # Load the FiQA dataset
    fiqa_data = load_fiqa_dataset()
    
    if fiqa_data is None or len(fiqa_data) == 0:
        print("Could not load FiQA dataset. Exiting.")
        return
    
    # Convert to LangChain documents
    documents = convert_fiqa_to_documents(fiqa_data)
    
    # Initialize the RAG system with NVIDIA embeddings and Milvus Lite
    print("\n=== Initializing NVIDIA RAG System ===")
    rag = NVIDIARAGSystem(
        milvus_db_path="./fiqa_milvus_lite.db",
        embedding_base_url="http://localhost:8089",
        embedding_model_name="nvidia/nv-embedqa-e5-v5",
        llm_base_url="http://localhost:8088",
        llm_model_name="meta/llama-3.1-nemotron-70b-instruct:latest"
    )
    
    try:
        # Initialize the complete pipeline for the documents
        rag.initialize_for_documents(documents)
        
        # Example financial queries for the FiQA dataset
        example_queries = [
            "What are people saying about Samsung?",
            "Explain what the DFA is according to the documents.",
            "What are the SEC requirements for accredited investors?",
            "Discuss health FSA limitations mentioned in the corpus."
        ]
        
        # Run example queries
        print("\n=== Example Queries ===")
        for query in example_queries:
            print(f"\nQuery: {query}")
            try:
                # First, show direct search results
                search_results = rag.search_documents(query, top_k=2)
                print(f"Direct search results:")
                for i, result in enumerate(search_results[0]):
                    print(f"Result {i+1}:")
                    print(f"Text: {result['entity']['text'][:150]}...")
                    
                # Then show RAG results
                result = rag.query(query)
                print(f"\nRAG Answer: {result['answer']}")
                print(f"Retrieved {len(result['context'])} documents")
            except Exception as e:
                print(f"Error processing query: {e}")
        
        # Interactive query loop
        print("\n=== NVIDIA RAG Query System for FiQA Dataset ===")
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
                
    except Exception as e:
        print(f"Error in main process: {e}")
        
if __name__ == "__main__":
    main()