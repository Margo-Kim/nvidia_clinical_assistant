"""
Main script to run a RAG system with FiQA dataset using NVIDIA embeddings and Milvus Lite
"""

import logging
import argparse

# Import from our modules
from rag_implementation import RAGSystem
from document_processing import process_documents

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run RAG with FiQA dataset")
    
    parser.add_argument(
        "--milvus_db", 
        type=str, 
        default="./fiqa_milvus_lite.db",
        help="Path to Milvus Lite database file"
    )
    
    parser.add_argument(
        "--dataset_limit", 
        type=int, 
        default=10,
        help="Number of samples to load from FiQA dataset"
    )
    
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
      "--eval",
       action="store_true",
      help="Generate answers for FiQA ragas_eval split"
    )
    parser.add_argument(
        "--eval_limit",
        type=int,
        default=None,
        help="Cap number of FiQA eval questions (for a quick run)"
    )
    return parser.parse_args()

def run_example_queries(rag_system):
    """Run example queries on the RAG system"""
    example_queries = [
        "What are people saying about Samsung?",
        "Explain what the DFA is according to the documents.",
        "What are the SEC requirements for accredited investors?",
        "Discuss health FSA limitations mentioned in the corpus."
    ]
    
    print("\n=== Example Queries ===")
    
    for query in example_queries:
        print(f"\nQuery: {query}")
        try:
            # First, perform a direct search
            search_results = rag_system.search_documents(query, top_k=2)
            
            # Display search results
            print(f"Direct search results:")
            for i, doc in enumerate(search_results["documents"]):
                print(f"Result {i+1}:")
                print(f"Text: {doc.page_content[:150]}...")
                
            # Then show RAG results
            rag_result = rag_system.query(query)
            print(f"\nRAG Answer: {rag_result['answer']}")
            
            if "context" in rag_result:
                print(f"Retrieved {len(rag_result['context'])} documents")
        except Exception as e:
            print(f"Error processing query: {e}")

def run_interactive_mode(rag_system):
    """Run interactive query mode"""
    print("\n=== RAG Query System for FiQA Dataset ===")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your question about financial topics: ")
        if query.lower() in ['exit', 'quit', 'q']:
            break
        
        if not query.strip():
            continue
            
        try:
            # Perform direct search
            search_results = rag_system.search_documents(query, top_k=3)
            
            print("\nDirect search results:")
            for i, doc in enumerate(search_results["documents"]):
                print(f"Result {i+1}:")
                print(f"Text: {doc.page_content[:100]}...")
            
            # Get RAG answer
            rag_result = rag_system.query(query)
            
            print("\nRAG Answer:")
            print(rag_result['answer'])
            
            # Optionally show retrieved documents
            if "context" in rag_result and rag_result["context"]:
                show_docs = input("Show retrieved document chunks? (y/n): ")
                if show_docs.lower() == 'y':
                    for i, doc in enumerate(rag_result["context"], 1):
                        print(f"\nChunk {i}:")
                        print(f"Content: {doc.page_content[:200]}...")
                        print(f"Metadata: {doc.metadata}")
        except Exception as e:
            print(f"Error processing query: {e}")

def main():
    """Main function"""
    args = parse_args()
    
    # Test the NVIDIA embedding service first
    print("\n=== Testing NVIDIA Embedding Service ===")
    rag_system = RAGSystem(milvus_db_path=args.milvus_db)
    
    # Test embedding service
    test_result = rag_system.test_embedding_service()
    if test_result is None:
        print("Error: NVIDIA embedding service test failed. Please check configuration and service availability.")
        return
    else:
        print(f"Success! NVIDIA embedding service is working. Vector dimension: {len(test_result)}")
    
    # Process the FiQA dataset
    documents = process_documents(limit=args.dataset_limit)
    
    if not documents:
        print("Could not process FiQA dataset. Exiting.")
        return
    
    # Initialize the RAG system with the documents
    print("\n=== Initializing RAG System ===")
    rag_system.initialize_for_documents(documents)
    
    if args.eval:
        from eval import load_fiqa_eval, fill_answers

        eval_df = load_fiqa_eval(limit=args.eval_limit)
        print(f"\n=== Generating answers for {len(eval_df)} FiQA eval questions ===\n")
        fill_answers(rag_system, eval_df, out_path="fiqa_answers.csv")
        print("Finished. CSV saved as fiqa_answers.csv\n")
    
    # Run example queries
    run_example_queries(rag_system)
    
    # Run interactive mode if requested
    if args.interactive:
        run_interactive_mode(rag_system)

if __name__ == "__main__":
    main()
# """
# Main script to run a RAG system with FiQA dataset using NVIDIA embeddings and Milvus Lite
# """

# import os
# import logging
# from rag_implementation import RAGSystem
# from langchain.schema import Document

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def load_fiqa_dataset():
#     """
#     Load the FiQA dataset using the Hugging Face datasets library
#     """
#     print("Loading the FiQA dataset from Hugging Face...")
    
#     try:
#         from datasets import load_dataset
        
#         # Load the FiQA corpus dataset
#         dataset = load_dataset("explodinggradients/fiqa", "corpus")
#         print(dataset)
        
#         # For demonstration purposes, let's use a small subset
#         # Start with just 10 samples to test the system
#         sample_data = dataset["corpus"].select(range(10))
        
#         print(f"Loaded {len(sample_data)} samples from the FiQA corpus dataset")
#         return sample_data
    
#     except Exception as e:
#         print(f"Error loading FiQA dataset: {e}")
#         return None

# def convert_fiqa_to_documents(fiqa_data):
#     """
#     Convert FiQA data to LangChain Document format
#     """
#     documents = []
    
#     for item in fiqa_data:
#         # The FiQA corpus dataset has a field called 'doc' that contains the text
#         content = item["doc"]
        
#         # Create metadata
#         metadata = {
#             "source": "fiqa_corpus",
#             "index": len(documents)
#         }
        
#         # Create the document
#         doc = Document(page_content=content, metadata=metadata)
#         documents.append(doc)
    
#     print(f"Converted {len(documents)} FiQA items to Document format")
#     return documents

# def main():
#     """
#     Main function to run the RAG system
#     """
#     # Test the NVIDIA embedding service first
#     print("\n=== Testing NVIDIA Embedding Service ===")
#     rag = RAGSystem(
#         embedding_base_url="http://localhost:8089",
#         embedding_model_name="nvidia/nv-embedqa-e5-v5",
#         llm_base_url="http://localhost:8088",
#         llm_model_name="meta/llama-3.1-nemotron-70b-instruct:latest"
#     )
    
#     # Test embedding service
#     test_result = rag.test_embedding_service("Testing the NVIDIA embedding service.")
#     if test_result is None:
#         print("Error: NVIDIA embedding service test failed. Please check configuration and service availability.")
#         return
#     else:
#         print(f"Success! NVIDIA embedding service is working. Vector dimension: {len(test_result)}")
    
#     # Load the FiQA dataset
#     fiqa_data = load_fiqa_dataset()
    
#     if fiqa_data is None or len(fiqa_data) == 0:
#         print("Could not load FiQA dataset. Exiting.")
#         return
    
#     # Convert to LangChain documents
#     documents = convert_fiqa_to_documents(fiqa_data)
    
#     # Initialize the RAG system with NVIDIA embeddings and Milvus Lite
#     print("\n=== Initializing RAG System ===")
#     rag = RAGSystem(
#         milvus_db_path="./fiqa_milvus_lite.db",
#         embedding_base_url="http://localhost:8089",
#         embedding_model_name="nvidia/nv-embedqa-e5-v5",
#         llm_base_url="http://localhost:8088",
#         llm_model_name="meta/llama-3.1-nemotron-70b-instruct:latest"
#     )
    
#     try:
#         # Initialize the complete pipeline for the documents
#         rag.initialize_for_documents(documents)
        
#         # Example financial queries for the FiQA dataset
#         example_queries = [
#             "What are people saying about Samsung?",
#             "Explain what the DFA is according to the documents.",
#             "What are the SEC requirements for accredited investors?",
#             "Discuss health FSA limitations mentioned in the corpus."
#         ]
        
#         # Run example queries
#         print("\n=== Example Queries ===")
#         for query in example_queries:
#             print(f"\nQuery: {query}")
#             try:
#                 # First, perform a direct search
#                 search_results = rag.search_documents(query, top_k=2)
                
#                 # Display search results
#                 print(f"Direct search results:")
#                 for i, doc in enumerate(search_results["documents"]):
#                     print(f"Result {i+1}:")
#                     print(f"Text: {doc.page_content[:150]}...")
                    
#                 # Then show RAG results
#                 rag_result = rag.query(query)
#                 print(f"\nRAG Answer: {rag_result['answer']}")
                
#                 if "context" in rag_result:
#                     print(f"Retrieved {len(rag_result['context'])} documents")
#             except Exception as e:
#                 print(f"Error processing query: {e}")
        
#         # Interactive query loop
#         print("\n=== RAG Query System for FiQA Dataset ===")
#         print("Type 'exit' to quit")
        
#         while True:
#             query = input("\nEnter your question about financial topics: ")
#             if query.lower() in ['exit', 'quit', 'q']:
#                 break
            
#             if not query.strip():
#                 continue
                
#             try:
#                 # Perform direct search
#                 search_results = rag.search_documents(query, top_k=3)
                
#                 print("\nDirect search results:")
#                 for i, doc in enumerate(search_results["documents"]):
#                     print(f"Result {i+1}:")
#                     print(f"Text: {doc.page_content[:100]}...")
                
#                 # Get RAG answer
#                 rag_result = rag.query(query)
                
#                 print("\nRAG Answer:")
#                 print(rag_result['answer'])
                
#                 # Optionally show retrieved documents
#                 if "context" in rag_result and rag_result["context"]:
#                     show_docs = input("Show retrieved document chunks? (y/n): ")
#                     if show_docs.lower() == 'y':
#                         for i, doc in enumerate(rag_result["context"], 1):
#                             print(f"\nChunk {i}:")
#                             print(f"Content: {doc.page_content[:200]}...")
#                             print(f"Metadata: {doc.metadata}")
#             except Exception as e:
#                 print(f"Error processing query: {e}")
                
#     except Exception as e:
#         print(f"Error in main process: {e}")
        
# if __name__ == "__main__":
#     main()