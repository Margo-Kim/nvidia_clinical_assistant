"""
LangChain RAG Implementation using NVIDIA AI Endpoints and Milvus Lite
"""

import os
import time
from typing import List, Dict, Any

# LangChain and related imports
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from pymilvus import connections, utility

# Set logging level to DEBUG for more information
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    def __init__(self, 
                 docs_folder: str = "documents",
                 milvus_db_path: str = "./milvus_lite.db",
                 embedding_base_url: str = "http://localhost:8089",
                 llm_base_url: str = "http://localhost:8088",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the RAG system.
        
        Args:
            docs_folder: Folder containing documents to process
            milvus_db_path: Path to the Milvus Lite database file
            embedding_base_url: NVIDIA embedding service URL
            llm_base_url: NVIDIA LLM service URL
            chunk_size: Document chunk size for splitting
            chunk_overlap: Overlap between chunks
        """
        self.docs_folder = docs_folder
        self.milvus_db_path = milvus_db_path
        self.embedding_base_url = embedding_base_url
        self.llm_base_url = llm_base_url
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.embedding_model = None
        self.chat_model = None
        self.vectorstore = None
        self.retriever = None
        self.rag_chain = None
        self.milvus_client = None
        
        # Connect to Milvus Lite
        self.milvus_client = MilvusClient(uri=self.milvus_db_path)
        logger.info(f"Connected to Milvus Lite database at {self.milvus_db_path}")

    def setup_embedding_model(self):
        """Set up NVIDIA embeddings model."""
        self.embedding_model = NVIDIAEmbeddings(
            model_name="nvidia/nv-embedqa-e5-v5",
            base_url=self.embedding_base_url
        )
        logger.info("Embedding model initialized")
        return self.embedding_model
    
    def setup_chat_model(self):
        """Set up NVIDIA chat completion model."""
        self.chat_model = ChatNVIDIA(
            model_name="meta/llama-3.1-nemotron-70b-instruct:latest",
            base_url=self.llm_base_url,
            max_tokens=32768,
            temperature=0.1,
            model_kwargs={
                "do_sample": True,
                "top_p": 0.9
            }
        )
        logger.info("Chat model initialized")
        return self.chat_model
    
    def load_and_split_documents(self) -> List[Document]:
        """
        Load documents from the specified folder and split them into chunks.
        
        Returns:
            List of document chunks
        """
        # Check if folder exists
        if not os.path.exists(self.docs_folder):
            logger.error(f"Documents folder {self.docs_folder} does not exist")
            raise FileNotFoundError(f"Folder {self.docs_folder} not found")
        
        # Load documents using DirectoryLoader with PyPDFLoader for PDF files
        loader = DirectoryLoader(
            self.docs_folder, 
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} documents from {self.docs_folder}")
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True
        )
        
        doc_chunks = text_splitter.split_documents(documents)
        logger.info(f"Split documents into {len(doc_chunks)} chunks")
        
        return doc_chunks
    
    def setup_vectorstore(self, documents: List[Document]) -> Milvus:
        """
        Set up Milvus Lite vector store with the provided documents.
        
        Args:
            documents: List of document chunks to embed and store
            
        Returns:
            Configured Milvus Lite vector store
        """
        # Ensure embedding model is initialized
        if self.embedding_model is None:
            self.setup_embedding_model()
        
        # Create collection name with timestamp to avoid conflicts
        collection_name = f"rag_collection_{int(time.time())}"
        
        # Get embedding dimension from model (NVIDIA embedqa-e5-v5 uses 1024 dimensions)
        embedding_dimension = 1024
        
        # First, create the collection in Milvus Lite
        if collection_name not in self.milvus_client.list_collections():
            self.milvus_client.create_collection(
                collection_name=collection_name,
                dimension=embedding_dimension
            )
            logger.info(f"Created Milvus Lite collection: {collection_name}")
        
        # Create a Milvus vector store with Milvus Lite
        self.vectorstore = Milvus.from_documents(
            documents=documents,
            embedding=self.embedding_model,
            collection_name=collection_name,
            connection_args={"uri": self.milvus_db_path},
            drop_old=True
        )
        
        logger.info(f"Created Milvus Lite vector store with collection: {collection_name}")
        
        # Set up retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Retrieve top 4 chunks
        )
        
        return self.vectorstore
    
    def setup_rag_chain(self):
        """
        Set up the RAG chain combining the retriever and LLM.
        
        Returns:
            Configured RAG chain
        """
        # Ensure chat model is initialized
        if self.chat_model is None:
            self.setup_chat_model()
            
        # Ensure retriever is set up
        if self.retriever is None:
            logger.error("Retriever not initialized. Please set up the vector store first.")
            raise ValueError("Retriever not initialized")
        
        # Create prompt template for the QA chain
        prompt = ChatPromptTemplate.from_template("""
        <context>
        {context}
        </context>
        
        Based on the provided context, please answer the following question accurately and concisely.
        If the answer cannot be found in the context, acknowledge that and provide general information if possible.
        
        Question: {input}
        """)
        
        # Create the document chain
        document_chain = create_stuff_documents_chain(self.chat_model, prompt)
        
        # Create the retrieval chain
        self.rag_chain = create_retrieval_chain(self.retriever, document_chain)
        
        logger.info("RAG chain initialized")
        return self.rag_chain
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a question through the RAG pipeline.
        
        Args:
            question: User question
            
        Returns:
            RAG response dict containing answer and retrieved documents
        """
        # Ensure RAG chain is initialized
        if self.rag_chain is None:
            logger.error("RAG chain not initialized")
            raise ValueError("RAG chain not initialized. Please set up the RAG chain first.")
        
        logger.info(f"Processing query: {question}")
        
        # Execute the chain
        response = self.rag_chain.invoke({"input": question})
        
        return response
    
    def initialize_full_pipeline(self):
        """
        Initialize the complete RAG pipeline at once.
        
        Returns:
            Self for method chaining
        """
        logger.info("Initializing full RAG pipeline...")
        
        # Setup embedding model
        self.setup_embedding_model()
        
        # Setup chat model
        self.setup_chat_model()
        
        # Load and split documents
        documents = self.load_and_split_documents()
        
        # Setup vector store and retriever
        self.setup_vectorstore(documents)
        
        # Setup RAG chain
        self.setup_rag_chain()
        
        logger.info("RAG pipeline fully initialized")
        return self
    
    def check_milvus_content(self, query="test", k=3):
        """
        Check what's in Milvus Lite by doing a simple similarity search.
        
        Args:
            query: Test query to search for
            k: Number of results to return
        """
        try:
            # List all collections
            print("\nCollections in Milvus Lite:")
            collections = self.milvus_client.list_collections()
            print(collections)
            
            if not collections:
                logger.warning("No collections found in Milvus Lite")
                return
            
            # Use the vectorstore for similarity search if available
            if self.vectorstore:
                docs = self.vectorstore.similarity_search(query=query, k=k)
                
                print(f"\nSample documents for query '{query}':")
                for i, doc in enumerate(docs, 1):
                    print(f"\nDocument {i}:")
                    print(f"Content: {doc.page_content[:200]}...")
                    print(f"Metadata: {doc.metadata}")
            else:
                logger.warning("Vector store not initialized")
                
        except Exception as e:
            logger.error(f"Error checking Milvus Lite content: {e}")


def main():
    """
    Main function to demonstrate the RAG pipeline.
    """
    # Initialize the RAG system with Milvus Lite
    rag_system = RAGSystem(
        docs_folder="documents",
        milvus_db_path="./milvus_lite.db"
    )
    
    # Initialize the full pipeline
    rag_system.initialize_full_pipeline()
    
    # Check what's in Milvus Lite
    rag_system.check_milvus_content()
    
    # Example queries
    example_queries = [
        "What are the main topics covered in the documents?",
        "Summarize the key points from the documents.",
        "What are the recommendations or conclusions in the documents?"
    ]
    
    for query in example_queries:
        print(f"\n\nQuery: {query}")
        result = rag_system.query(query)
        print(f"Answer: {result['answer']}")
        print(f"Retrieved {len(result['context'])} documents")


if __name__ == "__main__":
    main()