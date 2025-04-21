"""
RAG chain implementations
"""

import logging
from typing import Dict, Any

from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

logger = logging.getLogger(__name__)

def create_rag_chain(chat_model, retriever):
    """
    Create a RAG chain combining the retriever and LLM.
    
    Args:
        chat_model: The LLM to use
        retriever: The retriever to use
    
    Returns:
        Configured RAG chain
    """
    logger.info("Setting up RAG chain")
    
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
    document_chain = create_stuff_documents_chain(chat_model, prompt)
    
    # Create the retrieval chain
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    logger.info("RAG chain initialized")
    return rag_chain

def query_rag_chain(rag_chain, question):
    """
    Process a question through the RAG pipeline.
    
    Args:
        rag_chain: The RAG chain
        question: User question
    
    Returns:
        RAG response dict containing answer and retrieved documents
    """
    logger.info(f"Processing query: {question}")
    
    try:
        # Execute the chain
        response = rag_chain.invoke({"input": question})
        return response
    except Exception as e:
        logger.error(f"Error executing RAG chain: {e}")
        # Return a fallback response with error information
        return {
            "answer": f"I encountered an error processing your query. Error: {str(e)}",
            "context": []
        }