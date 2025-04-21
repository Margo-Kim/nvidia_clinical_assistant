"""
RAG chain implementations with improved error handling
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

    Instructions for answering:
    1. Answer using only the information found in the context. 
    2. Do not add any formatting or explanations. (NO ASTERISKS, NO BOLD)
    3. Keep the answer semantically and structurally close to the content in the documents.
    4. Focus on the financial information and advice relevant to the question.
    5. If the information isn't available in the context, state this clearly.    
    
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
        
        # Log success
        if 'context' in response and response['context']:
            doc_count = len(response['context'])
            logger.info(f"Successfully retrieved {doc_count} documents")
            
            # Debug log to check document structure
            if logger.isEnabledFor(logging.DEBUG):
                for i, doc in enumerate(response['context']):
                    logger.debug(f"Document {i+1} keys: {doc.__dict__.keys()}")
                    logger.debug(f"Document {i+1} content: {getattr(doc, 'page_content', 'NO CONTENT')[:100]}...")
        
        return response
    except Exception as e:
        logger.error(f"Error executing RAG chain: {e}")
        
        # Try to get more detailed error info
        import traceback
        logger.error(f"Detailed error: {traceback.format_exc()}")
        
        # Return a fallback response with error information
        return {
            "answer": f"I encountered an error processing your query. Error: {str(e)}",
            "context": []
        }