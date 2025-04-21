"""
LLM module with hardcoded NVIDIA configuration
"""

import logging
from langchain_nvidia_ai_endpoints import ChatNVIDIA

logger = logging.getLogger(__name__)

# Hardcoded configuration
LLM_URL = "http://localhost:8088"
LLM_MODEL = "meta/llama-3.1-nemotron-70b-instruct:latest"
MAX_TOKENS = 32768
TEMPERATURE = 0.2

def create_llm():
    """
    Create NVIDIA LLM model with hardcoded configuration.
    
    Returns:
        Configured ChatNVIDIA instance
    """
    logger.info(f"Setting up NVIDIA chat model {LLM_MODEL}")
    
    try:
        chat_model = ChatNVIDIA(
            model_name=LLM_MODEL,
            base_url=LLM_URL,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            model_kwargs={
                "do_sample": True,
                "top_p": 0.9
            }
        )
        
        logger.info("Chat model initialized successfully")
        return chat_model
    
    except Exception as e:
        logger.error(f"Error initializing chat model: {e}")
        raise