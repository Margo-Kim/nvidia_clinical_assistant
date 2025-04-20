"""
Debug script for NVIDIA embedding service
"""

import logging
import sys
import os
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

logger = logging.getLogger(__name__)

def test_nvidia_embeddings(
    base_url="http://localhost:8089", 
    model_name="nvidia/nv-embedqa-e5-v5"
):
    """
    Test the NVIDIA embedding service with detailed logging
    
    Args:
        base_url: URL of the NVIDIA embedding service
        model_name: Name of the embedding model
    """
    logger.info(f"Testing NVIDIA embeddings at {base_url} with model {model_name}")
    
    # Print environment information
    logger.info(f"Environment information:")
    env_vars = [
        "NVIDIA_API_KEY", 
        "NVIDIA_BASE_URL",
        "HTTP_PROXY", 
        "HTTPS_PROXY", 
        "NO_PROXY"
    ]
    
    for var in env_vars:
        value = os.environ.get(var, "Not set")
        if var.endswith("_KEY") and value != "Not set":
            logger.info(f"{var}: **Set but not shown**")
        else:
            logger.info(f"{var}: {value}")
    
    try:
        # Initialize the embedding model
        logger.info("Initializing NVIDIAEmbeddings...")
        embedding_model = NVIDIAEmbeddings(
            model_name=model_name,
            base_url=base_url,
            max_batch_size=1  # Start with just 1 to make debugging easier
        )
        
        # Check available models
        try:
            logger.info("Fetching available models...")
            available_models = embedding_model.available_models
            logger.info(f"Available models: {[model.id for model in available_models]}")
        except Exception as e:
            logger.error(f"Error fetching available models: {e}")
            logger.error(f"Error type: {type(e).__name__}")
        
        # Try to embed a single text
        logger.info("Attempting to embed single text...")
        test_text = "This is a test of the NVIDIA embedding service."
        
        try:
            embedding = embedding_model.embed_query(test_text)
            logger.info(f"Successfully generated embedding with dimension {len(embedding)}")
            logger.info(f"Embedding sample (first 5 values): {embedding[:5]}")
            
            return True, embedding
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            
            # Try to get more details
            if hasattr(e, "response"):
                try:
                    response = e.response
                    logger.error(f"Response status code: {response.status_code}")
                    logger.error(f"Response headers: {response.headers}")
                    logger.error(f"Response content: {response.content}")
                except:
                    logger.error("Could not extract response details")
            
            return False, str(e)
    
    except Exception as e:
        logger.error(f"Error initializing embedding model: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return False, str(e)

def try_multiple_configurations():
    """Try multiple configurations to see which one works"""
    
    # Test configurations
    configurations = [
        {
            "base_url": "http://localhost:8089",
            "model_name": "nvidia/nv-embedqa-e5-v5"
        },
        {
            "base_url": "http://localhost:8089/v1",
            "model_name": "nvidia/nv-embedqa-e5-v5"
        },
        {
            "base_url": "http://localhost:8089",
            "model_name": "nv-embedqa-e5-v5"
        },
        {
            "base_url": "http://localhost:8089/v1/embeddings",
            "model_name": "nvidia/nv-embedqa-e5-v5"
        }
    ]
    
    # Try each configuration
    for i, config in enumerate(configurations):
        logger.info(f"\n\n=== Testing Configuration {i+1} ===")
        success, result = test_nvidia_embeddings(**config)
        if success:
            logger.info(f"Configuration {i+1} SUCCESSFUL!")
            logger.info(f"Working configuration: {config}")
            return config
        else:
            logger.info(f"Configuration {i+1} failed: {result}")
    
    logger.info("All configurations failed")
    return None

if __name__ == "__main__":
    logger.info("Starting NVIDIA embedding service debug script")
    working_config = try_multiple_configurations()
    
    if working_config:
        print("\n\nSUCCESS! Found working configuration:")
        print(f"Base URL: {working_config['base_url']}")
        print(f"Model name: {working_config['model_name']}")
    else:
        print("\n\nAll configurations failed. Please check:")
        print("1. Is the NVIDIA embedding service running?")
        print("2. Is the service URL correct?")
        print("3. Is the model name correct?")
        print("4. Are any required API keys or environment variables set?")
        print("5. Check the service logs for more details.")