"""
Vector store module for Milvus Lite
"""

import logging
import time
from typing import List
from pymilvus import DataType
from langchain.schema import Document
from langchain_milvus import Milvus

logger = logging.getLogger(__name__)

# def create_vectorstore(documents, embedding_model, milvus_db_path):
#     """
#     Create a Milvus Lite vector store with the provided documents.
    
#     Args:
#         documents: List of documents to add to the vector store
#         embedding_model: Embedding model to use
#         milvus_db_path: Path to the Milvus Lite database file
    
#     Returns:
#         Configured Milvus vector store
#     """
#     logger.info("Setting up Milvus Lite vector store")
    
#     # Generate collection name
#     collection_name = f"rag_collection_{int(time.time())}"
    
#     try:
#         # Create a Milvus vector store with Milvus Lite
#         # Explicitly specify FLAT index type which is supported by Milvus Lite
#         vectorstore = Milvus.from_documents(
#             documents=documents,
#             embedding=embedding_model,
#             collection_name=collection_name,
#             connection_args={"uri": milvus_db_path},
#             index_params={"index_type": "FLAT", "metric_type": "L2"},  # Milvus Lite only supports FLAT, IVF_FLAT, AUTOINDEX
#             drop_old=True
#         )
        
#         logger.info(f"Created Milvus Lite vector store with collection: {collection_name}")
#         return vectorstore
    
#     except Exception as e:
#         logger.error(f"Error creating Milvus Lite vector store: {e}")
#         raise
from tenacity import retry, stop_after_attempt, wait_random_exponential

# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
BATCH = 3                 # must match max_batch_size above
MAX_RETRIES = 4

@retry(stop=stop_after_attempt(MAX_RETRIES),
       wait=wait_random_exponential(multiplier=1, max=5))
def _safe_embed(texts, embedder):
    """Embed a small batch with retry/back‑off on 5xx/429 errors."""
    return embedder.embed_documents(texts)


def _to_batches(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


# ------------------------------------------------------------------ #
#  Main entry point used by the rest of your code
# ------------------------------------------------------------------ #
def create_vectorstore(documents, embedding_model, milvus_db_path):
    """
    Build a Milvus‑Lite collection incrementally (streaming) so we never
    send >3 passages per request and never hold all vectors in RAM.
    """
    logger.info("Setting up Milvus Lite vector store (streaming insert)")
    from pymilvus import MilvusClient   # local import keeps deps thin

    client = MilvusClient(milvus_db_path)          # activates Lite
    collection_name = f"rag_collection_{int(time.time())}"

    if client.has_collection(collection_name):
        client.drop_collection(collection_name)

    # Create schema once (FLAT/COSINE index, 1024‑dim)
    client.create_collection(
        collection_name=collection_name,
        dimension=1024,
        primary_field_name="pk",
        vector_field_name="vector",
        id_type=DataType.INT64,
        metric_type="COSINE",
        auto_id=False, 
    )

    pk = 0
    texts, metas = [], []

    logger.info("Inserting documents in batches of %d …", BATCH)

    for doc in documents:
        texts.append(doc.page_content)
        metas.append(doc.metadata)

        if len(texts) == BATCH:
            vecs = _safe_embed(texts, embedding_model)
            client.insert(
                collection_name,
                data=[
                    {"pk": pk + i, "vector": vecs[i], "text": texts[i], **metas[i]}
                    for i in range(len(vecs))
                ],
            )
            pk += len(texts)
            texts, metas = [], []

    # flush final partial batch
    if texts:
        vecs = _safe_embed(texts, embedding_model)
        client.insert(
            collection_name,
            data=[
                {"pk": pk + i, "vector": vecs[i], "text": texts[i], **metas[i]}
                for i in range(len(vecs))
            ],
        )

    client.load_collection(collection_name)
    logger.info("Created Milvus Lite vector store with collection: %s", collection_name)

    # Wrap the MilvusClient in LangChain’s facade so the rest of your code is untouched
    return Milvus(
        embedding_function=embedding_model,
        connection_args={"uri": milvus_db_path},
        collection_name=collection_name,
    )


def search_documents(vectorstore, query_text, top_k=3):
    """
    Search for documents using the vector store.
    
    Args:
        vectorstore: The Milvus vector store
        query_text: Query text
        top_k: Number of results to return
    
    Returns:
        List of search results
    """
    logger.info(f"Searching for: '{query_text}'")
    
    try:
        results = vectorstore.similarity_search(query_text, k=top_k)
        logger.info(f"Search returned {len(results)} results")
        return {"documents": results}
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return {"documents": [], "error": str(e)}