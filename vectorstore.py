"""
Vector store module for Milvus Lite
"""

import logging
import time
from typing import Iterable, List
from pymilvus import DataType
from langchain.schema import Document
from langchain_milvus import Milvus
from tenacity import retry, wait_random_exponential, stop_after_attempt

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
BATCH = 5                 # must ≤ embedding_model.max_batch_size
MAX_RETRIES = 4

@retry(wait=wait_random_exponential(1, 4),
       stop=stop_after_attempt(MAX_RETRIES))
def _embed(texts: List[str], embedder):
    """
    Embed a batch. If the batch fails, fall back to single‑item calls
    so only the problematic passage is skipped.
    """
    try:
        return embedder.embed_documents(texts)
    except Exception as batch_err:
        good_vecs = []
        for t in texts:
            try:
                good_vecs.extend(embedder.embed_documents([t]))
            except Exception as e:
                logger.warning("❌ Dropping a passage that NIM still rejects (%s)", e)
        if not good_vecs:       # nothing worked → raise original error
            raise batch_err
        return good_vecs


def _batched(iterable: Iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def create_vectorstore(documents: List[Document], embedding_model, milvus_db_path):
    """
    Create a Milvus Lite collection and stream‑insert documents in batches of 3.
    """
    from pymilvus import FieldSchema, CollectionSchema, MilvusClient, DataType

    logger.info("Setting up Milvus Lite vector store (incremental insert)")

    coll_name = f"rag_collection_{int(time.time())}"
    client = MilvusClient(milvus_db_path)
    if client.has_collection(coll_name):
        client.drop_collection(coll_name)
    text_field = FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        max_length=4096,
        is_primary=False,
    )
    # define schema (auto_id=True lets Milvus assign pk)
    client.create_collection(
        collection_name=coll_name,
        dimension=1024,
        primary_field_name="pk",
        vector_field_name="vector",
        id_type=DataType.INT64,
        metric_type="COSINE",
        auto_id=True,
        enable_dynamic_field = True,
        fields=[text_field],  
    )

    # insert in safe batches

    for chunk in _batched(documents, BATCH):
        texts  = [d.page_content for d in chunk]
        metas  = [d.metadata     for d in chunk]
        vecs   = _embed(texts, embedding_model)
        client.insert(
            coll_name,
            data=[
                {"vector": v, "text": t, **m} for v, t, m in zip(vecs, texts, metas)
            ],
        )
        logger.info(f"Inserted {len(vecs)} vectors into Milvus Lite collection: {coll_name}")

    client.load_collection(coll_name)
    logger.info("Created Milvus Lite vector store: %s (%s docs)", coll_name, len(documents))

    # return LangChain wrapper so existing code stays the same
    return Milvus(
        embedding_function=embedding_model,
        connection_args={"uri": milvus_db_path},
        collection_name=coll_name,
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