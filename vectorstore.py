# vectorstore.py

import logging
import time
from typing import Iterable, List

from tenacity import retry, wait_random_exponential, stop_after_attempt
from langchain.schema import Document
from langchain_milvus import Milvus
from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType

logger = logging.getLogger(__name__)

# Must not exceed embedding_model.max_batch_size
BATCH = 3
MAX_RETRIES = 4

@retry(wait=wait_random_exponential(multiplier=1, max=5),
       stop=stop_after_attempt(MAX_RETRIES))
def _embed(texts: List[str], embedder):
    """Embed a small batch, with fallback to single‑item if needed."""
    try:
        return embedder.embed_documents(texts)
    except Exception as batch_err:
        # try individually so only the bad passage is dropped
        good_vecs = []
        for t in texts:
            try:
                good_vecs.extend(embedder.embed_documents([t]))
            except Exception as e:
                logger.warning("❌ Dropping bad passage: %s", e)
        if not good_vecs:
            raise batch_err
        return good_vecs

def _batched(iterable: Iterable, size: int):
    """Yield successive batches from `iterable` of length `size`."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch

def create_vectorstore(
    documents: List[Document],
    embedding_model,
    milvus_db_path: str
) -> Milvus:
    """
    Create a Milvus‑Lite collection and stream‑insert your documents in batches.
    Returns a LangChain Milvus wrapper for similarity_search().
    """
    coll_name = f"rag_collection_{int(time.time())}"
    client = MilvusClient(milvus_db_path)

    # Drop if exists
    if client.has_collection(coll_name):
        client.drop_collection(coll_name)

    # Define explicit schema (pk, vector, text)
    fields = [
        FieldSchema(name="pk",     dtype=DataType.INT64,       is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="text",   dtype=DataType.VARCHAR,      max_length=4096),
    ]
    schema = CollectionSchema(fields, description="RAG FiQA collection")
    client.create_collection(collection_name=coll_name, schema=schema)

    # Stream‑insert in small batches
    for chunk in _batched(documents, BATCH):
        texts = [doc.page_content for doc in chunk]
        metas = [doc.metadata     for doc in chunk]
        vecs  = _embed(texts, embedding_model)

        # Build row dicts
        rows = [
            {"vector": v, "text": t, **m}
            for v, t, m in zip(vecs, texts, metas)
        ]

        # *** correct signature: data=rows ***
        client.insert(coll_name, data=rows)
        logger.info("Inserted %d vectors into %s", len(rows), coll_name)

    client.load_collection(coll_name)
    logger.info("Milvus‑Lite vector store ready: %s (%d docs)", coll_name, len(documents))

    # Wrap for LangChain so similarity_search works unchanged
    return Milvus(
        embedding_function=embedding_model,
        connection_args={"uri": milvus_db_path},
        collection_name=coll_name,
    )

def search_documents(vectorstore: Milvus, query_text: str, top_k: int = 3):
    """
    Perform a similarity search and return LangChain Documents.
    """
    logger.info("Searching for: '%s'", query_text)
    try:
        docs = vectorstore.similarity_search(query_text, k=top_k)
        logger.info("Search returned %d results", len(docs))
        return {"documents": docs}
    except Exception as e:
        logger.error("Error searching documents: %s", e)
        return {"documents": [], "error": str(e)}
