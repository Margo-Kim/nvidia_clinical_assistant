# vectorstore.py

import logging
import time
from typing import Iterable, List

from tenacity import retry, wait_random_exponential, stop_after_attempt
from langchain.schema import Document
from langchain_milvus import Milvus
from pymilvus import MilvusClient, DataType

logger = logging.getLogger(__name__)

# Batch size must be ≤ your embedding_model.max_batch_size
BATCH = 3
MAX_RETRIES = 4

@retry(wait=wait_random_exponential(multiplier=1, max=5),
       stop=stop_after_attempt(MAX_RETRIES))
def _embed(texts: List[str], embedder):
    """Embed a batch — if it fails, retry items one by one so only the broken chunk is dropped."""
    try:
        return embedder.embed_documents(texts)
    except Exception as batch_err:
        good = []
        for t in texts:
            try:
                good.extend(embedder.embed_documents([t]))
            except Exception as e:
                logger.warning("❌ Dropping bad passage: %s", e)
        if not good:
            raise batch_err
        return good

def _batched(iterable: Iterable, size: int):
    """Yield consecutive lists of length `size`."""
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) == size:
            yield buf
            buf = []
    if buf:
        yield buf

def create_vectorstore(
    documents: List[Document],
    embedding_model,
    milvus_db_path: str
) -> Milvus:
    """
    1) Creates a Milvus‑Lite collection with:
         - pk (INT64 auto-generated)
         - vector (FLOAT_VECTOR, dim=1024)
         - text (VARCHAR, up to 4096 chars)
       AND dynamic fields turned on so all your metadata (source, chunk, etc.) are stored too.
    2) Streams inserts in tiny BATCH batches, with retry/fallback.
    3) Returns the LangChain Milvus wrapper for .similarity_search().
    """
    coll_name = f"rag_collection_{int(time.time())}"
    client = MilvusClient(uri=milvus_db_path)

    # Drop any leftover
    if client.has_collection(coll_name):
        client.drop_collection(coll_name)

    # Shortcut API + dynamic fields
    client.create_collection(
        collection_name      = coll_name,
        dimension            = 1024,
        primary_field_name   = "pk",
        vector_field_name    = "vector",
        id_type              = DataType.INT64,
        metric_type          = "COSINE",
        auto_id              = True,
        enable_dynamic_field = True,    # ← crucial
    )

    # Insert in safe, retryable batches
    for chunk in _batched(documents, BATCH):
        texts = [doc.page_content for doc in chunk]
        metas = [doc.metadata     for doc in chunk]
        vecs  = _embed(texts, embedding_model)

        rows = [
            { "vector": v, "text": t, **m }
            for v, t, m in zip(vecs, texts, metas)
        ]
        client.insert(coll_name, data=rows)
        logger.info("Inserted %d rows into %s", len(rows), coll_name)

    client.load_collection(coll_name)
    logger.info("Milvus‑Lite vector store ready: %s (%d docs)", coll_name, len(documents))

    # Wrap in LangChain facade
    return Milvus(
        embedding_function=embedding_model,
        connection_args={"uri": milvus_db_path},
        collection_name=coll_name,
    )

def search_documents(vectorstore: Milvus, query_text: str, top_k: int = 3):
    """
    Simple wrapper—calls vectorstore.similarity_search(...)
    """
    logger.info("Searching for: '%s'", query_text)
    try:
        docs = vectorstore.similarity_search(query_text, k=top_k)
        logger.info("Search returned %d hits", len(docs))
        return {"documents": docs}
    except Exception as e:
        logger.error("Error during search: %s", e)
        return {"documents": [], "error": str(e)}
