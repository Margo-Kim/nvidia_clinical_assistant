import logging
import time
from typing import Iterable, List

from tenacity import retry, wait_random_exponential, stop_after_attempt
from langchain.schema import Document
from langchain_milvus import Milvus
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

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
         - page_content (VARCHAR, up to 65535 chars)
       AND dynamic fields turned on so all your metadata (source, chunk, etc.) are stored too.
    2) Streams inserts in tiny BATCH batches, with retry/fallback.
    3) Returns the LangChain Milvus wrapper for .similarity_search().
    """
    coll_name = f"rag_collection_{int(time.time())}"
    client = MilvusClient(uri=milvus_db_path)

    # Drop any leftover
    if client.has_collection(coll_name):
        client.drop_collection(coll_name)

    # Log embedding model info for debugging
    logger.info(f"Using embedding model type: {type(embedding_model)}")
    try:
        embedding_dim = embedding_model.dimension
        logger.info(f"Embedding dimension: {embedding_dim}")
    except (AttributeError, TypeError):
        logger.warning("Could not determine embedding dimension, using default 1024")
        embedding_dim = 1024

    # Create explicit schema with required fields
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    
    schema = CollectionSchema(fields=fields, enable_dynamic_field=True)
    
    # Create collection with explicit schema
    client.create_collection(
        collection_name=coll_name, 
        schema=schema
    )

    # Insert in safe, retryable batches
    for chunk in _batched(documents, BATCH):
        texts = [doc.page_content for doc in chunk]
        metas = [doc.metadata for doc in chunk]
        vecs = _embed(texts, embedding_model)

        # Store document text as 'text' field (this is crucial)
        rows = []
        for v, t, m in zip(vecs, texts, metas):
            row = {
                "vector": v,
                "text": t  # The field name MUST match what we define in Milvus constructor
            }
            # Add metadata as separate fields
            for key, value in m.items():
                row[key] = value
            
            rows.append(row)
            
        # Debug logging
        logger.info("Sample row structure: %s", str({k: type(v).__name__ for k, v in rows[0].items()}))
        
        # Insert the rows
        try:
            client.insert(coll_name, data=rows)
            logger.info("Inserted %d rows into %s", len(rows), coll_name)
        except Exception as e:
            logger.error("Error inserting rows: %s", e)
            raise

    # Create index on vector field
    try:
        client.create_index(
            collection_name=coll_name,
            field_name="vector",
            index_params={
                "index_type": "FLAT",
                "metric_type": "COSINE",
                "params": {}
            }
        )
        logger.info("Index created successfully")
    except Exception as e:
        logger.error(f"Failed to create index: {e}")
        # This is non-fatal for Milvus Lite, can continue

    # Load collection
    client.load_collection(coll_name)
    logger.info("Milvus‑Lite vector store ready: %s (%d docs)", coll_name, len(documents))
    
    # Verify the fields
    sample = client.query(collection_name=coll_name, filter="", limit=1)
    if sample:
        logger.info("Sample document keys: %s", str(list(sample[0].keys())))

    # Wrap in LangChain facade with text_field="text"
    return Milvus(
        embedding_function=embedding_model,
        connection_args={"uri": milvus_db_path},
        collection_name=coll_name,
        text_field="text"  # This MUST match the field name used above
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