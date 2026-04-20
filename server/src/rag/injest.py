import asyncio
import logging
import uuid

from src.utils.config import get_settings
from src.rag.qdrant import qdrant_client
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from sqlalchemy import bindparam, text, update
from pathlib import Path
from src.utils.config import get_settings
from qdrant_client import models

settings = get_settings()
KB_PATH = Path(__file__).resolve().parent / "knowledge-base.md"
text = KB_PATH.read_text(encoding="utf-8")

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=settings.GOOGLE_API_KEY,
)

COLLECTION_NAME = "knowledge_base"

settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def injest_knowledge_base():
    logger.info("⏳ Starting batch processing job...")

    # Check if collection exists using the imported client
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=3072,
                distance=models.Distance.COSINE,
            ),
            quantization_config=models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(type=models.ScalarType.INT8, always_ram=True)
            ),
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client, collection_name=COLLECTION_NAME
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    node_parser = MarkdownNodeParser()

    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context, node_parser=node_parser
    )

    documents = [Document(text=text)]
    doc = Document(
        text=text,
        metadata={"source": "knowledge-base.md"},
    )

    try:
        await asyncio.to_thread(
            index.insert_nodes, documents
        )
        logger.info(f"✅ Successfully processed knowledge base")
    except Exception as e:
        logger.error(f"❌ Knowledge base Failed: {e}")
        raise e

    logger.info("🎉 Knowledge base job finished.")