import asyncio
import logging
from pathlib import Path

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.node_parser import MarkdownNodeParser
from qdrant_client import models

from src.utils.config import get_settings
from src.rag.qdrant import qdrant_client

settings = get_settings()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KB_PATH = Path(__file__).resolve().parent / "knowledge-base.md"
text = KB_PATH.read_text(encoding="utf-8")

Settings.embed_model = GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    api_key=settings.GOOGLE_API_KEY,
)

COLLECTION_NAME = "knowledge_base"


async def injest_knowledge_base():
    logger.info("⏳ Starting ingestion...")

    # 1. Create collection if not exists
    if not qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=3072,
                distance=models.Distance.COSINE,
            ),
        )

    # 2. Setup vector store
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    try:
        # 3. Create documents
        documents = [
            Document(
                text=text,
                metadata={"source": "knowledge-base.md"},
            )
        ]

        # 4. Create nodes
        parser = MarkdownNodeParser()

        nodes = parser.get_nodes_from_documents(documents)

        # 5. Build index 
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            store_nodes_override=True
        )

        # 5. Persist docstore (for BM25)
        storage_context.persist(persist_dir="./storage")

        logger.info("✅ Knowledge base ingested & persisted")
    except Exception as e:
        logger.error(f"❌ Knowledge base Failed: {e}")
        raise e