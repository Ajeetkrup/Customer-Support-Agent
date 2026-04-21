from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from src.utils.config import get_settings
from src.rag.qdrant import aqdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from src.rag.qdrant import redis_kv
from src.rag.query_helpers import get_llm_response_async
from llama_index.retrievers.bm25 import BM25Retriever  
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.postprocessor.sbert_rerank import SentenceTransformerRerank 
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine
from llama_index.core import load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

Settings.embed_model =  GoogleGenAIEmbedding(
    model_name="gemini-embedding-001",
    embed_batch_size=100,
    api_key=settings.GOOGLE_API_KEY,
    embeddings_cache=redis_kv,
)

Settings.llm = Groq(model="qwen/qwen3-32b", api_key=settings.GROQ_API_KEY, temperature=0.2, max_tokens=512,)
# Settings.llm = Gemini(
#     model="models/gemini-flash-latest", 
#     api_key=settings.GOOGLE_API_KEY,
#     temperature=0.2,          
#     max_output_tokens=512,
#     max_retries=2
# )

COLLECTION_NAME = "knowledge_base"

async def retrieve_from_qdrant(query: str):
    try:
        time_start = time.time()
        # 1. Connect to the existing Qdrant collection
        vector_store = QdrantVectorStore(
            aclient=aqdrant_client, 
            collection_name=COLLECTION_NAME
        )
        logger.info(f"Time taken to connect to Qdrant: {time.time() - time_start} seconds")
        time_start = time.time()
        storage_context = StorageContext.from_defaults(
            persist_dir="./storage",
            vector_store=vector_store
        )
        logger.info(f"Time taken to create storage context: {time.time() - time_start} seconds")
        time_start = time.time()
        # Load the Index from the storage context
        index = load_index_from_storage(storage_context)
        print("len(index.docstore.docs): ------- ", len(index.docstore.docs))
        logger.info(f"Time taken to load index: {time.time() - time_start} seconds")
        time_start = time.time()
        vector_retriever = index.as_retriever(similarity_top_k=5)

        bm25_retriever = BM25Retriever.from_defaults(
            docstore=index.docstore,
            similarity_top_k=5
        )
        logger.info(f"Time taken to create BM25 retriever: {time.time() - time_start} seconds")
        time_start = time.time()
        # hybrid retrieval
        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            num_queries=3, 
            mode="reciprocal_rerank"
        )
        logger.info(f"Time taken to create query fusion retriever: {time.time() - time_start} seconds")
        time_start = time.time()
        #reranker
        reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=3
        )
        logger.info(f"Time taken to create sentence transformer reranker: {time.time() - time_start} seconds")
        time_start = time.time()
        # query rewriting
        hyde = HyDEQueryTransform(include_original=True)
        logger.info(f"Time taken to create HyDE query transform: {time.time() - time_start} seconds")
        time_start = time.time()
        query_engine = RetrieverQueryEngine.from_args(
            retriever=retriever,
            node_postprocessors=[reranker]
        )
        logger.info(f"Time taken to create retriever query engine: {time.time() - time_start} seconds")
        time_start = time.time()
        # Apply HyDE on top
        query_engine = TransformQueryEngine(
            query_engine=query_engine,
            query_transform=hyde
        )
        logger.info(f"Time taken to create transform query engine: {time.time() - time_start} seconds")
        time_start = time.time()
        response = await get_llm_response_async(query, query_engine)
        logger.info(f"Time taken to get LLM response: {time.time() - time_start} seconds")
        time_start = time.time()

        text = getattr(response, "response", None) or str(response)
        return text

    except Exception as e:
        print(f"❌ Error querying Qdrant: {e}")
        raise e