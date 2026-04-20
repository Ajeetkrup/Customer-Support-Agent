from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.llms.gemini import Gemini
from llama_index.llms.groq import Groq
from src.utils.config import get_settings
from src.rag.qdrant import aqdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from src.rag.qdrant import redis_kv
from src.rag.query_helpers import get_llm_response_async

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
        # 1. Connect to the existing Qdrant collection
        vector_store = QdrantVectorStore(
            aclient=aqdrant_client, 
            collection_name=COLLECTION_NAME
        )

        # 2. Load the Index from the Vector Store
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # 3. Create Engine and Query
        query_engine = index.as_query_engine(similarity_top_k=3)
        
        response = await get_llm_response_async(query, query_engine)

        text = getattr(response, "response", None) or str(response)
        return text

    except Exception as e:
        print(f"❌ Error querying Qdrant: {e}")
        return None