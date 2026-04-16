import os
import logging
from datetime import datetime
from openai import AsyncAzureOpenAI
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents.aio import SearchClient
from azure.cosmos.aio import CosmosClient

logger = logging.getLogger("fuyu-engine")

class RAGEngine:
    def __init__(self):
        # 1. Credencial única
        self.credential = DefaultAzureCredential()
        
        # 2. Configuración OpenAI
        token_provider = get_bearer_token_provider(
            self.credential, 
            "https://cognitiveservices.azure.com/.default"
        )
        self.client = AsyncAzureOpenAI(
            azure_ad_token_provider=token_provider, 
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        self.index_name = "fuyu-enterprise-index"

        # 3. Configuración Cosmos DB (Persistencia)
        self.cosmos_endpoint = os.getenv("COSMOS_ENDPOINT")
        #self.cosmos_key = os.getenv("COSMOS_KEY")
        self.db_name = "fuyu-db"
        self.container_name = "chat-history"
        self._cosmos_client = None

    async def _get_cosmos_client(self):
        if not self._cosmos_client and self.cosmos_endpoint:
            # Usamos self.credential en lugar de self.cosmos_key
            self._cosmos_client = CosmosClient(self.cosmos_endpoint, self.credential)
        return self._cosmos_client

    async def get_context(self, question):
        """
        Recupera los fragmentos más relevantes desde Azure AI Search
        usando Búsqueda Híbrida + Semantic Re-ranking.
        """
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=self.index_name,
            credential=self.credential
        )
        
        try:
            # 1. Vectorización
            emb = await self.client.embeddings.create(
                input=question, 
                model="text-embedding-3-large"
            )
            
            context_list = []
            
            # 2. Búsqueda Híbrida con Re-ranking Semántico de Azure
            async with search_client:
                results = await search_client.search(
                    search_text=question,
                    vector_queries=[{
                        "vector": emb.data[0].embedding, 
                        "fields": "content_vector", 
                        "k": 10,
                        "kind": "vector"
                    }],
                    query_type="semantic", 
                    semantic_configuration_name="fuyu-semantic-config", 
                    top=5,
                    select=["content"]
                )
                async for res in results:
                    context_list.append(res["content"])
                    
            return context_list

        except Exception as e:
            logger.error(f"❌ Error en Recuperación Semántica: {e}")
            return await self._fallback_search(question)

    async def _fallback_search(self, question):
        """Búsqueda de respaldo si el ranker semántico falla."""
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=self.index_name,
            credential=self.credential
        )
        async with search_client:
            results = await search_client.search(search_text=question, top=3)
            return [res["content"] async for res in results]

    # --- Persistencia con Cosmos DB ---
    async def save_chat_history(self, session_id, history):
        """Guarda el historial en Azure Cosmos DB."""
        client = await self._get_cosmos_client()
        if not client: return

        try:
            db = client.get_database_client(self.db_name)
            container = db.get_container_client(self.container_name)
            data = {
                "id": session_id,
                "history": history,
                "last_updated": datetime.now().isoformat()
            }
            await container.upsert_item(data)
        except Exception as e:
            logger.error(f"❌ Error guardando en Cosmos: {e}")

    async def get_chat_history(self, session_id):
        """Recupera el historial desde Azure Cosmos DB."""
        client = await self._get_cosmos_client()
        if not client: return []

        try:
            db = client.get_database_client(self.db_name)
            container = db.get_container_client(self.container_name)
            item = await container.read_item(item=session_id, partition_key=session_id)
            return item.get("history", [])
        except Exception:
            return []

    async def close(self):
        """Cierra las sesiones activas."""
        await self.client.close()
        await self.credential.close()
        if self._cosmos_client:
            await self._cosmos_client.close()
        logger.info("🔌 Sesiones de Azure cerradas correctamente.")