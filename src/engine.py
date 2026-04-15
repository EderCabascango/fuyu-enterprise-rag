import os
import logging
from openai import AsyncAzureOpenAI
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents.aio import SearchClient

logger = logging.getLogger("fuyu-engine")

class RAGEngine:
    def __init__(self):
        # 1. Credencial única para todos los servicios de Azure (Identity)
        self.credential = DefaultAzureCredential()
        
        # 2. Proveedor de tokens para OpenAI (Manejo automático de rotación)
        token_provider = get_bearer_token_provider(
            self.credential, 
            "https://cognitiveservices.azure.com/.default"
        )

        # 3. Cliente de OpenAI persistente
        self.client = AsyncAzureOpenAI(
            azure_ad_token_provider=token_provider, 
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
        self.index_name = "fuyu-enterprise-index"

    async def get_context(self, question):
        """
        Recupera los 3 fragmentos más relevantes desde Azure AI Search
        usando búsqueda híbrida (Vector + Keyword).
        """
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name=self.index_name,
            credential=self.credential
        )
        
        try:
            # Vectorización de la consulta (Embeddings)
            emb = await self.client.embeddings.create(
                input=question, 
                model="text-embedding-3-large"
            )
            
            context_list = []
            
            # El bloque 'async with' cierra el SearchClient automáticamente al terminar
            async with search_client:
                results = await search_client.search(
                    search_text=question,
                    vector_queries=[{
                        "vector": emb.data[0].embedding, 
                        "fields": "content_vector", 
                        "k": 3, 
                        "kind": "vector"
                    }],
                    top=3,
                    select=["content"]
                )
                async for res in results:
                    context_list.append(res["content"])
                    
            return context_list

        except Exception as e:
            logger.error(f"❌ Error recuperando contexto: {e}")
            return []

    async def close(self):
        """
        Cierra las sesiones activas para evitar advertencias de 'Unclosed client session'.
        """
        await self.client.close()
        await self.credential.close()
        logger.info("🔌 Sesiones de Azure cerradas correctamente.")