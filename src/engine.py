import os
from openai import AsyncAzureOpenAI
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents.aio import SearchClient

class RAGEngine:
    def __init__(self):
        # 1. Credencial que detecta la identidad automáticamente
        self.credential = DefaultAzureCredential()
        
        # 2. Proveedor de tokens (reemplaza la API Key de OpenAI)
        token_provider = get_bearer_token_provider(
            self.credential, 
            "https://cognitiveservices.azure.com/.default"
        )

        self.client = AsyncAzureOpenAI(
            azure_ad_token_provider=token_provider, 
            api_version="2024-08-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.model = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")

    async def get_context(self, question):
        # 3. El cliente de búsqueda ahora usa la misma identidad
        search_client = SearchClient(
            endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            index_name="fuyu-enterprise-index",
            credential=self.credential # <-- Sin llaves aquí tampoco
        )
        
        # Vectorización (Embeddings)
        emb = await self.client.embeddings.create(
            input=question, 
            model="text-embedding-3-large"
        )
        
        context_list = []
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