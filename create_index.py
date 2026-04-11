import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
    VectorSearchAlgorithmKind
)

load_dotenv()

def create_fuyu_index():
    endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    key = os.getenv("AZURE_SEARCH_KEY")
    index_name = "fuyu-enterprise-index"

    client = SearchIndexClient(endpoint, AzureKeyCredential(key))

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String),
        SearchField(name="content_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=3072, vector_search_profile_name="fuyu-vector-profile"),
        SimpleField(name="metadata", type=SearchFieldDataType.String),
    ]

    vector_search = VectorSearch(
        profiles=[VectorSearchProfile(name="fuyu-vector-profile", algorithm_configuration_name="fuyu-algorithms")],
        algorithms=[HnswAlgorithmConfiguration(name="fuyu-algorithms", kind=VectorSearchAlgorithmKind.HNSW)]
    )

    index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
    
    try:
        client.create_index(index)
        print(f"✅ Índice '{index_name}' creado con éxito.")
    except Exception as e:
        print(f"❌ Error al crear el índice: {e}")

if __name__ == "__main__":
    create_fuyu_index()