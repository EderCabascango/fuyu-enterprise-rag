import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

load_dotenv()

def run_ingestion():
    # 1. Cargar Documentos
    print("📂 Leyendo PDFs de la carpeta data/...")
    loader = DirectoryLoader('data/', glob="./*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"📄 Se cargaron {len(docs)} páginas en total.")

    # 2. Fragmentación Inteligente
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    total_chunks = len(chunks)
    print(f"✂️  Creados {total_chunks} fragmentos de texto.")

    # 3. Modelos y Clientes
    embeddings_model = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-3-large",
        openai_api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY")
    )

    search_client = SearchClient(
        endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        index_name="fuyu-enterprise-index",
        credential=AzureKeyCredential(os.getenv("AZURE_SEARCH_KEY"))
    )

    # 4. Ingestión con Lotes y Pausas (Anti-Error 429)
    print("🚀 Iniciando subida a Suecia...")
    batch_size = 5  # Subimos de 5 en 5 para no saturar la cuota
    
    for i in range(0, total_chunks, batch_size):
        batch = []
        current_batch_chunks = chunks[i : i + batch_size]
        
        for j, chunk in enumerate(current_batch_chunks):
            # Generar el vector (el embedding)
            vector = embeddings_model.embed_query(chunk.page_content)
            
            # Limpiar metadatos (evitar errores de tipos de datos en Azure)
            clean_metadata = str(chunk.metadata).replace("'", '"')

            batch.append({
                "id": f"chunk-{i+j}",
                "content": chunk.page_content,
                "content_vector": vector,
                "metadata": clean_metadata
            })
        
        # Subir lote y esperar
        try:
            search_client.upload_documents(documents=batch)
            print(f"✅ [{i+len(current_batch_chunks)}/{total_chunks}] Fragmentos indexados...")
            time.sleep(2) # Pausa de 2 segundos para que Azure respire
        except Exception as e:
            print(f"❌ Error en el lote {i}: {e}")
            break

    print("\n🏆 ¡PROCESO TERMINADO! Tu base de conocimientos de la AI-300 está lista.")

if __name__ == "__main__":
    run_ingestion()