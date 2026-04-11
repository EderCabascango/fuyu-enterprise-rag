/*
==========================================================================
PROYECTO: Fuyu Enterprise RAG - Infraestructura Base
DESCRIPCIÓN: Despliegue de Azure AI Search y OpenAI con modelos GlobalStandard.
ESTÁNDAR: Bicep (Azure IaC) - Versión Abril 2026
==========================================================================
*/

@description('Ubicación regional (recomendado: uksouth según validación de cuota).')
param location string = resourceGroup().location

@description('Prefijo para nombres únicos de recursos.')
param prefix string = 'fuyu'

// --- 1. AZURE AI SEARCH ---
resource searchService 'Microsoft.Search/searchServices@2023-11-01' = {
  name: '${prefix}-search-service'
  location: location
  sku: {
    name: 'basic' // Requerido para búsqueda vectorial y RAG profesional.
  }
  properties: {
    replicaCount: 1
    partitionCount: 1
    hostingMode: 'default'
  }
}

// --- 2. AZURE OPENAI SERVICE ---
resource openAiService 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: '${prefix}-openai-service'
  location: location
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: '${prefix}-openai-service'
    publicNetworkAccess: 'Enabled'
    // Evita el error de "Soft-deleted" si el nombre ya se usó.
    restore: false 
  }
}

// --- 3. DESPLIEGUE DE MODELOS (LLM & EMBEDDINGS) ---

// GPT-4o-mini: Usamos GlobalStandard porque es donde reside tu cuota disponible.
resource gptMini 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openAiService
  name: 'gpt-4o-mini'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o-mini'
      version: '2024-07-18'
    }
  }
  sku: {
    name: 'GlobalStandard'
    capacity: 1 
  }
}

// Text-Embedding-3-Large: El traductor vectorial.
resource embedding 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openAiService
  name: 'text-embedding-3-large'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-3-large'
      version: '1'
    }
  }
  sku: {
    name: 'GlobalStandard'
    capacity: 1 
  }
}

// --- SALIDAS (OUTPUTS) ---
output openAiEndpoint string = openAiService.properties.endpoint
output searchEndpoint string = 'https://${searchService.name}.search.windows.net'
