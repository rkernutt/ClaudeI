# app.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import boto3
import json
import os
import uuid
from datetime import datetime
import logging
from elasticsearch import Elasticsearch
import PyPDF2
import docx
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG API with Amazon Bedrock")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
ELASTIC_ENDPOINT = os.getenv("ELASTIC_ENDPOINT")  # For Elastic Serverless
ELASTIC_CLOUD_ID = os.getenv("ELASTIC_CLOUD_ID")  # For Elastic Cloud (non-serverless)
ELASTIC_API_KEY = os.getenv("ELASTIC_API_KEY")
ES_INDEX = os.getenv("ES_INDEX", "rag_documents")  # Changed from rag_documents_semantic
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

# Initialize Elasticsearch client
if ELASTIC_ENDPOINT and ELASTIC_API_KEY:
    # Elastic Serverless configuration
    es = Elasticsearch(
        [ELASTIC_ENDPOINT],
        api_key=ELASTIC_API_KEY,
        request_timeout=30,
        verify_certs=True
    )
    logger.info(f"Connected to Elastic Serverless at {ELASTIC_ENDPOINT}")
elif ELASTIC_CLOUD_ID and ELASTIC_API_KEY:
    # Elastic Cloud (non-serverless) configuration
    es = Elasticsearch(
        cloud_id=ELASTIC_CLOUD_ID,
        api_key=ELASTIC_API_KEY,
        request_timeout=30
    )
    logger.info("Connected to Elastic Cloud")
else:
    # Fallback for local development
    ES_HOST = os.getenv("ELASTICSEARCH_HOST", "localhost:9200")
    es = Elasticsearch([f"http://{ES_HOST}"])
    logger.info(f"Connected to local Elasticsearch at {ES_HOST}")

bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = []
    top_k: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: List[dict] = []

class UploadResponse(BaseModel):
    id: str
    filename: str
    content_preview: str
    chunks: int

# Helper functions
def extract_text_from_pdf(file_content: bytes) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise

def extract_text_from_docx(file_content: bytes) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(io.BytesIO(file_content))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        raise

def extract_text_from_file(file: UploadFile) -> str:
    """Extract text based on file type"""
    content = file.file.read()
    
    if file.filename.endswith('.pdf'):
        return extract_text_from_pdf(content)
    elif file.filename.endswith('.docx'):
        return extract_text_from_docx(content)
    elif file.filename.endswith('.txt'):
        return content.decode('utf-8', errors='ignore')
    else:
        raise ValueError(f"Unsupported file type: {file.filename}")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

async def query_bedrock(prompt: str) -> str:
    """Query Amazon Bedrock Claude model using Messages API"""
    try:
        # Use Messages API format for Claude 3
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        # Invoke the model
        response = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )
        
        # Parse the response
        response_body = json.loads(response['body'].read())
        
        # Extract the assistant's response
        if 'content' in response_body and len(response_body['content']) > 0:
            return response_body['content'][0]['text']
        else:
            return "I couldn't generate a response. Please try again."
        
    except Exception as e:
        logger.error(f"Bedrock query error: {str(e)}")
        
        # If it's a validation error, try the old format
        if "ValidationException" in str(e) and "Messages API" in str(e):
            try:
                # Fallback to completion API for older models
                formatted_prompt = f"\n\nHuman: {prompt}\n\nAssistant:"
                request_body = {
                    "prompt": formatted_prompt,
                    "max_tokens_to_sample": 1000,
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
                
                response = bedrock.invoke_model(
                    modelId=BEDROCK_MODEL_ID,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response['body'].read())
                return response_body.get('completion', '').strip()
                
            except Exception as e2:
                logger.error(f"Fallback Bedrock query error: {str(e2)}")
                raise HTTPException(status_code=500, detail=f"LLM query failed: {str(e2)}")
        else:
            raise HTTPException(status_code=500, detail=f"LLM query failed: {str(e)}")

# API Endpoints
@app.get("/")
async def root():
    """Redirect to the frontend"""
    return {"message": "Visit /static/index.html for the frontend"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # For Elastic Serverless, use a simple search instead of cluster health
        try:
            # Just do a simple ping with a match_all query
            result = es.search(index=ES_INDEX, body={"query": {"match_all": {}}, "size": 0})
            es_status = "connected"
            doc_count = result['hits']['total']['value']
        except Exception as e:
            if "index_not_found_exception" in str(e):
                es_status = "connected (no index yet)"
                doc_count = 0
            else:
                es_status = f"error: {str(e)}"
                doc_count = 0
        
        # Check if ELSER is available (this might also not work in serverless)
        elser_available = False
        try:
            models = es.ml.get_trained_models(model_id=".elser*")
            elser_available = len(models.get("trained_model_configs", [])) > 0
        except:
            # ML APIs might not be available in serverless
            elser_available = "unknown (serverless mode)"
            
        return {
            "status": "healthy",
            "elasticsearch": es_status,
            "document_count": doc_count,
            "elser_available": elser_available,
            "mode": "serverless",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and index a document"""
    try:
        # Extract text from file
        text = extract_text_from_file(file)
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Index each chunk to Elasticsearch
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_{i}"
            
            # Index to Elasticsearch (ELSER will process via pipeline)
            es.index(
                index=ES_INDEX,
                id=chunk_id,
                body={
                    "document_id": doc_id,
                    "chunk_index": i,
                    "content": chunk,
                    "filename": file.filename,
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": {
                        "total_chunks": len(chunks),
                        "chunk_size": len(chunk.split())
                    }
                }
            )
        
        # Force refresh to make documents searchable immediately
        es.indices.refresh(index=ES_INDEX)
        
        return UploadResponse(
            id=doc_id,
            filename=file.filename,
            content_preview=text[:200] + "..." if len(text) > 200 else text,
            chunks=len(chunks)
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using text search (optimized for serverless)"""
    try:
        # For Elastic Serverless, use standard text search
        search_body = {
            "query": {
                "multi_match": {
                    "query": request.query,
                    "fields": ["content^2", "filename"],
                    "type": "best_fields",
                    "fuzziness": "AUTO"
                }
            },
            "size": request.top_k,
            "_source": ["content", "filename", "chunk_index", "document_id"],
            "highlight": {
                "fields": {
                    "content": {
                        "fragment_size": 150,
                        "number_of_fragments": 3
                    }
                }
            }
        }
        
        logger.info(f"Searching for: {request.query}")
        results = es.search(index=ES_INDEX, body=search_body)
        
        # Extract relevant chunks
        contexts = []
        sources = []
        
        for hit in results['hits']['hits']:
            source = hit['_source']
            contexts.append(source['content'])
            sources.append({
                "filename": source.get('filename', 'Unknown'),
                "chunk_index": source.get('chunk_index', 0),
                "score": hit['_score']
            })
        
        if not contexts:
            return QueryResponse(
                answer="I couldn't find any relevant information in the uploaded documents. Please make sure you've uploaded documents related to your question.",
                sources=[]
            )
        
        # Build the RAG prompt
        context_text = "\n\n---\n\n".join(contexts)
        
        # Include conversation history if provided
        history_text = ""
        if request.history:
            history_text = "\n".join([
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in request.history[-4:]  # Last 4 messages
            ])
            history_text = f"\nConversation history:\n{history_text}\n"
        
        # Create the prompt for Bedrock
        prompt = f"""You are Claude, a helpful AI assistant. Answer the question based on the provided context from the uploaded documents.

Context from documents:
{context_text}
{history_text}
Question: {request.query}

Please provide a clear, well-structured answer using markdown formatting:
- Use **bold** for important points
- Use bullet points or numbered lists where appropriate
- Use ### for section headers if the answer has multiple parts
- Use > for highlighting key quotes or facts
- Keep paragraphs short and readable

Base your answer on the context above. If the context doesn't contain enough information to fully answer the question, acknowledge what you can answer and what information is missing."""
        
        # Query Bedrock
        answer = await query_bedrock(prompt)
        
        return QueryResponse(
            answer=answer,
            sources=sources
        )
        
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    try:
        # Test connection with a simple request
        info = es.info()
        logger.info(f"Connected to Elasticsearch Serverless")
        
        # Try to create index - for serverless, keep it simple
        try:
            es.indices.create(
                index=ES_INDEX,
                body={
                    "mappings": {
                        "properties": {
                            "content": {"type": "text"},
                            "filename": {"type": "keyword"},
                            "document_id": {"type": "keyword"},
                            "chunk_index": {"type": "integer"},
                            "timestamp": {"type": "date"},
                            "metadata": {"type": "object"}
                        }
                    }
                }
            )
            logger.info(f"Created index: {ES_INDEX}")
        except Exception as e:
            if "resource_already_exists_exception" in str(e) or "400" in str(e):
                logger.info(f"Index {ES_INDEX} already exists")
            else:
                logger.warning(f"Could not create index: {e}")
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)