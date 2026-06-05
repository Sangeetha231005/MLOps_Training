import os
import sys
import shutil
import logging

# Add parent directory of backend to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.getcwd())

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv


# Import components
from backend.parser import SmartPDFParser
from backend.extractor import GeminiExtractor
from backend.graph_db import LegalGraphDB
from backend.vector_db import LegalVectorDB
from backend.rag_engine import HybridRAGEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load env variables
load_dotenv()

# Initialize FastAPI App
app = FastAPI(
    title="TruthTerms AI Backend",
    description="Graph RAG powered legal document analysis API",
    version="1.0"
)

# Configure CORS so local HTML files can communicate with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure data directory exists
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize core modules
db_path = os.path.join(DATA_DIR, "truth_terms.db")
chroma_dir = os.path.join(DATA_DIR, "chromadb")

extractor = GeminiExtractor()
graph_db = LegalGraphDB(db_path=db_path)
vector_db = LegalVectorDB(persist_dir=chroma_dir)
rag_engine = HybridRAGEngine(extractor=extractor, graph_db=graph_db, vector_db=vector_db)
pdf_parser = SmartPDFParser()

class QueryRequest(BaseModel):
    query: str

@app.get("/api/status")
async def get_status():
    """
    Checks the status of the backend, including API key status and database size.
    """
    has_api_key = extractor.api_configured
    
    # Calculate database size to check if document is indexed
    try:
        nodes_count = len(graph_db.get_all_graph_data()["nodes"])
    except Exception:
        nodes_count = 0
        
    return {
        "status": "ready",
        "gemini_api_key_configured": has_api_key,
        "is_indexed": nodes_count > 0,
        "indexed_nodes_count": nodes_count
    }

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a T&C PDF, chunk it, extract knowledge triplets,
    and build the local vector and graph databases.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    temp_path = os.path.join(DATA_DIR, file.filename)
    try:
        # Save file locally
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Parse text into structural chunks
        chunks = pdf_parser.chunk_document(temp_path)
        if not chunks:
            raise HTTPException(status_code=400, detail="Failed to extract any text from the PDF.")

        # Process chunks using Hybrid RAG (embed & parse relationships)
        result = rag_engine.process_document(chunks)
        return {
            "message": "File processed and indexed successfully.",
            "chunks_count": len(chunks),
            "engine_result": result
        }

    except Exception as e:
        logger.error(f"Error during PDF processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/query")
async def query_terms(payload: QueryRequest):
    """
    Query the indexed terms using Hybrid RAG (Vector similarity + Graph traversal).
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    try:
        response = rag_engine.answer_query(payload.query)
        return response
    except Exception as e:
        logger.error(f"Error answering query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to query database: {str(e)}")

@app.get("/api/graph")
async def get_graph():
    """
    Returns the entire Knowledge Graph nodes and edges for rendering.
    """
    try:
        graph_data = graph_db.get_all_graph_data()
        return graph_data
    except Exception as e:
        logger.error(f"Error retrieving graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load graph data: {str(e)}")

# --- Serve Frontend directly from FastAPI server ---

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

# Try to mount frontend directories if they exist
if os.path.exists(FRONTEND_DIR):
    # Route to serve the index.html directly at the root URL "/"
    @app.get("/")
    async def serve_index():
        index_path = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return JSONResponse({"status": "Frontend build files not found. Serve index.html manually."})

    # Serve other assets dynamically
    @app.get("/{filename}")
    async def serve_frontend_assets(filename: str):
        file_path = os.path.join(FRONTEND_DIR, filename)
        if os.path.exists(file_path):
            return FileResponse(file_path)
        raise HTTPException(status_code=404, detail="File not found")
else:
    logger.warning("Frontend folder not found. Serving as API only.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
