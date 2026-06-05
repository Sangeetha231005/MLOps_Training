import os
import json
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    logger.warning("GEMINI_API_KEY not found in environment variables. Gemini features will fail unless it is provided.")

# Define Pydantic Schemas for Structured JSON Output
class Entity(BaseModel):
    name: str = Field(description="Name of the legal entity, actor, section/clause, or definition (e.g. 'User', 'Company', 'Section 10', 'Personal Data')")
    type: str = Field(description="Type of entity: 'Clause' (for document sections), 'Actor' (parties involved), 'Risk' (user vulnerabilities), or 'Definition' (legal terms)")

class Relationship(BaseModel):
    source: str = Field(description="The source entity name (e.g. 'Indemnification')")
    relation: str = Field(description="The verb/relationship: 'REFERENCES', 'AGREES_TO', 'SHARED_WITH', 'LIMITS', 'CONTAINS', 'EXCLUDES', 'PREVENTS'")
    target: str = Field(description="The target entity name (e.g. 'Section 10')")

class Risk(BaseModel):
    name: str = Field(description="Name of the risk (e.g. 'Account Suspension', 'Auto Renewal')")
    severity: str = Field(description="Severity score: 'High', 'Medium', or 'Low'")
    description: str = Field(description="Detailed explanation of what this risk entails and why it matters to the user")

class ExtractionResult(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]
    risks: List[Risk]

class GeminiExtractor:
    """
    Handles communications with Gemini for Entity Extraction and Vector Embeddings.
    """
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
        self.api_configured = bool(os.getenv("GEMINI_API_KEY"))

    def extract_legal_triplets(self, chunk_title: str, chunk_text: str) -> Dict[str, Any]:
        """
        Calls Gemini to extract entities, relationships, and risks from a clause chunk in a structured schema.
        """
        if not self.api_configured:
            # Return empty/mock template if API key is missing to avoid crashes
            return {"entities": [], "relationships": [], "risks": []}

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "response_mime_type": "application/json",
                    "response_schema": ExtractionResult,
                }
            )

            prompt = f"""
            You are a legal expert analyzing a Terms & Conditions / Privacy Policy document.
            Analyze the following text section titled "{chunk_title}".
            
            Perform the following extractions:
            1. Extract core Entities (Actors like User/Company, Clauses like Section names, Risks like Data Sharing, and Definitions like Services).
            2. Extract Relationships (how these entities interact or reference each other). Pay extreme attention to cross-references (e.g. Section A references Section B).
            3. Extract Risks: identify any clauses that limit user rights, allow tracking, share data, auto-renew payments, or waive liabilities.
            
            Text to analyze:
            {chunk_text}
            """

            response = model.generate_content(prompt)
            data = json.loads(response.text)
            return data

        except Exception as e:
            logger.error(f"Error during Gemini extraction: {str(e)}")
            # Return fallback structure
            return {"entities": [], "relationships": [], "risks": []}

    def get_embedding(self, text: str) -> List[float]:
        """
        Computes 768-dimensional embeddings using Gemini's embedding model.
        """
        if not self.api_configured:
            # Fallback mock embedding (all zeros) for zero-config run
            return [0.0] * 768

        try:
            # Truncate text if too long for embedding limits
            text_truncated = text[:8000]
            result = genai.embed_content(
                model="models/gemini-embedding-001",
                content=text_truncated,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return [0.0] * 768

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generates the final response to a user query based on the retrieved Hybrid RAG context.
        """
        if not self.api_configured:
            return "Error: Gemini API Key is missing. Please add GEMINI_API_KEY in your .env file to run queries."

        try:
            model = genai.GenerativeModel(self.model_name)
            prompt = f"""
            You are TruthTerms AI, an intelligent assistant designed to help users understand complex Terms and Conditions.
            Use the following context retrieved from the legal document (both vector search and connected graph clauses) to answer the user's question.
            
            Provide a clear, detailed, and objective response. Be upfront about risks. Highlight specific sections, clauses, or definitions where applicable.
            Do not make up facts. If the information is not in the context, say that you could not find it.

            [RETRIEVED CONTEXT]
            {context}

            [USER QUESTION]
            {query}

            [INSTRUCTIONS]
            - Structure your answer with clear markdown bullet points/headings.
            - Explicitly cite section numbers (e.g. "[Section 5]") when referring to specific statements.
            - Summarize any potential risks or hidden catches mentioned in the context.
            """

            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return f"An error occurred while communicating with Gemini: {str(e)}"

if __name__ == "__main__":
    # Quick test if key exists
    extractor = GeminiExtractor()
    if extractor.api_configured:
        print("Embedding test:")
        emb = extractor.get_embedding("This is a test clause about refund policies.")
        print(f"Embedding length: {len(emb)}, Sample values: {emb[:5]}")
    else:
        print("Gemini key not configured. Mocking active.")
