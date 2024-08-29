import pandas as pd
from neo4j import GraphDatabase
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "tammu123"))

genai.configure(api_key="")

model = SentenceTransformer('all-MiniLM-L6-v2')

def query_neo4j(query: str, params: Dict = None) -> List[Dict]:
    """Execute a Cypher query and return the results."""
    with driver.session() as session:
        result = session.run(query, params)
        return [record.data() for record in result]

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts."""
    embedding1 = model.encode([text1])
    embedding2 = model.encode([text2])
    return cosine_similarity(embedding1, embedding2)[0][0]

def add_semantic_similarity(query: str):
    """Add semantic similarity attribute to Document nodes."""
    cypher_query = """
    MATCH (d:Document)
    RETURN d.case_number AS case_number, d.petitioner_name AS petitioner_name, 
           d.respondent_name AS respondent_name
    """
    documents = query_neo4j(cypher_query)
    
    for doc in documents:
        text = f"{doc['petitioner_name']} {doc['respondent_name']}"
        similarity = calculate_semantic_similarity(query, text)
        
        update_query = """
        MATCH (d:Document {case_number: $case_number})
        SET d.semantic_similarity = $similarity
        """
        query_neo4j(update_query, {"case_number": doc['case_number'], "similarity": float(similarity)})

def retrieve_relevant_documents(query: str, top_k: int = 3) -> List[Dict]:
    """Retrieve relevant documents based on the query and semantic similarity."""
    add_semantic_similarity(query)
    
    cypher_query = """
    MATCH (d:Document)
    RETURN d.case_number AS case_number, d.petitioner_name AS petitioner_name, 
           d.respondent_name AS respondent_name, d.semantic_similarity AS similarity
    ORDER BY d.semantic_similarity DESC
    LIMIT $top_k
    """
    return query_neo4j(cypher_query, {"top_k": top_k})

def retrieve_document_sections(case_number: str) -> List[Dict]:
    """Retrieve all sections for a given document."""
    cypher_query = """
    MATCH (d:Document {case_number: $case_number})-[:HAS_SECTION]->(s:Section)
    RETURN s.name AS section_name, s.content AS content
    """
    return query_neo4j(cypher_query, {"case_number": case_number})

def generate_response(query: str, context: str) -> str:
    """Generate a response using Gemini model."""
    generation_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "max_output_tokens": 1024,
    }
    model = genai.GenerativeModel(model_name="gemini-1.0-pro", generation_config=generation_config)
    
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = model.generate_content(prompt)
    return response.text

def graph_rag(query: str) -> str:
    """Perform Graph RAG to answer the query."""
    relevant_docs = retrieve_relevant_documents(query)
    
    context = []
    for doc in relevant_docs:
        sections = retrieve_document_sections(doc['case_number'])
        doc_context = f"Case Number: {doc['case_number']}\n"
        doc_context += f"Petitioner: {doc['petitioner_name']}\n"
        doc_context += f"Respondent: {doc['respondent_name']}\n"
        doc_context += f"Similarity: {doc['similarity']}\n"
        for section in sections:
            doc_context += f"{section['section_name']}:\n{section['content']}\n\n"
        context.append(doc_context)
    
    combined_context = "\n".join(context)
    
    response = generate_response(query, combined_context)
    
    return response

query = "What are the key principles of law laid down in cases related to trade tax?"
answer = graph_rag(query)
print(f"Query: {query}")
print(f"Answer: {answer}")

driver.close()
