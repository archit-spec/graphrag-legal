import pandas as pd
from neo4j import GraphDatabase
import openai
from typing import List, Dict

# Neo4j driver
driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "tammu123"))

# OpenAI configuration
openai.api_key = "your-openai-api-key"

def query_neo4j(query: str, params: Dict = None) -> List[Dict]:
    """Execute a Cypher query and return the results."""
    with driver.session() as session:
        result = session.run(query, params)
        return [record.data() for record in result]

def retrieve_relevant_documents(query: str, top_k: int = 3) -> List[Dict]:
    """Retrieve relevant documents based on the query."""
    cypher_query = """
    CALL db.index.fulltext.queryNodes("documentIndex", $query)
    YIELD node, score
    RETURN node.case_number AS case_number, node.petitioner_name AS petitioner_name, 
           node.respondent_name AS respondent_name, score
    ORDER BY score DESC
    LIMIT $top_k
    """
    return query_neo4j(cypher_query, {"query": query, "top_k": top_k})

def retrieve_document_sections(case_number: str) -> List[Dict]:
    """Retrieve all sections for a given document."""
    cypher_query = """
    MATCH (d:Document {case_number: $case_number})-[:HAS_SECTION]->(s:Section)
    RETURN s.name AS section_name, s.content AS content
    """
    return query_neo4j(cypher_query, {"case_number": case_number})

def generate_response(query: str, context: str) -> str:
    """Generate a response using OpenAI's GPT model."""
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

def graph_rag(query: str) -> str:
    """Perform Graph RAG to answer the query."""
    # Retrieve relevant documents
    relevant_docs = retrieve_relevant_documents(query)
    
    # Collect context from relevant documents
    context = []
    for doc in relevant_docs:
        sections = retrieve_document_sections(doc['case_number'])
        doc_context = f"Case Number: {doc['case_number']}\n"
        doc_context += f"Petitioner: {doc['petitioner_name']}\n"
        doc_context += f"Respondent: {doc['respondent_name']}\n"
        for section in sections:
            doc_context += f"{section['section_name']}:\n{section['content']}\n\n"
        context.append(doc_context)
    
    # Combine context
    combined_context = "\n".join(context)
    
    # Generate response
    response = generate_response(query, combined_context)
    
    return response

# Example usage
query = "What are the key principles of law laid down in cases related to trade tax?"
answer = graph_rag(query)
print(f"Query: {query}")
print(f"Answer: {answer}")

# Don't forget to close the driver when you're done
driver.close()
