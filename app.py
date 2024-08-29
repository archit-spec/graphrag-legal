import pandas as pd
from neo4j import GraphDatabase
import openai
import google.generativeai as genai
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx
from typing import List, Dict

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "tammu123"))

genai.configure(api_key="")

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
    """Generate a response using Google's Gemini API."""
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b-exp-0827",
        generation_config={
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
    )
    
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(prompt)
    return response.text.strip()

def graph_rag(query: str) -> str:
    """Perform Graph RAG to answer the query."""
    relevant_docs = retrieve_relevant_documents(query)
    
    context = []
    for doc in relevant_docs:
        sections = retrieve_document_sections(doc['case_number'])
        doc_context = f"Case Number: {doc['case_number']}\n"
        doc_context += f"Petitioner: {doc['petitioner_name']}\n"
        doc_context += f"Respondent: {doc['respondent_name']}\n"
        for section in sections:
            doc_context += f"{section['section_name']}:\n{section['content']}\n\n"
        context.append(doc_context)
    
    combined_context = "\n".join(context)
    
    response = generate_response(query, combined_context)
    
    return response

def visualize_kg():
    G = nx.Graph()

    G.add_node("Case 1", label="Case")
    G.add_node("Petitioner A", label="Petitioner")
    G.add_node("Respondent B", label="Respondent")
    G.add_edge("Case 1", "Petitioner A")
    G.add_edge("Case 1", "Respondent B")
    
    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=15)
    nx.draw_networkx_labels(G, pos, labels, font_size=12)
    plt.title("Knowledge Graph")
    plt.show()

def main():
    st.title("Legal Query Answering with Graph RAG and Gemini API")
    
    query = st.text_input("Enter your query:")
    if query:
        answer = graph_rag(query)
        st.write(f"Query: {query}")
        st.write(f"Answer: {answer}")
        
        st.write("### Knowledge Graph Visualization")
        st.pyplot(visualize_kg())

if __name__ == "__main__":
    main()

driver.close()

