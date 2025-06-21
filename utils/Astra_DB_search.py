import os
import streamlit as st
import cassio
from langchain.vectorstores.cassandra import Cassandra
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document


def get_astra_vectorstore():
    Astra_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    Astra_DB_ID = os.getenv("ASTRA_DB_ID")
    
    if not Astra_token or not Astra_DB_ID:
        st.error("❌ Astra DB credentials not found. Please set them in your environment variables.")
        print(f"Astra_token: {Astra_token}, Astra_DB_ID: {Astra_DB_ID}")  # Debugging
        return None
    
    try:
        cassio.init(
            database_id=Astra_DB_ID,
            token=Astra_token
        )
        embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Cassandra(
            embedding=embedding,
            table_name="Stock_Report",  # Updated table name
            session=None,
            keyspace=None
        )
    except Exception as e:
        # st.error(f"❌ Failed to initialize vectorstore: {e}")
        print(f"Exception: {e}")  # Debugging
        return None
    
def store_response(vectorstore, response, symbol):
    if not vectorstore:
        st.error("❌ Vectorstore is not initialized.")
        return
    
    # Create a Document with the response
    doc = Document(
        page_content=response,
        metadata={"symbol": symbol}
    )
    
    # Add the document to the vectorstore
    vectorstore.add_documents([doc])
    # st.success("✅ Response stored successfully in Astra DB.")
    
    
def search_response_plain(vectorstore, symbol, k=3):
    if not vectorstore:
        return []
    results = vectorstore.similarity_search(symbol, k=k)
    return [
        {
            "symbol": result.metadata.get("symbol", "N/A"),
            "content": result.page_content
        }
        for result in results
    ]

# 2. Create a tool function for the agent
def astra_search_tool(symbol: str, k: int = 3):
    vectorstore = get_astra_vectorstore()
    results = search_response_plain(vectorstore, symbol, k)
    if not results:
        return f"No results found for symbol: {symbol}"
    return "\n\n".join(
        f"Symbol: {r['symbol']}\nContent: {r['content']}" for r in results
    )
