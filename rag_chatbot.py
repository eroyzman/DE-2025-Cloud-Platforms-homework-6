import streamlit as st
from langchain.llms import Bedrock
from langchain.chains import RetrievalQA
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.embeddings import BedrockEmbeddings
from opensearchpy import RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3

# Initialize AWS session for authentication
session = boto3.Session()
credentials = session.get_credentials()
region = "us-east-1"
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    "aoss",
    session_token=credentials.token
)

# Initialize LLM
llm = Bedrock(model_id="anthropic.claude-v2", region_name="us-east-1")

# Initialize embeddings
embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", region_name="us-east-1")

# Initialize vector store with AWS authentication
vector_db = OpenSearchVectorSearch(
    opensearch_url="https://dashboards.us-east-1.aoss.amazonaws.com/_login/?collectionId=v5ekj0q8u2r16y65ox1m",  # Replace with your OpenSearch collection endpoint
    index_name="company-data-index",
    embedding_function=embeddings,
    is_aoss=True,
    connection_class=RequestsHttpConnection,
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True
)

# Create RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(search_kwargs={"k": 2}),
    return_source_documents=True
)

# Streamlit app
st.title("Company Data RAG Chatbot")
query = st.text_input("Ask a question about Company Data:")
if query:
    result = rag_chain({"query": query})
    st.write("**Answer**:")
    st.write(result["result"])
    st.write("**Citations**:")
    for doc in result["source_documents"]:
        st.write(f"- {doc.metadata['source']} (Page {doc.metadata['page']})")