import streamlit as st
from neo4j import GraphDatabase

try:
    # Load credentials from Streamlit secrets
    uri = st.secrets["NEO4J_URI"]
    username = st.secrets["NEO4J_USERNAME"]
    password = st.secrets["NEO4J_PASSWORD"]

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    print("Connection successful!")
    driver.close()
except Exception as e:
    print(f"Connection failed: {e}")