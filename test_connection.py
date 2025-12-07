from neo4j import GraphDatabase

try:
    uri = "neo4j+s://86285415.databases.neo4j.io" # Your Aura URI
    username = "neo4j"
    password = "Q_m7-uDAJL7ALAyHoA4obsfMOgBfU3mpe717pNu37GI"

    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    print("Connection successful!")
    driver.close()
except Exception as e:
    print(f"Connection failed: {e}")