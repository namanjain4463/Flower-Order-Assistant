from llm import llm
from graph import graph  # if storing vectors in Neo4j, otherwise load from your vector DB

def vector_search_colors(user_text):
    """
    Returns the top flower color semantic matches
    stored in your vector store.
    """
    query = """
    CALL db.index.vector.queryNodes('flowerColorIndex', 5, $embedding)
    YIELD node, score
    RETURN node.color AS color, score
    """
    embed = llm.embed_query(user_text)

    result = graph.query(query, {"embedding": embed})
    return result
