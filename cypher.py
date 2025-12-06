import streamlit as st
from llm import llm
from graph import graph
import csv
import uuid
import os

# Create the Cypher QA chain
from langchain_neo4j import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate

CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to extract information to put into a bill of materials and bill of order for a flower grabbing arm.
Convert the user's question based on the schema.

Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.

Do not return entire nodes or embedding properties.

Fine Tuning:

First find which colors the user wants to order, and use the FOLLOWS_ORDER and ENDS_AT relationships to access the information to put into the bill of materials and bill of order.
```


```

```

Schema:
{schema}

Question:
{question}
"""

cypher_prompt = PromptTemplate.from_template(CYPHER_GENERATION_TEMPLATE)

cypher_qa = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True,
    return_intermediate_steps=True
)
def get_bom(graph, order_list):
    """
    Returns color, quantity, pickupOrder, and dropoffLocation
    for each color in the order.
    """
    cypher = """
    UNWIND $orderList AS item
    MATCH (c:Color {name: item.color}) -[r:FOLLOWS_ORDER]->(o:Order)
    MATCH (c:Color {name: item.color}) -[s:ENDS_AT]->(e:EndLocation)
    RETURN
        c.name AS color,
        item.quantity AS Quantity,
        o.name AS PickupOrder,
        e.name AS DropoffLocation
    ORDER BY o.name ASC
    """
    result = graph.query(cypher, {"orderList": order_list})
    return result

# In your existing tools/cypher.py file, update the imports

import os
import csv
import uuid
from collections import Counter 
from robot_executor import execute_flower_order 


import os
import csv
import uuid
import json
import subprocess
from collections import Counter 

# NOTE: We NO LONGER import execute_flower_order directly.
# This prevents Streamlit/PyTorch conflicts.

def create_bill_of_order(order_bom):
    """
    1. Creates a CSV Bill of Order.
    2. Executes the flower picking process via a SUBPROCESS.
    3. Returns a combined log.
    """
    if not order_bom:
        return "Order confirmation received, but BOM was empty."

    # --- 1. Create the CSV file ---
    filename = f"bill_of_order_{uuid.uuid4().hex}.csv"
    filepath = os.path.join(os.getcwd(), filename)
    fieldnames = ["pickupOrder", "color", "quantity", "dropoffLocation"]

    try:
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in order_bom:
                writer.writerow(row)
        csv_log = f"Bill of Order created at: {filepath}"
    except Exception as e:
        csv_log = f"ERROR creating CSV: {e}"

    # --- 2. Call the Robot Execution (Via Subprocess) ---
    print("\n[AGENT] Launching robot subprocess...")
    
    # Define temp files for data exchange
    input_json_path = "robot_input.json"
    output_json_path = "robot_output.json"
    
    robot_log = ""
    unavailable_items = []

    try:
        # Write input data
        with open(input_json_path, 'w') as f:
            json.dump(order_bom, f)
            
        # Run the script as a separate process
        # This isolates it from Streamlit's environment
        process = subprocess.run(
            ["python3", "robot_executor.py", input_json_path, output_json_path],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        # Check if the script created the output file
        if os.path.exists(output_json_path):
            with open(output_json_path, 'r') as f:
                result = json.load(f)
                robot_log = result.get("log", "No log returned.")
                unavailable_items = result.get("unavailable", [])
        else:
            robot_log = f"Robot script failed silently.\nStderr: {process.stderr}\nStdout: {process.stdout}"

    except Exception as e:
        robot_log = f"FATAL SUBPROCESS ERROR: {e}"
        unavailable_items = []

    # --- 3. Format Response ---
    unavailable_summary = ""
    if unavailable_items:
        unavailable_counts = Counter(item['color'] for item in unavailable_items)
        list_text = "\n".join([f"- {count} x {color}" for color, count in unavailable_counts.items()])
        unavailable_summary = (
            "\n\n🚨 **ATTENTION:** The robot missed these items:\n"
            f"{list_text}\n"
        )
    
    final_response = (
        f"{csv_log}\n\n"
        f"--- Robot Execution Summary ---\n"
        f"{robot_log}"
        f"{unavailable_summary}"
    )

    return final_response