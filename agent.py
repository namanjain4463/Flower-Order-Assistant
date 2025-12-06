from llm import llm
from graph import graph

from tools.cypher import cypher_qa
from tools.cypher import get_bom, create_bill_of_order
from tools.vector import vector_search_colors  # you will create this

# Create a flower order chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert on flower ordering and are collecting orders from customers."),
        ("human", "{input}"),
    ]
)

flower_order = chat_prompt | llm | StrOutputParser()
# Create a set of tools
from langchain.tools import Tool

tools = [
    Tool.from_function(
        name="General Chat",
        description="General conversation with customer",
        func=flower_order.invoke,
    ),
    Tool.from_function(
        name="Generate_BOM",
        description="Generate bill of materials from a flower order using Neo4j.",
        func=lambda order_list: get_bom(graph, order_list),
    ),
    Tool.from_function(
        name="Generate_Bill_Of_Order",
        description="Convert BOM into a CSV bill of order for mechanical arm.",
        func=lambda bom: create_bill_of_order(bom),
    ),
    Tool.from_function(
        name="SemanticColorSearch",
        description="Use embedding-based vector search to match user text to flower colors.",
        func=vector_search_colors,
    )
]
# Create chat history callback
from langchain_neo4j import Neo4jChatMessageHistory

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)
# Create the agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain import hub

agent_prompt = ChatPromptTemplate.from_template("""
You are a flower-ordering expert following this process:

1. When user provides a natural-language flower order, use SemanticColorSearch
   to identify colors mentioned.
2. Use the LLM to extract quantities for each detected color.
3. Construct an order list like:
   [color: <str>, quantity: <int>]
4. Call Generate_BOM to retrieve pickup order and dropoff location.
5. Present the Bill of Materials and ask the user for confirmation.
6. If the user confirms with "yes", call Generate_Bill_Of_Order to create a CSV file.


TOOLS:
------

You have access to the following tools:

{tools}

{tool_names}

To use a tool, follow this format:

```
Thought: Do I need to use a tool? Yes
Action: <tool name>
Action Input: <JSON input>
```
When you have a final answer:
```
Thought: Do I need a tool? No
Final Answer: <answer>             
```
Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}                           
""")

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True
)

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)
# Create a handler to call the agent
from utils import get_session_id

def generate_response(user_input):
    session_id = get_session_id()

    # detect confirmation
    if user_input.lower().strip() in ["yes", "y", "confirm", "correct"]:
        response = chat_agent.invoke(
            {"input": "CONFIRM_ORDER"},
            {"configurable": {"session_id": session_id}},
        )
    else:
        response = chat_agent.invoke(
            {"input": user_input},
            {"configurable": {"session_id": session_id}},
        )

    return response["output"]