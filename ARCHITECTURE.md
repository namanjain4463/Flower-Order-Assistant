# System Architecture & Workflow Documentation

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Architecture](#component-architecture)
4. [Data Flow Diagrams](#data-flow-diagrams)
5. [Module Details](#module-details)
6. [Sequence Diagrams](#sequence-diagrams)
7. [Technology Stack](#technology-stack)

---

## System Overview

**Flower Order Assistant** is an AI-powered robotic flower picking and ordering system that combines:

- Natural language processing (LangChain + OpenAI)
- Knowledge graph database (Neo4j)
- Computer vision (YOLO)
- Robotic arm control (UR5 with RTDE)
- Web interface (Streamlit)

### Core Capabilities

- Accept natural language flower orders
- Semantic color matching via vector embeddings
- Generate Bill of Materials (BOM) from knowledge graph
- Create CSV bill of order for robot execution
- Autonomous robotic flower picking and placement
- Real-time position correction using KNN error compensation

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE LAYER                         │
│                        (Streamlit Web App)                          │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      AI AGENT ORCHESTRATION                          │
│              (LangChain ReAct Agent + Chat History)                 │
│                                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │
│  │ General Chat │  │ Vector Search│  │ BOM Generator│             │
│  │     Tool     │  │     Tool     │  │     Tool     │             │
│  └──────────────┘  └──────────────┘  └──────────────┘             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌────────────────┐   ┌──────────────────┐
│   LLM Layer   │   │  Knowledge      │   │  Robot Control   │
│  (OpenAI API) │   │  Graph (Neo4j)  │   │  Layer (UR5)     │
│               │   │                 │   │                  │
│ - Chat Model  │   │ - Cypher Queries│   │ - RTDE Control   │
│ - Embeddings  │   │ - Vector Index  │   │ - YOLO Vision    │
│               │   │ - Chat History  │   │ - MLP Joints     │
└───────────────┘   └────────────────┘   └──────────────────┘
```

---

## Component Architecture

### 1. Frontend Layer (`bot.py`)

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT APP                        │
│                                                         │
│  ┌─────────────────────────────────────────────────┐  │
│  │         Initialization & Setup                  │  │
│  │  - Validate secrets (API keys, DB credentials)  │  │
│  │  - Import agent & utilities                     │  │
│  └─────────────────────────────────────────────────┘  │
│                         │                              │
│  ┌─────────────────────────────────────────────────┐  │
│  │         Session State Management                │  │
│  │  - messages: List[Dict] - chat history          │  │
│  │  - pending_bom: str - current BOM text          │  │
│  │  - last_csv_path: str - last order CSV path     │  │
│  └─────────────────────────────────────────────────┘  │
│                         │                              │
│  ┌─────────────────────────────────────────────────┐  │
│  │         User Interaction Handler                │  │
│  │  - handle_submit(message)                       │  │
│  │    ├─> Generate response from agent             │  │
│  │    ├─> Detect BOM in response                   │  │
│  │    ├─> Handle user confirmation                 │  │
│  │    └─> Extract CSV path from response           │  │
│  └─────────────────────────────────────────────────┘  │
│                         │                              │
│  ┌─────────────────────────────────────────────────┐  │
│  │         Display Layer                           │  │
│  │  - Chat message rendering                       │  │
│  │  - CSV download button                          │  │
│  └─────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 2. Agent Layer (`agent.py`)

```
┌──────────────────────────────────────────────────────────────┐
│                    LANGCHAIN AGENT                           │
│                                                              │
│  ┌────────────────────────────────────────────────────┐     │
│  │           ReAct Agent Executor                     │     │
│  │  - Strategy: Thought → Action → Observation        │     │
│  │  - Iterative tool use until final answer           │     │
│  └────────────────────────────────────────────────────┘     │
│                           │                                  │
│  ┌────────────────────────┴────────────────────────────┐    │
│  │                    TOOLS                            │    │
│  │                                                     │    │
│  │  1. General Chat Tool                              │    │
│  │     └─> flower_order chain (LLM + StrOutputParser) │    │
│  │                                                     │    │
│  │  2. SemanticColorSearch                            │    │
│  │     └─> vector_search_colors()                     │    │
│  │         └─> Neo4j vector index query               │    │
│  │                                                     │    │
│  │  3. Generate_BOM                                   │    │
│  │     └─> get_bom(graph, order_list)                 │    │
│  │         └─> Cypher query for pickup/dropoff        │    │
│  │                                                     │    │
│  │  4. Generate_Bill_Of_Order                         │    │
│  │     └─> create_bill_of_order(bom)                  │    │
│  │         ├─> Create CSV file                        │    │
│  │         └─> Launch robot subprocess                │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│  ┌────────────────────────┴────────────────────────────┐    │
│  │         Message History (Neo4j)                     │    │
│  │  - Neo4jChatMessageHistory per session              │    │
│  │  - RunnableWithMessageHistory wrapper               │    │
│  └─────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

### 3. LLM & Knowledge Graph Layer (`llm.py`, `graph.py`, `vector.py`, `cypher.py`)

```
┌───────────────────────────────────────────────────────────────┐
│                  LLM LAYER (llm.py)                           │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  ChatOpenAI                                         │     │
│  │  - Model: Configured via secrets                    │     │
│  │  - API Key: From Streamlit secrets                  │     │
│  └─────────────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  OpenAIEmbeddings                                   │     │
│  │  - Used for vector search queries                   │     │
│  └─────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│               KNOWLEDGE GRAPH (graph.py)                      │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  Neo4jGraph                                         │     │
│  │  - URI: Neo4j Aura instance                         │     │
│  │  - Credentials: From Streamlit secrets              │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                               │
│  Graph Schema:                                                │
│  ┌─────────┐  FOLLOWS_ORDER  ┌───────┐                       │
│  │  Color  │─────────────────▶│ Order │                       │
│  └─────────┘                  └───────┘                       │
│       │                                                        │
│       │ ENDS_AT                                                │
│       ▼                                                        │
│  ┌──────────────┐                                             │
│  │ EndLocation  │                                             │
│  └──────────────┘                                             │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│              VECTOR SEARCH (vector.py)                        │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  vector_search_colors(user_text)                    │     │
│  │  1. Generate embedding from user text               │     │
│  │  2. Query Neo4j vector index 'flowerColorIndex'     │     │
│  │  3. Return top 5 matches with scores                │     │
│  └─────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────┐
│             CYPHER OPERATIONS (cypher.py)                     │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  GraphCypherQAChain                                 │     │
│  │  - Converts natural language to Cypher queries      │     │
│  └─────────────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  get_bom(graph, order_list)                         │     │
│  │  - Input: [{color, quantity}, ...]                  │     │
│  │  - Cypher: MATCH Color→Order, Color→EndLocation     │     │
│  │  - Output: [{color, Quantity, PickupOrder,          │     │
│  │              DropoffLocation}, ...]                 │     │
│  └─────────────────────────────────────────────────────┘     │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  create_bill_of_order(bom)                          │     │
│  │  1. Create CSV with UUID filename                   │     │
│  │  2. Prepare robot input JSON                        │     │
│  │  3. Launch subprocess: python robot_executor.py     │     │
│  │  4. Read robot output JSON                          │     │
│  │  5. Return combined log + unavailable items         │     │
│  └─────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────┘
```

### 4. Robot Execution Layer (`robot_executor.py`)

```
┌────────────────────────────────────────────────────────────────┐
│                  ROBOT EXECUTOR (Subprocess)                   │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  1. CONFIGURATION                                    │     │
│  │     - UR5 Robot IP: 169.254.152.222                  │     │
│  │     - Camera: 1280x720 USB camera                    │     │
│  │     - Servo: GPIO Pin 12 (gripper)                   │     │
│  │     - Models: YOLO + MLP joint predictor             │     │
│  │     - Drop positions: A, B, C, D, E                  │     │
│  └──────────────────────────────────────────────────────┘     │
│                           │                                    │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  2. MODEL LOADING                                    │     │
│  │     ┌────────────────────────────────────────┐       │     │
│  │     │ YOLOtoJointMLP (PyTorch)               │       │     │
│  │     │  - Input: Normalized Y position (1D)   │       │     │
│  │     │  - Output: 6 joint angles              │       │     │
│  │     │  - Layers: 1→32→128→16→6               │       │     │
│  │     └────────────────────────────────────────┘       │     │
│  │     ┌────────────────────────────────────────┐       │     │
│  │     │ YOLO Object Detection                  │       │     │
│  │     │  - Model: best_yolo_CLEAN.pt           │       │     │
│  │     │  - Detects: Flower colors + bboxes     │       │     │
│  │     └────────────────────────────────────────┘       │     │
│  │     ┌────────────────────────────────────────┐       │     │
│  │     │ Scalers (from checkpoint)              │       │     │
│  │     │  - x_scaler_mean, x_scaler_scale       │       │     │
│  │     │  - y_scaler_mean, y_scaler_scale       │       │     │
│  │     └────────────────────────────────────────┘       │     │
│  └──────────────────────────────────────────────────────┘     │
│                           │                                    │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  3. KNN ERROR COMPENSATOR                            │     │
│  │     - Stores position corrections in CSV             │     │
│  │     - K=3 nearest neighbors                          │     │
│  │     - Features: (feature_x, feature_y)               │     │
│  │     - Corrections: (diff_x, diff_y, diff_z)          │     │
│  └──────────────────────────────────────────────────────┘     │
│                           │                                    │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  4. EXECUTION LOGIC: execute_flower_order()          │     │
│  │                                                      │     │
│  │  FOR EACH item in BOM:                               │     │
│  │    1. Move to camera position                        │     │
│  │    2. Open gripper                                   │     │
│  │    3. Capture image                                  │     │
│  │    4. Run YOLO detection                             │     │
│  │    5. Filter for target color                        │     │
│  │    6. Select target (max Y position)                 │     │
│  │    7. Predict joint angles via MLP                   │     │
│  │    8. Apply KNN position correction                  │     │
│  │    9. Move to target position                        │     │
│  │   10. Lower gripper and grab                         │     │
│  │   11. Log position error                             │     │
│  │   12. Move to drop location                          │     │
│  │   13. Release flower                                 │     │
│  │                                                      │     │
│  │  RETURN: Execution log + unavailable items           │     │
│  └──────────────────────────────────────────────────────┘     │
│                           │                                    │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  5. SUBPROCESS INTERFACE                             │     │
│  │     INPUT:  robot_input.json (BOM list)              │     │
│  │     OUTPUT: robot_output.json                        │     │
│  │             {log: str, unavailable: list}            │     │
│  └──────────────────────────────────────────────────────┘     │
└────────────────────────────────────────────────────────────────┘
```

### 5. Utility Layer (`utils.py`)

```
┌─────────────────────────────────────────────────────┐
│                 UTILITIES                           │
│                                                     │
│  ┌───────────────────────────────────────────┐     │
│  │  write_message(role, content, save)       │     │
│  │  - Saves to session state                 │     │
│  │  - Renders in Streamlit chat UI           │     │
│  └───────────────────────────────────────────┘     │
│                                                     │
│  ┌───────────────────────────────────────────┐     │
│  │  get_session_id()                         │     │
│  │  - Returns current Streamlit session ID   │     │
│  │  - Used for chat history isolation        │     │
│  └───────────────────────────────────────────┘     │
│                                                     │
│  ┌───────────────────────────────────────────┐     │
│  │  extract_quantities_llm(text, colors)     │     │
│  │  - Uses LLM to extract quantities         │     │
│  │  - Returns JSON: [{color, quantity}]      │     │
│  └───────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
```

---

## Data Flow Diagrams

### Complete Order Processing Flow

```
┌──────┐
│ User │
└──┬───┘
   │ "I want 5 red and 3 white flowers"
   ▼
┌────────────────────────────────────┐
│  Streamlit UI (bot.py)             │
│  - Captures input                  │
│  - Calls handle_submit()           │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Agent (agent.py)                  │
│  - generate_response()             │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  ReAct Agent Loop                  │
│  Thought: Need to identify colors  │
└────────────┬───────────────────────┘
             │
             ▼ Action: SemanticColorSearch
┌────────────────────────────────────┐
│  Vector Search (vector.py)         │
│  1. Embed user text via OpenAI     │
│  2. Query Neo4j vector index       │
│  3. Return: ["red", "white"]       │
└────────────┬───────────────────────┘
             │ Observation: Colors found
             ▼
┌────────────────────────────────────┐
│  Agent continues                   │
│  Thought: Extract quantities       │
│  Uses LLM to parse: red=5, white=3 │
└────────────┬───────────────────────┘
             │
             ▼ Action: Generate_BOM
┌────────────────────────────────────┐
│  Cypher Query (cypher.py)          │
│  get_bom(graph, order_list)        │
│  Executes:                         │
│    MATCH (c:Color)-[r]->(o:Order)  │
│    MATCH (c:Color)-[s]->(e:End...) │
│  Returns:                          │
│    [{color: "red",                 │
│      Quantity: 5,                  │
│      PickupOrder: "1",             │
│      DropoffLocation: "A"},        │
│     {color: "white", ...}]         │
└────────────┬───────────────────────┘
             │ Observation: BOM generated
             ▼
┌────────────────────────────────────┐
│  Agent Final Answer                │
│  "Here's your BOM... Confirm?"     │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Streamlit displays BOM            │
│  Sets: pending_bom = response      │
└────────────┬───────────────────────┘
             │
             │ User: "yes"
             ▼
┌────────────────────────────────────┐
│  Streamlit handles confirmation    │
│  Calls: generate_response("yes")   │
└────────────┬───────────────────────┘
             │
             ▼ Action: Generate_Bill_Of_Order
┌────────────────────────────────────┐
│  create_bill_of_order (cypher.py)  │
│  1. Write CSV file                 │
│  2. Write robot_input.json         │
│  3. subprocess.run([               │
│       "python3",                   │
│       "robot_executor.py",         │
│       "robot_input.json",          │
│       "robot_output.json"          │
│     ])                             │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Robot Executor (robot_executor.py)│
│  - Loads YOLO + MLP models         │
│  - Connects to UR5 robot           │
│  - FOR EACH flower in BOM:         │
│    ├─ Move to camera position      │
│    ├─ Capture image                │
│    ├─ Detect flowers with YOLO     │
│    ├─ Predict joints with MLP      │
│    ├─ Apply KNN correction         │
│    ├─ Pick flower                  │
│    └─ Drop at location             │
│  - Write robot_output.json         │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  cypher.py reads output JSON       │
│  Returns: CSV path + robot log     │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Agent returns final response      │
│  "CSV created... Robot log..."     │
└────────────┬───────────────────────┘
             │
             ▼
┌────────────────────────────────────┐
│  Streamlit displays result         │
│  Shows download button for CSV     │
└────────────────────────────────────┘
```

### Robot Vision & Control Pipeline

```
┌────────────────────────────────────────────────────────────┐
│               ROBOT PICKING CYCLE                          │
└────────────────────────────────────────────────────────────┘

1. POSITIONING
   ┌──────────────┐
   │ Move to      │
   │ P_CAM joints │ ────▶ UR5 moves to overhead camera position
   └──────────────┘

2. IMAGE CAPTURE
   ┌──────────────┐
   │ USB Camera   │ ────▶ Capture 1280x720 image
   │ cv2.VideoC.. │       Save as snap_HHMMSS.jpg
   └──────────────┘

3. DETECTION
   ┌──────────────────────────────────────────┐
   │ YOLO Model (best_yolo_CLEAN.pt)          │
   │ - Input: Image                           │
   │ - Output: Bounding boxes (xywh)          │
   │           Color labels                   │
   │           Confidence scores               │
   └──────────┬───────────────────────────────┘
              │
              ▼
   ┌──────────────────────────────────────────┐
   │ Filter & Sort                            │
   │ - Keep only target color                 │
   │ - Sort by Y position (descending)        │
   │ - Select: max(y_pos) = closest flower    │
   └──────────┬───────────────────────────────┘
              │
              ▼
4. JOINT PREDICTION
   ┌──────────────────────────────────────────┐
   │ YOLOtoJointMLP                           │
   │ Input:  Normalized Y position [0-1]      │
   │ Process:                                 │
   │   yn_normalized = (yn - x_mean) / x_scale│
   │   joints_pred = model(yn_normalized)     │
   │   joints = joints_pred * y_scale + y_mean│
   │ Output: [j1, j2, j3, j4, j5, j6] radians │
   └──────────┬───────────────────────────────┘
              │
              ▼
5. ERROR CORRECTION
   ┌──────────────────────────────────────────┐
   │ KNN Error Compensator                    │
   │ - Query: feature_x, feature_y            │
   │ - Find k=3 nearest neighbors             │
   │ - Average corrections: Δx, Δy, Δz        │
   │ - Apply offset to TCP position           │
   └──────────┬───────────────────────────────┘
              │
              ▼
6. PICK SEQUENCE
   ┌──────────────────────────────────────────┐
   │ 1. servo.value = -0.14  (open gripper)   │
   │ 2. moveJ(predicted_joints)               │
   │ 3. moveL(tcp + knn_offset)               │
   │ 4. moveL(tcp.z -= 0.065)  (descend)      │
   │ 5. servo.max()  (close gripper)          │
   │ 6. moveL(tcp.z += 0.15)  (lift)          │
   │ 7. Log position error for KNN update     │
   └──────────┬───────────────────────────────┘
              │
              ▼
7. DROP SEQUENCE
   ┌──────────────────────────────────────────┐
   │ 1. moveJ(DROP_POSITIONS[location])       │
   │ 2. servo.mid()  (release)                │
   │ 3. Record result                         │
   └──────────────────────────────────────────┘
```

---

## Module Details

### `bot.py` - Streamlit Web Interface

**Purpose**: User-facing chat interface

**Key Features**:

- Defensive initialization with secret validation
- Session state management for chat history
- BOM pending state for confirmation flow
- CSV download functionality
- Error handling and display

**Session State Variables**:

- `messages`: Chat message history
- `pending_bom`: Current BOM awaiting confirmation
- `last_csv_path`: Path to most recent CSV file

---

### `agent.py` - LangChain Agent Orchestrator

**Purpose**: Coordinates all tools and manages conversation flow

**Agent Type**: ReAct (Reasoning + Acting)

**Agent Prompt Strategy**:

1. Identify colors via SemanticColorSearch
2. Extract quantities using LLM
3. Build order list: `[{color, quantity}, ...]`
4. Call Generate_BOM for pickup/dropoff info
5. Present BOM and await confirmation
6. On confirmation, call Generate_Bill_Of_Order

**Tools**:

- **General Chat**: Standard conversation
- **SemanticColorSearch**: Embedding-based color matching
- **Generate_BOM**: Neo4j Cypher query for order details
- **Generate_Bill_Of_Order**: CSV creation + robot execution

**Message History**: Stored in Neo4j per session ID

---

### `llm.py` - Language Model Configuration

**Components**:

- `ChatOpenAI`: Main conversational model
- `OpenAIEmbeddings`: Vector embeddings for semantic search

**Configuration**: Loaded from Streamlit secrets

---

### `graph.py` - Neo4j Knowledge Graph

**Connection**: Neo4j Aura cloud instance

**Graph Schema**:

```cypher
(Color)-[:FOLLOWS_ORDER]->(Order)
(Color)-[:ENDS_AT]->(EndLocation)
```

**Indices**:

- `flowerColorIndex`: Vector index on Color nodes

---

### `vector.py` - Semantic Search

**Function**: `vector_search_colors(user_text)`

**Process**:

1. Generate embedding from user text
2. Query Neo4j vector index
3. Return top 5 color matches with similarity scores

**Use Case**: Fuzzy matching "crimson" → "red"

---

### `cypher.py` - Database Operations & Robot Integration

**Key Functions**:

1. **`GraphCypherQAChain`**: Natural language to Cypher conversion

2. **`get_bom(graph, order_list)`**:

   - Input: `[{color: str, quantity: int}, ...]`
   - Cypher: Matches Color → Order, Color → EndLocation
   - Output: BOM with pickup orders and dropoff locations

3. **`create_bill_of_order(bom)`**:
   - Creates CSV file with UUID name
   - Launches robot executor as subprocess
   - Reads robot output JSON
   - Returns combined execution log

**Subprocess Isolation**: Prevents PyTorch/Streamlit conflicts

---

### `robot_executor.py` - Robotic Control System

**Hardware**:

- **Robot**: Universal Robots UR5
- **Camera**: USB camera (1280x720)
- **Gripper**: Servo on GPIO pin 12
- **Communication**: RTDE (Real-Time Data Exchange)

**Models**:

1. **YOLO (`best_yolo_CLEAN.pt`)**:

   - Detects flower colors
   - Returns bounding boxes (normalized xywh)

2. **MLP (`flower_joint_model_CLEAN.pth`)**:

   - Architecture: 1→32→128→16→6
   - Input: Normalized Y position
   - Output: 6 joint angles (radians)
   - Includes StandardScaler parameters

3. **KNN Error Compensator**:
   - Stores historical position corrections
   - Uses k=3 nearest neighbors
   - Improves accuracy over time

**Main Function**: `execute_flower_order(bom_list)`

**Robot Positions**:

- `P_CAM`: Camera/detection position
- `DROP_POSITIONS`: 5 drop locations (A-E)

**Error Handling**:

- Tracks unavailable items
- Logs failures per color
- Returns detailed execution report

---

### `utils.py` - Helper Functions

**Functions**:

- `write_message()`: Dual save/display to session state & UI
- `get_session_id()`: Retrieves Streamlit session identifier
- `extract_quantities_llm()`: LLM-based quantity extraction (currently unused in favor of agent's internal processing)

---

## Sequence Diagrams

### User Order to Robot Execution

```
User          Streamlit       Agent         Neo4j         Subprocess      UR5 Robot
 │                │             │             │                │              │
 │─"5 red, 3 white"─▶           │             │                │              │
 │                │             │             │                │              │
 │                │─generate_response()──▶    │                │              │
 │                │             │             │                │              │
 │                │             │─vector_search_colors()───▶   │              │
 │                │             │             │                │              │
 │                │             │◀────["red","white"]─────────│              │
 │                │             │             │                │              │
 │                │             │─get_bom()───▶               │              │
 │                │             │             │                │              │
 │                │             │◀────BOM─────│               │              │
 │                │             │             │                │              │
 │                │◀────"BOM + Confirm?"──────│               │              │
 │                │             │             │                │              │
 │◀────Display BOM──            │             │                │              │
 │                │             │             │                │              │
 │─────"yes"─────▶             │             │                │              │
 │                │             │             │                │              │
 │                │─generate_response("yes")─▶│                │              │
 │                │             │             │                │              │
 │                │             │─create_bill_of_order()       │              │
 │                │             │  │                           │              │
 │                │             │  │─Write CSV                 │              │
 │                │             │  │                           │              │
 │                │             │  │─subprocess.run()────▶     │              │
 │                │             │  │                           │              │
 │                │             │  │                      ┌────▼──────┐       │
 │                │             │  │                      │Load Models│       │
 │                │             │  │                      └────┬──────┘       │
 │                │             │  │                           │              │
 │                │             │  │                      ┌────▼──────┐       │
 │                │             │  │                      │Connect UR5│───────▶
 │                │             │  │                      └────┬──────┘       │
 │                │             │  │                           │              │
 │                │             │  │                      ┌────▼──────┐       │
 │                │             │  │                      │Move P_CAM │◀──────│
 │                │             │  │                      └────┬──────┘       │
 │                │             │  │                           │              │
 │                │             │  │                      ┌────▼──────┐       │
 │                │             │  │                      │Capture Img│       │
 │                │             │  │                      └────┬──────┘       │
 │                │             │  │                           │              │
 │                │             │  │                      ┌────▼──────┐       │
 │                │             │  │                      │YOLO Detect│       │
 │                │             │  │                      └────┬──────┘       │
 │                │             │  │                           │              │
 │                │             │  │                      ┌────▼──────┐       │
 │                │             │  │                      │MLP Predict│       │
 │                │             │  │                      └────┬──────┘       │
 │                │             │  │                           │              │
 │                │             │  │                      ┌────▼──────┐       │
 │                │             │  │                      │Pick Flower│───────▶
 │                │             │  │                      └────┬──────┘       │
 │                │             │  │                           │              │
 │                │             │  │                      ┌────▼──────┐       │
 │                │             │  │                      │Drop Flower│───────▶
 │                │             │  │                      └────┬──────┘       │
 │                │             │  │                           │              │
 │                │             │  │                      (Repeat for all)    │
 │                │             │  │                           │              │
 │                │             │  │                      ┌────▼──────┐       │
 │                │             │  │                      │Write JSON │       │
 │                │             │  │                      └────┬──────┘       │
 │                │             │  │                           │              │
 │                │             │  │◀──────robot_output.json───│              │
 │                │             │  │                                          │
 │                │             │◀─┘ Return log                               │
 │                │             │                                             │
 │                │◀────"CSV created + log"──────────────────────             │
 │                │                                                           │
 │◀────Display result + Download button                                      │
 │                                                                            │
```

---

## Technology Stack

### **Frontend**

- **Streamlit** 1.35.0 - Web UI framework
- **Python** 3.x

### **AI & NLP**

- **LangChain** 0.3.9 - Agent orchestration
- **LangChain-OpenAI** 0.2.10 - OpenAI integration
- **OpenAI API** 1.56.0 - LLM & embeddings
- **LangChain-Neo4j** 0.1.1 - Graph integration
- **LangChainHub** 0.1.21 - Prompt templates

### **Database**

- **Neo4j** 5.27.0 - Knowledge graph
- **Neo4j Aura** - Cloud-hosted instance
- Vector indices for semantic search

### **Computer Vision**

- **YOLO** (Ultralytics) - Object detection
- **OpenCV** (cv2) - Image processing
- **PyTorch** - Neural network inference

### **Robotics**

- **UR5 Robot** (Universal Robots)
- **RTDE** (Real-Time Data Exchange) - Robot communication
- **rtde_control** - Motion control
- **rtde_receive** - State feedback
- **gpiozero** - Servo control (Raspberry Pi)

### **Data Processing**

- **NumPy** - Numerical operations
- **Pandas** - CSV handling, error logging

### **Other**

- **UUID** - Unique file naming
- **JSON** - Data interchange
- **Subprocess** - Process isolation

---

## Key Design Patterns

### 1. **Process Isolation**

Robot execution runs in subprocess to prevent PyTorch/Streamlit conflicts

### 2. **Agent-Tool Architecture**

ReAct agent dynamically selects tools based on reasoning

### 3. **Knowledge Graph Integration**

Neo4j stores relationships between colors, orders, and locations

### 4. **Semantic Search**

Vector embeddings enable fuzzy color matching

### 5. **Error Compensation**

KNN-based learning improves robot accuracy over time

### 6. **Session Isolation**

Each user session has independent chat history in Neo4j

### 7. **Defensive Programming**

Extensive error handling and validation throughout

---

## Workflow Summary

1. **User Input** → Natural language order
2. **Vector Search** → Match colors semantically
3. **LLM Extraction** → Parse quantities
4. **Graph Query** → Fetch pickup/dropoff from Neo4j
5. **User Confirmation** → Present BOM, await "yes"
6. **CSV Generation** → Create bill of order file
7. **Subprocess Launch** → Execute robot_executor.py
8. **Robot Execution**:
   - Load YOLO & MLP models
   - Connect to UR5
   - For each flower: detect → predict joints → pick → drop
   - Log results
9. **Result Display** → Show execution log + unavailable items
10. **CSV Download** → Provide download button

---

## File Structure

```
chatbot/
├── bot.py                        # Streamlit web interface
├── agent.py                      # LangChain agent orchestration
├── llm.py                        # OpenAI LLM & embeddings
├── graph.py                      # Neo4j connection
├── vector.py                     # Semantic color search
├── cypher.py                     # Cypher queries & BOM generation
├── robot_executor.py             # UR5 robot control & vision
├── utils.py                      # Helper functions
├── test_connection.py            # Neo4j connection test
├── requirements.txt              # Python dependencies
├── flower_joint_model_CLEAN.pth  # MLP joint predictor model
├── best_yolo_CLEAN.pt            # YOLO detection model (not in listing)
├── xyz_error_log.csv             # KNN error compensation log (generated)
├── bill_of_order_*.csv           # Generated order files
├── robot_input.json              # Subprocess input (temporary)
├── robot_output.json             # Subprocess output (temporary)
└── snap_*.jpg                    # Captured images (temporary)
```

---

## Future Enhancements

1. **Multi-color bouquet support** - Handle complex arrangements
2. **Real-time video feedback** - Live camera stream in UI
3. **Order queue management** - Multiple concurrent orders
4. **Advanced error recovery** - Retry logic for failed picks
5. **Performance metrics** - Success rate, timing analytics
6. **3D visualization** - Robot workspace visualization
7. **Voice input** - Speech-to-text ordering
8. **Inventory management** - Track flower availability

---

_Last Updated: December 6, 2025_
