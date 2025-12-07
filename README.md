# Flower Order Assistant 🌸🤖

[![GitHub](https://img.shields.io/badge/GitHub-Flower--Order--Assistant-blue?logo=github)](https://github.com/namanjain4463/Flower-Order-Assistant)
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.9-green)](https://www.langchain.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.27.0-blue)](https://neo4j.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red)](https://streamlit.io/)

An AI-powered robotic flower picking and ordering system that combines natural language processing, computer vision, and robotic control to autonomously fulfill flower orders.

## 🎯 Overview

This system allows users to place flower orders in natural language through a chatbot web interface. The AI agent processes the order, generates a bill of materials from a Neo4j knowledge graph, and controls a UR5 robotic arm to physically pick and arrange the flowers.

**🔗 Repository**: [github.com/namanjain4463/Flower-Order-Assistant](https://github.com/namanjain4463/Flower-Order-Assistant)

### Key Features

- 🤖 **AI-Powered Agent**: LangChain ReAct agent with OpenAI GPT for natural language understanding
- 🔍 **Semantic Search**: Vector-based color matching for flexible order interpretation
- 🗄️ **Knowledge Graph**: Neo4j database for order relationships and pickup/dropoff logic
- 👁️ **Computer Vision**: YOLO object detection for flower identification
- 🦾 **Robotic Control**: UR5 arm with real-time position correction
- 💬 **Web Interface**: Clean Streamlit chat interface with session history

## 📋 Table of Contents

- [System Requirements](#system-requirements)
- [Hardware Requirements](#hardware-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [Architecture](#architecture)
- [Troubleshooting](#troubleshooting)
- [Development](#development)

## 🖥️ System Requirements

### Software Dependencies

- **Python**: 3.8 or higher
- **Operating System**: Linux (tested on Raspberry Pi OS/Ubuntu)
- **Neo4j**: Aura cloud instance or local Neo4j 5.x
- **OpenAI API**: Valid API key with GPT-4 access

### Python Packages

See [requirements.txt](requirements.txt) for the complete list. Main dependencies:

- `streamlit==1.35.0` - Web interface
- `langchain==0.3.9` - Agent orchestration
- `langchain-openai==0.2.10` - OpenAI integration
- `langchain-neo4j==0.1.1` - Neo4j integration
- `neo4j==5.27.0` - Database driver
- `openai==1.56.0` - OpenAI API client
- `ultralytics` - YOLO object detection
- `torch` - Neural network inference
- `opencv-python` - Image processing
- `pandas` - Data handling
- `numpy` - Numerical operations

## 🔧 Hardware Requirements

### Required Hardware

1. **Universal Robots UR5** robotic arm

   - IP Address: 169.254.152.222 (configurable in `robot_executor.py`)
   - RTDE interface enabled

2. **USB Camera**

   - Resolution: 1280x720 or higher
   - V4L2 compatible (Linux)

3. **Servo Gripper**

   - Connected to GPIO pin 12
   - Compatible with `gpiozero` library

4. **Raspberry Pi or Linux Computer**
   - For robot control and GPIO access
   - Connected to same network as UR5

### Optional Hardware

- **Monitor/Display** - For viewing robot camera feed during operation

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/namanjain4463/Flower-Order-Assistant.git
cd Flower-Order-Assistant
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# OR
.\venv\Scripts\Activate.ps1  # On Windows PowerShell
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install System Dependencies (Linux)

For camera and GPIO support:

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv
sudo apt-get install -y v4l-utils
```

For Raspberry Pi GPIO:

```bash
sudo apt-get install -y python3-gpiozero pigpio
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

### 5. Download Model Files

The large model files are included in this repository:

- `flower_joint_model_CLEAN.pth` - MLP joint angle predictor (included)
- `best_yolo_CLEAN.pt` - YOLO flower detection model (you may need to download separately if not present)

> **Note**: If the YOLO model is not included, you'll need to train it using your specific flower dataset or obtain it separately.

## ⚙️ Configuration

### Method 1: Using Streamlit Secrets (Recommended)

Create `.streamlit/secrets.toml` in the project directory:

```bash
mkdir -p .streamlit
```

Edit `.streamlit/secrets.toml`:

```toml
# OpenAI Configuration
OPENAI_API_KEY = "sk-your-openai-api-key-here"
OPENAI_MODEL = "gpt-4"

# Neo4j Configuration
NEO4J_URI = "neo4j+s://your-instance.databases.neo4j.io"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your-neo4j-password"
```

> **Important**: Never commit `secrets.toml` to version control. It's already in `.gitignore`.

### Method 2: Using Environment Variables (Alternative)

Copy the example file and edit it:

```bash
cp .env.example .env
```

Edit `.env` with your credentials:

```bash
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_MODEL=gpt-4
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-neo4j-password
```

> **Important**: Never commit `.env` to version control. It's already in `.gitignore`.

### 2. Neo4j Database Setup

#### Create the Graph Schema

```cypher
// Create Color nodes
CREATE (red:Color {name: 'red'})
CREATE (orange:Color {name: 'orange'})
CREATE (pink:Color {name: 'pink'})
CREATE (purple:Color {name: 'purple'})
CREATE (white:Color {name: 'white'})

// Create Order nodes (pickup sequence)
CREATE (o1:Order {name: '1'})
CREATE (o2:Order {name: '2'})
CREATE (o3:Order {name: '3'})
CREATE (o4:Order {name: '4'})
CREATE (o5:Order {name: '5'})

// Create EndLocation nodes (dropoff positions)
CREATE (locA:EndLocation {name: 'A'})
CREATE (locB:EndLocation {name: 'B'})
CREATE (locC:EndLocation {name: 'C'})
CREATE (locD:EndLocation {name: 'D'})
CREATE (locE:EndLocation {name: 'E'})

// Create relationships (example mapping)
MATCH (c:Color {name: 'red'}), (o:Order {name: '1'})
CREATE (c)-[:FOLLOWS_ORDER]->(o)

MATCH (c:Color {name: 'red'}), (loc:EndLocation {name: 'A'})
CREATE (c)-[:ENDS_AT]->(loc)

// Repeat for other colors...
```

#### Create Vector Index

```cypher
// Create vector index for semantic color search
CALL db.index.vector.createNodeIndex(
  'flowerColorIndex',
  'Color',
  'embedding',
  1536,
  'cosine'
)
```

#### Populate Embeddings

You'll need to generate and store embeddings for each color using OpenAI's embedding model:

```python
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase

embeddings = OpenAIEmbeddings(openai_api_key="your-key")

colors = ['red', 'orange', 'pink', 'purple', 'white']
driver = GraphDatabase.driver(uri, auth=(username, password))

for color in colors:
    embedding = embeddings.embed_query(color)

    with driver.session() as session:
        session.run(
            "MATCH (c:Color {name: $name}) SET c.embedding = $embedding",
            name=color,
            embedding=embedding
        )

driver.close()
```

### 3. Robot Configuration

Edit `robot_executor.py` if needed:

```python
# Line 18-19: Update UR5 IP address
UR_IP = "169.254.152.222"  # Change to your robot's IP

# Lines 26-35: Adjust drop positions (in degrees)
DROP_POSITIONS = {
    'A': [math.radians(x) for x in [80.52, -75.77, 88.12, -101.97, -93.01, 77.96]],
    'B': [math.radians(x) for x in [70.00, -75.77, 88.12, -101.97, -93.01, 77.96]],
    # ... adjust as needed for your workspace
}

# Line 23: Adjust camera position
P_CAM  = [math.radians(x) for x in [-61.24, -89.61, 50.15, -57.46, -92.83, 115.69]]
```

### 4. Camera Configuration

Test camera access:

```bash
# List available cameras
v4l2-ctl --list-devices

# Test camera capture
python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAILED')"
```

If using a different camera device:

```python
# In robot_executor.py, line 148
cap = cv2.VideoCapture(0)  # Change 0 to your camera index
```

## 🚀 Running the Application

### 1. Test Neo4j Connection

```bash
streamlit run test_connection.py
```

Expected output:

```
Connection successful!
```

> **Note**: `test_connection.py` now uses secrets from `.streamlit/secrets.toml` for security.

### 2. Start the Streamlit App

```bash
streamlit run bot.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Using the Application

1. **Place an Order**:

   ```
   I want 5 red and 3 white flowers
   ```

2. **Review BOM**:
   The agent will display a Bill of Materials with pickup orders and dropoff locations.

3. **Confirm**:

   ```
   yes
   ```

4. **Download CSV**:
   After robot execution, download the generated bill of order CSV.

## 📖 Usage Guide

### Example Conversations

**Simple Order**:

```
User: I need 10 purple flowers
Agent: [Searches colors, generates BOM]
      Here's your Bill of Materials:
      - 10 purple flowers
      - Pickup Order: 4
      - Dropoff Location: D

      Please confirm to proceed.
User: yes
Agent: [Creates CSV, executes robot]
      CSV file created at: bill_of_order_abc123.csv
      Robot Execution Summary: Picked 10 purple flowers
```

**Multi-Color Order**:

```
User: 3 red, 2 white, and 5 pink please
Agent: [Processes order]
      Bill of Materials:
      - 3 red (Pickup: 1, Drop: A)
      - 2 white (Pickup: 5, Drop: E)
      - 5 pink (Pickup: 3, Drop: C)

      Confirm?
User: confirm
Agent: [Executes order]
```

**Semantic Matching**:

```
User: I want crimson and ivory flowers
Agent: [Vector search matches crimson→red, ivory→white]
      How many of each color?
```

### Available Colors

- 🔴 Red
- 🟠 Orange
- 🩷 Pink
- 🟣 Purple
- ⚪ White

### Understanding the Output

**Bill of Materials (BOM)** includes:

- `color`: Flower color
- `Quantity`: Number requested
- `PickupOrder`: Sequence number for robot picking
- `DropoffLocation`: Where to place flowers (A-E)

**Robot Execution Log** shows:

- Successful picks
- Skipped items (if flower not found)
- Unavailable items summary

## 🏗️ Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed system architecture, component diagrams, and data flow documentation.

### Quick Architecture Overview

```
User ──▶ Streamlit UI ──▶ LangChain Agent ──▶ Tools
                              │
                              ├─▶ Vector Search (Neo4j)
                              ├─▶ BOM Generator (Cypher)
                              └─▶ Robot Executor (Subprocess)
                                      │
                                      ├─▶ YOLO Detection
                                      ├─▶ MLP Joint Prediction
                                      └─▶ UR5 Control
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Streamlit Shows "Application Setup Failed"

**Cause**: Missing or invalid secrets

**Solution**:

- Check `.streamlit/secrets.toml` exists
- Verify all required keys are present
- Test Neo4j connection separately with `test_connection.py`

#### 2. Robot Subprocess Fails

**Cause**: PyTorch model loading error or robot connection failure

**Solution**:

```bash
# Test robot executor independently
echo '[{"color": "red", "quantity": 1, "pickupOrder": "1", "dropoffLocation": "A"}]' > test_input.json
python3 robot_executor.py test_input.json test_output.json
cat test_output.json
```

#### 3. Camera Not Found

**Cause**: Camera not accessible or wrong index

**Solution**:

```bash
# List cameras
ls /dev/video*

# Test with different index
python3 -c "import cv2; cap = cv2.VideoCapture(1); print(cap.isOpened())"
```

#### 4. Vector Search Returns No Results

**Cause**: Embeddings not populated in Neo4j

**Solution**:

- Run the embedding population script (see Configuration step 2)
- Verify vector index exists:
  ```cypher
  SHOW INDEXES
  ```

#### 5. "Model file not found" Error

**Cause**: Missing `.pth` or `.pt` model files

**Solution**:

- Ensure `flower_joint_model_CLEAN.pth` is in project root
- Ensure `best_yolo_CLEAN.pt` is in project root
- Check file permissions

### Debug Mode

Enable verbose logging:

```python
# In agent.py, line 108
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # Already enabled - shows agent reasoning
    return_intermediate_steps=True,
    handle_parsing_errors=True
)
```

## 🛠️ Development

### Project Structure

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
├── .streamlit/
│   └── secrets.toml              # Configuration (DO NOT COMMIT)
├── README.md                     # This file
└── ARCHITECTURE.md               # System architecture documentation
```

### Adding New Colors

1. **Update Neo4j**:

   ```cypher
   CREATE (c:Color {name: 'blue'})
   MATCH (c:Color {name: 'blue'}), (o:Order {name: '6'})
   CREATE (c)-[:FOLLOWS_ORDER]->(o)
   MATCH (c:Color {name: 'blue'}), (loc:EndLocation {name: 'F'})
   CREATE (c)-[:ENDS_AT]->(loc)
   ```

2. **Add Embedding**:

   ```python
   embedding = embeddings.embed_query('blue')
   # Store in Neo4j
   ```

3. **Update YOLO Model** (if needed):

   - Retrain with blue flower images
   - Export new model

4. **Add Drop Position** (if needed):
   ```python
   # In robot_executor.py
   DROP_POSITIONS['F'] = [math.radians(x) for x in [...]]
   ```

### Testing

```bash
# Test individual components

# 1. Neo4j connection
python3 test_connection.py

# 2. Vector search
python3 -c "from vector import vector_search_colors; print(vector_search_colors('crimson'))"

# 3. LLM connection
python3 -c "from llm import llm; print(llm.invoke('Hello'))"

# 4. Robot (dry run)
# Edit robot_executor.py to add test mode or use JSON files
```

### Environment Variables (Alternative to Secrets)

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4"
export NEO4J_URI="neo4j+s://..."
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="..."

streamlit run bot.py
```

## 📝 Notes

- **Safety**: Ensure robot workspace is clear before execution
- **Calibration**: KNN error compensator improves over time - initial accuracy may vary
- **Concurrency**: Current design handles one order at a time
- **Storage**: CSV files and images accumulate - clean periodically

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Support

For issues and questions:

- 🐛 **Report bugs**: [GitHub Issues](https://github.com/namanjain4463/Flower-Order-Assistant/issues)
- 📖 **Documentation**: Check [Troubleshooting](#troubleshooting) section
- 🏗️ **Architecture**: Review [ARCHITECTURE.md](ARCHITECTURE.md) for system details
- 💬 **Discussions**: [GitHub Discussions](https://github.com/namanjain4463/Flower-Order-Assistant/discussions)

## 🙏 Acknowledgments

- **LangChain** - For the powerful agent framework
- **Neo4j** - For the knowledge graph database
- **OpenAI** - For GPT models and embeddings
- **Universal Robots** - For UR5 robotic arm platform
- **Ultralytics** - For YOLO object detection

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Repository**: [github.com/namanjain4463/Flower-Order-Assistant](https://github.com/namanjain4463/Flower-Order-Assistant)

---

**Version**: 1.0  
**Last Updated**: December 6, 2025  
**Status**: Production Ready
