# A2A RAG Microservice Demo (Complete Solution)

This document contains the complete guide and source code for a decoupled RAG (Retrieval-Augmented Generation) application, accessible via an A2A-compliant interface. The system is built as a microservice architecture.

## Overview

The architecture consists of four separate components:

- **A2A Server (a2a_server.py):** The public endpoint listening on port 8000. It speaks the A2A protocol, manages the lifecycle of tasks, and orchestrates requests to the internal services.  
- **Adapter Service (adapter_service.py):** An internal microservice on port 8001. Its sole purpose is to translate between the A2A data format and the legacy bot's format.  
- **Legacy Bot Service (legacy_bot_service.py):** The core logic, a RAG bot listening on port 8002. It answers questions using LangChain and LangGraph.  
- **Client (client.py):** An interactive command-line application to communicate with the A2A Server and conduct a conversation.

Data flow:

- Client → A2A Server (HTTP, A2A Protocol)  
- A2A Server → Adapter Service (JSON-RPC)  
- Adapter Service → Legacy RAG Bot (JSON-RPC)  
- Legacy Bot → OpenAI API (internal call)

## Setup and Execution

Follow these steps to run the entire system.

### 1. Project Structure

Ensure you have the following four Python files in one directory:

- a2a_server.py  
- adapter_service.py  
- legacy_bot_service.py  
- client.py

### 2. Create .env File

Create a file named `.env` in the same directory and add your OpenAI API key:

OPENAI_API_KEY="sk-..."

### 3. Install Dependencies (with uv)

Open a terminal in your project directory and run the following commands:

Create a virtual environment:  
uv venv

Activate the environment (Linux/macOS):  
source .venv/bin/activate

(Windows alternative: .venv\Scripts\activate)

Install required packages:  
uv add "uvicorn[standard]" fastapi langchain-openai langchain-core httpx a2a-sdk langchain-community chromadb langgraph python-dotenv

### 4. Start the Services

You will need three separate terminals to run the services in parallel.

**Terminal 1 (Legacy RAG Bot):**  
uv run python legacy_bot_service.py

**Terminal 2 (Adapter Service):**  
uv run python adapter_service.py

**Terminal 3 (A2A Server):**  
uv run python a2a_server.py

### 5. Start the Client

Open a fourth terminal:

uv run python client.py

### Example Interaction

Start a conversation with the bot:

- "Who is the owner of Bella Vista?"  
- "What are his opening hours?"