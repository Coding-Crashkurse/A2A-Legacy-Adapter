# A2A Stateless RAG Microservice Demo

This document provides a complete guide and source code for a decoupled RAG (Retrieval-Augmented Generation) application, accessible via an A2A-compliant interface. The system is built as a microservice architecture with a stateless, synchronous communication model that requires no tasks or polling.

## Overview

The architecture consists of four separate components:

- **A2A Server (`a2a_server.py`)**: The public endpoint listening on port 8000. It speaks the A2A protocol and forwards requests directly to the adapter service to get an immediate response.
- **Adapter Service (`adapter_service.py`)**: An internal microservice on port 8001. Its task is to translate between the A2A message format and the simple JSON-RPC format used by the legacy bot.
- **Legacy Bot Service (`legacy_bot_service.py`)**: The core logic, a RAG bot listening on port 8002. It answers questions using LangChain and LangGraph and has no knowledge of A2A.
- **Client (`client.py`)**: An interactive command-line application that communicates directly with the A2A server and waits for instant responses.

## Data Flow

The data flow is a simple, synchronous request-response cycle:

- Client → A2A Server (A2A Message)  
- A2A Server → Adapter Service (A2A Message)  
- Adapter Service → Legacy RAG Bot (Plain JSON-RPC)  
- Legacy Bot → Adapter Service (Response)  
- Adapter Service → A2A Server (A2A Message)  
- A2A Server → Client (A2A Message)

## Setup and Execution

### 1. Project Structure

Ensure the following four Python files are in the same directory:

- a2a_server.py  
- adapter_service.py  
- legacy_bot_service.py  
- client.py

### 2. Create `.env` File

Create a file named `.env` in the same directory and add your OpenAI API key:

OPENAI_API_KEY="sk-..."

### 3. Install Dependencies (with `uv`)

Open a terminal in your project directory and run the following commands:

- Create a virtual environment:  
  uv venv

- Activate the environment (Linux/macOS):  
  source .venv/bin/activate

  (Windows alternative: .venv\Scripts\activate)

- Install required packages:  
  uv add "uvicorn[standard]" fastapi langchain-openai langchain-core httpx a2a-sdk langchain-community chromadb langgraph python-dotenv pydantic

### 4. Start the Services

You’ll need three separate terminals to run the services in parallel.

**Terminal 1 (Legacy RAG Bot):**  
uv run python legacy_bot_service.py

**Terminal 2 (Adapter Service):**  
uv run python adapter_service.py

**Terminal 3 (A2A Server):**  
uv run python a2a_server.py

### 5. Start the Client

Open a fourth terminal and also activate the virtual environment:

uv run python client.py

## Example Interaction

Start a conversation with the bot. You’ll notice that the responses are immediate, with no visible polling messages.

- You: Who is the owner of Bella Vista?  
- You: What are the opening hours?  
- You: Tell me about the menu