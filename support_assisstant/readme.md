# Zepto Support Assistant

## Module 3 — Support Assistant

This module implements a complete local GenAI/RAG-based support assistant for Zepto.

The application uses a collection of Zepto policy documents, generates local embeddings using `all-MiniLM-L6-v2`, stores and searches those embeddings using ChromaDB, and uses LangGraph to route user queries through an intent-classification and retrieval workflow.

The application is exposed through a FastAPI REST API and can also be packaged and executed using Docker.

The required graded implementation runs entirely offline using the deterministic `MOCK_LLM` mode. No LLM API key, cloud LLM provider, or network connection to an LLM provider is required.

*************************************************************************

# Project Objective

The objective of this module is to build a small but complete customer-support assistant capable of answering questions related to Zepto's policies.


The complete pipeline consists of:

1. Document chromadbcretionion
2. Document chunking
3. Local embedding generation
4. ChromaDB vector storage
5. Query embedding
6. Similarity-based retrieval
7. LangGraph intent routing
8. Mock/optional LLM answer generation
9. Pydantic response validation
10. FastAPI API serving
11. Docker containerization

*************************************************************************
# Project Structure

```text
support_assistant/
│
├── Documents/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
├── chroma_db/
│
├── chromadbcretion.py
├── Langraph.py
├── fastAPI.py
├── requirements.txt
├── Dockerfile
└── README.md
```

### File responsibilities

| File / Folder       | Purpose                                                                |
|******************************************************************************************************
| `Documents/`        | Contains the 8 Zepto policy documents                                  |
| `chromadbcretion.py`| Loads documents, creates embeddings and stores them in ChromaDB        |
| `Langraph.py`       | Contains LangGraph, routing, retrieval, generation and response schema |
| `fastAPI.py`        | FastAPI application and `/ask` endpoint                                |
| `requirements.txt`  | Python dependencies                                                    |
| `Dockerfile`        | Container configuration                                                |
| `chroma_db/`        | Persistent ChromaDB vector database                                    |
| `README.md`         | Project documentation                                                  |

*************************************************************************

# Policy Document Corpus

The project uses eight Zepto policy documents.

Each document contains a different area of Zepto's support policy.

### Document 1 — Delivery Policy
`doc_01.txt`
Contains information about:
* Delivery time
* Serviceable pin codes
* Standard delivery charges
* Free delivery threshold
* Priority delivery

*************************************************************************

### Document 2 — Returns & Refunds
`doc_02.txt`
Contains information about:
* Return windows
* Perishable items
* Non-perishable items
* Refund processing
* Wallet refunds
* Personal-care item returns
* Return pickup

*************************************************************************

### Document 3 — Membership Tiers
`doc_03.txt`
Contains information about:
* Basic membership
* Zepto Pass
* Zepto Pass+
* Membership pricing
* Delivery benefits
* Membership cancellation

*************************************************************************

### Document 4 — Order Tracking
`doc_04.txt`
Contains information about:
* Live rider tracking
* Track Order screen
* Estimated delivery time
* Delayed orders
* Contacting support

*************************************************************************

### Document 5 — Order Cancellation
`doc_05.txt`
Contains information about:
* Order cancellation
* Cancellation before packing
* Cancellation after packing
* Automatic cancellation
* Refunds

*************************************************************************

### Document 6 — Damaged or Missing Items
`doc_06.txt`
Contains information about:
* Damaged products
* Spoiled products
* Missing products
* Reporting an issue
* Replacement
* Refund
* Photo requirement for orders over INR 1000

*************************************************************************

### Document 7 — Gift Cards
`doc_07.txt`
Contains information about:
* Gift card denominations
* Gift card delivery
* Validity
* Combining payment methods
* Cash redemption

*************************************************************************

### Document 8 — Customer Support Hours
`doc_08.txt`
Contains information about:
* In-app chat
* Support availability
* Response time
* Email support
* Phone support

*************************************************************************

# Installation

## Step 1 — Create a virtual environment

From the `support_assistant` directory:

```bash
py -m venv .venv
```

### Windows

```powershell
.venv\Scripts\activate
```

*************************************************************************

## Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

The main libraries used are:

```text
pandas
numpy
chromadb
sentence-transformers
langgraph
langchain-core
fastapi
uvicorn
pydantic
```

*************************************************************************

# Document chromadbcretion

The document chromadbcretionion process is implemented in:

```text
chromadbcretion.py
```

The chromadbcretionion pipeline is:

```text
docs/*.txt-> Load documents-> One chunk per document-> all-MiniLM-L6-v2-> Generate embeddings-> ChromaDB
```
Because the supplied documents are short, each document is treated as a single chunk.

This provides eight chunks in total.

*************************************************************************

# 7. Running the chromadbcretionion Pipeline

Run:

```bash
py chromadbcretion.py
```

Expected output:

```text
Loaded 8 documents.
Stored 8 documents in ChromaDB.
Collection: zepto_policies
```

The process creates the persistent:

```text
chroma_db/
```

directory.

The ChromaDB collection is:

```text
zepto_policies
```

*************************************************************************

# Embedding Model

The project uses:

```text
all-MiniLM-L6-v2
```

through the `sentence-transformers` library.
The model is executed locally.
No external embedding API is required.
The same embedding model is used for:
1. Creating document embeddings during chromadbcretionion.
2. Creating the query embedding during retrieval.
This ensures that document vectors and query vectors exist in the same embedding space.

*************************************************************************

# ChromaDB

ChromaDB is used as the vector database.

The collection name is:

```text
zepto_policies
```
The collection stores:
* Document IDs
* Document text
* Embeddings
* Source metadata
The retrieval process uses cosine similarity.
For a policy question, the application retrieves the top three most similar chunks.

*************************************************************************

# LangGraph Architecture

The LangGraph workflow is implemented in:

```text
Langraph.py
```

The graph contains three required nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

The graph structure is:

```text

**User Question->The user submits a question to the support assistant.

**Intent-> The classify_intent node analyzes the question.
	It determines whether the question is:
	Policy Question
	General Question
	
**Policy Question->
	The request is sent to the retrieve_and_answer node.
	The system searches ChromaDB for relevant policy documents.
	The Top-3 relevant chunks are retrieved.
	The retrieved context is used to generate a grounded answer.
**General Question
	The request is sent to the direct_answer node.
	In MOCK_LLM=1 mode, the system returns a fixed mock response indicating that it can only answer Zepto policy questions.

**Pydantic Validation-> The generated response is validated using the Pydantic response schema.

**The response contains:->
	answer
	sources
	confidence
	JSON Response
	After successful validation, the final response is returned as structured JSON.

```

*************************************************************************

# LangGraph State

The application uses a typed state based on keywords.

The state contains information such as:

```text
query
intent
retrieved_documents
retrieved_ids
answer
sources
confidence
error
```

This state is passed between the LangGraph nodes.

*************************************************************************

# Node 1 — `classify_intent`

The first node determines whether the incoming query is:

```text
policy_question
```

or:

```text
general_question
```

In the required mock mode, classification uses a deterministic keyword heuristic.
For example:
```text
"What is the delivery fee?"
"Can I cancel my order?"
```

contains:
```text
delivery
cancel
```

Therefore it becomes:

```text
policy_question
```

An unrelated query such as:

```text
"What is the capital of india?"
```

does not contain a policy keyword and therefore becomes:

```text
general_question
```

*************************************************************************

# Node 2 — `retrieve_and_answer`

This node is executed for:

```text
policy_question
```

queries.

The retrieval process is:

```text
User Query-> all-MiniLM-L6-v2-> Query Embedding-> ChromaDB-> Cosine Similarity-> Top 3 Chunks
```

The retrieved document IDs are stored in the response `sources` field.

The retrieval process executes in both mock and optional real-LLM modes because embeddings and ChromaDB do not require an LLM API.

*************************************************************************

# Mock Answer Generation

When:
```text
MOCK_LLM=1
```
the application does not make an LLM API call.
Instead, the application takes the most relevant retrieved chunk and generates:
```text
Based on the retrieved context: <top chunk excerpt>
```
The excerpt is limited to approximately the first 200 characters.

For example:

```text
Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes...
```

This provides a deterministic answer that is directly grounded in the retrieved policy document.

*************************************************************************

# Node 3 — `direct_answer`

This node handles:

```text
general_question
```
In mock mode, it returns:

```text
I can only answer questions about Zepto policies right now.
```
No ChromaDB retrieval is required for this route.
The response contains:
```json
{
  "sources": []
}
```

This demonstrates that the graph correctly distinguishes policy queries from unrelated queries.

*************************************************************************

# Conditional Routing

A conditional edge is configured after:

```text
classify_intent
```

The routing logic is:

```text
if intent == "policy_question":
    retrieve_and_answer

else:
    direct_answer
```

Therefore:

```text
Policy query->classify_intent->policy_question->retrieve_and_answer
```

while:

```text
General query->classify_intent->general_question->direct_answer
```

The routing logic itself is independent of the `MOCK_LLM` setting.

The toggle only affects the generation/classification implementation.

*************************************************************************

# Structured Prompt Template

The project contains a structured prompt template for the optional real-LLM path.
The prompt follows the required:

```text
Role
Context
Task
Format
Length
```

structure.

It also includes a negative constraint:

```text
Do not answer using information that is not present in the provided context.
Do not invent Zepto policies.
```

This ensures that the real LLM is instructed to remain grounded in the retrieved documents.

*************************************************************************

# Few-Shot Example

The prompt also contains a few-shot example.
The example demonstrates how a user question about delivery charges should be answered using the supplied policy context.
The example includes:

```text
User question
Context
Expected structured response
```

This gives the optional real LLM path an explicit example of the expected response style and structure.

*************************************************************************

# Pydantic Response Schema

The final API response is validated using Pydantic.
The schema contains three fields:
```text
answer
sources
confidence
```
Example:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_02"
  ],
  "confidence": 1.0
}
```

### `answer`
Contains the generated assistant response.

### `sources`
Contains the IDs of retrieved chunks/documents used for a policy question.


For general questions:
```json
"sources": []
```

### `confidence`
Contains a floating-point value between:

```text
0.0 and 1.0
```

In mock mode, the implementation uses:

```text
1.0
```

because the output is deterministic and generated directly from the application's retrieved context.

*************************************************************************

# FastAPI

The FastAPI application is implemented in:

```text
fastAPI.py
```

The application exposes:

```text
POST /ask
```

The request model is:

```json
{
  "query": "What is the delivery fee?"
}
```

The response model is the validated:

```text
AssistantResponse
```

*************************************************************************

# Running FastAPI

Run:

```bash
uvicorn fastAPI:app --reload
```

The application will be available at:

```text
http://127.0.0.1:8000
```

FastAPI Swagger documentation is available at:

```text
http://127.0.0.1:8000/docs
```

*************************************************************************

# API Test 1 — Policy Question

Use the following request:

```json
{
  "query": "What is the delivery fee?"
}
```

This query contains the keyword:

```text
delivery
```

Therefore, the expected routing is:

```text
classify_intent->policy_question->retrieve_and_answer
```

The answer should follow the mock response format:

```text
  "answer": "Based on the retrieved context: Delivery Policy: \"Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order vo",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_02"
  ],
  "confidence": 1
```
*************************************************************************

# API Test 2 — General Question

Use:

```json
{
  "query": "What is the capital of France?"
}
```

This query does not contain any of the specified Zepto policy keywords.

Therefore, the expected routing is:

```text
classify_intent->general_question->direct_answer
```

Expected response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

This test demonstrates the second conditional path in the LangGraph.

*************************************************************************

# Additional Retrieval Tests

The following queries can be used to verify that retrieval is returning relevant documents.

### Delivery --->  What is the delivery fee? --->
Expected relevant document:doc_01

*************************************************************************

### Returns--->How long do I have to return a grocery item?
Expected relevant document:doc_02

*************************************************************************

### Membership--->How much does Zepto Pass cost?
Expected relevant document:doc_03
*************************************************************************

### Tracking--->How can I track my order?
Expected relevant document:doc_04
*************************************************************************

### Cancellation--->Can I cancel my order?
Expected relevant document:doc_05
*************************************************************************

### Damaged or Missing Items--->What should I do if an item is missing?
Expected relevant document:doc_06
*************************************************************************

### Gift Cards--->What gift card denominations are available?
Expected relevant document:doc_07
*************************************************************************

### Support--->What are the support hours?
Expected relevant document:doc_08

These tests help demonstrate that the retrieval stage returns content relevant to the user's question rather than simply returning arbitrary documents.

*************************************************************************

# MOCK_LLM Mode

The graded baseline uses:

```text
MOCK_LLM=1
```

or leaves the variable unset.

### Mock mode provides:

* Deterministic intent classification
* No LLM API calls
* No API key requirement
* Local embeddings
* ChromaDB retrieval
* Deterministic answer generation
* Deterministic Pydantic validation

To explicitly enable mock mode on Windows PowerShell:

```powershell
$env:MOCK_LLM="1"
```

Then run:

```powershell
uvicorn main:app --reload
```

*************************************************************************

# Optional Real-LLM Mode

The project structure also supports an optional:

```text
MOCK_LLM=0
```

extension.

This path is not required for the graded baseline.

When using the real-LLM path, the structured prompt is used to provide:

* Role
* Retrieved context
* User question
* Required output format
* Length requirements
* Grounding constraints
* Few-shot example

The real-LLM implementation should validate the raw LLM response against the Pydantic schema.

If validation fails, the implementation should retry with a corrective instruction up to two additional times.

The required submission remains fully functional without this optional extension.

*************************************************************************

# End-to-End Data Flow

The complete data flow is:

```text
                    INGESTION
                       |
                       v
              8 Policy Documents
                       |
                       v
                 Chunking
                       |
                       v
             Sentence Transformer
             all-MiniLM-L6-v2
                       |
                       v
                    Embeddings
                       |
                       v
                    ChromaDB
                       |
                       |
                       |       QUERY TIME
                       |           |
                       |           v
                       |       User Query
                       |           |
                       |           v
                       |    classify_intent
                       |           |
                       |       +-------+
                       |       |       |
                       |       v       v
                       |    Policy   General
                       |       |       |
                       |       v       v
                       |   Retrieval  Direct
                       |       |
                       |       v
                       |    Top-3 Chunks
                       |       |
                       |       v
                       |    Generation
                       |       |
                       |       v
                       |  Pydantic Schema
                       |       |
                       |       v
                       |    JSON Response
```

*************************************************************************

# Pipeline Components

## Ingestion

Handled by:

```text
chromadbcretion.py
```

Responsible for:

* Reading the 8 `.txt` files
* Creating document chunks
* Generating embeddings
* Storing vectors in ChromaDB

*************************************************************************

## Embedding

Handled using:

```text
SentenceTransformer
all-MiniLM-L6-v2
```

This happens locally without an external API.

*************************************************************************

## Retrieval

Handled by:

```text
retrieve_and_answer()
```

inside:

```text
Langraph.py
```

The function:

1. Embeds the user query.
2. Queries ChromaDB.
3. Retrieves the top 3 chunks.
4. Passes the most relevant chunk into the mock answer generation.

*************************************************************************

## Generation

Mock generation is handled by:

```text
retrieve_and_answer()
```

for policy questions and:

```text
direct_answer()
```

for general questions.

The optional real-LLM generation uses the structured prompt template.

*************************************************************************

# Where `MOCK_LLM` Changes the Pipeline

The important distinction is:

```text
                 MOCK_LLM
                    |
        +-----------+-----------+
        |                       |
        v                       v
      MOCK                     REAL
        |                       |
        v                       v
 Deterministic              Optional
 rule-based                 LLM calls
 generation
```

### In mock mode

```text
Classification
      |
      v
Keyword heuristic

Retrieval
      |
      v
ChromaDB

Generation
      |
      v
Deterministic canned output
```

### In optional real mode

The architecture remains the same, but the relevant generation/classification step can use a real LLM.

Retrieval continues to use the local:

```text
Sentence Transformers + ChromaDB
```

pipeline.

*************************************************************************

# Docker

The application includes a `Dockerfile` for local container execution.

The Docker image:

1. Uses Python 3.11.
2. Creates `/app` as the working directory.
3. Installs Python dependencies.
4. Copies the project files.
5. Sets `MOCK_LLM=1`.
6. Exposes port `8000`.
7. Starts Uvicorn.

*************************************************************************

# Build Docker Image

From the `support_assistant` directory:

```bash
docker build -t zepto-support-assistant .
```

A successful build indicates that the Dockerfile and dependencies can be installed correctly.

*************************************************************************

# Run Docker Container

Run:

```bash
docker run -p 8000:8000 zepto-support-assistant
```

The application will then be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

The API endpoint is:

```text
POST /ask
```

*************************************************************************

# Docker API Test

Example request:

```json
{
  "query": "What is the delivery fee?"
}
```

The request should be sent to:

```text
POST http://localhost:8000/ask
```

The response should contain:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["..."],
  "confidence": 1.0
}
```

*************************************************************************

#  Validation and Error Prevention

The project uses Pydantic to ensure the final response follows the required structure.

The `confidence` field is constrained to:

```text
0.0 <= confidence <= 1.0
```

The required fields are:

```text
answer
sources
confidence
```

This prevents the application from returning an incorrectly structured response.

In the required mock mode, the response is constructed directly by the application and therefore does not depend on unpredictable LLM output.

*************************************************************************

#  Why Mock Mode Is Used

The project deliberately uses mock mode as the baseline because it provides:

* Reproducible results
* No API key dependency
* No external LLM network calls
* Deterministic grading
* Consistent routing
* Consistent response structure

This makes the application suitable for offline evaluation.

*************************************************************************
Final Verification

Before submission, verify that the complete workflow operates successfully:

Documents → Embeddings → ChromaDB → Intent Classification → Retrieval/Direct Answer → Pydantic Validation → JSON Response

The application should work end-to-end in offline mock mode without requiring an external LLM API or API key.
*************************************************************************

# Summary

This project implements a complete local RAG support assistant for Zepto.

The application begins with eight Zepto policy documents. These documents are loaded and treated as individual chunks because of their short length. The chunks are embedded locally using `all-MiniLM-L6-v2` and stored in a persistent ChromaDB collection called `zepto_policies`.

At query time, LangGraph first classifies the query using the required keyword-based mock heuristic. Policy questions are sent to `retrieve_and_answer`, where the query is embedded and the top three relevant ChromaDB chunks are retrieved. The mock generation path creates a deterministic answer from the most similar chunk.

General questions are routed to `direct_answer` and receive the fixed response indicating that the assistant currently answers Zepto policy questions only.

The final response is validated using Pydantic and contains:

```text
answer
sources
confidence
```

FastAPI exposes the workflow through:

```text
POST /ask
```

The complete application can also be packaged and executed locally using Docker.


