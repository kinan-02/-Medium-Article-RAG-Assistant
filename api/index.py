import os
from fastapi import FastAPI
from pinecone import Pinecone
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# 1. Load environment variables from .env file
load_dotenv()

# --- Configuration & Hyperparameters ---
CHUNK_SIZE = 512
OVERLAP_RATIO = 0.1  # 50 tokens overlap / 512 chunk size ≈ 10%
TOP_K = 5  # Let's retrieve top 5 relevant passages

LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
LLMOD_BASE_URL = "https://api.llmod.ai/v1"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medium-rag-index"

EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL = "4UHRUIN-gpt-5-mini"
# --- Initialization ---
app = FastAPI()

# LangChain components pointing to LLMod.ai
embeddings_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_base=LLMOD_BASE_URL,
    openai_api_key=LLMOD_API_KEY
)

llm = ChatOpenAI(
    api_key=LLMOD_API_KEY,
    base_url=LLMOD_BASE_URL,
    model=CHAT_MODEL,
    temperature=1.0
)

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# Exact Assignment System Prompt
SYSTEM_PROMPT_TEXT = (
    "You are a Medium-article assistant that answers questions strictly and only "
    "based on the Medium articles dataset context provided to you (metadata "
    "and article passages). You must not use any external knowledge, the open "
    "internet, or information that is not explicitly contained in the retrieved "
    "context. If the answer cannot be determined from the provided context, "
    "respond: \"I don't know based on the provided Medium articles data.\" "
    "Always explain your answer using the given context, quoting or "
    "paraphrasing the relevant article passage or metadata when helpful."
)

rag_prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT_TEXT),
    ("user", "Context:\n{context}\n\nQuestion: {question}")
])


# --- Pydantic Schemas for Validation ---
class QueryRequest(BaseModel):
    question: str


# --- HTTP Endpoints ---

@app.get("/api/stats")
def get_stats():
    """Returns the strict JSON configuration of the RAG system."""
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K
    }


@app.post("/api/prompt")
def query_rag(payload: QueryRequest):
    """Handles vector search and context-augmented generation."""
    question = payload.question

    # 1. Embed the user question
    query_vector = embeddings_model.embed_query(question)

    # 2. Query Pinecone for relevant contexts
    search_response = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True
    )

    # 3. Format context blocks and gather metadata for output payload
    context_list = []
    formatted_context_strings = []

    for match in search_response.matches:
        meta = match.metadata
        # Create the exact structured dictionary required by the grading script
        context_list.append({
            "article_id": meta.get("article_id", ""),
            "title": meta.get("title", ""),
            "chunk": meta.get("text", ""),
            "score": float(match.score)
        })

        # Compile a clean text block for the LLM context prompt
        formatted_context_strings.append(
            f"Title: {meta.get('title')}\n"
            f"Authors: {meta.get('authors')}\n"
            f"Passage: {meta.get('text')}\n"
            f"---"
        )

    combined_context_text = "\n".join(formatted_context_strings)

    # 4. Generate Compiled Prompt Strings to expose in the response
    # We invoke the template formatting directly to extract the raw strings
    compiled_messages = rag_prompt_template.format_messages(
        context=combined_context_text,
        question=question
    )
    system_prompt_compiled = compiled_messages[0].content
    user_prompt_compiled = compiled_messages[1].content

    # 5. Get answer from GPT-5-mini
    llm_response = llm.invoke(compiled_messages)

    # 6. Return response matching the exact JSON structural requirements
    return {
        "response": llm_response.content,
        "context": context_list,
        "Augmented_prompt": {
            "System": system_prompt_compiled,
            "User": user_prompt_compiled
        }
    }