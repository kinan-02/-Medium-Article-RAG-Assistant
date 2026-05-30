import os
import pandas as pd
import tiktoken
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# 1. Load environment variables from .env file
load_dotenv()

# --- Configuration & Hyperparameters ---
CSV_PATH = "data/medium-english-50mb.csv"
CHUNK_SIZE = 512
OVERLAP = 50
ENCODING_NAME = "cl100k_base"

# LLMod.ai Settings
LLMOD_API_KEY = os.getenv("LLMOD_API_KEY")
LLMOD_BASE_URL = "https://api.llmod.ai/v1"

# Pinecone Settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medium-rag-index"
EMBEDDING_DIMENSION = 1536


# --- Chunking Logic ---
def get_tokenizer():
    return tiktoken.get_encoding(ENCODING_NAME)


def chunk_text(text, chunk_size, overlap):
    if not isinstance(text, str): return []
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text)

    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i: i + chunk_size]
        chunks.append(tokenizer.decode(chunk_tokens))
        i += (chunk_size - overlap)
    return chunks


# --- Main Ingestion Pipeline ---
def main():
    # 1. Load Data
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)

    # 2. Chunking
    all_chunks = []
    print(f"Chunking {len(df)} articles...")
    for index, row in df.iterrows():
        article_chunks = chunk_text(row['text'], CHUNK_SIZE, OVERLAP)

        # Safe string conversion handling for missing data (NaNs)
        url = str(row['url']) if pd.notna(row['url']) else ""
        timestamp = str(row['timestamp']) if pd.notna(row['timestamp']) else ""
        tags = str(row['tags']) if pd.notna(row['tags']) else ""
        title = str(row['title']) if pd.notna(row['title']) else "Untitled"
        authors = str(row['authors']) if pd.notna(row['authors']) else "Unknown"

        for chunk_idx, chunk in enumerate(article_chunks):
            all_chunks.append({
                "id": f"article_{index}_chunk_{chunk_idx}",
                "article_id": str(index),
                "title": title,
                "authors": authors,
                "url": url,
                "timestamp": timestamp,
                "tags": tags,
                "text": chunk
            })

    # Save a local backup of the chunks for manual inspection
    chunks_df = pd.DataFrame(all_chunks)
    chunks_df.to_csv("data/processed_chunks.csv", index=False)
    print(f"Saved {len(all_chunks)} local chunks to 'data/processed_chunks.csv'.")

    # 3. Initialize Pinecone
    print("\nConnecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # Free tier default region
            )
        )

    index = pc.Index(INDEX_NAME)

    # 4. Initialize LangChain Embeddings via LLMod
    print("\nInitializing Embeddings model...")
    embeddings_model = OpenAIEmbeddings(
        model="4UHRUIN-text-embedding-3-small",
        openai_api_base=LLMOD_BASE_URL,
        openai_api_key=LLMOD_API_KEY,
        chunk_size=256
    )

    # 5. Embed and Upsert in Batches
    print("Embedding and upserting data to Pinecone...")
    pinecone_batch_size = 100

    for i in range(0, len(all_chunks), pinecone_batch_size):
        batch_chunks = all_chunks[i: i + pinecone_batch_size]

        # Extract the raw text for the embedding model
        texts_to_embed = [chunk['text'] for chunk in batch_chunks]

        # Generate the vector embeddings via LLMod
        vectors = embeddings_model.embed_documents(texts_to_embed)

        # Construct the full data payload Pinecone requires
        upsert_data = []
        for j, chunk in enumerate(batch_chunks):
            metadata = {
                "article_id": chunk["article_id"],
                "title": chunk["title"],
                "authors": chunk["authors"],
                "url": chunk["url"],
                "timestamp": chunk["timestamp"],
                "tags": chunk["tags"],
                "text": chunk["text"]  # The text chunk used for generation
            }
            upsert_data.append((chunk["id"], vectors[j], metadata))

        # Push the batch to the vector database
        index.upsert(vectors=upsert_data)
        print(f"  Upserted batch {i // pinecone_batch_size + 1} (items {i} to {i + len(batch_chunks)})")

    print("\nSuccess! Pipeline complete.")


if __name__ == "__main__":
    main()