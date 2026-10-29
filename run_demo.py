"""
End-to-End Live Execution Test of RAG System
"""
import os
from rag_qa_system import (
    extract_text_from_pdfs,
    chunk_documents,
    RAGVectorStore,
    generate_grounded_answer,
    init_gemini_client
)

print("=" * 70)
print("🚀 RUNNING END-TO-END RAG DEMONSTRATION")
print("=" * 70)

# Step 1: Document Ingestion
pdf_path = "sample_ai_overview.pdf"
print(f"\n[1/4] Extracting text from '{pdf_path}'...")
extracted_pages = extract_text_from_pdfs([pdf_path])
print(f"      Extracted {len(extracted_pages)} readable pages.")
for p in extracted_pages:
    print(f"      - Page {p['page']}: {len(p['text'])} chars")

# Step 2: Semantic Chunking
print("\n[2/4] Chunking document (chunk_size=300, chunk_overlap=60)...")
chunks = chunk_documents(extracted_pages, chunk_size=300, chunk_overlap=60)
print(f"      Generated {len(chunks)} text chunks.")
for c in chunks:
    snippet = c['text'].replace('\n', ' ')[:75]
    print(f"      - Chunk #{c['id']} [Page {c['page']}]: \"{snippet}...\"")

# Step 3: Embeddings & FAISS Indexing
print("\n[3/4] Building FAISS Vector Index with 'all-MiniLM-L6-v2'...")
store = RAGVectorStore()
store.build_index(chunks)

# Step 4: Test Queries and Retrieval
test_queries = [
    "What is the budget for Project Titan?",
    "How does RAG reduce hallucinations?",
    "Who won the 1994 World Cup in soccer?"
]

print("\n[4/4] Executing Retrieval & Similarity Search...")
print("=" * 70)

for q in test_queries:
    print(f"\n❓ QUERY: \"{q}\"")
    matches = store.search(q, k=2, threshold=0.15)
    
    if not matches:
        print("   ❌ No chunks matched above similarity threshold.")
    else:
        print(f"   🔍 Retrieved {len(matches)} relevant chunk(s):")
        for idx, m in enumerate(matches, start=1):
            print(f"      [{idx}] Source: {m['filename']} (Page {m['page']}) | Cosine Score: {m['score']:.4f}")
            snippet = m['text'].replace('\n', ' ')
            print(f"          \"{snippet}\"")
            
print("\n" + "=" * 70)
print("✅ INGESTION, CHUNKING, EMBEDDING & FAISS RETRIEVAL COMPLETED PERFECTLY!")
print("=" * 70)
