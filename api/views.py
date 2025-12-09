from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from django.db import connection
import tempfile, os, logging
import numpy as np
import json
import time
from django.http import StreamingHttpResponse , JsonResponse
from rest_framework.permissions import AllowAny
from django.views.decorators.csrf import csrf_exempt
from .models import UploadedFile, DocumentChunk, User, Chat
from .utils.process_text import (
    extract_text_from_file,
    chunk_text,
)
from .utils.ollama_llm import (
    ask_llama,
    stream_llama,
    generate_embeddings,
    generate_query_embedding,
    list_models,
    set_model
)
from api.rerank import rerank_chunks_with_llama
from api.prompting import build_prompt
from api.safety import is_harmful_query


logger = logging.getLogger(__name__)

# --------------------------------
# List available models
# --------------------------------
def get_ollama_models(request):
    models = list_models()
    return JsonResponse({"models": models})

@csrf_exempt
def select_model(request):
    if request.method == "POST":
        data = json.loads(request.body)
        model = data.get("model")
        if model:
            set_model(model)
            return JsonResponse({"status": "success", "model": model})
        return JsonResponse({"status": "error", "message": "Model not provided"}, status=400)


# --------------------------------
# Upload File + Store Embeddings
# --------------------------------
class UploadView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        # Handle anonymous users gracefully
        user = getattr(request, "user", None)
        if not user or user.is_anonymous:
            user = User.objects.first()
            logger.info("Anonymous upload — using default user")

        org = getattr(user, "organization", None)

        uploaded_file = request.FILES.get("file")
        if not uploaded_file:
            return Response({"error": "No file uploaded"}, status=400)

        # Save to temporary file
        temp_path = tempfile.mktemp(suffix=uploaded_file.name)
        with open(temp_path, "wb+") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # Extract text using unified handler
        try:
            text = extract_text_from_file(temp_path)
        except Exception as e:
            logger.error(f"Extraction failed for {uploaded_file.name}: {e}")
            os.remove(temp_path)
            return Response({"error": "File extraction failed"}, status=500)

        if not text.strip():
            os.remove(temp_path)
            return Response({"error": "No readable text found in file"}, status=400)

        # Save uploaded file record
        uploaded_obj = UploadedFile.objects.create(
            user=user,
            organization=org,
            file_name=uploaded_file.name,
            file_size=uploaded_file.size,
            file_path=uploaded_file,
            extracted_text=text,
        )

        # --- Chunk the extracted text ---
        chunks = chunk_text(text)
        logger.info(f"Total chunks created: {len(chunks)}")

        # --- Generate embeddings ---
        embeddings = generate_embeddings(chunks)

        saved_count = 0
        for idx, (chunk_text_block, emb) in enumerate(zip(chunks, embeddings)):

            if emb is None or len(emb) == 0:
                logger.warning(f"Skipping empty embedding for chunk {idx}")
                continue

            if isinstance(emb, np.ndarray):
                emb = emb.tolist()

            DocumentChunk.objects.create(
                uploaded_file=uploaded_obj,
                user=user,
                organization=org,
                content=chunk_text_block,
                embedding=emb,
                chunk_index=idx,
            )
            saved_count += 1

        # Cleanup
        os.remove(temp_path)

        logger.info(f"✅ File '{uploaded_file.name}' processed — {saved_count}/{len(chunks)} chunks saved")

        return Response({
            "status": "success",
            "file": uploaded_file.name,
            "chunks_saved": saved_count
        })

# --------------------------------
# Query (Ask LLaMA)
# --------------------------------
# class QueryView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         user = getattr(request, "user", None)
#         if not user or user.is_anonymous:
#             user = User.objects.first()

#         org = getattr(user, "organization", None)
#         org_id = org.id if org else None
#         query = request.data.get("query")

#         if not query:
#             return Response({"error": "Query is required"}, status=400)

#         # --- Generate query embedding ---
#         q_emb = generate_query_embedding(query)

#         # ✅ Convert embedding to comma-separated vector
#         embedding_str = "[" + ", ".join(f"{x:.6f}" for x in q_emb) + "]"

#         # --- Perform similarity search ---
#         with connection.cursor() as cur:
#             cur.execute(
#                 """
#                 SELECT content, embedding <-> %s::vector AS distance
#                 FROM api_documentchunk
#                 WHERE organization_id IS NOT DISTINCT FROM %s
#                 ORDER BY distance
#                 LIMIT 5;
#                 """,
#                 [embedding_str, org.id if org else None],
#             )
#             rows = cur.fetchall()

#         if not rows:
#             return Response({"answer": "No relevant data found."})

#         # --- Combine top chunks ---
#         context = "\n\n".join([r[0] for r in rows])
#         prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

#         # --- Get answer from LLaMA or model ---
#         answer = ask_llama(prompt)

#         # --- Save chat ---
#         Chat.objects.create(user=user, organization=org, query=query, response=answer)

#         return Response({"answer": answer})

# --------------------------------
# Query (Ask LLaMA) - Stream response
# --------------------------------
# class QueryView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         query = request.data.get("query")
#         if not query:
#             return Response({"error": "Query is required"}, status=400)
        
#         start = time.time()

#         # Generate embedding
#         q_emb = generate_query_embedding(query)
#         print("⏱️ Embedding Time:", time.time() - start)
#         embedding_str = "[" + ", ".join(f"{x:.6f}" for x in q_emb) + "]"

#         # Find relevant docs
#         with connection.cursor() as cur:
#             cur.execute("""
#                 SELECT content, embedding <-> %s::vector AS distance
#                 FROM api_documentchunk
#                 ORDER BY distance
#                 LIMIT 15;
#             """, [embedding_str])
#             rows = cur.fetchall()

#         context = "\n\n".join([r[0] for r in rows][:8])
#         prompt = f"""
#         You are an academic assistant. Answer the question ONLY using the context below.
#         Do NOT add anything not present in the context.
#         If the answer is not in the document, say "The document does not contain this information."

#         CONTEXT:
#         {context}

#         QUESTION:
#         {query}

#         ANSWER:
#         """

#         # ✅ Stream response
#         return StreamingHttpResponse(stream_llama(prompt), content_type="text/plain")
    
# # --------------------------------
# # ask LLaMA - Stream response with embedding cache
# # --------------------------------
# class QueryView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         query = request.data.get("query")
#         if not query:
#             return Response({"error": "Query is required"}, status=400)

#         user = getattr(request, "user", None)
#         if not user or user.is_anonymous:
#             user = User.objects.first()
#         org = getattr(user, "organization", None)
#         start = time.time()
#         q_emb = generate_query_embedding(query)
#         # ❗ Stop if embedding failed or empty
#         if q_emb is None or len(q_emb) == 0:
#             return Response({"error": "Embedding generation failed"}, status=500)

#         # Optional: also block fallback zero vectors
#         import numpy as np
#         if np.all(q_emb == 0):
#             return Response({"error": "Embedding generation failed (zero vector)"}, status=500)
#         print("⏱️ Embedding Time:", time.time() - start)
#         embedding_str = "[" + ", ".join(f"{x:.6f}" for x in q_emb) + "]"

#         with connection.cursor() as cur:
#             cur.execute("""
#                  SELECT content, embedding <-> %s::vector AS distance
#                  FROM api_documentchunk
#                  ORDER BY distance
#                  LIMIT 15;
#              """, [embedding_str])
#             rows = cur.fetchall()

#         if not rows:
#             return Response({"answer": "No relevant data found."})

#         context = "\n\n".join([r[0] for r in rows][:8])
#         prompt = f"""
#         You are an academic assistant. Answer the question ONLY using the context below.
#         Do NOT add anything not present in the context.
#         If the answer is not in the document, say "The document does not contain this information."

#         CONTEXT:
#         {context}

#         QUESTION:
#         {query}

#         ANSWER:
#         """
#         print("response time:", time.time() - start)

#         def event_stream():
#             answer = ""
#             for token in stream_llama(prompt):
#                 answer += token
#                 yield f"data: {token}\n\n"
#             Chat.objects.create(user=user, organization=org, query=query, response=answer)

#         return StreamingHttpResponse(event_stream(), content_type="text/event-stream")

# --------------------------------
# Hybrid BM25 + Vector Search + Summary Mode
# --------------------------------
# class QueryView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         query = request.data.get("query")
#         if not query:
#             return Response({"error": "Query is required"}, status=400)

#         # ---- USER HANDLING ----
#         user = getattr(request, "user", None)
#         if not user or user.is_anonymous:
#             user = User.objects.first()

#         org = getattr(user, "organization", None)

#         # ---- Detect if user wants a summary ----
#         summary_mode = any(word in query.lower() for word in [
#             "summary", "summarize", "overview", "describe the whole document",
#             "entire document", "full document", "complete document"
#         ])

#         # ---- 1. EMBEDDING ----
#         start = time.time()
#         q_emb = generate_query_embedding(query)

#         if q_emb is None or len(q_emb) == 0:
#             return Response({"error": "Embedding generation failed"}, status=500)

#         import numpy as np
#         if np.all(q_emb == 0):
#             return Response({"error": "Embedding generation failed (zero vector)"}, status=500)

#         if isinstance(q_emb, np.ndarray):
#             q_emb = q_emb.tolist()

#         print("⏱️ Embedding Time:", time.time() - start)

#         # ---- 2. HYBRID BM25 + Vector Search ----
#         with connection.cursor() as cur:
#             cur.execute("""
#                 SELECT content, vec_distance, bm25_score, hybrid_score
#                 FROM (
#                     SELECT 
#                         content,
#                         embedding <-> %s::vector AS vec_distance,
#                         ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS bm25_score,
#                         (0.7 * (embedding <-> %s::vector)
#                          - 0.3 * ts_rank_cd(search_vector, plainto_tsquery('english', %s))
#                         ) AS hybrid_score
#                     FROM api_documentchunk
#                     WHERE organization_id IS NOT DISTINCT FROM %s
#                 ) AS sub
#                 ORDER BY hybrid_score ASC
#                 LIMIT 18;
#             """, [q_emb, query, q_emb, query, org.id if org else None])

#             rows = cur.fetchall()

#         if not rows:
#             return Response({"answer": "No relevant data found."})

#         # ---- 3. BUILD CONTEXT ----
#         # rows = [content, vec_distance, bm25_score, hybrid_score]
#         context_chunks = [row[0] for row in rows[:12]]
#         context = "\n\n".join(context_chunks)

#         # ---- 4. PROMPTS ----
#         if summary_mode:
#             print("📝 SUMMARY MODE")
#             prompt = f"""
# You are an academic assistant.
# Your task is to summarize the ENTIRE document using ONLY the context provided.
# Do NOT refuse. Do NOT say the summary is missing.
# Create a complete, well-structured summary.

# Use:
# - Headings
# - Bullet points
# - Clear explanations
# - Section-wise organization

# CONTEXT:
# {context}

# SUMMARY:
# """
#         else:
#             print("📘 NORMAL RAG MODE")
#             prompt = f"""
# You are an academic assistant. Answer the question ONLY using the context below.
# Do NOT add anything not present in the context.
# If the answer is not in the document, reply exactly:
# "The document does not contain this information."

# CONTEXT:
# {context}

# QUESTION:
# {query}

# ANSWER:
# """

#         print("response time:", time.time() - start)

#         # ---- 5. STREAM RESPONSE ----
#         def event_stream():
#             full_answer = ""
#             for token in stream_llama(prompt):
#                 full_answer += token
#                 yield f"data: {token}\n\n"

#             Chat.objects.create(
#                 user=user,
#                 organization=org,
#                 query=query,
#                 response=full_answer
#             )

#         return StreamingHttpResponse(event_stream(), content_type="text/event-stream")

#query view with re-ranking
# class QueryView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request):
#         query = request.data.get("query")
#         if not query:
#             return Response({"error": "Query is required"}, status=400)

#         # ---- USER HANDLING ----
#         user = getattr(request, "user", None)
#         if not user or user.is_anonymous:
#             user = User.objects.first()
#         org = getattr(user, "organization", None)

#         # ---- Detect summary intent (optional) ----
#         summary_mode = any(word in query.lower() for word in [
#             "summary", "summarize", "overview", "entire document", "whole document"
#         ])

#         # ---- 1. EMBEDDING ----
#         start = time.time()
#         q_emb = generate_query_embedding(query)
#         if q_emb is None or len(q_emb) == 0:
#             return Response({"error": "Embedding generation failed"}, status=500)

#         if np.all(q_emb == 0):
#             return Response({"error": "Embedding generation failed (zero vector)"}, status=500)

#         if isinstance(q_emb, np.ndarray):
#             q_emb = q_emb.tolist()

#         print("⏱️ Embedding Time:", time.time() - start)

#         # ---- 2. HYBRID VECTOR + BM25 SEARCH ----
#         with connection.cursor() as cur:
#             cur.execute("""
#                 SELECT content, vec_distance, bm25_score, hybrid_score
#                 FROM (
#                     SELECT 
#                         content,
#                         embedding <-> %s::vector AS vec_distance,
#                         ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS bm25_score,
#                         (0.7 * (embedding <-> %s::vector)
#                          - 0.3 * ts_rank_cd(search_vector, plainto_tsquery('english', %s))
#                         ) AS hybrid_score
#                     FROM api_documentchunk
#                     WHERE organization_id IS NOT DISTINCT FROM %s
#                 ) AS sub
#                 ORDER BY hybrid_score ASC
#                 LIMIT 20;
#             """, [q_emb, query, q_emb, query, org.id if org else None])

#             rows = cur.fetchall()

#         if not rows:
#             return Response({"answer": "No relevant data found."})

#         # rows: [content, vec_distance, bm25_score, hybrid_score]
#         initial_chunks = [r[0] for r in rows]

#         # ---- 3. RE-RANK CHUNKS WITH LLaMA ----
#         print("🔁 Running LLaMA re-ranking on retrieved chunks...")
#         reranked_chunks = rerank_chunks_with_llama(
#             query=query,
#             chunks=initial_chunks,
#             top_k=8,  # use top 8 for final context
#         )

#         # Build final context
#         context = "\n\n".join(reranked_chunks)

#         # ---- 4. PROMPT CONSTRUCTION ----
#         if summary_mode:
#             print("📝 SUMMARY MODE (with reranked context)")
#             prompt = f"""
# You are an academic assistant.
# Your task is to summarize the ENTIRE document using ONLY the context provided.
# Do NOT refuse. Do NOT say the summary is missing.
# Create a complete, well-structured summary.

# Use:
# - Headings
# - Bullet points
# - Clear explanations
# - Section-wise organization

# CONTEXT:
# {context}

# SUMMARY:
# """
#         else:
#             print("📘 NORMAL RAG MODE (with reranked context)")
#             prompt = f"""
# You are an academic assistant. Answer the question ONLY using the context below.
# Do NOT add anything not present in the context.
# If the answer is not in the document, reply exactly:
# "The document does not contain this information."

# CONTEXT:
# {context}

# QUESTION:
# {query}

# ANSWER:
# """

#         print("🕒 Total pre-generation time:", time.time() - start)

#         # ---- 5. STREAMING RESPONSE ----
#         def event_stream():
#             full_answer = ""
#             for token in stream_llama(prompt):
#                 full_answer += token
#                 yield f"data: {token}\n\n"

#             Chat.objects.create(
#                 user=user,
#                 organization=org,
#                 query=query,
#                 response=full_answer
#             )

#         return StreamingHttpResponse(event_stream(), content_type="text/event-stream")

#query view with re ranking and safe prompting
class QueryView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        query = request.data.get("query")
        if not query:
            return Response({"error": "Query is required"}, status=400)
        
        # --------------------------------
        # SAFETY GUARDRAIL (very important)
        # --------------------------------
        if is_harmful_query(query):
            return Response({
                "answer": "I’m sorry, but I cannot assist with that request."
            }, status=400)

        # ---- USER HANDLING ----
        user = getattr(request, "user", None)
        if not user or user.is_anonymous:
            user = User.objects.first()
        org = getattr(user, "organization", None)

        # ---- Detect summary intent ----
        summary_mode = any(word in query.lower() for word in [
            "summary", "summarize", "overview",
            "entire document", "whole document", "full document"
        ])

        # ---- 1. EMBEDDING ----
        start = time.time()
        q_emb = generate_query_embedding(query)
        if q_emb is None or len(q_emb) == 0:
            return Response({"error": "Embedding generation failed"}, status=500)

        if np.all(q_emb == 0):
            return Response({"error": "Embedding generation failed (zero vector)"}, status=500)

        if isinstance(q_emb, np.ndarray):
            q_emb = q_emb.tolist()

        print("⏱️ Embedding Time:", time.time() - start)

        # ---- 2. HYBRID VECTOR + BM25 SEARCH ----
        with connection.cursor() as cur:
            cur.execute("""
                SELECT content, vec_distance, bm25_score, hybrid_score
                FROM (
                    SELECT 
                        content,
                        embedding <-> %s::vector AS vec_distance,
                        ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS bm25_score,
                        (0.7 * (embedding <-> %s::vector)
                         - 0.3 * ts_rank_cd(search_vector, plainto_tsquery('english', %s))
                        ) AS hybrid_score
                    FROM api_documentchunk
                    WHERE organization_id IS NOT DISTINCT FROM %s
                ) AS sub
                ORDER BY hybrid_score ASC
                LIMIT 20;
            """, [q_emb, query, q_emb, query, org.id if org else None])

            rows = cur.fetchall()

        if not rows:
            return Response({"answer": "No relevant data found."})

        initial_chunks = [r[0] for r in rows]

        # ---- 3. RE-RANK CHUNKS WITH LLaMA ----
        reranked_chunks = rerank_chunks_with_llama(
            query=query,
            chunks=initial_chunks,
            top_k=8,
        )

        context = "\n\n".join(reranked_chunks)

        # ---- 4. BUILD SAFE, CoT-SUPPRESSED PROMPT ----
        if summary_mode:
            mode = "summary"
            print("📝 SUMMARY MODE (CoT suppressed)")
        else:
            mode = "qa"
            print("📘 QA MODE (CoT suppressed)")

        prompt = build_prompt(
            mode=mode,
            context=context,
            query=query if mode == "qa" else None,
        )

        print("🕒 Total pre-generation time:", time.time() - start)

        # ---- 5. STREAMING RESPONSE ----
        def event_stream():
            full_answer = ""
            for token in stream_llama(prompt):
                full_answer += token
                yield f"data: {token}\n\n"

            Chat.objects.create(
                user=user,
                organization=org,
                query=query,
                response=full_answer
            )

        return StreamingHttpResponse(event_stream(), content_type="text/event-stream")
