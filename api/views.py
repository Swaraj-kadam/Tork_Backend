from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from django.db import connection
import tempfile, os, logging
import numpy as np
import time
from django.http import StreamingHttpResponse
from rest_framework.permissions import AllowAny
from .models import UploadedFile, DocumentChunk, User, Chat
from .utils.process_text import (
    extract_and_clean_pdf,
    extract_text_from_pdf_with_ocr,
    extract_and_clean_ppt,
    extract_and_clean_excel,
    chunk_text,
)
from .utils.ollama_llm import (
    ask_llama,
    stream_llama,
    generate_embeddings,
    generate_query_embedding,
    list_models,
)

logger = logging.getLogger(__name__)

# --------------------------------
# List available models
# --------------------------------
@api_view(["GET"])
def get_models(request):
    return Response({"models": list_models()})


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

        # Save temp file
        temp_path = tempfile.mktemp(suffix=uploaded_file.name)
        with open(temp_path, "wb+") as f:
            for chunk in uploaded_file.chunks():
                f.write(chunk)

        # Extract text based on file type
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".pdf"):
            text = extract_and_clean_pdf(temp_path)
            if not text.strip():  # fallback to OCR
                text = extract_text_from_pdf_with_ocr(temp_path)
        elif file_name.endswith((".ppt", ".pptx")):
            text = extract_and_clean_ppt(temp_path)
        elif file_name.endswith((".xlsx", ".xls")):
            text = extract_and_clean_excel(temp_path)
        else:
            os.remove(temp_path)
            return Response({"error": "Unsupported file format"}, status=400)

        # Create UploadedFile record
        uploaded_obj = UploadedFile.objects.create(
            user=user,
            organization=org,
            file_name=uploaded_file.name,
            file_size=uploaded_file.size,
            file_path=uploaded_file,
            extracted_text=text,
        )

        # --- Chunk text ---
        chunks = chunk_text(text)
        logger.info(f"Total chunks created: {len(chunks)}")

        # --- Generate embeddings ---
        embeddings = generate_embeddings(chunks)

        created = 0
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            if emb is None or len(emb) == 0:
                logger.warning(f"Skipping empty embedding for chunk {idx}")
                continue

            # Ensure list format for pgvector
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()

            DocumentChunk.objects.create(
                uploaded_file=uploaded_obj,
                user=user,
                organization=org,
                content=chunk,
                embedding=emb,
                chunk_index=idx,
            )
            created += 1

        os.remove(temp_path)
        logger.info(f"✅ File '{uploaded_file.name}' processed — {created}/{len(chunks)} chunks saved")

        return Response({
            "status": "success",
            "file": uploaded_file.name,
            "chunks_saved": created
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
class QueryView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        query = request.data.get("query")
        if not query:
            return Response({"error": "Query is required"}, status=400)
        
        start = time.time()

        # Generate embedding
        q_emb = generate_query_embedding(query)
        print("⏱️ Embedding Time:", time.time() - start)
        embedding_str = "[" + ", ".join(f"{x:.6f}" for x in q_emb) + "]"

        # Find relevant docs
        with connection.cursor() as cur:
            cur.execute("""
                SELECT content, embedding <-> %s::vector AS distance
                FROM api_documentchunk
                ORDER BY distance
                LIMIT 5;
            """, [embedding_str])
            rows = cur.fetchall()

        context = "\n\n".join([r[0] for r in rows])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        # ✅ Stream response
        return StreamingHttpResponse(stream_llama(prompt), content_type="text/plain")
    
# --------------------------------
# ask LLaMA - Stream response with embedding cache
# --------------------------------
class QueryView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        query = request.data.get("query")
        if not query:
            return Response({"error": "Query is required"}, status=400)

        user = getattr(request, "user", None)
        if not user or user.is_anonymous:
            user = User.objects.first()
        org = getattr(user, "organization", None)

        q_emb = generate_query_embedding(query)
        emb_str = "[" + ", ".join(f"{x:.6f}" for x in q_emb) + "]"

        with connection.cursor() as cur:
            cur.execute("""
                SELECT content, embedding <-> %s::vector AS distance
                FROM api_documentchunk
                WHERE organization_id IS NOT DISTINCT FROM %s
                ORDER BY distance LIMIT 5;
            """, [emb_str, org.id if org else None])
            rows = cur.fetchall()

        if not rows:
            return Response({"answer": "No relevant data found."})

        context = "\n\n".join([r[0] for r in rows])
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

        def event_stream():
            answer = ""
            for token in stream_llama(prompt):
                answer += token
                yield f"data: {token}\n\n"
            Chat.objects.create(user=user, organization=org, query=query, response=answer)

        return StreamingHttpResponse(event_stream(), content_type="text/event-stream")
