from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, cast, Any
from pathlib import Path
import os, requests
import re
import io
import uuid
from PIL import Image
from openai import OpenAI, BadRequestError, NotFoundError
from openai.types.chat import ChatCompletionMessageParam
from fastapi import UploadFile, File, HTTPException
from typing import Optional, cast
from tavily import TavilyClient
from dotenv import load_dotenv

# RAG imports
import fitz  # PyMuPDF for PDF extraction
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

# Load environment variables from .env file
load_dotenv(Path(__file__).parent / ".env")

app = FastAPI()

# CORS (ok to keep even if same-origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve paths relative to this file
BASE_DIR = Path(__file__).parent

# Optionally serve static files under /static if folder exists
STATIC_DIR = BASE_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Uploads directory (always ensure and mount)
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Serve index.html at root
@app.get("/")
async def index():
    # Serve the app's index.html located alongside this file
    return FileResponse(str(BASE_DIR / "index.html"))


# OpenAI-compatible chat completions client (e.g., vLLM server)
openai_api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
openai_api_base = os.getenv("OPENAI_API_BASE", "")

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
    default_headers={"ngrok-skip-browser-warning": "true"},
)

# Tavily Web Search client
tavily_api_key = os.getenv("TAVILY_API_KEY", "")
tavily_client = TavilyClient(api_key=tavily_api_key) if tavily_api_key else None

# ============== RAG Configuration ==============
# Embedding model for RAG
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))  # characters per chunk
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))  # overlap between chunks
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))  # number of chunks to retrieve
RAG_COLLECTION_NAME = "documents"

# Initialize embedding model (lazy loading)
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model: {RAG_EMBEDDING_MODEL}...")
        _embedding_model = SentenceTransformer(RAG_EMBEDDING_MODEL)
        print("Embedding model loaded.")
    return _embedding_model

# Initialize Qdrant client (in-memory)
qdrant_client = QdrantClient(":memory:")

# Create collection on startup
def init_qdrant_collection():
    """Initialize Qdrant collection if it doesn't exist."""
    collections = qdrant_client.get_collections().collections
    if not any(c.name == RAG_COLLECTION_NAME for c in collections):
        # all-MiniLM-L6-v2 produces 384-dimensional vectors
        qdrant_client.create_collection(
            collection_name=RAG_COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )
        print(f"Created Qdrant collection: {RAG_COLLECTION_NAME}")

init_qdrant_collection()

# Store document metadata (in-memory)
document_metadata: Dict[str, Dict[str, Any]] = {}  # doc_id -> {filename, user_id, chunk_count, ...}
# ============== End RAG Configuration ==============

user_histories: Dict[str, List[ChatCompletionMessageParam]] = {}
vision_capability_overrides: Dict[str, bool] = {}  # Store user overrides for vision capability
SYSTEM_PROMPT = "You are a helpful assistant."

# Chat behavior knobs (override via env vars if needed)
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
CONTEXT_MARGIN = int(os.getenv("CONTEXT_MARGIN", "16"))  # safety headroom tokens
SUFFIX_MARGIN_TOKENS = int(os.getenv("SUFFIX_MARGIN_TOKENS", "24"))
TRUNCATION_SUFFIX = os.getenv("TRUNCATION_SUFFIX", "… Would you like me to continue?")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "")
VISION_MODELS_ENV = {m.strip() for m in os.getenv("VISION_MODELS", "").split(",") if m.strip()}
INLINE_LOCAL_UPLOADS = os.getenv("INLINE_LOCAL_UPLOADS", "1") in {"1", "true", "yes", "on"}

# Max history turns per model type (user+assistant pairs)
MAX_HISTORY_TURNS_TEXT = int(os.getenv("MAX_HISTORY_TURNS_TEXT", "6"))
MAX_HISTORY_TURNS_VISION = int(os.getenv("MAX_HISTORY_TURNS_VISION", "2"))

# Image compression settings
IMAGE_SIZE_THRESHOLD = int(os.getenv("IMAGE_SIZE_THRESHOLD", str(500 * 1024)))  # 500 KB
IMAGE_MAX_SIZE_THRESHOLD = int(os.getenv("IMAGE_MAX_SIZE_THRESHOLD", str(1 * 1024 * 1024)))  # 1 MB
IMAGE_MAX_DIMENSION = int(os.getenv("IMAGE_MAX_DIMENSION", "2048"))  # Max width or height
IMAGE_QUALITY = int(os.getenv("IMAGE_QUALITY", "85"))  # JPEG/WebP quality (1-100)
VISION_PROBE_CACHE: dict[str, bool] = {}


class ChatRequest(BaseModel):
    user_id: str
    message: str
    attachments: Optional[List["Attachment"]] = None
    model: Optional[str] = None
    vision_enabled: Optional[bool] = None  # User toggle override for vision capability
    web_search: Optional[bool] = False  # Enable web search for this request
    rag_enabled: Optional[bool] = False  # Enable RAG for this request


class ChatResponse(BaseModel):
    reply: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    duration_ms: int | None = None
    tokens_per_sec: float | None = None


class Attachment(BaseModel):
    filename: str
    url: str
    mime_type: str
    text: Optional[str] = None


# ============== RAG Helper Functions ==============

def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from a PDF file using PyMuPDF."""
    try:
        doc = fitz.open(str(file_path))
        text_parts = []
        for page in doc:
            text_parts.append(page.get_text())
        doc.close()
        return "\n".join(text_parts)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""


def chunk_text(text: str, chunk_size: int = RAG_CHUNK_SIZE, overlap: int = RAG_CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    if not text:
        return []
    
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at sentence or paragraph boundary
        if end < text_len:
            # Look for last period, newline, or space
            for sep in ['\n\n', '\n', '. ', ' ']:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size // 2:  # Only break if we're past halfway
                    chunk = chunk[:last_sep + len(sep)]
                    end = start + len(chunk)
                    break
        
        chunks.append(chunk.strip())
        start = end - overlap
        
        # Prevent infinite loop
        if start >= text_len - overlap:
            break
    
    return [c for c in chunks if c]  # Filter empty chunks


def index_document(user_id: str, doc_id: str, filename: str, text: str) -> int:
    """Index a document's text chunks into Qdrant."""
    chunks = chunk_text(text)
    if not chunks:
        return 0
    
    model = get_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=False)
    
    points = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        point_id = str(uuid.uuid4())
        points.append(PointStruct(
            id=point_id,
            vector=embedding.tolist(),
            payload={
                "user_id": user_id,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": i,
                "text": chunk,
            }
        ))
    
    qdrant_client.upsert(collection_name=RAG_COLLECTION_NAME, points=points)
    
    # Store metadata
    document_metadata[doc_id] = {
        "filename": filename,
        "user_id": user_id,
        "chunk_count": len(chunks),
        "text_length": len(text),
    }
    
    return len(chunks)


def search_documents(user_id: str, query: str, top_k: int = RAG_TOP_K) -> List[Dict[str, Any]]:
    """Search for relevant document chunks for a user's query."""
    model = get_embedding_model()
    query_embedding = model.encode([query], show_progress_bar=False)[0]
    
    # Use query_points for newer qdrant-client versions, fallback to search for older versions
    try:
        results = qdrant_client.query_points(
            collection_name=RAG_COLLECTION_NAME,
            query=query_embedding.tolist(),
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=top_k,
        ).points
    except AttributeError:
        # Fallback for older qdrant-client versions
        results = qdrant_client.search(
            collection_name=RAG_COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=top_k,
        )
    
    return [
        {
            "text": hit.payload.get("text", ""),
            "filename": hit.payload.get("filename", ""),
            "score": hit.score,
            "chunk_index": hit.payload.get("chunk_index", 0),
        }
        for hit in results
    ]


def get_user_documents(user_id: str) -> List[Dict[str, Any]]:
    """Get list of documents indexed for a user."""
    return [
        {"doc_id": doc_id, **meta}
        for doc_id, meta in document_metadata.items()
        if meta.get("user_id") == user_id
    ]


def delete_document(user_id: str, doc_id: str) -> bool:
    """Delete a document and its chunks from the index."""
    if doc_id not in document_metadata:
        return False
    
    meta = document_metadata[doc_id]
    if meta.get("user_id") != user_id:
        return False
    
    # Delete points with matching doc_id
    qdrant_client.delete(
        collection_name=RAG_COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        ),
    )
    
    del document_metadata[doc_id]
    return True


def perform_rag_search(user_id: str, query: str) -> str:
    """Perform RAG search and return formatted context."""
    results = search_documents(user_id, query)
    
    if not results:
        return ""
    
    parts = ["**[Document Context]**\n"]
    for i, r in enumerate(results, 1):
        score_pct = int(r["score"] * 100)
        parts.append(f"\n**[{i}. {r['filename']} (relevance: {score_pct}%)]**\n{r['text']}\n")
    
    return "\n".join(parts)


# ============== End RAG Helper Functions ==============


def probe_vision_capability(base_url: str, model_id: str, timeout: int = 5) -> bool:
    if model_id in VISION_PROBE_CACHE:
        return VISION_PROBE_CACHE[model_id]

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "about:blank"},
                    {"type": "text", "text": "ping"},
                ],
            }
        ],
        "max_tokens": 1,
    }

    # Normalize the base URL - strip trailing /v1 if present, then add it back
    # This ensures consistency whether base_url ends with /v1 or not
    normalized_base = base_url.rstrip("/")
    if normalized_base.endswith("/v1"):
        normalized_base = normalized_base[:-3]
    
    try:
        r = requests.post(
            f"{normalized_base}/v1/chat/completions",
            json=payload,
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=timeout,
        )

        if r.status_code == 200:
            VISION_PROBE_CACHE[model_id] = True
            return True

        error_text = r.text.lower()

        if any(k in error_text for k in [
            "image",
            "vision",
            "multimodal",
            "image_url",
            "image token",
        ]):
            VISION_PROBE_CACHE[model_id] = True
            return True

        VISION_PROBE_CACHE[model_id] = False
        return False

    except requests.RequestException:
        VISION_PROBE_CACHE[model_id] = False
        return False


def get_vision_capability(user_id: str, model_id: str) -> bool:
    """
    Get vision capability for a model, checking for user override first.
    If user has set an override for this model, use that.
    Otherwise, use the probed capability.
    """
    override_key = f"{user_id}:{model_id}"
    override = vision_capability_overrides.get(override_key)
    if override is not None:
        return override
    return probe_vision_capability(openai_api_base, model_id)


def get_vision_capability_from_request(user_id: str, model_id: str, vision_enabled_override: Optional[bool] = None) -> bool:
    """
    Get vision capability, checking user's request override first.
    Priority: request override > stored override > probed capability
    """
    if vision_enabled_override is not None:
        return vision_enabled_override
    return get_vision_capability(user_id, model_id)


def get_max_history_turns_for_model(model_id: str, user_id: str = "", vision_enabled_override: Optional[bool] = None) -> int:
    vision_capable = get_vision_capability_from_request(user_id, model_id, vision_enabled_override)
    return MAX_HISTORY_TURNS_VISION if vision_capable else MAX_HISTORY_TURNS_TEXT


def validate_attachments_for_model(model_id: str, attachments: Optional[List[Attachment]], user_id: str = "", vision_enabled_override: Optional[bool] = None):
    if not attachments:
        return
    vision_capable = get_vision_capability_from_request(user_id, model_id, vision_enabled_override)
    if vision_capable:
        return
    for att in attachments:
        mt = (att.mime_type or "").lower()
        # Only allow text/* for text-only models
        if not mt.startswith("text/"):
            raise HTTPException(status_code=400, detail="Selected model is text-only; remove non-text attachments (images/PDFs).")



def _inject_attachments_into_message(base_text: str, attachments: Optional[List[Attachment]]) -> str:
    if not attachments:
        return base_text
    lines: List[str] = []
    for att in attachments:
        header = f"[Attachment] {att.filename} ({att.mime_type}) URL: {att.url}"
        if att.text:
            # Limit attachment text to avoid huge prompts
            preview = att.text[:5000]
            header += f"\nContent preview:\n{preview}"
        lines.append(header)
    return base_text + "\n\n" + "\n\n".join(lines)


def _build_user_content(user_message: str,
                        attachments: Optional[List[Attachment]],
                        model_id: str,
                        user_id: str = "",
                        vision_enabled_override: Optional[bool] = None) -> Any:
    vision_capable = get_vision_capability_from_request(user_id, model_id, vision_enabled_override)
    if vision_capable:
        parts: List[Dict[str, Any]] = [{"type": "text", "text": user_message}]
        for att in attachments or []:
            mt = (att.mime_type or "").lower()
            if mt.startswith("image/"):
                # Ensure absolute URL; client attempts this already, but be tolerant
                url = str(att.url)
                try:
                    # If it's relative, make it absolute to our origin
                    if not (url.startswith("http://") or url.startswith("https://") or url.startswith("data:")):
                        url = url if url.startswith("/") else "/" + url
                        # We cannot know host here reliably; keep as-is and rely on client normalization
                except Exception:
                    pass
                # Optionally inline local uploads as data URLs so the model server
                # doesn't need to fetch from our FastAPI host.
                if INLINE_LOCAL_UPLOADS:
                    try:
                        from urllib.parse import urlparse
                        import base64
                        parsed = urlparse(url)
                        path = parsed.path or ""
                        if path.startswith("/uploads/"):
                            fname = os.path.basename(path)
                            fpath = UPLOAD_DIR / fname
                            if fpath.exists() and fpath.is_file():
                                raw = fpath.read_bytes()
                                b64 = base64.b64encode(raw).decode("ascii")
                                url = f"data:{mt};base64,{b64}"
                    except Exception:
                        # Fall back to original URL if any error occurs
                        pass
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": url}
                })
            elif mt.startswith("text/") and att.text:
                parts.append({"type": "text", "text": f"[Attachment {att.filename}]\n{att.text[:5000]}"})
            # PDFs are kept as reference only (no OCR here)
        return parts
    # text-only fallback: merge into a single string (with lightweight attachment previews)
    if not attachments:
        return user_message
    lines = [user_message]
    for att in attachments:
        mt = (att.mime_type or "").lower()
        if mt.startswith("text/") and att.text:
            lines.append(f"[Attachment {att.filename}]\n{att.text[:1000]}")
        elif mt.startswith("image/"):
            lines.append(f"[Image reference] {att.url}")
    return "\n\n".join(lines)


def normalize_messages_for_vllm(
    messages: List[ChatCompletionMessageParam],
    model_id: str,
    user_id: str = "",
    vision_enabled_override: Optional[bool] = None
) -> List[ChatCompletionMessageParam]:
    # If the target model is vision-capable, keep structured parts intact
    vision_capable = get_vision_capability_from_request(user_id, model_id, vision_enabled_override)
    if vision_capable:
        return messages
    normalized: List[ChatCompletionMessageParam] = []

    for m in messages:
        content = m.get("content")

        if isinstance(content, str):
            normalized.append(m)
            continue

        if isinstance(content, list):
            parts: List[str] = []
            for p in content:
                if isinstance(p, dict):
                    if p.get("type") == "text":
                        parts.append(p.get("text", ""))
                    elif p.get("type") == "image_url":
                        url = p.get("image_url", {}).get("url", "")
                        if url:
                            parts.append(f"[Image] {url}")

            normalized.append(
                cast(ChatCompletionMessageParam, {
                    "role": m.get("role", "user"),
                    "content": "\n".join(parts),
                })
            )
            continue

        normalized.append(
            cast(ChatCompletionMessageParam, {
                "role": m.get("role", "user"),
                "content": str(content),
            })
        )

    return normalized


def perform_web_search(query: str, max_results: int = 5) -> str:
    """
    Perform web search using Tavily and return formatted results.
    Returns empty string if search fails or is not configured.
    """
    if not tavily_client:
        return ""
    
    try:
        response = tavily_client.search(
            query=query,
            search_depth="basic",  # "basic" for faster, "advanced" for more thorough
            max_results=max_results,
            include_answer=True,  # Get a direct answer if available
        )
        
        # Format results for LLM context
        parts = []
        
        # Include direct answer if available
        if response.get("answer"):
            parts.append(f"**Direct Answer:** {response['answer']}")
        
        # Include search results
        results = response.get("results", [])
        if results:
            parts.append("\n**Web Search Results:**")
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                url = r.get("url", "")
                content = r.get("content", "")[:500]  # Limit content length
                parts.append(f"\n{i}. **{title}**\n   URL: {url}\n   {content}")
        
        return "\n".join(parts) if parts else ""
    
    except Exception as e:
        print(f"Web search failed: {e}")
        return ""


def build_messages(user_id: str, user_message: str, attachments: Optional[List[Attachment]] = None, model_id: str = DEFAULT_MODEL, vision_enabled_override: Optional[bool] = None, web_search: bool = False, rag_enabled: bool = False) -> List[ChatCompletionMessageParam]:
    history = user_histories.get(user_id)
    if history is None:
        history = cast(List[ChatCompletionMessageParam], [{"role": "system", "content": SYSTEM_PROMPT}])

    # Keep system message (first) and only last N turns to reduce context growth
    system_msg: List[ChatCompletionMessageParam] = []
    tail: List[ChatCompletionMessageParam] = []
    if history and isinstance(history[0], dict) and history[0].get("role") == "system":
        system_msg = [history[0]]  # type: ignore[index]
        tail = history[1:]
    else:
        tail = history

    # Each turn is two messages (user+assistant); keep last N turns per model type
    max_tail_msgs = get_max_history_turns_for_model(model_id) * 2
    trimmed_tail = tail[-max_tail_msgs:]

    messages: List[ChatCompletionMessageParam] = [*system_msg, *trimmed_tail]
    
    # Augment user message with context from various sources
    augmented_message = user_message
    context_parts = []
    
    # Add RAG context if enabled
    if rag_enabled:
        rag_context = perform_rag_search(user_id, user_message)
        if rag_context:
            context_parts.append(rag_context)
    
    # Add web search results if enabled
    if web_search:
        search_results = perform_web_search(user_message)
        if search_results:
            context_parts.append(f"**[Web Search Context]**\n{search_results}")
    
    # Combine contexts
    if context_parts:
        combined_context = "\n\n---\n".join(context_parts)
        augmented_message = f"{user_message}\n\n---\n{combined_context}\n---\n\nPlease use the above context to help answer my question. Cite sources when relevant."
    
    user_content: Any = _build_user_content(augmented_message, attachments, model_id, user_id, vision_enabled_override)
    messages.append({"role": "user", "content": user_content})
    return messages


def _parse_allowed_tokens_from_error(msg: str) -> int | None:
    # Typical format:
    # "This model's maximum context length is 1024 tokens and your request has 899 input tokens (256 > 1024 - 899)."
    try:
        max_ctx_match = re.search(r"maximum context length is (\d+) tokens", msg)
        input_match = re.search(r"your request has (\d+) input tokens", msg)
        if not max_ctx_match or not input_match:
            return None
        max_ctx = int(max_ctx_match.group(1))
        input_tokens = int(input_match.group(1))
        allowed = max_ctx - input_tokens - CONTEXT_MARGIN
        return allowed if allowed > 0 else 0
    except Exception:
        return None


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    model_name = (req.model or DEFAULT_MODEL).strip()
    if not model_name:
        raise HTTPException(status_code=400, detail="No model selected. Please pick a model from the dropdown.")
    validate_attachments_for_model(model_name, req.attachments, req.user_id, req.vision_enabled)
    messages = build_messages(req.user_id, req.message, req.attachments, model_name, req.vision_enabled, req.web_search or False, req.rag_enabled or False)
    messages = normalize_messages_for_vllm(messages, model_name, req.user_id, req.vision_enabled)
    max_tokens = DEFAULT_MAX_TOKENS
    # Reserve some tokens for suffix if we hit the length limit
    effective_max_tokens = max(16, max_tokens - SUFFIX_MARGIN_TOKENS)

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=effective_max_tokens,
            temperature=0.8,
            top_p=0.95,
        )
    except BadRequestError as e:
        allowed = _parse_allowed_tokens_from_error(str(e))
        if allowed is None:
            # Fallback: halve and retry once
            effective_max_tokens = max(16, effective_max_tokens // 2)
        else:
            effective_max_tokens = max(16, min(effective_max_tokens, allowed - max(0, SUFFIX_MARGIN_TOKENS)))

        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=effective_max_tokens,
            temperature=0.8,
            top_p=0.95,
        )
    except (NotFoundError, Exception) as e:
        # Fallback to web search when model is unavailable
        if tavily_client:
            search_results = perform_web_search(req.message)
            if search_results:
                fallback_answer = f"**Model is not available, results are from web:**\n\n{search_results}"
                return ChatResponse(
                    reply=fallback_answer,
                    prompt_tokens=None,
                    completion_tokens=None,
                    total_tokens=None,
                    duration_ms=None,
                    tokens_per_sec=None,
                )
        # If no web search available or failed, raise the original error
        if isinstance(e, NotFoundError):
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' not found on server. Choose an available model.")
        raise HTTPException(status_code=500, detail=f"Model unavailable and web search fallback failed: {str(e)}")

    finish_reason = getattr(resp.choices[0], "finish_reason", None)
    answer = resp.choices[0].message.content or ""
    if finish_reason == "length" and TRUNCATION_SUFFIX:
        answer = f"{answer}{TRUNCATION_SUFFIX}"

    # Usage metrics (if provided by server)
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)
    # tokens/sec not precisely known; approximate via server timing not available
    # Let duration be None for non-streaming unless later instrumented server-side

    if req.user_id not in user_histories:
        user_histories[req.user_id] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": req.message},
            {"role": "assistant", "content": answer},
        ]
    else:
        user_histories[req.user_id].append({"role": "user", "content": req.message})
        user_histories[req.user_id].append({"role": "assistant", "content": answer})

    # Prune stored history to respect max turns per selected model
    hist = user_histories.get(req.user_id)
    if hist:
        system_msg: List[ChatCompletionMessageParam] = []
        tail: List[ChatCompletionMessageParam] = []
        if isinstance(hist[0], dict) and hist[0].get("role") == "system":
            system_msg = [hist[0]]  # type: ignore[index]
            tail = hist[1:]
        else:
            tail = hist
        max_tail_msgs = get_max_history_turns_for_model(model_name, req.user_id, req.vision_enabled) * 2
        trimmed_tail = tail[-max_tail_msgs:]
        user_histories[req.user_id] = [*system_msg, *trimmed_tail]

    return ChatResponse(
        reply=answer,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        duration_ms=None,
        tokens_per_sec=(completion_tokens / 1.0) if completion_tokens else None,
    )


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    model_name = (req.model or DEFAULT_MODEL).strip()
    if not model_name:
        def gen_err():
            yield "No model selected. Please pick a model from the dropdown."
        return StreamingResponse(gen_err(), media_type="text/plain")
    validate_attachments_for_model(model_name, req.attachments, req.user_id, req.vision_enabled)
    messages = build_messages(req.user_id, req.message, req.attachments, model_name, req.vision_enabled, req.web_search or False, req.rag_enabled or False)
    messages = normalize_messages_for_vllm(messages, model_name, req.user_id, req.vision_enabled)

    def token_generator():
        buffer: List[str] = []
        max_tokens = DEFAULT_MAX_TOKENS
        effective_max_tokens = max(16, max_tokens - SUFFIX_MARGIN_TOKENS)
        last_finish: str | None = None
        import time
        start_t = time.perf_counter()

        def do_stream(toks: int):
            return client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=toks,
                temperature=0.8,
                top_p=0.95,
                stream=True,
            )

        try:
            stream = do_stream(effective_max_tokens)
        except BadRequestError as e:
            allowed = _parse_allowed_tokens_from_error(str(e))
            if allowed is None:
                effective_max_tokens = max(16, effective_max_tokens // 2)
            else:
                effective_max_tokens = max(16, min(effective_max_tokens, allowed - max(0, SUFFIX_MARGIN_TOKENS)))
            try:
                stream = do_stream(effective_max_tokens)
            except BadRequestError:
                yield "Context limit reached. Please start a new chat or /reset."
                return
        except NotFoundError:
            # Fallback to web search when model is unavailable
            if tavily_client:
                search_results = perform_web_search(req.message)
                if search_results:
                    yield f"**Model is not available, results are from web:**\n\n{search_results}"
                    return
            yield f"Model '{model_name}' not found on server. Choose an available model."
            return
        except Exception as e:
            # Fallback to web search for any other model errors
            if tavily_client:
                search_results = perform_web_search(req.message)
                if search_results:
                    yield f"**Model is not available, results are from web:**\n\n{search_results}"
                    return
            yield f"Model unavailable: {str(e)}"
            return

        for chunk in stream:
            choice0 = chunk.choices[0]
            last_finish = getattr(choice0, "finish_reason", last_finish)
            delta = getattr(choice0, "delta", None)
            if delta is None:
                continue
            piece = getattr(delta, "content", None) or ""
            if piece:
                buffer.append(piece)
                yield piece

        final_answer = "".join(buffer)
        if last_finish == "length" and TRUNCATION_SUFFIX:
            # Append suffix to the stream output
            final_answer = f"{final_answer}{TRUNCATION_SUFFIX}"
            yield TRUNCATION_SUFFIX
        end_t = time.perf_counter()
        dur_ms = int((end_t - start_t) * 1000)
        # Rough token estimate: 4 chars ≈ 1 token
        approx_tokens = max(1, len(final_answer) // 4)
        tps = approx_tokens / max(0.001, (end_t - start_t))
        # Emit a final metrics line that the client can parse (optional)
        yield f"\n\n[throughput] duration_ms={dur_ms} tokens_per_sec={tps:.2f} approx_tokens={approx_tokens}\n"
        if req.user_id not in user_histories:
            user_histories[req.user_id] = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": req.message},
                {"role": "assistant", "content": final_answer},
            ]
        else:
            user_histories[req.user_id].append({"role": "user", "content": req.message})
            user_histories[req.user_id].append({"role": "assistant", "content": final_answer})
        # Prune stored history to respect max turns per selected model
        hist = user_histories.get(req.user_id)
        if hist:
            system_msg: List[ChatCompletionMessageParam] = []
            tail: List[ChatCompletionMessageParam] = []
            if isinstance(hist[0], dict) and hist[0].get("role") == "system":
                system_msg = [hist[0]]  # type: ignore[index]
                tail = hist[1:]
            else:
                tail = hist
            max_tail_msgs = get_max_history_turns_for_model(model_name, req.user_id, req.vision_enabled) * 2
            trimmed_tail = tail[-max_tail_msgs:]
            user_histories[req.user_id] = [*system_msg, *trimmed_tail]

    return StreamingResponse(token_generator(), media_type="text/plain")


class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 5


@app.post("/search")
async def web_search(req: SearchRequest):
    """
    Standalone web search endpoint for testing or direct use.
    """
    if not tavily_client:
        raise HTTPException(status_code=503, detail="Web search is not configured. Set TAVILY_API_KEY.")
    
    try:
        response = tavily_client.search(
            query=req.query,
            search_depth="basic",
            max_results=req.max_results or 5,
            include_answer=True,
        )
        return {
            "query": req.query,
            "answer": response.get("answer"),
            "results": response.get("results", []),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search/status")
def search_status():
    """Check if web search is configured and available."""
    return {
        "available": tavily_client is not None,
        "provider": "tavily" if tavily_client else None,
    }


# ============== RAG Endpoints ==============

class RAGUploadRequest(BaseModel):
    user_id: str


@app.post("/rag/upload")
async def rag_upload(user_id: str, file: UploadFile = File(...)):
    """Upload and index a PDF document for RAG."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported for RAG")
    
    # Save file temporarily
    doc_id = str(uuid.uuid4())
    unique_name = f"{doc_id}{ext}"
    dest = UPLOAD_DIR / unique_name
    
    with dest.open("wb") as out:
        content = await file.read()
        out.write(content)
    
    # Extract text from PDF
    text = extract_text_from_pdf(dest)
    if not text.strip():
        dest.unlink()  # Clean up
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")
    
    # Index the document
    chunk_count = index_document(user_id, doc_id, file.filename, text)
    
    return {
        "status": "ok",
        "doc_id": doc_id,
        "filename": file.filename,
        "chunk_count": chunk_count,
        "text_length": len(text),
    }


@app.get("/rag/documents")
def rag_list_documents(user_id: str):
    """List all documents indexed for a user."""
    docs = get_user_documents(user_id)
    return {"documents": docs}


@app.delete("/rag/documents/{doc_id}")
def rag_delete_document(doc_id: str, user_id: str):
    """Delete a document from the RAG index."""
    success = delete_document(user_id, doc_id)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found or access denied")
    return {"status": "ok", "doc_id": doc_id}


class RAGSearchRequest(BaseModel):
    user_id: str
    query: str
    top_k: Optional[int] = 5


@app.post("/rag/search")
def rag_search(req: RAGSearchRequest):
    """Search indexed documents (for testing)."""
    results = search_documents(req.user_id, req.query, req.top_k or RAG_TOP_K)
    return {"query": req.query, "results": results}


@app.get("/rag/status")
def rag_status():
    """Check RAG system status."""
    doc_count = len(document_metadata)
    return {
        "available": True,
        "embedding_model": RAG_EMBEDDING_MODEL,
        "collection": RAG_COLLECTION_NAME,
        "document_count": doc_count,
    }


# ============== End RAG Endpoints ==============


class ResetRequest(BaseModel):
    user_id: str | None = None


@app.post("/reset")
def reset_chat(req: ResetRequest):
    if req.user_id:
        user_histories.pop(req.user_id, None)
        return {"status": "ok", "cleared": "user", "user_id": req.user_id}
    else:
        user_histories.clear()
        return {"status": "ok", "cleared": "all"}


@app.get("/models")
def list_models():
    items: List[Dict[str, Any]] = []

    try:
        data = client.models.list()
        for m in getattr(data, "data", []) or []:
            mid = getattr(m, "id", None) or getattr(m, "model", None) or ""
            if not isinstance(mid, str) or not mid:
                continue

            vision = probe_vision_capability(openai_api_base, mid)

            items.append({
                "id": mid,
                "vision": vision,
            })

    except Exception:
        pass

    default_id = DEFAULT_MODEL if DEFAULT_MODEL else (items[0]["id"] if items else None)
    return {
        "models": items,
        "default": default_id,
    }


class VisionOverrideRequest(BaseModel):
    model_id: str
    user_id: str
    vision_enabled: bool


@app.post("/vision-override")
def set_vision_override(req: VisionOverrideRequest):
    """
    Allow user to override vision capability detection for a model.
    Stores the override in session-like dict keyed by user_id:model_id.
    """
    override_key = f"{req.user_id}:{req.model_id}"
    vision_capability_overrides[override_key] = req.vision_enabled
    return {"status": "ok", "override_key": override_key, "vision_enabled": req.vision_enabled}


@app.get("/vision-override")
def get_vision_override(user_id: str, model_id: str):
    """
    Retrieve the override for a specific user+model combo, or None if not set.
    """
    override_key = f"{user_id}:{model_id}"
    override = vision_capability_overrides.get(override_key)
    return {"override_key": override_key, "vision_enabled": override}



ALLOWED_EXTENSIONS = {
    ".txt", ".md", ".markdown",
    ".pdf",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"
}


def compress_image(file_path: Path, mime_type: str) -> None:
    """
    Compress image in-place if it exceeds size threshold.
    Resizes large images and re-encodes with quality tuning.
    Thresholds: 500 KB soft limit, 1 MB aggressive limit.
    """
    try:
        file_size = file_path.stat().st_size
        if file_size < IMAGE_SIZE_THRESHOLD:
            return  # No compression needed
        
        img = Image.open(file_path)
        original_mode = img.mode
        
        # Convert RGBA/LA/P to RGB if saving as JPEG (JPEG doesn't support transparency)
        if img.mode in ("RGBA", "LA", "P"):
            if mime_type in ("image/jpeg", "application/octet-stream"):
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA":
                    rgb_img.paste(img, mask=img.split()[-1])
                else:
                    rgb_img.paste(img)
                img = rgb_img
        
        # Resize if dimensions are too large
        if img.width > IMAGE_MAX_DIMENSION or img.height > IMAGE_MAX_DIMENSION:
            img.thumbnail((IMAGE_MAX_DIMENSION, IMAGE_MAX_DIMENSION), Image.Resampling.LANCZOS)
        
        # Determine output format and quality
        if mime_type == "image/png":
            output_format = "PNG"
            save_kwargs = {"optimize": True}
        elif mime_type == "image/webp":
            output_format = "WEBP"
            save_kwargs = {"quality": IMAGE_QUALITY}
        else:  # Default to JPEG for unknown/JPEG types
            output_format = "JPEG"
            save_kwargs = {"quality": IMAGE_QUALITY, "optimize": True}
        
        # Save to in-memory buffer first to check size
        buffer = io.BytesIO()
        img.save(buffer, format=output_format, **save_kwargs)
        compressed_size = buffer.tell()
        
        # Only write back if compressed size is smaller
        if compressed_size < file_size:
            buffer.seek(0)
            file_path.write_bytes(buffer.getvalue())
    
    except Exception as e:
        # Log error but don't fail upload if compression fails
        print(f"Warning: Image compression failed for {file_path.name}: {e}")


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    saved = []
    for uf in files:
        name = os.path.basename(uf.filename or "")
        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"File type not allowed: {ext}")

        # Use unique filename to avoid collisions
        unique = f"{os.urandom(8).hex()}{ext}"
        dest = UPLOAD_DIR / unique
        # Persist file to disk
        with dest.open("wb") as out:
            while True:
                chunk = await uf.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)

        # Compress image if it exceeds size threshold
        if ext in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
            compress_image(dest, uf.content_type or "application/octet-stream")

        item = {
            "filename": name,
            "url": f"/uploads/{unique}",
            "mime_type": uf.content_type or "application/octet-stream",
        }

        # If text file, attach a text preview
        if ext in {".txt", ".md", ".markdown"}:
            try:
                txt = (UPLOAD_DIR / unique).read_text(encoding="utf-8", errors="ignore")
                item["text"] = txt[:20000]
            except Exception:
                pass

        # PDFs and images are stored and referenced via URL; no extraction here
        saved.append(item)

    return {"files": saved}