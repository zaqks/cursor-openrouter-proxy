#!/usr/bin/env python3
"""
Cursor OpenRouter Proxy Server
A high-performance HTTP/2-enabled proxy server that enables Cursor IDE to use any LLM
available through OpenRouter.
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S.%f",
)
logger = logging.getLogger(__name__)

# Constants
OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o"

# Load environment variables
load_dotenv()


# Global configuration
class Config:
    def __init__(self):
        self.endpoint = OPENROUTER_ENDPOINT
        self.model = os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")

        # Validate API key
        if not self.api_key.startswith("sk-or-"):
            logger.error("OPENROUTER_API_KEY must start with 'sk-or-'")
            sys.exit(1)
        if len(self.api_key) < 32:
            logger.error("OPENROUTER_API_KEY seems too short to be valid")
            sys.exit(1)

        # Validate model
        if self.model and "/" not in self.model:
            logger.error(
                f"Invalid model: {self.model}. Must contain a provider prefix (e.g. openai/gpt-4o)"
            )
            sys.exit(1)

        logger.info(
            f"Initialized Cursor-OpenRouter proxy with model: {self.model} using endpoint: {self.endpoint}"
        )


active_config = Config()
debug_mode = os.getenv("DEBUG", "false").lower() == "true"

# HTTP client with connection pooling
http_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app"""
    global http_client

    # Startup: Create HTTP client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(300.0),  # 5 minutes
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        http2=True,
    )
    logger.info("HTTP client initialized")

    yield

    # Shutdown: Close HTTP client
    if http_client:
        await http_client.aclose()
        logger.info("HTTP client closed")


# Initialize FastAPI app
app = FastAPI(
    title="Cursor OpenRouter Proxy",
    description="Proxy server for Cursor IDE to use OpenRouter models",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Length"],
)


# Pydantic models
class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None


class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class Tool(BaseModel):
    type: str
    function: Function


class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    functions: Optional[List[Function]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Any] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


class OpenRouterRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[str] = None


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[Model]


class ConfigUpdateRequest(BaseModel):
    model: str


# Helper functions
def debug_log(message: str, *args):
    """Log debug messages if debug mode is enabled"""
    if debug_mode:
        logger.debug(message, *args)


def mask_api_key(key: str) -> str:
    """Mask API key for logging"""
    if len(key) <= 12:
        return "***"
    return f"{key[:6]}...{key[-6:]}"


def convert_tool_choice(choice: Any) -> str:
    """Convert tool choice to OpenRouter format"""
    if choice is None:
        return ""

    if isinstance(choice, str):
        if choice in ["auto", "none"]:
            return choice

    if isinstance(choice, dict):
        if choice.get("type") == "function":
            return "auto"  # OpenRouter doesn't support specific function selection

    return ""


def convert_messages(messages: List[Message]) -> List[Message]:
    """Convert messages to OpenRouter format"""
    converted = []

    for i, msg in enumerate(messages):
        logger.info(f"Converting message {i} - Role: {msg.role}")
        new_msg = msg.model_copy()

        # Handle function response messages
        if msg.role == "function":
            logger.info("Converting function response to tool response")
            new_msg.role = "tool"

        converted.append(new_msg)

    return converted


def truncate_string(s: str, max_len: int = 50) -> str:
    """Truncate string for logging"""
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


# Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test OpenRouter connection
        headers = {
            "Authorization": f"Bearer {active_config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/pezzos/cursor-proxy",
            "X-Title": "Cursor Proxy",
        }

        response = await http_client.get(
            f"{OPENROUTER_ENDPOINT}/models",
            headers=headers,
        )

        if response.status_code != 200:
            logger.error(
                f"Health check failed with status {response.status_code}: {response.text}"
            )
            raise HTTPException(
                status_code=response.status_code,
                detail=f"OpenRouter returned {response.status_code}",
            )

        return {
            "status": "ok",
            "endpoint": OPENROUTER_ENDPOINT,
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Connection failed")


@app.get("/v1/models")
async def get_models():
    """Get available models from OpenRouter"""
    try:
        headers = {
            "Content-Type": "application/json",
        }

        response = await http_client.get(
            f"{OPENROUTER_ENDPOINT}/models",
            headers=headers,
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=response.status_code,
                detail="Failed to fetch models",
            )

        return Response(
            content=response.content,
            media_type="application/json",
            status_code=response.status_code,
        )
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail="Error fetching models")


@app.get("/v1/config")
async def get_config():
    """Get current configuration"""
    return {"model": active_config.model}


@app.post("/v1/config")
async def update_config(config: ConfigUpdateRequest):
    """Update model configuration"""
    if not config.model:
        raise HTTPException(status_code=400, detail="Model is required")

    active_config.model = config.model
    logger.info(f"Updated model to: {active_config.model}")

    return {
        "status": "success",
        "model": active_config.model,
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Handle chat completions requests"""
    debug_log(f"Received request: POST /v1/chat/completions")

    # Validate authorization header
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header",
        )

    user_api_key = authorization.replace("Bearer ", "").strip()
    if not user_api_key.startswith("sk-"):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key format",
        )

    # Parse request body
    try:
        body = await request.json()
        chat_req = ChatRequest(**body)
    except Exception as e:
        logger.error(f"Error parsing request JSON: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON")

    logger.info(f"Parsed request: model={chat_req.model}, stream={chat_req.stream}")

    # Always use the configured model from .env, regardless of what Cursor sends
    logger.info(
        f"Using configured model: {active_config.model} (endpoint: {active_config.endpoint})"
    )
    original_model = chat_req.model
    chat_req.model = active_config.model
    logger.info(f"Model converted from {original_model} to: {active_config.model}")

    # Convert to OpenRouter request format
    openrouter_req = OpenRouterRequest(
        model=active_config.model,
        messages=convert_messages(chat_req.messages),
        stream=chat_req.stream,
    )

    # Apply model-specific adjustments
    if active_config.model.startswith("mistralai/") or active_config.model.startswith(
        "google/"
    ):
        if chat_req.temperature is not None:
            openrouter_req.temperature = min(chat_req.temperature, 1.0)
    else:
        openrouter_req.temperature = chat_req.temperature

    if chat_req.max_tokens is not None:
        openrouter_req.max_tokens = chat_req.max_tokens

    # Handle tools/functions
    if chat_req.tools:
        openrouter_req.tools = chat_req.tools
        tool_choice = convert_tool_choice(chat_req.tool_choice)
        if tool_choice:
            openrouter_req.tool_choice = tool_choice
    elif chat_req.functions:
        openrouter_req.tools = [
            Tool(type="function", function=fn) for fn in chat_req.functions
        ]
        tool_choice = convert_tool_choice(chat_req.tool_choice)
        if tool_choice:
            openrouter_req.tool_choice = tool_choice

    # Create request body
    modified_body = openrouter_req.model_dump(exclude_none=True)
    logger.info(f"Modified request body: {json.dumps(modified_body, indent=2)}")

    # Create proxy request
    target_url = f"{active_config.endpoint}/chat/completions"
    headers = {
        "Authorization": f"Bearer {active_config.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream" if chat_req.stream else "application/json",
        "User-Agent": "cursor-proxy/1.0",
        "HTTP-Referer": "https://github.com/pezzos/cursor-proxy",
        "X-Title": "Cursor Proxy",
        "OpenAI-Organization": "cursor-proxy",
    }

    # Model-specific headers
    if active_config.model.startswith("mistralai/"):
        headers["X-Model-Provider"] = "mistral"
    elif active_config.model.startswith("google/"):
        headers["X-Model-Provider"] = "google"

    try:
        if chat_req.stream:
            return await handle_streaming_response(target_url, modified_body, headers)
        else:
            return await handle_regular_response(target_url, modified_body, headers)
    except Exception as e:
        logger.error(f"Error forwarding request: {e}")
        raise HTTPException(status_code=502, detail="Error forwarding request")


async def handle_streaming_response(
    target_url: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
) -> StreamingResponse:
    """Handle streaming response from OpenRouter"""
    debug_log("Starting streaming response handling")

    async def event_generator() -> AsyncIterator[bytes]:
        try:
            async with http_client.stream(
                "POST",
                target_url,
                json=body,
                headers=headers,
            ) as response:
                logger.info(f"OpenRouter response status: {response.status_code}")

                if response.status_code >= 400:
                    error_body = await response.aread()
                    logger.error(f"Error response: {error_body.decode()}")
                    yield f"data: {error_body.decode()}\n\n".encode()
                    return

                async for line in response.aiter_lines():
                    if line.strip():
                        yield f"{line}\n".encode()

        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n".encode()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


async def handle_regular_response(
    target_url: str,
    body: Dict[str, Any],
    headers: Dict[str, str],
) -> Response:
    """Handle regular (non-streaming) response from OpenRouter"""
    debug_log("Handling regular (non-streaming) response")

    response = await http_client.post(
        target_url,
        json=body,
        headers=headers,
    )

    logger.info(f"OpenRouter response status: {response.status_code}")

    if response.status_code >= 400:
        error_body = response.json()
        logger.error(f"Error response: {error_body}")

        return Response(
            content=json.dumps(error_body),
            media_type="application/json",
            status_code=response.status_code,
        )

    # Parse OpenRouter response
    try:
        openrouter_resp = response.json()
        debug_log(f"Original response: {json.dumps(openrouter_resp, indent=2)}")

        # Check for errors
        if "error" in openrouter_resp:
            return Response(
                content=json.dumps(openrouter_resp),
                media_type="application/json",
                status_code=response.status_code,
            )

        # Convert to OpenAI format
        openai_resp = {
            "id": openrouter_resp.get("id"),
            "object": "chat.completion",
            "created": openrouter_resp.get("created"),
            "model": openrouter_resp.get("model", active_config.model),
            "choices": openrouter_resp.get("choices", []),
            "usage": openrouter_resp.get("usage", {}),
        }

        modified_body = json.dumps(openai_resp)
        debug_log(f"Modified response: {modified_body}")

        return Response(
            content=modified_body,
            media_type="application/json",
            status_code=response.status_code,
        )
    except Exception as e:
        logger.error(f"Error parsing response: {e}")
        raise HTTPException(status_code=500, detail="Error parsing response")


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        log_level="info",
    )
