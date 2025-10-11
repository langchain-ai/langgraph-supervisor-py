"""
User profile tooling that wraps YouTube Data API access and OpenAI embeddings.

Exposes three LangChain tools:
- yt_list_liked_videos
- yt_list_subscriptions
- embed_user_preferences
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence

from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3
DEFAULT_MAX_RESULTS = 50
SCOPES_ENV = "YOUTUBE_SCOPES"
TOKEN_ENV = "YOUTUBE_TOKEN_PATH"
DEFAULT_SCOPE = "https://www.googleapis.com/auth/youtube.readonly"

try:  # Optional dependencies; tests monkeypatch the YouTube client to avoid importing these.
    from google.auth.exceptions import RefreshError  # type: ignore[import]
except ImportError:

    class RefreshError(Exception):  # type: ignore[override]
        """Fallback when google-auth is not installed."""


try:
    from google.oauth2.credentials import Credentials  # type: ignore[import]
    from google.auth.transport.requests import Request  # type: ignore[import]
except ImportError:  # pragma: no cover - handled by raising at runtime if actually needed
    Credentials = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]


try:
    from googleapiclient.discovery import build  # type: ignore[import]
    from googleapiclient.errors import HttpError  # type: ignore[import]
except ImportError:  # pragma: no cover - handled by raising at runtime if actually needed
    build = None  # type: ignore[assignment]

    class HttpError(Exception):  # type: ignore[override]
        """Lightweight stand-in to simplify testing without google-api-python-client."""

        def __init__(self, resp: Any, content: Any, uri: str | None = None) -> None:
            super().__init__(content)
            self.resp = resp
            self.content = content
            self.uri = uri

        @property
        def status_code(self) -> Optional[int]:
            return getattr(self.resp, "status", None)


def _load_scopes() -> Sequence[str]:
    raw = os.getenv(SCOPES_ENV)
    if not raw:
        return (DEFAULT_SCOPE,)
    scopes = [scope.strip() for scope in raw.split(",") if scope.strip()]
    return scopes or (DEFAULT_SCOPE,)


def _token_path() -> str:
    return os.getenv(TOKEN_ENV, "token.json")


def _ensure_dependencies() -> None:
    if Credentials is None or Request is None:
        raise RuntimeError(
            "google-auth is required. Install google-auth and google-auth-oauthlib to use "
            "the YouTube tools."
        )
    if build is None:
        raise RuntimeError(
            "google-api-python-client is required. Install google-api-python-client to use "
            "the YouTube tools."
        )


@lru_cache(maxsize=1)
def _get_youtube_client():
    """Return an authorized YouTube Data API client, refreshing the token if needed."""
    _ensure_dependencies()

    scopes = _load_scopes()
    token_path = _token_path()
    if not os.path.exists(token_path):
        raise FileNotFoundError(
            f"OAuth token not found at '{token_path}'. "
            "Complete the authorization flow and save the token file."
        )

    creds = Credentials.from_authorized_user_file(token_path, scopes)  # type: ignore[misc]
    if not creds:
        raise RuntimeError("Unable to load OAuth credentials from token file.")

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())  # type: ignore[misc]
        except RefreshError as exc:  # pragma: no cover - depends on live creds
            raise RuntimeError("Failed to refresh YouTube OAuth credentials.") from exc
        with open(token_path, "w", encoding="utf-8") as fh:
            fh.write(creds.to_json())

    if not creds.valid:
        raise RuntimeError("YouTube credentials are not valid and could not be refreshed.")

    return build("youtube", "v3", credentials=creds, cache_discovery=False)  # type: ignore[misc]


def _coerce_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _http_status(err: BaseException) -> Optional[int]:
    for attr in ("status_code", "status"):
        status = getattr(err, attr, None)
        if isinstance(status, int):
            return status

    resp = getattr(err, "resp", None)
    status = getattr(resp, "status", None)
    return status if isinstance(status, int) else None


def _error_payload(err: BaseException) -> Dict[str, Any]:
    status = _http_status(err)
    message = str(err)
    if len(message) > 240:
        message = message[:237] + "..."
    payload: Dict[str, Any] = {"message": message}
    if status is not None:
        payload["status"] = status
    return payload


def _execute_with_retries(request: Any):
    last_error: Optional[BaseException] = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            return request.execute()
        except HttpError as err:  # type: ignore[misc]
            last_error = err
            status = _http_status(err)
            if status in RETRYABLE_STATUS_CODES and attempt < MAX_ATTEMPTS - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError("Request execution failed unexpectedly.")


def _clamp_max_results(max_results: int) -> int:
    if max_results <= 0:
        return 1
    if max_results > DEFAULT_MAX_RESULTS:
        return DEFAULT_MAX_RESULTS
    return max_results


def _subscription_rows(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items:
        snippet = item.get("snippet", {})
        resource = snippet.get("resourceId", {})
        channel_id = resource.get("channelId")
        if not channel_id:
            continue
        rows.append(
            {
                "channel_id": channel_id,
                "channel_title": snippet.get("title"),
                "subscribed_at": snippet.get("publishedAt"),
            }
        )
    return rows


def _liked_rows(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in items:
        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})
        rows.append(
            {
                "video_id": item.get("id"),
                "title": snippet.get("title"),
                "channel_title": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"),
                "duration": item.get("contentDetails", {}).get("duration"),
                "view_count": _coerce_int(statistics.get("viewCount")),
                "like_count": _coerce_int(statistics.get("likeCount")),
            }
        )
    return rows


def _empty_result(error: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    result: Dict[str, Any] = {"items": [], "next_page_token": None}
    if error:
        result["error"] = error
    return result


@tool("yt_list_subscriptions")
def yt_list_subscriptions(
    max_results: int = DEFAULT_MAX_RESULTS,
    page_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the user's subscribed channels (channel_id, title, subscribed_at)."""
    service = _get_youtube_client()
    request = (
        service.subscriptions()  # type: ignore[no-untyped-call]
        .list(
            part="snippet",
            mine=True,
            maxResults=_clamp_max_results(max_results),
            pageToken=page_token,
            fields="nextPageToken,items(snippet(title,publishedAt,resourceId/channelId))",
        )
    )
    try:
        response = _execute_with_retries(request)
    except HttpError as err:  # type: ignore[misc]
        status = _http_status(err)
        if status in RETRYABLE_STATUS_CODES:
            return _empty_result(_error_payload(err))
        raise

    return {
        "items": _subscription_rows(response.get("items", [])),
        "next_page_token": response.get("nextPageToken"),
    }


@tool("yt_list_liked_videos")
def yt_list_liked_videos(
    max_results: int = DEFAULT_MAX_RESULTS,
    page_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the user's liked videos (video_id, title, channel_title, published_at, duration)."""
    service = _get_youtube_client()
    request = (
        service.videos()  # type: ignore[no-untyped-call]
        .list(
            part="id,snippet,contentDetails,statistics",
            myRating="like",
            maxResults=_clamp_max_results(max_results),
            pageToken=page_token,
            fields=(
                "nextPageToken,items("
                "id,snippet(title,channelTitle,publishedAt),"
                "contentDetails(duration),statistics(viewCount,likeCount)"
                ")"
            ),
        )
    )

    try:
        response = _execute_with_retries(request)
    except HttpError as err:  # type: ignore[misc]
        status = _http_status(err)
        if status in RETRYABLE_STATUS_CODES:
            return _empty_result(_error_payload(err))
        raise

    return {
        "items": _liked_rows(response.get("items", [])),
        "next_page_token": response.get("nextPageToken"),
    }


class EmbedInput(BaseModel):
    text: str = Field(..., description="Text to embed. Concise bullet-style summary is recommended.")
    model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model identifier (e.g., text-embedding-3-small / text-embedding-3-large).",
    )
    normalize: bool = Field(
        default=True,
        description="If true, L2-normalize the output vector (recommended for cosine/dot retrieval).",
    )
    max_chars: Optional[int] = Field(
        default=8000,
        description="Optional hard cap to avoid extremely long inputs; set None to skip.",
    )


@lru_cache(maxsize=4)
def _get_embedder(model: str) -> OpenAIEmbeddings:
    # langchain-openai reads OPENAI_API_KEY from env
    return OpenAIEmbeddings(model=model)


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm > 0 else vec


@tool("embed_user_preferences", args_schema=EmbedInput)
def embed_user_preferences(
    text: str,
    model: str = "text-embedding-3-small",
    normalize: bool = True,
    max_chars: Optional[int] = 8000,
) -> Dict[str, Any]:
    """
    Return an OpenAI embedding vector for the supplied text.

    Returns:
      {
        "model": str,
        "embedding": List[float],
        "dim": int,
        "normalized": bool,
        "error": str | None
      }
    Never raises: on failure returns {"error": "...", "embedding": []}.
    """
    try:
        if not os.getenv("OPENAI_API_KEY"):
            return {
                "model": model,
                "embedding": [],
                "dim": 0,
                "normalized": False,
                "error": "missing_openai_api_key",
            }

        if not isinstance(text, str):
            return {
                "model": model,
                "embedding": [],
                "dim": 0,
                "normalized": False,
                "error": "invalid_text_type",
            }

        clean = text.strip()
        if not clean:
            return {
                "model": model,
                "embedding": [],
                "dim": 0,
                "normalized": False,
                "error": "empty_text",
            }

        if isinstance(max_chars, int) and max_chars > 0 and len(clean) > max_chars:
            clean = clean[:max_chars]

        embedder = _get_embedder(model)
        vec = embedder.embed_query(clean)

        if not isinstance(vec, list) or not vec:
            return {
                "model": model,
                "embedding": [],
                "dim": 0,
                "normalized": False,
                "error": "empty_embedding",
            }

        if normalize:
            vec = _l2_normalize(vec)

        return {
            "model": model,
            "embedding": vec,
            "dim": len(vec),
            "normalized": bool(normalize),
            "error": None,
        }

    except Exception as exc:  # pragma: no cover - defensive fallback
        return {
            "model": model,
            "embedding": [],
            "dim": 0,
            "normalized": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


__all__ = ["yt_list_subscriptions", "yt_list_liked_videos", "embed_user_preferences"]

