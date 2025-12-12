"""
VRChat tools for Gabriel

Expose function calls for:
- list_vrchat_friend_requests
- accept_vrchat_friend_request
- deny_vrchat_friend_request

Configuration is read from environment variables or config.yml under `vrchat`.
Never hardcode credentials; use env vars VRC_USERNAME/VRC_PASSWORD and optional VRC_TOTP_SECRET.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from google.genai import types
import os
import json
import time

from vrchatapi import build_client_from_env_or_config, VRChatAPIError

logger = logging.getLogger(__name__)


# Module-level rate limiting state
_LAST_VRCHAT_CALL_TS: float = 0.0
_VRCHAT_RATE_LIMIT_SECONDS: float = float(os.environ.get("VRCHAT_TOOL_RATE_LIMIT_SECONDS", "30") or 30)


AVATAR_STORE_PATH = os.path.join(os.getcwd(), "avatars.json")
LAST_AVATAR_PATH = os.path.join(os.getcwd(), "last_avatar.json")

VRCHAT_FUNCTION_DECLARATIONS = [
    {
        "name": "list_vrchat_friend_requests",
        "description": "List incoming VRChat friend requests (notification type=friendRequest).",
        "parameters": {
            "type": "object",
            "properties": {
                "include_hidden": {"type": "boolean", "description": "Include hidden friend request notifications", "default": False},
                "limit": {"type": "integer", "description": "Max number of notifications to fetch (1-100)", "default": 60}
            }
        }
    },
    {
        "name": "accept_vrchat_friend_request",
        "description": "Accept an incoming VRChat friend request by its notification id (frq_...).",
        "parameters": {
            "type": "object",
            "properties": {
                "notification_id": {"type": "string", "description": "The notification ID (frq_...) to accept"}
            },
            "required": ["notification_id"]
        }
    },
    {
        "name": "deny_vrchat_friend_request",
        "description": "Deny (hide) an incoming VRChat friend request by its notification id (frq_...).",
        "parameters": {
            "type": "object",
            "properties": {
                "notification_id": {"type": "string", "description": "The notification ID (frq_...) to deny/hide"}
            },
            "required": ["notification_id"]
        }
    },
    {
        "name": "get_own_avatar",
        "description": "Get the currently equipped avatar for the authenticated user and save it into a local avatars list.",
        "parameters": {
            "type": "object",
            "properties": {
                "save": {"type": "boolean", "description": "If true, save/update the avatar in the local avatars list", "default": True}
            }
        }
    },
    {
        "name": "select_avatar",
        "description": "Switch to an avatar by its ID. If the ID exists in the local avatars list, metadata will be returned.",
        "parameters": {
            "type": "object",
            "properties": {
                "avatar_id": {"type": "string", "description": "The avatar ID (avtr_...) to select"}
            },
            "required": ["avatar_id"]
        }
    },
    {
        "name": "list_saved_avatars",
        "description": "List avatars saved locally by get_own_avatar. Returns id, name, author, images, and tags.",
        "parameters": {"type": "object", "properties": {}}
    },
]

# Cached authenticated VRChat client to avoid re-login and repeated 2FA on every call
_CACHED_VRCHAT_INIT: Optional[Dict[str, Any]] = None


def _get_vrchat_client_cached(force_refresh: bool = False) -> Dict[str, Any]:
    """Get a cached authenticated VRChat client; build it if missing or force_refresh.

    Returns the same dict structure as build_client_from_env_or_config.
    """
    global _CACHED_VRCHAT_INIT
    if not force_refresh and _CACHED_VRCHAT_INIT and _CACHED_VRCHAT_INIT.get("success") and _CACHED_VRCHAT_INIT.get("client"):
        return _CACHED_VRCHAT_INIT

    init = build_client_from_env_or_config()
    if init.get("success"):
        _CACHED_VRCHAT_INIT = init
    else:
        _CACHED_VRCHAT_INIT = None
    return init


def _clear_vrchat_client_cache() -> None:
    global _CACHED_VRCHAT_INIT
    _CACHED_VRCHAT_INIT = None


async def handle_vrchat_function_calls(function_call) -> types.FunctionResponse:
    global _LAST_VRCHAT_CALL_TS, _VRCHAT_RATE_LIMIT_SECONDS
    name = function_call.name
    args = function_call.args or {}

    # Global rate-limiter: 30s between any VRChat calls to avoid spamming
    now = time.time()
    if _LAST_VRCHAT_CALL_TS and (now - _LAST_VRCHAT_CALL_TS) < _VRCHAT_RATE_LIMIT_SECONDS:
        wait = int(_VRCHAT_RATE_LIMIT_SECONDS - (now - _LAST_VRCHAT_CALL_TS))
        return types.FunctionResponse(
            id=function_call.id,
            name=name,
            response={
                "success": False,
                "message": f"VRChat tools are rate-limited. Please wait {wait}s before trying again.",
                "rate_limited": True,
                "retry_after_seconds": wait,
            },
        )

    # Use a cached authenticated client so we don't re-login and trigger 2FA every call.
    init = _get_vrchat_client_cached()
    if not init.get("success"):
        return types.FunctionResponse(
            id=function_call.id,
            name=name,
            response={"success": False, "message": init.get("message", "Failed to initialize VRChat client")},
        )

    client = init["client"]

    try:
        if name == "list_vrchat_friend_requests":
            include_hidden = bool(args.get("include_hidden", False))
            limit = int(args.get("limit", 60))
            items = client.list_friend_requests(include_hidden=include_hidden, n=limit)
            # Normalize output to essential fields
            simplified = [
                {
                    "id": it.get("id"),
                    "type": it.get("type"),
                    "senderUserId": it.get("senderUserId"),
                    "senderUsername": it.get("senderUsername"),
                    "message": it.get("message"),
                    "created_at": it.get("created_at"),
                    "seen": it.get("seen"),
                }
                for it in items
            ]
            result: Dict[str, Any] = {"success": True, "requests": simplified, "count": len(simplified)}

        elif name == "accept_vrchat_friend_request":
            nid = args.get("notification_id")
            if not nid:
                result = {"success": False, "message": "notification_id is required"}
            else:
                result = client.accept_friend_request(nid)
                if isinstance(result, dict) and result.get("status_code") in (401, 403):
                    _clear_vrchat_client_cache()
                    init2 = _get_vrchat_client_cached(force_refresh=True)
                    if init2.get("success"):
                        client = init2["client"]
                        result = client.accept_friend_request(nid)

        elif name == "deny_vrchat_friend_request":
            nid = args.get("notification_id")
            if not nid:
                result = {"success": False, "message": "notification_id is required"}
            else:
                result = client.deny_friend_request(nid)
                if isinstance(result, dict) and result.get("status_code") in (401, 403):
                    _clear_vrchat_client_cache()
                    init2 = _get_vrchat_client_cached(force_refresh=True)
                    if init2.get("success"):
                        client = init2["client"]
                        result = client.deny_friend_request(nid)
        elif name == "get_own_avatar":
            save_flag = bool(args.get("save", True))
            try:
                avatar = client.get_own_avatar()
            except VRChatAPIError as e:
                # If unauthorized, clear cache and retry once
                if "unauthorized" in str(e).lower() or "401" in str(e) or "403" in str(e):
                    _clear_vrchat_client_cache()
                    init2 = _get_vrchat_client_cached(force_refresh=True)
                    if not init2.get("success"):
                        raise
                    client = init2["client"]
                    avatar = client.get_own_avatar()
                else:
                    raise
            result = {"success": True, "avatar": _simplify_avatar(avatar)}
            if save_flag and isinstance(avatar, dict):
                try:
                    saved = _load_avatar_store()
                    entry = _avatar_entry_from_avatar(avatar)
                    # upsert by id
                    updated = False
                    for i, existing in enumerate(saved):
                        if existing.get("id") == entry["id"]:
                            saved[i] = entry
                            updated = True
                            break
                    if not updated:
                        saved.append(entry)
                    _save_avatar_store(saved)
                    _save_last_avatar(entry)
                    result["saved_count"] = len(saved)
                except Exception as e:
                    result["save_error"] = str(e)

        elif name == "select_avatar":
            avatar_id = str(args.get("avatar_id") or "").strip()
            if not avatar_id:
                result = {"success": False, "message": "avatar_id is required"}
            else:
                sel = client.select_avatar(avatar_id)
                # If unauthorized, clear cache and retry once
                if isinstance(sel, dict) and sel.get("status_code") in (401, 403):
                    _clear_vrchat_client_cache()
                    init2 = _get_vrchat_client_cached(force_refresh=True)
                    if init2.get("success"):
                        client = init2["client"]
                        sel = client.select_avatar(avatar_id)
                ok = bool(sel.get("success")) if isinstance(sel, dict) else False
                # Try to enrich with local metadata if present
                meta = None
                try:
                    for it in _load_avatar_store():
                        if it.get("id") == avatar_id:
                            meta = it
                            break
                except Exception:
                    pass
                # If switch succeeded, record last used avatar (prefer metadata if available)
                if ok:
                    try:
                        entry = meta or {"id": avatar_id, "name": None, "authorName": None}
                        _save_last_avatar(entry)
                    except Exception:
                        pass
                result = {"success": ok, "response": sel, "metadata": meta}

        elif name == "list_saved_avatars":
            items = _load_avatar_store()
            result = {"success": True, "avatars": items, "count": len(items), "path": AVATAR_STORE_PATH}

        else:
            result = {"success": False, "message": f"Unknown VRChat function: {name}"}

        # Set rate-limit timestamp on successful or handled result to avoid spamming
        _LAST_VRCHAT_CALL_TS = time.time()
        return types.FunctionResponse(id=function_call.id, name=name, response=result)

    except VRChatAPIError as e:
        logger.error(f"VRChat API error in {name}: {e}")
        # If unauthorized, clear cached client so a future call can re-authenticate
        try:
            if "unauthorized" in str(e).lower() or "401" in str(e) or "403" in str(e):
                global _CACHED_VRCHAT_INIT
                _CACHED_VRCHAT_INIT = None
        except Exception:
            pass
        _LAST_VRCHAT_CALL_TS = time.time()
        return types.FunctionResponse(
            id=function_call.id,
            name=name,
            response={"success": False, "message": str(e)},
        )
    except Exception as e:
        logger.error(f"Unhandled error in VRChat tool {name}: {e}")
        _LAST_VRCHAT_CALL_TS = time.time()
        return types.FunctionResponse(
            id=function_call.id,
            name=name,
            response={"success": False, "message": f"Error executing {name}: {e}"},
        )


def _load_avatar_store() -> List[Dict[str, Any]]:
    if not os.path.exists(AVATAR_STORE_PATH):
        return []
    try:
        with open(AVATAR_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        # migrate old dict format if any
        if isinstance(data, dict) and "avatars" in data and isinstance(data["avatars"], list):
            return data["avatars"]
        return []
    except Exception:
        return []


def _save_avatar_store(items: List[Dict[str, Any]]) -> None:
    tmp_path = AVATAR_STORE_PATH + ".tmp"
    os.makedirs(os.path.dirname(AVATAR_STORE_PATH) or ".", exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, AVATAR_STORE_PATH)


def _save_last_avatar(entry: Dict[str, Any]) -> None:
    data = {
        "id": entry.get("id"),
        "name": entry.get("name"),
        "authorName": entry.get("authorName"),
        "imageUrl": entry.get("imageUrl") or entry.get("thumbnailImageUrl"),
        "saved_at": int(time.time()),
    }
    tmp_path = LAST_AVATAR_PATH + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, LAST_AVATAR_PATH)


def _avatar_entry_from_avatar(avatar: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": avatar.get("id"),
        "name": avatar.get("name"),
        "authorId": avatar.get("authorId"),
        "authorName": avatar.get("authorName"),
        "imageUrl": avatar.get("imageUrl") or avatar.get("thumbnailImageUrl"),
        "thumbnailImageUrl": avatar.get("thumbnailImageUrl") or avatar.get("imageUrl"),
        "releaseStatus": avatar.get("releaseStatus"),
        "tags": avatar.get("tags", []),
        "updated_at": avatar.get("updated_at") or avatar.get("created_at"),
    }


def _simplify_avatar(avatar: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return _avatar_entry_from_avatar(avatar)
    except Exception:
        return {"id": avatar.get("id"), "name": avatar.get("name")}

