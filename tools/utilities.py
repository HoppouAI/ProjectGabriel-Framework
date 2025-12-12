"""
Utility tools for Gabriel
Provides time, note-taking, and mode switching capabilities.
"""

import logging
import os
import time
import hashlib
from datetime import datetime
from typing import Dict, Any
from google.genai import types
from .memory import memory_system
try:  # Optional dependency, already in requirements
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# Set up logging
logger = logging.getLogger(__name__)

_key_injector = None


def _ensure_key_injector():
    global _key_injector
    if _key_injector is None:
        try:
            import pydirectinput as _pdi
            try:
                _pdi.PAUSE = 0.0
            except Exception:
                pass
            _key_injector = _pdi
        except Exception as e:
            raise RuntimeError(f"PyDirectInput not available for keyboard inputs: {e}")
    return _key_injector

# Additional utility function declarations that might be useful
UTILITY_FUNCTION_DECLARATIONS = [
    {
        "name": "get_current_time",
        "description": "Get the current date and time",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "take_note",
    "description": "Take a quick note (timestamped key). Notes are rate-limited and saved as quick_note by default (auto-clears in ~6 hours).",
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note content"
                },
                "category": {
                    "type": "string",
                    "description": "Category for the note",
                    "default": "notes"
        },
        "memory_type": {
            "type": "string",
            "description": "Override note memory type",
            "enum": ["long_term", "short_term", "quick_note"],
            "default": "quick_note"
                }
            },
            "required": ["note"]
        }
    },
    {
        "name": "switch_to_v2_mode",
        "description": "This provides you more voice quality and expression. you can do tons of accents, voice impressions etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Optional reason for switching to V2 mode",
                    "default": "V2 Requested"
                }
            }
        }
    },
    {
        "name": "switch_to_v1_mode",
        "description": "Switch back to V1 mode using Gemini 2.0 Flash Live. This provides the original feature set and session management capabilities.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Optional reason for switching to V1 mode",
                    "default": "Returning to V1 mode requested"
                }
            }
        }
    },
    {
        "name": "trigger_clip_shortcut",
        "description": "Send the Alt+F10 capture shortcut to save the current highlight (works with any software using that binding).",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]

async def handle_utility_function_call(function_call) -> types.FunctionResponse:
    """Handle utility function calls."""
    function_name = function_call.name
    args = function_call.args
    
    try:
        if function_name == "get_current_time":
            result = {
                "success": True,
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": datetime.now().isoformat()
            }
        
        elif function_name == "take_note":
            # Config and rate limit controls
            cfg = _get_note_config()

            if not cfg.get("enabled", True):
                result = {
                    "success": True,
                    "message": "Notes are disabled by configuration; note not saved.",
                    "skipped": True,
                }
            else:
                # Rate limit: prevent too-frequent notes
                now = time.time()
                min_gap = float(cfg.get("min_interval_seconds", 120))

                note_text = str(args["note"]).strip()
                content_hash = hashlib.sha256(note_text.lower().encode("utf-8")).hexdigest()

                global _last_note_ts, _last_note_hash
                if _last_note_ts is None:
                    _last_note_ts = 0.0

                if now - _last_note_ts < min_gap:
                    result = {
                        "success": True,
                        "message": f"Note rate-limited (wait {int(min_gap - (now - _last_note_ts))}s); not saved.",
                        "skipped": True,
                        "reason": "rate_limited"
                    }
                else:
                    # De-duplicate recent identical notes
                    dedupe_window = float(cfg.get("dedupe_window_seconds", 300))
                    if _last_note_hash == content_hash and now - _last_note_ts < dedupe_window:
                        result = {
                            "success": True,
                            "message": "Duplicate recent note suppressed; not saved.",
                            "skipped": True,
                            "reason": "duplicate"
                        }
                    else:
                        # Generate a timestamped key for the note
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        key = f"note_{timestamp}"

                        # Determine memory type: prefer arg, fall back to config, default quick_note
                        memory_type = args.get("memory_type") or str(cfg.get("default_type", "quick_note"))

                        result = memory_system.save_memory(
                            key=key,
                            content=note_text,
                            category=args.get("category", "notes"),
                            memory_type=memory_type,
                            tags=["quick_note"]
                        )

                        # Update rate-limit trackers only if actually saved
                        if result.get("success"):
                            _last_note_ts = now
                            _last_note_hash = content_hash
        
        elif function_name == "switch_to_v2_mode":
            reason = args.get("reason", "Better voice quality requested")
            logger.info(f"V2 mode switch requested: {reason}")
            
            # Set a flag that can be checked by the main application
            # This will be handled by the main application loop
            result = {
                "success": True,
                "message": f"Switching to V2 mode. Reason: {reason}",
                "action": "switch_to_v2",
                "reason": reason
            }
        
        elif function_name == "switch_to_v1_mode":
            reason = args.get("reason", "Returning to V1 mode requested")
            logger.info(f"V1 mode switch requested: {reason}")
            
            # This function will be called from V2 mode to switch back to V1
            result = {
                "success": True,
                "message": f"Switching to V1 mode. Reason: {reason}",
                "action": "switch_to_v1",
                "reason": reason
            }
        elif function_name == "trigger_clip_shortcut":
            injector = _ensure_key_injector()
            try:
                injector.keyDown("alt")
                injector.press("f10")
            finally:
                try:
                    injector.keyUp("alt")
                except Exception:
                    pass
            result = {
                "success": True,
                "message": "Triggered Alt+F10 capture shortcut",
                "action": "trigger_clip_shortcut"
            }
        
        else:
            result = {
                "success": False,
                "message": f"Unknown utility function: {function_name}"
            }
        
        return types.FunctionResponse(
            id=function_call.id,
            name=function_name,
            response=result
        )
        
    except Exception as e:
        logger.error(f"Error handling utility function call {function_name}: {e}")
        return types.FunctionResponse(
            id=function_call.id,
            name=function_name,
            response={
                "success": False,
                "message": f"Error executing {function_name}: {str(e)}"
            }
        )

# --- Internal configuration and state for notes ---
_last_note_ts: float | None = None
_last_note_hash: str | None = None

def _get_note_config() -> Dict[str, Any]:
    """Load note-related configuration from config.yml with safe defaults.

    Returns a dict with keys:
      - enabled: bool
      - default_type: str ("short_term" | "quick_note" | "long_term")
      - min_interval_seconds: float
      - dedupe_window_seconds: float
    """
    defaults = {
        "enabled": True,
    "default_type": "quick_note",
        "min_interval_seconds": 120.0,
        "dedupe_window_seconds": 300.0,
    }
    try:
        config_path = os.path.join(os.getcwd(), "config.yml")
        if yaml and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            mem_cfg = (cfg.get("memory") or {}).get("notes") or {}
            if isinstance(mem_cfg, dict):
                return {
                    **defaults,
                    **{k: mem_cfg.get(k, defaults[k]) for k in defaults.keys()}
                }
    except Exception as e:
        logger.debug(f"Note config load failed; using defaults: {e}")
    return defaults
