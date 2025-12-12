"""
Yap mode tools for Gabriel
Allows the AI to temporarily disable audio input so it cannot be cut off
by user speech until it explicitly disables yap mode again.
"""

import logging
import asyncio
from typing import Dict, Any, Optional
from google.genai import types

logger = logging.getLogger(__name__)

# Module-level states
_YAP_MODE_ENABLED: bool = False
_AI_SPEAKING: bool = False
_YAP_TIMER_TASK: Optional[asyncio.Task] = None


async def _yap_mode_auto_disable() -> None:
    """Auto-disable yap mode after 60 seconds."""
    try:
        await asyncio.sleep(60)
        global _YAP_MODE_ENABLED
        if _YAP_MODE_ENABLED:
            logger.info("Yap mode auto-disabled after 60 seconds")
            _YAP_MODE_ENABLED = False
    except asyncio.CancelledError:
        logger.debug("Yap mode timer cancelled")
    except Exception as e:
        logger.error(f"Error in yap mode auto-disable: {e}")


def is_yap_mode_enabled() -> bool:
    """Return whether yap mode is currently enabled."""
    return _YAP_MODE_ENABLED


def set_yap_mode(enabled: bool) -> None:
    """Set yap mode on or off."""
    global _YAP_MODE_ENABLED
    global _YAP_TIMER_TASK
    prev = _YAP_MODE_ENABLED
    _YAP_MODE_ENABLED = bool(enabled)
    
    if _YAP_TIMER_TASK and not _YAP_TIMER_TASK.done():
        _YAP_TIMER_TASK.cancel()
        _YAP_TIMER_TASK = None
    
    if _YAP_MODE_ENABLED:
        _YAP_TIMER_TASK = asyncio.create_task(_yap_mode_auto_disable())
        logger.info(f"Yap mode ENABLED (was {'ENABLED' if prev else 'DISABLED'}); will auto-disable in 60 seconds")
    else:
        logger.info(f"Yap mode DISABLED (was {'ENABLED' if prev else 'DISABLED'})")


def is_ai_speaking() -> bool:
    """Return whether the AI is currently speaking (output audio playing)."""
    return _AI_SPEAKING


def set_ai_speaking(speaking: bool) -> None:
    """Set whether the AI is currently speaking.

    This should be toggled by the audio receive/playback pipeline when
    output audio starts and ends.
    """
    global _AI_SPEAKING
    prev = _AI_SPEAKING
    _AI_SPEAKING = bool(speaking)
    if prev != _AI_SPEAKING:
        logger.info(f"AI speaking state: {'STARTED' if _AI_SPEAKING else 'ENDED'}")


# Function declarations exposed to the model
YAP_FUNCTION_DECLARATIONS = [
    {
        "name": "enable_yap_mode",
        "description": (
            "Disable microphone input so the user cannot interrupt you. "
            "Use this when you need to speak without being cut off. You must later call disable_yap_mode to re-enable input."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Optional explanation shown in logs/UI for enabling yap mode.",
                    "default": "AI requires uninterrupted speaking"
                }
            }
        }
    },
    {
        "name": "disable_yap_mode",
        "description": "Re-enable microphone input so the user can speak to you again.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Optional explanation shown in logs/UI for disabling yap mode.",
                    "default": "AI finished speaking uninterrupted"
                }
            }
        }
    },
    {
        "name": "get_yap_mode_status",
        "description": "Get whether yap mode is currently enabled (audio input disabled).",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]


async def handle_yap_function_calls(function_call) -> types.FunctionResponse:
    """Handle yap mode related function calls."""
    fname = function_call.name
    args: Dict[str, Any] = function_call.args or {}
    try:
        if fname == "enable_yap_mode":
            reason = args.get("reason", "AI requires uninterrupted speaking")
            set_yap_mode(True)
            response = {
                "success": True,
                "message": f"Yap mode enabled. {reason}",
                "yap_mode_enabled": True
            }
        elif fname == "disable_yap_mode":
            reason = args.get("reason", "AI finished speaking uninterrupted")
            set_yap_mode(False)
            response = {
                "success": True,
                "message": f"Yap mode disabled. {reason}",
                "yap_mode_enabled": False
            }
        elif fname == "get_yap_mode_status":
            response = {
                "success": True,
                "yap_mode_enabled": is_yap_mode_enabled()
            }
        else:
            response = {
                "success": False,
                "message": f"Unknown yap function: {fname}"
            }

        return types.FunctionResponse(
            id=function_call.id,
            name=fname,
            response=response,
        )
    except Exception as e:
        logger.error(f"Error handling yap function {fname}: {e}")
        return types.FunctionResponse(
            id=function_call.id,
            name=fname,
            response={
                "success": False,
                "message": f"Error executing {fname}: {str(e)}"
            }
        )
