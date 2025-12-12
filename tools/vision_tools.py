"""
Vision tools for Gabriel
Provides vision-based player following functionality.
"""

import logging
from typing import Dict, Any
from google.genai import types

# Set up logging
logger = logging.getLogger(__name__)

# Import Vision functionality
try:
    from vision import vision as vision_module
    VISION_AVAILABLE = True
    logger.info("Vision module loaded successfully")
except Exception as e:
    VISION_AVAILABLE = False
    vision_module = None
    logger.warning(f"Vision module not available: {e}")

# Vision function declarations
VISION_FUNCTION_DECLARATIONS = [
    {
        "name": "vision_start_following",
        "description": "Begin following nearby players using the vision system and send OSC movement controls.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "vision_stop_following",
        "description": "Stop following players and halt OSC movement.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "vision_status",
        "description": "Get current vision system status including device, model readiness, and follow state.",
        "parameters": {"type": "object", "properties": {}}
    }
]

async def handle_vision_function_calls(function_call) -> types.FunctionResponse:
    function_name = function_call.name
    try:
        if not VISION_AVAILABLE:
            result = {"success": False, "message": "Vision module not available"}
        else:
            if function_name == "vision_start_following":
                ok = vision_module.start_following()
                result = {"success": bool(ok), "message": "Vision following started" if ok else "Already running"}
            elif function_name == "vision_stop_following":
                ok = vision_module.stop_following()
                result = {"success": bool(ok), "message": "Vision following stopped"}
            elif function_name == "vision_status":
                status = vision_module.get_status() if hasattr(vision_module, 'get_status') else {}
                result = {"success": True, "status": status}
            else:
                result = {"success": False, "message": f"Unknown vision function: {function_name}"}

        return types.FunctionResponse(
            id=function_call.id,
            name=function_name,
            response=result
        )
    except Exception as e:
        logger.error(f"Error handling vision function call {function_name}: {e}")
        return types.FunctionResponse(
            id=function_call.id,
            name=function_name,
            response={
                "success": False,
                "message": f"Error executing {function_name}: {str(e)}"
            }
        )
