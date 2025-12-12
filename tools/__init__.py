"""
Tools package for function calling with Gemini Live API
Provides modular tool system with memory, utilities, and integrations.
"""

import logging
from typing import List, Dict, Any
from google.genai import types

# Import all tool modules
from .memory import (
    MemorySystem,
    memory_system,
    MEMORY_FUNCTION_DECLARATIONS,
    handle_memory_function_call,
    get_memory_tools
)

from .utilities import (
    UTILITY_FUNCTION_DECLARATIONS,
    handle_utility_function_call
)

from .vision_tools import (
    VISION_FUNCTION_DECLARATIONS,
    handle_vision_function_calls
)

from .integrations import (
    MYINSTANTS_AVAILABLE,
    SFX_AVAILABLE,
    PERSONALITIES_AVAILABLE,
    MOVEMENT_AVAILABLE,
    VRCHAT_AVAILABLE,
    FISHING_AVAILABLE,
    MYINSTANTS_FUNCTION_DECLARATIONS,
    SFX_FUNCTION_DECLARATIONS,
    PERSONALITY_FUNCTION_DECLARATIONS,
    MOVEMENT_FUNCTION_DECLARATIONS,
    FISHING_FUNCTION_DECLARATIONS,
    handle_myinstants_function_calls,
    handle_sfx_function_calls,
    handle_personality_function_calls,
    handle_movement_function_calls,
    handle_fishing_function_calls,
    VRCHAT_FUNCTION_DECLARATIONS,
    handle_vrchat_function_calls,
)

# Yap mode tools
from .yap_tools import (
    YAP_FUNCTION_DECLARATIONS,
    handle_yap_function_calls,
    is_yap_mode_enabled,
    set_yap_mode,
    is_ai_speaking,
    set_ai_speaking,
)
from .audio_tools import (
    AUDIO_FUNCTION_DECLARATIONS,
    handle_audio_function_calls,
)
from .image_generation import (
    IMAGE_FUNCTION_DECLARATIONS,
    handle_image_generation_function_calls,
)

# Set up logging
logger = logging.getLogger(__name__)

def get_all_tools(config=None):
    """Get all available tools for Gemini Live API."""
    # Start with memory declarations
    all_declarations = MEMORY_FUNCTION_DECLARATIONS[:]
    
    # Filter utility declarations based on V2 mode config
    v2_enabled = True
    if config:
        v2_enabled = config.get('v2_mode', {}).get('enabled', True)
    
    # Add utility functions, filtering switch_to_v2_mode if V2 is disabled
    for decl in UTILITY_FUNCTION_DECLARATIONS:
        if decl.get('name') == 'switch_to_v2_mode' and not v2_enabled:
            continue  # Skip V2 switch function if V2 mode is disabled
        all_declarations.append(decl)
    
    # Add MyInstants tools if available
    if MYINSTANTS_AVAILABLE:
        all_declarations.extend(MYINSTANTS_FUNCTION_DECLARATIONS)
    
    # Add SFX tools if available
    if SFX_AVAILABLE:
        all_declarations.extend(SFX_FUNCTION_DECLARATIONS)
    
    # Add Personalities tools if available
    if PERSONALITIES_AVAILABLE:
        all_declarations.extend(PERSONALITY_FUNCTION_DECLARATIONS)
    
    # Add Vision tools
    all_declarations.extend(VISION_FUNCTION_DECLARATIONS)
    
    # Add Movement tools if available
    if MOVEMENT_AVAILABLE:
        all_declarations.extend(MOVEMENT_FUNCTION_DECLARATIONS)

    # Add Fishing tools if available
    if FISHING_AVAILABLE:
        all_declarations.extend(FISHING_FUNCTION_DECLARATIONS)

    # Add Yap mode tools
    all_declarations.extend(YAP_FUNCTION_DECLARATIONS)

    # Add VRChat tools if available
    if VRCHAT_AVAILABLE:
        all_declarations.extend(VRCHAT_FUNCTION_DECLARATIONS)
    all_declarations.extend(AUDIO_FUNCTION_DECLARATIONS)
    all_declarations.extend(IMAGE_FUNCTION_DECLARATIONS)
    
    return [{"function_declarations": all_declarations}]

async def handle_function_call(function_call) -> types.FunctionResponse:
    """Main function call handler that routes to appropriate handlers."""
    function_name = function_call.name
    
    # Memory functions
    if function_name in [
        "save_memory", "read_memory", "update_memory", "delete_memory",
        "list_memories", "search_memories", "get_memory_stats", "cleanup_expired_memories"
    ]:
        return await handle_memory_function_call(function_call)
    
    # Utility functions
    elif function_name in ["get_current_time", "take_note", "switch_to_v2_mode", "switch_to_v1_mode", "trigger_clip_shortcut"]:
        return await handle_utility_function_call(function_call)
    
    # MyInstants functions
    elif MYINSTANTS_AVAILABLE and function_name.startswith("myinstants_") or function_name in [
        "search_myinstants_sounds", "play_myinstants_sound", "get_myinstants_sound_details",
        "get_trending_myinstants_sounds", "get_recent_myinstants_sounds", "stop_myinstants_sound",
        "set_myinstants_volume", "get_myinstants_cache_info", "clear_myinstants_cache"
    ]:
        return await handle_myinstants_function_calls(function_call)
    
    # SFX functions
    elif SFX_AVAILABLE and function_name in [
        "play_sfx", "stop_sfx", "list_sfx", "search_sfx", "get_sfx_categories",
        "set_sfx_volume", "get_sfx_status", "scan_sfx_files"
    ]:
        return await handle_sfx_function_calls(function_call)
    
    # Personality functions
    elif PERSONALITIES_AVAILABLE and function_name in [
        "switch_personality", "get_current_personality", "list_personalities",
        "add_personality", "update_personality", "get_personality_history"
    ]:
        return await handle_personality_function_calls(function_call)
    
    # Vision functions
    elif function_name in [
    "vision_start_following", "vision_stop_following", "vision_status"
    ]:
        return await handle_vision_function_calls(function_call)
    
    # Movement functions
    elif MOVEMENT_AVAILABLE and function_name in [
    "look_behind", "look_turn", "move_direction", "jump", "crouch", "crawl", "stop_all_inputs"
    ]:
        return await handle_movement_function_calls(function_call)
    
    # Fishing functions
    elif FISHING_AVAILABLE and function_name in [
        "vr_fishing_cast", "vr_fishing_reel", "vr_set_fishing_mode"
    ]:
        return await handle_fishing_function_calls(function_call)
    
    # Yap mode functions
    elif function_name in [
        "enable_yap_mode", "disable_yap_mode", "get_yap_mode_status"
    ]:
        return await handle_yap_function_calls(function_call)

    # VRChat functions
    elif VRCHAT_AVAILABLE and function_name in [
        "list_vrchat_friend_requests", "accept_vrchat_friend_request", "deny_vrchat_friend_request",
    "get_own_avatar", "select_avatar", "list_saved_avatars"
    ]:
        return await handle_vrchat_function_calls(function_call)
    elif function_name in [
        "stop_all_audio_playback"
    ]:
        return await handle_audio_function_calls(function_call)
    elif function_name in [
        "generate_image_to_webhook"
    ]:
        return await handle_image_generation_function_calls(function_call)
    
    else:
        logger.warning(f"Unknown function call: {function_name}")
        return types.FunctionResponse(
            id=function_call.id,
            name=function_name,
            response={
                "success": False,
                "message": f"Unknown function: {function_name}"
            }
        )

# Export commonly used items
__all__ = [
    'get_all_tools',
    'handle_function_call',
    'memory_system',
    'MemorySystem',
    'get_memory_tools',
    'is_yap_mode_enabled',
    'set_yap_mode',
    'is_ai_speaking',
    'set_ai_speaking',
]
