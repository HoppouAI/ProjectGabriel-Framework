"""
Integration tools for Gabriel
Provides access to MyInstants, SFX, Personalities, and Movement modules.
"""

import logging

# Set up logging
logger = logging.getLogger(__name__)

# Import MyInstants functionality
try:
    from myinstants import (
        get_all_myinstants_tools, 
        handle_myinstants_function_calls,
        MYINSTANTS_FUNCTION_DECLARATIONS
    )
    MYINSTANTS_AVAILABLE = True
    logger.info("MyInstants module loaded successfully")
except ImportError as e:
    MYINSTANTS_AVAILABLE = False
    MYINSTANTS_FUNCTION_DECLARATIONS = []
    handle_myinstants_function_calls = None
    logger.warning(f"MyInstants module not available: {e}")

# Import SFX functionality
try:
    from sfx import (
        get_all_sfx_tools,
        handle_sfx_function_calls,
        SFX_FUNCTION_DECLARATIONS
    )
    SFX_AVAILABLE = True
    logger.info("SFX module loaded successfully")
except ImportError as e:
    SFX_AVAILABLE = False
    SFX_FUNCTION_DECLARATIONS = []
    handle_sfx_function_calls = None
    logger.warning(f"SFX module not available: {e}")

# Import Personalities functionality
try:
    from personalities import (
        get_personality_tools,
        handle_personality_function_calls,
        PERSONALITY_FUNCTION_DECLARATIONS
    )
    PERSONALITIES_AVAILABLE = True
    logger.info("Personalities module loaded successfully")
except ImportError as e:
    PERSONALITIES_AVAILABLE = False
    PERSONALITY_FUNCTION_DECLARATIONS = []
    handle_personality_function_calls = None
    logger.warning(f"Personalities module not available: {e}")

# Import Movement functionality (OSC input without vision)
try:
    from movement import (
        get_movement_tools,
        handle_movement_function_calls,
        MOVEMENT_FUNCTION_DECLARATIONS,
    )
    MOVEMENT_AVAILABLE = True
    logger.info("Movement module loaded successfully")
except Exception as e:
    MOVEMENT_AVAILABLE = False
    MOVEMENT_FUNCTION_DECLARATIONS = []
    handle_movement_function_calls = None
    logger.warning(f"Movement module not available: {e}")

# Import VRChat functionality (friend requests)
try:
    from .vrchat_tools import (
        VRCHAT_FUNCTION_DECLARATIONS,
        handle_vrchat_function_calls,
    )
    VRCHAT_AVAILABLE = True
    logger.info("VRChat tools loaded successfully")
except Exception as e:
    VRCHAT_AVAILABLE = False
    VRCHAT_FUNCTION_DECLARATIONS = []
    handle_vrchat_function_calls = None
    logger.warning(f"VRChat tools not available: {e}")

# Import Fishing functionality
try:
    from .fishing import (
        FISHING_FUNCTION_DECLARATIONS,
        handle_fishing_function_calls,
    )
    FISHING_AVAILABLE = True
    logger.info("Fishing tools loaded successfully")
except Exception as e:
    FISHING_AVAILABLE = False
    FISHING_FUNCTION_DECLARATIONS = []
    handle_fishing_function_calls = None
    logger.warning(f"Fishing tools not available: {e}")

# Re-export availability flags and handlers for easy access
__all__ = [
    'MYINSTANTS_AVAILABLE',
    'SFX_AVAILABLE', 
    'PERSONALITIES_AVAILABLE',
    'MOVEMENT_AVAILABLE',
    'VRCHAT_AVAILABLE',
    'FISHING_AVAILABLE',
    'MYINSTANTS_FUNCTION_DECLARATIONS',
    'SFX_FUNCTION_DECLARATIONS',
    'PERSONALITY_FUNCTION_DECLARATIONS', 
    'MOVEMENT_FUNCTION_DECLARATIONS',
    'FISHING_FUNCTION_DECLARATIONS',
    'VRCHAT_FUNCTION_DECLARATIONS',
    'handle_myinstants_function_calls',
    'handle_sfx_function_calls',
    'handle_personality_function_calls',
    'handle_movement_function_calls',
    'handle_fishing_function_calls',
    'handle_vrchat_function_calls',
]
