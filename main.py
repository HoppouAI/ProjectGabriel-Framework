import os
import asyncio
import base64
import io
import traceback
import yaml
import json
import logging
import time
import random

import cv2
import pyaudio
import PIL.Image
import mss
try:
    from mss import tools as mss_tools
except Exception:
    mss_tools = None

import argparse

from google import genai
from google.genai import types

try:
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    ConnectionClosedError = Exception
    ConnectionClosedOK = Exception

import tools as memory_tools

# Import OSC module for VRChat integration
import osc

# Import append system for system prompt enhancement
import append

# Import MyInstants for audio timing notifications
try:
    from myinstants import myinstants_client
    MYINSTANTS_AVAILABLE = True
except ImportError:
    MYINSTANTS_AVAILABLE = False
    myinstants_client = None

# Import Chat API
try:
    from api import chat as chat_api
    CHAT_API_AVAILABLE = True
except ImportError:
    CHAT_API_AVAILABLE = False
    chat_api = None

# Import WebUI Server
try:
    from api import webui_server
    WEBUI_SERVER_AVAILABLE = True
except ImportError:
    WEBUI_SERVER_AVAILABLE = False
    webui_server = None

# Import V2 mode
try:
    import v2
    V2_AVAILABLE = True
except ImportError:
    V2_AVAILABLE = False
    print("V2 mode not available")

# Attempt to preload vision model in background
try:
    from vision import vision as _vision_module
    _VISION_AVAILABLE = True
except Exception:
    _vision_module = None
    _VISION_AVAILABLE = False

# Idle gaze support
try:
    import idle as _idle
    _IDLE_AVAILABLE = True
except Exception:
    _idle = None
    _IDLE_AVAILABLE = False

# Session persistence
try:
    from session_persistence import get_persistence_manager
    SESSION_PERSISTENCE_AVAILABLE = True
except Exception:
    get_persistence_manager = None
    SESSION_PERSISTENCE_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Custom exception for V2 mode switching
class V2ModeSwitchRequested(Exception):
    """Exception raised when V2 mode switch is requested."""
    pass


class ControlledReconnectRequested(Exception):
    """Exception raised to proactively restart the Live session at a safe boundary."""
    pass


def setup_logging(config):
    """Setup logging configuration based on config file."""
    logging_config = config.get('logging', {})
    
    # Set main logging level
    log_level = logging_config.get('level', 'INFO').upper()
    level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(level=level)
    
    # Configure specific loggers
    loggers_config = logging_config.get('loggers', {})
    for logger_name, logger_level in loggers_config.items():
        specific_logger = logging.getLogger(logger_name)
        specific_level = getattr(logging, logger_level.upper(), logging.ERROR)
        specific_logger.setLevel(specific_level)
    
    # Suppress Google GenAI warnings if enabled
    if logging_config.get('suppress_genai_warnings', True):
        logging.getLogger('google_genai.types').setLevel(logging.ERROR)


def load_config(config_path="config.yml"):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default values.")
        return {}
    except yaml.YAMLError as e:
        print(f"Error parsing config file: {e}")
        return {}


def load_prompts(prompts_path="prompts.json"):
    """Load prompts from JSON file."""
    try:
        with open(prompts_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Prompts file {prompts_path} not found. Using default prompt.")
        return {
            "normal": {
                "name": "Default Assistant",
                "description": "Gabriel, a helpful assistant",
                "prompt": "You are Gabriel, a helpful assistant."
            }
        }
    except json.JSONDecodeError as e:
        print(f"Error parsing prompts file: {e}")
        return {
            "normal": {
                "name": "Default Assistant", 
                "description": "Gabriel, a helpful assistant",
                "prompt": "You are Gabriel, a helpful assistant."
            }
        }


def get_system_instruction(config, prompts):
    """Get the system instruction based on config selection."""
    live_config = config.get('live_connect', {})
    prompt_key = live_config.get('prompt', 'normal')
    
    # Handle custom prompt
    if prompt_key == 'custom':
        custom_prompt = live_config.get('custom_prompt')
        if custom_prompt:
            base_instruction = custom_prompt
        else:
            logger.warning("Custom prompt selected but custom_prompt not found in config. Using default.")
            prompt_key = 'normal'
            base_instruction = prompts.get('normal', {}).get('prompt', 'You are Gabriel, a helpful assistant.')
    else:
        # Get prompt from prompts.json
        if prompt_key in prompts:
            prompt_info = prompts[prompt_key]
            logger.info(f"Using prompt: {prompt_info.get('name', prompt_key)} - {prompt_info.get('description', 'No description')}")
            base_instruction = prompt_info['prompt']
        else:
            logger.warning(f"Prompt '{prompt_key}' not found in prompts.json. Available prompts: {', '.join(prompts.keys())}. Using 'normal'.")
            base_instruction = prompts.get('normal', {}).get('prompt', 'You are Gabriel, a helpful assistant.')
    
    # Apply append system to enhance the base instruction
    append_config = config.get('append_system', {})
    appends_path = append_config.get('file', 'appends.json')
    append_variables = append_config.get('variables', {})
    
    # Add config-based variables
    if 'add_config_info' in append_config and append_config['add_config_info']:
        append_variables.update({
            'model_name': config.get('model', {}).get('name', 'Unknown'),
            'audio_sample_rate': str(config.get('audio', {}).get('send_sample_rate', 16000)),
            'video_mode': config.get('defaults', {}).get('mode', 'camera')
        })
    # Ensure we append personality name and description only (not full content)
    append_variables.setdefault('personalities_include_description', True)
    
    enhanced_instruction = append.append_to_system_instruction(
        base_instruction, 
        appends_path, 
        append_variables,
        config  # Pass full config for memory integration
    )
    
    return enhanced_instruction


def setup_globals(config, prompts=None):
    """Setup global variables from configuration."""
    global FORMAT, CHANNELS, SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE, CHUNK_SIZE
    global MODEL, DEFAULT_MODE, client, tools, CONFIG
    
    # Load prompts if not provided
    if prompts is None:
        prompts = load_prompts()
    
    # Audio configuration
    FORMAT = getattr(pyaudio, f"paInt{config.get('audio', {}).get('format', 16)}")
    CHANNELS = config.get('audio', {}).get('channels', 1)
    SEND_SAMPLE_RATE = config.get('audio', {}).get('send_sample_rate', 16000)
    RECEIVE_SAMPLE_RATE = config.get('audio', {}).get('receive_sample_rate', 24000)
    CHUNK_SIZE = config.get('audio', {}).get('chunk_size', 1024)

    # Model configuration
    MODEL = config.get('model', {}).get('name', "models/gemini-2.0-flash-live-001")

    # Default mode
    DEFAULT_MODE = config.get('defaults', {}).get('mode', "camera")

    # API configuration
    api_config = config.get('api', {})
    
    # Handle API key from config or environment variable
    api_key = None
    if 'api_key' in api_config:
        # Use API key directly from config
        api_key = api_config['api_key']
        logger.info("Using API key from config file")
    elif 'key_env_var' in api_config:
        # Use environment variable specified in config
        api_key = os.environ.get(api_config['key_env_var'])
        if api_key:
            logger.info(f"Using API key from environment variable: {api_config['key_env_var']}")
        else:
            logger.warning(f"Environment variable {api_config['key_env_var']} not found")
    else:
        # Fallback to default environment variable
        api_key = os.environ.get('GEMINI_API_KEY')
        if api_key:
            logger.info("Using API key from default environment variable: GEMINI_API_KEY")
        else:
            logger.warning("Default environment variable GEMINI_API_KEY not found")
    
    # Validate API key
    if not api_key:
        raise ValueError(
            "No API key found! Please either:\n"
            "1. Add 'api_key' directly to your config.yml file, or\n"
            "2. Set the GEMINI_API_KEY environment variable, or\n"
            "3. Add 'key_env_var' to your config.yml to specify a custom environment variable name"
        )
    
    if not api_key.strip():
        raise ValueError("API key is empty or contains only whitespace")
    
    logger.info(f"API key loaded successfully (length: {len(api_key)} characters)")
    
    client = genai.Client(
        http_options={"api_version": api_config.get('version', 'v1beta')},
        api_key=api_key,
    )

    # Tools configuration
    tools_config = config.get('tools', {})
    tools = []

    # Add Google Search tool if enabled
    if tools_config.get('google_search', True):
        tools.append(types.Tool(google_search=types.GoogleSearch()))

    # Add memory system tools if enabled (default: True)
    if tools_config.get('memory_system', True):
        memory_system_tools = memory_tools.get_all_tools(config)
        for tool in memory_system_tools:
            tools.append(types.Tool(function_declarations=tool['function_declarations']))

    # Add custom function declarations from config
    function_declarations = tools_config.get('function_declarations', [])
    if function_declarations:
        tools.append(types.Tool(function_declarations=function_declarations))

    # Live connect configuration
    live_config = config.get('live_connect', {})
    speech_config = live_config.get('speech', {})
    voice_config = speech_config.get('voice', {})
    context_window_config = live_config.get('context_window', {})
    session_resumption_config = live_config.get('session_resumption', {})

    # Setup session resumption configuration
    session_resumption = None
    if session_resumption_config.get('enabled', False):
        session_resumption = types.SessionResumptionConfig(
            # The handle parameter is set dynamically by SessionManager
            # when resuming a session. For new sessions, this will be None.
            handle=None
        )

    # Check if using native audio model (doesn't support language_code)
    is_native_audio_model = MODEL == "models/gemini-2.5-flash-native-audio-preview-09-2025"
    
    # Build SpeechConfig based on model type
    if is_native_audio_model:
        # Native audio model doesn't support language_code parameter
        speech_cfg = types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_config.get('name', 'Puck')
                )
            )
        )
    else:
        # Standard models support language_code
        speech_cfg = types.SpeechConfig(
            language_code=speech_config.get('language_code', 'ta-IN'),
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_config.get('name', 'Puck')
                )
            )
        )
    
    # Configure context window based on model capabilities
    if is_native_audio_model:
        # Native audio model supports up to 128k context window
        context_compression = types.ContextWindowCompressionConfig(
            trigger_tokens=context_window_config.get('trigger_tokens_native', 102400),
            sliding_window=types.SlidingWindow(
                target_tokens=context_window_config.get('sliding_window_target_tokens_native', 51200)
            ),
        )
    else:
        # Standard models use default context window settings
        context_compression = types.ContextWindowCompressionConfig(
            trigger_tokens=context_window_config.get('trigger_tokens', 25600),
            sliding_window=types.SlidingWindow(
                target_tokens=context_window_config.get('sliding_window_target_tokens', 12800)
            ),
        )
    
    CONFIG = types.LiveConnectConfig(
        response_modalities=live_config.get('response_modalities', ["AUDIO"]),
        media_resolution=live_config.get('media_resolution', "MEDIA_RESOLUTION_MEDIUM"),
        speech_config=speech_cfg,
        context_window_compression=context_compression,
        session_resumption=session_resumption,
        tools=tools,
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=get_system_instruction(config, prompts))],
            role="user"
        ),
        # Enable audio transcription for VRChat text output when using AUDIO mode
        output_audio_transcription=live_config.get('output_audio_transcription', {}),
        # Enable input audio transcription to capture user speech
        input_audio_transcription=live_config.get('input_audio_transcription', {}),
    )

    # Initialize VRChat OSC client
    osc.initialize_osc_client(config)
    
    # Initialize VRChat controls for webui
    try:
        from api.webui import initialize_vrchat_controls
        initialize_vrchat_controls(config)
        logger.info("VRChat controls initialized")
    except Exception as e:
        logger.warning(f"Unable to initialize VRChat controls: {e}")
    
    # Initialize movement module with config so it can send OSC inputs
    try:
        import movement
        movement.initialize_movement(config)
        logger.info("Movement module initialized")
    except Exception as e:
        logger.warning(f"Unable to initialize movement module: {e}")
    
    # Preload vision model in background to avoid startup pauses
    try:
        if _VISION_AVAILABLE and hasattr(_vision_module, 'initialize_in_background'):
            _vision_module.initialize_in_background()
            logger.info("Vision model preload triggered in background")
            # Only auto-start following if explicitly configured for auto-start
            try:
                vcfg = getattr(_vision_module, 'config', {})
                if vcfg and vcfg.get('auto_start_following', False) and hasattr(_vision_module, 'start_following'):
                    _vision_module.start_following()
                    logger.info("Vision follower started automatically")
                else:
                    logger.info("Vision follower available but not auto-started (use voice commands to control)")
            except Exception as vision_start_err:
                logger.warning(f"Unable to start vision follower: {vision_start_err}")
    except Exception as e:
        logger.warning(f"Unable to preload vision model: {e}")

    # Start idle gaze in background
    try:
        if _IDLE_AVAILABLE and hasattr(_idle, 'start_idle_gaze'):
            _idle.start_idle_gaze()
            logger.info("Idle gaze started")
    except Exception as e:
        logger.warning(f"Unable to start idle gaze: {e}")

    # Start Chat API server
    try:
        if CHAT_API_AVAILABLE and chat_api:
            chat_api.start_chat_api(config)
            logger.info("Chat API initialization attempted")
        else:
            logger.info("Chat API not available")
    except Exception as e:
        logger.warning(f"Unable to start Chat API: {e}")
    
    # Start WebUI server
    try:
        if WEBUI_SERVER_AVAILABLE and webui_server:
            webui_server.start_webui_server(config)
            logger.info("WebUI server initialization attempted")
        else:
            logger.info("WebUI server not available")
    except Exception as e:
        logger.warning(f"Unable to start WebUI server: {e}")

pya = pyaudio.PyAudio()

# Global variables that will be set by setup_globals
FORMAT = None
CHANNELS = None
SEND_SAMPLE_RATE = None
RECEIVE_SAMPLE_RATE = None
CHUNK_SIZE = None
MODEL = None
DEFAULT_MODE = "camera"  # Default fallback value
client = None
tools = None
CONFIG = None


class SessionManager:
    """Manages session resumption and automatic reconnection."""
    
    def __init__(self, config):
        self.config = config
        self.session_config = config.get('session_management', {})
        self.auto_reconnect_config = self.session_config.get('auto_reconnect', {})
        self.monitoring_config = self.session_config.get('monitoring', {})
        
        # Session state
        self.session_handle = None
        self.last_consumed_message_index = None
        self.is_reconnecting = False
        self.connection_start_time = None
        self.last_heartbeat = None
        
        self.fresh_start_requested = False
        self.reconnect_requested = False
        
        # Reconnection settings
        self.max_retries = self.auto_reconnect_config.get('max_retries', 5)
        self.initial_delay = self.auto_reconnect_config.get('initial_delay', 1.0)
        self.max_delay = self.auto_reconnect_config.get('max_delay', 30.0)
        self.exponential_base = self.auto_reconnect_config.get('exponential_base', 2.0)
        self.jitter = self.auto_reconnect_config.get('jitter', 0.1)
        
        self.persistence = None
        self.persistence_task = None
        if SESSION_PERSISTENCE_AVAILABLE:
            save_interval = self.session_config.get('save_interval', 30)
            self.persistence = get_persistence_manager(save_interval)
            logger.info(f"Session persistence enabled with {save_interval}s interval")
        
        logger.info("SessionManager initialized")
    
    def get_session_config(self):
        """Get the current session configuration with resumption handle if available."""
        global CONFIG
        
        if self.session_handle and hasattr(CONFIG, 'session_resumption') and CONFIG.session_resumption:
            # Update the session resumption config with our handle
            CONFIG.session_resumption.handle = self.session_handle
            logger.info(f"Using session resumption handle: {self.session_handle[:20]}...")
        
        return CONFIG
    
    def handle_session_resumption_update(self, update):
        """Handle incoming session resumption update."""
        if hasattr(update, 'new_handle') and update.new_handle:
            self.session_handle = update.new_handle
            logger.info(f"Updated session handle: {self.session_handle[:20]}...")
        
        if hasattr(update, 'resumable'):
            logger.info(f"Session resumable: {update.resumable}")
        
        if hasattr(update, 'last_consumed_client_message_index'):
            self.last_consumed_message_index = update.last_consumed_client_message_index
            logger.info(f"Last consumed message index: {self.last_consumed_message_index}")
    
    def handle_go_away(self, go_away):
        """Handle GoAway message indicating impending disconnection."""
        logger.warning("GoAway message received from server")
        
        if hasattr(go_away, 'time_left'):
            time_left = go_away.time_left
            logger.warning(f"Server will disconnect in: {time_left}")
            
            # Parse the duration string (format: "XXXs" for seconds)
            if isinstance(time_left, str) and time_left.endswith('s'):
                try:
                    seconds = float(time_left[:-1])
                    if seconds < 30:  # If less than 30 seconds, prepare for reconnection
                        logger.info("Preparing for imminent disconnection...")
                        self.is_reconnecting = True
                except ValueError:
                    logger.warning(f"Could not parse time_left: {time_left}")
        
        # Always set reconnecting flag when GoAway is received
        self.is_reconnecting = True
        logger.info("GoAway received - marking session for reconnection")
        
        # Log additional GoAway details if available
        if hasattr(go_away, 'reason'):
            logger.warning(f"GoAway reason: {go_away.reason}")
        if hasattr(go_away, 'debug_description'):
            logger.warning(f"GoAway debug info: {go_away.debug_description}")
    
    async def calculate_reconnect_delay(self, retry_count):
        """Calculate delay before reconnection attempt with exponential backoff and jitter."""
        if not self.auto_reconnect_config.get('enabled', True):
            return None
        
        if retry_count >= self.max_retries:
            logger.error(f"Max retries ({self.max_retries}) exceeded. Giving up.")
            return None
        
        # Exponential backoff
        delay = min(self.initial_delay * (self.exponential_base ** retry_count), self.max_delay)
        
        # Add jitter
        jitter_amount = delay * self.jitter * (random.random() * 2 - 1)  # +/- jitter%
        delay += jitter_amount
        
        delay = max(0, delay)  # Ensure non-negative
        logger.info(f"Reconnection attempt {retry_count + 1}/{self.max_retries} in {delay:.2f}s")
        
        return delay
    
    def reset_connection_state(self):
        """Reset connection state for new session."""
        self.connection_start_time = time.time()
        self.last_heartbeat = self.connection_start_time
        self.is_reconnecting = False
    
    def request_fresh_start(self):
        """Request a fresh start (clear session and restart)."""
        logger.info("Fresh start requested via API")
        self.clear_session_resumption(reason="Fresh start requested by user")
        if self.persistence:
            self.persistence.clear_session_handle()
        self.fresh_start_requested = True
    
    def request_reconnect(self):
        """Request a reconnect with saved session."""
        logger.info("Reconnect requested via API")
        if self.persistence:
            saved_data = self.persistence.load_session_handle()
            if saved_data and saved_data.get('handle'):
                self.session_handle = saved_data.get('handle')
                logger.info(f"Loaded session handle for reconnect: {self.session_handle[:20]}...")
            else:
                logger.warning("No saved session handle found for reconnect")
        self.reconnect_requested = True
    
    def start_periodic_save(self, mode: str = "v1"):
        if self.persistence and not self.persistence_task:
            handle_getter = lambda: self.session_handle
            metadata_getter = lambda: {
                "last_consumed_index": self.last_consumed_message_index,
                "connection_start_time": self.connection_start_time
            }
            self.persistence_task = asyncio.create_task(
                self.persistence.start_periodic_save(handle_getter, mode, metadata_getter)
            )
            logger.info(f"Started periodic session save for {mode} mode")
    
    def stop_periodic_save(self, mode: str = "v1"):
        if self.persistence:
            self.persistence.stop_periodic_save()
            if self.session_handle:
                metadata = {
                    "last_consumed_index": self.last_consumed_message_index,
                    "connection_start_time": self.connection_start_time
                }
                self.persistence.save_session_handle(self.session_handle, mode, metadata)
                logger.info(f"Saved final session handle on stop for {mode} mode")
        if self.persistence_task:
            self.persistence_task.cancel()
            self.persistence_task = None
        logger.info("Connection state reset")
    
    def should_attempt_reconnect(self, error):
        """Determine if we should attempt to reconnect based on the error."""
        if not self.auto_reconnect_config.get('enabled', True):
            return False
        
        # Handle policy violation 1008: session not found -> clear session handle and reconnect
        if self._is_policy_session_not_found(error):
            self.clear_session_resumption(reason="Server reported session not found (1008 policy violation)")
            logger.info("Treating 1008 session-not-found as reconnectable after clearing session handle")
            return True

        # If we're already marked for reconnection (e.g., due to GoAway), attempt it
        if self.is_reconnecting:
            logger.info("Reconnection flagged due to GoAway message")
            return True
        
        # Handle ExceptionGroup (from TaskGroup) by checking its exceptions
        if hasattr(error, 'exceptions'):  # ExceptionGroup
            for exc in error.exceptions:
                # Handle policy violation 1008 inside ExceptionGroup
                if self._is_policy_session_not_found(exc):
                    self.clear_session_resumption(reason="Server reported session not found (1008 policy violation)")
                    logger.info("Treating 1008 session-not-found (inner) as reconnectable after clearing session handle")
                    return True
                # Check for V2ModeSwitchRequested in the exception group
                if isinstance(exc, V2ModeSwitchRequested):
                    logger.info("V2ModeSwitchRequested found in ExceptionGroup - not a reconnection case")
                    return False
                if self._is_reconnectable_error(exc):
                    # Log specific details about WebSocket errors
                    if '1011' in str(exc) or 'deadline expired' in str(exc).lower():
                        logger.warning(f"Detected WebSocket deadline/timeout error: {exc}")
                    return True
            return False
        
        # Check for standalone V2ModeSwitchRequested
        if isinstance(error, V2ModeSwitchRequested):
            logger.info("V2ModeSwitchRequested - not a reconnection case")
            return False
        
        return self._is_reconnectable_error(error)
    
    def _is_reconnectable_error(self, error):
        """Check if a single error is reconnectable."""
        # Ignore local screen capture errors (not network related)
        try:
            ename = error.__class__.__name__.lower()
            estr = str(error).lower()
        except Exception:
            ename = ""
            estr = ""
        if ("screenshoterror" in ename) or ("gdi32.getdibits" in estr) or ("mss" in ename and "grab" in estr):
            logger.info(f"Local screen capture error detected (non-network): {type(error).__name__}: {error}")
            return False

        # Always attempt reconnect for connection-related errors
        if isinstance(error, (ConnectionError, asyncio.TimeoutError)):
            return True
        
        # Also attempt reconnect for cancelled errors (could be from GoAway)
        if isinstance(error, asyncio.CancelledError):
            return True
        
        # Handle WebSocket-specific exceptions if available
        if WEBSOCKETS_AVAILABLE:
            if isinstance(error, (ConnectionClosedError, ConnectionClosedOK)):
                logger.info(f"WebSocket connection closed: {error}")
                return True
        
        # Handle WebSocket-specific errors that might occur
        if hasattr(error, '__class__'):
            error_name = error.__class__.__name__.lower()
            # Add specific WebSocket error types
            if any(keyword in error_name for keyword in [
                'websocket', 'connection', 'network', 'timeout', 'aborted',
                'connectionclosederror', 'connectionclosed', 'websocketclosed'
            ]):
                return True
        
    # Handle general exceptions that might indicate disconnection
        # These are common patterns for network disconnections
        if isinstance(error, Exception):
            error_str = str(error).lower()
            if any(keyword in error_str for keyword in [
                'connection', 'disconnect', 'timeout', 'aborted', 
                'websocket', 'network', 'broken pipe', 'reset by peer',
                'deadline expired', 'connectionclosederror', 'internal error',
                'websocket closed', 'connection closed', '1011 (internal error)',
                '1011', '1007', 'invalid argument', 'invalid frame payload',
                'service is currently unavailable', 'server error', 'cancelled'
            ]):
                logger.info(f"Reconnectable error detected: {error}")
                return True
        
        logger.warning(f"Error not configured for reconnection: {type(error).__name__}: {error}")
        return False

    def _is_policy_session_not_found(self, error) -> bool:
        """Detect WebSocket 1008 policy violation with 'session not found' message."""
        try:
            msg = str(error)
            low = msg.lower()
        except Exception:
            return False
        # Look for 1008 and session not found indicators
        if ("1008" in low or "policy violation" in low) and ("session not found" in low or "bidigeneratecontent" in low):
            logger.warning(f"Detected 1008 policy violation / session not found: {msg}")
            return True
        return False

    def clear_session_resumption(self, reason: str | None = None):
        """Clear session resumption state so next session starts fresh."""
        global CONFIG
        if reason:
            logger.info(f"Clearing session resumption state: {reason}")
        else:
            logger.info("Clearing session resumption state")
        # Clear local state
        self.session_handle = None
        self.last_consumed_message_index = None
        # Best-effort clear on global CONFIG, if available
        try:
            if hasattr(CONFIG, 'session_resumption') and CONFIG.session_resumption is not None:
                CONFIG.session_resumption.handle = None
        except Exception as e:
            logger.warning(f"Failed to clear session resumption handle on CONFIG: {e}")


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode
        self.config = config  # Store config reference
        
        # Get video configuration
        self.video_config = self.config.get('video', {})
        
        # Get queue configuration
        queue_config = self.config.get('queues', {})
        self.output_queue_maxsize = queue_config.get('output_queue_maxsize', 5)
        
        # Get debug configuration
        self.debug_config = self.config.get('debug', {})
        
        # Initialize session manager
        self.session_manager = SessionManager(self.config)

        self.audio_in_queue = None
        self.out_queue = None

        self.session = None
        
        self.receive_audio_task = None
        self.play_audio_task = None
        
        # V2 mode switching
        self.switch_to_v2_requested = False
        self.returning_from_v2 = False
        
        # Proactive session refresh
        self.deferred_reconnect_requested = False
        self._refresh_notice_sent = False

        # Audio stream handles
        self.audio_out_stream = None

    def _get_frame(self, cap):
        # Get image configuration
        image_config = self.video_config.get('image', {})
        thumbnail_size = image_config.get('thumbnail_size', [1024, 1024])
        image_format = image_config.get('format', 'jpeg')
        mime_type = image_config.get('mime_type', 'image/jpeg')
        
        # Read the frame
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail(thumbnail_size)

        image_io = io.BytesIO()
        img.save(image_io, format=image_format)
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # Get camera configuration
        camera_config = self.video_config.get('camera', {})
        device_index = camera_config.get('device_index', 0)
        frame_delay = camera_config.get('frame_delay', 1.0)
        
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, device_index
        )  # device_index represents the camera to use

        try:
            while True:
                frame = await asyncio.to_thread(self._get_frame, cap)
                if frame is None:
                    break

                await asyncio.sleep(frame_delay)

                await self.out_queue.put(frame)
        except asyncio.CancelledError:
            logger.debug("get_frames task cancelled")
            raise
        except Exception as cam_err:
            logger.warning(f"get_frames encountered error: {cam_err}")
            raise
        finally:
            try:
                cap.release()
            except Exception:
                pass

    def _get_screen(self):
        # Get screen and image configuration
        screen_config = self.video_config.get('screen', {})
        image_config = self.video_config.get('image', {})
        monitor_index = screen_config.get('monitor_index', 0)
        image_format = image_config.get('format', 'jpeg')
        mime_type = image_config.get('mime_type', 'image/jpeg')
        
        # Perform a single screen grab with robust error handling
        sct = None
        try:
            sct = mss.mss()
            # Validate monitor index against available monitors
            monitors = getattr(sct, 'monitors', [])
            if not monitors:
                logger.debug("No monitors reported by mss; skipping frame")
                return None
            # Clamp index into valid range
            safe_index = monitor_index
            if safe_index < 0 or safe_index >= len(monitors):
                logger.warning(f"Configured monitor_index {monitor_index} out of range (0..{len(monitors)-1}); using 0")
                safe_index = 0
            monitor = monitors[safe_index]

            try:
                i = sct.grab(monitor)
            except Exception as grab_err:
                # Intermittent Windows capture error (e.g., gdi32.GetDIBits failures)
                logger.debug(f"Screen grab failed: {grab_err}")
                return None

            try:
                if mss_tools is not None:
                    image_bytes = mss_tools.to_png(i.rgb, i.size)
                else:
                    raise RuntimeError("mss.tools not available for encoding screenshot")
                img = PIL.Image.open(io.BytesIO(image_bytes))

                image_io = io.BytesIO()
                img.save(image_io, format=image_format)
                image_io.seek(0)

                image_bytes = image_io.read()
                return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
            except Exception as enc_err:
                logger.debug(f"Screen encode failed: {enc_err}")
                return None
        finally:
            try:
                if sct is not None:
                    sct.close()
            except Exception:
                pass

    async def get_screen(self):
        # Get screen configuration
        screen_config = self.video_config.get('screen', {})
        frame_delay = screen_config.get('frame_delay', 1.0)

        try:
            while True:
                frame = await asyncio.to_thread(self._get_screen)
                if frame is None:
                    # Transient capture failure; back off slightly and continue
                    await asyncio.sleep(min(frame_delay, 0.2))
                    continue

                await asyncio.sleep(frame_delay)

                await self.out_queue.put(frame)
        except asyncio.CancelledError:
            logger.debug("get_screen task cancelled")
            raise
        except Exception as scr_err:
            logger.warning(f"get_screen encountered error: {scr_err}")
            raise

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            
            if msg is None:
                continue
            try:
                if isinstance(msg, dict):
                    mime = msg.get('mime_type')
                    data = msg.get('data')

                    if data is None and 'text' not in msg:
                        logger.warning(f"Invalid message format in send_realtime, skipping: {msg}")
                        continue

                    if 'text' in msg and (not mime and data is None):
                        await self.session.send_realtime_input(text=msg['text'])
                        continue

                    # If data is a base64-encoded string, decode it; otherwise keep bytes
                    raw = data
                    if isinstance(data, str):
                        try:
                            raw = base64.b64decode(data)
                        except Exception:
                            raw = data.encode('utf-8')

                    if mime and isinstance(mime, str) and mime.startswith('audio/'):
                        blob = types.Blob(data=bytes(raw) if raw is not None else None, mime_type=mime)
                        await self.session.send_realtime_input(audio=blob)
                    elif mime and isinstance(mime, str) and (mime.startswith('image/') or mime.startswith('video/')):
                        blob = types.Blob(data=bytes(raw) if raw is not None else None, mime_type=mime)
                        await self.session.send_realtime_input(media=blob)
                    else:
                        if isinstance(raw, (bytes, bytearray)):
                            blob = types.Blob(data=bytes(raw), mime_type=mime or 'application/octet-stream')
                            await self.session.send_realtime_input(media=blob)
                        else:
                            await self.session.send_realtime_input(text=str(raw))
                else:
                    if isinstance(msg, (bytes, bytearray)):
                        blob = types.Blob(data=bytes(msg), mime_type='audio/pcm')
                        await self.session.send_realtime_input(audio=blob)
                    elif isinstance(msg, str):
                        await self.session.send_realtime_input(text=msg)
                    else:
                        await self.session.send_realtime_input(text=str(msg))
            except Exception as send_err:
                logger.warning(f"send_realtime encountered error when sending message: {send_err}")
                continue

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )

        # Use debug configuration
        if __debug__ and not self.debug_config.get('exception_on_overflow', False):
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}

        try:
            while True:
                data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
                # If yap mode is enabled AND AI is speaking, drop mic input to prevent interruptions
                try:
                    if (
                        getattr(memory_tools, 'is_yap_mode_enabled', lambda: False)()
                        and getattr(memory_tools, 'is_ai_speaking', lambda: False)()
                    ):
                        continue
                except Exception:
                    # If tools not initialized yet, fall back to sending
                    pass
                await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})
        except asyncio.CancelledError:
            logger.debug("listen_audio task cancelled")
            raise
        except Exception as mic_err:
            logger.warning(f"listen_audio encountered error: {mic_err}")
            # Propagate to trigger reconnect logic
            raise
        finally:
            try:
                if self.audio_stream is not None:
                    try:
                        if hasattr(self.audio_stream, 'stop_stream'):
                            self.audio_stream.stop_stream()
                    except Exception:
                        pass
                    try:
                        self.audio_stream.close()
                    except Exception as e:
                        logger.debug(f"Error closing input stream: {e}")
            finally:
                self.audio_stream = None

    async def receive_audio(self):
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        current_text_response = ""  # Accumulate text for VRChat
        
        while True:
            try:
                turn = self.session.receive()
                if asyncio.iscoroutine(turn):
                    turn = await turn
            except Exception as recv_err:
                logger.warning(f"Error obtaining turn iterator from session.receive(): {recv_err}")
                raise

            async for response in turn:
                # Handle audio data
                if data := response.data:
                    try:
                        import sfx as _sfx
                        if hasattr(_sfx, 'sfx_manager') and _sfx.sfx_manager.is_music_playing():
                            pass
                        else:
                            self.audio_in_queue.put_nowait(data)
                            try:
                                if MYINSTANTS_AVAILABLE and myinstants_client:
                                    myinstants_client.notify_ai_audio_received()
                            except Exception as myinstants_notify_error:
                                logger.warning(f"MyInstants audio notification error: {myinstants_notify_error}")
                            try:
                                osc.notify_ai_speech_start()
                            except Exception as osc_start_error:
                                logger.warning(f"OSC speech start notification error: {osc_start_error}")
                            try:
                                if hasattr(memory_tools, 'set_ai_speaking'):
                                    memory_tools.set_ai_speaking(True)
                            except Exception as yap_start_error:
                                logger.warning(f"YAP state start update error: {yap_start_error}")
                    except Exception as audio_data_error:
                        logger.warning(f"Audio data processing error: {audio_data_error}")
                        # Fallback: still queue the audio data
                        self.audio_in_queue.put_nowait(data)
                        try:
                            if MYINSTANTS_AVAILABLE and myinstants_client:
                                myinstants_client.notify_ai_audio_received()
                        except Exception:
                            pass
                        try:
                            osc.notify_ai_speech_start()
                        except Exception:
                            pass
                        try:
                            if hasattr(memory_tools, 'set_ai_speaking'):
                                memory_tools.set_ai_speaking(True)
                        except Exception:
                            pass
                    continue
                
                # Handle text responses (for TEXT mode)
                if text := response.text:
                    # Accumulate text for VRChat
                    current_text_response += text
                    
                    # Broadcast to WebSocket clients if available
                    if CHAT_API_AVAILABLE and chat_api:
                        try:
                            chat_api.broadcast_gabriel_response(text, "text_chunk")
                        except Exception as e:
                            logger.warning(f"Failed to broadcast text to WebSocket clients: {e}")
                
                # Handle audio transcriptions (for AUDIO mode with transcription enabled)
                if hasattr(response, 'server_content') and response.server_content:
                    # Handle user input transcription
                    if hasattr(response.server_content, 'input_transcription') and response.server_content.input_transcription:
                        input_text = response.server_content.input_transcription.text
                        if input_text:
                            logger.info(f"[User]: {input_text}")
                            
                            # Broadcast user input transcription to WebSocket clients
                            if CHAT_API_AVAILABLE and chat_api:
                                try:
                                    chat_api.broadcast_gabriel_response(input_text, "user_input_transcription")
                                except Exception as e:
                                    logger.warning(f"Failed to broadcast user input transcription to WebSocket clients: {e}")
                    
                    # Handle Gabriel output transcription
                    if hasattr(response.server_content, 'output_transcription') and response.server_content.output_transcription:
                        transcription_text = response.server_content.output_transcription.text
                        if transcription_text:
                            # Accumulate transcription text for VRChat
                            current_text_response += transcription_text
                            
                            # Broadcast to WebSocket clients if available
                            if CHAT_API_AVAILABLE and chat_api:
                                try:
                                    chat_api.broadcast_gabriel_response(transcription_text, "transcription_chunk")
                                except Exception as e:
                                    logger.warning(f"Failed to broadcast transcription to WebSocket clients: {e}")
                
                # Handle tool calls
                if hasattr(response, 'tool_call') and response.tool_call:
                    function_responses = []
                    for fc in response.tool_call.function_calls:
                        logger.info(f"Function call: {fc.name} with args: {fc.args}")
                        
                        if CHAT_API_AVAILABLE and chat_api:
                            try:
                                chat_api.broadcast_to_websockets("function_call", {
                                    "name": fc.name,
                                    "args": fc.args
                                })
                            except Exception as e:
                                logger.warning(f"Failed to broadcast function call to WebSocket clients: {e}")
                        
                        try:
                            function_response = await memory_tools.handle_function_call(fc)
                            function_responses.append(function_response)
                            logger.info(f"Function response: {function_response.response}")
                            
                            if CHAT_API_AVAILABLE and chat_api:
                                try:
                                    chat_api.broadcast_to_websockets("function_response", {
                                        "name": fc.name,
                                        "response": function_response.response
                                    })
                                except Exception as e:
                                    logger.warning(f"Failed to broadcast function response to WebSocket clients: {e}")
                            
                            # Check if this is a V2 mode switch request
                            if (fc.name == "switch_to_v2_mode" and 
                                function_response.response.get("success") and 
                                function_response.response.get("action") == "switch_to_v2"):
                                logger.info("V2 mode switch requested, setting flag")
                                # Set a flag that the main loop can check
                                self.switch_to_v2_requested = True
                                
                        except Exception as e:
                            logger.error(f"Error handling function call {fc.name}: {e}")
                            # Create error response
                            error_response = types.FunctionResponse(
                                id=fc.id,
                                name=fc.name,
                                response={
                                    "success": False,
                                    "message": f"Error executing {fc.name}: {str(e)}"
                                }
                            )
                            function_responses.append(error_response)
                    
                    # Send function responses back to the model
                    if function_responses:
                        await self.session.send_tool_response(function_responses=function_responses)
                
                # Handle session resumption updates
                if hasattr(response, 'session_resumption_update') and response.session_resumption_update:
                    self.session_manager.handle_session_resumption_update(response.session_resumption_update)
                
                # Handle GoAway messages
                if hasattr(response, 'go_away') and response.go_away:
                    self.session_manager.handle_go_away(response.go_away)
                    # Request a controlled reconnect after the current turn completes
                    self.deferred_reconnect_requested = True

            # Turn complete - send accumulated text to VRChat and notify services
            if current_text_response.strip():
                try:
                    await osc.send_to_vrchat(current_text_response.strip())
                except Exception as osc_error:
                    logger.warning(f"Failed to send message to VRChat: {osc_error}")
                    # Don't let OSC errors crash the session
                
                # Broadcast complete response to WebSocket clients
                if CHAT_API_AVAILABLE and chat_api:
                    try:
                        chat_api.broadcast_gabriel_response(current_text_response.strip(), "complete_response")
                    except Exception as e:
                        logger.warning(f"Failed to broadcast complete response to WebSocket clients: {e}")
                        
                current_text_response = ""  # Reset for next turn
                
            # Turn complete - notify MyInstants that speech has ended
            try:
                if MYINSTANTS_AVAILABLE and myinstants_client:
                    myinstants_client.notify_ai_speech_ended()
            except Exception as myinstants_error:
                logger.warning(f"MyInstants notification error: {myinstants_error}")
            
            # Notify OSC client that Gabriel's speech has ended
            try:
                osc.notify_ai_speech_end()
            except Exception as osc_notify_error:
                logger.warning(f"OSC speech end notification error: {osc_notify_error}")
            
            try:
                if hasattr(memory_tools, 'set_ai_speaking'):
                    memory_tools.set_ai_speaking(False)
            except Exception as yap_error:
                logger.warning(f"YAP state update error: {yap_error}")
            
            try:
                if hasattr(self, 'switch_to_v2_requested') and self.switch_to_v2_requested:
                    logger.info("V2 mode switch detected, raising signal")
                    # Clear the flag first
                    self.switch_to_v2_requested = False
                    raise V2ModeSwitchRequested("V2 mode switch requested by AI")
            except V2ModeSwitchRequested:
                # Re-raise the V2ModeSwitchRequested exception - this is the expected behavior
                raise
            except Exception as v2_check_error:
                logger.warning(f"Error checking V2 mode switch state: {v2_check_error}")
                # Reset flag to prevent stuck state
                if hasattr(self, 'switch_to_v2_requested'):
                    self.switch_to_v2_requested = False
            
            # If a controlled reconnect was requested (GoAway or lifetime monitor), do it now at safe boundary
            if self.deferred_reconnect_requested:
                logger.info("Controlled reconnect requested; restarting session at turn boundary")
                self.deferred_reconnect_requested = False
                # Clear any queued audio to prevent playback issues
                try:
                    while not self.audio_in_queue.empty():
                        self.audio_in_queue.get_nowait()
                except Exception as clear_error:
                    logger.warning(f"Error clearing audio queue during reconnect: {clear_error}")
                raise ControlledReconnectRequested("Proactive session refresh at turn boundary")
            
            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        self.audio_out_stream = stream
        try:
            while True:
                bytestream = await self.audio_in_queue.get()
                await asyncio.to_thread(stream.write, bytestream)
        except asyncio.CancelledError:
            logger.debug("play_audio task cancelled")
            raise
        except Exception as spk_err:
            logger.warning(f"play_audio encountered error: {spk_err}")
            raise
        finally:
            try:
                if self.audio_out_stream is not None:
                    try:
                        if hasattr(self.audio_out_stream, 'stop_stream'):
                            self.audio_out_stream.stop_stream()
                    except Exception:
                        pass
                    try:
                        self.audio_out_stream.close()
                    except Exception as e:
                        logger.debug(f"Error closing output stream: {e}")
            finally:
                self.audio_out_stream = None

    async def run_session(self):
        """Run a single session with the configured setup."""
        session_config = self.session_manager.get_session_config()
        session_start_time = time.time()
        
        try:
            async with (
                client.aio.live.connect(model=MODEL, config=session_config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.session_manager.reset_connection_state()
                logger.info("Session connected successfully")
                
                self.session_manager.start_periodic_save(mode="v1")

                # Register session with Chat API
                try:
                    if CHAT_API_AVAILABLE and chat_api:
                        chat_api.register_session(session)
                        logger.info("Session registered with Chat API")
                except Exception as api_reg_error:
                    logger.warning(f"Failed to register session with Chat API: {api_reg_error}")

                # Send V1 mode system instruction if returning from V2 mode
                if self.returning_from_v2:
                    try:
                        logger.info("Sending V1 mode system instruction to Gabriel")
                        await session.send_client_content(
                            turns={"role": "user", "parts": [{"text": "SYSTEM INSTRUCTION: You are now in V1 mode"}]},
                            turn_complete=True
                        )
                        self.returning_from_v2 = False  # Reset the flag
                    except Exception as system_msg_error:
                        logger.warning(f"Failed to send V1 mode system instruction: {system_msg_error}")
                        self.returning_from_v2 = False  # Reset the flag even on error

                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=self.output_queue_maxsize)

                tg.create_task(self.send_realtime())
                tg.create_task(self.listen_audio())
                if self.video_mode == "camera":
                    tg.create_task(self.get_frames())
                elif self.video_mode == "screen":
                    tg.create_task(self.get_screen())

                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())
                tg.create_task(self._monitor_connection_lifetime())

                # Keep the session running indefinitely until interrupted
                while True:
                    if self.session_manager.fresh_start_requested:
                        logger.info("Fresh start requested in main loop - exiting session")
                        raise asyncio.CancelledError("Fresh start requested")
                    
                    if self.session_manager.reconnect_requested:
                        logger.info("Reconnect requested in main loop - exiting session")
                        raise asyncio.CancelledError("Reconnect requested")
                    
                    await asyncio.sleep(0.5)
                    
        except KeyboardInterrupt:
            session_duration = time.time() - session_start_time
            logger.info(f"KeyboardInterrupt received after {session_duration:.2f}s, ending session")
            raise asyncio.CancelledError("User requested exit")
            
        except Exception as e:
            # Check if this exception contains V2ModeSwitchRequested
            if hasattr(e, 'exceptions'):
                for exc in e.exceptions:
                    if isinstance(exc, V2ModeSwitchRequested):
                        session_duration = time.time() - session_start_time
                        logger.info(f"V2 mode switch found in ExceptionGroup after {session_duration:.2f}s")
                        raise V2ModeSwitchRequested("V2 mode switch requested by AI")
            
            # Check if this is a standalone V2ModeSwitchRequested
            if isinstance(e, V2ModeSwitchRequested):
                session_duration = time.time() - session_start_time
                logger.info(f"V2 mode switch requested after {session_duration:.2f}s, ending V1 session")
                raise e
            
            # Re-raise other exceptions
            raise e
            
        finally:
            session_duration = time.time() - session_start_time
            logger.info(f"Session ended after {session_duration:.2f} seconds")
            
            self.session_manager.stop_periodic_save(mode="v1")
            
            # Unregister session from Chat API
            try:
                if CHAT_API_AVAILABLE and chat_api:
                    chat_api.unregister_session()
                    logger.debug("Session unregistered from Chat API")
            except Exception as api_unreg_error:
                logger.warning(f"Failed to unregister session from Chat API: {api_unreg_error}")

    async def run(self):
        """Main run method with automatic reconnection support and V2 mode switching."""
        retry_count = 0
        consecutive_failures = 0
        last_successful_connection = None
        
        logger.info(f"Starting main session loop with auto-reconnect (max_retries: {self.session_manager.max_retries})")
        logger.info(f"Session manager configuration: auto_reconnect={self.session_manager.auto_reconnect_config.get('enabled', True)}, initial_delay={self.session_manager.initial_delay}s")
        
        while True:
            connection_start = time.time()
            logger.info(f"Starting session attempt {retry_count + 1} at {time.strftime('%H:%M:%S')}")
            logger.info(f"DEBUG: About to start session, returning_from_v2={getattr(self, 'returning_from_v2', False)}")
            
            try:
                logger.info(f"Starting session (attempt {retry_count + 1})")
                await self.run_session()
                
                # If we reach here, user requested exit
                logger.info("Session ended normally by user request")
                break
                
            except asyncio.CancelledError as e:
                if str(e) == "User requested exit":
                    logger.info("User requested exit")
                    break
                elif str(e) == "Fresh start requested":
                    logger.info("Fresh start requested - restarting with fresh session")
                    self.session_manager.fresh_start_requested = False
                    consecutive_failures = 0
                    retry_count = 0
                    await asyncio.sleep(0.5)
                    continue
                elif str(e) == "Reconnect requested":
                    logger.info("Reconnect requested - restarting with saved session")
                    self.session_manager.reconnect_requested = False
                    consecutive_failures = 0
                    retry_count = 0
                    await asyncio.sleep(0.5)
                    continue
                else:
                    # This might be from a GoAway or connection issue
                    logger.warning(f"Session cancelled: {e}, checking if reconnection is needed")
                    # Even if not explicitly configured, proceed with conservative reconnect
                    if not self.session_manager.should_attempt_reconnect(e):
                        logger.warning("CancelledError not marked reconnectable; proceeding with conservative reconnect")
            
            except ControlledReconnectRequested as e:
                # Proactive refresh: do not penalize or backoff aggressively
                logger.info(f"Proactive session refresh requested: {e}")
                logger.info("This is a planned reconnection, not an error - resetting failure counters")
                # Reset counters to avoid artificial backoff
                consecutive_failures = 0
                retry_count = 0
                # Continue to reconnection logic without delay
                continue
            
            except V2ModeSwitchRequested as e:
                logger.info("V2 mode switch requested, attempting to switch")
                
                if not V2_AVAILABLE:
                    logger.error("V2 mode not available, continuing with V1")
                    # Send a message to the user via VRChat
                    await osc.send_to_vrchat("V2 mode is not available. Continuing with V1 mode.")
                else:
                    try:
                        # Clean up current audio streams before switching
                        if hasattr(self, 'audio_stream') and self.audio_stream:
                            try:
                                if hasattr(self.audio_stream, 'stop_stream'):
                                    self.audio_stream.stop_stream()
                            except Exception:
                                pass
                            try:
                                self.audio_stream.close()
                            except Exception as cleanup_e:
                                logger.warning(f"Error closing input audio stream during V2 switch: {cleanup_e}")
                            finally:
                                self.audio_stream = None
                        if hasattr(self, 'audio_out_stream') and self.audio_out_stream:
                            try:
                                if hasattr(self.audio_out_stream, 'stop_stream'):
                                    self.audio_out_stream.stop_stream()
                            except Exception:
                                pass
                            try:
                                self.audio_out_stream.close()
                            except Exception as cleanup_e:
                                logger.warning(f"Error closing output audio stream during V2 switch: {cleanup_e}")
                            finally:
                                self.audio_out_stream = None
                        
                        # Notify user about the switch
                        await osc.send_to_vrchat("Switching to V2 mode with enhanced voice quality...")
                        
                        # Switch to V2 mode
                        logger.info("Switching to V2 mode...")
                        result = await v2.run_v2_mode(
                            self.config, 
                            client, 
                            video_mode=self.video_mode,
                            session_handle=self.session_manager.session_handle
                        )
                        
                        if result == "switch_to_v1":
                            logger.info("V2 mode requested switch back to V1, restarting main loop")
                            await osc.send_to_vrchat("Switched back to V1 mode.")
                            # Set flag to notify Gabriel about V1 mode when the next session starts
                            self.returning_from_v2 = True
                            # Reset the switch flag and break out of retry loop to restart
                            self.switch_to_v2_requested = False
                            logger.info("Breaking out of retry loop to restart V1 session")
                            break
                        else:
                            logger.info("V2 mode ended, restarting V1 mode")
                            await osc.send_to_vrchat("Returned to V1 mode.")
                            # Set flag to notify Gabriel about V1 mode when the next session starts
                            self.returning_from_v2 = True
                            # Reset the switch flag and break out of retry loop to restart
                            self.switch_to_v2_requested = False
                            logger.info("Breaking out of retry loop to restart V1 session")
                            break
                            
                    except Exception as v2_error:
                        logger.error(f"Error in V2 mode: {v2_error}")
                        logger.info("Falling back to V1 mode")
                        await osc.send_to_vrchat("V2 mode encountered an error. Falling back to V1 mode.")
                        # Set flag to notify Gabriel about V1 mode when the next session starts
                        self.returning_from_v2 = True
                        # Reset the switch flag and break out of retry loop to restart
                        self.switch_to_v2_requested = False
                        # Don't increment retry count for V2 failures, just restart
                        logger.info("Breaking out of retry loop to restart V1 session after V2 error")
                        break
            
            except Exception as e:
                proactive_refresh = False
                if hasattr(e, 'exceptions') and e.exceptions:
                    for inner in flatten_exception_group(e):
                        if isinstance(inner, ControlledReconnectRequested):
                            proactive_refresh = True
                            break
                if proactive_refresh:
                    logger.info("Proactive session refresh requested via ExceptionGroup")
                    logger.info("This is a planned reconnection, not an error - resetting failure counters")
                    consecutive_failures = 0
                    retry_count = 0
                    continue
                logger.info(f"EXCEPTION CAUGHT: {type(e).__name__}: {str(e)[:200]}")
                reconnectable = self.session_manager.should_attempt_reconnect(e)
                if reconnectable:
                    logger.warning(f"Reconnectable session error: {e}")
                else:
                    logger.error(f"Session error: {e}")
                
                # Add detailed logging for exception type
                logger.info(f"Exception type: {type(e).__name__}")
                logger.info(f"Exception has 'exceptions' attribute: {hasattr(e, 'exceptions')}")
                
                exceptions_attr = getattr(e, 'exceptions', None)
                if exceptions_attr:
                    v2_switch_handled = False
                    logger.info(f"Processing ExceptionGroup with {len(exceptions_attr)} exceptions")
                    if not reconnectable:
                        logger.error(f"ExceptionGroup contains {len(exceptions_attr)} exceptions:")
                    for i, exc in enumerate(exceptions_attr):
                        logger.info(f"  Checking exception {i+1}: {type(exc).__name__}: {exc}")
                        if not reconnectable:
                            logger.error(f"  Exception {i+1}: {type(exc).__name__}: {exc}")
                        if isinstance(exc, V2ModeSwitchRequested):
                            logger.info("Found V2ModeSwitchRequested in ExceptionGroup, handling V2 switch")
                            v2_switch_handled = True
                            if not V2_AVAILABLE:
                                logger.error("V2 mode not available, continuing with V1")
                                await osc.send_to_vrchat("V2 mode is not available. Continuing with V1 mode.")
                            else:
                                try:
                                    if hasattr(self, 'audio_stream') and self.audio_stream:
                                        try:
                                            if hasattr(self.audio_stream, 'stop_stream'):
                                                self.audio_stream.stop_stream()
                                        except Exception:
                                            pass
                                        try:
                                            self.audio_stream.close()
                                        except Exception as cleanup_e:
                                            logger.warning(f"Error closing input audio stream during V2 switch: {cleanup_e}")
                                        finally:
                                            self.audio_stream = None
                                    if hasattr(self, 'audio_out_stream') and self.audio_out_stream:
                                        try:
                                            if hasattr(self.audio_out_stream, 'stop_stream'):
                                                self.audio_out_stream.stop_stream()
                                        except Exception:
                                            pass
                                        try:
                                            self.audio_out_stream.close()
                                        except Exception as cleanup_e:
                                            logger.warning(f"Error closing output audio stream during V2 switch: {cleanup_e}")
                                        finally:
                                            self.audio_out_stream = None
                                    await osc.send_to_vrchat("Switching to V2 mode with enhanced voice quality...")
                                    logger.info("Switching to V2 mode...")
                                    result = await v2.run_v2_mode(
                                        self.config,
                                        client,
                                        video_mode=self.video_mode,
                                        session_handle=self.session_manager.session_handle
                                    )
                                    if result == "switch_to_v1":
                                        logger.info("V2 mode requested switch back to V1, breaking retry loop")
                                        await osc.send_to_vrchat("Switched back to V1 mode.")
                                        self.returning_from_v2 = True
                                        self.switch_to_v2_requested = False
                                        v2_switch_handled = True
                                        break  # Break out of retry loop to restart V1
                                    else:
                                        logger.info("V2 mode ended, breaking retry loop to return to V1")
                                        await osc.send_to_vrchat("Returned to V1 mode.")
                                        self.returning_from_v2 = True
                                        self.switch_to_v2_requested = False
                                        v2_switch_handled = True
                                        break  # Break out of retry loop to restart V1
                                except Exception as v2_error:
                                    logger.error(f"Error in V2 mode: {v2_error}")
                                    logger.info("Falling back to V1 mode")
                                    await osc.send_to_vrchat("V2 mode encountered an error. Falling back to V1 mode.")
                                    self.returning_from_v2 = True
                                    self.switch_to_v2_requested = False
                                    v2_switch_handled = True
                                    break  # Break out of retry loop to restart V1
                            if v2_switch_handled:
                                break  # Break out of the main retry loop as well
                
                # Check connection duration to see if it was a quick failure
                connection_duration = time.time() - connection_start
                if connection_duration < 10:  # Failed within 10 seconds
                    consecutive_failures += 1
                    logger.warning(f"Quick failure detected ({connection_duration:.2f}s), consecutive failures: {consecutive_failures}")
                else:
                    consecutive_failures = 0  # Reset on successful longer connection
                    # Also reset retry count for longer connections (successful sessions)
                    if connection_duration > 300:  # 5 minutes or more
                        retry_count = 0
                        logger.info(f"Long-lived session ({connection_duration:.2f}s), resetting retry count")
                
                if not reconnectable:
                    logger.warning("Error not configured for reconnection; proceeding with conservative reconnect")
                    # Add a small delay to avoid tight loop
                    try:
                        await asyncio.sleep(5)
                    except asyncio.CancelledError:
                        logger.info("Conservative delay interrupted by cancellation")
                        break
                
                # If we have too many consecutive quick failures, wait longer
                if consecutive_failures >= 3:
                    extra_delay = min(consecutive_failures * 5, 60)  # Up to 60 seconds extra
                    logger.warning(f"Multiple consecutive failures, adding {extra_delay}s extra delay")
                    try:
                        await asyncio.sleep(extra_delay)
                    except asyncio.CancelledError:
                        logger.info("Extra delay sleep interrupted by cancellation")
                        break
                    except Exception as sleep_e:
                        logger.warning(f"Exception during extra delay sleep: {sleep_e}")
            
            finally:
                # Clean up audio stream if it exists
                cleanup_errors = []
                
                if hasattr(self, 'audio_stream') and self.audio_stream:
                    try:
                        try:
                            if hasattr(self.audio_stream, 'stop_stream'):
                                self.audio_stream.stop_stream()
                        except Exception:
                            pass
                        self.audio_stream.close()
                        logger.debug("Input audio stream cleaned up successfully")
                    except Exception as e:
                        cleanup_errors.append(f"input audio stream: {e}")
                    finally:
                        self.audio_stream = None
                        
                if hasattr(self, 'audio_out_stream') and self.audio_out_stream:
                    try:
                        try:
                            if hasattr(self.audio_out_stream, 'stop_stream'):
                                self.audio_out_stream.stop_stream()
                        except Exception:
                            pass
                        self.audio_out_stream.close()
                        logger.debug("Output audio stream cleaned up successfully")
                    except Exception as e:
                        cleanup_errors.append(f"output audio stream: {e}")
                    finally:
                        self.audio_out_stream = None
                
                # Log any cleanup errors
                if cleanup_errors:
                    logger.warning(f"Cleanup errors: {'; '.join(cleanup_errors)}")
                else:
                    logger.debug("All audio streams cleaned up successfully")
            
            # Calculate reconnection delay
            delay = await self.session_manager.calculate_reconnect_delay(retry_count)
            if delay is None:
                # Fallback to a safe default delay to prevent full exit
                delay = 5.0
                logger.warning("No reconnection delay available (possibly disabled or max retries). Using default 5s and continuing.")
            
            retry_count += 1
            logger.info(f"Attempting reconnection {retry_count}/{self.session_manager.max_retries} in {delay:.2f} seconds...")
            logger.info(f"Connection failure stats: consecutive_failures={consecutive_failures}, retry_count={retry_count}")
            
            # Sleep before reconnection attempt with proper exception handling
            try:
                logger.debug(f"Sleeping for {delay:.2f}s before reconnection attempt")
                await asyncio.sleep(delay)
                logger.debug("Reconnection delay completed, proceeding with new session")
            except asyncio.CancelledError:
                logger.info("Reconnection sleep interrupted by cancellation")
                break
            except Exception as e:
                logger.warning(f"Exception during reconnection sleep: {e}")
                # Continue with reconnection attempt despite sleep interruption
            
        logger.info("Main session loop ended")

    async def _monitor_connection_lifetime(self):
        # Proactive refresh monitor based on configured limits and margins
        sm_cfg = self.config.get('session_management', {})
        pr_cfg = sm_cfg.get('proactive_refresh', {})
        enabled = pr_cfg.get('enabled', True)
        if not enabled:
            return
        # Defaults: 600s lifetime, refresh 55s before cutoff
        lifetime = float(pr_cfg.get('connection_lifetime_seconds', 600))
        margin = float(pr_cfg.get('refresh_margin_seconds', 55))
        if margin < 5:
            margin = 5.0
        threshold = max(0.0, lifetime - margin)
        notice_sent = False
        while True:
            try:
                await asyncio.sleep(1)
                
                start = self.session_manager.connection_start_time or time.time()
                elapsed = time.time() - start
                if elapsed >= threshold and not self.deferred_reconnect_requested:
                    self.deferred_reconnect_requested = True
                    if not notice_sent and not self._refresh_notice_sent:
                        try:
                            await osc.send_to_vrchat("Refreshing connection soon to avoid server cutoff...")
                        except Exception:
                            pass
                        self._refresh_notice_sent = True
                        notice_sent = True
            except asyncio.CancelledError:
                break
            except Exception as mon_e:
                logger.debug(f"Lifetime monitor error: {mon_e}")
                # Continue monitoring


def flatten_exception_group(error):
    stack = [error]
    while stack:
        current = stack.pop()
        if hasattr(current, "exceptions") and current.exceptions:
            stack.extend(current.exceptions)
        else:
            yield current


def is_recoverable_main_error(error, session_manager):
    if not session_manager:
        return False
    recoverable = False
    for inner in flatten_exception_group(error):
        if isinstance(inner, V2ModeSwitchRequested):
            return False
        if isinstance(inner, ControlledReconnectRequested):
            recoverable = True
            continue
        if isinstance(inner, asyncio.CancelledError):
            recoverable = True
            continue
        if session_manager._is_reconnectable_error(inner):
            recoverable = True
            continue
        return False
    return recoverable


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=None,  # Will use config default if not specified
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="path to configuration file",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default="prompts.json",
        help="path to prompts file",
    )
    args = parser.parse_args()
    
    try:
        # Load configuration with custom path
        config = load_config(args.config)
        
        # Setup logging based on configuration
        setup_logging(config)
        
        # Load prompts with custom path
        prompts = load_prompts(args.prompts)
        
        # Setup global variables from configuration
        setup_globals(config, prompts)
        
        # Use command line mode if specified, otherwise use config default
        video_mode = args.mode if args.mode is not None else DEFAULT_MODE
        
        main = AudioLoop(video_mode=video_mode)
        
        if CHAT_API_AVAILABLE and chat_api:
            try:
                chat_api.register_session_manager(main.session_manager)
                logger.info("Session manager registered with Chat API")
            except Exception as api_reg_error:
                logger.warning(f"Failed to register session manager with Chat API: {api_reg_error}")
        
        logger.info("Starting ProjectGabriel main application")
        retry = 0
        max_main_retries = 10  # Limit main loop retries to prevent infinite loops
        
        while retry < max_main_retries:
            try:
                logger.info(f"Starting main run loop (main retry {retry + 1}/{max_main_retries})")
                result = asyncio.run(main.run())
                # If run() returns normally (None) we keep the process alive and restart the loop
                if result is not None:
                    logger.info(f"Main run returned: {result}")
                # Short pause to avoid tight restart loop
                logger.info("Main run completed normally, restarting...")
                time.sleep(1)
                retry = 0  # Reset retry counter on successful completion
                continue
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt received in main loop")
                raise
            except Exception as unhandled:
                if is_recoverable_main_error(unhandled, getattr(main, "session_manager", None)):
                    logger.warning(f"Recoverable main loop error {type(unhandled).__name__}: {str(unhandled)[:200]}")
                    time.sleep(1)
                    continue
                retry += 1
                logger.error(f"Unhandled error in main loop (attempt {retry}/{max_main_retries}): {unhandled}")
                logger.error(f"Exception type: {type(unhandled).__name__}")
                try:
                    logger.error(f"Exception details: {traceback.format_exc()}")
                except Exception:
                    logger.error("Could not format exception traceback")
                backoff = min(30, 2 ** min(retry, 5))
                logger.info(f"Restarting main loop in {backoff}s (attempt {retry}/{max_main_retries})...")
                time.sleep(backoff)
                continue
            except BaseException as base_exc:
                # Re-raise normal termination signals so the process can exit when intended
                if isinstance(base_exc, (KeyboardInterrupt, SystemExit)):
                    logger.info(f"Received termination signal: {type(base_exc).__name__}")
                    raise
                # Handle ExceptionGroup (thrown by TaskGroup) which is a BaseException in Python 3.11+
                if hasattr(base_exc, 'exceptions'):
                    if is_recoverable_main_error(base_exc, getattr(main, "session_manager", None)):
                        logger.warning(f"Recoverable ExceptionGroup in main loop: {str(base_exc)[:200]}")
                        time.sleep(1)
                        continue
                    retry += 1
                    logger.error(f"Unhandled ExceptionGroup in main loop (attempt {retry}/{max_main_retries}): {base_exc}")
                    for i, ie in enumerate(base_exc.exceptions):
                        logger.error(f"  Inner exception {i+1}: {type(ie).__name__}: {ie}")
                    try:
                        logger.error('ExceptionGroup traceback:')
                        logger.error(''.join(traceback.format_exception(type(base_exc), base_exc, base_exc.__traceback__)))
                    except Exception:
                        logger.error("Could not format ExceptionGroup traceback")
                    backoff = min(30, 2 ** min(retry, 5))
                    logger.info(f"Restarting main loop in {backoff}s (attempt {retry}/{max_main_retries})...")
                    time.sleep(backoff)
                    continue
                # Unknown BaseException: log and re-raise to avoid masking critical signals
                logger.error(f"Critical BaseException in main loop (re-raising): {type(base_exc).__name__}: {base_exc}")
                raise
        
        # If we reach here, we've exceeded max retries
        logger.error(f"Exceeded maximum main loop retries ({max_main_retries}). Process will exit.")
        logger.error("This may indicate a persistent issue that requires manual intervention.")
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Fatal error in main application: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception details: {traceback.format_exc()}")
    finally:
        try:
            if _VISION_AVAILABLE and (_vision_module is not None) and hasattr(_vision_module, 'stop_following'):
                _vision_module.stop_following()
                logger.info("Vision follower stopped")
        except Exception as vision_stop_err:
            logger.warning(f"Error stopping vision follower: {vision_stop_err}")
        logger.info("ProjectGabriel application terminated")
