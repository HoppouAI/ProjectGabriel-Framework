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

import argparse

from google import genai
from google.genai import types

# Import GenAI errors for quota detection
try:
    from google.genai import errors
    GENAI_ERRORS_AVAILABLE = True
except ImportError:
    GENAI_ERRORS_AVAILABLE = False
    errors = None

# Try to import common WebSocket exceptions that might be raised
try:
    from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    # If websockets library is not available, we'll handle these as generic exceptions
    WEBSOCKETS_AVAILABLE = False
    ConnectionClosedError = Exception  # Fallback
    ConnectionClosedOK = Exception  # Fallback

# Import our tools module
import tools as memory_tools

# Import OSC module for VRChat integration
import osc

# Import append system for system prompt enhancement
import append

# Import Chat API for WebUI functionality
try:
    from api import chat as chat_api
    CHAT_API_AVAILABLE = True
except ImportError:
    CHAT_API_AVAILABLE = False
    chat_api = None

# Import MyInstants for audio timing notifications
try:
    from myinstants import myinstants_client
    MYINSTANTS_AVAILABLE = True
except ImportError:
    MYINSTANTS_AVAILABLE = False
    myinstants_client = None

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

# Model and default configuration for V2
V2_MODEL = "models/gemini-2.5-flash-native-audio-preview-09-2025"
V2_DEFAULT_MODE = "screen"

# Custom exception for V1 mode switching
class V1ModeSwitchRequested(Exception):
    """Exception raised when V1 mode switch is requested."""
    pass


class APIKeyManager:
    """Manages API keys with automatic failover for quota exceeded errors."""
    
    def __init__(self, config):
        self.config = config
        api_config = config.get('api', {})
        
        # Primary API key
        self.primary_key = None
        if 'api_key' in api_config:
            self.primary_key = api_config['api_key']
        elif 'key_env_var' in api_config:
            self.primary_key = os.environ.get(api_config['key_env_var'])
        else:
            self.primary_key = os.environ.get('GEMINI_API_KEY')
        
        # Backup API keys
        self.backup_keys = api_config.get('backup_api_keys', [])
        
        # Current state
        self.current_key_index = -1  # -1 means using primary key
        self.failed_keys = set()  # Track keys that have failed
        self.last_switch_time = 0
        self.switch_cooldown = 60  # 60 seconds cooldown between switches
        
        # Validate we have at least a primary key
        if not self.primary_key:
            raise ValueError("No primary API key found!")
        
        logger.info(f"APIKeyManager initialized with primary key and {len(self.backup_keys)} backup keys")
    
    def get_current_key(self):
        """Get the currently active API key."""
        if self.current_key_index == -1:
            return self.primary_key
        elif 0 <= self.current_key_index < len(self.backup_keys):
            return self.backup_keys[self.current_key_index]
        else:
            # Fallback to primary if index is invalid
            self.current_key_index = -1
            return self.primary_key
    
    def get_current_key_description(self):
        """Get a description of the current key for logging."""
        if self.current_key_index == -1:
            return "primary"
        else:
            return f"backup #{self.current_key_index + 1}"
    
    def is_quota_error(self, error):
        """Check if an error is a quota exceeded error."""
        if not GENAI_ERRORS_AVAILABLE:
            # Fallback to string matching if errors module not available
            # Consider common quota/billing phrases regardless of HTTP code presence
            error_str = str(error).lower()
            # Capture attributes from common websocket/client errors
            code = getattr(error, 'code', None) or getattr(error, 'status', None)
            reason = getattr(error, 'reason', None)
            if isinstance(reason, bytes):
                try:
                    reason = reason.decode('utf-8', errors='ignore')
                except Exception:
                    reason = str(reason)
            reason_str = (str(reason).lower() if reason is not None else '')

            quota_keywords = [
                'quota', 'exceeded your current quota', 'exceeded quota', 'billing', 'plan',
                'rate limit', 'resource_exhausted', 'resource exhausted', 'over quota'
            ]

            # If websocket closed with 1011 and reason mentions quota/billing
            if (code == 1011) and any(k in reason_str or k in error_str for k in quota_keywords):
                return True

            # If message clearly states quota/billing regardless of code
            if any(k in error_str for k in quota_keywords):
                return True

            # As a final heuristic, allow classic 429 + keywords
            return (
                '429' in error_str and any(k in error_str for k in quota_keywords)
            )
        
        # Use proper error detection with google.genai.errors
        if isinstance(error, errors.ClientError):
            # Check for 429 status code (Too Many Requests)
            if error.code == 429:
                return True
            
            # Check for RESOURCE_EXHAUSTED status
            if error.status and 'RESOURCE_EXHAUSTED' in str(error.status).upper():
                return True
                
            # Check message content for quota-related terms
            if error.message:
                message_lower = error.message.lower()
                return any(keyword in message_lower for keyword in [
                    'quota', 'exceeded', 'rate limit', 'resource_exhausted', 'billing', 'plan'
                ])
        
        # Handle ExceptionGroup that might contain quota errors
        if hasattr(error, 'exceptions'):
            for exc in error.exceptions:
                if self.is_quota_error(exc):
                    return True
        
        # Fallback to string matching without requiring 429
        error_str = str(error).lower()
        quota_keywords = [
            'quota', 'exceeded your current quota', 'exceeded quota', 'billing', 'plan',
            'rate limit', 'resource_exhausted', 'resource exhausted', 'over quota'
        ]
        return any(k in error_str for k in quota_keywords)
    
    def can_switch_key(self, ignore_cooldown: bool = False):
        """Check if we can switch to another API key.

        Args:
            ignore_cooldown: If True, ignore the switch cooldown (useful for hard quota errors).
        """
        # Check cooldown unless explicitly ignored
        if not ignore_cooldown and (time.time() - self.last_switch_time < self.switch_cooldown):
            return False
        
        # Check if we have more keys available
        total_keys = 1 + len(self.backup_keys)  # primary + backups
        available_keys = total_keys - len(self.failed_keys)
        
        return available_keys > 1  # Need at least one more key to switch to
    
    def switch_to_next_key(self, ignore_cooldown: bool = False):
        """Switch to the next available API key.

        Args:
            ignore_cooldown: If True, bypass cooldown checks to rotate keys immediately.
        """
        if not self.can_switch_key(ignore_cooldown=ignore_cooldown):
            logger.warning("Cannot switch API key: cooldown active or no more keys available")
            return False
        
        # Mark current key as failed
        current_key = self.get_current_key()
        self.failed_keys.add(current_key)
        
        # Find next available key
        next_key_found = False
        
        # Try backup keys first
        for i, backup_key in enumerate(self.backup_keys):
            if backup_key not in self.failed_keys:
                self.current_key_index = i
                next_key_found = True
                break
        
        # If no backup keys available, try primary (if not failed)
        if not next_key_found and self.primary_key not in self.failed_keys:
            self.current_key_index = -1
            next_key_found = True
        
        if next_key_found:
            # Update last switch time unless we're bypassing cooldown (still record moment for telemetry)
            self.last_switch_time = time.time()
            new_key_desc = self.get_current_key_description()
            logger.info(f"Switched to {new_key_desc} API key due to quota exceeded")
            return True
        else:
            logger.error("No more available API keys to switch to!")
            return False
    
    def reset_failed_keys(self):
        """Reset the failed keys set (useful for periodic retry)."""
        self.failed_keys.clear()
        logger.info("Reset failed API keys - all keys available for retry")
    
    def create_client(self):
        """Create a new client with the current API key."""
        current_key = self.get_current_key()
        api_config = self.config.get('api', {})
        
        if not current_key or not current_key.strip():
            raise ValueError("Current API key is empty!")
        
        client = genai.Client(
            http_options={"api_version": api_config.get('version', 'v1beta')},
            api_key=current_key,
        )
        
        logger.info(f"Created new client with {self.get_current_key_description()} API key")
        return client

# Global variables that will be set by setup_v2_globals
FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

pya = pyaudio.PyAudio()

# Global client instance - will be shared with main.py
client = None
CONFIG = None


def setup_v2_globals(main_config, main_client):
    """Setup V2 globals using the main configuration and client."""
    global client, CONFIG, FORMAT, CHANNELS, SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE, CHUNK_SIZE
    
    # Use the existing client from main.py
    client = main_client
    
    # Audio configuration from main config
    audio_config = main_config.get('audio', {})
    FORMAT = getattr(pyaudio, f"paInt{audio_config.get('format', 16)}")
    CHANNELS = audio_config.get('channels', 1)
    SEND_SAMPLE_RATE = audio_config.get('send_sample_rate', 16000)
    RECEIVE_SAMPLE_RATE = audio_config.get('receive_sample_rate', 24000)
    CHUNK_SIZE = audio_config.get('chunk_size', 1024)

    # Tools configuration - use same tools as main
    tools_config = main_config.get('tools', {})
    tools = []

    # Add Google Search tool if enabled
    if tools_config.get('google_search', True):
        tools.append(types.Tool(google_search=types.GoogleSearch()))

    # Add memory system tools if enabled (default: True)
    if tools_config.get('memory_system', True):
        memory_system_tools = memory_tools.get_all_tools(main_config)
        for tool in memory_system_tools:
            tools.append(types.Tool(function_declarations=tool['function_declarations']))

    # Add custom function declarations from config
    function_declarations = tools_config.get('function_declarations', [])
    if function_declarations:
        tools.append(types.Tool(function_declarations=function_declarations))

    # V2 specific configuration using main config as base
    live_config = main_config.get('live_connect', {})
    speech_config = live_config.get('speech', {})
    voice_config = speech_config.get('voice', {})
    context_window_config = live_config.get('context_window', {})
    
    # Get system instruction from main config
    # Import functions directly to avoid circular import
    def load_prompts_v2(prompts_path="prompts.json"):
        """Load prompts from JSON file."""
        try:
            with open(prompts_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except FileNotFoundError:
            logger.warning(f"Prompts file {prompts_path} not found. Using default prompt.")
            return {
                "normal": {
                    "name": "Default Assistant",
                    "description": "Gabriel, a helpful assistant",
                    "prompt": "You are Gabriel, a helpful assistant."
                }
            }
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing prompts file: {e}")
            return {
                "normal": {
                    "name": "Default Assistant", 
                    "description": "Gabriel, a helpful assistant",
                    "prompt": "You are Gabriel, a helpful assistant."
                }
            }

    def get_system_instruction_v2(config, prompts):
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

    prompts = load_prompts_v2()
    system_instruction = get_system_instruction_v2(main_config, prompts)

    # Live connect configuration
    live_config = main_config.get('live_connect', {})
    speech_config = live_config.get('speech', {})
    voice_config = speech_config.get('voice', {})
    context_window_config = live_config.get('context_window', {})
    session_resumption_config = live_config.get('session_resumption', {})

    # Setup session resumption configuration for V2
    # Note: Session resumption may not be supported on all V2 models
    session_resumption = None
    if session_resumption_config.get('enabled', False):
        try:
            session_resumption = types.SessionResumptionConfig(
                # The handle parameter is set dynamically by V2SessionManager
                # when resuming a session. For new sessions, this will be None.
                handle=None
            )
            logger.info("V2 session resumption enabled")
        except Exception as e:
            logger.warning(f"V2 session resumption not available: {e}")
            session_resumption = None

    CONFIG = types.LiveConnectConfig(
        response_modalities=live_config.get('response_modalities', ["AUDIO"]),
        media_resolution=live_config.get('media_resolution', "MEDIA_RESOLUTION_MEDIUM"),
        thinking_config=types.ThinkingConfig(
            thinking_budget=live_config.get('thinking_budget', 1024),
            include_thoughts=True,
        ),
        speech_config=types.SpeechConfig(
            # Note: V2 mode doesn't support language_code, so we omit it
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_config.get('name', 'Puck')
                )
            )
        ),
        realtime_input_config=types.RealtimeInputConfig(
            turn_coverage="TURN_INCLUDES_ALL_INPUT"
        ),
        context_window_compression=types.ContextWindowCompressionConfig(
            trigger_tokens=context_window_config.get('trigger_tokens', 25600),
            sliding_window=types.SlidingWindow(
                target_tokens=context_window_config.get('sliding_window_target_tokens', 12800)
            ),
        ),
        session_resumption=session_resumption,
        tools=tools,
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=system_instruction)],
            role="user"
        ),
        # Enable audio transcription for VRChat text output when using AUDIO mode
        output_audio_transcription=live_config.get('output_audio_transcription', {}),
    )

    logger.info("V2 mode globals initialized successfully")


class V2SessionManager:
    """Manages session resumption and automatic reconnection for V2 mode."""
    
    def __init__(self, config, api_key_manager):
        self.config = config
        self.api_key_manager = api_key_manager
        self.session_config = config.get('session_management', {})
        self.auto_reconnect_config = self.session_config.get('auto_reconnect', {})
        self.monitoring_config = self.session_config.get('monitoring', {})
        
        # Session state
        self.session_handle = None
        self.last_consumed_message_index = None
        self.is_reconnecting = False
        self.connection_start_time = None
        self.last_heartbeat = None
        
        # Reconnection settings
        self.max_retries = self.auto_reconnect_config.get('max_retries', 5)
        self.initial_delay = self.auto_reconnect_config.get('initial_delay', 1.0)
        self.max_delay = self.auto_reconnect_config.get('max_delay', 30.0)
        self.exponential_base = self.auto_reconnect_config.get('exponential_base', 2.0)
        self.jitter = self.auto_reconnect_config.get('jitter', 0.1)
        
        # API key switching state
        self.quota_retry_count = 0
        self.max_quota_retries = 3  # Max retries per API key
        
        self.persistence = None
        self.persistence_task = None
        if SESSION_PERSISTENCE_AVAILABLE:
            save_interval = self.session_config.get('save_interval', 30)
            self.persistence = get_persistence_manager(save_interval)
            logger.info(f"V2 session persistence enabled with {save_interval}s interval")
        
        logger.info("V2SessionManager initialized with API key management")
    
    def get_session_config(self):
        """Get the current session configuration with resumption handle if available."""
        global CONFIG
        
        # Check if session resumption is available and enabled for V2
        if hasattr(CONFIG, 'session_resumption') and CONFIG.session_resumption is not None:
            if self.session_handle:
                try:
                    # Update the session resumption config with our handle
                    CONFIG.session_resumption.handle = self.session_handle
                    logger.info(f"V2 using session resumption handle: {self.session_handle[:20]}...")
                except Exception as e:
                    logger.warning(f"V2 failed to set session resumption handle: {e}")
            else:
                # Guard against stale handle lingering in CONFIG when we intentionally cleared it
                try:
                    if getattr(CONFIG.session_resumption, 'handle', None):
                        CONFIG.session_resumption.handle = None
                        logger.info("V2 cleared stale session resumption handle in config (fresh session)")
                except Exception as e:
                    logger.warning(f"V2 failed to clear stale session handle: {e}")
        
        return CONFIG
    
    def handle_session_resumption_update(self, update):
        """Handle incoming session resumption update."""
        if hasattr(update, 'new_handle') and update.new_handle:
            self.session_handle = update.new_handle
            logger.info(f"V2 updated session handle: {self.session_handle[:20]}...")
        
        if hasattr(update, 'resumable'):
            logger.info(f"V2 session resumable: {update.resumable}")
        
        if hasattr(update, 'last_consumed_client_message_index'):
            self.last_consumed_message_index = update.last_consumed_client_message_index
            logger.info(f"V2 last consumed message index: {self.last_consumed_message_index}")
    
    def handle_go_away(self, go_away):
        """Handle GoAway message indicating impending disconnection."""
        logger.warning("V2 GoAway message received from server")
        
        if hasattr(go_away, 'time_left'):
            time_left = go_away.time_left
            logger.warning(f"V2 server will disconnect in: {time_left}")
            
            # Parse the duration string (format: "XXXs" for seconds)
            if isinstance(time_left, str) and time_left.endswith('s'):
                try:
                    seconds = float(time_left[:-1])
                    if seconds < 30:  # If less than 30 seconds, prepare for reconnection
                        logger.info("V2 preparing for imminent disconnection...")
                        self.is_reconnecting = True
                except ValueError:
                    logger.warning(f"V2 could not parse time_left: {time_left}")
        
        # Always set reconnecting flag when GoAway is received
        self.is_reconnecting = True
        logger.info("V2 GoAway received - marking session for reconnection")
        
        # Log additional GoAway details if available
        if hasattr(go_away, 'reason'):
            logger.warning(f"V2 GoAway reason: {go_away.reason}")
        if hasattr(go_away, 'debug_description'):
            logger.warning(f"V2 GoAway debug info: {go_away.debug_description}")
    
    async def handle_quota_exceeded(self, error):
        """Handle quota exceeded errors by switching API keys."""
        logger.warning(f"V2 quota exceeded detected: {error}")
        
        # On clear quota/billing errors we should bypass cooldown to recover faster
        if self.api_key_manager.can_switch_key(ignore_cooldown=True):
            # Try to switch to next available API key
            if self.api_key_manager.switch_to_next_key(ignore_cooldown=True):
                # Reset quota retry count for new key
                self.quota_retry_count = 0
                # IMPORTANT: Session handles cannot cross API keys; clear any existing handle
                self.clear_session_resumption(reason="API key switched due to quota exceeded")
                logger.info("V2 switched to backup API key, will retry connection")
                return True
            else:
                logger.error("V2 no more backup API keys available")
                return False
        else:
            self.quota_retry_count += 1
            logger.warning(f"V2 quota retry {self.quota_retry_count}/{self.max_quota_retries} for current API key")
            
            if self.quota_retry_count >= self.max_quota_retries:
                # Reset failed keys and try switching again as last resort
                logger.info("V2 max quota retries reached, resetting failed keys and trying again")
                self.api_key_manager.reset_failed_keys()
                if self.api_key_manager.switch_to_next_key():
                    self.quota_retry_count = 0
                    return True
                else:
                    logger.error("V2 all API keys exhausted, cannot continue")
                    return False
            
            # Wait a bit before retrying with same key
            await asyncio.sleep(min(5 * self.quota_retry_count, 30))
            return True
    
    async def calculate_reconnect_delay(self, retry_count):
        """Calculate delay before reconnection attempt with exponential backoff and jitter."""
        if not self.auto_reconnect_config.get('enabled', True):
            return None
        
        if retry_count >= self.max_retries:
            logger.error(f"V2 max retries ({self.max_retries}) exceeded. Giving up.")
            return None
        
        # Exponential backoff
        delay = min(self.initial_delay * (self.exponential_base ** retry_count), self.max_delay)
        
        # Add jitter
        jitter_amount = delay * self.jitter * (random.random() * 2 - 1)  # +/- jitter%
        delay += jitter_amount
        
        delay = max(0, delay)  # Ensure non-negative
        logger.info(f"V2 reconnection attempt {retry_count + 1}/{self.max_retries} in {delay:.2f}s")
        
        return delay
    
    def reset_connection_state(self):
        """Reset connection state for new session."""
        self.connection_start_time = time.time()
        self.last_heartbeat = self.connection_start_time
        self.is_reconnecting = False
        logger.info("V2 connection state reset")
    
    def should_attempt_reconnect(self, error):
        """Determine if we should attempt to reconnect based on the error."""
        if not self.auto_reconnect_config.get('enabled', True):
            return False
        
        # Handle policy violation 1008: session not found -> clear session handle and reconnect
        if self._is_policy_session_not_found(error):
            self.clear_session_resumption(reason="Server reported session not found (1008 policy violation)")
            logger.info("V2 treating 1008 session-not-found as reconnectable after clearing session handle")
            return True

        # Check for quota exceeded errors first
        if self.api_key_manager.is_quota_error(error):
            logger.info("V2 quota exceeded error detected - will attempt API key switch")
            return True
        
        # If we're already marked for reconnection (e.g., due to GoAway), attempt it
        if self.is_reconnecting:
            logger.info("V2 reconnection flagged due to GoAway message")
            return True
        
        # Handle ExceptionGroup (from TaskGroup) by checking its exceptions
        if hasattr(error, 'exceptions'):  # ExceptionGroup
            for exc in error.exceptions:
                # Handle policy violation 1008 inside ExceptionGroup
                if self._is_policy_session_not_found(exc):
                    self.clear_session_resumption(reason="Server reported session not found (1008 policy violation)")
                    logger.info("V2 treating 1008 session-not-found (inner) as reconnectable after clearing session handle")
                    return True
                # Check for V1ModeSwitchRequested in the exception group
                if isinstance(exc, V1ModeSwitchRequested):
                    logger.info("V1ModeSwitchRequested found in ExceptionGroup - not a reconnection case")
                    return False
                # Check for quota errors in exception group
                if self.api_key_manager.is_quota_error(exc):
                    logger.info("V2 quota exceeded error found in ExceptionGroup")
                    return True
                if self._is_reconnectable_error(exc):
                    # Log specific details about WebSocket errors
                    if '1011' in str(exc) or 'deadline expired' in str(exc).lower():
                        logger.warning(f"V2 detected WebSocket deadline/timeout error: {exc}")
                    return True
            return False
        
        # Check for standalone V1ModeSwitchRequested
        if isinstance(error, V1ModeSwitchRequested):
            logger.info("V1ModeSwitchRequested - not a reconnection case")
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
            logger.info(f"V2 local screen capture error detected (non-network): {type(error).__name__}: {error}")
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
                logger.info(f"V2 WebSocket connection closed: {error}")
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
                '1011', 'service is currently unavailable', 'server error'
            ]):
                logger.info(f"V2 reconnectable error detected: {error}")
                return True
        
        logger.warning(f"V2 error not configured for reconnection: {type(error).__name__}: {error}")
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
            logger.warning(f"V2 detected 1008 policy violation / session not found: {msg}")
            return True
        return False

    def clear_session_resumption(self, reason: str | None = None):
        """Clear session resumption state so next session starts fresh.

        This is required when changing API keys because server session handles
        are not valid across different credentials.
        """
        global CONFIG
        if reason:
            logger.info(f"V2 clearing session resumption state: {reason}")
        else:
            logger.info("V2 clearing session resumption state")
        # Clear local state
        self.session_handle = None
        self.last_consumed_message_index = None
        # Best-effort clear on global CONFIG, if available
        try:
            if hasattr(CONFIG, 'session_resumption') and CONFIG.session_resumption is not None:
                CONFIG.session_resumption.handle = None
        except Exception as e:
            logger.warning(f"V2 failed to clear session resumption handle on CONFIG: {e}")
    
    def start_periodic_save(self, mode: str = "v2"):
        if self.persistence and not self.persistence_task:
            handle_getter = lambda: self.session_handle
            metadata_getter = lambda: {
                "last_consumed_index": self.last_consumed_message_index,
                "connection_start_time": self.connection_start_time
            }
            self.persistence_task = asyncio.create_task(
                self.persistence.start_periodic_save(handle_getter, mode, metadata_getter)
            )
            logger.info(f"V2 started periodic session save for {mode} mode")
    
    def stop_periodic_save(self, mode: str = "v2"):
        if self.persistence:
            self.persistence.stop_periodic_save()
            if self.session_handle:
                metadata = {
                    "last_consumed_index": self.last_consumed_message_index,
                    "connection_start_time": self.connection_start_time
                }
                self.persistence.save_session_handle(self.session_handle, mode, metadata)
                logger.info(f"V2 saved final session handle on stop for {mode} mode")
        if self.persistence_task:
            self.persistence_task.cancel()
            self.persistence_task = None


class V2AudioLoop:
    def __init__(self, video_mode=V2_DEFAULT_MODE, session_handle=None, config=None, api_key_manager=None):
        self.video_mode = video_mode
        self.session_handle = session_handle  # For session transfer
        self.config = config or {}  # Store config reference
        self.api_key_manager = api_key_manager
        
        # Get video configuration
        self.video_config = self.config.get('video', {})
        
        # Get queue configuration
        queue_config = self.config.get('queues', {})
        self.output_queue_maxsize = queue_config.get('output_queue_maxsize', 5)
        
        # Get debug configuration
        self.debug_config = self.config.get('debug', {})
        
        # Initialize session manager for V2
        self.session_manager = V2SessionManager(self.config, self.api_key_manager)
        # Set session handle if provided (for session transfer from V1)
        if session_handle:
            self.session_manager.session_handle = session_handle
            logger.info(f"V2 initialized with transferred session handle: {session_handle[:20]}...")
        
        self.audio_in_queue = None
        self.out_queue = None
        self.session = None
        self.receive_audio_task = None
        self.play_audio_task = None
        
        # V1 mode switching
        self.switch_to_v1_requested = False
        
        # Current client reference (will be updated on API key switches)
        self.current_client = None
        # Thinking streaming state
        self._thinking_task = None
        self._thinking_active = False
        self._current_thinking_text = ""
        self._current_text_response = ""
        self._thinking_marquee_started = False
        self._final_marquee_started = False

    async def _stop_final_marquee_after_delay(self, client, key: str, delay: float):
        try:
            await asyncio.sleep(delay)
            try:
                client.stop_ui_marquee(key + ":final")
            except Exception:
                pass
            self._final_marquee_started = False
            # Clear the accumulated final response and return to idle UI
            try:
                self._current_text_response = ""
            except Exception:
                pass
            try:
                client.maybe_send_idle_ui()
            except Exception:
                pass
        except asyncio.CancelledError:
            return
        except Exception:
            return

    def _get_frame(self, cap):
        """Get frame from camera with V2 optimizations."""
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
        """Camera frame capture for V2 mode."""
        # Get camera configuration
        camera_config = self.video_config.get('camera', {})
        device_index = camera_config.get('device_index', 0)
        frame_delay = camera_config.get('frame_delay', 1.0)
        
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, device_index
        )  # device_index represents the camera to use

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(frame_delay)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        """Get screen capture for V2 mode."""
        # Get screen and image configuration
        screen_config = self.video_config.get('screen', {})
        image_config = self.video_config.get('image', {})
        monitor_index = screen_config.get('monitor_index', 0)
        image_format = image_config.get('format', 'jpeg')
        mime_type = image_config.get('mime_type', 'image/jpeg')
        
        sct = None
        try:
            sct = mss.mss()
            monitors = getattr(sct, 'monitors', [])
            if not monitors:
                logger.debug("V2 no monitors reported by mss; skipping frame")
                return None
            safe_index = monitor_index
            if safe_index < 0 or safe_index >= len(monitors):
                logger.warning(f"V2 configured monitor_index {monitor_index} out of range (0..{len(monitors)-1}); using 0")
                safe_index = 0
            monitor = monitors[safe_index]

            try:
                i = sct.grab(monitor)
            except Exception as grab_err:
                logger.debug(f"V2 screen grab failed: {grab_err}")
                return None

            try:
                image_bytes = mss.tools.to_png(i.rgb, i.size)
                img = PIL.Image.open(io.BytesIO(image_bytes))

                image_io = io.BytesIO()
                img.save(image_io, format=image_format)
                image_io.seek(0)

                image_bytes = image_io.read()
                return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}
            except Exception as enc_err:
                logger.debug(f"V2 screen encode failed: {enc_err}")
                return None
        finally:
            try:
                if sct is not None:
                    sct.close()
            except Exception:
                pass

    async def get_screen(self):
        """Screen capture loop for V2 mode."""
        # Get screen configuration
        screen_config = self.video_config.get('screen', {})
        frame_delay = screen_config.get('frame_delay', 1.0)

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                # Transient capture failure; short backoff and continue
                await asyncio.sleep(min(frame_delay, 0.2))
                continue

            await asyncio.sleep(frame_delay)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        """Send real-time data to the session."""
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        """Audio input capture for V2 mode."""
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
            
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            # If yap mode is enabled AND AI is speaking, drop mic input to prevent interruptions
            try:
                import tools as _tools
                if (
                    getattr(_tools, 'is_yap_mode_enabled', lambda: False)()
                    and getattr(_tools, 'is_ai_speaking', lambda: False)()
                ):
                    continue
            except Exception:
                pass
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        """Background task to read from the websocket and write pcm chunks to the output queue."""
        current_text_response = ""  # Accumulate text for VRChat
        # Thinking stream control variables are stored on the instance
        
        while True:
            turn = self.session.receive()
            async for response in turn:
                # Handle audio data
                if data := response.data:
                    try:
                        import sfx as _sfx
                        if hasattr(_sfx, 'sfx_manager') and _sfx.sfx_manager.is_music_playing():
                            pass
                        else:
                            self.audio_in_queue.put_nowait(data)
                            if MYINSTANTS_AVAILABLE and myinstants_client:
                                myinstants_client.notify_ai_audio_received()
                            osc.notify_ai_speech_start()
                            try:
                                import tools as _tools
                                if hasattr(_tools, 'set_ai_speaking'):
                                    _tools.set_ai_speaking(True)
                            except Exception:
                                pass
                    except Exception:
                        self.audio_in_queue.put_nowait(data)
                        if MYINSTANTS_AVAILABLE and myinstants_client:
                            myinstants_client.notify_ai_audio_received()
                        osc.notify_ai_speech_start()
                        try:
                            import tools as _tools
                            if hasattr(_tools, 'set_ai_speaking'):
                                _tools.set_ai_speaking(True)
                        except Exception:
                            pass
                    continue
                
                # Handle text responses
                if text := response.text:
                    # Don't log streaming text to console to avoid spam
                    # Accumulate text for VRChat in instance buffer
                    self._current_text_response += text
                
                # Handle audio transcriptions (for AUDIO mode with transcription enabled)
                if hasattr(response, 'server_content') and response.server_content:
                    if hasattr(response.server_content, 'output_transcription') and response.server_content.output_transcription:
                        transcription_text = response.server_content.output_transcription.text
                        if transcription_text:
                            # Don't log transcription streaming to console
                            # Accumulate transcription text for VRChat in instance buffer
                            self._current_text_response += transcription_text

                    # Handle model 'thought' parts (thinking). These parts are marked in the
                    # LiveServerMessage model_turn.parts with part.thought == True. When present,
                    # stream them to WebUI and VRChat periodically until the turn is complete.
                    try:
                        mt = getattr(response.server_content, 'model_turn', None)
                        parts = getattr(mt, 'parts', None)
                        if parts:
                            for part in parts:
                                if getattr(part, 'thought', False):
                                    text = getattr(part, 'text', '') or ''
                                    if text:
                                        # Update current thinking buffer
                                        self._current_thinking_text += text
                                        # Broadcast thinking chunk to WebUI if available
                                        try:
                                            if CHAT_API_AVAILABLE and chat_api:
                                                chat_api.broadcast_gabriel_response(text, "thinking_chunk")
                                        except Exception:
                                            pass
                                        # Start background task to stream thinking to VRChat if not running
                                        if not self._thinking_active:
                                            self._thinking_active = True
                                            client = None
                                            try:
                                                client = osc.get_osc_client()
                                            except Exception:
                                                client = None
                                            key = f"v2:{id(self)}"
                                            try:
                                                if client and getattr(client, 'ui_two_part_enabled', False):
                                                    # Pass both thinking and current accumulated response so both parts are visible
                                                    client.start_ui_marquee(key + ":thinking", thinking=lambda: self._current_thinking_text, final=lambda: self._current_text_response, interval_seconds=getattr(client, 'marquee_interval_seconds', 1.0), window_chars=getattr(client, 'marquee_window_chars', 40))
                                                    self._thinking_marquee_started = True
                                                else:
                                                    self._thinking_task = asyncio.create_task(self._stream_thinking())
                                                    self._thinking_marquee_started = False
                                            except Exception:
                                                self._thinking_task = None
                                                self._thinking_marquee_started = False
                    except Exception as thinking_err:
                        logger.debug(f"V2 thinking handling error: {thinking_err}")
                
                # Handle tool calls
                if hasattr(response, 'tool_call') and response.tool_call:
                    function_responses = []
                    for fc in response.tool_call.function_calls:
                        logger.info(f"V2 Function call: {fc.name} with args: {fc.args}")
                        try:
                            function_response = await memory_tools.handle_function_call(fc)
                            function_responses.append(function_response)
                            logger.info(f"V2 Function response: {function_response.response}")
                            
                            # Check if this is a V1 mode switch request
                            if (fc.name == "switch_to_v1_mode" and 
                                function_response.response.get("success") and 
                                function_response.response.get("action") == "switch_to_v1"):
                                logger.info("V1 mode switch requested from V2, setting flag")
                                # Set a flag that the V2 loop can check
                                self.switch_to_v1_requested = True
                                
                        except Exception as e:
                            logger.error(f"Error handling V2 function call {fc.name}: {e}")
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

            # Turn complete - send accumulated text to VRChat and notify services
            if self._current_text_response.strip():
                final_text = self._current_text_response.strip()
                client = osc.get_osc_client()
                key = f"v2:{id(self)}"
                if client and getattr(client, 'ui_two_part_enabled', False):
                    try:
                        # Stop thinking marquee if it was running
                        if getattr(self, '_thinking_marquee_started', False):
                            try:
                                client.stop_ui_marquee(key + ":thinking")
                            except Exception:
                                pass
                            self._thinking_marquee_started = False

                        # Send an immediate two-part UI showing final text
                        try:
                            await client.send_two_part_ui('', final_text)
                        except Exception:
                            await osc.send_to_vrchat(final_text)

                        # Start a final marquee so the Response: line remains visible and can scroll
                        try:
                            client.start_ui_marquee(key + ":final", thinking=lambda: "", final=lambda: self._current_text_response, interval_seconds=getattr(client, 'marquee_interval_seconds', 1.0), window_chars=getattr(client, 'marquee_window_chars', 40))
                            self._final_marquee_started = True
                            # Schedule stopping the final marquee and returning to idle after auto_clear_delay
                            delay = getattr(client, 'auto_clear_delay', 0) or 10.0
                            try:
                                asyncio.create_task(self._stop_final_marquee_after_delay(client, key, delay))
                            except Exception:
                                pass
                        except Exception:
                            pass
                    except Exception:
                        await osc.send_to_vrchat(final_text)
                else:
                    await osc.send_to_vrchat(final_text)
            # Turn complete - stop any thinking stream that might be running
            if getattr(self, '_thinking_active', False):
                try:
                    self._thinking_active = False
                    if getattr(self, '_thinking_marquee_started', False):
                        try:
                            client = osc.get_osc_client()
                            client.stop_ui_marquee(f"v2:{id(self)}:thinking")
                        except Exception:
                            pass
                        self._thinking_marquee_started = False
                    if self._thinking_task and not self._thinking_task.done():
                        self._thinking_task.cancel()
                        try:
                            await self._thinking_task
                        except asyncio.CancelledError:
                            pass
                except Exception as stop_err:
                    logger.debug(f"Error stopping thinking task: {stop_err}")
                finally:
                    self._thinking_task = None
                    self._current_thinking_text = ""
                
            # Turn complete - notify MyInstants that speech has ended
            if MYINSTANTS_AVAILABLE and myinstants_client:
                myinstants_client.notify_ai_speech_ended()
            
            # Notify OSC client that Gabriel's speech has ended
            osc.notify_ai_speech_end()
            try:
                import tools as _tools
                if hasattr(_tools, 'set_ai_speaking'):
                    _tools.set_ai_speaking(False)
            except Exception:
                pass
            
            if hasattr(self, 'switch_to_v1_requested') and self.switch_to_v1_requested:
                logger.info("V1 mode switch detected in V2, raising signal")
                # Clear the flag first
                self.switch_to_v1_requested = False
                raise V1ModeSwitchRequested("V1 mode switch requested by AI")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        """Audio output playback for V2 mode."""
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def _stream_thinking(self):
        """Background task to stream current thinking text to VRChat every 0.5s.

        Keeps sending the most recent thinking buffer until `_thinking_active` is cleared.
        """
        try:
            while getattr(self, '_thinking_active', False):
                try:
                    text = getattr(self, '_current_thinking_text', '')
                    if text and text.strip():
                        thinking_msg = f"[Thinking]:\n{str(text).strip()}"
                        await osc.send_to_vrchat(thinking_msg)
                except Exception as send_err:
                    logger.debug(f"V2 thinking send error: {send_err}")
                try:
                    await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Thinking stream task error: {e}")

    async def run_session(self):
        """Run a single V2 session with the configured setup."""
        session_config = self.session_manager.get_session_config()
        session_start_time = time.time()
        
        # Ensure we have a current client
        if not self.current_client:
            self.current_client = self.api_key_manager.create_client()
        
        try:
            async with (
                self.current_client.aio.live.connect(model=V2_MODEL, config=session_config) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                self.session_manager.reset_connection_state()
                logger.info("V2 session connected successfully")
                
                self.session_manager.start_periodic_save(mode="v2")

                # Register session with Chat API for WebUI functionality
                if CHAT_API_AVAILABLE and chat_api:
                    try:
                        chat_api.register_session(session)
                        logger.info("V2 session registered with Chat API")
                    except Exception as api_reg_error:
                        logger.warning(f"Failed to register V2 session with Chat API: {api_reg_error}")

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

                # Send system instruction to notify Gabriel about V2 mode
                try:
                    logger.info("Sending V2 mode system instruction to Gabriel")
                    await session.send_client_content(
                        turns={"role": "user", "parts": [{"text": "SYSTEM INSTRUCTION: You are now in V2 mode"}]},
                        turn_complete=True
                    )
                except Exception as system_msg_error:
                    logger.warning(f"Failed to send V2 mode system instruction: {system_msg_error}")

                # Keep the session running indefinitely until interrupted
                while True:
                    await asyncio.sleep(1)
                    
        except KeyboardInterrupt:
            session_duration = time.time() - session_start_time
            logger.info(f"V2 KeyboardInterrupt received after {session_duration:.2f}s, ending session")
            raise asyncio.CancelledError("User requested exit")
            
        except Exception as e:
            # Check if this exception contains V1ModeSwitchRequested
            if hasattr(e, 'exceptions'):
                for exc in e.exceptions:
                    if isinstance(exc, V1ModeSwitchRequested):
                        session_duration = time.time() - session_start_time
                        logger.info(f"V1 mode switch found in ExceptionGroup after {session_duration:.2f}s")
                        raise V1ModeSwitchRequested("V1 mode switch requested by AI")
            
            # Check if this is a standalone V1ModeSwitchRequested
            if isinstance(e, V1ModeSwitchRequested):
                session_duration = time.time() - session_start_time
                logger.info(f"V1 mode switch requested after {session_duration:.2f}s, ending V2 session")
                raise e
            
            # Re-raise other exceptions
            raise e
            
        finally:
            session_duration = time.time() - session_start_time
            logger.info(f"V2 session ended after {session_duration:.2f} seconds")
            
            self.session_manager.stop_periodic_save(mode="v2")
            
            # Unregister session from Chat API
            if CHAT_API_AVAILABLE and chat_api:
                try:
                    chat_api.unregister_session()
                    logger.info("V2 session unregistered from Chat API")
                except Exception as api_unreg_error:
                    logger.warning(f"Failed to unregister V2 session from Chat API: {api_unreg_error}")

            # Ensure any final marquee is stopped on session end
            try:
                client = osc.get_osc_client()
                if client and getattr(self, '_final_marquee_started', False):
                    try:
                        client.stop_ui_marquee(f"v2:{id(self)}:final")
                    except Exception:
                        pass
                    self._final_marquee_started = False
            except Exception:
                pass

    async def run(self):
        """Main V2 run method with automatic reconnection support, V1 mode switching, and API key failover."""
        retry_count = 0
        consecutive_failures = 0
        last_successful_connection = None
        
        logger.info(f"Starting V2 session loop with auto-reconnect (max_retries: {self.session_manager.max_retries})")
        
        while True:
            connection_start = time.time()
            
            try:
                logger.info(f"Starting V2 session (attempt {retry_count + 1})")
                await self.run_session()
                
                # If we reach here, user requested exit
                logger.info("V2 session ended normally by user request")
                break
            
            except asyncio.CancelledError as e:
                if str(e) == "User requested exit":
                    logger.info("V2 user requested exit")
                    break
                else:
                    # This might be from a GoAway or connection issue
                    logger.warning(f"V2 session cancelled: {e}, checking if reconnection is needed")
                    if not self.session_manager.should_attempt_reconnect(e):
                        logger.warning("V2 CancelledError not marked reconnectable; proceeding with conservative reconnect")
                        try:
                            await asyncio.sleep(5)
                        except asyncio.CancelledError:
                            logger.info("V2 conservative delay interrupted by cancellation")
                            break
                        except Exception as sleep_e:
                            logger.warning(f"V2 exception during conservative delay: {sleep_e}")
                        # Continue to compute delay and retry
            
            except V1ModeSwitchRequested as e:
                logger.info("V1 mode switch requested from V2, returning switch signal")
                return "switch_to_v1"
            
            except Exception as e:
                is_connection_error = False
                
                if hasattr(e, 'exceptions'):
                    for exc in e.exceptions:
                        if WEBSOCKETS_AVAILABLE and isinstance(exc, (ConnectionClosedError, ConnectionClosedOK)):
                            is_connection_error = True
                            break
                        if isinstance(exc, (ConnectionError, asyncio.TimeoutError)):
                            is_connection_error = True
                            break
                        error_str = str(exc).lower()
                        if any(keyword in error_str for keyword in ['connection closed', '1011', 'websocket closed', 'internal error']):
                            is_connection_error = True
                            break
                
                if is_connection_error:
                    logger.info(f"V2 connection error (will reconnect): {e}")
                else:
                    logger.error(f"V2 session error: {e}")
                
                if hasattr(e, 'exceptions'):
                    if is_connection_error:
                        logger.info(f"V2 ExceptionGroup contains {len(e.exceptions)} exceptions:")
                        for i, exc in enumerate(e.exceptions):
                            logger.info(f"  V2 Exception {i+1}: {type(exc).__name__}: {exc}")
                    else:
                        logger.error(f"V2 ExceptionGroup contains {len(e.exceptions)} exceptions:")
                        for i, exc in enumerate(e.exceptions):
                            logger.error(f"  V2 Exception {i+1}: {type(exc).__name__}: {exc}")
                    
                    for exc in e.exceptions:
                        if isinstance(exc, V1ModeSwitchRequested):
                            logger.info("Found V1ModeSwitchRequested in ExceptionGroup, returning switch signal")
                            return "switch_to_v1"
                
                # Check for quota exceeded errors and handle API key switching
                if self.api_key_manager.is_quota_error(e):
                    logger.warning("V2 quota exceeded error detected")
                    
                    # Try to handle the quota error by switching API keys
                    if await self.session_manager.handle_quota_exceeded(e):
                        # Create new client with the switched API key
                        try:
                            self.current_client = self.api_key_manager.create_client()
                            logger.info("V2 successfully switched API key, will retry")
                            # Don't increment retry count for quota switches
                            await asyncio.sleep(2)  # Brief pause before retry
                            continue
                        except Exception as client_error:
                            logger.error(f"V2 failed to create client with new API key: {client_error}")
                            # Fall through to normal error handling
                    else:
                        logger.error("V2 failed to handle quota exceeded error")
                        # Proceed to conservative retry instead of exiting entirely
                        try:
                            await asyncio.sleep(5)
                        except asyncio.CancelledError:
                            logger.info("V2 conservative delay interrupted by cancellation")
                            break
                        except Exception as sleep_e:
                            logger.warning(f"V2 exception during conservative delay: {sleep_e}")
                
                # Check connection duration to see if it was a quick failure
                connection_duration = time.time() - connection_start
                if connection_duration < 10:  # Failed within 10 seconds
                    consecutive_failures += 1
                    logger.warning(f"V2 quick failure detected ({connection_duration:.2f}s), consecutive failures: {consecutive_failures}")
                else:
                    consecutive_failures = 0  # Reset on successful longer connection
                    # Also reset retry count for longer connections (successful sessions)
                    if connection_duration > 300:  # 5 minutes or more
                        retry_count = 0
                        logger.info(f"V2 long-lived session ({connection_duration:.2f}s), resetting retry count")
                
                if not self.session_manager.should_attempt_reconnect(e):
                    logger.warning("V2 error not configured for reconnection; proceeding with conservative reconnect")
                    # Add a small delay to avoid tight loop
                    try:
                        await asyncio.sleep(5)
                    except asyncio.CancelledError:
                        logger.info("V2 conservative delay interrupted by cancellation")
                        break
                    except Exception as sleep_e:
                        logger.warning(f"V2 exception during conservative delay: {sleep_e}")
                
                # If we have too many consecutive quick failures, wait longer
                if consecutive_failures >= 3:
                    extra_delay = min(consecutive_failures * 5, 60)  # Up to 60 seconds extra
                    logger.warning(f"V2 multiple consecutive failures, adding {extra_delay}s extra delay")
                    try:
                        await asyncio.sleep(extra_delay)
                    except asyncio.CancelledError:
                        logger.info("V2 extra delay sleep interrupted by cancellation")
                        break
                    except Exception as sleep_e:
                        logger.warning(f"V2 exception during extra delay sleep: {sleep_e}")
            
            finally:
                # Clean up audio stream if it exists
                if hasattr(self, 'audio_stream') and self.audio_stream:
                    try:
                        try:
                            if self.audio_stream.is_active():
                                self.audio_stream.stop_stream()
                        except Exception:
                            pass
                        self.audio_stream.close()
                    except Exception as e:
                        logger.warning(f"V2 error closing audio stream: {e}")
            
            # Calculate reconnection delay
            delay = await self.session_manager.calculate_reconnect_delay(retry_count)
            if delay is None:
                # Fallback to a safe default delay to prevent full exit
                delay = 5.0
                logger.warning("V2 no reconnection delay available (possibly disabled or max retries). Using default 5s and continuing.")
            
            retry_count += 1
            logger.info(f"V2 attempting reconnection {retry_count}/{self.session_manager.max_retries} in {delay:.2f} seconds...")
            
            # Sleep before reconnection attempt with proper exception handling
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                logger.info("V2 reconnection sleep interrupted by cancellation")
                break
            except Exception as e:
                logger.warning(f"V2 exception during reconnection sleep: {e}")
                # Continue with reconnection attempt despite sleep interruption
            
        logger.info("V2 main session loop ended")
        return None


async def run_v2_mode(main_config, main_client, video_mode=V2_DEFAULT_MODE, session_handle=None):
    """
    Run V2 mode with Gemini 2.5 Flash and backup API key support.
    
    Args:
        main_config: Configuration from main.py
        main_client: Client instance from main.py (used for compatibility, V2 creates its own)
        video_mode: Video mode to use
        session_handle: Optional session handle for continuity
    
    Returns:
        String indicating next action or None
    """
    try:
        # Setup V2 globals using main configuration
        setup_v2_globals(main_config, main_client)
        
        # Initialize API Key Manager for backup API key support
        api_key_manager = APIKeyManager(main_config)
        logger.info(f"V2 initialized with API key management: {api_key_manager.get_current_key_description()} key active")
        
        # Create and run V2 audio loop with API key manager
        v2_loop = V2AudioLoop(
            video_mode=video_mode, 
            session_handle=session_handle, 
            config=main_config,
            api_key_manager=api_key_manager
        )
        result = await v2_loop.run()
        
        return result
        
    except Exception as e:
        logger.error(f"Error running V2 mode: {e}")
        traceback.print_exception(type(e), e, e.__traceback__)
        return None


if __name__ == "__main__":
    """Standalone V2 mode for testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=V2_DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()
    
    # For standalone mode, create a simple client
    standalone_client = genai.Client(
        http_options={"api_version": "v1beta"},
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    
    # Simple config for standalone mode
    standalone_config = {
        'api': {
            'version': 'v1beta',
            'api_key': os.environ.get("GEMINI_API_KEY"),
            'backup_api_keys': []  # No backup keys for standalone mode
        },
        'audio': {
            'format': 16,
            'channels': 1,
            'send_sample_rate': 16000,
            'receive_sample_rate': 24000,
            'chunk_size': 1024
        },
        'live_connect': {
            'speech': {
                'voice': {
                    'name': 'Puck'
                }
            },
            'session_resumption': {
                'enabled': False  # Disable for standalone mode
            }
        },
        'tools': {
            'google_search': True,
            'memory_system': True
        },
        'session_management': {
            'auto_reconnect': {
                'enabled': True,
                'max_retries': 5,
                'initial_delay': 1.0,
                'max_delay': 30.0,
                'exponential_base': 2.0,
                'jitter': 0.1
            }
        },
        'video': {
            'camera': {
                'device_index': 0,
                'frame_delay': 1.0
            },
            'screen': {
                'monitor_index': 0,
                'frame_delay': 1.0
            },
            'image': {
                'thumbnail_size': [1024, 1024],
                'format': 'jpeg',
                'mime_type': 'image/jpeg'
            }
        },
        'queues': {
            'output_queue_maxsize': 5
        },
        'debug': {
            'exception_on_overflow': False
        }
    }
    
    logger.info("Running V2 mode in standalone mode")
    asyncio.run(run_v2_mode(standalone_config, standalone_client, video_mode=args.mode))
