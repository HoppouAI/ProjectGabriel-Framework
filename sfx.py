"""
Custom SFX (Sound Effects) module for playing audio files from the sfx folder.
Provides functionality to manage and play custom audio files stored locally.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from google.genai import types
import threading


try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False


# Optional MyInstants integration (fall back if a file isn't available locally)
try:
    from myinstants import myinstants_client
    MYINSTANTS_AVAILABLE = True
except Exception as e:
    myinstants_client = None
    MYINSTANTS_AVAILABLE = False
    # Do not spam logs at import time; leave info for callers

logger = logging.getLogger(__name__)

class SFXManager:
    """Manager for custom sound effects and audio files."""
    
    def __init__(self, sfx_base_path: str = "sfx"):
        self.sfx_base_path = Path(sfx_base_path)
        self.cache_file = self.sfx_base_path / "sfx_cache.json"
        self.audio_cache = {}
        self.current_player = None
        self.is_playing = False
        self.volume = 0.7
        self.current_playing_info = None
        self._monitor_thread = None
        self._monitor_stop_event = threading.Event()
        self.supported_formats = {
            '.mp3', '.wav', '.ogg', '.m4a', '.aac', '.flac', '.wma'
        }

        self.init_audio_system()
        self.load_cache()
        self.scan_audio_files()
    
    def _normalize(self, s: str) -> str:
        t = s.lower().replace('\\', '/').replace('_', ' ')
        out = []
        for ch in t:
            if ch.isalnum() or ch.isspace() or ch in {'/', '.'}:
                out.append(ch)
        return ' '.join(''.join(out).split())
    
    def _ensure_supported_ext(self, name: str) -> Optional[str]:
        p = Path(name)
        if p.suffix:
            return str(p)
        for ext in self.supported_formats:
            candidate = str(p) + ext
            if any(candidate.lower() == info['relative_path'].lower() or candidate.lower() == Path(info['path']).name.lower() for info in self.audio_cache.values()):
                return candidate
        return None
    
    def init_audio_system(self):
        """Initialize pygame audio system."""
        self.audio_system = None
        
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self.audio_system = "pygame"
                logger.info("Initialized pygame audio system")
                try:
                    self._start_monitor_thread()
                except Exception:
                    pass
                return
            except Exception as e:
                logger.warning(f"Failed to initialize pygame: {e}")
        
        logger.error("Pygame audio system not available!")
        self.audio_system = None

    def _monitor_loop(self):
        while not self._monitor_stop_event.is_set():
            try:
                if self.audio_system == "pygame":
                    busy = False
                    try:
                        busy = pygame.mixer.music.get_busy()
                    except Exception:
                        busy = False
                    if self.is_playing:
                        info = self.current_playing_info or {}
                        if str(info.get('category', '')).lower() == 'music':
                            if not busy:
                                self.is_playing = False
                                self.current_playing_info = None
                                logger.info('Music finished playback, cleared playing state')
                else:
                    pass
            except Exception as e:
                logger.debug(f'Monitor loop error: {e}')
            finally:
                time.sleep(0.5)

    def _start_monitor_thread(self):
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        self._monitor_stop_event.clear()
        t = threading.Thread(target=self._monitor_loop, daemon=True)
        t.start()
        self._monitor_thread = t

    def _stop_monitor_thread(self):
        try:
            self._monitor_stop_event.set()
            if self._monitor_thread:
                self._monitor_thread.join(timeout=1.0)
        except Exception:
            pass
    
    def load_cache(self):
        """Load audio file cache from disk."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.audio_cache = json.load(f)
                logger.info(f"Loaded {len(self.audio_cache)} cached audio files")
            else:
                self.audio_cache = {}
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            self.audio_cache = {}
    
    def save_cache(self):
        """Save audio file cache to disk."""
        try:
            
            self.sfx_base_path.mkdir(exist_ok=True)
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.audio_cache, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved cache with {len(self.audio_cache)} audio files")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def scan_audio_files(self):
        """Scan the sfx directory for audio files and update cache."""
        if not self.sfx_base_path.exists():
            logger.warning(f"SFX directory {self.sfx_base_path} does not exist")
            return
        
        new_files = 0
        updated_files = 0
        
        for root, dirs, files in os.walk(self.sfx_base_path):
            
            if 'myinstants' in Path(root).parts:
                continue
                
            for file in files:
                file_path = Path(root) / file
                
                
                if file_path.suffix.lower() not in self.supported_formats:
                    continue
                
                
                relative_path = file_path.relative_to(self.sfx_base_path)
                file_key = str(relative_path).replace('\\', '/')
                
                
                file_stats = file_path.stat()
                file_info = {
                    'path': str(file_path),
                    'relative_path': file_key,
                    'name': file_path.stem,
                    'category': file_path.parent.name if file_path.parent != self.sfx_base_path else 'general',
                    'format': file_path.suffix.lower(),
                    'size': file_stats.st_size,
                    'modified': file_stats.st_mtime,
                    'duration': None  
                }
                
                
                if file_key not in self.audio_cache:
                    new_files += 1
                elif self.audio_cache[file_key]['modified'] != file_stats.st_mtime:
                    updated_files += 1
                
                self.audio_cache[file_key] = file_info
        
        logger.info(f"Scanned audio files: {new_files} new, {updated_files} updated, {len(self.audio_cache)} total")
        
        if new_files > 0 or updated_files > 0:
            self.save_cache()
    
    def get_audio_files(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of available audio files, optionally filtered by category."""
        files = []
        
        for file_key, file_info in self.audio_cache.items():
            if category and file_info['category'].lower() != category.lower():
                continue
            files.append(file_info)
        
        return sorted(files, key=lambda x: x['name'].lower())
    
    def search_audio_files(self, query: str) -> List[Dict[str, Any]]:
        """Search for audio files by name or category."""
        query = query.lower()
        results = []
        
        for file_key, file_info in self.audio_cache.items():
            if (query in file_info['name'].lower() or 
                query in file_info['category'].lower() or
                query in file_info['relative_path'].lower()):
                results.append(file_info)
        
        return sorted(results, key=lambda x: x['name'].lower())
    
    def get_categories(self) -> List[str]:
        """Get list of available categories."""
        categories = set()
        for file_info in self.audio_cache.values():
            categories.add(file_info['category'])
        
        return sorted(list(categories))
    
    def play_audio_file(self, file_identifier: str, play_mode: str = None, count: int = 1) -> Dict[str, Any]:
        """Play an audio file by name, relative path, or exact match.

        play_mode: Optional; when falling back to MyInstants, pass-through mode 'full' or 'instant'.
        count: how many times to play the file (default 1)
        """
        if not self.audio_system:
            return {
                "success": False,
                "message": "No audio playback system available"
            }

        file_info = self._find_audio_file(file_identifier)
        if not file_info:
            # Attempt fallback to MyInstants if local file not found
            try:
                if MYINSTANTS_AVAILABLE and myinstants_client is not None:
                    logger.info(f"Local audio '{file_identifier}' not found, trying MyInstants search")
                    search_res = myinstants_client.search_sounds(file_identifier, limit=5)
                    if search_res.get("success") and search_res.get("count", 0) > 0:
                        top = search_res["sounds"][0]
                        logger.info(f"Playing MyInstants sound as fallback: {top.get('title')} ({top.get('id')})")
                        play_res = myinstants_client.play_sound(
                            sound_id=top.get("id"),
                            title=top.get("title"),
                            mp3_url=top.get("mp3"),
                            volume=self.volume,
                            immediate=False,
                            play_mode=play_mode or "full",
                            count=int(count)
                        )
                        # Normalize and sanitize response for sfx callers
                        try:
                            sanitized_play_res = _sanitize_for_response(play_res) if isinstance(play_res, dict) else play_res
                        except Exception:
                            sanitized_play_res = {"success": bool(getattr(play_res, 'get', lambda k, d=None: None)('success', False)), "message": str(play_res)}

                        return {
                            "success": bool(sanitized_play_res.get("success") if isinstance(sanitized_play_res, dict) else False),
                            "message": sanitized_play_res.get("message") if isinstance(sanitized_play_res, dict) else str(sanitized_play_res),
                            "myinstants": True,
                            "result": sanitized_play_res
                        }
            except Exception as e:
                logger.debug(f"MyInstants fallback error: {e}")

            return {
                "success": False,
                "message": f"Audio file '{file_identifier}' not found"
            }
        
        file_path = file_info['path']
        
        
        self.stop_audio()
        
        try:
            if self.audio_system == "pygame":
                # Play 'count' times honoring play_mode
                last_res = None
                interval = 0.1 if (play_mode == "instant") else 0.05
                for i in range(max(1, int(count))):
                    res = self._play_with_pygame(file_path, file_info)
                    last_res = res
                    if not res.get("success"):
                        return res
                    if play_mode == "full":
                        # Wait until music finishes
                        try:
                            while pygame.mixer.music.get_busy():
                                time.sleep(0.05)
                        except Exception:
                            pass
                    else:
                        # instant: short gap between starts (note: with pygame.music this restarts stream)
                        time.sleep(interval)
                return last_res
            else:
                return {
                    "success": False,
                    "message": "Pygame audio system not available"
                }
                
        except Exception as e:
            logger.error(f"Error playing audio file {file_path}: {e}")
            return {
                "success": False,
                "message": f"Error playing audio: {str(e)}"
            }
    
    def _find_audio_file(self, identifier: str) -> Optional[Dict[str, Any]]:
        id_raw = identifier.strip()
        id_low = id_raw.lower().replace('\\', '/').strip('/ ')
        if not id_low:
            return None
        try:
            direct_path = self.sfx_base_path / id_low
            if direct_path.exists() and direct_path.is_file():
                key = str(direct_path.relative_to(self.sfx_base_path)).replace('\\', '/')
                for info in self.audio_cache.values():
                    if info['relative_path'].lower() == key.lower():
                        return info
        except Exception:
            pass
        ensured = self._ensure_supported_ext(id_low)
        if ensured:
            direct_path2 = self.sfx_base_path / ensured
            if direct_path2.exists() and direct_path2.is_file():
                key = str(direct_path2.relative_to(self.sfx_base_path)).replace('\\', '/')
                for info in self.audio_cache.values():
                    if info['relative_path'].lower() == key.lower():
                        return info
        for info in self.audio_cache.values():
            if info['relative_path'].lower() == id_low:
                return info
        for info in self.audio_cache.values():
            if info['name'].lower() == id_low:
                return info
        norm_id = self._normalize(id_low)
        want_music = ' music ' in f' {norm_id} '
        best = None
        best_score = -1
        for info in self.audio_cache.values():
            if want_music and info.get('category', '').lower() != 'music':
                continue
            name_norm = self._normalize(info['name'])
            rel_norm = self._normalize(info['relative_path'])
            score = 0
            if norm_id == name_norm or norm_id == rel_norm:
                score = 100
            elif norm_id in name_norm or norm_id in rel_norm:
                score = max(len(norm_id), 1)
            else:
                tokens = set(norm_id.split())
                if tokens:
                    match_tokens = sum(1 for t in tokens if t in name_norm or t in rel_norm)
                    score = match_tokens
            if info.get('category', '').lower() == 'music':
                score += 1
            if score > best_score:
                best = info
                best_score = score
        return best
    
    def _play_with_pygame(self, file_path: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Play audio using pygame."""
        try:
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.set_volume(self.volume)
            pygame.mixer.music.play()
            self.is_playing = True
            self.current_playing_info = file_info
            
            return {
                "success": True,
                "message": f"Playing '{file_info['name']}' using pygame",
                "file_info": file_info,
                "player": "pygame"
            }
        except Exception as e:
            raise Exception(f"Pygame playback error: {e}")
    
    def stop_audio(self) -> Dict[str, Any]:
        """Stop any currently playing audio."""
        try:
            if not self.is_playing:
                return {
                    "success": True,
                    "message": "No audio currently playing"
                }
            
            if self.audio_system == "pygame":
                pygame.mixer.music.stop()
            
            self.is_playing = False
            self.current_playing_info = None
            
            return {
                "success": True,
                "message": "Audio stopped"
            }
            
        except Exception as e:
            logger.error(f"Error stopping audio: {e}")
            return {
                "success": False,
                "message": f"Error stopping audio: {str(e)}"
            }
    
    def set_volume(self, volume: float) -> Dict[str, Any]:
        """Set playback volume (0.0 to 1.0)."""
        try:
            volume = max(0.0, min(1.0, volume))
            self.volume = volume
            
            
            if self.is_playing and self.audio_system == "pygame":
                pygame.mixer.music.set_volume(volume)
            
            return {
                "success": True,
                "message": f"Volume set to {volume:.1%}",
                "volume": volume
            }
            
        except Exception as e:
            logger.error(f"Error setting volume: {e}")
            return {
                "success": False,
                "message": f"Error setting volume: {str(e)}"
            }
    
    def get_playback_status(self) -> Dict[str, Any]:
        """Get current playback status."""
        return {
            "is_playing": self.is_playing,
            "volume": self.volume,
            "audio_system": self.audio_system,
            "total_files": len(self.audio_cache),
            "categories": self.get_categories()
        }

    def is_music_playing(self) -> bool:
        try:
            if not self.is_playing:
                return False
            info = self.current_playing_info or {}
            return str(info.get('category', '')).lower() == 'music'
        except Exception:
            return False



sfx_manager = SFXManager()


SFX_FUNCTION_DECLARATIONS = [
    {
        "name": "play_sfx",
        "description": "Play a custom sound effect or audio file from the sfx folder. Use file name, relative path, or search term.",
        "parameters": {
            "type": "object",
            "properties": {
                "file_identifier": {
                    "type": "string",
                    "description": "Name, relative path, or search term for the audio file (e.g., 'explosion', 'music/song.mp3', 'dramatic')"
                },
                "play_mode": {
                    "type": "string",
                    "description": "Playback behavior when queued: 'full' to wait for each sound to finish, 'instant' to play with short spacing allowing overlap",
                    "enum": ["full", "instant"],
                    "default": "full"
                },
                "count": {
                    "type": "integer",
                    "description": "Number of times to play the sound (default: 1)",
                    "default": 1,
                    "minimum": 1
                }
            },
            "required": ["file_identifier"]
        }
    },
    {
        "name": "stop_sfx",
        "description": "Stop any currently playing sound effect or audio.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "list_sfx",
        "description": "List available sound effects and audio files, optionally filtered by category.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category to filter by (e.g., 'music', 'effects', 'voices')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of files to return",
                    "default": 20
                }
            }
        }
    },
    {
        "name": "search_sfx",
        "description": "Search for sound effects and audio files by name or category.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term for finding audio files"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_sfx_categories",
        "description": "Get list of available sound effect categories.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "set_sfx_volume",
        "description": "Set the volume for sound effect playback.",
        "parameters": {
            "type": "object",
            "properties": {
                "volume": {
                    "type": "number",
                    "description": "Volume level from 0.0 (silent) to 1.0 (maximum)",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["volume"]
        }
    },
    {
        "name": "get_sfx_status",
        "description": "Get current sound effect playback status and system information.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "scan_sfx_files",
        "description": "Scan the sfx folder for new or updated audio files.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]

def _sanitize_for_response(obj):
    def _sanitize(value):
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                if str(k).startswith('_'):
                    continue
                out[k] = _sanitize(v)
            return out
        elif isinstance(value, list):
            return [_sanitize(v) for v in value]
        elif isinstance(value, (str, int, float, bool)) or value is None:
            return value
        else:
            try:
                json.dumps(value)
                return value
            except Exception:
                return repr(value)
    return _sanitize(obj)


async def handle_sfx_function_calls(function_call) -> types.FunctionResponse:
    """Handle SFX-related function calls."""
    function_name = function_call.name
    args = function_call.args
    
    try:
        if function_name == "play_sfx":
            play_mode = args.get("play_mode", None)
            count = args.get("count", 1)
            result = sfx_manager.play_audio_file(args["file_identifier"], play_mode=play_mode, count=count)
        
        elif function_name == "stop_sfx":
            result = sfx_manager.stop_audio()
        
        elif function_name == "list_sfx":
            files = sfx_manager.get_audio_files(args.get("category"))
            limit = args.get("limit", 20)
            
            result = {
                "success": True,
                "files": files[:limit],
                "total_count": len(files),
                "showing": min(len(files), limit)
            }
        
        elif function_name == "search_sfx":
            files = sfx_manager.search_audio_files(args["query"])
            limit = args.get("limit", 10)
            
            result = {
                "success": True,
                "files": files[:limit],
                "total_matches": len(files),
                "showing": min(len(files), limit),
                "query": args["query"]
            }
        
        elif function_name == "get_sfx_categories":
            categories = sfx_manager.get_categories()
            result = {
                "success": True,
                "categories": categories,
                "count": len(categories)
            }
        
        elif function_name == "set_sfx_volume":
            result = sfx_manager.set_volume(args["volume"])
        
        elif function_name == "get_sfx_status":
            result = {
                "success": True,
                **sfx_manager.get_playback_status()
            }
        
        elif function_name == "scan_sfx_files":
            sfx_manager.scan_audio_files()
            result = {
                "success": True,
                "message": "SFX files scanned successfully",
                "total_files": len(sfx_manager.audio_cache)
            }
        
        else:
            result = {
                "success": False,
                "message": f"Unknown SFX function: {function_name}"
            }
        
        sanitized = _sanitize_for_response(result) if isinstance(result, dict) else result
        return types.FunctionResponse(
            id=function_call.id,
            name=function_name,
            response=(sanitized if isinstance(sanitized, dict) else {"result": sanitized})
        )
        
    except Exception as e:
        logger.error(f"Error handling SFX function call {function_name}: {e}")
        return types.FunctionResponse(
            id=function_call.id,
            name=function_name,
            response={
                "success": False,
                "message": f"Error executing {function_name}: {str(e)}"
            }
        )

def get_sfx_tools():
    """Get the SFX tools configuration for Gemini Live API."""
    return [{"function_declarations": SFX_FUNCTION_DECLARATIONS}]

def get_all_sfx_tools():
    """Get all SFX function declarations."""
    return SFX_FUNCTION_DECLARATIONS
