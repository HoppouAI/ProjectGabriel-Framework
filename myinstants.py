"""
MyInstants module for function calling with Gemini Live API
Provides tools to search, download, and play sound effects from MyInstants.
"""

import os
import json
import logging
import asyncio
import hashlib
import requests
import pygame
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import quote, urlparse
from google.genai import types


logger = logging.getLogger(__name__)

class SimpleSoundQueue:
    """Simple sound queue that plays sounds after Gabriel's TTS finishes."""
    
    def __init__(self):
        self.queued_sounds = []
        self.is_ai_speaking = False
        
    def queue_sound(self, sound_info: Dict[str, Any]):
        """Queue a sound for playback after Gabriel's TTS ends."""
        self.queued_sounds.append(sound_info)
        logger.info(f"Queued sound: {sound_info.get('title', 'Unknown')}")
        
    def set_ai_speaking(self, speaking: bool):
        """Set whether Gabriel's TTS is currently speaking."""
        self.is_ai_speaking = speaking
        if not speaking:
            logger.info("Gabriel's TTS stopped - will play queued sounds")
        
    async def process_queue(self, client_instance):
        """Play all queued sounds if Gabriel is not speaking."""
        if self.is_ai_speaking or not self.queued_sounds:
            return
            
        
        sounds_to_play = self.queued_sounds.copy()
        self.queued_sounds.clear()
        
        for sound_info in sounds_to_play:
            await self._play_queued_sound(sound_info, client_instance)
            
    async def _play_queued_sound(self, sound_info: Dict[str, Any], client_instance):
        """Actually play a queued sound."""
        try:
            result = client_instance._play_sound_immediate(
                sound_info["sound_id"],
                sound_info.get("title"),
                sound_info.get("mp3_url"),
                sound_info.get("volume", 0.7)
            )
            if result["success"]:
                logger.info(f"Played queued sound: {sound_info.get('title', 'Unknown')}")
            else:
                logger.error(f"Failed to play queued sound: {result.get('message', 'Unknown error')}")
        except Exception as e:
            logger.error(f"Error playing queued sound: {e}")


class MyInstantsClient:
    """Client for interacting with MyInstants API and managing sound effects."""
    
    def __init__(self, cache_dir: str = "sfx/myinstants"):
        self.base_url = "https://myinstants-api.vercel.app"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        
        self.sound_queue = SimpleSoundQueue()
        
        
        try:
            pygame.mixer.init()
            self.mixer_initialized = True
            logger.info("Pygame mixer initialized successfully")
        except Exception as e:
            self.mixer_initialized = False
            logger.error(f"Failed to initialize pygame mixer: {e}")
        
        
        self.playing_sounds = {}
        self.sound_cache = {}
        
        
        self._queue_task = None
        self._start_queue_processor()
    
    def _start_queue_processor(self):
        """Start the background queue processor."""
        def run_queue_processor():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._queue_processor_loop())
            except Exception as e:
                logger.error(f"Queue processor error: {e}")
            finally:
                loop.close()
        
        if self._queue_task is None or self._queue_task.done():
            self._queue_task = threading.Thread(target=run_queue_processor, daemon=True)
            self._queue_task.start()
    
    async def _queue_processor_loop(self):
        """Background loop to process the sound queue."""
        while True:
            try:
                await self.sound_queue.process_queue(self)
                await asyncio.sleep(0.1)  
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(1.0)  
    
    def notify_ai_tts_started(self):
        """Call this when Gabriel's TTS starts speaking."""
        self.sound_queue.set_ai_speaking(True)
        logger.info("Gabriel's TTS started")
    
    def notify_ai_tts_ended(self):
        """Call this when Gabriel's TTS stops speaking."""
        self.sound_queue.set_ai_speaking(False)
        logger.info("Gabriel's TTS ended")
    
    
    def notify_ai_audio_received(self):
        """Compatibility method: Called when Gabriel's audio is received."""
        if not self.sound_queue.is_ai_speaking:
            self.notify_ai_tts_started()
    
    def notify_ai_speech_ended(self):
        """Compatibility method: Called when Gabriel's speech/turn ends."""
        self.notify_ai_tts_ended()
    
    def _generate_cache_filename(self, sound_id: str, title: str) -> str:
        """Generate a safe filename for caching."""
        
        hash_obj = hashlib.md5(sound_id.encode())
        hash_str = hash_obj.hexdigest()[:8]
        
        
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')[:50]  
        
        return f"{safe_title}_{hash_str}.mp3"
    
    def _get_cache_path(self, sound_id: str, title: str) -> Path:
        """Get the full cache path for a sound file."""
        filename = self._generate_cache_filename(sound_id, title)
        return self.cache_dir / filename
    
    def _download_sound(self, mp3_url: str, cache_path: Path) -> bool:
        """Download a sound file to the cache directory."""
        try:
            logger.info(f"Downloading sound from {mp3_url} to {cache_path}")
            
            response = requests.get(mp3_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(cache_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Successfully downloaded sound to {cache_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download sound: {e}")
            if cache_path.exists():
                cache_path.unlink()  
            return False
    
    def search_sounds(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search for sounds using the MyInstants API."""
        try:
            url = f"{self.base_url}/search"
            params = {"q": quote(query)}
            
            logger.info(f"Searching for sounds with query: {query}")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            
            if isinstance(data, dict) and "data" in data:
                sounds = data["data"]
                if isinstance(sounds, list) and len(sounds) > 0:
                    
                    results = sounds[:limit]
                    
                    return {
                        "success": True,
                        "sounds": results,
                        "count": len(results),
                        "query": query
                    }
                else:
                    return {
                        "success": False,
                        "message": f"No sounds found for query: {query}",
                        "sounds": [],
                        "count": 0
                    }
            
            elif isinstance(data, list) and len(data) > 0:
                
                results = data[:limit]
                
                return {
                    "success": True,
                    "sounds": results,
                    "count": len(results),
                    "query": query
                }
            else:
                return {
                    "success": False,
                    "message": f"No sounds found for query: {query}",
                    "sounds": [],
                    "count": 0
                }
                
        except Exception as e:
            logger.error(f"Error searching sounds: {e}")
            return {
                "success": False,
                "message": f"Failed to search sounds: {str(e)}",
                "sounds": [],
                "count": 0
            }
    
    def get_sound_details(self, sound_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific sound."""
        try:
            url = f"{self.base_url}/detail"
            params = {"id": sound_id}
            
            logger.info(f"Getting details for sound ID: {sound_id}")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            
            response_text = response.text
            
            
            try:
                
                json_start = response_text.find('{')
                if json_start != -1:
                    json_text = response_text[json_start:]
                    data = json.loads(json_text)
                else:
                    data = response.json()
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse JSON response: {e}")
                logger.debug(f"Response text: {response_text}")
                return {
                    "success": False,
                    "message": f"Failed to parse API response for sound ID '{sound_id}'"
                }
            
            
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], dict) and "id" in data["data"]:
                    return {
                        "success": True,
                        "sound": data["data"]
                    }
                elif "id" in data:  
                    return {
                        "success": True,
                        "sound": data
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Sound with ID '{sound_id}' not found"
                    }
            else:
                return {
                    "success": False,
                    "message": f"Sound with ID '{sound_id}' not found"
                }
                
        except Exception as e:
            logger.error(f"Error getting sound details: {e}")
            return {
                "success": False,
                "message": f"Failed to get sound details: {str(e)}"
            }
    
    def get_trending_sounds(self, region: str = "us", limit: int = 10) -> Dict[str, Any]:
        """Get trending sounds for a specific region."""
        try:
            url = f"{self.base_url}/trending"
            params = {"q": region}
            
            logger.info(f"Getting trending sounds for region: {region}")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            
            if isinstance(data, dict) and "data" in data:
                sounds = data["data"]
                if isinstance(sounds, list) and len(sounds) > 0:
                    
                    results = sounds[:limit]
                    
                    return {
                        "success": True,
                        "sounds": results,
                        "count": len(results),
                        "region": region
                    }
                else:
                    return {
                        "success": False,
                        "message": f"No trending sounds found for region: {region}",
                        "sounds": [],
                        "count": 0
                    }
            
            elif isinstance(data, list) and len(data) > 0:
                
                results = data[:limit]
                
                return {
                    "success": True,
                    "sounds": results,
                    "count": len(results),
                    "region": region
                }
            else:
                return {
                    "success": False,
                    "message": f"No trending sounds found for region: {region}",
                    "sounds": [],
                    "count": 0
                }
                
        except Exception as e:
            logger.error(f"Error getting trending sounds: {e}")
            return {
                "success": False,
                "message": f"Failed to get trending sounds: {str(e)}",
                "sounds": [],
                "count": 0
            }
    
    def get_recent_sounds(self, limit: int = 10) -> Dict[str, Any]:
        """Get recently uploaded sounds."""
        try:
            url = f"{self.base_url}/recent"
            
            logger.info("Getting recent sounds")
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            
            if isinstance(data, dict) and "data" in data:
                sounds = data["data"]
                if isinstance(sounds, list) and len(sounds) > 0:
                    
                    results = sounds[:limit]
                    
                    return {
                        "success": True,
                        "sounds": results,
                        "count": len(results)
                    }
                else:
                    return {
                        "success": False,
                        "message": "No recent sounds found",
                        "sounds": [],
                        "count": 0
                    }
            
            elif isinstance(data, list) and len(data) > 0:
                
                results = data[:limit]
                
                return {
                    "success": True,
                    "sounds": results,
                    "count": len(results)
                }
            else:
                return {
                    "success": False,
                    "message": "No recent sounds found",
                    "sounds": [],
                    "count": 0
                }
                
        except Exception as e:
            logger.error(f"Error getting recent sounds: {e}")
            return {
                "success": False,
                "message": f"Failed to get recent sounds: {str(e)}",
                "sounds": [],
                "count": 0
            }
    
    def play_sound(self, sound_id: str, title: str = None, mp3_url: str = None, volume: float = 0.7, immediate: bool = False) -> Dict[str, Any]:
        """Play a sound effect. By default, queues for playback after Gabriel's TTS ends.
        
        Args:
            sound_id: Sound ID to play
            title: Optional title for caching
            mp3_url: Optional direct MP3 URL
            volume: Volume level (0.0-1.0)
            immediate: If True, play immediately. If False, queue for after Gabriel's TTS ends (default)
        """
        if not self.mixer_initialized:
            return {
                "success": False,
                "message": "Pygame mixer not initialized. Cannot play sounds."
            }
        
        try:
            
            if not mp3_url or not title:
                sound_details = self.get_sound_details(sound_id)
                if not sound_details["success"]:
                    return sound_details
                
                sound_data = sound_details["sound"]
                mp3_url = sound_data.get("mp3")
                title = sound_data.get("title", sound_id)
            
            if not mp3_url:
                return {
                    "success": False,
                    "message": f"No MP3 URL found for sound ID: {sound_id}"
                }
            
            if immediate:
                
                return self._play_sound_immediate(sound_id, title, mp3_url, volume)
            else:
                
                sound_info = {
                    "sound_id": sound_id,
                    "title": title,
                    "mp3_url": mp3_url,
                    "volume": volume
                }
                
                if self.sound_queue.is_ai_speaking:
                    
                    self.sound_queue.queue_sound(sound_info)
                    return {
                        "success": True,
                        "message": f"Queued sound '{title}' for playback after Gabriel's TTS ends",
                        "sound_id": sound_id,
                        "title": title,
                        "queued": True
                    }
                else:
                    
                    return self._play_sound_immediate(sound_id, title, mp3_url, volume)
                
        except Exception as e:
            logger.error(f"Error in play_sound: {e}")
            return {
                "success": False,
                "message": f"Failed to play sound: {str(e)}"
            }
    
    def _play_sound_immediate(self, sound_id: str, title: str = None, mp3_url: str = None, volume: float = 0.7) -> Dict[str, Any]:
        """Internal method to play a sound immediately without queuing."""
        try:
            
            cache_path = self._get_cache_path(sound_id, title)
            
            if not cache_path.exists():
                
                if not self._download_sound(mp3_url, cache_path):
                    return {
                        "success": False,
                        "message": f"Failed to download sound: {title}"
                    }
            else:
                logger.info(f"Using cached sound: {cache_path}")
            
            
            try:
                
                if sound_id in self.playing_sounds:
                    self.playing_sounds[sound_id].stop()
                
                
                sound = pygame.mixer.Sound(str(cache_path))
                sound.set_volume(volume)
                
                
                channel = sound.play()
                self.playing_sounds[sound_id] = sound
                
                logger.info(f"Playing sound: {title}")
                
                return {
                    "success": True,
                    "message": f"Playing sound: {title}",
                    "sound_id": sound_id,
                    "title": title,
                    "cached": True,
                    "cache_path": str(cache_path)
                }
                
            except Exception as e:
                logger.error(f"Error playing sound: {e}")
                return {
                    "success": False,
                    "message": f"Failed to play sound: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Error in _play_sound_immediate: {e}")
            return {
                "success": False,
                "message": f"Failed to play sound: {str(e)}"
            }
    
    def stop_sound(self, sound_id: str = None) -> Dict[str, Any]:
        """Stop a specific sound or all sounds."""
        try:
            if sound_id:
                if sound_id in self.playing_sounds:
                    self.playing_sounds[sound_id].stop()
                    del self.playing_sounds[sound_id]
                    return {
                        "success": True,
                        "message": f"Stopped sound: {sound_id}"
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Sound {sound_id} is not currently playing"
                    }
            else:
                
                pygame.mixer.stop()
                self.playing_sounds.clear()
                return {
                    "success": True,
                    "message": "Stopped all sounds"
                }
                
        except Exception as e:
            logger.error(f"Error stopping sound: {e}")
            return {
                "success": False,
                "message": f"Failed to stop sound: {str(e)}"
            }
    
    def set_volume(self, volume: float) -> Dict[str, Any]:
        """Set the master volume for all sounds."""
        try:
            volume = max(0.0, min(1.0, volume))  
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
                "message": f"Failed to set volume: {str(e)}"
            }
    
    def clear_cache(self, confirm: bool = False) -> Dict[str, Any]:
        """Clear the sound cache directory."""
        if not confirm:
            return {
                "success": False,
                "message": "Cache clearing requires confirmation. Set confirm=True to proceed."
            }
        
        try:
            
            self.stop_sound()
            
            
            files_removed = 0
            for file_path in self.cache_dir.glob("*.mp3"):
                try:
                    file_path.unlink()
                    files_removed += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")
            
            return {
                "success": True,
                "message": f"Cache cleared. Removed {files_removed} files.",
                "files_removed": files_removed
            }
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return {
                "success": False,
                "message": f"Failed to clear cache: {str(e)}"
            }
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache directory."""
        try:
            cache_files = list(self.cache_dir.glob("*.mp3"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "success": True,
                "cache_directory": str(self.cache_dir),
                "cached_files": len(cache_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "files": [f.name for f in cache_files]
            }
            
        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {
                "success": False,
                "message": f"Failed to get cache info: {str(e)}"
            }
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get information about the current sound queue."""
        try:
            return {
                "success": True,
                "queued_sounds": len(self.sound_queue.queued_sounds),
                "is_ai_speaking": self.sound_queue.is_ai_speaking,
                "queue_details": [
                    {
                        "title": sound_info.get("title", "Unknown"),
                        "sound_id": sound_info.get("sound_id", "Unknown")
                    }
                    for sound_info in self.sound_queue.queued_sounds
                ]
            }
        except Exception as e:
            logger.error(f"Error getting queue status: {e}")
            return {
                "success": False,
                "message": f"Failed to get queue status: {str(e)}"
            }
    
    def clear_sound_queue(self) -> Dict[str, Any]:
        """Clear all queued sounds."""
        try:
            cleared_count = len(self.sound_queue.queued_sounds)
            self.sound_queue.queued_sounds.clear()
            
            return {
                "success": True,
                "message": f"Cleared {cleared_count} queued sounds",
                "cleared_count": cleared_count
            }
        except Exception as e:
            logger.error(f"Error clearing sound queue: {e}")
            return {
                "success": False,
                "message": f"Failed to clear sound queue: {str(e)}"
            }
    
    def set_ai_tts_state(self, speaking: bool) -> Dict[str, Any]:
        """Set Gabriel's TTS speaking state."""
        try:
            if speaking:
                self.notify_ai_tts_started()
            else:
                self.notify_ai_tts_ended()
            
            return {
                "success": True,
                "message": f"Gabriel's TTS state set to: {'speaking' if speaking else 'not speaking'}",
                "is_speaking": speaking
            }
        except Exception as e:
            logger.error(f"Error setting Gabriel's TTS state: {e}")
            return {
                "success": False,
                "message": f"Failed to set Gabriel's TTS state: {str(e)}"
            }



myinstants_client = MyInstantsClient()


MYINSTANTS_FUNCTION_DECLARATIONS = [
    {
        "name": "search_myinstants_sounds",
        "description": "Search for sound effects on MyInstants by keyword or phrase. Returns a list of available sounds with their details.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search term or phrase to find sounds (e.g., 'laugh', 'applause', 'sad trombone')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10, max: 50)",
                    "default": 10
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "play_myinstants_sound",
        "description": "Play a sound effect from MyInstants. By default, sounds are queued to play after Gabriel's TTS finishes speaking for better conversational flow. The sound will be downloaded and cached automatically for future use.",
        "parameters": {
            "type": "object",
            "properties": {
                "sound_id": {
                    "type": "string",
                    "description": "The unique ID of the sound to play (obtained from search results)"
                },
                "title": {
                    "type": "string",
                    "description": "Optional title of the sound (helps with caching)"
                },
                "mp3_url": {
                    "type": "string",
                    "description": "Optional direct MP3 URL (if available from search results)"
                },
                "volume": {
                    "type": "number",
                    "description": "Volume level from 0.0 to 1.0 (default: 0.7)",
                    "default": 0.7
                },
                "immediate": {
                    "type": "boolean",
                    "description": "If true, play immediately. If false (default), queue for playback after Gabriel's TTS ends",
                    "default": False
                }
            },
            "required": ["sound_id"]
        }
    },
    {
        "name": "get_myinstants_sound_details",
        "description": "Get detailed information about a specific sound including title, description, tags, and MP3 URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "sound_id": {
                    "type": "string",
                    "description": "The unique ID of the sound to get details for"
                }
            },
            "required": ["sound_id"]
        }
    },
    {
        "name": "get_trending_myinstants_sounds",
        "description": "Get trending/popular sound effects from MyInstants for a specific region.",
        "parameters": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "Region code (e.g., 'us', 'uk', 'de', 'fr') - default: 'us'",
                    "default": "us"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)",
                    "default": 10
                }
            }
        }
    },
    {
        "name": "get_recent_myinstants_sounds",
        "description": "Get recently uploaded sound effects from MyInstants.",
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)",
                    "default": 10
                }
            }
        }
    },
    {
        "name": "stop_myinstants_sound",
        "description": "Stop playing sound effects. Can stop a specific sound or all sounds.",
        "parameters": {
            "type": "object",
            "properties": {
                "sound_id": {
                    "type": "string",
                    "description": "Optional sound ID to stop. If not provided, stops all sounds."
                }
            }
        }
    },
    {
        "name": "set_myinstants_volume",
        "description": "Set the volume level for MyInstants sound playback.",
        "parameters": {
            "type": "object",
            "properties": {
                "volume": {
                    "type": "number",
                    "description": "Volume level from 0.0 (mute) to 1.0 (full volume)",
                    "minimum": 0.0,
                    "maximum": 1.0
                }
            },
            "required": ["volume"]
        }
    },
    {
        "name": "get_myinstants_cache_info",
        "description": "Get information about the MyInstants sound cache (cached files, total size, etc.).",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "clear_myinstants_cache",
        "description": "Clear the MyInstants sound cache to free up disk space. Use with caution as it will remove all cached sound files.",
        "parameters": {
            "type": "object",
            "properties": {
                "confirm": {
                    "type": "boolean",
                    "description": "Must be set to true to confirm cache clearing",
                    "default": False
                }
            }
        }
    },
    {
        "name": "get_myinstants_queue_status",
        "description": "Get information about the current sound queue and timing status.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "clear_myinstants_queue",
        "description": "Clear all queued sounds that haven't played yet.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "set_ai_tts_state",
        "description": "Set whether Gabriel's TTS is currently speaking. This controls when queued sounds will play.",
        "parameters": {
            "type": "object",
            "properties": {
                "speaking": {
                    "type": "boolean",
                    "description": "True if Gabriel's TTS is speaking, False if it has stopped"
                }
            },
            "required": ["speaking"]
        }
    }
]

async def handle_myinstants_function_call(function_call) -> types.FunctionResponse:
    """Handle MyInstants-related function calls."""
    function_name = function_call.name
    args = function_call.args
    
    try:
        if function_name == "search_myinstants_sounds":
            result = myinstants_client.search_sounds(
                query=args["query"],
                limit=args.get("limit", 10)
            )
        
        elif function_name == "play_myinstants_sound":
            result = myinstants_client.play_sound(
                sound_id=args["sound_id"],
                title=args.get("title"),
                mp3_url=args.get("mp3_url"),
                volume=args.get("volume", 0.7),
                immediate=args.get("immediate", False)
            )
        
        elif function_name == "get_myinstants_sound_details":
            result = myinstants_client.get_sound_details(args["sound_id"])
        
        elif function_name == "get_trending_myinstants_sounds":
            result = myinstants_client.get_trending_sounds(
                region=args.get("region", "us"),
                limit=args.get("limit", 10)
            )
        
        elif function_name == "get_recent_myinstants_sounds":
            result = myinstants_client.get_recent_sounds(
                limit=args.get("limit", 10)
            )
        
        elif function_name == "stop_myinstants_sound":
            result = myinstants_client.stop_sound(
                sound_id=args.get("sound_id")
            )
        
        elif function_name == "set_myinstants_volume":
            result = myinstants_client.set_volume(args["volume"])
        
        elif function_name == "get_myinstants_cache_info":
            result = myinstants_client.get_cache_info()
        
        elif function_name == "clear_myinstants_cache":
            result = myinstants_client.clear_cache(
                confirm=args.get("confirm", False)
            )
        
        elif function_name == "get_myinstants_queue_status":
            result = myinstants_client.get_queue_status()
        
        elif function_name == "clear_myinstants_queue":
            result = myinstants_client.clear_sound_queue()
        
        elif function_name == "set_ai_tts_state":
            result = myinstants_client.set_ai_tts_state(args["speaking"])
        
        else:
            result = {
                "success": False,
                "message": f"Unknown MyInstants function: {function_name}"
            }
        
        return types.FunctionResponse(
            id=function_call.id,
            name=function_name,
            response={
                **result,
                "scheduling": "SILENT"
            }
        )
        
    except Exception as e:
        logger.error(f"Error handling MyInstants function call {function_name}: {e}")
        return types.FunctionResponse(
            id=function_call.id,
            name=function_name,
            response={
                "success": False,
                "message": f"Error executing {function_name}: {str(e)}",
                "scheduling": "SILENT"
            }
        )

def get_myinstants_tools():
    """Get the MyInstants tools configuration for Gemini Live API."""
    return [{"function_declarations": MYINSTANTS_FUNCTION_DECLARATIONS}]


def get_all_myinstants_tools():
    """Get all MyInstants tools for integration with other tool modules."""
    return MYINSTANTS_FUNCTION_DECLARATIONS

async def handle_myinstants_function_calls(function_call) -> types.FunctionResponse:
    """Main function call handler for MyInstants functions."""
    return await handle_myinstants_function_call(function_call)
