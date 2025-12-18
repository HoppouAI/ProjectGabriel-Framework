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
import time
from collections import deque
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
        """Play all queued sounds if Gabriel is not speaking.

        Implements burst/rapid-play for consecutive repeats of the same sound
        when the client's repeat configuration threshold is hit.
        """
        if self.is_ai_speaking or not self.queued_sounds:
            return

        sounds_to_play = self.queued_sounds.copy()
        self.queued_sounds.clear()

        i = 0
        n = len(sounds_to_play)
        while i < n:
            # Group consecutive identical sound_ids
            current = sounds_to_play[i]
            sid = current.get("sound_id")
            j = i + 1
            group = [current]
            while j < n and sounds_to_play[j].get("sound_id") == sid:
                group.append(sounds_to_play[j])
                j += 1

            # Compute total repeats requested for this consecutive group
            total_count = sum(int(g.get("count", 1)) for g in group)

            # Check client's repeat config to decide if we should burst-play
            cfg = getattr(client_instance, "repeat_config", None)
            do_burst = False
            rapid_interval = 0.05
            if cfg and cfg.get("enabled") and total_count >= int(cfg.get("threshold", 5)):
                do_burst = True
                rapid_interval = float(cfg.get("rapid_interval", 0.05))

            if do_burst:
                logger.info(f"Burst-playing sound '{sid}' {total_count} times with interval {rapid_interval}s")
                # Play quickly in succession total_count times (small awaits so loop doesn't starve event loop)
                for k in range(total_count):
                    try:
                        g = group[0]
                        client_instance._play_sound_immediate(
                            g["sound_id"],
                            g.get("title"),
                            g.get("mp3_url"),
                            g.get("volume", 0.7)
                        )
                    except Exception as e:
                        logger.error(f"Error burst-playing sound {sid}: {e}")
                    await asyncio.sleep(rapid_interval)
            else:
                # Play each normally (maintains existing behavior)
                for g in group:
                    await self._play_queued_sound(g, client_instance)
                    # if the user requested 'instant' spacing, add a small delay between starts
                    play_mode = g.get("play_mode", "full")
                    if play_mode == "instant":
                        interval = g.get("rapid_interval") or client_instance.repeat_config.get("rapid_interval", 0.05)
                        # default to 0.1s for instant spacing if not configured
                        interval = interval if interval is not None else 0.1
                        await asyncio.sleep(interval)

            i = j
    async def _play_queued_sound(self, sound_info: Dict[str, Any], client_instance):
        """Actually play a queued sound.

        Supports two playback modes:
          - 'full'   : wait for the channel to finish before returning (sequential play)
          - 'instant': fire-and-forget with short spacing (overlap allowed)
        """
        try:
            play_mode = sound_info.get("play_mode", "full")
            count = int(sound_info.get("count", 1))

            for idx in range(max(1, count)):
                # request channel back so we can optionally wait for completion
                result = client_instance._play_sound_immediate(
                    sound_info["sound_id"],
                    sound_info.get("title"),
                    sound_info.get("mp3_url"),
                    sound_info.get("volume", 0.7),
                    return_channel=True
                )
                if not result.get("success"):
                    logger.error(f"Failed to play queued sound: {result.get('message', 'Unknown error')}")
                    return

                channel = result.get("_channel")
                logger.info(f"Played queued sound: {sound_info.get('title', 'Unknown')} (mode={play_mode}) [{idx+1}/{count}]")

                if play_mode == "full" and channel is not None:
                    # Wait until the sound finishes playing
                    try:
                        while channel.get_busy():
                            await asyncio.sleep(0.1)
                    except Exception as e:
                        logger.debug(f"Error while waiting for channel to finish: {e}")
                else:
                    # instant: short delay between immediate starts
                    await asyncio.sleep(sound_info.get("rapid_interval") or client_instance.repeat_config.get("rapid_interval", 0.05))

        except Exception as e:
            logger.error(f"Error playing queued sound: {e}")


class MyInstantsClient:
    """Client for interacting with MyInstants API and managing sound effects."""
    
    def __init__(self, cache_dir: str = "sfx/myinstants"):
        self.base_url = "https://myinstants.barricade.dev"
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

        # Repeat/burst configuration
        self.repeat_config = {
            "enabled": True,
            "threshold": 5,           # number of repeats to trigger burst mode
            "rapid_interval": 0.05,   # seconds between rapid plays (very short)
            "detection_period": 5.0   # seconds window to count repeats
        }
        self._play_history: Dict[str, deque] = {}
        self._repeat_lock = threading.Lock()

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
    
    def _safe_title(self, title: str) -> str:
        """Return a filesystem-safe shortened title used for cache lookup."""
        if not title:
            return ""
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')[:50]
        return safe_title

    def _find_cached_by_title(self, title: str):
        """If a cached file already exists for the given title, return its Path.

        This lets multiple identical-sounding uploads (same title) reuse a single
        cached file instead of re-downloading for each sound id.
        """
        safe = self._safe_title(title)
        if not safe:
            return None

        # Prefer files using the standard generated filename pattern
        for p in self.cache_dir.glob(f"{safe}_*.mp3"):
            return p

        # Allow exact filename without hash
        candidate = self.cache_dir / f"{safe}.mp3"
        if candidate.exists():
            return candidate

        # Fallback: any file beginning with the safe title
        matches = list(self.cache_dir.glob(f"{safe}*.mp3"))
        return matches[0] if matches else None

    def _get_cache_path(self, sound_id: str, title: str) -> Path:
        """Get the full cache path for a sound file.

        If a file already exists in cache with the same title, return that path
        so we reuse cached sounds by name rather than re-downloading.
        """
        if title:
            existing = self._find_cached_by_title(title)
            if existing:
                return existing

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
            params = {"q": query}
            
            logger.info(f"Searching for sounds with query: {query} (url={url})")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = self._safe_parse_json(response)
            if data is None:
                return {
                    "success": False,
                    "message": f"Failed to parse JSON response from search (status: {response.status_code}). Response truncated: {response.text[:400]!r}",
                    "sounds": [],
                    "count": 0
                }
            # If the API returns an error object (status/message), surface it instead of silently returning no results
            if isinstance(data, dict) and ("status" in data or "message" in data):
                status = str(data.get("status", "")).strip()
                msg = data.get("message") or data.get("error") or "API error"
                if status and status != "200":
                    logger.warning(f"MyInstants API error on search: status={status} message={msg}")
                    return {
                        "success": False,
                        "message": f"API error: {msg} (status: {status})",
                        "sounds": [],
                        "count": 0,
                        "raw_api_response": data
                    }
            
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
    
    def _normalize_sound_id(self, sound_id: str) -> str:
        """Normalize different forms of sound identifiers into the slug used by the API.

        Accepts full URLs, paths like '/en/instant/slug', numeric IDs, or slug forms and
        returns a best-effort slug to use with the `/detail?id=` API.
        """
        if not sound_id:
            return sound_id

        # If it's a URL, extract the last path segment
        try:
            parsed = urlparse(sound_id)
            if parsed.scheme in ("http", "https") and parsed.path:
                path = parsed.path.rstrip('/')
                # If the path contains '/instant/', use the following segment
                if '/instant/' in path:
                    return path.split('/instant/')[-1].lstrip('/')
                # Otherwise, use the final segment
                return path.split('/')[-1].lstrip('/')
        except Exception:
            pass

        # If it looks like '/en/instant/slug' or '/instant/slug', strip prefixes and slashes
        if sound_id.startswith('/'):
            parts = sound_id.strip('/').split('/')
            if 'instant' in parts:
                idx = parts.index('instant')
                if idx + 1 < len(parts):
                    return parts[idx + 1]
            return parts[-1]

        return sound_id

    def get_sound_details(self, sound_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific sound.

        Accepts various ID formats and will attempt to normalize and retry.
        """
        try:
            normalized_id = self._normalize_sound_id(sound_id)

            url = f"{self.base_url}/detail"
            params = {"id": normalized_id}
            
            logger.info(f"Getting details for sound ID: {sound_id} (normalized: {normalized_id})")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = self._safe_parse_json(response)
            if data is None:
                return {
                    "success": False,
                    "message": f"Failed to parse API response for sound ID '{normalized_id}'"
                }

            if isinstance(data, dict):
                # Surface API-level errors
                if ("status" in data or "message" in data) and str(data.get("status", "")).strip() != "200":
                    return {
                        "success": False,
                        "message": data.get("message", "API error"),
                        "raw_api_response": data
                    }

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
    
    def _safe_parse_json(self, response) -> Optional[Any]:
        """Try to parse JSON robustly, including responses that contain extra HTML or text.

        Returns parsed JSON (dict/list) on success or None on failure.
        """
        try:
            return response.json()
        except (ValueError, json.JSONDecodeError) as e:
            text = response.text or ''
            # Try to find the start of the JSON payload inside HTML or other wrappers
            json_start = text.find('[')
            if json_start == -1:
                json_start = text.find('{')
            if json_start != -1:
                try:
                    return json.loads(text[json_start:])
                except Exception as e2:
                    logger.debug(f"Failed fallback JSON parse: {e2}; response text truncated: {text[:400]!r}")
            logger.error(f"Failed to parse JSON from {getattr(response, 'url', 'unknown')} (status {getattr(response,'status_code', 'unknown')}). Response truncated: {text[:400]!r}")
            return None
    
    def get_trending_sounds(self, region: str = "us", limit: int = 10) -> Dict[str, Any]:
        """Get trending sounds for a specific region."""
        try:
            url = f"{self.base_url}/trending"
            params = {"q": region}
            
            logger.info(f"Getting trending sounds for region: {region}")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = self._safe_parse_json(response)
            if data is None:
                return {
                    "success": False,
                    "message": f"Failed to parse JSON response from trending (status: {response.status_code}). Response truncated: {response.text[:400]!r}",
                    "sounds": [],
                    "count": 0
                }
            # Surface API-level errors from the backend service
            if isinstance(data, dict) and ("status" in data or "message" in data):
                status = str(data.get("status", "")).strip()
                msg = data.get("message") or data.get("error") or "API error"
                if status and status != "200":
                    logger.warning(f"MyInstants API error on trending: status={status} message={msg}")
                    return {
                        "success": False,
                        "message": f"API error: {msg} (status: {status})",
                        "sounds": [],
                        "count": 0,
                        "raw_api_response": data
                    }
            
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
            
            data = self._safe_parse_json(response)
            if data is None:
                return {
                    "success": False,
                    "message": f"Failed to parse JSON response from recent (status: {response.status_code}). Response truncated: {response.text[:400]!r}",
                    "sounds": [],
                    "count": 0
                }
            # Surface API-level errors from the backend service
            if isinstance(data, dict) and ("status" in data or "message" in data):
                status = str(data.get("status", "")).strip()
                msg = data.get("message") or data.get("error") or "API error"
                if status and status != "200":
                    logger.warning(f"MyInstants API error on recent: status={status} message={msg}")
                    return {
                        "success": False,
                        "message": f"API error: {msg} (status: {status})",
                        "sounds": [],
                        "count": 0,
                        "raw_api_response": data
                    }
            
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
    
    def play_sound(self, sound_id: str, title: str = None, mp3_url: str = None, volume: float = 0.7, immediate: bool = False, play_mode: str = "full", count: int = 1) -> Dict[str, Any]:
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
                # First attempt: resolve via the detail endpoint using normalized id
                sound_details = self.get_sound_details(sound_id)

                # If detail lookup failed, try sensible fallbacks (search by title, then by id string)
                if not sound_details["success"]:
                    logger.info(f"Detail lookup failed for '{sound_id}': {sound_details.get('message')}; trying fallbacks")

                    # If a title was provided, prefer searching by it
                    if title:
                        fallback = self.search_sounds(title, limit=5)
                        if fallback.get("success") and fallback.get("count", 0) > 0:
                            found = fallback["sounds"][0]
                            logger.info(f"Found fallback sound by title search: {found.get('id')}")
                            sound_id = found.get("id")
                            mp3_url = found.get("mp3")
                            title = found.get("title", title)
                            sound_details = {"success": True, "sound": found}

                    # If still no result, try searching by the provided sound_id text (useful when numeric ids were given)
                    if not sound_details["success"]:
                        fallback2 = self.search_sounds(str(sound_id), limit=5)
                        if fallback2.get("success") and fallback2.get("count", 0) > 0:
                            found = fallback2["sounds"][0]
                            logger.info(f"Found fallback sound by id-string search: {found.get('id')}")
                            sound_id = found.get("id")
                            mp3_url = found.get("mp3")
                            title = found.get("title", title)
                            sound_details = {"success": True, "sound": found}

                    # If still not found, return the original failure (with helpful message)
                    if not sound_details["success"]:
                        return {
                            "success": False,
                            "message": f"Sound not found. Tried detail lookup for '{sound_id}' and fallback searches.",
                            "original_error": sound_details.get("message")
                        }
                else:
                    sound_data = sound_details["sound"]
                    mp3_url = mp3_url or sound_data.get("mp3")
                    title = title or sound_data.get("title", sound_id)
            
            if not mp3_url:
                return {
                    "success": False,
                    "message": f"No MP3 URL found for sound ID: {sound_id}"
                }
            
            if immediate:
                # Play synchronously 'count' times. For 'full' mode wait for each to finish;
                # for 'instant' mode start each with a short spacing.
                last_res = None
                interval = self.repeat_config.get("rapid_interval", 0.05) if play_mode == "instant" else 0.05
                for i in range(max(1, int(count))):
                    res = self._play_sound_immediate(sound_id, title, mp3_url, volume, return_channel=True)
                    last_res = res
                    if not res.get("success"):
                        # sanitize channel if present before returning
                        if isinstance(res, dict):
                            res.pop("_channel", None)
                        return res
                    ch = res.get("_channel")
                    if play_mode == "full" and ch is not None:
                        # Wait synchronously until channel finishes
                        try:
                            while getattr(ch, "get_busy", lambda: False)():
                                time.sleep(0.05)
                        except Exception:
                            pass
                    else:
                        # instant spacing between starts
                        time.sleep(interval)
                if isinstance(last_res, dict):
                    last_res.pop("_channel", None)
                return last_res or {"success": False, "message": "Failed to play sound"}
            else:
                
                sound_info = {
                    "sound_id": sound_id,
                    "title": title,
                    "mp3_url": mp3_url,
                    "volume": volume,
                    "play_mode": play_mode,
                    "count": int(count)
                }
                
                if self.sound_queue.is_ai_speaking:
                    
                    self.sound_queue.queue_sound(sound_info)
                    return {
                        "success": True,
                        "message": f"Queued sound '{title}' for playback after Gabriel's TTS ends",
                        "sound_id": sound_id,
                        "title": title,
                        "queued": True,
                        "count": int(count)
                    }
                else:
                    
                    # Not speaking, play immediately 'count' times (honor play_mode)
                    last_res = None
                    interval = self.repeat_config.get("rapid_interval", 0.05) if play_mode == "instant" else 0.05
                    for i in range(max(1, int(count))):
                        res = self._play_sound_immediate(sound_id, title, mp3_url, volume, return_channel=True)
                        last_res = res
                        if not res.get("success"):
                            if isinstance(res, dict):
                                res.pop("_channel", None)
                            return res
                        ch = res.get("_channel")
                        if play_mode == "full" and ch is not None:
                            try:
                                while getattr(ch, "get_busy", lambda: False)():
                                    time.sleep(0.05)
                            except Exception:
                                pass
                        else:
                            time.sleep(interval)
                    if isinstance(last_res, dict):
                        last_res.pop("_channel", None)
                    return last_res or {"success": False, "message": "Failed to play sound"}
                
        except Exception as e:
            logger.error(f"Error in play_sound: {e}")
            return {
                "success": False,
                "message": f"Failed to play sound: {str(e)}"
            }
    
    def _play_sound_immediate(self, sound_id: str, title: str = None, mp3_url: str = None, volume: float = 0.7, return_channel: bool = False) -> Dict[str, Any]:
        """Internal method to play a sound immediately without queuing."""
        try:
            
            normalized_id = self._normalize_sound_id(sound_id)
            # If caller didn't provide a title but we have an mp3 URL, derive a simple title from the URL
            if not title and mp3_url:
                try:
                    parsed_mp3 = urlparse(mp3_url)
                    candidate = os.path.splitext(os.path.basename(parsed_mp3.path))[0]
                    if candidate:
                        title = candidate
                except Exception:
                    pass

            cache_path = self._get_cache_path(normalized_id, title)
            
            if not cache_path.exists():
                
                if not self._download_sound(mp3_url, cache_path):
                    return {
                        "success": False,
                        "message": f"Failed to download sound: {title}"
                    }
                # Record mapping for quick future lookups
                try:
                    self.sound_cache[normalized_id] = cache_path
                except Exception:
                    pass
            else:
                logger.info(f"Using cached sound: {cache_path}")
                try:
                    self.sound_cache[normalized_id] = cache_path
                except Exception:
                    pass
            
            
            try:
                
                # Clean up finished channels for this sound id
                try:
                    channels = self.playing_sounds.get(normalized_id, [])
                    active_channels = [ch for ch in channels if getattr(ch, 'get_busy', lambda: False)()]
                    self.playing_sounds[normalized_id] = active_channels
                except Exception:
                    pass

                sound = pygame.mixer.Sound(str(cache_path))
                sound.set_volume(volume)

                channel = sound.play()
                # Store channel so we can optionally wait on it; allow multiple channels per sound id
                try:
                    self.playing_sounds.setdefault(normalized_id, []).append(channel)
                except Exception:
                    self.playing_sounds[normalized_id] = [channel]

                logger.info(f"Playing sound: {title}")

                response = {
                    "success": True,
                    "message": f"Playing sound: {title}",
                    "sound_id": normalized_id,
                    "title": title,
                    "cached": True,
                    "cache_path": str(cache_path)
                }
                # Optionally include channel for internal awaiting callers
                if return_channel:
                    try:
                        response["_channel"] = channel
                    except Exception:
                        pass
                return response
                
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
                normalized_id = self._normalize_sound_id(sound_id)
                if normalized_id in self.playing_sounds:
                    try:
                        # If we stored channels list, stop each; otherwise attempt stop
                        channels = self.playing_sounds.get(normalized_id)
                        if isinstance(channels, list):
                            for ch in channels:
                                try:
                                    ch.stop()
                                except Exception:
                                    pass
                        else:
                            try:
                                channels.stop()
                            except Exception:
                                pass
                    except Exception:
                        pass

                    try:
                        del self.playing_sounds[normalized_id]
                    except Exception:
                        self.playing_sounds.pop(normalized_id, None)

                    return {
                        "success": True,
                        "message": f"Stopped sound: {normalized_id}"
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

    def set_repeat_config(self, enabled: bool = True, threshold: int = 5, rapid_interval: float = 0.05, detection_period: float = 5.0) -> Dict[str, Any]:
        """Configure burst/rapid-play behavior when same sound is queued repeatedly.

        Args:
            enabled: Enable/disable the feature
            threshold: Number of repeated queued plays to trigger burst mode
            rapid_interval: Seconds between rapid plays when bursting
            detection_period: Time window (seconds) used when heuristics elsewhere track repeats (not used in queue grouping)
        """
        try:
            self.repeat_config["enabled"] = bool(enabled)
            self.repeat_config["threshold"] = int(threshold)
            self.repeat_config["rapid_interval"] = float(rapid_interval)
            self.repeat_config["detection_period"] = float(detection_period)

            return {
                "success": True,
                "message": "Repeat configuration updated",
                "repeat_config": self.repeat_config
            }
        except Exception as e:
            logger.error(f"Error setting repeat config: {e}")
            return {
                "success": False,
                "message": f"Failed to set repeat config: {str(e)}"
            }

    def get_repeat_config(self) -> Dict[str, Any]:
        try:
            return {
                "success": True,
                "repeat_config": self.repeat_config
            }
        except Exception as e:
            logger.error(f"Error getting repeat config: {e}")
            return {
                "success": False,
                "message": f"Failed to get repeat config: {str(e)}"
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
                },
                "play_mode": {
                    "type": "string",
                    "description": "Playback behavior when queued: 'full' to wait for full length sequential playback, 'instant' to play with short spacing allowing overlap",
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
        "name": "configure_myinstants_repeat",
        "description": "Configure burst/rapid-play behavior: when the same sound is requested repeated times, play them quickly.",
        "parameters": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean", "description": "Enable or disable rapid repeat detection", "default": True},
                "threshold": {"type": "integer", "description": "Number of repeats to trigger burst mode (default: 5)", "default": 5},
                "rapid_interval": {"type": "number", "description": "Seconds between rapid plays in burst mode (default: 0.05)", "default": 0.05},
                "detection_period": {"type": "number", "description": "Detection window in seconds (informational)", "default": 5.0}
            }
        }
    },
    {
        "name": "get_myinstants_repeat_config",
        "description": "Get the current rapid/burst repeat configuration.",
        "parameters": {
            "type": "object",
            "properties": {}
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

def _sanitize_for_response(obj):
    """Sanitize a result dict for JSON/function response by removing internal keys and non-serializable objects.

    - Removes keys that start with '_' (internal)
    - Converts non-serializable values to their repr()
    """
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
            # Attempt JSON serialization, fallback to repr
            try:
                json.dumps(value)
                return value
            except Exception:
                return repr(value)

    return _sanitize(obj)


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
                immediate=args.get("immediate", False),
                play_mode=args.get("play_mode", "full"),
                count=args.get("count", 1)
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
        
        elif function_name == "configure_myinstants_repeat":
            result = myinstants_client.set_repeat_config(
                enabled=args.get("enabled", True),
                threshold=args.get("threshold", 5),
                rapid_interval=args.get("rapid_interval", 0.05),
                detection_period=args.get("detection_period", 5.0)
            )
        
        elif function_name == "get_myinstants_repeat_config":
            result = myinstants_client.get_repeat_config()
        
        else:
            result = {
                "success": False,
                "message": f"Unknown MyInstants function: {function_name}"
            }
        
        sanitized = _sanitize_for_response(result) if isinstance(result, dict) else result
        return types.FunctionResponse(
            id=function_call.id,
            name=function_name,
            response={
                **(sanitized if isinstance(sanitized, dict) else {"result": sanitized}),
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
