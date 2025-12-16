"""
Memory reader module for retrieving memories from the persistent storage.
12/16/2025 - Added SQLite support as an alternative backend.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

try:
    from pymongo import DESCENDING, MongoClient
    from pymongo.collection import Collection
    from pymongo.errors import PyMongoError
except Exception:
    DESCENDING = None
    MongoClient = None
    Collection = Any
    PyMongoError = Exception

from tools.memory import (
    MEMORY_TYPE_LONG_TERM,
    MEMORY_TYPE_SHORT_TERM,
    MEMORY_TYPE_QUICK_NOTE,
    get_mongo_connection_settings,
    memory_system,
)

logger = logging.getLogger(__name__)


class MemoryReader:
    """Class to handle memory retrieval and formatting."""

    def __init__(self, connection_settings: Optional[Dict[str, Any]] = None):
        self.settings = connection_settings or get_mongo_connection_settings()
        self._use_direct_mongo = bool(getattr(memory_system, "backend", "mongo") == "mongo" and MongoClient is not None)
        self.collection: Optional[Collection] = None
        self.client: Optional[MongoClient] = None
        if self._use_direct_mongo:
            self._initialize_collection()

    def _initialize_collection(self):
        if not self._use_direct_mongo:
            return
        existing_collection = getattr(memory_system, "collection", None) if memory_system is not None else None
        if existing_collection is not None:
            self.collection = existing_collection
            return
        uri = self.settings.get("uri") or ""
        if not uri:
            logger.error("MongoDB URI not configured for MemoryReader")
            return
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            database_name = self.settings.get("database") or "gabriel"
            collection_name = self.settings.get("collection") or "memories"
            self.collection = self.client[database_name][collection_name]
        except Exception as exc:
            logger.error(f"Failed to connect to MongoDB for memory reader: {exc}")
            self.collection = None

    def _ensure_collection(self) -> bool:
        if not self._use_direct_mongo:
            return False
        if self.collection is not None:
            return True
        self._initialize_collection()
        return self.collection is not None
    
    def get_recent_memories(self, count: int = 10) -> List[Dict[str, Any]]:
        try:
            getter = getattr(memory_system, "get_recent_memories_for_prompt", None)
            if callable(getter):
                memories = getter(count)
                logger.debug(f"Retrieved {len(memories)} recent memories")
                return memories
        except Exception:
            pass
        if not self._ensure_collection():
            logger.error("Memory collection unavailable")
            return []
        try:
            real_limit = max(1, int(count * 0.8))
            note_limit = max(0, count - real_limit)
            real_docs = list(
                self.collection.find(
                    {"memory_type": {"$in": [MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM]}},
                    {
                        "key": 1,
                        "content": 1,
                        "category": 1,
                        "created_at": 1,
                        "tags": 1,
                    },
                ).sort("created_at", DESCENDING).limit(real_limit)
            )
            note_docs: List[Dict[str, Any]] = []
            if note_limit:
                note_docs = list(
                    self.collection.find(
                        {"memory_type": MEMORY_TYPE_QUICK_NOTE},
                        {
                            "key": 1,
                            "content": 1,
                            "category": 1,
                            "created_at": 1,
                            "tags": 1,
                        },
                    ).sort("created_at", DESCENDING).limit(note_limit)
                )
            docs = real_docs + note_docs
            memories: List[Dict[str, Any]] = []
            for doc in docs:
                created_at = doc.get("created_at")
                if isinstance(created_at, datetime):
                    created = created_at.isoformat()
                elif created_at:
                    created = str(created_at)
                else:
                    created = ""
                tags = doc.get("tags") or []
                if isinstance(tags, list):
                    tags_str = ",".join(tags)
                else:
                    tags_str = str(tags)
                memories.append({
                    "id": str(doc.get("_id")),
                    "key": doc.get("key"),
                    "content": doc.get("content"),
                    "category": doc.get("category", "general"),
                    "created_at": created,
                    "tags": tags_str
                })
            logger.debug(f"Retrieved {len(memories)} recent memories")
            return memories
        except PyMongoError as exc:
            logger.error(f"Database error retrieving memories: {exc}")
            return []
        except Exception as exc:
            logger.error(f"Unexpected error retrieving memories: {exc}")
            return []
    
    def format_memories_for_prompt(
        self, 
        memories: List[Dict[str, Any]], 
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format memories for inclusion in a system prompt.
        
        Args:
            memories: List of memory dictionaries
            config: Configuration dictionary for formatting options
            
        Returns:
            Formatted string containing memory information
        """
        if not memories:
            return ""
        
        # Default configuration
        format_config = {
            'include_timestamps': True,
            'include_categories': True,
            'max_content_length': 200,
            'separator': '\n'
        }
        
        # Update with provided config
        if config:
            format_config.update(config)
        
        formatted_parts = []
        
        for i, memory in enumerate(memories, 1):
            # Create a clear, structured format for each memory
            memory_lines = []
            
            # Memory header with number and key
            if memory['key']:
                header = f"{i}. Memory: {memory['key']}"
            else:
                header = f"{i}. Memory (ID: {memory['id']})"
            
            # Add category if enabled and not general
            if format_config['include_categories'] and memory['category'] != 'general':
                header += f" (Category: {memory['category']})"
            
            memory_lines.append(header)
            
            # Add content with clear labeling
            content = memory['content']
            max_length = format_config['max_content_length']
            if len(content) > max_length:
                content = content[:max_length] + "... (use memory tools to fetch the rest of this if needed)"
            memory_lines.append(f"   Content: {content}")
            
            # Add timestamp if enabled
            if format_config['include_timestamps'] and memory['created_at']:
                try:
                    # Parse the timestamp and format it nicely
                    if memory['created_at']:
                        created_date = datetime.fromisoformat(memory['created_at'].replace('Z', '+00:00'))
                        formatted_date = created_date.strftime('%B %d, %Y')
                        memory_lines.append(f"   Date: {formatted_date}")
                except (ValueError, TypeError):
                    # If timestamp parsing fails, skip it
                    pass
            
            # Join lines for this memory
            memory_text = '\n'.join(memory_lines)
            formatted_parts.append(memory_text)
        
        # Join all memories with double newlines for clear separation
        return '\n\n'.join(formatted_parts)
    
    def get_formatted_recent_memories(
        self, 
        count: int = 10, 
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Get and format recent memories in one step.
        
        Args:
            count: Number of recent memories to retrieve
            config: Configuration dictionary for formatting options
            
        Returns:
            Formatted string ready for inclusion in system prompt
        """
        memories = self.get_recent_memories(count)
        if not memories:
            logger.debug("No memories found to format")
            return ""
        
        formatted = self.format_memories_for_prompt(memories, config)
        logger.info(f"Formatted {len(memories)} memories for system prompt")
        return formatted
    
    def check_database_exists(self) -> bool:
        try:
            checker = getattr(memory_system, "is_available", None)
            if callable(checker):
                return bool(checker())
        except Exception:
            pass
        if not self._ensure_collection():
            return False
        try:
            self.collection.estimated_document_count()
            return True
        except PyMongoError as exc:
            logger.error(f"Error checking memory collection: {exc}")
            return False
    
    def get_memory_count(self) -> int:
        try:
            counter = getattr(memory_system, "get_memory_count", None)
            if callable(counter):
                return int(counter())
        except Exception:
            pass
        if not self._ensure_collection():
            return 0
        try:
            return int(self.collection.count_documents({}))
        except PyMongoError as exc:
            logger.error(f"Error counting memories: {exc}")
            return 0


def get_memory_content_for_prompt(config: Dict[str, Any]) -> str:
    """
    Convenience function to get memory content for system prompts.
    
    Args:
        config: Application configuration dictionary
        
    Returns:
        Formatted memory content string for inclusion in system prompt
    """
    memory_config = config.get('memory', {})
    
    # Check if memory system is enabled
    if not memory_config.get('enabled', False):
        logger.debug("Memory system is disabled")
        return ""
    
    # Get configuration values
    count = memory_config.get('recent_memories_count', 10)
    format_config = memory_config.get('format', {})
    mongo_overrides = memory_config.get('mongo') if isinstance(memory_config.get('mongo'), dict) else None

    reader = MemoryReader(mongo_overrides)

    if not reader.check_database_exists():
        logger.warning("Memory collection is not available")
        return ""
    
    # Get memory count
    total_memories = reader.get_memory_count()
    if total_memories == 0:
        logger.debug("No memories found in database")
        return ""
    
    logger.info(f"Found {total_memories} total memories, retrieving {min(count, total_memories)} recent ones")
    
    # Get formatted memory content
    content = reader.get_formatted_recent_memories(count, format_config)
    
    if content:
        # Add a clear header to the memory section
        count_text = f"{min(count, total_memories)} most recent" if count < total_memories else "all"
        header = f"\n\n=== MEMORY SYSTEM ===\nThe following are your {count_text} memories from previous conversations:\n"
        return header + content
    
    return ""


if __name__ == "__main__":
    # Test the memory reader
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Testing memory reader...")
        
        # Test with sample config
        test_config = {
            'memory': {
                'enabled': True,
                'recent_memories_count': 10,
                'mongo': {},
                'format': {
                    'include_timestamps': True,
                    'include_categories': True,
                    'max_content_length': 100,
                    'separator': '\n- '
                }
            }
        }
        
        content = get_memory_content_for_prompt(test_config)
        if content:
            print("Memory content:")
            print(content)
        else:
            print("No memory content generated")
            
    else:
        print("Usage:")
        print("  python memory_reader.py test   - Test the memory reader")
