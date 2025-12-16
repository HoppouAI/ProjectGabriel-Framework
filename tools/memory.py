"""
Memory System
Provides a simple but effective memory system using MongoDB & SQLite.
Supports three memory types:
- Long-term: Permanent memories
- Short-term: Auto-deleted after 7 days
- Quick notes: Auto-deleted after 6 hours
12/16/2025 - added SQLite support as an alternative backend.
"""

import logging
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from google.genai import types
try:
    from pymongo import ASCENDING, DESCENDING, MongoClient
    from pymongo.collection import Collection, ReturnDocument
    from pymongo.errors import PyMongoError
except Exception:
    ASCENDING = None
    DESCENDING = None
    MongoClient = None
    Collection = None
    ReturnDocument = None
    PyMongoError = Exception
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

# Set up logging
logger = logging.getLogger(__name__)

# Memory type constants
MEMORY_TYPE_LONG_TERM = "long_term"
MEMORY_TYPE_SHORT_TERM = "short_term"
MEMORY_TYPE_QUICK_NOTE = "quick_note"

def _load_config_file() -> Dict[str, Any]:
    config_path = os.path.join(os.getcwd(), "config.yml")
    if not yaml or not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.debug(f"Config load failed; using defaults: {exc}")
    return {}


def _get_memory_config() -> Dict[str, Any]:
    config = _load_config_file()
    memory_cfg = config.get("memory")
    if isinstance(memory_cfg, dict):
        return memory_cfg
    return {}


def _get_memory_backend() -> str:
    cfg = _get_memory_config()
    backend = cfg.get("backend") or cfg.get("storage") or "mongo"
    try:
        backend = str(backend).strip().lower()
    except Exception:
        backend = "mongo"
    if backend in {"mongodb", "mongo"}:
        return "mongo"
    if backend in {"sqlite", "sqlite3"}:
        return "sqlite"
    return "mongo"


def _get_sqlite_config() -> Dict[str, Any]:
    cfg = _get_memory_config()
    sqlite_cfg = cfg.get("sqlite")
    if isinstance(sqlite_cfg, dict):
        return sqlite_cfg
    return {}


def get_mongo_connection_settings(overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "uri_env_var": "GABRIEL_MONGO_URI",
        "password_env_var": "GABRIEL_MONGO_PASSWORD",
        "username_env_var": "GABRIEL_MONGO_USERNAME",
        "database_env_var": "GABRIEL_MONGO_DB",
        "collection_env_var": "GABRIEL_MONGO_COLLECTION",
        "database": "gabriel",
        "collection": "memories",
        "options": "retryWrites=true&w=majority",
        "username": "gabriel_hoppouai_db",
    }
    mongo_cfg = _get_memory_config().get("mongo")
    if isinstance(mongo_cfg, dict):
        defaults.update({k: v for k, v in mongo_cfg.items() if v is not None})
    if overrides:
        defaults.update({k: v for k, v in overrides.items() if v is not None})

    uri_env_var = defaults.get("uri_env_var", "GABRIEL_MONGO_URI")
    uri = os.environ.get(uri_env_var, "") or defaults.get("uri", "")
    if not uri:
        username = os.environ.get(defaults.get("username_env_var", "GABRIEL_MONGO_USERNAME"), "") or defaults.get("username", "")
        password = os.environ.get(defaults.get("password_env_var", "GABRIEL_MONGO_PASSWORD"), "")
        host = defaults.get("host", "")
        options = defaults.get("options", "")
        database_name = defaults.get("database", "")
        if username and password and host:
            option_suffix = ""
            if options:
                option_suffix = options if options.startswith("?") else f"?{options}"
            path_suffix = f"/{database_name}" if database_name else ""
            uri = f"mongodb+srv://{username}:{password}@{host}{path_suffix}{option_suffix}"

    db_env_var = defaults.get("database_env_var", "GABRIEL_MONGO_DB")
    collection_env_var = defaults.get("collection_env_var", "GABRIEL_MONGO_COLLECTION")
    db_override = os.environ.get(db_env_var)
    collection_override = os.environ.get(collection_env_var)
    if db_override:
        defaults["database"] = db_override
    if collection_override:
        defaults["collection"] = collection_override

    defaults["uri"] = uri
    return defaults


class MemorySystem:
    def __init__(self, mongo_uri: Optional[str] = None, database: Optional[str] = None, collection: Optional[str] = None):
        self.backend = _get_memory_backend()
        overrides: Dict[str, Any] = {}
        if mongo_uri is not None:
            overrides["uri"] = mongo_uri
        if database is not None:
            overrides["database"] = database
        if collection is not None:
            overrides["collection"] = collection
        self.settings = get_mongo_connection_settings(overrides if overrides else None)
        self.client: Optional[MongoClient] = None
        self.collection: Optional[Collection] = None
        self.sqlite_conn: Optional[sqlite3.Connection] = None
        self.sqlite_path: Optional[str] = None
        self._sqlite_lock = threading.RLock()
        self.cleanup_thread: Optional[threading.Thread] = None
        self.cleanup_running = False
        self._connect()
        if self._is_storage_ready():
            self.start_cleanup_thread()

    def __del__(self):
        self.close()

    def close(self):
        self.stop_cleanup_thread()
        if self.sqlite_conn is not None:
            try:
                self.sqlite_conn.close()
            except Exception as exc:
                logger.debug(f"SQLite close during shutdown: {exc}")
            finally:
                self.sqlite_conn = None
        if self.client is not None:
            try:
                self.client.close()
            except Exception as exc:
                logger.debug(f"Mongo client close during shutdown: {exc}")
            finally:
                self.client = None

    def _connect(self):
        backend = _get_memory_backend()
        self.backend = backend
        if backend == "sqlite":
            self._connect_sqlite()
            return
        self._connect_mongo()

    def _connect_mongo(self):
        uri = self.settings.get("uri") or ""
        if not uri:
            logger.error("MongoDB URI is not configured; memory system disabled")
            return
        if MongoClient is None:
            logger.error("pymongo is not available; cannot use mongo memory backend")
            return
        try:
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command("ping")
            database_name = self.settings.get("database") or "gabriel"
            collection_name = self.settings.get("collection") or "memories"
            self.collection = self.client[database_name][collection_name]
            self.init_collection()
            logger.info(f"Memory system connected to MongoDB at {database_name}.{collection_name}")
        except Exception as exc:
            logger.error(f"Failed to initialize MongoDB memory storage: {exc}")
            self.collection = None

    def _connect_sqlite(self):
        try:
            sqlite_cfg = _get_sqlite_config()
            path = sqlite_cfg.get("path") or sqlite_cfg.get("file") or "gabriel_memories.sqlite"
            path = str(path)
            if not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)
            parent = os.path.dirname(path)
            if parent and not os.path.exists(parent):
                try:
                    os.makedirs(parent, exist_ok=True)
                except Exception:
                    pass
            with self._sqlite_lock:
                self.sqlite_path = path
                self.sqlite_conn = sqlite3.connect(path, check_same_thread=False)
                self.sqlite_conn.row_factory = sqlite3.Row
                try:
                    self.sqlite_conn.execute("PRAGMA journal_mode=WAL")
                except Exception:
                    pass
                self.init_collection()
            logger.info(f"Memory system connected to SQLite at {path}")
        except Exception as exc:
            logger.error(f"Failed to initialize SQLite memory storage: {exc}")
            self.sqlite_conn = None
            self.sqlite_path = None

    def init_collection(self):
        if self.backend == "sqlite":
            conn = self.sqlite_conn
            if conn is None:
                return
            with self._sqlite_lock:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS memories ("
                    "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                    "key TEXT NOT NULL UNIQUE,"
                    "content TEXT NOT NULL,"
                    "category TEXT NOT NULL,"
                    "memory_type TEXT NOT NULL,"
                    "tags_json TEXT NOT NULL,"
                    "content_hash TEXT NOT NULL,"
                    "created_at TEXT NOT NULL,"
                    "updated_at TEXT,"
                    "access_count INTEGER NOT NULL DEFAULT 0"
                    ")"
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON memories(category)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type_created ON memories(memory_type, created_at)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON memories(content_hash)")
                conn.commit()
            return
        if self.collection is None:
            return
        try:
            self.collection.create_index([("key", ASCENDING)], unique=True, name="idx_key_unique")
            self.collection.create_index([("category", ASCENDING)], name="idx_category")
            self.collection.create_index([("created_at", DESCENDING)], name="idx_created_at")
            self.collection.create_index([("memory_type", ASCENDING)], name="idx_memory_type")
            self.collection.create_index([("memory_type", ASCENDING), ("created_at", DESCENDING)], name="idx_memory_type_created")
            self.collection.create_index([("content_hash", ASCENDING)], name="idx_content_hash")
        except PyMongoError as exc:
            logger.error(f"Failed to ensure memory indexes: {exc}")

    def _is_storage_ready(self) -> bool:
        if self.backend == "sqlite":
            return self.sqlite_conn is not None
        return self.collection is not None

    def is_available(self) -> bool:
        return self._ensure_collection()

    def _ensure_collection(self) -> bool:
        if self.backend == "sqlite":
            if self.sqlite_conn is not None:
                return True
            self._connect()
            return self.sqlite_conn is not None
        if self.collection is not None:
            return True
        self._connect()
        return self.collection is not None
    
    def start_cleanup_thread(self):
        if not self._is_storage_ready():
            return
        if self.cleanup_running:
            return
        self.cleanup_running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self.cleanup_thread.start()
        logger.info("Memory cleanup thread started")

    def stop_cleanup_thread(self):
        was_running = self.cleanup_running
        self.cleanup_running = False
        thread = self.cleanup_thread
        if thread and thread.is_alive():
            thread.join()
        self.cleanup_thread = None
        if was_running:
            logger.info("Memory cleanup thread stopped")

    def _cleanup_worker(self):
        while self.cleanup_running:
            try:
                if self._is_storage_ready():
                    self.cleanup_expired_memories()
                time.sleep(600)
            except Exception as exc:
                logger.error(f"Error in cleanup worker: {exc}")
                time.sleep(60)
    
    def cleanup_expired_memories(self) -> Dict[str, Any]:
        if not self._ensure_collection():
            return {"error": "Memory storage unavailable"}
        try:
            now = datetime.utcnow()
            note_cfg = _get_note_config()
            ttl_hours = float(note_cfg.get("ttl_hours", 6))
            quick_cutoff = now - timedelta(hours=ttl_hours)
            st_cfg = _get_short_term_config()
            ttl_days = float(st_cfg.get("ttl_days", 7))
            short_cutoff = now - timedelta(days=ttl_days)
            quick_deleted = 0
            short_deleted = 0
            if self.backend == "sqlite":
                conn = self.sqlite_conn
                if conn is None:
                    return {"error": "Memory storage unavailable"}
                quick_iso = quick_cutoff.isoformat()
                short_iso = short_cutoff.isoformat()
                with self._sqlite_lock:
                    qr = conn.execute(
                        "DELETE FROM memories WHERE memory_type = ? AND created_at < ? AND (tags_json NOT LIKE ?)",
                        (MEMORY_TYPE_QUICK_NOTE, quick_iso, '%"pinned"%')
                    )
                    quick_deleted = int(qr.rowcount or 0)
                    sr = conn.execute(
                        "DELETE FROM memories WHERE memory_type = ? AND created_at < ? AND (tags_json NOT LIKE ?)",
                        (MEMORY_TYPE_SHORT_TERM, short_iso, '%"pinned"%')
                    )
                    short_deleted = int(sr.rowcount or 0)
                    conn.commit()
            else:
                quick_filter = {
                    "memory_type": MEMORY_TYPE_QUICK_NOTE,
                    "created_at": {"$lt": quick_cutoff},
                    "tags": {"$nin": ["pinned"]}
                }
                short_filter = {
                    "memory_type": MEMORY_TYPE_SHORT_TERM,
                    "created_at": {"$lt": short_cutoff},
                    "tags": {"$nin": ["pinned"]}
                }
                quick_result = self.collection.delete_many(quick_filter)
                short_result = self.collection.delete_many(short_filter)
                quick_deleted = quick_result.deleted_count if quick_result else 0
                short_deleted = short_result.deleted_count if short_result else 0
            if quick_deleted or short_deleted:
                logger.info(f"Cleanup completed: {quick_deleted} quick notes, {short_deleted} short-term memories deleted")
            return {
                "quick_notes_deleted": quick_deleted,
                "short_term_deleted": short_deleted
            }
        except Exception as exc:
            logger.error(f"Error during cleanup: {exc}")
            return {"error": str(exc)}
    
    def save_memory(self, key: str, content: str, category: str = "general", memory_type: str = MEMORY_TYPE_LONG_TERM, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        if not self._ensure_collection():
            return {"success": False, "message": "Memory storage unavailable"}
        valid_types = [MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM, MEMORY_TYPE_QUICK_NOTE]
        if memory_type not in valid_types:
            return {
                "success": False,
                "message": f"Invalid memory type. Must be one of: {', '.join(valid_types)}"
            }
        tags_list = list(tags) if tags else []
        chash = _hash_text(str(content))
        now = datetime.utcnow()
        try:
            if self.backend == "sqlite":
                import json
                conn = self.sqlite_conn
                if conn is None:
                    return {"success": False, "message": "Memory storage unavailable"}
                now_iso = now.isoformat()
                tags_json = json.dumps(tags_list, ensure_ascii=False)
                with self._sqlite_lock:
                    conn.execute(
                        "INSERT INTO memories(key, content, category, memory_type, tags_json, content_hash, created_at, updated_at, access_count) "
                        "VALUES(?, ?, ?, ?, ?, ?, ?, ?, 0) "
                        "ON CONFLICT(key) DO UPDATE SET "
                        "content=excluded.content, "
                        "category=excluded.category, "
                        "memory_type=excluded.memory_type, "
                        "tags_json=excluded.tags_json, "
                        "content_hash=excluded.content_hash, "
                        "updated_at=excluded.updated_at",
                        (key, content, category, memory_type, tags_json, chash, now_iso, now_iso)
                    )
                    row = conn.execute("SELECT id FROM memories WHERE key = ?", (key,)).fetchone()
                    conn.commit()
                mid = str(row[0]) if row else ""
                logger.info(f"Memory saved: {key} (type: {memory_type})")
                return {
                    "success": True,
                    "message": f"Memory '{key}' saved successfully as {memory_type}",
                    "id": mid,
                    "key": key,
                    "memory_type": memory_type
                }
            doc = self.collection.find_one_and_update(
                {"key": key},
                {
                    "$set": {
                        "content": content,
                        "category": category,
                        "memory_type": memory_type,
                        "tags": tags_list,
                        "content_hash": chash,
                        "updated_at": now,
                    },
                    "$setOnInsert": {
                        "created_at": now,
                        "access_count": 0,
                    },
                },
                upsert=True,
                return_document=ReturnDocument.AFTER,
            )
            if doc is None:
                return {"success": False, "message": "Failed to persist memory"}
            logger.info(f"Memory saved: {key} (type: {memory_type})")
            return {
                "success": True,
                "message": f"Memory '{key}' saved successfully as {memory_type}",
                "id": str(doc.get("_id")),
                "key": key,
                "memory_type": memory_type
            }
        except Exception as exc:
            logger.error(f"Error saving memory: {exc}")
            return {
                "success": False,
                "message": f"Failed to save memory: {str(exc)}"
            }

    def has_recent_duplicate(self, content_hash: str, window_seconds: float, types: Optional[List[str]] = None) -> bool:
        if not self._ensure_collection():
            return False
        try:
            since = datetime.utcnow() - timedelta(seconds=window_seconds)
            if self.backend == "sqlite":
                conn = self.sqlite_conn
                if conn is None:
                    return False
                since_iso = since.isoformat()
                with self._sqlite_lock:
                    if types:
                        placeholders = ",".join(["?"] * len(types))
                        row = conn.execute(
                            f"SELECT id FROM memories WHERE content_hash = ? AND created_at > ? AND memory_type IN ({placeholders}) LIMIT 1",
                            (content_hash, since_iso, *types)
                        ).fetchone()
                    else:
                        row = conn.execute(
                            "SELECT id FROM memories WHERE content_hash = ? AND created_at > ? LIMIT 1",
                            (content_hash, since_iso)
                        ).fetchone()
                return row is not None
            query: Dict[str, Any] = {
                "content_hash": content_hash,
                "created_at": {"$gt": since},
            }
            if types:
                query["memory_type"] = {"$in": types}
            doc = self.collection.find_one(query, {"_id": 1})
            return doc is not None
        except Exception:
            return False
    
    def read_memory(self, key: str) -> Dict[str, Any]:
        if not self._ensure_collection():
            return {"success": False, "message": "Memory storage unavailable"}
        try:
            if self.backend == "sqlite":
                conn = self.sqlite_conn
                if conn is None:
                    return {"success": False, "message": "Memory storage unavailable"}
                import json
                with self._sqlite_lock:
                    row = conn.execute(
                        "SELECT id, key, content, category, memory_type, tags_json, created_at, updated_at, access_count FROM memories WHERE key = ?",
                        (key,)
                    ).fetchone()
                    if not row:
                        return {"success": False, "message": f"Memory '{key}' not found"}
                    conn.execute("UPDATE memories SET access_count = access_count + 1 WHERE key = ?", (key,))
                    conn.commit()
                    try:
                        tags = json.loads(row["tags_json"]) if row["tags_json"] else []
                    except Exception:
                        tags = []
                    memory = {
                        "id": str(row["id"]),
                        "key": row["key"],
                        "content": row["content"],
                        "category": row["category"],
                        "memory_type": row["memory_type"],
                        "tags": tags,
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "access_count": int(row["access_count"] or 0) + 1,
                    }
                logger.info(f"Memory read: {key} (type: {memory['memory_type']})")
                return {"success": True, "memory": memory}
            doc = self.collection.find_one_and_update(
                {"key": key},
                {"$inc": {"access_count": 1}},
                return_document=ReturnDocument.AFTER,
            )
            if doc:
                memory = self._format_memory_doc(doc)
                logger.info(f"Memory read: {key} (type: {memory['memory_type']})")
                return {
                    "success": True,
                    "memory": memory
                }
            return {
                "success": False,
                "message": f"Memory '{key}' not found"
            }
        except Exception as exc:
            logger.error(f"Error reading memory: {exc}")
            return {
                "success": False,
                "message": f"Failed to read memory: {str(exc)}"
            }
    
    def update_memory(self, key: str, content: Optional[str] = None, category: Optional[str] = None, memory_type: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        if not self._ensure_collection():
            return {"success": False, "message": "Memory storage unavailable"}
        try:
            if self.backend == "sqlite":
                conn = self.sqlite_conn
                if conn is None:
                    return {"success": False, "message": "Memory storage unavailable"}
                with self._sqlite_lock:
                    row = conn.execute("SELECT id FROM memories WHERE key = ?", (key,)).fetchone()
                    if not row:
                        return {"success": False, "message": f"Memory '{key}' not found"}
                if memory_type is not None:
                    valid_types = [MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM, MEMORY_TYPE_QUICK_NOTE]
                    if memory_type not in valid_types:
                        return {
                            "success": False,
                            "message": f"Invalid memory type. Must be one of: {', '.join(valid_types)}"
                        }
                updates: Dict[str, Any] = {}
                if content is not None:
                    updates["content"] = content
                    updates["content_hash"] = _hash_text(str(content))
                if category is not None:
                    updates["category"] = category
                if memory_type is not None:
                    updates["memory_type"] = memory_type
                if tags is not None:
                    import json
                    updates["tags_json"] = json.dumps(list(tags), ensure_ascii=False)
                if updates:
                    updates["updated_at"] = datetime.utcnow().isoformat()
                    set_sql = ",".join([f"{k} = ?" for k in updates.keys()])
                    values = list(updates.values())
                    values.append(key)
                    with self._sqlite_lock:
                        conn.execute(f"UPDATE memories SET {set_sql} WHERE key = ?", values)
                        conn.commit()
                logger.info(f"Memory updated: {key}")
                return {"success": True, "message": f"Memory '{key}' updated successfully"}

            existing = self.collection.find_one({"key": key})
            if not existing:
                return {
                    "success": False,
                    "message": f"Memory '{key}' not found"
                }
            if memory_type is not None:
                valid_types = [MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM, MEMORY_TYPE_QUICK_NOTE]
                if memory_type not in valid_types:
                    return {
                        "success": False,
                        "message": f"Invalid memory type. Must be one of: {', '.join(valid_types)}"
                    }
            updates: Dict[str, Any] = {}
            if content is not None:
                updates["content"] = content
                updates["content_hash"] = _hash_text(str(content))
            if category is not None:
                updates["category"] = category
            if memory_type is not None:
                updates["memory_type"] = memory_type
            if tags is not None:
                updates["tags"] = list(tags)
            if updates:
                updates["updated_at"] = datetime.utcnow()
                self.collection.update_one({"key": key}, {"$set": updates})
            logger.info(f"Memory updated: {key}")
            return {
                "success": True,
                "message": f"Memory '{key}' updated successfully"
            }
        except Exception as exc:
            logger.error(f"Error updating memory: {exc}")
            return {
                "success": False,
                "message": f"Failed to update memory: {str(exc)}"
            }
    
    def delete_memory(self, key: str) -> Dict[str, Any]:
        if not self._ensure_collection():
            return {"success": False, "message": "Memory storage unavailable"}
        try:
            if self.backend == "sqlite":
                conn = self.sqlite_conn
                if conn is None:
                    return {"success": False, "message": "Memory storage unavailable"}
                with self._sqlite_lock:
                    cur = conn.execute("DELETE FROM memories WHERE key = ?", (key,))
                    deleted = int(cur.rowcount or 0)
                    conn.commit()
                if deleted:
                    logger.info(f"Memory deleted: {key}")
                    return {"success": True, "message": f"Memory '{key}' deleted successfully"}
                return {"success": False, "message": f"Memory '{key}' not found"}
            result = self.collection.delete_one({"key": key})
            if result.deleted_count:
                logger.info(f"Memory deleted: {key}")
                return {
                    "success": True,
                    "message": f"Memory '{key}' deleted successfully"
                }
            return {
                "success": False,
                "message": f"Memory '{key}' not found"
            }
        except Exception as exc:
            logger.error(f"Error deleting memory: {exc}")
            return {
                "success": False,
                "message": f"Failed to delete memory: {str(exc)}"
            }
    
    def list_memories(self, category: Optional[str] = None, memory_type: Optional[str] = None, limit: int = 50) -> Dict[str, Any]:
        if not self._ensure_collection():
            return {"success": False, "message": "Memory storage unavailable"}
        try:
            if self.backend == "sqlite":
                import json
                conn = self.sqlite_conn
                if conn is None:
                    return {"success": False, "message": "Memory storage unavailable"}
                wh = []
                params: List[Any] = []
                if category:
                    wh.append("category = ?")
                    params.append(category)
                if memory_type:
                    wh.append("memory_type = ?")
                    params.append(memory_type)
                where_sql = (" WHERE " + " AND ".join(wh)) if wh else ""
                lim = max(1, int(limit))
                sql = (
                    "SELECT id, key, content, category, memory_type, tags_json, created_at, updated_at, access_count "
                    "FROM memories" + where_sql + " ORDER BY COALESCE(updated_at, created_at) DESC LIMIT ?"
                )
                params2 = params + [lim]
                memories: List[Dict[str, Any]] = []
                with self._sqlite_lock:
                    rows = conn.execute(sql, params2).fetchall()
                for row in rows:
                    content = row["content"] or ""
                    if len(content) > 200:
                        content = content[:200] + "..."
                    try:
                        tags = json.loads(row["tags_json"]) if row["tags_json"] else []
                    except Exception:
                        tags = []
                    memories.append({
                        "id": str(row["id"]),
                        "key": row["key"],
                        "content": content,
                        "category": row["category"] or "general",
                        "memory_type": row["memory_type"] or MEMORY_TYPE_LONG_TERM,
                        "tags": tags,
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "access_count": int(row["access_count"] or 0),
                    })
                return {"success": True, "memories": memories, "count": len(memories)}

            filters: Dict[str, Any] = {}
            if category:
                filters["category"] = category
            if memory_type:
                filters["memory_type"] = memory_type
            cursor = self.collection.find(
                filters,
                {
                    "_id": 1,
                    "key": 1,
                    "content": 1,
                    "category": 1,
                    "memory_type": 1,
                    "tags": 1,
                    "created_at": 1,
                    "updated_at": 1,
                    "access_count": 1,
                },
            ).sort([("updated_at", DESCENDING)]).limit(limit)
            memories = []
            for doc in cursor:
                content = doc.get("content", "")
                if len(content) > 200:
                    content = content[:200] + "..."
                memories.append({
                    "id": str(doc.get("_id")),
                    "key": doc.get("key"),
                    "content": content,
                    "category": doc.get("category", "general"),
                    "memory_type": doc.get("memory_type", MEMORY_TYPE_LONG_TERM),
                    "tags": doc.get("tags") or [],
                    "created_at": self._serialize_datetime(doc.get("created_at")),
                    "updated_at": self._serialize_datetime(doc.get("updated_at")),
                    "access_count": doc.get("access_count", 0)
                })
            return {
                "success": True,
                "memories": memories,
                "count": len(memories)
            }
        except Exception as exc:
            logger.error(f"Error listing memories: {exc}")
            return {
                "success": False,
                "message": f"Failed to list memories: {str(exc)}"
            }
    
    def search_memories(self, search_term: str, memory_type: Optional[str] = None, limit: int = 20) -> Dict[str, Any]:
        if not self._ensure_collection():
            return {"success": False, "message": "Memory storage unavailable"}
        try:
            if self.backend == "sqlite":
                conn = self.sqlite_conn
                if conn is None:
                    return {"success": False, "message": "Memory storage unavailable"}
                term = str(search_term)
                like = f"%{term}%"
                wh = ["(key LIKE ? OR content LIKE ?)"]
                params: List[Any] = [like, like]
                if memory_type:
                    wh.append("memory_type = ?")
                    params.append(memory_type)
                lim = max(1, int(limit))
                params.append(lim)
                sql = (
                    "SELECT id, key, content, category, memory_type, created_at, updated_at, access_count "
                    "FROM memories WHERE " + " AND ".join(wh) + " "
                    "ORDER BY access_count DESC, COALESCE(updated_at, created_at) DESC LIMIT ?"
                )
                memories: List[Dict[str, Any]] = []
                with self._sqlite_lock:
                    rows = conn.execute(sql, params).fetchall()
                for row in rows:
                    content = row["content"] or ""
                    if len(content) > 200:
                        content = content[:200] + "..."
                    memories.append({
                        "key": row["key"],
                        "content": content,
                        "category": row["category"] or "general",
                        "memory_type": row["memory_type"] or MEMORY_TYPE_LONG_TERM,
                        "created_at": row["created_at"],
                        "updated_at": row["updated_at"],
                        "access_count": int(row["access_count"] or 0)
                    })
                return {"success": True, "memories": memories, "count": len(memories), "search_term": search_term}
            pattern = re.escape(search_term)
            regex = {"$regex": pattern, "$options": "i"}
            query: Dict[str, Any] = {
                "$or": [
                    {"key": regex},
                    {"content": regex},
                ]
            }
            if memory_type:
                query["memory_type"] = memory_type
            cursor = self.collection.find(
                query,
                {
                    "key": 1,
                    "content": 1,
                    "category": 1,
                    "memory_type": 1,
                    "created_at": 1,
                    "updated_at": 1,
                    "access_count": 1,
                },
            ).sort([("access_count", DESCENDING), ("updated_at", DESCENDING)]).limit(limit)
            memories = []
            for doc in cursor:
                content = doc.get("content", "")
                if len(content) > 200:
                    content = content[:200] + "..."
                memories.append({
                    "key": doc.get("key"),
                    "content": content,
                    "category": doc.get("category", "general"),
                    "memory_type": doc.get("memory_type", MEMORY_TYPE_LONG_TERM),
                    "created_at": self._serialize_datetime(doc.get("created_at")),
                    "updated_at": self._serialize_datetime(doc.get("updated_at")),
                    "access_count": doc.get("access_count", 0)
                })
            return {
                "success": True,
                "memories": memories,
                "count": len(memories),
                "search_term": search_term
            }
        except Exception as exc:
            logger.error(f"Error searching memories: {exc}")
            return {
                "success": False,
                "message": f"Failed to search memories: {str(exc)}"
            }
    
    def get_memory_stats(self) -> Dict[str, Any]:
        if not self._ensure_collection():
            return {"success": False, "message": "Memory storage unavailable"}
        try:
            if self.backend == "sqlite":
                conn = self.sqlite_conn
                if conn is None:
                    return {"success": False, "message": "Memory storage unavailable"}
                stats = {
                    "total": 0,
                    "long_term": 0,
                    "short_term": 0,
                    "quick_note": 0
                }
                with self._sqlite_lock:
                    rows = conn.execute("SELECT memory_type, COUNT(1) AS c FROM memories GROUP BY memory_type").fetchall()
                for row in rows:
                    mt = row[0] or "unknown"
                    c = int(row[1] or 0)
                    stats[mt] = c
                    stats["total"] += c
                return {"success": True, "stats": stats}
            pipeline = [
                {"$group": {"_id": "$memory_type", "count": {"$sum": 1}}},
            ]
            stats_cursor = self.collection.aggregate(pipeline)
            stats = {
                "total": 0,
                "long_term": 0,
                "short_term": 0,
                "quick_note": 0
            }
            for row in stats_cursor:
                memory_type = row.get("_id") or "unknown"
                count = row.get("count", 0)
                stats[memory_type] = count
                stats["total"] += count
            return {
                "success": True,
                "stats": stats
            }
        except Exception as exc:
            logger.error(f"Error getting memory stats: {exc}")
            return {
                "success": False,
                "message": f"Failed to get memory stats: {str(exc)}"
            }

    def get_memory_count(self) -> int:
        if not self._ensure_collection():
            return 0
        try:
            if self.backend == "sqlite":
                conn = self.sqlite_conn
                if conn is None:
                    return 0
                with self._sqlite_lock:
                    row = conn.execute("SELECT COUNT(1) FROM memories").fetchone()
                return int(row[0] if row else 0)
            return int(self.collection.count_documents({}))
        except Exception:
            return 0

    def get_recent_memories_for_prompt(self, count: int = 10) -> List[Dict[str, Any]]:
        if not self._ensure_collection():
            return []
        try:
            real_limit = max(1, int(int(count) * 0.8))
            note_limit = max(0, int(count) - real_limit)
            if self.backend == "sqlite":
                import json
                conn = self.sqlite_conn
                if conn is None:
                    return []
                memories: List[Dict[str, Any]] = []
                with self._sqlite_lock:
                    rows_real = conn.execute(
                        "SELECT id, key, content, category, created_at, tags_json FROM memories "
                        "WHERE memory_type IN (?, ?) ORDER BY created_at DESC LIMIT ?",
                        (MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM, real_limit)
                    ).fetchall()
                    rows_note: List[sqlite3.Row] = []
                    if note_limit:
                        rows_note = conn.execute(
                            "SELECT id, key, content, category, created_at, tags_json FROM memories "
                            "WHERE memory_type = ? ORDER BY created_at DESC LIMIT ?",
                            (MEMORY_TYPE_QUICK_NOTE, note_limit)
                        ).fetchall()
                for row in list(rows_real) + list(rows_note):
                    try:
                        tags_list = json.loads(row["tags_json"]) if row["tags_json"] else []
                    except Exception:
                        tags_list = []
                    tags_str = ",".join(tags_list) if isinstance(tags_list, list) else str(tags_list)
                    memories.append({
                        "id": str(row["id"]),
                        "key": row["key"],
                        "content": row["content"],
                        "category": row["category"] or "general",
                        "created_at": row["created_at"] or "",
                        "tags": tags_str
                    })
                return memories
            docs: List[Dict[str, Any]] = []
            real_docs = list(
                self.collection.find(
                    {"memory_type": {"$in": [MEMORY_TYPE_LONG_TERM, MEMORY_TYPE_SHORT_TERM]}},
                    {"key": 1, "content": 1, "category": 1, "created_at": 1, "tags": 1},
                ).sort("created_at", DESCENDING).limit(real_limit)
            )
            note_docs: List[Dict[str, Any]] = []
            if note_limit:
                note_docs = list(
                    self.collection.find(
                        {"memory_type": MEMORY_TYPE_QUICK_NOTE},
                        {"key": 1, "content": 1, "category": 1, "created_at": 1, "tags": 1},
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
            return memories
        except Exception:
            return []

    def _format_memory_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": str(doc.get("_id")),
            "key": doc.get("key"),
            "content": doc.get("content"),
            "category": doc.get("category", "general"),
            "memory_type": doc.get("memory_type", MEMORY_TYPE_LONG_TERM),
            "tags": doc.get("tags") or [],
            "created_at": self._serialize_datetime(doc.get("created_at")),
            "updated_at": self._serialize_datetime(doc.get("updated_at")),
            "access_count": doc.get("access_count", 0)
        }

    @staticmethod
    def _serialize_datetime(value: Any) -> Optional[str]:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str):
            return value
        return None

# Function declarations for Gemini Live API
MEMORY_FUNCTION_DECLARATIONS = [
    {
        "name": "save_memory",
        "description": "Save a new memory or update an existing one. Choose the appropriate memory type: 'long_term' for permanent memories, 'short_term' for memories that auto-delete after 7 days, or 'quick_note' for temporary notes that clear after 6 hours.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "A unique identifier for the memory (e.g., 'user_name', 'favorite_color', 'project_details')"
                },
                "content": {
                    "type": "string",
                    "description": "The content to remember"
                },
                "category": {
                    "type": "string",
                    "description": "Category for organizing memories (e.g., 'personal', 'work', 'preferences')",
                    "default": "general"
                },
                "memory_type": {
                    "type": "string",
                    "description": "Type of memory: 'long_term' (permanent), 'short_term' (7 days), or 'quick_note' (6 hours)",
                    "enum": ["long_term", "short_term", "quick_note"],
                    "default": "long_term"
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tags for better organization"
                }
            },
            "required": ["key", "content"]
        }
    },
    {
        "name": "promote_memory",
        "description": "Promote a memory to a higher persistence type.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "The memory key to promote"},
                "new_type": {"type": "string", "enum": ["short_term", "long_term"], "description": "Target type"}
            },
            "required": ["key", "new_type"]
        }
    },
    {
        "name": "pin_memory",
        "description": "Pin or unpin a memory to prevent cleanup.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "pin": {"type": "boolean", "default": True}
            },
            "required": ["key"]
        }
    },
    {
        "name": "read_memory",
        "description": "Read a specific memory by its key. Use this to recall previously stored information.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The unique identifier of the memory to read"
                }
            },
            "required": ["key"]
        }
    },
    {
        "name": "update_memory",
        "description": "Update an existing memory with new information.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The unique identifier of the memory to update"
                },
                "content": {
                    "type": "string",
                    "description": "New content for the memory"
                },
                "category": {
                    "type": "string",
                    "description": "New category for the memory"
                },
                "memory_type": {
                    "type": "string",
                    "description": "New memory type: 'long_term', 'short_term', or 'quick_note'",
                    "enum": ["long_term", "short_term", "quick_note"]
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "New tags for the memory"
                }
            },
            "required": ["key"]
        }
    },
    {
        "name": "delete_memory",
        "description": "Delete a memory permanently. Use with caution.",
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The unique identifier of the memory to delete"
                }
            },
            "required": ["key"]
        }
    },
    {
        "name": "list_memories",
        "description": "List all memories or memories filtered by category and/or memory type.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Optional category to filter memories"
                },
                "memory_type": {
                    "type": "string",
                    "description": "Optional memory type to filter: 'long_term', 'short_term', or 'quick_note'",
                    "enum": ["long_term", "short_term", "quick_note"]
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of memories to return",
                    "default": 50
                }
            }
        }
    },
    {
        "name": "search_memories",
        "description": "Search through memories by content or key, optionally filtered by memory type.",
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "type": "string",
                    "description": "The term to search for in memory keys and content"
                },
                "memory_type": {
                    "type": "string",
                    "description": "Optional memory type to filter: 'long_term', 'short_term', or 'quick_note'",
                    "enum": ["long_term", "short_term", "quick_note"]
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 20
                }
            },
            "required": ["search_term"]
        }
    },
    {
        "name": "get_memory_stats",
        "description": "Get statistics about memory usage by type.",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "cleanup_expired_memories",
        "description": "Manually trigger cleanup of expired memories (quick notes older than 6 hours, short-term memories older than 7 days).",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    }
]

async def handle_memory_function_call(function_call) -> types.FunctionResponse:
    """Handle memory-related function calls."""
    function_name = function_call.name
    args = function_call.args
    
    try:
        if function_name == "save_memory":
            key = args["key"]
            content = args["content"]
            category = args.get("category", "general")

            # Determine memory type with note_* convention
            provided_type = args.get("memory_type")
            if key.startswith("note_") and not provided_type:
                # Use configured default for notes (quick_note by default in config.yml)
                note_cfg = _get_note_config()
                mem_type = str(note_cfg.get("default_type", MEMORY_TYPE_QUICK_NOTE))
            else:
                mem_type = provided_type or MEMORY_TYPE_LONG_TERM

            # Rate limit and de-duplicate note saves even if called directly
            if key.startswith("note_"):
                cfg = _get_note_config()
                if not cfg.get("enabled", True):
                    result = {
                        "success": True,
                        "message": "Notes are disabled by configuration; not saved.",
                        "skipped": True,
                    }
                else:
                    now = time.time()
                    min_gap = float(cfg.get("min_interval_seconds", 120))
                    content_hash = _hash_text(str(content))
                    global _note_last_ts, _note_last_hash
                    if _note_last_ts is None:
                        _note_last_ts = 0.0
                    if now - _note_last_ts < min_gap:
                        result = {
                            "success": True,
                            "message": f"Note rate-limited (wait {int(min_gap - (now - _note_last_ts))}s); not saved.",
                            "skipped": True,
                            "reason": "rate_limited"
                        }
                    else:
                        dedupe_window = float(cfg.get("dedupe_window_seconds", 300))
                        if _note_last_hash == content_hash and now - _note_last_ts < dedupe_window:
                            result = {
                                "success": True,
                                "message": "Duplicate recent note suppressed; not saved.",
                                "skipped": True,
                                "reason": "duplicate"
                            }
                        else:
                            if memory_system.has_recent_duplicate(content_hash, dedupe_window, [MEMORY_TYPE_QUICK_NOTE, MEMORY_TYPE_SHORT_TERM, MEMORY_TYPE_LONG_TERM]):
                                result = {
                                    "success": True,
                                    "message": "Duplicate recent note in database suppressed; not saved.",
                                    "skipped": True,
                                    "reason": "duplicate_db"
                                }
                            else:
                                result = memory_system.save_memory(
                                key=key,
                                content=content,
                                category=category,
                                memory_type=mem_type,
                                tags=args.get("tags") or ["quick_note"]
                            )
                            if result.get("success"):
                                _note_last_ts = now
                                _note_last_hash = content_hash
            else:
                result = memory_system.save_memory(
                    key=key,
                    content=content,
                    category=category,
                    memory_type=mem_type,
                    tags=args.get("tags")
                )
        
        elif function_name == "read_memory":
            result = memory_system.read_memory(args["key"])
        
        elif function_name == "update_memory":
            result = memory_system.update_memory(
                key=args["key"],
                content=args.get("content"),
                category=args.get("category"),
                memory_type=args.get("memory_type"),
                tags=args.get("tags")
            )
        
        elif function_name == "delete_memory":
            result = memory_system.delete_memory(args["key"])
        
        elif function_name == "list_memories":
            result = memory_system.list_memories(
                category=args.get("category"),
                memory_type=args.get("memory_type"),
                limit=args.get("limit", 50)
            )
        
        elif function_name == "search_memories":
            result = memory_system.search_memories(
                search_term=args["search_term"],
                memory_type=args.get("memory_type"),
                limit=args.get("limit", 20)
            )
        
        elif function_name == "get_memory_stats":
            result = memory_system.get_memory_stats()
        
        elif function_name == "cleanup_expired_memories":
            result = memory_system.cleanup_expired_memories()
        
        elif function_name == "promote_memory":
            nt = args["new_type"]
            if nt not in [MEMORY_TYPE_SHORT_TERM, MEMORY_TYPE_LONG_TERM]:
                result = {"success": False, "message": "Invalid new_type"}
            else:
                result = memory_system.update_memory(key=args["key"], memory_type=nt)
        
        elif function_name == "pin_memory":
            pin = bool(args.get("pin", True))
            cur = memory_system.read_memory(args["key"]) or {}
            if not cur.get("success"):
                result = cur
            else:
                mem = cur["memory"]
                tags = mem.get("tags") or []
                if pin and "pinned" not in tags:
                    tags.append("pinned")
                if not pin and "pinned" in tags:
                    tags = [t for t in tags if t != "pinned"]
                result = memory_system.update_memory(key=args["key"], tags=tags)
        
        else:
            result = {
                "success": False,
                "message": f"Unknown function: {function_name}"
            }
        
        return types.FunctionResponse(
            id=function_call.id,
            name=function_name,
            response=result
        )
        
    except Exception as e:
        logger.error(f"Error handling function call {function_name}: {e}")
        return types.FunctionResponse(
            id=function_call.id,
            name=function_name,
            response={
                "success": False,
                "message": f"Error executing {function_name}: {str(e)}"
            }
        )

def get_memory_tools():
    """Get the memory tools configuration for Gemini Live API."""
    return [{"function_declarations": MEMORY_FUNCTION_DECLARATIONS}]

# --- Internal helpers and state ---
_note_last_ts: float | None = None
_note_last_hash: str | None = None

def _hash_text(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()

def _get_note_config() -> Dict[str, Any]:
    defaults = {
        "enabled": True,
        "default_type": MEMORY_TYPE_QUICK_NOTE,
        "ttl_hours": 6.0,
        "min_interval_seconds": 120.0,
        "dedupe_window_seconds": 300.0,
    }
    memory_cfg = _get_memory_config()
    note_cfg = memory_cfg.get("notes") if isinstance(memory_cfg, dict) else {}
    if isinstance(note_cfg, dict):
        merged = defaults.copy()
        for key in defaults:
            if key in note_cfg and note_cfg[key] is not None:
                merged[key] = note_cfg[key]
        return merged
    return defaults


def _get_short_term_config() -> Dict[str, Any]:
    defaults = {
        "ttl_days": 7.0,
    }
    memory_cfg = _get_memory_config()
    short_cfg = memory_cfg.get("short_term") if isinstance(memory_cfg, dict) else {}
    if isinstance(short_cfg, dict):
        merged = defaults.copy()
        for key in defaults:
            if key in short_cfg and short_cfg[key] is not None:
                merged[key] = short_cfg[key]
        return merged
    return defaults


# Global memory system instance
memory_system = MemorySystem()
