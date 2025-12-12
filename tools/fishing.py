import asyncio
import logging
from typing import Any, Dict, Optional

from google.genai import types

logger = logging.getLogger(__name__)

_injector = None
_active_press_tasks: Dict[str, asyncio.Task] = {}
_active_left_label: Optional[str] = None
_queued_cast_duration: Optional[float] = None


def _ensure_injector():
    global _injector
    if _injector is None:
        try:
            import pydirectinput as _pdi  # type: ignore
            _injector = _pdi
            try:
                _pdi.PAUSE = 0.0
            except Exception:
                pass
        except Exception as e:
            raise RuntimeError(f"PyDirectInput not available for mouse inputs: {e}")
    return _injector


def _mouse_down(button: str) -> None:
    inj = _ensure_injector()
    inj.mouseDown(button=button)


def _mouse_up(button: str) -> None:
    inj = _ensure_injector()
    inj.mouseUp(button=button)


async def _press_mouse(button: str, duration: float) -> None:
    try:
        _mouse_down(button)
        await asyncio.sleep(max(0.0, float(duration)))
    finally:
        try:
            _mouse_up(button)
        except Exception as e:
            logger.warning(f"Failed to release mouse {button}: {e}")


def _spawn_unique_press(tag: str, coro_factory, label: Optional[str] = None) -> None:
    global _active_left_label
    existing = _active_press_tasks.get(tag)
    if existing and not existing.done():
        existing.cancel()
    task = asyncio.create_task(coro_factory())
    _active_press_tasks[tag] = task
    if tag == "mouse:left":
        _active_left_label = label

        def _on_done(_):
            global _active_left_label, _queued_cast_duration
            try:
                if _active_press_tasks.get(tag) is task:
                    del _active_press_tasks[tag]
            except Exception:
                pass
            prev_label = _active_left_label
            _active_left_label = None
            if prev_label == "reel":
                dur = _queued_cast_duration
                _queued_cast_duration = None
                if dur is not None:
                    _spawn_unique_press("mouse:left", lambda: _press_mouse("left", float(dur)), label="cast")

        task.add_done_callback(_on_done)


async def vr_fishing_cast(duration: Optional[float] = None, when_reeling: Optional[str] = None) -> Dict[str, Any]:
    dur = float(2.0 if duration is None else duration)
    mode = (when_reeling or "interrupt").strip().lower()
    try:
        existing = _active_press_tasks.get("mouse:left")
        if existing and not existing.done() and _active_left_label == "reel":
            if mode not in {"interrupt", "queue"}:
                mode = "interrupt"
            if mode == "queue":
                global _queued_cast_duration
                _queued_cast_duration = dur
                return {"success": True, "action": "vr_fishing_cast", "queued": True, "after": "reel", "duration": dur}
            else:
                try:
                    existing.cancel()
                    try:
                        await asyncio.wait_for(existing, timeout=0.3)
                    except Exception:
                        pass
                except Exception:
                    pass
        _spawn_unique_press("mouse:left", lambda: _press_mouse("left", dur), label="cast")
        return {"success": True, "action": "vr_fishing_cast", "button": "left", "duration": dur}
    except Exception as e:
        return {"success": False, "message": f"Fishing cast failed: {e}"}


async def vr_fishing_reel(duration: Optional[float] = None) -> Dict[str, Any]:
    dur = float(30.0 if duration is None else duration)
    try:
        _spawn_unique_press("mouse:left", lambda: _press_mouse("left", dur), label="reel")
        return {"success": True, "action": "vr_fishing_reel", "button": "left", "duration": dur}
    except Exception as e:
        return {"success": False, "message": f"Fishing reel failed: {e}"}


def _to_bool(val: Optional[Any]) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in {"true", "1", "yes", "y", "on", "enable", "enabled"}:
            return True
        if s in {"false", "0", "no", "n", "off", "disable", "disabled"}:
            return False
    return False


async def vr_set_fishing_mode(enabled: Optional[Any] = None) -> Dict[str, Any]:
    state = _to_bool(enabled)
    try:
        import idle as idle_mod
        if state:
            idle_mod.stop_idle_gaze()
        else:
            idle_mod.start_idle_gaze()
        return {"success": True, "action": "vr_set_fishing_mode", "fishing_mode": state}
    except Exception as e:
        return {"success": False, "message": f"Setting fishing mode failed: {e}"}


FISHING_FUNCTION_DECLARATIONS = [
    {
        "name": "vr_fishing_cast",
        "description": "Hold left mouse button to cast a fishing line in VRChat. Defaults to 2 seconds.",
        "parameters": {
            "type": "object",
            "properties": {
                "duration": {"type": "number", "description": "Seconds to hold left click", "default": 1.0},
                "when_reeling": {"type": "string", "description": "Behavior if a reeling action is active: 'interrupt' to stop reeling and cast now, or 'queue' to cast after reeling.", "enum": ["interrupt", "queue"], "default": "interrupt"}
            }
        }
    },
    {
        "name": "vr_fishing_reel",
        "description": "Hold left mouse button to reel in after a catch. Defaults to about 15 seconds.",
        "parameters": {
            "type": "object",
            "properties": {
                "duration": {"type": "number", "description": "Seconds to hold left click", "default": 15.0}
            }
        }
    },
    {
        "name": "vr_set_fishing_mode",
        "description": "Enable or disable fishing mode.",
        "parameters": {
            "type": "object",
            "properties": {
                "enabled": {"type": "boolean", "description": "Set to true to enable fishing mode (disables idle gaze)."}
            },
            "required": ["enabled"]
        }
    }
]


async def handle_fishing_function_calls(function_call) -> types.FunctionResponse:
    name = function_call.name
    args = function_call.args or {}

    try:
        if name == "vr_fishing_cast":
            result = await vr_fishing_cast(args.get("duration"), args.get("when_reeling"))
        elif name == "vr_fishing_reel":
            result = await vr_fishing_reel(args.get("duration"))
        elif name == "vr_set_fishing_mode":
            result = await vr_set_fishing_mode(args.get("enabled"))
        else:
            result = {"success": False, "message": f"Unknown fishing function: {name}"}

        return types.FunctionResponse(id=function_call.id, name=name, response=result)
    except Exception as e:
        logger.error(f"Fishing function {name} failed: {e}")
        return types.FunctionResponse(
            id=function_call.id,
            name=name,
            response={"success": False, "message": f"Error executing {name}: {str(e)}"}
        )


__all__ = [
    "FISHING_FUNCTION_DECLARATIONS",
    "handle_fishing_function_calls",
]
