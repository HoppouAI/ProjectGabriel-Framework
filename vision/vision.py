import os
import threading
import mss
import numpy as np
import cv2
import time
import json

# Optional imports with graceful degradation
try:
    import pygetwindow as gw
except Exception:
    gw = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from pythonosc.udp_client import SimpleUDPClient
except Exception:
    SimpleUDPClient = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    import torch
except Exception:
    torch = None

def load_config():
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)
        
config = load_config()

# OSC Setup (VRChat Controls)
osc_client = None
def _ensure_osc():
    global osc_client
    if osc_client is None and SimpleUDPClient is not None:
        host = config.get("osc_host", "127.0.0.1")
        port = int(config.get("osc_port", 9000))
        try:
            osc_client = SimpleUDPClient(host, port)
        except Exception:
            osc_client = None

def send_osc_command(direction, value):
    _ensure_osc()
    if osc_client is not None:
        try:
            osc_client.send_message(f"/input/{direction}", value)
        except Exception:
            pass

# Sprint helpers

def _get_sprint_input_name():
    # Allow override via vision/config.json: sprint_input (default "Run")
    return str(config.get("sprint_input", "Run"))


def start_sprint():
    """Hold the Run input to sprint in VRChat."""
    global _sprinting
    if not config.get("sprint_enabled", True):
        return
    if not _sprinting:
        send_osc_command(_get_sprint_input_name(), 1)
        _sprinting = True


def stop_sprint():
    """Release the Run input to stop sprinting."""
    global _sprinting
    if _sprinting:
        send_osc_command(_get_sprint_input_name(), 0)
        _sprinting = False

# Movement and rotation helpers

def rotate_left(steps: int = 1):
    for _ in range(steps):
        send_osc_command("LookLeft", 1)
        time.sleep(0.1)
        send_osc_command("LookLeft", 0)


def rotate_right(steps: int = 1):
    for _ in range(steps):
        send_osc_command("LookRight", 1)
        time.sleep(0.1)
        send_osc_command("LookRight", 0)


def move_forward():
    send_osc_command("MoveForward", 1)


def stop_forward():
    send_osc_command("MoveForward", 0)


def move_backward():
    send_osc_command("MoveBackward", 1)


def stop_backward():
    send_osc_command("MoveBackward", 0)

# Global model state and background initialization
model = None
device = "cpu"
_model_init_started = False
_model_ready_event = threading.Event()
_follower_thread = None
_stop_event = threading.Event()
_running = False
_sprinting = False
_window_initialized = False
_preview_failed_once = False

def _select_device():
    global device
    prefer = str(config.get("detection_device", "cuda")).lower()
    if torch is not None and prefer == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

def _get_model_path():
    """Return model identifier: allow bare names like 'yolov11n.pt' for auto-download.

    If the value looks like a plain filename (no path separators) we return as-is so
    Ultralytics can fetch it from the hub. If it's a relative or absolute path, resolve
    to an absolute filesystem path.
    """
    path = str(config.get("model_path", "yolov8n.pt"))
    # If it contains a path separator or starts with . or drive letter, treat as path
    if any(sep in path for sep in ("/", "\\")) or path.startswith(".") or (len(path) > 1 and path[1] == ":"):
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(__file__), path)
        return path
    # Bare model name (e.g., 'yolov11n.pt')
    return path

def _init_model():
    global model
    try:
        _select_device()
        if YOLO is None:
            print("Ultralytics YOLO not available")
            return
        primary = _get_model_path()
        candidates = [primary]
        # Fallbacks if provided name is not available in current ultralytics build/cache
        def _maybe_add(x):
            if x not in candidates:
                candidates.append(x)
        # Prefer v10 and v8 as fallbacks for broader compatibility
        _maybe_add("yolov10n.pt")
        _maybe_add("yolov8n.pt")

        last_err = None
        for w in candidates:
            try:
                if device == "cuda" and torch is not None:
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                print(f"Loading YOLO weights: {w}")
                m = YOLO(w)
                m = m.to(device)
                # Manual .half() conversion removed to fix "struct c10::Half != float" on GTX 1650
                # during model fusion. Ultralytics handles precision automatically.
                model = m
                print(f"Loaded YOLO model: {w}")
                break
            except Exception as e:
                last_err = e
                print(f"Failed to load model '{w}': {e}")
                continue
        if model is None and last_err is not None:
            print("All YOLO model load attempts failed; vision will run without detection.")
    finally:
        _model_ready_event.set()
        print(f"Vision model init complete on device: {device}")

def initialize_in_background():
    global _model_init_started
    if _model_init_started:
        return
    _model_init_started = True
    _model_ready_event.clear()
    t = threading.Thread(target=_init_model, name="vision-model-init", daemon=True)
    t.start()
    return t

def get_game_window():
    if gw is None:
        print("pygetwindow not available")
        return None
    title = config.get("window_title", "VRChat")
    windows = gw.getWindowsWithTitle(title)
    if windows:
        game_window = windows[0]
        try:
            game_window.activate()
        except Exception:
            pass
        return game_window.left, game_window.top, game_window.width, game_window.height
    else:
        print("Game window not found!")
        return None

def capture_screen(left, top, width, height):
    # Robust single-capture helper that tolerates intermittent failures
    with mss.mss() as sct:
        monitor = {"top": top, "left": left, "width": width, "height": height}
        try:
            screenshot = sct.grab(monitor)
        except Exception as e:
            # Common on Windows: ScreenShotError: gdi32.GetDIBits() failed
            # Return None to let caller decide how to proceed
            try:
                print(f"capture_screen grab failed: {e}")
            except Exception:
                pass
            return None
        frame = np.array(screenshot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

def detect_players(frame):
    if model is None:
        return []
    resized_frame = cv2.resize(frame, (640, 640))
    # Run model inference with verbose logging disabled to prevent console spam
    try:
        results = model(resized_frame, verbose=False)
    except TypeError:
        # Older ultralytics may not accept verbose kwarg
        results = model(resized_frame)
    players = []
    
    scale_x = frame.shape[1] / 640
    scale_y = frame.shape[0] / 640
    conf_thresh = max(0.30, float(config.get("confidence_threshold", 0.25)))
    names = None
    try:
        # ultralytics model has names mapping
        names = getattr(getattr(model, 'model', None), 'names', None) or getattr(model, 'names', None)
    except Exception:
        names = None

    allowed_labels = {"person", "human", "player", "people", "man", "woman"}
    allowed_ids = set()
    try:
        if names is not None:
            if isinstance(names, (list, tuple)):
                for i, nm in enumerate(names):
                    nl = str(nm).strip().lower() if nm is not None else ""
                    if any(lbl in nl for lbl in allowed_labels):
                        allowed_ids.add(int(i))
            elif isinstance(names, dict):
                for i, nm in names.items():
                    try:
                        idx = int(i)
                    except Exception:
                        continue
                    nl = str(nm).strip().lower() if nm is not None else ""
                    if any(lbl in nl for lbl in allowed_labels):
                        allowed_ids.add(idx)
    except Exception:
        allowed_ids = set()
    if not allowed_ids:
        allowed_ids = {0}
    
    for result in results:
        for box in result.boxes:
            try:
                cls_id = int(box.cls)
                conf_val = float(box.conf)
            except Exception:
                cls_id = int(box.cls) if hasattr(box, 'cls') else -1
                conf_val = 1.0
            if conf_val >= conf_thresh and cls_id in allowed_ids:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
                y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
                box_height = max(1, y2 - y1)
                distance = estimate_distance(box_height)
                label = None
                try:
                    if names is not None and 0 <= cls_id < len(names):
                        label = str(names[cls_id])
                except Exception:
                    label = None
                players.append({
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "distance": float(distance),
                    "cls": cls_id,
                    "label": label if label is not None else str(cls_id),
                    "conf": float(conf_val),
                })
    
    players.sort(key=lambda p: p.get("distance", 1e9))
    return players

def estimate_distance(box_height, reference_height=200, reference_distance=1.0):
    return reference_distance * (reference_height / box_height)

def read_name_tag(frame, player_box, frame_count):
    if frame_count % 10 != 0:  # Only run OCR every 10 frames
        return ""
    if isinstance(player_box, dict):
        x1, y1, x2 = player_box["x1"], player_box["y1"], player_box["x2"]
    else:
        x1, y1, x2, _ = player_box
    name_tag_region = frame[max(0, y1 - 30):y1, x1:x2]
    if name_tag_region.size == 0 or pytesseract is None:
        return ""
    gray = cv2.cvtColor(name_tag_region, cv2.COLOR_BGR2GRAY)
    try:
        text = pytesseract.image_to_string(gray, config='--psm 7')
    except Exception:
        return ""
    return text.strip()

def track_and_rotate(frame, width, height, last_target=None, last_direction=None, no_player_time=0, frame_count=0, players=None):
    if players is None:
        players = detect_players(frame)
    if players:
        no_player_time = 0

        player_box = last_target if (isinstance(last_target, dict) and last_target in players) else players[0]
        x1, y1, x2, y2 = player_box["x1"], player_box["y1"], player_box["x2"], player_box["y2"]
        distance = float(player_box.get("distance", 1.0))
        player_center_x = (x1 + x2) // 2
        screen_center_x = width // 2
        name_tag = read_name_tag(frame, player_box, frame_count)

        obj_name = player_box.get("label", str(player_box.get("cls", "")))
    # Debug print suppressed to avoid console spam
    # print(f"Tracking {obj_name} at Distance: {distance:.2f} | Name: {name_tag}")

        # Movement control
        moving_forward = False
        if distance > config["max_distance"]:
            move_forward()
            stop_backward()
            moving_forward = True
        elif distance < config["min_distance"]:
            move_backward()
            stop_forward()
            moving_forward = False
        else:
            stop_forward()
            stop_backward()
            moving_forward = False

        # Sprint control based on distance thresholds
        sprint_enabled = bool(config.get("sprint_enabled", True))
        sprint_dist = float(config.get("sprint_distance", config.get("max_distance", 0.5) * 1.5))
        sprint_catchup = float(config.get("sprint_catchup_distance", config.get("max_distance", 0.5)))
        if sprint_enabled and moving_forward and distance > sprint_dist:
            start_sprint()
        elif _sprinting and (not moving_forward or distance <= sprint_catchup):
            stop_sprint()

        dead_zone = width * config["deadzone"]
        deviation = player_center_x - screen_center_x

        if deviation > dead_zone and last_direction != "right":
            # print("Stepping Right...")
            rotate_right(steps=1)
            last_direction = "right"
        elif deviation < -dead_zone and last_direction != "left":
            # print("Stepping Left...")
            rotate_left(steps=1)
            last_direction = "left"
        else:
            last_direction = None
        last_target = player_box
    else:
    # print("No player detected. Stopping all movement.")
        stop_forward()
        stop_backward()
        stop_sprint()
        no_player_time += 1
        if no_player_time > 50:
            # print("Searching for player...")
            rotate_right(steps=1)
        last_direction = None
        last_target = None

    return last_target, last_direction, no_player_time

def _draw_overlays(frame, players, frame_count):
    try:
        for det in players:
            if isinstance(det, dict):
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                distance = float(det.get("distance", 0.0))
                cname = str(det.get("label", det.get("cls", "")))
                conf = det.get("conf", None)
            else:
                x1, y1, x2, y2, distance = det
                cname = "object"
                conf = None
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if conf is not None:
                label = f"{cname} {conf:.2f} d:{distance:.2f}"
            else:
                label = f"{cname} d:{distance:.2f}"
            cv2.putText(frame, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        h, w = frame.shape[:2]
        cx = w // 2
        dead_zone = int(w * float(config.get("deadzone", 0.05)))
        cv2.line(frame, (cx, 0), (cx, h), (255, 255, 0), 1)
        cv2.rectangle(frame, (cx - dead_zone, 0), (cx + dead_zone, 5), (0, 255, 255), -1)
        status = f"objects:{len(players)} device:{device} frame:{frame_count}"
        cv2.putText(frame, status, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    except Exception:
        pass

def _show_frame(frame):
    global _window_initialized, _preview_failed_once
    try:
        if not _window_initialized:
            try:
                cv2.namedWindow("YOLO Player Detection", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("YOLO Player Detection", 960, 540)
            except Exception:
                pass
            _window_initialized = True
        cv2.imshow("YOLO Player Detection", frame)
        key = cv2.waitKey(1) & 0xFF
        return True, key
    except Exception as e:
        if not _preview_failed_once:
            _preview_failed_once = True
            try:
                print(f"Preview window unavailable: {e}")
            except Exception:
                pass
        return False, -1

def _follow_loop():
    global _running
    # Wait for model to be ready
    if not _model_ready_event.is_set():
        _model_ready_event.wait(timeout=60)
    game_window = get_game_window()
    if not game_window:
        _running = False
        return
    left, top, width, height = game_window

    last_target = None
    last_direction = None
    no_player_time = 0
    frame_count = 0

    _running = True
    with mss.mss() as sct:
        while not _stop_event.is_set():
            monitor = {"top": top, "left": left, "width": width, "height": height}
            try:
                screenshot = sct.grab(monitor)
            except Exception as e:
                # Intermittent screen grab failure; back off and continue
                try:
                    print(f"vision follow grab failed: {e}")
                except Exception:
                    pass
                time.sleep(0.05)
                continue
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            players = detect_players(frame) if model is not None else []

            if bool(config.get("show_window", True)):
                _draw_overlays(frame, players, frame_count)
                ok, key = _show_frame(frame)
                if ok and (key == 27 or key == ord('q')):
                    _stop_event.set()
                    break

            last_target, last_direction, no_player_time = track_and_rotate(
                frame, width, height, last_target, last_direction, no_player_time, frame_count, players
            )

            frame_count += 1
            # Small sleep to reduce CPU usage
            time.sleep(0.005)

    stop_forward()
    stop_backward()
    stop_sprint()
    try:
        if bool(config.get("show_window", True)):
            cv2.destroyAllWindows()
    except Exception:
        pass
    _running = False

def start_following():
    global _follower_thread
    initialize_in_background()
    if _follower_thread and _follower_thread.is_alive():
        return True
    _stop_event.clear()
    _follower_thread = threading.Thread(target=_follow_loop, name="vision-follow", daemon=True)
    _follower_thread.start()
    return True

def stop_following():
    _stop_event.set()
    stop_sprint()
    return True

def get_status():
    return {
        "device": device,
        "model_ready": _model_ready_event.is_set(),
        "following": _running,
        "sprinting": _sprinting,
    }

def main():
    initialize_in_background()
    # Blocking run with window display for manual testing
    game_window = get_game_window()
    if not game_window:
        return
    left, top, width, height = game_window

    last_target = None
    last_direction = None
    no_player_time = 0
    frame_count = 0

    # Wait up to 60s for model
    _model_ready_event.wait(timeout=60)

    with mss.mss() as sct:
        while True:
            monitor = {"top": top, "left": left, "width": width, "height": height}
            try:
                screenshot = sct.grab(monitor)
            except Exception as e:
                try:
                    print(f"vision main grab failed: {e}")
                except Exception:
                    pass
                time.sleep(0.05)
                continue
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            players = detect_players(frame) if model is not None else []

            last_target, last_direction, no_player_time = track_and_rotate(
                frame, width, height, last_target, last_direction, no_player_time, frame_count, players
            )

            if bool(config.get("show_window", True)):
                _draw_overlays(frame, players, frame_count)
                _show_frame(frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()
    stop_forward()
    stop_backward()

if __name__ == "__main__":
    main()
