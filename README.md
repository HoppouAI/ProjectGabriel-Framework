# Project Gabriel Framework

The Code is based on the Project Gabriel Framework by [Hoppou.AI](https://hoppou.ai/) which we use for our AI in VRChat named Gabriel — the Indian guy in the blue polo shirt.

![Gabriel Picture](https://github.com/HoppouAI/ProjectGabriel-Framework/blob/main/Other%20Stuff/Gabriel_Picture.png?raw=true)

## Quick summary

- Language: Python 3.10+ (codebase is Python-first)
- Purpose: Live assistant / VRChat AI (audio I/O, VRChat OSC, memory, web UI)
- Main entry points: `main.py`, `supervisor.py`, `v2.py`, `api/webui_server.py` (see folder `api/webui/` for the static UI)
- Important assets: `yolo11m.pt` (vision model — large binary, optional)

## Prerequisites

- Python 3.10+ (Reccomended 3.13.3)
- If you plan to use as a AI In VRChat you will need 2 Virtual Audio Cables, https://vb-audio.com/Cable/ and https://vac.muzychenko.net/en/download.htm Download the "VAC 4.xx" Lite version
- Install dependencies:

```bash
pip install -r requirements.txt
```

If you need CUDA wheels for PyTorch, follow the commented suggestion in `requirements.txt`.

## Run (basic)

- Start the main app (default behavior wired in `main.py`):

```bash
python main.py
```

The WebUI is located under `api/webui/` Default ports are configured in `config.yml`.

## Configuration — `config.yml`

The project is configured via the YAML file `config.yml` in the repository root. Below are the most-common configuration sections and recommended placeholders. Keep secrets (API keys, passwords) secure

Top-level sections you will commonly edit:

- `audio` — audio settings
	- `format`: sample format (integer)
	- `channels`: 1 for mono, 2 for stereo
	- `send_sample_rate` / `receive_sample_rate`: rates used for streaming
	- `chunk_size`: buffer size for audio frames

- `model` — model selector
	- `name`: path or name of the model used for Gemini/LLM calls (example: `models/gemini-live-2.5-flash-preview`)

- `defaults` — general defaults
	- `mode`: `screen`, `camera`, or `none` (affects capture behavior)

- `api` — REST API and WebUI server
	- `version`: API version string
	- `api_key`: primary API key or a URL where to obtain one (replace with real key)
	- `backup_api_keys`: list of fallback keys for failover
	- `chat.enabled`: enable/disable REST chat API
	- `chat.host` / `chat.port`: host and port for API
	- `webui.enabled`: whether the embeddable WebUI server should start
	- `webui.host` / `webui.port`: host/port for UI

- `webhooks` — webhook URLs (e.g., `image_generation`)

- `live_connect` — Gemini Live settings
	- `response_modalities`: list like `['AUDIO']` or `['AUDIO','TEXT']`
	- `media_resolution`: `MEDIA_RESOLUTION_LOW` or `MEDIA_RESOLUTION_HIGH`
	- `speech.language_code`: e.g., `en-US` or other supported codes
	- `speech.voice.name`: voice name (depends on the target speech API)
	- `context_window`: tuning for context compression and windowing
	- `prompt` / `custom_prompt`: `prompt` selects mode; when `custom` use `custom_prompt`

- `tools` — enable/disable optional integrations
	- `google_search`, `memory_system`, `function_declarations` (array)

- `video` — camera / screen capture settings and image processing options

- `queues` — queue sizes for audio/effects

- `logging` — global log level and per-module overrides

- `session_management` — reconnection and heartbeat tuning

- `vrchat` — VRChat-specific settings
	- `application`: `name`, `version`, `contact` (identify your app)
	- `credentials`: `username`, `password`, `two_factor_code`, `totp_secret`

- `osc` — VRChat OSC integration
	- `host` / `port`: where to send OSC messages (VRChat usually listens on `127.0.0.1:9000`)
	- `chatbox` settings: `max_length`, `send_immediately`, `auto_clear_delay`, etc.

- `memory` — memory/mongo settings
	- `enabled`: master switch
	- `recent_memories_count`: how many to include in system prompts
	- `mongo.uri`, `mongo.host`, `uri_env_var`, `username`, `password_env_var`, etc. (use env vars for credentials)

Example minimal edits (place in `config.yml`):

```yaml
api:
	webui:
		enabled: true
		host: "0.0.0.0"
		port: 5555

model:
	name: "models/gemini-2.5-flash-native-audio-preview-09-2025"

vrchat:
	application:
		name: "MyAI"
		version: "0.1"
		contact: "me@example.com"

	credentials:
		username: "VRC_USERNAME"
		password: "VRC_PASSWORD"

osc:
	enabled: true
	host: "127.0.0.1"
	port: 9000

memory:
	enabled: false

```

## Troubleshooting & notes

- If you run into audio device issues on Windows, make sure `pyaudio` is installed and your microphone, and output devices are properly set Using Windows settings/volume mixer for Python
- If the WebUI or API fails to bind, check `config.yml` ports and that no other service is using them.

## Where to look in the code

- App main: `main.py` and `v2.py`
- WebUI server: `api/webui_server.py` and `api/webui/` static files
- VRChat / OSC helpers: `vrchatapi.py` and `tools/vrchat_tools.py`
- Memory: `tools/memory.py` and related `mongo` settings in `config.yml`

## Contributing

If you'd like to contribute, please open issues or PRs.

---
