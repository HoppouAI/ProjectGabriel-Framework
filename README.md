# Project Gabriel Framework

The Code is based on the Project Gabriel Framework by [Hoppou.AI](https://hoppou.ai/) which we use for our AI in VRChat named Gabriel, the Indian guy in the blue polo shirt.

![Gabriel Picture](https://github.com/HoppouAI/ProjectGabriel-Framework/blob/main/Other%20Stuff/Gabriel_Picture.png?raw=true)

## Summary

This is a Python-based framework for running a live AI assistant in VRChat. It handles audio I/O, VRChat OSC integration (for movement and chatbox), memory management, and vision capabilities.

- **Main Entry Point:** `supervisor.py`
- **Key Features:** Live audio streaming, Actual Vision (Gemini Live), following/looking vision (YOLOv11), OSC control.

## Prerequisites

Before setting up the code, you need to set up the environment and audio drivers.

1.  **Virtual Audio Cables**: You need two separate virtual lines to route audio to and from VRChat.
    *   [VB-Audio Cable](https://vb-audio.com/Cable/) (Standard)
    *   [Virtual Audio Cable (Hi-Fi)](https://vb-audio.com/Cable/#DownloadASIOBridge) (Or any secondary cable)
2.  **Gemini API Key**: Obtain a key from Google AI Studio.

## Installation

We recommend using **uv** for installation. It is a fast package manager that will automatically fetch the correct version of Python (3.13.3) for this project when you create the environment.

### Option 1: Using UV (Recommended)

official docs at: https://docs.astral.sh/uv/getting-started/installation/

First, install uv using PowerShell:
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Once installed, restart your terminal and run the following commands in the project folder. 

**Note:** The `uv venv` command below will automatically download and install Python 3.13.3 for you if you don't have it.

```bash
# Creates the virtual environment and installs Python 3.13.3 automatically
uv venv --python 3.13.3

# Activate environment (Windows)
.venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Option 2: Standard Python (Manual)

If you prefer not to use uv, you must download and install **Python 3.13.3** manually first. Make sure "Add Python to PATH" is checked during the installer setup.

```bash
# Create virtual environment
python -m venv venv

# Activate environment (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Support (Optional)

If you have an NVIDIA GPU, install the CUDA-enabled version of PyTorch to improve vision performance. Run this command *after* installing the main requirements:

```bash
# If using UV

# First 
uv pip uninstall torch torchvision torchaudio

# Then install CUDA version
uv pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# If using Standard Pip

# First
pip uninstall torch torchvision torchaudio

# Then install CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Configuration

### 1. API Key
Open `config.yml` in the root directory. Locate the `api` section and paste your Gemini API key:

```yaml
api:
  api_key: "YOUR_KEY_HERE"
```

### 2. Personality
Open `prompts.json`. This controls the AI's persona.
*   **Important:** Ensure the main configuration key is named `"default"`.
*   Modify the `"prompt"` field to change the backstory (e.g., "You are a helpful assistant...").

### 3. Performance Tuning
If running on a lower-end PC or without a dedicated GPU, you should disable the vision system in `vision/config.json` to save resources:
*   Set `"enabled": false,` at the top of the config.json

## Audio Routing

For the AI to communicate in VRChat, you must configure the Windows Volume Mixer and VRChat settings correctly.

**Note:** You must run the application (`python supervisor.py`) for it to appear in the Volume Mixer.

### Windows Volume Mixer Settings

| Application | Output Device (Top Bar) | Input Device (Bottom Bar) |
| :--- | :--- | :--- |
| **Python** | `CABLE Input` (VB-Audio Virtual Cable) | `Hi-Fi Cable Output` (VB-Audio Hi-Fi) |
| **VRChat** | `Hi-Fi Cable Input` (VB-Audio Hi-Fi) | Default / Microphone |

### VRChat In-Game Settings

Go to Audio Settings -> Microphone in VRChat:

1.  **Microphone Device:** `CABLE Output` (VB-Audio Virtual Cable)
2.  **Noise Suppression:** OFF (Required for audio to pass cleanly)
3.  **Activation Threshold:** 0%
4.  **Volume:** Mute Music/SFX and keep Voices at 100%.

## Usage

To start the framework, run the supervisor script inside your activated environment:

```bash
python supervisor.py
```

To stop the application, press `CTRL+C` in the terminal.

## Project Structure

- `supervisor.py`: Process manager that launches the necessary components.
- `main.py`: Core application logic.
- `config.yml`: Main configuration file for ports, keys, and toggles.
- `appends.json`: Content that is appended to the system prompt, e.g. extra information.
- `prompts.json`: System prompts and character definitions.
- `requirements.txt`: Python dependency list.
- `vision/config.json`: Vision system configuration.