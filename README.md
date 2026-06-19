<div align="center">

# Project Gabriel Framework <sub>(archived)</sub>

[![Use the new version: Project Gabriel Remastered](https://img.shields.io/badge/use%20the%20new%20version-Project%20Gabriel%20Remastered-6366f1?style=for-the-badge&logo=github&logoColor=white)](https://github.com/HoppouAI/ProjectGabriel-Remastered)
[![Discord](https://img.shields.io/badge/Discord-join-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/ZNWTYTk4Vq)
[![Website](https://img.shields.io/badge/Website-hoppou.ai-ff66c4?style=for-the-badge)](https://hoppou.ai/)
[![Status](https://img.shields.io/badge/status-archived-8b949e?style=for-the-badge)](#)

</div>

> [!IMPORTANT]
> **This is the original Project Gabriel Framework, and it is no longer the version we recommend.**
>
> A full rewrite, **[Project Gabriel Remastered](https://github.com/HoppouAI/ProjectGabriel-Remastered)**, is out now and it is better in just about every way: much easier to set up, far more stable, and loaded with features this version never had. If you are starting fresh, grab Remastered instead. This original is archived and only gets limited support from here on.

### Why Remastered?

- **Far easier setup.** Download a release, run `setup.bat`, then `run.bat`. It installs Python, the dependencies, and GPU support for you, no manual environment wrangling.
- **A lot more features.** Real time Gemini Live voice, persistent memory with semantic search, switchable personalities, spatial navigation, a full WebUI dashboard, a plugin system, a Discord bot, and an optional fully local backend (LM Studio plus local speech to text and TTS, so nothing leaves your PC).
- **Stable and actively maintained.** All the development happens over there now.
- **VRChat is optional.** Remastered also runs standalone through Discord, the WebUI, or any other interface.

➡️ **[Get Project Gabriel Remastered](https://github.com/HoppouAI/ProjectGabriel-Remastered)** and hop in the **[Discord](https://discord.gg/ZNWTYTk4Vq)** if you need a hand or want updates.

> [!NOTE]
> Everything below is the original documentation for this archived framework, kept for reference. Support here is limited, so the Discord is your best bet for help (and it will point you at Remastered anyway).

---

# Project Gabriel Framework

The Code is based on the Project Gabriel Framework by [Hoppou.AI](https://hoppou.ai/) which we use for our AI in VRChat named Gabriel, the Indian guy in the blue polo shirt.

![Gabriel Picture](https://github.com/HoppouAI/ProjectGabriel-Framework/blob/main/Other%20Stuff/Gabriel_Picture.png?raw=true)


# Tutorials:
Easy install tutorial: https://www.youtube.com/watch?v=TPZSzJMIVKQ

Longer full install: https://www.youtube.com/watch?v=E4CQecBAXNM

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

## Installation (Easy)

We have included a setup script that handles everything for you.

1.  Run `setup.bat` in the main folder.
2.  It will automatically install `uv` (our package manager) to a local folder so it doesn't mess with your system.
3.  It will create the virtual environment and install Python 3.13.3.
4.  It will ask if you want to install for **NVIDIA GPU** or **CPU Only**.
    *   Choose **1** if you have an NVIDIA card (better performance for vision).
    *   Choose **2** if you don't.

Once it finishes, you are ready to configure the AI.

## Manual Installation

If you prefer to set things up yourself or the script doesn't work for you, you can follow these steps.

We recommend using **uv** for installation. It is a fast package manager that will automatically fetch the correct version of Python (3.13.3) for this project when you create the environment.

### Option 1: Using UV (Manual)

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

### GPU Support (Manual)

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

**Note:** You must run the application (`run.bat` or `python supervisor.py`) for it to appear in the Volume Mixer.

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

### Easy Start
Just run `run.bat` in the main folder. It will start the supervisor script for you.

### Manual Start
To start the framework manually, run the supervisor script inside your activated environment:

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
