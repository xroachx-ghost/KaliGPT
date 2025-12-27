# KaliGPT Workstation

An advanced desktop GUI that combines a ChatGPT-style assistant with a dedicated panel for
computer control and activity monitoring.

## Features

- Two-pane layout with chat history and workstation controls.
- Optional OpenAI integration (set `OPENAI_API_KEY`).
- First-run wizard for choosing a provider, entering API keys, and enabling desktop control.
- Desktop snapshot preview and automation log.
- Startup update checks with an in-app status banner (can be disabled).
- System tray menu with quick access to open, pause, or quit the agent.
- Persistent chat history stored in `~/.kaligpt/history.json`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/main.py
```

## Notes

- Desktop control uses `pyautogui`, which may require accessibility permissions on macOS.
- The OpenAI integration is optional; the UI will fall back to stub responses when
  credentials are not provided.
- If no API key is configured for the selected provider, the header shows a
  “Demo mode: API key missing” banner to indicate stubbed responses.
- Update checks use the `packaging/version.json` manifest by default. Set
  `KALIGPT_UPDATE_URL` to a hosted JSON manifest (or release API endpoint you control)
  to surface real update notifications.

## Packaging (PyInstaller)

1. Install dependencies, including PyInstaller:

   ```bash
   pip install -r requirements.txt
   pip install pyinstaller
   ```

2. Build the desktop binary:

   ```bash
   ./packaging/build_pyinstaller.sh
   ```

3. The bundled output is available in `dist/KaliGPT/`.

Update `packaging/version.json` (and/or `KALIGPT_UPDATE_URL`) to point to your release
notes and download URLs so the UI can surface update notifications.

## System tray controls

- **Open**: restores the main window.
- **Pause agent**: pauses or resumes the agent loop (agent mode only).
- **Quit**: exits KaliGPT.

## Troubleshooting

- **Automation unavailable banner**: install `pyautogui` (`pip install pyautogui`) and
  enable accessibility/automation permissions for your OS (macOS may also require Screen
  Recording permissions) before restarting the app.
