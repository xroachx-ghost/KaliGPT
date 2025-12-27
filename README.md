# KaliGPT Workstation

An advanced desktop GUI that combines a ChatGPT-style assistant with a dedicated panel for
computer control and activity monitoring.

## Features

- Two-pane layout with chat history and workstation controls.
- Optional OpenAI integration (set `OPENAI_API_KEY`).
- First-run wizard for choosing a provider, entering API keys, and enabling desktop control.
- Desktop snapshot preview and automation log.
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
