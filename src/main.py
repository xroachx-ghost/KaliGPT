"""KaliGPT desktop controller GUI."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

try:
    import pyautogui
except ImportError:  # pragma: no cover - handled in UI
    pyautogui = None

try:
    import openai
except ImportError:  # pragma: no cover - handled in UI
    openai = None


@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime

    def to_display(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{time_str}] {self.role.title()}: {self.content}"


class ChatModel(QtCore.QAbstractListModel):
    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._messages: list[ChatMessage] = []

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._messages)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._messages)):
            return None
        message = self._messages[index.row()]
        if role == QtCore.Qt.DisplayRole:
            return message.to_display()
        if role == QtCore.Qt.UserRole:
            return message
        return None

    def add_message(self, message: ChatMessage) -> None:
        self.beginInsertRows(QtCore.QModelIndex(), len(self._messages), len(self._messages))
        self._messages.append(message)
        self.endInsertRows()

    def as_openai_messages(self) -> list[dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in self._messages]


class ComputerControlPanel(QtWidgets.QGroupBox):
    screenshot_captured = QtCore.Signal(QtGui.QPixmap)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Computer Control", parent)
        self.setCheckable(True)
        self.setChecked(False)

        self.status_label = QtWidgets.QLabel("Control disabled.")
        self.status_label.setWordWrap(True)

        self.screenshot_label = QtWidgets.QLabel()
        self.screenshot_label.setFixedHeight(240)
        self.screenshot_label.setAlignment(QtCore.Qt.AlignCenter)
        self.screenshot_label.setStyleSheet("background: #111; color: #eee; border-radius: 6px;")
        self.screenshot_label.setText("No preview")

        self.take_screenshot_button = QtWidgets.QPushButton("Take Desktop Snapshot")
        self.take_screenshot_button.clicked.connect(self.handle_screenshot)

        self.activity_log = QtWidgets.QPlainTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setPlaceholderText("Automation log will appear here.")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.screenshot_label)
        layout.addWidget(self.take_screenshot_button)
        layout.addWidget(self.activity_log)

        self.toggled.connect(self.handle_toggle)
        self.screenshot_captured.connect(self.update_preview)

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.activity_log.appendPlainText(f"[{timestamp}] {message}")

    def handle_toggle(self, enabled: bool) -> None:
        if enabled:
            self.status_label.setText(
                "Control enabled. Keep this window visible so you can monitor the automation."
            )
            self.log("Control enabled.")
        else:
            self.status_label.setText("Control disabled.")
            self.log("Control disabled.")

    def handle_screenshot(self) -> None:
        if pyautogui is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Missing dependency",
                "pyautogui is not installed. Install it to capture screenshots.",
            )
            return
        self.log("Capturing screenshot...")
        screenshot = pyautogui.screenshot()
        image = screenshot.convert("RGBA")
        qt_image = QtGui.QImage(
            image.tobytes("raw", "RGBA"),
            image.width,
            image.height,
            QtGui.QImage.Format_RGBA8888,
        )
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        self.screenshot_captured.emit(pixmap)

    def update_preview(self, pixmap: QtGui.QPixmap) -> None:
        scaled = pixmap.scaled(
            self.screenshot_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.screenshot_label.setPixmap(scaled)
        self.log("Screenshot updated.")


class ChatWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("KaliGPT Workstation")
        self.resize(1200, 720)

        self.chat_model = ChatModel(self)
        self.chat_view = QtWidgets.QListView()
        self.chat_view.setModel(self.chat_model)
        self.chat_view.setWordWrap(True)
        self.chat_view.setStyleSheet("font-size: 14px;")

        self.message_input = QtWidgets.QTextEdit()
        self.message_input.setPlaceholderText("Ask KaliGPT to help with your workflow...")
        self.message_input.setFixedHeight(120)

        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.clicked.connect(self.handle_send)

        self.controls_panel = ComputerControlPanel()

        self.api_status = QtWidgets.QLabel()
        self.api_status.setText(self._api_status_text())
        self.api_status.setStyleSheet("color: #888;")

        chat_layout = QtWidgets.QVBoxLayout()
        chat_layout.addWidget(QtWidgets.QLabel("Conversation"))
        chat_layout.addWidget(self.chat_view)
        chat_layout.addWidget(self.message_input)

        send_layout = QtWidgets.QHBoxLayout()
        send_layout.addWidget(self.api_status)
        send_layout.addStretch(1)
        send_layout.addWidget(self.send_button)
        chat_layout.addLayout(send_layout)

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addLayout(chat_layout, stretch=3)
        main_layout.addWidget(self.controls_panel, stretch=2)

        self._load_history()

    def _api_status_text(self) -> str:
        if openai is None:
            return "OpenAI client not installed. Responses will be stubbed."
        if not os.getenv("OPENAI_API_KEY"):
            return "Set OPENAI_API_KEY to enable live responses."
        return "Connected to OpenAI API."

    def _load_history(self) -> None:
        history_path = self._history_path()
        if not history_path.exists():
            return
        try:
            with history_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for item in payload:
                message = ChatMessage(
                    role=item["role"],
                    content=item["content"],
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                )
                self.chat_model.add_message(message)
        except (OSError, ValueError, KeyError):
            return

    def _save_history(self) -> None:
        history_path = self._history_path()
        history_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in self.chat_model._messages
        ]
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _history_path(self) -> Path:
        return Path.home() / ".kaligpt" / "history.json"

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._save_history()
        super().closeEvent(event)

    def handle_send(self) -> None:
        content = self.message_input.toPlainText().strip()
        if not content:
            return
        self._add_message("user", content)
        self.message_input.clear()
        response = self._generate_response()
        self._add_message("assistant", response)

    def _add_message(self, role: str, content: str) -> None:
        message = ChatMessage(role=role, content=content, timestamp=datetime.now())
        self.chat_model.add_message(message)
        self.chat_view.scrollToBottom()

    def _generate_response(self) -> str:
        if openai is None or not os.getenv("OPENAI_API_KEY"):
            return (
                "I'm ready to help. Install the OpenAI SDK and set OPENAI_API_KEY "
                "to enable live model responses."
            )
        client = openai.OpenAI()
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.chat_model.as_openai_messages(),
            )
            return completion.choices[0].message.content
        except Exception as exc:  # pragma: no cover - network call
            return f"API error: {exc}"


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ChatWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
