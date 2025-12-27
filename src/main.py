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
        self.screenshot_label.setStyleSheet(
            "background: #1f2023; color: #9aa0a6; border-radius: 8px; border: 1px solid #33343a;"
        )
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


class ChatBubbleDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._bubble_padding = 12
        self._text_padding = 12
        self._max_width_ratio = 0.7

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> None:
        message: ChatMessage | None = index.data(QtCore.Qt.UserRole)
        if message is None:
            return

        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        rect = option.rect
        view_width = option.widget.width() if option.widget else rect.width()
        max_width = max(280, int(view_width * self._max_width_ratio))
        font = option.font
        metrics = QtGui.QFontMetrics(font)

        text_rect = metrics.boundingRect(
            0,
            0,
            max_width - 2 * self._text_padding,
            10_000,
            QtCore.Qt.TextWordWrap,
            message.content,
        )
        bubble_width = text_rect.width() + 2 * self._text_padding
        bubble_height = text_rect.height() + 2 * self._text_padding

        if message.role == "user":
            bubble_color = QtGui.QColor("#10a37f")
            text_color = QtGui.QColor("#ffffff")
            x = rect.right() - bubble_width - self._bubble_padding
        else:
            bubble_color = QtGui.QColor("#444654")
            text_color = QtGui.QColor("#ffffff")
            x = rect.left() + self._bubble_padding

        y = rect.top() + self._bubble_padding // 2
        bubble_rect = QtCore.QRect(x, y, bubble_width, bubble_height)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(bubble_color)
        painter.drawRoundedRect(bubble_rect, 12, 12)

        text_draw_rect = bubble_rect.adjusted(
            self._text_padding, self._text_padding, -self._text_padding, -self._text_padding
        )
        painter.setPen(text_color)
        painter.drawText(text_draw_rect, QtCore.Qt.TextWordWrap, message.content)

        painter.restore()

    def sizeHint(
        self,
        option: QtWidgets.QStyleOptionViewItem,
        index: QtCore.QModelIndex,
    ) -> QtCore.QSize:
        message: ChatMessage | None = index.data(QtCore.Qt.UserRole)
        if message is None:
            return QtCore.QSize(0, 0)
        view_width = option.widget.width() if option.widget else 600
        max_width = max(280, int(view_width * self._max_width_ratio))
        metrics = QtGui.QFontMetrics(option.font)
        text_rect = metrics.boundingRect(
            0,
            0,
            max_width - 2 * self._text_padding,
            10_000,
            QtCore.Qt.TextWordWrap,
            message.content,
        )
        height = text_rect.height() + 2 * self._text_padding + self._bubble_padding
        return QtCore.QSize(view_width, height)


class ModeSelectionDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Mode")
        self.setModal(True)

        title = QtWidgets.QLabel("Choose a startup mode")
        title.setStyleSheet("font-size: 16px; font-weight: 600;")

        self.normal_mode_radio = QtWidgets.QRadioButton("Normal AI mode")
        self.normal_mode_radio.setChecked(True)
        self.agent_mode_radio = QtWidgets.QRadioButton("Agent mode")

        normal_description = QtWidgets.QLabel(
            "Best for chat-only workflows without desktop automation controls."
        )
        normal_description.setWordWrap(True)
        normal_description.setStyleSheet("color: #9aa0a6;")
        agent_description = QtWidgets.QLabel(
            "Includes computer control tools and automation activity panels."
        )
        agent_description.setWordWrap(True)
        agent_description.setStyleSheet("color: #9aa0a6;")

        normal_layout = QtWidgets.QVBoxLayout()
        normal_layout.addWidget(self.normal_mode_radio)
        normal_layout.addWidget(normal_description)

        agent_layout = QtWidgets.QVBoxLayout()
        agent_layout.addWidget(self.agent_mode_radio)
        agent_layout.addWidget(agent_description)

        options_layout = QtWidgets.QHBoxLayout()
        options_layout.addLayout(normal_layout)
        options_layout.addLayout(agent_layout)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(options_layout)
        layout.addWidget(buttons)

    def selected_mode(self) -> str:
        return "agent" if self.agent_mode_radio.isChecked() else "normal"


class ChatWindow(QtWidgets.QWidget):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.setWindowTitle("KaliGPT Workstation")
        self.resize(1200, 720)

        self.chat_model = ChatModel(self)
        self.chat_view = QtWidgets.QListView()
        self.chat_view.setModel(self.chat_model)
        self.chat_view.setWordWrap(True)
        self.chat_view.setSpacing(8)
        self.chat_view.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.chat_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.chat_view.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.chat_view.setItemDelegate(ChatBubbleDelegate(self.chat_view))

        self.message_input = QtWidgets.QTextEdit()
        self.message_input.setPlaceholderText("Message KaliGPT...")
        self.message_input.setFixedHeight(100)

        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.clicked.connect(self.handle_send)

        self.controls_panel = ComputerControlPanel()

        self.api_status = QtWidgets.QLabel()
        self.api_status.setText(self._api_status_text())
        self.api_status.setStyleSheet("color: #9aa0a6; font-size: 12px;")

        header_title = QtWidgets.QLabel("KaliGPT")
        header_title.setObjectName("HeaderTitle")
        header_subtitle = QtWidgets.QLabel("Your AI assistant for secure workflows")
        header_subtitle.setObjectName("HeaderSubtitle")

        header_layout = QtWidgets.QVBoxLayout()
        header_layout.addWidget(header_title)
        header_layout.addWidget(header_subtitle)
        header_layout.setSpacing(2)

        header_widget = QtWidgets.QWidget()
        header_widget.setLayout(header_layout)

        chat_layout = QtWidgets.QVBoxLayout()
        chat_layout.addWidget(header_widget)
        chat_layout.addWidget(self.chat_view)

        input_layout = QtWidgets.QHBoxLayout()
        input_layout.addWidget(self.message_input, stretch=1)
        input_layout.addWidget(self.send_button)
        chat_layout.addLayout(input_layout)
        chat_layout.addWidget(self.api_status)

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addLayout(chat_layout, stretch=3)
        main_layout.addWidget(self.controls_panel, stretch=2)

        self._apply_theme()
        self._configure_mode()
        self._load_history()

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background: #202123;
                color: #e8e8e8;
                font-size: 14px;
            }
            QListView {
                background: #202123;
                border: none;
            }
            QTextEdit {
                background: #2b2c2f;
                border: 1px solid #3e3f4b;
                border-radius: 12px;
                padding: 10px;
                color: #f5f5f5;
            }
            QPushButton {
                background: #10a37f;
                border: none;
                border-radius: 10px;
                padding: 10px 18px;
                color: white;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #0f8b6b;
            }
            QPushButton:pressed {
                background: #0b6f55;
            }
            QPushButton:disabled {
                background: #3a3b3f;
                color: #9aa0a6;
            }
            QGroupBox {
                background: #2b2c2f;
                border: 1px solid #3e3f4b;
                border-radius: 12px;
                margin-top: 14px;
                padding: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #e8e8e8;
                font-weight: 600;
            }
            QLabel#HeaderTitle {
                font-size: 22px;
                font-weight: 600;
            }
            QLabel#HeaderSubtitle {
                color: #9aa0a6;
                font-size: 13px;
            }
            """
        )

    def _configure_mode(self) -> None:
        is_agent_mode = self.mode == "agent"
        self.controls_panel.setVisible(is_agent_mode)
        self.controls_panel.setEnabled(is_agent_mode)
        if not is_agent_mode:
            self.controls_panel.setChecked(False)

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
    dialog = ModeSelectionDialog()
    if dialog.exec() != QtWidgets.QDialog.Accepted:
        return 0
    window = ChatWindow(dialog.selected_mode())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
