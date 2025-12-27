"""KaliGPT desktop controller GUI."""
from __future__ import annotations

import importlib
import json
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

try:
    import pyautogui
except ImportError:  # pragma: no cover - handled in UI
    pyautogui = None

try:
    import openai
except ImportError:  # pragma: no cover - handled in UI
    openai = None

try:
    import anthropic
except ImportError:  # pragma: no cover - handled in UI
    anthropic = None


PROVIDER_REGISTRY = {
    "openai": {
        "label": "OpenAI",
        "env_key": "OPENAI_API_KEY",
        "models": [
            {"id": "gpt-4o-mini", "display": "GPT-4o Mini"},
            {"id": "gpt-4o", "display": "GPT-4o"},
        ],
    },
    "anthropic": {
        "label": "Anthropic",
        "env_key": "ANTHROPIC_API_KEY",
        "models": [
            {"id": "claude-3-5-sonnet-latest", "display": "Claude 3.5 Sonnet"},
            {"id": "claude-3-5-haiku-latest", "display": "Claude 3.5 Haiku"},
        ],
    },
    "deepseek": {
        "label": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "models": [
            {"id": "deepseek-chat", "display": "DeepSeek Chat"},
            {"id": "deepseek-reasoner", "display": "DeepSeek Reasoner"},
        ],
    },
    "gemini": {
        "label": "Google Gemini",
        "env_key": "GEMINI_API_KEY",
        "models": [
            {"id": "gemini-1.5-pro", "display": "Gemini 1.5 Pro"},
            {"id": "gemini-1.5-flash", "display": "Gemini 1.5 Flash"},
        ],
    },
    "groq": {
        "label": "Groq",
        "env_key": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "models": [
            {"id": "llama-3.1-70b-versatile", "display": "Llama 3.1 70B"},
            {"id": "llama-3.1-8b-instant", "display": "Llama 3.1 8B"},
            {"id": "mixtral-8x7b-32768", "display": "Mixtral 8x7B"},
        ],
    },
    "mistral": {
        "label": "Mistral",
        "env_key": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        "models": [
            {"id": "mistral-large-latest", "display": "Mistral Large"},
            {"id": "mistral-small-latest", "display": "Mistral Small"},
        ],
    },
    "perplexity": {
        "label": "Perplexity",
        "env_key": "PERPLEXITY_API_KEY",
        "base_url": "https://api.perplexity.ai",
        "models": [
            {"id": "sonar", "display": "Sonar"},
            {"id": "sonar-pro", "display": "Sonar Pro"},
        ],
    },
    "cohere": {
        "label": "Cohere",
        "env_key": "COHERE_API_KEY",
        "models": [
            {"id": "command-r", "display": "Command R"},
            {"id": "command-r-plus", "display": "Command R+"},
        ],
    },
}

@dataclass
class ChatMessage:
    role: str
    content: str
    timestamp: datetime

    def to_display(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{time_str}] {self.role.title()}: {self.content}"


@dataclass
class Task:
    title: str
    completed: bool = False

    def to_payload(self) -> dict[str, str | bool]:
        return {"title": self.title, "completed": self.completed}

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "Task":
        title = str(payload.get("title", "")).strip()
        return cls(title=title, completed=bool(payload.get("completed", False)))


class AgentState(str, Enum):
    RUNNING = "Running"
    PAUSED = "Paused"
    ERROR = "Error"
    COMPLETED = "Completed"


class AgentRunner(QtCore.QObject):
    state_changed = QtCore.Signal(str)
    log_message = QtCore.Signal(str)
    task_completed = QtCore.Signal(str, int)

    def __init__(
        self,
        task_provider: Callable[[], list[Task]],
        interval_ms: int = 2500,
        max_retries: int = 3,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._task_provider = task_provider
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._execute_step)
        self._state = AgentState.PAUSED
        self._cancelled = False
        self._paused = True
        self._step_count = 0
        self._retry_count = 0
        self._max_retries = max_retries
        self._retry_delay_ms = 1200

    @QtCore.Slot()
    def start(self) -> None:
        if self._state == AgentState.RUNNING:
            return
        self._cancelled = False
        self._paused = False
        self._retry_count = 0
        self._step_count = 0
        self._set_state(AgentState.RUNNING)
        self.log_message.emit("Agent loop started.")
        if not self._timer.isActive():
            self._timer.start()
        QtCore.QTimer.singleShot(0, self._execute_step)

    @QtCore.Slot()
    def pause(self) -> None:
        if self._paused or self._state == AgentState.COMPLETED:
            return
        self._paused = True
        self._timer.stop()
        self._set_state(AgentState.PAUSED)
        self.log_message.emit("Agent loop paused.")

    @QtCore.Slot()
    def resume(self) -> None:
        if not self._paused or self._state == AgentState.COMPLETED:
            return
        self._paused = False
        self._set_state(AgentState.RUNNING)
        self.log_message.emit("Agent loop resumed.")
        if not self._timer.isActive():
            self._timer.start()

    @QtCore.Slot()
    def cancel(self) -> None:
        if self._cancelled:
            return
        self._cancelled = True
        self._paused = True
        self._timer.stop()
        self._set_state(AgentState.PAUSED)
        self.log_message.emit("Agent loop cancelled by user.")

    def _set_state(self, state: AgentState) -> None:
        if self._state == state:
            return
        self._state = state
        self.state_changed.emit(state.value)

    def _execute_step(self) -> None:
        if self._cancelled or self._paused:
            return
        try:
            tasks = self._task_provider()
            pending = [task for task in tasks if not task.completed]
            if not pending:
                self._timer.stop()
                self._set_state(AgentState.COMPLETED)
                self.log_message.emit("All tasks completed. Agent loop finished.")
                return
            self._step_count += 1
            self._run_step(pending[0])
            self._retry_count = 0
        except Exception as exc:
            self._handle_step_error(exc)

    def _run_step(self, task: Task) -> None:
        self.log_message.emit(
            f"Agent step {self._step_count}: focusing on '{task.title}'."
        )
        self.task_completed.emit(task.title, self._step_count)

    def _handle_step_error(self, exc: Exception) -> None:
        self._retry_count += 1
        self._timer.stop()
        self._set_state(AgentState.ERROR)
        self.log_message.emit(f"Agent error: {exc}")
        if self._retry_count <= self._max_retries:
            self.log_message.emit(
                f"Attempting recovery ({self._retry_count}/{self._max_retries})..."
            )
            QtCore.QTimer.singleShot(self._retry_delay_ms, self._recover_from_error)
            return
        self.log_message.emit("Recovery attempts exceeded. Continuing with next step.")
        self._retry_count = 0
        QtCore.QTimer.singleShot(self._retry_delay_ms, self._recover_from_error)

    def _recover_from_error(self) -> None:
        if self._cancelled or self._paused:
            return
        self._set_state(AgentState.RUNNING)
        if not self._timer.isActive():
            self._timer.start()
        self._execute_step()


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


class TaskModel(QtCore.QAbstractTableModel):
    tasks_changed = QtCore.Signal()
    _headers = ("Done", "Task")

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._tasks: list[Task] = []

    def rowCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._tasks)

    def columnCount(self, parent: QtCore.QModelIndex = QtCore.QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._headers)

    def headerData(
        self,
        section: int,
        orientation: QtCore.Qt.Orientation,
        role: int = QtCore.Qt.DisplayRole,
    ):
        if role != QtCore.Qt.DisplayRole or orientation != QtCore.Qt.Horizontal:
            return None
        if 0 <= section < len(self._headers):
            return self._headers[section]
        return None

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._tasks)):
            return None
        task = self._tasks[index.row()]
        if index.column() == 0 and role == QtCore.Qt.CheckStateRole:
            return QtCore.Qt.Checked if task.completed else QtCore.Qt.Unchecked
        if index.column() == 1 and role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            return task.title
        if role == QtCore.Qt.UserRole:
            return task
        return None

    def flags(self, index: QtCore.QModelIndex) -> QtCore.Qt.ItemFlags:
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled
        flags = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable
        if index.column() == 0:
            flags |= QtCore.Qt.ItemIsUserCheckable
        return flags

    def setData(self, index: QtCore.QModelIndex, value, role: int = QtCore.Qt.EditRole) -> bool:
        if not index.isValid() or not (0 <= index.row() < len(self._tasks)):
            return False
        task = self._tasks[index.row()]
        updated = False
        if index.column() == 0 and role == QtCore.Qt.CheckStateRole:
            task.completed = value == QtCore.Qt.Checked
            updated = True
        elif index.column() == 1 and role == QtCore.Qt.EditRole:
            title = str(value).strip()
            if title:
                task.title = title
                updated = True
        if updated:
            self.dataChanged.emit(index, index, [role])
            self.tasks_changed.emit()
        return updated

    def add_task(self, title: str) -> None:
        title = title.strip()
        if not title:
            return
        self.beginInsertRows(QtCore.QModelIndex(), len(self._tasks), len(self._tasks))
        self._tasks.append(Task(title=title))
        self.endInsertRows()
        self.tasks_changed.emit()

    def update_task(self, row: int, title: str) -> None:
        if not (0 <= row < len(self._tasks)):
            return
        title = title.strip()
        if not title:
            return
        self._tasks[row].title = title
        index = self.index(row, 1)
        self.dataChanged.emit(index, index, [QtCore.Qt.DisplayRole, QtCore.Qt.EditRole])
        self.tasks_changed.emit()

    def toggle_complete(self, row: int) -> None:
        if not (0 <= row < len(self._tasks)):
            return
        self._tasks[row].completed = not self._tasks[row].completed
        index = self.index(row, 0)
        self.dataChanged.emit(index, index, [QtCore.Qt.CheckStateRole])
        self.tasks_changed.emit()

    def remove_task(self, row: int) -> None:
        if not (0 <= row < len(self._tasks)):
            return
        self.beginRemoveRows(QtCore.QModelIndex(), row, row)
        self._tasks.pop(row)
        self.endRemoveRows()
        self.tasks_changed.emit()

    def move_task(self, source_row: int, target_row: int) -> None:
        if not (0 <= source_row < len(self._tasks)):
            return
        if not (0 <= target_row < len(self._tasks)):
            return
        if source_row == target_row:
            return
        destination = target_row + (1 if target_row > source_row else 0)
        self.beginMoveRows(QtCore.QModelIndex(), source_row, source_row, QtCore.QModelIndex(), destination)
        task = self._tasks.pop(source_row)
        self._tasks.insert(target_row, task)
        self.endMoveRows()
        self.tasks_changed.emit()

    def set_tasks(self, tasks: list[Task]) -> None:
        self.beginResetModel()
        self._tasks = list(tasks)
        self.endResetModel()
        self.tasks_changed.emit()

    def tasks(self) -> list[Task]:
        return list(self._tasks)


class ComputerControlPanel(QtWidgets.QGroupBox):
    screenshot_captured = QtCore.Signal(QtGui.QPixmap)
    preview_interval_ms = 1000

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Computer Control", parent)
        self._agent_mode_active = False
        self._emergency_stopped = False

        self.status_label = QtWidgets.QLabel("Control disabled.")
        self.status_label.setWordWrap(True)

        self.enable_control_toggle = QtWidgets.QCheckBox("Enable Control")
        self.enable_control_toggle.toggled.connect(self.handle_control_toggle)

        self.emergency_stop_button = QtWidgets.QPushButton("Emergency Stop")
        self.emergency_stop_button.setStyleSheet(
            "background: #a81818; color: white; font-weight: 700;"
        )
        self.emergency_stop_button.clicked.connect(self.handle_emergency_stop)

        self.reset_stop_button = QtWidgets.QPushButton("Reset Stop")
        self.reset_stop_button.setEnabled(False)
        self.reset_stop_button.clicked.connect(self.handle_reset_stop)

        self.screenshot_label = QtWidgets.QLabel()
        self.screenshot_label.setFixedHeight(240)
        self.screenshot_label.setAlignment(QtCore.Qt.AlignCenter)
        self.screenshot_label.setStyleSheet(
            "background: #1f2023; color: #9aa0a6; border-radius: 8px; border: 1px solid #33343a;"
        )
        self.screenshot_label.setText("No preview")

        self.take_screenshot_button = QtWidgets.QPushButton("Take Desktop Snapshot")
        self.take_screenshot_button.clicked.connect(self.handle_screenshot)

        self.preview_toggle_button = QtWidgets.QPushButton("Pause Preview")
        self.preview_toggle_button.setCheckable(True)
        self.preview_toggle_button.toggled.connect(self.handle_preview_toggle)
        self.preview_toggle_button.setEnabled(False)

        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.setInterval(self.preview_interval_ms)
        self.preview_timer.timeout.connect(self.capture_desktop_frame)

        self.delay_min_spin = QtWidgets.QDoubleSpinBox()
        self.delay_min_spin.setRange(0.0, 5.0)
        self.delay_min_spin.setSingleStep(0.05)
        self.delay_min_spin.setValue(0.15)
        self.delay_min_spin.setSuffix(" s")

        self.delay_max_spin = QtWidgets.QDoubleSpinBox()
        self.delay_max_spin.setRange(0.0, 5.0)
        self.delay_max_spin.setSingleStep(0.05)
        self.delay_max_spin.setValue(0.45)
        self.delay_max_spin.setSuffix(" s")

        self.move_duration_spin = QtWidgets.QDoubleSpinBox()
        self.move_duration_spin.setRange(0.0, 3.0)
        self.move_duration_spin.setSingleStep(0.05)
        self.move_duration_spin.setValue(0.2)
        self.move_duration_spin.setSuffix(" s")

        self.typing_interval_spin = QtWidgets.QDoubleSpinBox()
        self.typing_interval_spin.setRange(0.0, 1.0)
        self.typing_interval_spin.setSingleStep(0.01)
        self.typing_interval_spin.setValue(0.05)
        self.typing_interval_spin.setSuffix(" s")

        delay_form = QtWidgets.QFormLayout()
        delay_form.addRow("Action delay min", self.delay_min_spin)
        delay_form.addRow("Action delay max", self.delay_max_spin)
        delay_form.addRow("Move duration", self.move_duration_spin)
        delay_form.addRow("Typing interval", self.typing_interval_spin)

        delay_widget = QtWidgets.QWidget()
        delay_widget.setLayout(delay_form)

        self.activity_log = QtWidgets.QPlainTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setPlaceholderText("Automation log will appear here.")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.enable_control_toggle)
        control_row = QtWidgets.QHBoxLayout()
        control_row.addWidget(self.emergency_stop_button)
        control_row.addWidget(self.reset_stop_button)
        layout.addLayout(control_row)
        layout.addWidget(self.screenshot_label)
        layout.addWidget(self.take_screenshot_button)
        layout.addWidget(self.preview_toggle_button)
        layout.addWidget(delay_widget)
        layout.addWidget(self.activity_log)
        self.screenshot_captured.connect(self.update_preview)

    def set_agent_mode(self, active: bool) -> None:
        self._agent_mode_active = active
        if not active:
            self.preview_timer.stop()
            self.preview_toggle_button.blockSignals(True)
            self.preview_toggle_button.setChecked(True)
            self.preview_toggle_button.setText("Resume Preview")
            self.preview_toggle_button.blockSignals(False)
            self.preview_toggle_button.setEnabled(False)
        else:
            self.preview_toggle_button.setEnabled(True)
            self._update_preview_loop()

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.activity_log.appendPlainText(f"[{timestamp}] {message}")

    def handle_control_toggle(self, enabled: bool) -> None:
        if self._emergency_stopped:
            self.enable_control_toggle.blockSignals(True)
            self.enable_control_toggle.setChecked(False)
            self.enable_control_toggle.blockSignals(False)
            self.log("Control toggle ignored: emergency stop is active.")
            return
        if enabled:
            self.status_label.setText(
                "Control enabled. Keep this window visible so you can monitor automation."
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
        pixmap = self._pixmap_from_screenshot(screenshot)
        self.screenshot_captured.emit(pixmap)

    def handle_preview_toggle(self, paused: bool) -> None:
        if paused:
            self.preview_toggle_button.setText("Resume Preview")
            self.log("Preview paused.")
        else:
            self.preview_toggle_button.setText("Pause Preview")
            self.log("Preview resumed.")
        self._update_preview_loop()

    def capture_desktop_frame(self) -> None:
        if pyautogui is None:
            self.preview_timer.stop()
            self.preview_toggle_button.setChecked(True)
            self.screenshot_label.setText("Preview unavailable")
            self.log("Preview stopped: pyautogui is not installed.")
            return
        screenshot = pyautogui.screenshot()
        pixmap = self._pixmap_from_screenshot(screenshot)
        self.screenshot_captured.emit(pixmap)

    def update_preview(self, pixmap: QtGui.QPixmap) -> None:
        scaled = pixmap.scaled(
            self.screenshot_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.screenshot_label.setPixmap(scaled)
        if not self.preview_timer.isActive():
            self.log("Screenshot updated.")

    def _pixmap_from_screenshot(self, screenshot) -> QtGui.QPixmap:
        image = screenshot.convert("RGBA")
        qt_image = QtGui.QImage(
            image.tobytes("raw", "RGBA"),
            image.width,
            image.height,
            QtGui.QImage.Format_RGBA8888,
        )
        return QtGui.QPixmap.fromImage(qt_image)

    def _update_preview_loop(self) -> None:
        should_run = (
            self._agent_mode_active
            and self.isEnabled()
            and not self.preview_toggle_button.isChecked()
        )
        if should_run and pyautogui is None:
            self.preview_toggle_button.setChecked(True)
            return
        if should_run and not self.preview_timer.isActive():
            self.preview_timer.start()
            self.log("Preview loop started.")
        elif not should_run and self.preview_timer.isActive():
            self.preview_timer.stop()
            self.log("Preview loop stopped.")

    def handle_emergency_stop(self) -> None:
        if self._emergency_stopped:
            return
        self._emergency_stopped = True
        self.enable_control_toggle.blockSignals(True)
        self.enable_control_toggle.setChecked(False)
        self.enable_control_toggle.blockSignals(False)
        self.reset_stop_button.setEnabled(True)
        self.status_label.setText(
            "Emergency stop engaged. Control actions are blocked until reset."
        )
        self.log("Emergency stop activated. All control actions blocked.")

    def handle_reset_stop(self) -> None:
        if not self._emergency_stopped:
            return
        self._emergency_stopped = False
        self.reset_stop_button.setEnabled(False)
        self.status_label.setText("Control disabled.")
        self.log("Emergency stop cleared. Control remains disabled.")

    def _control_allowed(self, action_label: str) -> bool:
        if pyautogui is None:
            self.log(f"{action_label} blocked: pyautogui is not installed.")
            return False
        if self._emergency_stopped:
            self.log(f"{action_label} blocked: emergency stop is active.")
            return False
        if not self.enable_control_toggle.isChecked():
            self.log(f"{action_label} blocked: control is disabled.")
            return False
        return True

    def _action_delay_seconds(self) -> float:
        minimum = self.delay_min_spin.value()
        maximum = self.delay_max_spin.value()
        if maximum < minimum:
            minimum, maximum = maximum, minimum
        if maximum == 0:
            return 0.0
        return random.uniform(minimum, maximum)

    def _apply_action_delay(self) -> None:
        delay = self._action_delay_seconds()
        if delay > 0:
            time.sleep(delay)

    def move_mouse(self, x: int, y: int, duration: Optional[float] = None) -> bool:
        if not self._control_allowed("Move mouse"):
            return False
        move_duration = self.move_duration_spin.value() if duration is None else duration
        self.log(f"Moving mouse to ({x}, {y}) over {move_duration:.2f}s.")
        self._apply_action_delay()
        pyautogui.moveTo(x, y, duration=move_duration)
        return True

    def click_mouse(
        self,
        button: str = "left",
        clicks: int = 1,
        interval: float = 0.0,
    ) -> bool:
        if not self._control_allowed("Mouse click"):
            return False
        self.log(f"Clicking mouse: button={button}, clicks={clicks}.")
        self._apply_action_delay()
        pyautogui.click(button=button, clicks=clicks, interval=interval)
        return True

    def type_text(self, text: str) -> bool:
        if not self._control_allowed("Type text"):
            return False
        if not text:
            self.log("Type text skipped: empty input.")
            return False
        interval = self.typing_interval_spin.value()
        self.log(f"Typing {len(text)} characters at {interval:.2f}s interval.")
        self._apply_action_delay()
        pyautogui.write(text, interval=interval)
        return True

    def press_key(self, key: str) -> bool:
        if not self._control_allowed("Key press"):
            return False
        self.log(f"Pressing key: {key}.")
        self._apply_action_delay()
        pyautogui.press(key)
        return True


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

        self._memory = self._load_memory()
        self._load_api_keys()

        self.chat_model = ChatModel(self)
        self.task_model = TaskModel(self)
        self.task_model.tasks_changed.connect(self._save_tasks)
        self._task_snapshot_lock = threading.Lock()
        self._task_snapshot: list[Task] = []
        self.task_model.tasks_changed.connect(self._refresh_task_snapshot)
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

        self.provider_label = QtWidgets.QLabel("Provider")
        self.provider_selector = QtWidgets.QComboBox()
        self.provider_selector.currentIndexChanged.connect(self._handle_provider_change)

        self.model_label = QtWidgets.QLabel("Model")
        self.model_selector = QtWidgets.QComboBox()
        self.model_selector.currentIndexChanged.connect(self._handle_model_change)
        self._populate_provider_selector()

        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.clicked.connect(self.handle_send)

        self.controls_panel = ComputerControlPanel()
        self.task_panel = QtWidgets.QGroupBox("Tasks")
        self.task_view = QtWidgets.QTableView()
        self.task_view.setModel(self.task_model)
        self.task_view.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.task_view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.task_view.setEditTriggers(QtWidgets.QAbstractItemView.SelectedClicked)
        self.task_view.verticalHeader().setVisible(False)
        self.task_view.horizontalHeader().setStretchLastSection(True)
        self.task_view.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        self.task_view.setAlternatingRowColors(True)

        self.add_task_button = QtWidgets.QPushButton("Add")
        self.add_task_button.clicked.connect(self._add_task)
        self.edit_task_button = QtWidgets.QPushButton("Edit")
        self.edit_task_button.clicked.connect(self._edit_task)
        self.remove_task_button = QtWidgets.QPushButton("Remove")
        self.remove_task_button.clicked.connect(self._remove_task)
        self.move_up_button = QtWidgets.QPushButton("Move Up")
        self.move_up_button.clicked.connect(lambda: self._move_task(-1))
        self.move_down_button = QtWidgets.QPushButton("Move Down")
        self.move_down_button.clicked.connect(lambda: self._move_task(1))
        self.complete_task_button = QtWidgets.QPushButton("Toggle Complete")
        self.complete_task_button.clicked.connect(self._toggle_complete)

        self.agent_toggle_button = QtWidgets.QPushButton("Start Agent Loop")
        self.agent_toggle_button.setCheckable(True)
        self.agent_toggle_button.toggled.connect(self._toggle_agent_loop)
        self._agent_thread = QtCore.QThread(self)
        self._agent_runner = AgentRunner(self._get_task_snapshot)
        self._agent_runner.moveToThread(self._agent_thread)
        self._agent_runner.log_message.connect(self.controls_panel.log)
        self._agent_runner.state_changed.connect(self._handle_agent_state)
        self._agent_runner.task_completed.connect(self._prompt_task_feedback)
        self._agent_thread.start()

        self.api_status = QtWidgets.QLabel()
        self.api_status.setText(self._api_status_text())
        self.api_status.setStyleSheet("color: #9aa0a6; font-size: 12px;")

        header_title = QtWidgets.QLabel("KaliGPT")
        header_title.setObjectName("HeaderTitle")
        header_subtitle = QtWidgets.QLabel("Your AI assistant for secure workflows")
        header_subtitle.setObjectName("HeaderSubtitle")
        self.preferences_label = QtWidgets.QLabel()
        self.preferences_label.setObjectName("HeaderPreferences")
        self._update_preferences_label()

        header_layout = QtWidgets.QVBoxLayout()
        header_layout.addWidget(header_title)
        header_layout.addWidget(header_subtitle)
        header_layout.addWidget(self.preferences_label)
        header_layout.setSpacing(2)

        header_widget = QtWidgets.QWidget()
        header_widget.setLayout(header_layout)

        chat_layout = QtWidgets.QVBoxLayout()
        chat_layout.addWidget(header_widget)
        chat_layout.addWidget(self.chat_view)

        input_layout = QtWidgets.QHBoxLayout()
        input_layout.addWidget(self.message_input, stretch=1)
        input_layout.addWidget(self.send_button)

        model_layout = QtWidgets.QHBoxLayout()
        model_layout.addWidget(self.provider_label)
        model_layout.addWidget(self.provider_selector)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_selector)
        model_layout.addStretch()

        chat_layout.addLayout(input_layout)
        chat_layout.addLayout(model_layout)
        chat_layout.addWidget(self.api_status)

        task_button_layout = QtWidgets.QGridLayout()
        task_button_layout.addWidget(self.add_task_button, 0, 0)
        task_button_layout.addWidget(self.edit_task_button, 0, 1)
        task_button_layout.addWidget(self.remove_task_button, 0, 2)
        task_button_layout.addWidget(self.move_up_button, 1, 0)
        task_button_layout.addWidget(self.move_down_button, 1, 1)
        task_button_layout.addWidget(self.complete_task_button, 1, 2)

        task_layout = QtWidgets.QVBoxLayout(self.task_panel)
        task_layout.addWidget(self.task_view)
        task_layout.addLayout(task_button_layout)
        task_layout.addWidget(self.agent_toggle_button)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.task_panel)
        right_layout.addWidget(self.controls_panel)
        right_layout.addStretch()

        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_layout)

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addLayout(chat_layout, stretch=3)
        main_layout.addWidget(right_widget, stretch=2)

        self._apply_theme()
        self._configure_mode()
        self._load_history()
        self._load_tasks()
        self._refresh_task_snapshot()
        self._refresh_task_controls()
        self.task_view.selectionModel().selectionChanged.connect(
            lambda *_: self._refresh_task_controls()
        )

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
            QLabel#HeaderPreferences {
                color: #9aa0a6;
                font-size: 12px;
            }
            QTableView {
                background: #202123;
                border: 1px solid #3e3f4b;
                border-radius: 8px;
                gridline-color: #3e3f4b;
            }
            QHeaderView::section {
                background: #2b2c2f;
                color: #e8e8e8;
                padding: 6px;
                border: none;
                font-weight: 600;
            }
            """
        )

    def _configure_mode(self) -> None:
        is_agent_mode = self.mode == "agent"
        self.controls_panel.setVisible(is_agent_mode)
        self.controls_panel.setEnabled(is_agent_mode)
        self.controls_panel.set_agent_mode(is_agent_mode)
        if not is_agent_mode:
            self.controls_panel.setChecked(False)
            self.agent_toggle_button.setChecked(False)
            self.agent_toggle_button.setEnabled(False)
        else:
            self.agent_toggle_button.setEnabled(True)

    def _api_status_text(self) -> str:
        provider = self._selected_provider()
        if not provider:
            return "No provider selected."
        provider_info = PROVIDER_REGISTRY[provider]
        env_key = provider_info["env_key"]
        if provider in {"openai", "deepseek", "groq", "mistral", "perplexity"}:
            if openai is None:
                return "OpenAI client not installed. Responses will be stubbed."
        if provider == "anthropic":
            if anthropic is None:
                return "Anthropic client not installed. Responses will be stubbed."
        if provider == "gemini":
            if self._optional_module("google.generativeai") is None:
                return "Google Generative AI client not installed. Responses will be stubbed."
        if provider == "cohere":
            if self._optional_module("cohere") is None:
                return "Cohere client not installed. Responses will be stubbed."
        if not os.getenv(env_key):
            return f"Set {env_key} to enable {provider_info['label']} responses."
        return f"Connected to {provider_info['label']} API."

    def _populate_provider_selector(self) -> None:
        self.provider_selector.clear()
        preferences = self._memory.get("preferences", {})
        preferred_provider = None
        if isinstance(preferences, dict):
            preferred_provider = preferences.get("provider")
        providers = list(PROVIDER_REGISTRY.keys())
        selected_index = 0
        for index, provider in enumerate(providers):
            self.provider_selector.addItem(PROVIDER_REGISTRY[provider]["label"], provider)
            if provider == preferred_provider:
                selected_index = index
        self.provider_selector.setCurrentIndex(selected_index)
        self._populate_model_selector()
        self._persist_model_selection()

    def _populate_model_selector(self) -> None:
        provider = self._selected_provider()
        if not provider:
            return
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        models = PROVIDER_REGISTRY[provider]["models"]
        preferences = self._memory.get("preferences", {})
        model_map = {}
        if isinstance(preferences, dict):
            model_map = preferences.get("model_map", {}) or {}
        preferred_model = model_map.get(provider)
        selected_index = 0
        for index, model in enumerate(models):
            self.model_selector.addItem(model["display"], model["id"])
            if model["id"] == preferred_model:
                selected_index = index
        self.model_selector.setCurrentIndex(selected_index)
        self.model_selector.blockSignals(False)

    def _selected_provider(self) -> str | None:
        data = self.provider_selector.currentData()
        if isinstance(data, str):
            return data
        return None

    def _selected_model_id(self) -> str:
        data = self.model_selector.currentData()
        if isinstance(data, str):
            return data
        provider = self._selected_provider()
        if provider:
            return PROVIDER_REGISTRY[provider]["models"][0]["id"]
        return ""

    def _persist_model_selection(self) -> None:
        preferences = self._memory.setdefault("preferences", {})
        if not isinstance(preferences, dict):
            preferences = {}
            self._memory["preferences"] = preferences
        provider = self._selected_provider()
        if provider:
            preferences["provider"] = provider
            model_map = preferences.get("model_map")
            if not isinstance(model_map, dict):
                model_map = {}
            model_map[provider] = self._selected_model_id()
            preferences["model_map"] = model_map
        self._save_memory()
        self._update_preferences_label()

    def _handle_provider_change(self, *_: object) -> None:
        self._populate_model_selector()
        self._persist_model_selection()
        self._maybe_prompt_api_key()
        self.api_status.setText(self._api_status_text())

    def _handle_model_change(self, *_: object) -> None:
        self._persist_model_selection()
        self.api_status.setText(self._api_status_text())

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

    def _memory_path(self) -> Path:
        return Path.home() / ".kaligpt" / "memory.json"

    def _tasks_path(self) -> Path:
        return Path.home() / ".kaligpt" / "tasks.json"

    def _load_memory(self) -> dict[str, object]:
        memory_path = self._memory_path()
        if not memory_path.exists():
            return {"preferences": {}, "task_outcomes": [], "failures": [], "api_keys": {}}
        try:
            with memory_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return {
                "preferences": dict(payload.get("preferences", {})),
                "task_outcomes": list(payload.get("task_outcomes", [])),
                "failures": list(payload.get("failures", [])),
                "api_keys": dict(payload.get("api_keys", {})),
            }
        except (OSError, ValueError, TypeError):
            return {"preferences": {}, "task_outcomes": [], "failures": [], "api_keys": {}}

    def _save_memory(self) -> None:
        memory_path = self._memory_path()
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        with memory_path.open("w", encoding="utf-8") as handle:
            json.dump(self._memory, handle, indent=2)

    def _load_api_keys(self) -> None:
        api_keys = self._memory.get("api_keys", {})
        if not isinstance(api_keys, dict):
            return
        for provider, key in api_keys.items():
            if provider in PROVIDER_REGISTRY and isinstance(key, str) and key:
                env_key = PROVIDER_REGISTRY[provider]["env_key"]
                if not os.getenv(env_key):
                    os.environ[env_key] = key

    def _maybe_prompt_api_key(self) -> None:
        provider = self._selected_provider()
        if not provider:
            return
        provider_info = PROVIDER_REGISTRY[provider]
        env_key = provider_info["env_key"]
        if os.getenv(env_key):
            return
        label = provider_info["label"]
        prompt = f"Enter {label} API key ({env_key})"
        key, ok = QtWidgets.QInputDialog.getText(
            self,
            "API Key Required",
            prompt,
            QtWidgets.QLineEdit.Password,
        )
        if not ok:
            return
        key = key.strip()
        if not key:
            return
        os.environ[env_key] = key
        api_keys = self._memory.setdefault("api_keys", {})
        if isinstance(api_keys, dict):
            api_keys[provider] = key
        self._save_memory()

    def _optional_module(self, module_name: str):
        if importlib.util.find_spec(module_name) is None:
            return None
        return importlib.import_module(module_name)

    def _summarize_preferences(self) -> str:
        preferences = self._memory.get("preferences", {})
        if not isinstance(preferences, dict) or not preferences:
            return "Preferences: none saved"
        provider = preferences.get("provider")
        model_map = preferences.get("model_map")
        model = None
        if isinstance(model_map, dict) and provider in model_map:
            model = model_map.get(provider)
        if provider and model:
            return f"Preferences: provider={provider}, model={model}"
        return f"Preferences: provider={provider or 'none'}"

    def _update_preferences_label(self) -> None:
        self.preferences_label.setText(self._summarize_preferences())

    def _load_tasks(self) -> None:
        tasks_path = self._tasks_path()
        if not tasks_path.exists():
            return
        try:
            with tasks_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            tasks: list[Task] = []
            for item in payload:
                task = Task.from_payload(item)
                if task.title:
                    tasks.append(task)
            self.task_model.set_tasks(tasks)
        except (OSError, ValueError, TypeError):
            return

    def _save_tasks(self) -> None:
        tasks_path = self._tasks_path()
        tasks_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [task.to_payload() for task in self.task_model.tasks()]
        with tasks_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _find_task_row(self, title: str) -> int | None:
        for index, task in enumerate(self.task_model.tasks()):
            if not task.completed and task.title == title:
                return index
        return None

    @QtCore.Slot(str, int)
    def _prompt_task_feedback(self, title: str, step_count: int) -> None:
        prompt = f"Was task '{title}' successful?"
        response = QtWidgets.QMessageBox.question(
            self,
            "Task Feedback",
            prompt,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        success = response == QtWidgets.QMessageBox.Yes
        summary, ok = QtWidgets.QInputDialog.getText(
            self,
            "Task Feedback",
            "What worked or failed?",
        )
        summary_text = summary.strip() if ok else ""
        self._record_task_feedback(title, step_count, success, summary_text)

    def _record_task_feedback(
        self,
        title: str,
        step_count: int,
        success: bool,
        summary: str,
    ) -> None:
        timestamp = datetime.now().isoformat()
        outcome = {
            "title": title,
            "step": step_count,
            "success": success,
            "summary": summary,
            "timestamp": timestamp,
        }
        self._memory.setdefault("task_outcomes", []).append(outcome)
        if not success:
            self._memory.setdefault("failures", []).append(
                {"title": title, "summary": summary, "timestamp": timestamp}
            )
        self._save_memory()
        status = "succeeded" if success else "failed"
        message = f"Feedback recorded: '{title}' {status}."
        if summary:
            message = f"{message} Summary: {summary}"
        self.controls_panel.log(message)
        if success:
            row = self._find_task_row(title)
            if row is not None:
                self.task_model.toggle_complete(row)
                self._refresh_task_controls()

    def _refresh_task_snapshot(self) -> None:
        with self._task_snapshot_lock:
            self._task_snapshot = self.task_model.tasks()

    def _get_task_snapshot(self) -> list[Task]:
        with self._task_snapshot_lock:
            return list(self._task_snapshot)

    def _selected_task_row(self) -> int | None:
        selection = self.task_view.selectionModel()
        if selection is None:
            return None
        indexes = selection.selectedRows()
        if not indexes:
            return None
        return indexes[0].row()

    def _refresh_task_controls(self) -> None:
        row = self._selected_task_row()
        has_row = row is not None
        self.edit_task_button.setEnabled(has_row)
        self.remove_task_button.setEnabled(has_row)
        self.move_up_button.setEnabled(has_row and row > 0)
        self.move_down_button.setEnabled(has_row and row is not None and row < self.task_model.rowCount() - 1)
        self.complete_task_button.setEnabled(has_row)

    def _add_task(self) -> None:
        title, ok = QtWidgets.QInputDialog.getText(self, "Add Task", "Task")
        if not ok:
            return
        self.task_model.add_task(title)
        self.task_view.scrollToBottom()
        self._refresh_task_controls()

    def _edit_task(self) -> None:
        row = self._selected_task_row()
        if row is None:
            return
        task = self.task_model.tasks()[row]
        title, ok = QtWidgets.QInputDialog.getText(
            self, "Edit Task", "Task", text=task.title
        )
        if not ok:
            return
        self.task_model.update_task(row, title)
        self._refresh_task_controls()

    def _remove_task(self) -> None:
        row = self._selected_task_row()
        if row is None:
            return
        self.task_model.remove_task(row)
        self._refresh_task_controls()

    def _move_task(self, direction: int) -> None:
        row = self._selected_task_row()
        if row is None:
            return
        target = row + direction
        if not (0 <= target < self.task_model.rowCount()):
            return
        self.task_model.move_task(row, target)
        self.task_view.selectRow(target)
        self._refresh_task_controls()

    def _toggle_complete(self) -> None:
        row = self._selected_task_row()
        if row is None:
            return
        self.task_model.toggle_complete(row)
        self._refresh_task_controls()

    def _toggle_agent_loop(self, running: bool) -> None:
        if running:
            self.agent_toggle_button.setText("Stop Agent Loop")
            QtCore.QMetaObject.invokeMethod(
                self._agent_runner, "start", QtCore.Qt.QueuedConnection
            )
        else:
            self.agent_toggle_button.setText("Start Agent Loop")
            QtCore.QMetaObject.invokeMethod(
                self._agent_runner, "cancel", QtCore.Qt.QueuedConnection
            )

    def _handle_agent_state(self, state: str) -> None:
        if state == AgentState.COMPLETED.value:
            self.agent_toggle_button.blockSignals(True)
            self.agent_toggle_button.setChecked(False)
            self.agent_toggle_button.blockSignals(False)
            self.agent_toggle_button.setText("Start Agent Loop")
        elif state == AgentState.PAUSED.value:
            self.agent_toggle_button.setText("Start Agent Loop")
        elif state == AgentState.ERROR.value:
            self.agent_toggle_button.setText("Stop Agent Loop (recovering)")
        elif state == AgentState.RUNNING.value:
            self.agent_toggle_button.setText("Stop Agent Loop")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        self._save_history()
        self._save_tasks()
        self._save_memory()
        QtCore.QMetaObject.invokeMethod(
            self._agent_runner, "cancel", QtCore.Qt.BlockingQueuedConnection
        )
        self._agent_thread.quit()
        self._agent_thread.wait()
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
        provider = self._selected_provider()
        if not provider:
            return "Select a provider to start chatting."
        model_id = self._selected_model_id()
        provider_info = PROVIDER_REGISTRY[provider]
        env_key = provider_info["env_key"]

        if provider in {"openai", "deepseek", "groq", "mistral", "perplexity"}:
            if openai is None:
                return (
                    "I'm ready to help. Install the OpenAI SDK to enable "
                    "live model responses."
                )
            self._maybe_prompt_api_key()
            api_key = os.getenv(env_key)
            if not api_key:
                return f"Set {env_key} to enable {provider_info['label']} responses."
            client_kwargs = {"api_key": api_key}
            base_url = provider_info.get("base_url")
            if base_url:
                client_kwargs["base_url"] = base_url
            client = openai.OpenAI(**client_kwargs)
            try:
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=self.chat_model.as_openai_messages(),
                )
                return completion.choices[0].message.content
            except Exception as exc:  # pragma: no cover - network call
                return f"API error: {exc}"

        if provider == "anthropic":
            if anthropic is None:
                return (
                    "I'm ready to help. Install the Anthropic SDK to enable "
                    "live model responses."
                )
            self._maybe_prompt_api_key()
            if not os.getenv(env_key):
                return f"Set {env_key} to enable {provider_info['label']} responses."
            client = anthropic.Anthropic(api_key=os.getenv(env_key))
            try:
                response = client.messages.create(
                    model=model_id,
                    max_tokens=1024,
                    messages=self.chat_model.as_openai_messages(),
                )
                if response.content:
                    return response.content[0].text
                return "No response content returned from Anthropic."
            except Exception as exc:  # pragma: no cover - network call
                return f"API error: {exc}"

        if provider == "gemini":
            self._maybe_prompt_api_key()
            api_key = os.getenv(env_key)
            if not api_key:
                return f"Set {env_key} to enable {provider_info['label']} responses."
            genai = self._optional_module("google.generativeai")
            if genai is None:
                return (
                    "Install the Google Generative AI SDK (google-generativeai) "
                    "to enable Gemini responses."
                )
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_id)
            prompt = "\n".join(
                f"{msg['role']}: {msg['content']}"
                for msg in self.chat_model.as_openai_messages()
            )
            try:
                response = model.generate_content(prompt)
                return response.text if response.text else "No response content returned from Gemini."
            except Exception as exc:  # pragma: no cover - network call
                return f"API error: {exc}"

        if provider == "cohere":
            self._maybe_prompt_api_key()
            api_key = os.getenv(env_key)
            if not api_key:
                return f"Set {env_key} to enable {provider_info['label']} responses."
            cohere = self._optional_module("cohere")
            if cohere is None:
                return (
                    "Install the Cohere SDK (cohere) to enable Cohere responses."
                )
            client = cohere.Client(api_key)
            prompt = "\n".join(
                f"{msg['role']}: {msg['content']}"
                for msg in self.chat_model.as_openai_messages()
            )
            try:
                response = client.chat(model=model_id, message=prompt)
                return response.text if response.text else "No response content returned from Cohere."
            except Exception as exc:  # pragma: no cover - network call
                return f"API error: {exc}"

        return f"No client available for provider '{provider}'."


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
