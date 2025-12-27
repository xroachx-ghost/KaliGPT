"""KaliGPT desktop controller GUI."""
from __future__ import annotations

import csv
import importlib
import json
import os
import random
import re
import shutil
import subprocess
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


class ProviderHealthState(Enum):
    UNKNOWN = "unknown"
    OK = "ok"
    ERROR = "error"
    RETRYING = "retrying"


@dataclass
class ProviderHealth:
    state: ProviderHealthState = ProviderHealthState.UNKNOWN
    last_error_time: Optional[datetime] = None
    last_error_message: Optional[str] = None
    retry_after_seconds: Optional[float] = None


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


@dataclass
class RoutineAction:
    action: str
    parameters: dict[str, object]
    condition: Optional[str] = None

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "action": self.action,
            "parameters": self.parameters,
        }
        if self.condition:
            payload["condition"] = self.condition
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> Optional["RoutineAction"]:
        action = payload.get("action")
        parameters = payload.get("parameters")
        if not isinstance(action, str) or not isinstance(parameters, dict):
            return None
        condition_value = payload.get("condition")
        condition = str(condition_value).strip() if condition_value else None
        return cls(action=action, parameters=parameters, condition=condition)


@dataclass
class Routine:
    name: str
    actions: list[RoutineAction]
    created_at: str

    def to_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "created_at": self.created_at,
            "actions": [action.to_payload() for action in self.actions],
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> Optional["Routine"]:
        name = str(payload.get("name", "")).strip()
        actions_raw = payload.get("actions", [])
        created_at = str(payload.get("created_at", "")).strip()
        if not name or not isinstance(actions_raw, list):
            return None
        actions: list[RoutineAction] = []
        for item in actions_raw:
            if isinstance(item, dict):
                action = RoutineAction.from_payload(item)
                if action:
                    actions.append(action)
        if not actions:
            return None
        return cls(name=name, actions=actions, created_at=created_at or datetime.now().isoformat())


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
    SEARCH_ROLE = QtCore.Qt.UserRole + 1

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._messages: list[ChatMessage] = []
        self._search_query = ""

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
        if role == self.SEARCH_ROLE:
            return self._search_query
        return None

    def add_message(self, message: ChatMessage) -> None:
        self.beginInsertRows(QtCore.QModelIndex(), len(self._messages), len(self._messages))
        self._messages.append(message)
        self.endInsertRows()

    def set_messages(self, messages: list[ChatMessage]) -> None:
        self.beginResetModel()
        self._messages = list(messages)
        self.endResetModel()

    def set_search_query(self, query: str) -> None:
        self._search_query = query
        if self._messages:
            top = self.index(0, 0)
            bottom = self.index(len(self._messages) - 1, 0)
            self.dataChanged.emit(top, bottom, [self.SEARCH_ROLE])

    def search_query(self) -> str:
        return self._search_query

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
    export_audit_requested = QtCore.Signal()
    manual_action_triggered = QtCore.Signal(dict)
    preview_interval_ms = 1000
    AUTOMATION_UNAVAILABLE_MESSAGE = (
        "Automation unavailable. Install pyautogui and grant OS accessibility permissions "
        "to enable desktop control."
    )
    AUTOMATION_UNAVAILABLE_TOOLTIP = (
        "Desktop control requires pyautogui.\n"
        "Install it with: pip install pyautogui\n"
        "Then grant accessibility/automation permissions in your OS settings."
    )

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Computer Control", parent)
        self._agent_mode_active = False
        self._emergency_stopped = False
        self._automation_message_logged = False

        self.status_label = QtWidgets.QLabel("Control disabled.")
        self.status_label.setWordWrap(True)

        self.automation_banner = QtWidgets.QLabel(self.AUTOMATION_UNAVAILABLE_MESSAGE)
        self.automation_banner.setWordWrap(True)
        self.automation_banner.setStyleSheet(
            "background: #3b2f1b; color: #f6d26b; border-radius: 6px; padding: 6px;"
        )
        self.automation_banner.setVisible(False)

        self.enable_control_toggle = QtWidgets.QCheckBox("Enable Control")
        self.enable_control_toggle.toggled.connect(self.handle_control_toggle)

        self.batch_review_toggle = QtWidgets.QCheckBox(
            "Review action batches before execution"
        )
        self.batch_review_toggle.setChecked(True)

        self.dry_run_toggle = QtWidgets.QCheckBox("Dry run (log actions only)")
        self.dry_run_toggle.setToolTip(
            "When enabled, actions are logged but not executed."
        )
        self.dry_run_toggle.toggled.connect(lambda _: self._update_permission_summary())

        self.screenshot_context_toggle = QtWidgets.QCheckBox(
            "Use screenshot context before responses"
        )
        self.screenshot_context_toggle.setToolTip(
            "Capture a screenshot, analyze it, then respond with actions."
        )

        self.permission_scope_panel = QtWidgets.QGroupBox("Permission Scope")
        self.permission_scope_label = QtWidgets.QLabel()
        self.permission_scope_label.setWordWrap(True)
        permission_layout = QtWidgets.QVBoxLayout(self.permission_scope_panel)
        permission_layout.addWidget(self.permission_scope_label)

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

        self.manual_controls_group = QtWidgets.QGroupBox("Manual Actions")

        self.mouse_x_spin = QtWidgets.QSpinBox()
        self.mouse_x_spin.setRange(0, 10_000)
        self.mouse_y_spin = QtWidgets.QSpinBox()
        self.mouse_y_spin.setRange(0, 10_000)

        coordinate_row = QtWidgets.QHBoxLayout()
        coordinate_row.addWidget(QtWidgets.QLabel("X"))
        coordinate_row.addWidget(self.mouse_x_spin)
        coordinate_row.addSpacing(8)
        coordinate_row.addWidget(QtWidgets.QLabel("Y"))
        coordinate_row.addWidget(self.mouse_y_spin)
        coordinate_row.addStretch()

        self.move_mouse_button = QtWidgets.QPushButton("Move Mouse")
        self.move_mouse_button.clicked.connect(self.handle_move_mouse)

        self.click_button_combo = QtWidgets.QComboBox()
        self.click_button_combo.addItem("Left", "left")
        self.click_button_combo.addItem("Right", "right")
        self.click_button_combo.addItem("Middle", "middle")

        self.click_count_spin = QtWidgets.QSpinBox()
        self.click_count_spin.setRange(1, 10)
        self.click_count_spin.setValue(1)

        click_row = QtWidgets.QHBoxLayout()
        click_row.addWidget(self.click_button_combo)
        click_row.addWidget(QtWidgets.QLabel("Clicks"))
        click_row.addWidget(self.click_count_spin)
        click_row.addStretch()

        self.click_mouse_button = QtWidgets.QPushButton("Click Mouse")
        self.click_mouse_button.clicked.connect(self.handle_click_mouse)

        self.type_text_input = QtWidgets.QPlainTextEdit()
        self.type_text_input.setPlaceholderText("Enter text to type")
        self.type_text_input.setFixedHeight(70)

        self.type_text_button = QtWidgets.QPushButton("Type Text")
        self.type_text_button.clicked.connect(self.handle_type_text)

        self.key_input = QtWidgets.QLineEdit()
        self.key_input.setPlaceholderText("e.g., enter, esc, ctrl")

        self.press_key_button = QtWidgets.QPushButton("Press Key")
        self.press_key_button.clicked.connect(self.handle_press_key)

        key_row = QtWidgets.QHBoxLayout()
        key_row.addWidget(self.key_input)
        key_row.addWidget(self.press_key_button)

        manual_layout = QtWidgets.QGridLayout(self.manual_controls_group)
        manual_layout.setColumnStretch(1, 1)
        manual_layout.addWidget(QtWidgets.QLabel("Move to"), 0, 0)
        manual_layout.addLayout(coordinate_row, 0, 1)
        manual_layout.addWidget(self.move_mouse_button, 1, 1)
        manual_layout.addWidget(QtWidgets.QLabel("Click"), 2, 0)
        manual_layout.addLayout(click_row, 2, 1)
        manual_layout.addWidget(self.click_mouse_button, 3, 1)
        manual_layout.addWidget(QtWidgets.QLabel("Type text"), 4, 0)
        manual_layout.addWidget(self.type_text_input, 4, 1)
        manual_layout.addWidget(self.type_text_button, 5, 1)
        manual_layout.addWidget(QtWidgets.QLabel("Press key"), 6, 0)
        manual_layout.addLayout(key_row, 6, 1)

        self.routines_group = QtWidgets.QGroupBox("Routines")
        self.routine_selector = QtWidgets.QComboBox()
        self.routine_selector.addItem("No routines saved", None)
        self.record_routine_button = QtWidgets.QPushButton("Record Routine")
        self.record_routine_button.setCheckable(True)
        self.save_routine_button = QtWidgets.QPushButton("Save Routine")
        self.run_routine_button = QtWidgets.QPushButton("Run Routine")
        self.run_routine_button.setEnabled(False)

        routine_buttons = QtWidgets.QHBoxLayout()
        routine_buttons.addWidget(self.record_routine_button)
        routine_buttons.addWidget(self.save_routine_button)
        routine_buttons.addWidget(self.run_routine_button)

        routine_layout = QtWidgets.QVBoxLayout(self.routines_group)
        routine_layout.addWidget(self.routine_selector)
        routine_layout.addLayout(routine_buttons)

        self.activity_log = QtWidgets.QPlainTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setPlaceholderText("Automation log will appear here.")

        self.export_audit_button = QtWidgets.QPushButton("Export Audit Log")
        self.export_audit_button.clicked.connect(self.export_audit_requested.emit)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.automation_banner)
        layout.addWidget(self.enable_control_toggle)
        layout.addWidget(self.batch_review_toggle)
        layout.addWidget(self.dry_run_toggle)
        layout.addWidget(self.screenshot_context_toggle)
        layout.addWidget(self.permission_scope_panel)
        control_row = QtWidgets.QHBoxLayout()
        control_row.addWidget(self.emergency_stop_button)
        control_row.addWidget(self.reset_stop_button)
        layout.addLayout(control_row)
        layout.addWidget(self.screenshot_label)
        layout.addWidget(self.take_screenshot_button)
        layout.addWidget(self.preview_toggle_button)
        layout.addWidget(delay_widget)
        layout.addWidget(self.manual_controls_group)
        layout.addWidget(self.routines_group)
        layout.addWidget(self.activity_log)
        layout.addWidget(self.export_audit_button)
        self.screenshot_captured.connect(self.update_preview)
        self._update_automation_availability(log_message=True)
        self._update_permission_summary()

    def set_agent_mode(self, active: bool) -> None:
        self._agent_mode_active = active
        self._update_automation_availability(log_message=active)
        if not active:
            self.preview_timer.stop()
            self.preview_toggle_button.blockSignals(True)
            self.preview_toggle_button.setChecked(True)
            self.preview_toggle_button.setText("Resume Preview")
            self.preview_toggle_button.blockSignals(False)
            self.preview_toggle_button.setEnabled(False)
        else:
            if pyautogui is not None:
                self.preview_toggle_button.setEnabled(True)
                self._update_preview_loop()

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.activity_log.appendPlainText(f"[{timestamp}] {message}")

    def is_dry_run_enabled(self) -> bool:
        return self.dry_run_toggle.isChecked()

    def is_screenshot_context_enabled(self) -> bool:
        return self.screenshot_context_toggle.isChecked()

    def set_routines(self, routines: list[Routine]) -> None:
        self.routine_selector.blockSignals(True)
        self.routine_selector.clear()
        if not routines:
            self.routine_selector.addItem("No routines saved", None)
            self.run_routine_button.setEnabled(False)
        else:
            for routine in routines:
                label = f"{routine.name} ({len(routine.actions)} actions)"
                self.routine_selector.addItem(label, routine.name)
            self.run_routine_button.setEnabled(True)
        self.routine_selector.blockSignals(False)

    def selected_routine_name(self) -> Optional[str]:
        data = self.routine_selector.currentData()
        return data if isinstance(data, str) else None

    def set_recording_state(self, recording: bool) -> None:
        if recording:
            self.record_routine_button.setText("Stop Recording")
        else:
            self.record_routine_button.setText("Record Routine")

    def handle_control_toggle(self, enabled: bool) -> None:
        if pyautogui is None:
            self.enable_control_toggle.blockSignals(True)
            self.enable_control_toggle.setChecked(False)
            self.enable_control_toggle.blockSignals(False)
            self._note_automation_unavailable()
            self._update_permission_summary()
            return
        if self._emergency_stopped:
            self.enable_control_toggle.blockSignals(True)
            self.enable_control_toggle.setChecked(False)
            self.enable_control_toggle.blockSignals(False)
            self.log("Control toggle ignored: emergency stop is active.")
            self._update_permission_summary()
            return
        if enabled:
            self.status_label.setText(
                "Control enabled. Keep this window visible so you can monitor automation."
            )
            self.log("Control enabled.")
        else:
            self.status_label.setText("Control disabled.")
            self.log("Control disabled.")
        self._update_permission_summary()

    def handle_screenshot(self) -> None:
        if pyautogui is None:
            self._note_automation_unavailable()
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
            self._note_automation_unavailable()
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
            self._note_automation_unavailable()
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
        self._update_permission_summary()

    def handle_reset_stop(self) -> None:
        if not self._emergency_stopped:
            return
        self._emergency_stopped = False
        self.reset_stop_button.setEnabled(False)
        self.status_label.setText("Control disabled.")
        self.log("Emergency stop cleared. Control remains disabled.")
        self._update_permission_summary()

    def _control_allowed(self, action_label: str) -> bool:
        if pyautogui is None:
            self._note_automation_unavailable()
            return False
        if self._emergency_stopped:
            self.log(f"{action_label} blocked: emergency stop is active.")
            return False
        if not self.enable_control_toggle.isChecked():
            self.log(f"{action_label} blocked: control is disabled.")
            return False
        return True

    def _note_automation_unavailable(self) -> None:
        self.status_label.setText(self.AUTOMATION_UNAVAILABLE_MESSAGE)
        if not self._automation_message_logged:
            self.log(self.AUTOMATION_UNAVAILABLE_MESSAGE)
            self._automation_message_logged = True

    def _update_automation_availability(self, log_message: bool = False) -> None:
        unavailable = pyautogui is None
        self.automation_banner.setVisible(unavailable)
        if unavailable:
            if log_message:
                self._note_automation_unavailable()
            self.take_screenshot_button.setEnabled(False)
            self.preview_toggle_button.setEnabled(False)
            self.manual_controls_group.setEnabled(False)
            self.enable_control_toggle.setToolTip(self.AUTOMATION_UNAVAILABLE_TOOLTIP)
            self.take_screenshot_button.setToolTip(self.AUTOMATION_UNAVAILABLE_TOOLTIP)
            self.preview_toggle_button.setToolTip(self.AUTOMATION_UNAVAILABLE_TOOLTIP)
            self.manual_controls_group.setToolTip(self.AUTOMATION_UNAVAILABLE_TOOLTIP)
            self.screenshot_label.setText("Preview unavailable")
        else:
            self.automation_banner.setVisible(False)
            self.take_screenshot_button.setEnabled(True)
            self.manual_controls_group.setEnabled(True)
            self.preview_toggle_button.setEnabled(self._agent_mode_active)
            self.enable_control_toggle.setToolTip("")
            self.take_screenshot_button.setToolTip("")
            self.preview_toggle_button.setToolTip("")
            self.manual_controls_group.setToolTip("")
        self._update_permission_summary()

    def _update_permission_summary(self) -> None:
        if pyautogui is None:
            allowed = "none (automation unavailable)"
        else:
            allowed = "screenshot, move mouse, click, type, press key"
        if self._emergency_stopped:
            status = "blocked (emergency stop)"
        elif not self.enable_control_toggle.isChecked():
            status = "disabled"
        else:
            status = "enabled"
        dry_run = "on" if self.dry_run_toggle.isChecked() else "off"
        self.permission_scope_label.setText(
            f"Allowed: {allowed}.\nStatus: {status}.\nDry run: {dry_run}."
        )

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

    def handle_move_mouse(self) -> None:
        x = self.mouse_x_spin.value()
        y = self.mouse_y_spin.value()
        self.manual_action_triggered.emit(
            {"action": "move_mouse", "parameters": {"x": x, "y": y}}
        )
        self.move_mouse(x, y)

    def handle_click_mouse(self) -> None:
        button_data = self.click_button_combo.currentData()
        button = button_data if isinstance(button_data, str) else "left"
        clicks = self.click_count_spin.value()
        self.manual_action_triggered.emit(
            {
                "action": "click_mouse",
                "parameters": {"button": button, "clicks": clicks, "interval": 0.0},
            }
        )
        self.click_mouse(button=button, clicks=clicks, interval=0.0)

    def handle_type_text(self) -> None:
        text = self.type_text_input.toPlainText()
        if text:
            self.manual_action_triggered.emit(
                {"action": "type_text", "parameters": {"text": text}}
            )
        self.type_text(text)

    def handle_press_key(self) -> None:
        key = self.key_input.text().strip()
        if not key:
            self.log("Key press skipped: no key provided.")
            return
        self.manual_action_triggered.emit(
            {"action": "press_key", "parameters": {"key": key}}
        )
        self.press_key(key)

    def move_mouse(self, x: int, y: int, duration: Optional[float] = None) -> bool:
        if not self._control_allowed("Move mouse"):
            return False
        move_duration = self.move_duration_spin.value() if duration is None else duration
        self.log(f"Moving mouse to ({x}, {y}) over {move_duration:.2f}s.")
        if self.dry_run_toggle.isChecked():
            self.log("Dry run: move mouse skipped.")
            return True
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
        if self.dry_run_toggle.isChecked():
            self.log("Dry run: mouse click skipped.")
            return True
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
        if self.dry_run_toggle.isChecked():
            self.log("Dry run: typing skipped.")
            return True
        self._apply_action_delay()
        pyautogui.write(text, interval=interval)
        return True

    def press_key(self, key: str) -> bool:
        if not self._control_allowed("Key press"):
            return False
        self.log(f"Pressing key: {key}.")
        if self.dry_run_toggle.isChecked():
            self.log("Dry run: key press skipped.")
            return True
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
        model = index.model()
        search_query = ""
        if hasattr(model, "search_query"):
            search_query = model.search_query()

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

        normalized_query = search_query.strip().lower()
        if normalized_query and normalized_query in message.content.lower():
            highlight_pen = QtGui.QPen(QtGui.QColor("#fbbf24"))
            highlight_pen.setWidth(2)
            painter.setPen(highlight_pen)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRoundedRect(bubble_rect.adjusted(-2, -2, 2, 2), 12, 12)

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


API_KEY_HINTS = {
    "openai": ["sk-"],
    "anthropic": ["sk-ant-"],
    "deepseek": ["sk-"],
    "gemini": ["AIza"],
    "groq": ["gsk_"],
    "mistral": ["sk-"],
    "perplexity": ["pplx-"],
}


class FirstRunWizard(QtWidgets.QWizard):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("KaliGPT first-run setup")
        self.setWizardStyle(QtWidgets.QWizard.ModernStyle)

        self.provider_selector = QtWidgets.QComboBox()
        self.api_key_inputs: dict[str, QtWidgets.QLineEdit] = {}
        self.api_key_status: dict[str, QtWidgets.QLabel] = {}
        self.desktop_control_toggle = QtWidgets.QCheckBox("Enable desktop control tools")

        self._build_pages()

    def _build_pages(self) -> None:
        provider_page = QtWidgets.QWizardPage()
        provider_page.setTitle("Choose your default provider")
        provider_page.setSubTitle("Pick the provider you want to use for chat responses.")
        provider_layout = QtWidgets.QVBoxLayout(provider_page)
        provider_layout.addWidget(QtWidgets.QLabel("Provider"))
        for provider, info in PROVIDER_REGISTRY.items():
            self.provider_selector.addItem(info["label"], provider)
        provider_layout.addWidget(self.provider_selector)
        provider_layout.addStretch()
        self.addPage(provider_page)

        api_keys_page = QtWidgets.QWizardPage()
        api_keys_page.setTitle("Add API keys")
        api_keys_page.setSubTitle(
            "Keys are stored locally in ~/.kaligpt/memory.json. Leave a field blank to use demo mode."
        )
        api_layout = QtWidgets.QFormLayout(api_keys_page)
        for provider, info in PROVIDER_REGISTRY.items():
            entry_row = QtWidgets.QHBoxLayout()
            key_input = QtWidgets.QLineEdit()
            key_input.setEchoMode(QtWidgets.QLineEdit.Password)
            key_input.setPlaceholderText(f"{info['label']} API key")
            help_label = QtWidgets.QLabel(
                f"Env: {info['env_key']}"
                + (
                    f" • Common prefix: {', '.join(API_KEY_HINTS[provider])}"
                    if provider in API_KEY_HINTS
                    else ""
                )
            )
            help_label.setStyleSheet("color: #9aa0a6; font-size: 11px;")
            status_label = QtWidgets.QLabel("Missing (demo mode)")
            status_label.setStyleSheet("color: #f0b429; font-size: 11px;")
            entry_row.addWidget(key_input, stretch=2)
            entry_row.addWidget(help_label, stretch=3)
            entry_row.addWidget(status_label, stretch=1)
            api_layout.addRow(info["label"], entry_row)
            self.api_key_inputs[provider] = key_input
            self.api_key_status[provider] = status_label
            key_input.textChanged.connect(
                lambda text, provider=provider: self._update_api_key_status(provider, text)
            )
        self.addPage(api_keys_page)

        control_page = QtWidgets.QWizardPage()
        control_page.setTitle("Desktop control")
        control_page.setSubTitle(
            "Enable desktop control to allow automation tools when in agent mode."
        )
        control_layout = QtWidgets.QVBoxLayout(control_page)
        control_layout.addWidget(self.desktop_control_toggle)
        control_layout.addStretch()
        self.addPage(control_page)

    def selected_provider(self) -> str:
        data = self.provider_selector.currentData()
        if isinstance(data, str):
            return data
        return list(PROVIDER_REGISTRY.keys())[0]

    def api_keys(self) -> dict[str, str]:
        keys: dict[str, str] = {}
        for provider, input_widget in self.api_key_inputs.items():
            value = input_widget.text().strip()
            if value:
                keys[provider] = value
        return keys

    def desktop_control_enabled(self) -> bool:
        return self.desktop_control_toggle.isChecked()

    def _update_api_key_status(self, provider: str, text: str) -> None:
        status_label = self.api_key_status[provider]
        value = text.strip()
        if not value:
            status_label.setText("Missing (demo mode)")
            status_label.setStyleSheet("color: #f0b429; font-size: 11px;")
            return
        hints = API_KEY_HINTS.get(provider, [])
        if hints and not any(value.startswith(prefix) for prefix in hints):
            status_label.setText(f"Check format (expected {', '.join(hints)})")
            status_label.setStyleSheet("color: #f0b429; font-size: 11px;")
            return
        status_label.setText("Looks good")
        status_label.setStyleSheet("color: #6ee7b7; font-size: 11px;")

class ChatWindow(QtWidgets.QWidget):
    DEFAULT_PREFERENCES = {
        "persona": "General assistant",
        "tone": "Neutral",
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    DEFAULT_TEMPLATES = [
        {
            "name": "Daily routine",
            "tasks": [
                "Review overnight alerts",
                "Check inbox and calendar",
                "Plan top 3 priorities",
                "Share standup update",
                "Wrap up with end-of-day notes",
            ],
            "built_in": True,
        },
        {
            "name": "Triage checklist",
            "tasks": [
                "Acknowledge the alert",
                "Collect relevant logs",
                "Assess scope and severity",
                "Contain immediate risks",
                "Notify stakeholders",
                "Document findings and next steps",
            ],
            "built_in": True,
        },
    ]

    def __init__(self, mode: str) -> None:
        super().__init__()
        self.mode = mode
        self.setWindowTitle("KaliGPT Workstation")
        self.resize(1200, 720)

        self._memory = self._load_memory()
        self._audit_log = self._load_audit_log()
        self._first_run = self._needs_first_run_wizard()
        self._suppress_api_prompt = self._first_run
        self._load_api_keys()
        self._provider_health = {
            provider: ProviderHealth() for provider in PROVIDER_REGISTRY
        }
        self._conversations: list[dict[str, str]] = []
        self._active_conversation_id: str | None = None
        self._templates: list[dict[str, object]] = []
        self._routines: list[Routine] = []
        self._recording_actions: list[RoutineAction] = []
        self._recording_active = False

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

        self.menu_bar = QtWidgets.QMenuBar()
        self.menu_bar.setNativeMenuBar(False)
        file_menu = self.menu_bar.addMenu("File")
        self.new_conversation_action = QtGui.QAction("New Conversation", self)
        self.import_json_action = QtGui.QAction("Import JSON...", self)
        self.import_markdown_action = QtGui.QAction("Import Markdown...", self)
        self.export_json_action = QtGui.QAction("Export JSON...", self)
        self.export_markdown_action = QtGui.QAction("Export Markdown...", self)
        file_menu.addAction(self.new_conversation_action)
        file_menu.addSeparator()
        file_menu.addAction(self.import_json_action)
        file_menu.addAction(self.import_markdown_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_json_action)
        file_menu.addAction(self.export_markdown_action)
        self.new_conversation_action.triggered.connect(self._prompt_new_conversation)
        self.import_json_action.triggered.connect(self._import_conversation_json)
        self.import_markdown_action.triggered.connect(self._import_conversation_markdown)
        self.export_json_action.triggered.connect(self._export_conversation_json)
        self.export_markdown_action.triggered.connect(self._export_conversation_markdown)

        self.conversation_label = QtWidgets.QLabel("Conversation")
        self.conversation_selector = QtWidgets.QComboBox()
        self.conversation_selector.currentIndexChanged.connect(self._handle_conversation_change)
        self.new_conversation_button = QtWidgets.QPushButton("New")
        self.new_conversation_button.clicked.connect(self._prompt_new_conversation)

        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Search conversation...")
        self.search_input.textChanged.connect(self._handle_search_change)

        self.provider_label = QtWidgets.QLabel("Provider")
        self.provider_selector = QtWidgets.QComboBox()
        self.provider_selector.currentIndexChanged.connect(self._handle_provider_change)
        self.provider_status_widget = QtWidgets.QWidget()
        self.provider_status_layout = QtWidgets.QHBoxLayout(self.provider_status_widget)
        self.provider_status_layout.setContentsMargins(0, 0, 0, 0)
        self.provider_status_layout.setSpacing(6)
        self._provider_status_labels: dict[str, QtWidgets.QLabel] = {}

        self.model_label = QtWidgets.QLabel("Model")
        self.model_selector = QtWidgets.QComboBox()
        self.model_selector.currentIndexChanged.connect(self._handle_model_change)
        self._populate_provider_selector()

        self.behavior_group = QtWidgets.QGroupBox("Behavior")
        self.persona_input = QtWidgets.QLineEdit()
        self.persona_input.setPlaceholderText("e.g., Security analyst or Red team coach")
        self.persona_input.textChanged.connect(self._persist_behavior_preferences)

        self.tone_selector = QtWidgets.QComboBox()
        self.tone_selector.addItems(
            [
                "Neutral",
                "Friendly",
                "Direct",
                "Formal",
                "Concise",
                "Analytical",
                "Supportive",
            ]
        )
        self.tone_selector.currentIndexChanged.connect(self._persist_behavior_preferences)

        self.temperature_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.temperature_slider.setRange(0, 100)
        self.temperature_slider.setSingleStep(1)
        self.temperature_slider.valueChanged.connect(self._handle_temperature_change)
        self.temperature_value_label = QtWidgets.QLabel()

        self.max_tokens_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.max_tokens_slider.setRange(256, 4096)
        self.max_tokens_slider.setSingleStep(64)
        self.max_tokens_slider.valueChanged.connect(self._handle_max_tokens_change)
        self.max_tokens_value_label = QtWidgets.QLabel()

        behavior_form = QtWidgets.QFormLayout(self.behavior_group)
        behavior_form.addRow("Persona", self.persona_input)
        behavior_form.addRow("Tone", self.tone_selector)
        behavior_form.addRow("Temperature", self._build_slider_row(self.temperature_slider, self.temperature_value_label))
        behavior_form.addRow("Max tokens", self._build_slider_row(self.max_tokens_slider, self.max_tokens_value_label))

        self.send_button = QtWidgets.QPushButton("Send")
        self.send_button.clicked.connect(self.handle_send)

        self.controls_panel = ComputerControlPanel()
        self.controls_panel.enable_control_toggle.toggled.connect(
            self._persist_desktop_control_preference
        )
        self.controls_panel.batch_review_toggle.toggled.connect(
            self._persist_action_review_preference
        )
        self.controls_panel.dry_run_toggle.toggled.connect(
            self._persist_dry_run_preference
        )
        self.controls_panel.screenshot_context_toggle.toggled.connect(
            self._persist_screenshot_context_preference
        )
        self.controls_panel.record_routine_button.toggled.connect(
            self._handle_record_routine_toggle
        )
        self.controls_panel.save_routine_button.clicked.connect(
            self._save_recorded_routine
        )
        self.controls_panel.run_routine_button.clicked.connect(
            self._run_selected_routine
        )
        self.controls_panel.manual_action_triggered.connect(
            self._handle_manual_action
        )
        self.controls_panel.export_audit_requested.connect(self._export_audit_log)
        self.monitoring_panel = QtWidgets.QGroupBox("Monitoring")
        self.monitoring_window_label = QtWidgets.QLabel("Active window: Unavailable")
        self.monitoring_process_label = QtWidgets.QLabel("Foreground process: Unavailable")
        self.monitoring_cpu_label = QtWidgets.QLabel("CPU load: Unavailable")
        self.monitoring_memory_label = QtWidgets.QLabel("Memory usage: Unavailable")
        self._monitor_last_window: Optional[str] = None
        self._monitor_last_process: Optional[str] = None
        self._monitor_last_cpu: Optional[float] = None
        self._monitor_last_memory: Optional[float] = None
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
        self.apply_template_button = QtWidgets.QPushButton("Apply Template")
        self.apply_template_button.clicked.connect(self._apply_template)
        self.save_template_button = QtWidgets.QPushButton("Save as Template")
        self.save_template_button.clicked.connect(self._save_template)

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
        self.api_status.setStyleSheet("color: #9aa0a6; font-size: 12px;")

        header_title = QtWidgets.QLabel("KaliGPT")
        header_title.setObjectName("HeaderTitle")
        self.demo_mode_banner = QtWidgets.QLabel("Demo mode: API key missing")
        self.demo_mode_banner.setObjectName("DemoModeBanner")
        header_subtitle = QtWidgets.QLabel("Your AI assistant for secure workflows")
        header_subtitle.setObjectName("HeaderSubtitle")
        self.preferences_label = QtWidgets.QLabel()
        self.preferences_label.setObjectName("HeaderPreferences")
        self._update_preferences_label()

        header_layout = QtWidgets.QVBoxLayout()
        header_layout.addWidget(header_title)
        header_layout.addWidget(self.demo_mode_banner)
        header_layout.addWidget(header_subtitle)
        header_layout.addWidget(self.preferences_label)
        header_layout.setSpacing(2)

        header_widget = QtWidgets.QWidget()
        header_widget.setLayout(header_layout)

        conversation_layout = QtWidgets.QHBoxLayout()
        conversation_layout.addWidget(self.conversation_label)
        conversation_layout.addWidget(self.conversation_selector, stretch=1)
        conversation_layout.addWidget(self.new_conversation_button)

        search_layout = QtWidgets.QHBoxLayout()
        search_layout.addWidget(QtWidgets.QLabel("Search"))
        search_layout.addWidget(self.search_input, stretch=1)

        chat_layout = QtWidgets.QVBoxLayout()
        chat_layout.addWidget(self.menu_bar)
        chat_layout.addWidget(header_widget)
        chat_layout.addLayout(conversation_layout)
        chat_layout.addLayout(search_layout)
        chat_layout.addWidget(self.chat_view)

        input_layout = QtWidgets.QHBoxLayout()
        input_layout.addWidget(self.message_input, stretch=1)
        input_layout.addWidget(self.send_button)

        model_layout = QtWidgets.QHBoxLayout()
        model_layout.addWidget(self.provider_label)
        model_layout.addWidget(self.provider_selector)
        model_layout.addWidget(self.provider_status_widget)
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_selector)
        model_layout.addStretch()

        chat_layout.addLayout(input_layout)
        chat_layout.addLayout(model_layout)
        chat_layout.addWidget(self.behavior_group)
        chat_layout.addWidget(self.api_status)

        task_button_layout = QtWidgets.QGridLayout()
        task_button_layout.addWidget(self.add_task_button, 0, 0)
        task_button_layout.addWidget(self.edit_task_button, 0, 1)
        task_button_layout.addWidget(self.remove_task_button, 0, 2)
        task_button_layout.addWidget(self.move_up_button, 1, 0)
        task_button_layout.addWidget(self.move_down_button, 1, 1)
        task_button_layout.addWidget(self.complete_task_button, 1, 2)

        template_button_layout = QtWidgets.QHBoxLayout()
        template_button_layout.addWidget(self.apply_template_button)
        template_button_layout.addWidget(self.save_template_button)

        task_layout = QtWidgets.QVBoxLayout(self.task_panel)
        task_layout.addWidget(self.task_view)
        task_layout.addLayout(task_button_layout)
        task_layout.addLayout(template_button_layout)
        task_layout.addWidget(self.agent_toggle_button)

        monitoring_layout = QtWidgets.QVBoxLayout(self.monitoring_panel)
        monitoring_layout.addWidget(self.monitoring_window_label)
        monitoring_layout.addWidget(self.monitoring_process_label)
        monitoring_layout.addWidget(self.monitoring_cpu_label)
        monitoring_layout.addWidget(self.monitoring_memory_label)

        right_layout = QtWidgets.QVBoxLayout()
        right_layout.addWidget(self.task_panel)
        right_layout.addWidget(self.controls_panel)
        right_layout.addWidget(self.monitoring_panel)
        right_layout.addStretch()

        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right_layout)

        main_layout = QtWidgets.QHBoxLayout(self)
        main_layout.addLayout(chat_layout, stretch=3)
        main_layout.addWidget(right_widget, stretch=2)

        self._apply_theme()
        self._configure_mode()
        self._build_provider_status_widgets()
        self._load_desktop_control_preference()
        self._load_action_review_preference()
        self._load_dry_run_preference()
        self._load_screenshot_context_preference()
        self._load_behavior_preferences()
        self._load_conversations()
        self._load_templates()
        self._load_tasks()
        self._load_routines()
        self._refresh_task_snapshot()
        self._refresh_task_controls()
        self.task_view.selectionModel().selectionChanged.connect(
            lambda *_: self._refresh_task_controls()
        )
        self._monitor_timer = QtCore.QTimer(self)
        self._monitor_timer.setInterval(2500)
        self._monitor_timer.timeout.connect(self._refresh_monitoring)
        self._monitor_timer.start()
        self._refresh_monitoring()
        self._update_api_status()
        self._maybe_show_first_run_wizard()

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
            QLabel#DemoModeBanner {
                background: #2f1f1f;
                color: #f87171;
                border-radius: 6px;
                padding: 4px 8px;
                font-weight: 600;
            }
            QLabel#ProviderStatusLabel {
                font-size: 11px;
                color: #9aa0a6;
                border: 1px solid #3e3f4b;
                border-radius: 10px;
                padding: 2px 6px;
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

    def _refresh_monitoring(self) -> None:
        window_title, process_name = self._active_window_info()
        cpu_load = self._cpu_load()
        memory_usage = self._memory_usage()

        self.monitoring_window_label.setText(f"Active window: {window_title}")
        self.monitoring_process_label.setText(f"Foreground process: {process_name}")
        self.monitoring_cpu_label.setText(
            "CPU load: Unavailable" if cpu_load is None else f"CPU load: {cpu_load:.1f}%"
        )
        self.monitoring_memory_label.setText(
            "Memory usage: Unavailable"
            if memory_usage is None
            else f"Memory usage: {memory_usage:.1f}%"
        )

        if window_title != self._monitor_last_window:
            if self._monitor_last_window is not None:
                self._log_monitor_event(f"Active window changed to '{window_title}'.")
            self._monitor_last_window = window_title

        if process_name != self._monitor_last_process:
            if self._monitor_last_process is not None:
                self._log_monitor_event(f"Foreground process changed to '{process_name}'.")
            self._monitor_last_process = process_name

        self._log_usage_change("CPU load", cpu_load, 10.0, "cpu")
        self._log_usage_change("Memory usage", memory_usage, 5.0, "memory")

    def _log_usage_change(self, label: str, value: Optional[float], threshold: float, kind: str) -> None:
        if value is None:
            return
        if kind == "cpu":
            previous = self._monitor_last_cpu
            self._monitor_last_cpu = value
        else:
            previous = self._monitor_last_memory
            self._monitor_last_memory = value

        if previous is None:
            return
        if abs(value - previous) >= threshold:
            self._log_monitor_event(f"{label} shifted to {value:.1f}%.")

    def _log_monitor_event(self, message: str) -> None:
        self.controls_panel.log(f"Monitoring: {message}")
        self._append_audit_entry(
            {
                "event": "monitoring",
                "message": message,
            }
        )

    def _active_window_info(self) -> tuple[str, str]:
        if not sys.platform.startswith("linux"):
            return ("Unavailable", "Unavailable")
        if not shutil.which("xdotool"):
            return ("Unavailable", "Unavailable")

        window_id = self._run_command(["xdotool", "getactivewindow"]).strip()
        if not window_id:
            return ("Unavailable", "Unavailable")

        title = self._run_command(["xdotool", "getactivewindow", "getwindowname"]).strip()
        pid_raw = self._run_command(["xdotool", "getactivewindow", "getwindowpid"]).strip()
        process = self._process_name_from_pid(pid_raw)

        return (title or "Unknown", process or "Unknown")

    def _process_name_from_pid(self, pid_raw: str) -> str:
        if not pid_raw.isdigit():
            return "Unknown"
        pid = pid_raw.strip()
        comm_path = Path("/proc") / pid / "comm"
        if comm_path.exists():
            return comm_path.read_text(encoding="utf-8").strip()
        output = self._run_command(["ps", "-p", pid, "-o", "comm="]).strip()
        return output or "Unknown"

    def _cpu_load(self) -> Optional[float]:
        try:
            load_1, _, _ = os.getloadavg()
        except (AttributeError, OSError):
            return None
        cpu_count = os.cpu_count() or 1
        return min(100.0, (load_1 / cpu_count) * 100.0)

    def _memory_usage(self) -> Optional[float]:
        meminfo = Path("/proc/meminfo")
        if not meminfo.exists():
            return None
        total = None
        available = None
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                total = self._parse_kib(line)
            elif line.startswith("MemAvailable:"):
                available = self._parse_kib(line)
        if total is None or available is None or total == 0:
            return None
        used = total - available
        return (used / total) * 100.0

    def _parse_kib(self, line: str) -> Optional[int]:
        match = re.search(r"(\d+)", line)
        if not match:
            return None
        return int(match.group(1))

    def _run_command(self, command: list[str]) -> str:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout or ""

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

    def _build_provider_status_widgets(self) -> None:
        while self.provider_status_layout.count():
            item = self.provider_status_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._provider_status_labels.clear()
        for provider, info in PROVIDER_REGISTRY.items():
            label = QtWidgets.QLabel()
            label.setObjectName("ProviderStatusLabel")
            label.setTextFormat(QtCore.Qt.RichText)
            label.setToolTip(f"{info['label']} provider status.")
            self._provider_status_labels[provider] = label
            self.provider_status_layout.addWidget(label)
        self._update_provider_status_labels()

    def _format_provider_status(self, provider: str) -> tuple[str, str]:
        info = PROVIDER_REGISTRY[provider]
        health = self._provider_health.get(provider, ProviderHealth())
        label = info["label"]
        if health.state == ProviderHealthState.OK:
            return ("#34d399", f"{label}: Ready")
        if health.state == ProviderHealthState.RETRYING:
            retry = health.retry_after_seconds or 0
            return ("#fbbf24", f"{label}: Retrying in {retry:.1f}s…")
        if health.state == ProviderHealthState.ERROR:
            time_str = (
                health.last_error_time.strftime("%H:%M:%S")
                if health.last_error_time
                else "Unknown time"
            )
            return ("#f87171", f"{label}: Error at {time_str}")
        return ("#9aa0a6", f"{label}: Idle")

    def _update_provider_status_labels(self) -> None:
        for provider, label in self._provider_status_labels.items():
            color, text = self._format_provider_status(provider)
            label.setText(f"<span style='color:{color}'>●</span> {text}")
            health = self._provider_health.get(provider)
            if health and health.last_error_message:
                label.setToolTip(health.last_error_message)
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
        self._update_api_status()

    def _handle_model_change(self, *_: object) -> None:
        self._persist_model_selection()
        self._update_api_status()

    def _load_conversations(self) -> None:
        self._load_conversation_index()
        self._populate_conversation_selector()
        if self._active_conversation_id:
            self._load_conversation_messages(self._active_conversation_id)

    def _load_conversation_index(self) -> None:
        index_path = self._conversation_index_path()
        self._conversations = []
        self._active_conversation_id = None
        if index_path.exists():
            try:
                with index_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                conversations = payload.get("conversations", [])
                if isinstance(conversations, list):
                    self._conversations = [
                        item for item in conversations if isinstance(item, dict) and item.get("id")
                    ]
                active_id = payload.get("active_id")
                if isinstance(active_id, str):
                    self._active_conversation_id = active_id
            except (OSError, ValueError, TypeError, AttributeError):
                self._conversations = []
                self._active_conversation_id = None

        if not self._conversations:
            legacy_path = self._history_path()
            if legacy_path.exists():
                messages = self._read_messages_from_json(legacy_path)
                conversation_id = self._create_conversation(
                    title="Migrated Conversation",
                    messages=messages,
                    switch_to=True,
                )
                self._active_conversation_id = conversation_id
            else:
                conversation_id = self._create_conversation(
                    title="Conversation 1",
                    messages=[],
                    switch_to=True,
                )
                self._active_conversation_id = conversation_id
        elif self._active_conversation_id is None:
            self._active_conversation_id = self._conversations[0]["id"]
        else:
            known_ids = {meta.get("id") for meta in self._conversations}
            if self._active_conversation_id not in known_ids:
                self._active_conversation_id = self._conversations[0]["id"]

        self._save_conversation_index()

    def _save_conversation_index(self) -> None:
        index_path = self._conversation_index_path()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "active_id": self._active_conversation_id,
            "conversations": self._conversations,
        }
        with index_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _populate_conversation_selector(self) -> None:
        self.conversation_selector.blockSignals(True)
        self.conversation_selector.clear()
        active_index = 0
        for index, meta in enumerate(self._conversations):
            title = meta.get("title") or "Conversation"
            conversation_id = meta.get("id", "")
            self.conversation_selector.addItem(title, conversation_id)
            if conversation_id == self._active_conversation_id:
                active_index = index
        if self.conversation_selector.count():
            self.conversation_selector.setCurrentIndex(active_index)
        self.conversation_selector.blockSignals(False)

    def _handle_conversation_change(self, index: int) -> None:
        if index < 0:
            return
        conversation_id = self.conversation_selector.currentData()
        if not isinstance(conversation_id, str) or not conversation_id:
            return
        if conversation_id == self._active_conversation_id:
            return
        self._save_conversation_messages()
        self._active_conversation_id = conversation_id
        self._save_conversation_index()
        self._load_conversation_messages(conversation_id)

    def _prompt_new_conversation(self) -> None:
        title, ok = QtWidgets.QInputDialog.getText(
            self, "New Conversation", "Conversation title"
        )
        if not ok:
            return
        title = title.strip() or f"Conversation {len(self._conversations) + 1}"
        self._create_conversation(title=title, messages=[], switch_to=True)

    def _create_conversation(
        self,
        title: str,
        messages: list[ChatMessage],
        switch_to: bool = False,
    ) -> str:
        conversation_id = self._new_conversation_id()
        timestamp = datetime.now().isoformat()
        meta = {
            "id": conversation_id,
            "title": title,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        self._conversations.append(meta)
        self._save_conversation_messages(conversation_id, messages)
        if switch_to:
            self._active_conversation_id = conversation_id
            self._populate_conversation_selector()
            self._load_conversation_messages(conversation_id)
        self._save_conversation_index()
        return conversation_id

    def _load_conversation_messages(self, conversation_id: str) -> None:
        path = self._conversation_path(conversation_id)
        messages = []
        if path.exists():
            messages = self._read_messages_from_json(path)
        self.chat_model.set_messages(messages)
        self.chat_view.scrollToBottom()

    def _save_conversation_messages(
        self,
        conversation_id: str | None = None,
        messages: Optional[list[ChatMessage]] = None,
    ) -> None:
        conversation_id = conversation_id or self._active_conversation_id
        if not conversation_id:
            return
        payload = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in (messages or self.chat_model._messages)
        ]
        path = self._conversation_path(conversation_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self._touch_conversation(conversation_id)

    def _touch_conversation(self, conversation_id: str) -> None:
        for meta in self._conversations:
            if meta.get("id") == conversation_id:
                meta["updated_at"] = datetime.now().isoformat()
                break
        self._save_conversation_index()

    def _new_conversation_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d%H%M%S")
        suffix = random.randint(1000, 9999)
        return f"conv-{stamp}-{suffix}"

    def _read_messages_from_json(self, path: Path) -> list[ChatMessage]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                return []
            messages = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", "assistant"))
                content = str(item.get("content", ""))
                timestamp_raw = item.get("timestamp")
                try:
                    timestamp = (
                        datetime.fromisoformat(timestamp_raw)
                        if isinstance(timestamp_raw, str)
                        else datetime.now()
                    )
                except ValueError:
                    timestamp = datetime.now()
                messages.append(ChatMessage(role=role, content=content, timestamp=timestamp))
            return messages
        except (OSError, ValueError, TypeError):
            return []

    def _conversation_index_path(self) -> Path:
        return self._conversations_dir() / "index.json"

    def _conversation_path(self, conversation_id: str) -> Path:
        return self._conversations_dir() / f"{conversation_id}.json"

    def _conversations_dir(self) -> Path:
        return Path.home() / ".kaligpt" / "conversations"

    def _history_path(self) -> Path:
        return Path.home() / ".kaligpt" / "history.json"

    def _handle_search_change(self, text: str) -> None:
        self.chat_model.set_search_query(text)
        if text.strip():
            self.chat_view.viewport().update()

    def _export_conversation_json(self) -> None:
        conversation = self._active_conversation_meta()
        if not conversation:
            return
        start_dir = str(Path.home())
        suggested = f"{conversation.get('title', 'conversation')}.json"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Conversation (JSON)",
            str(Path(start_dir) / suggested),
            "JSON Files (*.json)",
        )
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != ".json":
            destination = destination.with_suffix(".json")
        payload = [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
            }
            for msg in self.chat_model._messages
        ]
        try:
            with destination.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Export Conversation",
                f"Failed to export conversation: {exc}",
            )

    def _export_conversation_markdown(self) -> None:
        conversation = self._active_conversation_meta()
        if not conversation:
            return
        start_dir = str(Path.home())
        suggested = f"{conversation.get('title', 'conversation')}.md"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Conversation (Markdown)",
            str(Path(start_dir) / suggested),
            "Markdown Files (*.md)",
        )
        if not path:
            return
        destination = Path(path)
        if destination.suffix.lower() != ".md":
            destination = destination.with_suffix(".md")
        try:
            destination.write_text(self._format_markdown_conversation(), encoding="utf-8")
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Export Conversation",
                f"Failed to export conversation: {exc}",
            )

    def _import_conversation_json(self) -> None:
        start_dir = str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Conversation (JSON)",
            start_dir,
            "JSON Files (*.json)",
        )
        if not path:
            return
        source = Path(path)
        messages = self._read_messages_from_json(source)
        if not messages:
            QtWidgets.QMessageBox.warning(
                self,
                "Import Conversation",
                "No valid messages were found in the selected file.",
            )
            return
        title = source.stem.replace("_", " ").title()
        self._create_conversation(title=title, messages=messages, switch_to=True)

    def _import_conversation_markdown(self) -> None:
        start_dir = str(Path.home())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Conversation (Markdown)",
            start_dir,
            "Markdown Files (*.md)",
        )
        if not path:
            return
        source = Path(path)
        try:
            markdown_text = source.read_text(encoding="utf-8")
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Import Conversation",
                f"Failed to read file: {exc}",
            )
            return
        messages = self._parse_markdown_conversation(markdown_text)
        if not messages:
            QtWidgets.QMessageBox.warning(
                self,
                "Import Conversation",
                "No messages could be parsed from the Markdown file.",
            )
            return
        title = source.stem.replace("_", " ").title()
        self._create_conversation(title=title, messages=messages, switch_to=True)

    def _format_markdown_conversation(self) -> str:
        conversation = self._active_conversation_meta() or {}
        title = conversation.get("title", "Conversation")
        lines = [f"# {title}", ""]
        for message in self.chat_model._messages:
            timestamp = message.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"### {message.role.title()} · {timestamp}")
            lines.append("")
            lines.append(message.content)
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _parse_markdown_conversation(self, markdown_text: str) -> list[ChatMessage]:
        messages: list[ChatMessage] = []
        current_role = None
        current_timestamp = datetime.now()
        buffer: list[str] = []
        header_pattern = re.compile(r"^###\s+(.+?)(?:\s+·\s+(.+))?$")

        def flush() -> None:
            nonlocal buffer, current_role, current_timestamp
            if current_role and buffer:
                content = "\n".join(line.rstrip() for line in buffer).strip()
                if content:
                    messages.append(
                        ChatMessage(
                            role=current_role,
                            content=content,
                            timestamp=current_timestamp,
                        )
                    )
            buffer = []

        for line in markdown_text.splitlines():
            match = header_pattern.match(line.strip())
            if match:
                flush()
                role = match.group(1).strip().lower()
                if role not in {"user", "assistant", "system"}:
                    role = "assistant"
                current_role = role
                timestamp_text = match.group(2)
                if timestamp_text:
                    try:
                        current_timestamp = datetime.fromisoformat(timestamp_text.strip())
                    except ValueError:
                        try:
                            current_timestamp = datetime.strptime(
                                timestamp_text.strip(), "%Y-%m-%d %H:%M:%S"
                            )
                        except ValueError:
                            current_timestamp = datetime.now()
                else:
                    current_timestamp = datetime.now()
                continue
            buffer.append(line)
        flush()
        return messages

    def _active_conversation_meta(self) -> Optional[dict[str, str]]:
        for meta in self._conversations:
            if meta.get("id") == self._active_conversation_id:
                return meta
        return None

    def _memory_path(self) -> Path:
        return Path.home() / ".kaligpt" / "memory.json"

    def _tasks_path(self) -> Path:
        return Path.home() / ".kaligpt" / "tasks.json"

    def _audit_log_path(self) -> Path:
        return Path.home() / ".kaligpt" / "audit.json"

    def _templates_path(self) -> Path:
        return Path.home() / ".kaligpt" / "templates.json"

    def _load_memory(self) -> dict[str, object]:
        memory_path = self._memory_path()
        if not memory_path.exists():
            return {
                "preferences": {},
                "task_outcomes": [],
                "failures": [],
                "api_keys": {},
                "routines": [],
            }
        try:
            with memory_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return {
                "preferences": dict(payload.get("preferences", {})),
                "task_outcomes": list(payload.get("task_outcomes", [])),
                "failures": list(payload.get("failures", [])),
                "api_keys": dict(payload.get("api_keys", {})),
                "routines": list(payload.get("routines", [])),
            }
        except (OSError, ValueError, TypeError):
            return {
                "preferences": {},
                "task_outcomes": [],
                "failures": [],
                "api_keys": {},
                "routines": [],
            }

    def _save_memory(self) -> None:
        memory_path = self._memory_path()
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        with memory_path.open("w", encoding="utf-8") as handle:
            json.dump(self._memory, handle, indent=2)

    def _normalize_template(self, payload: object) -> Optional[dict[str, object]]:
        if not isinstance(payload, dict):
            return None
        name = str(payload.get("name", "")).strip()
        tasks_raw = payload.get("tasks", [])
        if not name or not isinstance(tasks_raw, list):
            return None
        tasks = [str(item).strip() for item in tasks_raw if str(item).strip()]
        if not tasks:
            return None
        template: dict[str, object] = {"name": name, "tasks": tasks}
        if payload.get("built_in"):
            template["built_in"] = True
        return template

    def _load_templates(self) -> None:
        templates: list[dict[str, object]] = []
        templates_path = self._templates_path()
        if templates_path.exists():
            try:
                with templates_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, list):
                    raw_templates = payload
                elif isinstance(payload, dict):
                    raw_templates = payload.get("templates", [])
                else:
                    raw_templates = []
                if isinstance(raw_templates, list):
                    for item in raw_templates:
                        normalized = self._normalize_template(item)
                        if normalized:
                            templates.append(normalized)
            except (OSError, ValueError, TypeError):
                templates = []

        existing_names = {template["name"] for template in templates}
        for default_template in self.DEFAULT_TEMPLATES:
            if default_template["name"] not in existing_names:
                templates.append(dict(default_template))
        self._templates = templates
        self._save_templates()

    def _save_templates(self) -> None:
        templates_path = self._templates_path()
        templates_path.parent.mkdir(parents=True, exist_ok=True)
        with templates_path.open("w", encoding="utf-8") as handle:
            json.dump({"templates": self._templates}, handle, indent=2)

    def _export_audit_log(self) -> None:
        if not self._audit_log:
            QtWidgets.QMessageBox.information(
                self,
                "Export Audit Log",
                "No audit entries are available to export yet.",
            )
            return
        start_dir = str(Path.home())
        path, selected_filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Audit Log",
            start_dir,
            "JSON Files (*.json);;CSV Files (*.csv)",
        )
        if not path:
            return
        destination = Path(path)
        export_csv = destination.suffix.lower() == ".csv" or "CSV" in selected_filter
        if not destination.suffix:
            destination = destination.with_suffix(".csv" if export_csv else ".json")
        try:
            if export_csv:
                self._write_audit_csv(destination)
            else:
                with destination.open("w", encoding="utf-8") as handle:
                    json.dump(self._audit_log, handle, indent=2)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Export Audit Log",
                f"Failed to export audit log: {exc}",
            )

    def _write_audit_csv(self, destination: Path) -> None:
        fieldnames: list[str] = []
        for entry in self._audit_log:
            for key in entry.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for entry in self._audit_log:
                row: dict[str, object] = {}
                for key in fieldnames:
                    value = entry.get(key)
                    if isinstance(value, (dict, list)):
                        row[key] = json.dumps(value)
                    else:
                        row[key] = value
                writer.writerow(row)

    def _load_audit_log(self) -> list[dict[str, object]]:
        audit_path = self._audit_log_path()
        if not audit_path.exists():
            return []
        try:
            with audit_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]
        except (OSError, ValueError, TypeError):
            return []
        return []

    def _save_audit_log(self) -> None:
        audit_path = self._audit_log_path()
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        with audit_path.open("w", encoding="utf-8") as handle:
            json.dump(self._audit_log, handle, indent=2)

    def _append_audit_entry(self, entry: dict[str, object]) -> None:
        entry.setdefault("timestamp", datetime.now().isoformat())
        self._audit_log.append(entry)
        self._save_audit_log()

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
        if self._suppress_api_prompt:
            return
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
        self._update_api_status()

    def _update_api_status(self) -> None:
        status_text = self._api_status_text()
        provider = self._selected_provider()
        if provider:
            status_text = f"{status_text} {self._provider_status_message(provider)}"
        self.api_status.setText(status_text)
        self._update_demo_mode_banner(status_text)

    def _provider_status_message(self, provider: str) -> str:
        health = self._provider_health.get(provider)
        if health is None:
            return ""
        if health.state == ProviderHealthState.OK:
            return "Status: Ready."
        if health.state == ProviderHealthState.RETRYING:
            retry = health.retry_after_seconds or 0
            return f"Status: Retrying in {retry:.1f}s…"
        if health.state == ProviderHealthState.ERROR:
            time_str = (
                health.last_error_time.strftime("%H:%M:%S")
                if health.last_error_time
                else "Unknown time"
            )
            return f"Status: Error at {time_str}."
        return "Status: Idle."

    def _set_provider_health(
        self,
        provider: str,
        state: ProviderHealthState,
        *,
        error_message: Optional[str] = None,
        retry_after_seconds: Optional[float] = None,
    ) -> None:
        health = self._provider_health.setdefault(provider, ProviderHealth())
        health.state = state
        health.retry_after_seconds = retry_after_seconds
        if error_message:
            health.last_error_message = error_message
        if state in {ProviderHealthState.ERROR, ProviderHealthState.RETRYING}:
            health.last_error_time = datetime.now()
        self._update_provider_status_labels()
        self._update_api_status()

    def _update_demo_mode_banner(self, status_text: str) -> None:
        text = status_text.lower()
        show_banner = "stubbed" in text or ("set " in text and " to enable" in text)
        self.demo_mode_banner.setVisible(show_banner)

    def _needs_first_run_wizard(self) -> bool:
        memory_path = self._memory_path()
        if not memory_path.exists():
            return True
        preferences = self._memory.get("preferences")
        return not isinstance(preferences, dict) or not preferences

    def _maybe_show_first_run_wizard(self) -> None:
        if not self._first_run:
            return
        wizard = FirstRunWizard(self)
        result = wizard.exec()
        if result == QtWidgets.QDialog.Accepted:
            self._apply_first_run_settings(wizard)
        self._suppress_api_prompt = False

    def _apply_first_run_settings(self, wizard: FirstRunWizard) -> None:
        preferences = self._memory.setdefault("preferences", {})
        if not isinstance(preferences, dict):
            preferences = {}
            self._memory["preferences"] = preferences
        provider = wizard.selected_provider()
        preferences["provider"] = provider
        model_map = preferences.get("model_map")
        if not isinstance(model_map, dict):
            model_map = {}
        model_map[provider] = PROVIDER_REGISTRY[provider]["models"][0]["id"]
        preferences["model_map"] = model_map
        preferences["desktop_control_enabled"] = wizard.desktop_control_enabled()

        api_keys = self._memory.setdefault("api_keys", {})
        if not isinstance(api_keys, dict):
            api_keys = {}
            self._memory["api_keys"] = api_keys
        for provider_name, key in wizard.api_keys().items():
            api_keys[provider_name] = key
            env_key = PROVIDER_REGISTRY[provider_name]["env_key"]
            if not os.getenv(env_key):
                os.environ[env_key] = key

        self._save_memory()
        self._populate_provider_selector()
        self._load_desktop_control_preference()
        self._update_api_status()

    def _load_desktop_control_preference(self) -> None:
        preferences = self._memory.get("preferences", {})
        if not isinstance(preferences, dict):
            return
        enabled = bool(preferences.get("desktop_control_enabled", False))
        self.controls_panel.enable_control_toggle.blockSignals(True)
        self.controls_panel.enable_control_toggle.setChecked(enabled)
        self.controls_panel.enable_control_toggle.blockSignals(False)

    def _persist_desktop_control_preference(self, enabled: bool) -> None:
        preferences = self._memory.setdefault("preferences", {})
        if not isinstance(preferences, dict):
            preferences = {}
            self._memory["preferences"] = preferences
        preferences["desktop_control_enabled"] = enabled
        self._save_memory()

    def _load_action_review_preference(self) -> None:
        preferences = self._memory.get("preferences", {})
        if not isinstance(preferences, dict):
            return
        enabled = bool(preferences.get("batch_action_review_enabled", True))
        self.controls_panel.batch_review_toggle.blockSignals(True)
        self.controls_panel.batch_review_toggle.setChecked(enabled)
        self.controls_panel.batch_review_toggle.blockSignals(False)

    def _persist_action_review_preference(self, enabled: bool) -> None:
        preferences = self._memory.setdefault("preferences", {})
        if not isinstance(preferences, dict):
            preferences = {}
            self._memory["preferences"] = preferences
        preferences["batch_action_review_enabled"] = enabled
        self._save_memory()

    def _load_dry_run_preference(self) -> None:
        preferences = self._memory.get("preferences", {})
        enabled = bool(preferences.get("dry_run_enabled", False))
        self.controls_panel.dry_run_toggle.blockSignals(True)
        self.controls_panel.dry_run_toggle.setChecked(enabled)
        self.controls_panel.dry_run_toggle.blockSignals(False)
        self.controls_panel._update_permission_summary()

    def _persist_dry_run_preference(self, enabled: bool) -> None:
        preferences = self._memory.setdefault("preferences", {})
        if not isinstance(preferences, dict):
            preferences = {}
            self._memory["preferences"] = preferences
        preferences["dry_run_enabled"] = enabled
        self._save_memory()

    def _load_screenshot_context_preference(self) -> None:
        preferences = self._memory.get("preferences", {})
        enabled = bool(preferences.get("screenshot_context_enabled", False))
        self.controls_panel.screenshot_context_toggle.blockSignals(True)
        self.controls_panel.screenshot_context_toggle.setChecked(enabled)
        self.controls_panel.screenshot_context_toggle.blockSignals(False)

    def _persist_screenshot_context_preference(self, enabled: bool) -> None:
        preferences = self._memory.setdefault("preferences", {})
        if not isinstance(preferences, dict):
            preferences = {}
            self._memory["preferences"] = preferences
        preferences["screenshot_context_enabled"] = enabled
        self._save_memory()

    def _load_routines(self) -> None:
        routines_raw = self._memory.get("routines", [])
        routines: list[Routine] = []
        if isinstance(routines_raw, list):
            for item in routines_raw:
                if isinstance(item, dict):
                    routine = Routine.from_payload(item)
                    if routine:
                        routines.append(routine)
        self._routines = routines
        self.controls_panel.set_routines(self._routines)

    def _save_routines(self) -> None:
        self._memory["routines"] = [routine.to_payload() for routine in self._routines]
        self._save_memory()
        self.controls_panel.set_routines(self._routines)

    def _handle_record_routine_toggle(self, active: bool) -> None:
        self._recording_active = active
        self.controls_panel.set_recording_state(active)
        if active:
            self._recording_actions = []
            self.controls_panel.log("Routine recording started.")
        else:
            self.controls_panel.log(
                f"Routine recording stopped. {len(self._recording_actions)} actions captured."
            )

    def _handle_manual_action(self, payload: dict) -> None:
        if not self._recording_active:
            return
        action = payload.get("action")
        parameters = payload.get("parameters")
        if not isinstance(action, str) or not isinstance(parameters, dict):
            return
        self._recording_actions.append(
            RoutineAction(action=action, parameters=parameters)
        )
        self.controls_panel.log(f"Recorded action: {self._format_action_summary(action, parameters)}")

    def _save_recorded_routine(self) -> None:
        if not self._recording_actions:
            QtWidgets.QMessageBox.information(
                self,
                "Save Routine",
                "Record at least one manual action before saving.",
            )
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Save Routine",
            "Routine name",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        existing_index = next(
            (index for index, routine in enumerate(self._routines) if routine.name == name),
            None,
        )
        if existing_index is not None:
            response = QtWidgets.QMessageBox.question(
                self,
                "Save Routine",
                f"Routine '{name}' already exists. Replace it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if response != QtWidgets.QMessageBox.Yes:
                return
        routine = Routine(
            name=name,
            actions=list(self._recording_actions),
            created_at=datetime.now().isoformat(),
        )
        if existing_index is None:
            self._routines.append(routine)
        else:
            self._routines[existing_index] = routine
        self._save_routines()
        self.controls_panel.log(f"Routine saved: {name} ({len(routine.actions)} actions).")

    def _condition_allows(self, condition: Optional[str]) -> bool:
        if not condition:
            return True
        normalized = condition.strip().lower()
        context = {
            "control_enabled": self.controls_panel.enable_control_toggle.isChecked(),
            "dry_run": self.controls_panel.is_dry_run_enabled(),
        }
        if normalized == "control_enabled":
            return context["control_enabled"]
        if normalized == "control_disabled":
            return not context["control_enabled"]
        if normalized == "dry_run":
            return context["dry_run"]
        if normalized in {"not dry_run", "dry_run_disabled"}:
            return not context["dry_run"]
        self.controls_panel.log(f"Routine condition skipped: '{condition}' is not supported.")
        return False

    def _run_selected_routine(self) -> None:
        routine_name = self.controls_panel.selected_routine_name()
        if not routine_name:
            QtWidgets.QMessageBox.information(
                self,
                "Run Routine",
                "Select a routine to run.",
            )
            return
        routine = next((item for item in self._routines if item.name == routine_name), None)
        if not routine:
            QtWidgets.QMessageBox.information(
                self,
                "Run Routine",
                "Selected routine could not be found.",
            )
            return
        self.controls_panel.log(f"Running routine: {routine.name}.")
        self._append_audit_entry(
            {"event": "routine_started", "name": routine.name, "actions": len(routine.actions)}
        )
        eligible_actions: list[RoutineAction] = []
        summaries: list[str] = []
        for action in routine.actions:
            if not self._condition_allows(action.condition):
                self.controls_panel.log(
                    f"Skipping action due to condition: {action.action}"
                )
                continue
            eligible_actions.append(action)
            summary = self._format_action_summary(action.action, action.parameters)
            if action.condition:
                summary = f"{summary} (if {action.condition})"
            summaries.append(summary)
        if not eligible_actions:
            self.controls_panel.log("Routine completed: no eligible actions to run.")
            self._append_audit_entry(
                {"event": "routine_completed", "name": routine.name, "actions_run": 0}
            )
            return
        if (
            len(eligible_actions) == 1
            or not self.controls_panel.batch_review_toggle.isChecked()
        ):
            for action in eligible_actions:
                self._dispatch_action(action.to_payload(), confirm=True)
        else:
            approved_indices = self._review_actions(summaries)
            for index in approved_indices:
                self._dispatch_action(eligible_actions[index].to_payload(), confirm=False)
        self.controls_panel.log("Routine execution finished.")
        self._append_audit_entry(
            {
                "event": "routine_completed",
                "name": routine.name,
                "actions_run": len(eligible_actions),
            }
        )

    def _optional_module(self, module_name: str):
        if importlib.util.find_spec(module_name) is None:
            return None
        return importlib.import_module(module_name)

    def _summarize_preferences(self) -> str:
        preferences = self._memory.get("preferences", {})
        if not isinstance(preferences, dict):
            preferences = {}
        provider = preferences.get("provider")
        model_map = preferences.get("model_map")
        model = None
        if isinstance(model_map, dict) and provider in model_map:
            model = model_map.get(provider)
        behavior = self._behavior_preferences()
        persona = behavior["persona"]
        tone = behavior["tone"]
        temperature = behavior["temperature"]
        max_tokens = behavior["max_tokens"]
        if provider and model:
            return (
                "Preferences: "
                f"provider={provider}, model={model}, persona={persona}, tone={tone}, "
                f"temp={temperature}, max_tokens={max_tokens}"
            )
        return (
            "Preferences: "
            f"provider={provider or 'none'}, persona={persona}, tone={tone}, "
            f"temp={temperature}, max_tokens={max_tokens}"
        )

    def _update_preferences_label(self) -> None:
        self.preferences_label.setText(self._summarize_preferences())

    def _behavior_preferences(self) -> dict[str, object]:
        preferences = self._memory.get("preferences", {})
        if not isinstance(preferences, dict):
            preferences = {}
        temperature = self._coerce_float(
            preferences.get("temperature"), self.DEFAULT_PREFERENCES["temperature"]
        )
        max_tokens = self._coerce_int(
            preferences.get("max_tokens"), self.DEFAULT_PREFERENCES["max_tokens"]
        )
        return {
            "persona": preferences.get("persona") or self.DEFAULT_PREFERENCES["persona"],
            "tone": preferences.get("tone") or self.DEFAULT_PREFERENCES["tone"],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

    def _coerce_float(self, value: object, fallback: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return fallback

    def _coerce_int(self, value: object, fallback: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    def _build_slider_row(
        self, slider: QtWidgets.QSlider, label: QtWidgets.QLabel
    ) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(label)
        return row

    def _load_behavior_preferences(self) -> None:
        preferences = self._behavior_preferences()
        self.persona_input.setText(str(preferences["persona"]))
        tone = str(preferences["tone"])
        tone_index = self.tone_selector.findText(tone)
        if tone_index == -1:
            self.tone_selector.addItem(tone)
            tone_index = self.tone_selector.findText(tone)
        self.tone_selector.setCurrentIndex(tone_index)
        temperature = float(preferences["temperature"])
        self.temperature_slider.setValue(int(round(temperature * 100)))
        self._update_temperature_label(temperature)
        max_tokens = int(preferences["max_tokens"])
        self.max_tokens_slider.setValue(max_tokens)
        self._update_max_tokens_label(max_tokens)

    def _handle_temperature_change(self, value: int) -> None:
        temperature = value / 100
        self._update_temperature_label(temperature)
        self._persist_behavior_preferences()

    def _handle_max_tokens_change(self, value: int) -> None:
        self._update_max_tokens_label(value)
        self._persist_behavior_preferences()

    def _update_temperature_label(self, temperature: float) -> None:
        self.temperature_value_label.setText(f"{temperature:.2f}")

    def _update_max_tokens_label(self, max_tokens: int) -> None:
        self.max_tokens_value_label.setText(str(max_tokens))

    def _persist_behavior_preferences(self) -> None:
        preferences = self._memory.setdefault("preferences", {})
        if not isinstance(preferences, dict):
            preferences = {}
            self._memory["preferences"] = preferences
        persona = self.persona_input.text().strip() or self.DEFAULT_PREFERENCES["persona"]
        tone = self.tone_selector.currentText().strip() or self.DEFAULT_PREFERENCES["tone"]
        temperature = self.temperature_slider.value() / 100
        max_tokens = self.max_tokens_slider.value()
        preferences.update(
            {
                "persona": persona,
                "tone": tone,
                "temperature": round(temperature, 2),
                "max_tokens": max_tokens,
            }
        )
        self._save_memory()
        self._update_preferences_label()

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
        self._append_audit_entry(
            {
                "event": "task_outcome",
                "title": title,
                "step": step_count,
                "success": success,
                "summary": summary,
            }
        )
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

    def _apply_template(self) -> None:
        if not self._templates:
            QtWidgets.QMessageBox.information(
                self,
                "Apply Template",
                "No templates are available yet.",
            )
            return
        template_names = [template["name"] for template in self._templates]
        selected, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Apply Template",
            "Template",
            template_names,
            0,
            False,
        )
        if not ok:
            return
        selected_name = selected.strip()
        if not selected_name:
            return
        template = next(
            (item for item in self._templates if item.get("name") == selected_name),
            None,
        )
        if not template:
            return
        if self.task_model.rowCount() > 0:
            response = QtWidgets.QMessageBox.question(
                self,
                "Apply Template",
                "Replace the current task list with this template?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if response != QtWidgets.QMessageBox.Yes:
                return
        tasks = [Task(title=title) for title in template.get("tasks", [])]
        self.task_model.set_tasks(tasks)
        self.task_view.scrollToBottom()
        self._refresh_task_controls()

    def _save_template(self) -> None:
        tasks = [task.title for task in self.task_model.tasks() if task.title]
        if not tasks:
            QtWidgets.QMessageBox.information(
                self,
                "Save Template",
                "Add at least one task before saving a template.",
            )
            return
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Save Template",
            "Template name",
        )
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        existing_index = next(
            (index for index, item in enumerate(self._templates) if item.get("name") == name),
            None,
        )
        if existing_index is not None:
            response = QtWidgets.QMessageBox.question(
                self,
                "Save Template",
                f"Template '{name}' already exists. Replace it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if response != QtWidgets.QMessageBox.Yes:
                return
        template_payload = {"name": name, "tasks": tasks}
        if existing_index is None:
            self._templates.append(template_payload)
        else:
            self._templates[existing_index] = template_payload
        self._save_templates()

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
        self._save_conversation_messages()
        self._save_conversation_index()
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
        if role == "user":
            meta = self._active_conversation_meta()
            if meta:
                title = meta.get("title", "")
                if title.lower().startswith("conversation"):
                    snippet = content.strip().splitlines()[0][:60]
                    meta["title"] = snippet or title
                    self._save_conversation_index()
                    self._populate_conversation_selector()
        self._save_conversation_messages()

    def _extract_actions(self, response: str) -> list[dict[str, object]]:
        action_blocks: list[dict[str, object]] = []
        for match in re.finditer(r"```json\\s*(\\{.*?\\})\\s*```", response, re.DOTALL):
            snippet = match.group(1)
            try:
                payload = json.loads(snippet)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and "action" in payload:
                action_blocks.append(payload)
            elif isinstance(payload, dict) and "actions" in payload:
                actions = payload.get("actions")
                if isinstance(actions, list):
                    action_blocks.extend(action for action in actions if isinstance(action, dict))
        return action_blocks

    def _format_action_summary(self, action: str, parameters: dict[str, object]) -> str:
        if action == "move_mouse":
            return f"Move mouse to ({parameters.get('x')}, {parameters.get('y')})"
        if action == "click_mouse":
            return (
                "Click mouse "
                f"button={parameters.get('button', 'left')}, "
                f"clicks={parameters.get('clicks', 1)}"
            )
        if action == "type_text":
            text = str(parameters.get("text", ""))
            preview = text if len(text) <= 60 else f"{text[:57]}..."
            return f"Type text: {preview!r}"
        if action == "press_key":
            return f"Press key: {parameters.get('key')}"
        return f"Unknown action: {action}"

    def _dispatch_action(
        self, action_payload: dict[str, object], confirm: bool = True
    ) -> None:
        action = action_payload.get("action")
        if not isinstance(action, str):
            return
        parameters = action_payload.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
        summary = self._format_action_summary(action, parameters)
        self._append_audit_entry(
            {
                "event": "control_action_requested",
                "action": action,
                "parameters": parameters,
                "summary": summary,
                "confirmation_required": confirm,
            }
        )
        self.controls_panel.log(f"Action requested: {summary}")
        if not self.controls_panel.isEnabled():
            self.controls_panel.log("Action skipped: control panel is disabled.")
            self._append_audit_entry(
                {
                    "event": "control_action_skipped",
                    "action": action,
                    "summary": summary,
                    "reason": "control panel disabled",
                }
            )
            return
        if self.controls_panel.is_dry_run_enabled():
            self.controls_panel.log(f"Dry run: action not executed ({summary}).")
            self._append_audit_entry(
                {
                    "event": "control_action_dry_run",
                    "action": action,
                    "summary": summary,
                    "parameters": parameters,
                }
            )
            return
        if confirm:
            prompt = f"Execute action?\n\n{summary}"
            response = QtWidgets.QMessageBox.question(
                self,
                "Confirm Action",
                prompt,
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            approved = response == QtWidgets.QMessageBox.Yes
            self._append_audit_entry(
                {
                    "event": "control_action_confirmation",
                    "action": action,
                    "summary": summary,
                    "approved": approved,
                }
            )
            if not approved:
                self.controls_panel.log("Action canceled by user.")
                self._append_audit_entry(
                    {
                        "event": "control_action_canceled",
                        "action": action,
                        "summary": summary,
                        "reason": "user declined confirmation",
                    }
                )
                return
        dispatch_map = {
            "move_mouse": self.controls_panel.move_mouse,
            "click_mouse": self.controls_panel.click_mouse,
            "type_text": self.controls_panel.type_text,
            "press_key": self.controls_panel.press_key,
        }
        handler = dispatch_map.get(action)
        if handler is None:
            self.controls_panel.log(f"Action ignored: unsupported action '{action}'.")
            self._append_audit_entry(
                {
                    "event": "control_action_unsupported",
                    "action": action,
                    "summary": summary,
                }
            )
            return
        outcome = handler(**parameters)
        self.controls_panel.log("Action executed.")
        self._append_audit_entry(
            {
                "event": "control_action_executed",
                "action": action,
                "summary": summary,
                "success": bool(outcome),
            }
        )

    def _review_actions(self, summaries: list[str]) -> list[int]:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Review Actions")
        dialog.setModal(True)

        info_label = QtWidgets.QLabel(
            "Review planned actions before execution. Uncheck any action to skip."
        )
        info_label.setWordWrap(True)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_contents = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_contents)
        checkboxes: list[QtWidgets.QCheckBox] = []
        for summary in summaries:
            checkbox = QtWidgets.QCheckBox(summary)
            checkbox.setChecked(True)
            scroll_layout.addWidget(checkbox)
            checkboxes.append(checkbox)
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_contents)

        approve_all_button = QtWidgets.QPushButton("Approve All")
        reject_all_button = QtWidgets.QPushButton("Reject All")
        execute_selected_button = QtWidgets.QPushButton("Execute Selected")

        decision: dict[str, str] = {}

        def set_decision(value: str) -> None:
            decision["value"] = value
            dialog.accept()

        approve_all_button.clicked.connect(lambda: set_decision("approve_all"))
        reject_all_button.clicked.connect(lambda: set_decision("reject_all"))
        execute_selected_button.clicked.connect(
            lambda: set_decision("execute_selected")
        )

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(approve_all_button)
        button_row.addWidget(reject_all_button)
        button_row.addStretch()
        button_row.addWidget(execute_selected_button)

        layout = QtWidgets.QVBoxLayout(dialog)
        layout.addWidget(info_label)
        layout.addWidget(scroll_area)
        layout.addLayout(button_row)

        result = dialog.exec()
        selection = decision.get("value")
        if result != QtWidgets.QDialog.Accepted or selection is None:
            return []
        if selection == "approve_all":
            return list(range(len(summaries)))
        if selection == "execute_selected":
            return [index for index, checkbox in enumerate(checkboxes) if checkbox.isChecked()]
        return []

    def _dispatch_actions(self, response: str) -> None:
        actions = self._extract_actions(response)
        if not actions:
            return
        if (
            len(actions) == 1
            or not self.controls_panel.batch_review_toggle.isChecked()
        ):
            for action in actions:
                self._dispatch_action(action, confirm=True)
            return
        summaries = []
        for action_payload in actions:
            action = action_payload.get("action")
            parameters = action_payload.get("parameters")
            if not isinstance(action, str):
                summaries.append("Unknown action: <invalid>")
                continue
            summary_parameters = (
                parameters if isinstance(parameters, dict) else {}
            )
            summaries.append(self._format_action_summary(action, summary_parameters))
        approved_indices = self._review_actions(summaries)
        approved_set = set(approved_indices)
        self.controls_panel.log(
            f"Batch review: approved {len(approved_indices)}/{len(actions)} actions."
        )
        for index, summary in enumerate(summaries):
            if index in approved_set:
                self.controls_panel.log(f"Action approved: {summary}")
            else:
                self.controls_panel.log(f"Action rejected: {summary}")
        for index in approved_indices:
            self._dispatch_action(actions[index], confirm=False)

    def _call_with_retries(self, provider: str, api_call: Callable[[], str]) -> str:
        max_retries = 3
        base_delay = 1.2
        for attempt in range(max_retries + 1):
            try:
                result = api_call()
            except Exception:  # pragma: no cover - network call
                if attempt >= max_retries:
                    self._set_provider_health(
                        provider,
                        ProviderHealthState.ERROR,
                        error_message=self._friendly_api_status(provider),
                    )
                    raise
                delay = base_delay * (2**attempt)
                delay += random.uniform(0, 0.4)
                self._set_provider_health(
                    provider,
                    ProviderHealthState.RETRYING,
                    error_message=self._friendly_retry_status(provider, delay),
                    retry_after_seconds=delay,
                )
                QtWidgets.QApplication.processEvents()
                time.sleep(delay)
                continue
            self._set_provider_health(provider, ProviderHealthState.OK)
            return result
        raise RuntimeError("Failed to obtain response after retries.")

    def _friendly_retry_status(self, provider: str, delay: float) -> str:
        label = PROVIDER_REGISTRY[provider]["label"]
        return f"{label} API error. Retrying in {delay:.1f}s…"

    def _friendly_api_status(self, provider: str) -> str:
        label = PROVIDER_REGISTRY[provider]["label"]
        return f"{label} API error. Please try again shortly."

    def _friendly_api_error(self, provider: str) -> str:
        label = PROVIDER_REGISTRY[provider]["label"]
        health = self._provider_health.get(provider)
        time_str = (
            health.last_error_time.strftime("%H:%M:%S")
            if health and health.last_error_time
            else "an unknown time"
        )
        return (
            f"{label} API is unavailable right now. "
            f"Last error at {time_str}. Please try again shortly."
        )

    def _request_model_response(
        self,
        provider: str,
        model_id: str,
        system_message: str,
        messages: list[dict[str, str]],
        messages_without_system: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[bool, str]:
        provider_info = PROVIDER_REGISTRY[provider]
        env_key = provider_info["env_key"]

        if provider in {"openai", "deepseek", "groq", "mistral", "perplexity"}:
            if openai is None:
                return (
                    False,
                    "I'm ready to help. Install the OpenAI SDK to enable "
                    "live model responses.",
                )
            self._maybe_prompt_api_key()
            api_key = os.getenv(env_key)
            if not api_key:
                return (
                    False,
                    f"Set {env_key} to enable {provider_info['label']} responses.",
                )
            client_kwargs = {"api_key": api_key}
            base_url = provider_info.get("base_url")
            if base_url:
                client_kwargs["base_url"] = base_url
            client = openai.OpenAI(**client_kwargs)
            try:
                response = self._call_with_retries(
                    provider,
                    lambda: client.chat.completions.create(
                        model=model_id,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ).choices[0].message.content,
                )
                return True, response
            except Exception:  # pragma: no cover - network call
                self._set_provider_health(
                    provider,
                    ProviderHealthState.ERROR,
                    error_message=self._friendly_api_status(provider),
                )
                return False, self._friendly_api_error(provider)

        if provider == "anthropic":
            if anthropic is None:
                return (
                    False,
                    "I'm ready to help. Install the Anthropic SDK to enable "
                    "live model responses.",
                )
            self._maybe_prompt_api_key()
            if not os.getenv(env_key):
                return (
                    False,
                    f"Set {env_key} to enable {provider_info['label']} responses.",
                )
            client = anthropic.Anthropic(api_key=os.getenv(env_key))
            try:
                def _anthropic_call() -> str:
                    response = client.messages.create(
                        model=model_id,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        system=system_message,
                        messages=messages_without_system,
                    )
                    if response.content:
                        return response.content[0].text
                    return ""

                content = self._call_with_retries(provider, _anthropic_call)
                if content:
                    return True, content
                return True, "No response content returned from Anthropic."
            except Exception:  # pragma: no cover - network call
                self._set_provider_health(
                    provider,
                    ProviderHealthState.ERROR,
                    error_message=self._friendly_api_status(provider),
                )
                return False, self._friendly_api_error(provider)

        if provider == "gemini":
            self._maybe_prompt_api_key()
            api_key = os.getenv(env_key)
            if not api_key:
                return (
                    False,
                    f"Set {env_key} to enable {provider_info['label']} responses.",
                )
            genai = self._optional_module("google.generativeai")
            if genai is None:
                return (
                    False,
                    "Install the Google Generative AI SDK (google-generativeai) "
                    "to enable Gemini responses.",
                )
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_id)
            prompt = "\n".join(
                f"{msg['role']}: {msg['content']}"
                for msg in messages
            )
            try:
                content = self._call_with_retries(
                    provider,
                    lambda: model.generate_content(prompt).text
                    or "No response content returned from Gemini.",
                )
                return True, content
            except Exception:  # pragma: no cover - network call
                self._set_provider_health(
                    provider,
                    ProviderHealthState.ERROR,
                    error_message=self._friendly_api_status(provider),
                )
                return False, self._friendly_api_error(provider)

        if provider == "cohere":
            self._maybe_prompt_api_key()
            api_key = os.getenv(env_key)
            if not api_key:
                return (
                    False,
                    f"Set {env_key} to enable {provider_info['label']} responses.",
                )
            cohere = self._optional_module("cohere")
            if cohere is None:
                return (
                    False,
                    "Install the Cohere SDK (cohere) to enable Cohere responses.",
                )
            client = cohere.Client(api_key)
            prompt = "\n".join(
                f"{msg['role']}: {msg['content']}"
                for msg in messages
            )
            try:
                content = self._call_with_retries(
                    provider,
                    lambda: client.chat(model=model_id, message=prompt).text
                    or "No response content returned from Cohere.",
                )
                return True, content
            except Exception:  # pragma: no cover - network call
                self._set_provider_health(
                    provider,
                    ProviderHealthState.ERROR,
                    error_message=self._friendly_api_status(provider),
                )
                return False, self._friendly_api_error(provider)

        return False, f"No client available for provider '{provider}'."

    def _capture_screenshot_context(self) -> Optional[dict[str, object]]:
        if pyautogui is None:
            self.controls_panel.log("Screenshot context skipped: pyautogui unavailable.")
            return None
        try:
            snapshot = pyautogui.screenshot()
        except Exception as exc:  # pragma: no cover - environment dependent
            self.controls_panel.log(f"Screenshot capture failed: {exc}")
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = Path.home() / ".kaligpt" / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_dir / f"snapshot_{timestamp}.png"
        try:
            snapshot.save(snapshot_path)
        except Exception as exc:  # pragma: no cover - environment dependent
            self.controls_panel.log(f"Snapshot save failed: {exc}")
            return None
        self.controls_panel.log(f"Captured screenshot for context: {snapshot_path}")
        return {
            "path": str(snapshot_path),
            "width": snapshot.width,
            "height": snapshot.height,
            "timestamp": timestamp,
        }

    def _run_screenshot_analysis(
        self,
        provider: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        chat_messages: list[dict[str, str]],
    ) -> Optional[str]:
        if not self.controls_panel.is_screenshot_context_enabled():
            return None
        context = self._capture_screenshot_context()
        if not context:
            return None
        analysis_system = (
            "You are analyzing a desktop screenshot for an automation assistant. "
            "Summarize the visible UI, active application cues, and anything that "
            "matters for the next action. Do not include JSON action blocks."
        )
        last_user = next(
            (msg["content"] for msg in reversed(chat_messages) if msg.get("role") == "user"),
            "",
        )
        user_lines = [
            "Screenshot metadata:",
            f"- path: {context['path']}",
            f"- size: {context['width']}x{context['height']}",
            f"- captured: {context['timestamp']}",
        ]
        if last_user:
            user_lines.append(f"User goal: {last_user}")
        user_message = "\n".join(user_lines)
        analysis_messages = [
            {"role": "system", "content": analysis_system},
            {"role": "user", "content": user_message},
        ]
        analysis_messages_without_system = [
            {"role": "user", "content": user_message}
        ]
        success, analysis = self._request_model_response(
            provider,
            model_id,
            analysis_system,
            analysis_messages,
            analysis_messages_without_system,
            temperature,
            max_tokens,
        )
        if not success:
            self.controls_panel.log("Screenshot analysis skipped: model unavailable.")
            return None
        self.controls_panel.log("Screenshot analysis completed.")
        return analysis.strip() if analysis else None

    def _generate_response(self) -> str:
        provider = self._selected_provider()
        if not provider:
            return "Select a provider to start chatting."
        model_id = self._selected_model_id()
        behavior = self._behavior_preferences()
        system_message = (
            "You are {persona}. Respond in a {tone} tone. "
            "When you want to request a desktop automation action, include a JSON snippet "
            "wrapped in a ```json code fence after your response. "
            "Use the format: {\"action\": \"move_mouse|click_mouse|type_text|press_key\", "
            "\"parameters\": {\"x\": 0, \"y\": 0}}. "
            "Use parameters appropriate to the action: move_mouse(x, y, duration), "
            "click_mouse(button, clicks, interval), type_text(text), press_key(key). "
            "You may include multiple actions by returning multiple JSON blocks or a single "
            "{\"actions\": [...]} block. Only use valid JSON."
        ).format(persona=behavior["persona"], tone=behavior["tone"])
        chat_messages = self.chat_model.as_openai_messages()
        messages = [{"role": "system", "content": system_message}]
        analysis_context = self._run_screenshot_analysis(
            provider,
            model_id,
            temperature=0.2,
            max_tokens=min(512, int(behavior["max_tokens"])),
            chat_messages=chat_messages,
        )
        if analysis_context:
            messages.append(
                {
                    "role": "system",
                    "content": f"Screenshot analysis context:\n{analysis_context}",
                }
            )
        messages.extend(chat_messages)
        temperature = float(behavior["temperature"])
        max_tokens = int(behavior["max_tokens"])

        success, response = self._request_model_response(
            provider,
            model_id,
            system_message,
            messages,
            chat_messages,
            temperature,
            max_tokens,
        )
        if success:
            self._dispatch_actions(response)
        return response


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
