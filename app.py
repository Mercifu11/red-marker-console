#!/usr/bin/env python3
"""PUBG-style GUI integrating red scanning, template similarity and slicing."""

from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
import time
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QProcess, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import red_slice
import scan_red
import similarity


NAV_BG = "#10121F"
CARD_BG = "rgba(255,255,255,0.055)"
ACCENT = "#F97316"
SUCCESS = "#34D399"
TEXT = "#F8FAFC"


class Worker(QThread):
    progress = Signal(int, int, str)
    log = Signal(str, str)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, target, parent=None):
        super().__init__(parent)
        self.target = target

    def run(self):
        try:
            result = self.target(self)
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class TimelineSlider(QSlider):
    """Video scrubbing slider with a narrow vertical I-shaped needle."""

    def __init__(self, parent=None):
        super().__init__(Qt.Horizontal, parent)
        self.setRange(0, 100)
        self.setValue(0)
        self.dark = True

    def set_theme(self, dark: bool):
        self.dark = dark
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        groove_y = self.height() // 2
        painter.setPen(Qt.NoPen)
        track_color = QColor(255, 255, 255, 45) if self.dark else QColor("#CBD5E1")
        painter.setBrush(track_color)
        painter.drawRoundedRect(2, groove_y - 3, self.width() - 4, 6, 3, 3)

        minimum = self.minimum()
        maximum = max(minimum + 1, self.maximum())
        ratio = (self.value() - minimum) / (maximum - minimum)
        x = int(4 + ratio * max(0, self.width() - 8))
        painter.setPen(QPen(QColor("#F97316"), 3))
        painter.drawLine(x, 8, x, self.height() - 8)
        cap_color = QColor("#F8FAFC") if self.dark else QColor("#0F172A")
        painter.setPen(QPen(cap_color, 8))
        painter.drawLine(x, 5, x, 12)
        painter.drawLine(x, self.height() - 12, x, self.height() - 5)


class Card(QFrame):
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 14)
        self.layout.setSpacing(10)
        if title:
            label = QLabel(title)
            label.setObjectName("cardTitle")
            self.layout.addWidget(label)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(32)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(15, 23, 42, 45))
        self.setGraphicsEffect(shadow)


class RoiSelector(QWidget):
    roi_changed = Signal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("roiSelector")
        self.setMinimumWidth(720)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.pixmap: Optional[QPixmap] = None
        self.orig_size = (1920, 1080)
        self.roi = (888, 759, 137, 33)
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.selecting = False
        self.start_point = None
        self.end_point = None

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return max(1, round(width * 9 / 16))

    def set_image(self, image: np.ndarray):
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimage = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format_RGB888,
        ).copy()
        self.pixmap = QPixmap.fromImage(qimage)
        self.orig_size = (1920, 1080)
        self.update()

    def set_roi(self, x, y, w, h):
        self.roi = (x, y, w, h)
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if self.selecting and event.button() == Qt.LeftButton:
            self.start_point = (event.position().x(), event.position().y())
            self.end_point = self.start_point

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.selecting and self.start_point is not None:
            self.end_point = (event.position().x(), event.position().y())
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self.selecting and self.start_point is not None:
            self.end_point = (event.position().x(), event.position().y())
            if self.pixmap is not None:
                x1, y1 = self._to_orig(self.start_point)
                x2, y2 = self._to_orig(self.end_point)
                left = min(x1, x2)
                top = min(y1, y2)
                self.roi = (
                    max(0, left),
                    max(0, top),
                    max(1, abs(x2 - x1)),
                    max(1, abs(y2 - y1)),
                )
                self.roi_changed.emit(*self.roi)
        self.start_point = None
        self.end_point = None
        self.update()

    def _to_orig(self, point):
        x = int((point[0] - self.offset_x) / self.scale)
        y = int((point[1] - self.offset_y) / self.scale)
        x = max(0, min(self.orig_size[0] - 1, x))
        y = max(0, min(self.orig_size[1] - 1, y))
        return x, y

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101521"))
        if self.pixmap is None:
            painter.setPen(QColor(255, 255, 255, 120))
            painter.drawText(self.rect(), Qt.AlignCenter, "上传预览图后进行框选")
            return
        scale = min(
            self.width() / self.pixmap.width(),
            self.height() / self.pixmap.height(),
        )
        self.scale = scale
        display_w = int(self.pixmap.width() * scale)
        display_h = int(self.pixmap.height() * scale)
        self.offset_x = (self.width() - display_w) / 2
        self.offset_y = (self.height() - display_h) / 2
        painter.drawPixmap(
            int(self.offset_x),
            int(self.offset_y),
            display_w,
            display_h,
            self.pixmap,
        )
        x, y, rw, rh = self.roi
        painter.setPen(QPen(QColor(ACCENT), 3))
        painter.drawRect(
            int(self.offset_x + x * scale),
            int(self.offset_y + y * scale),
            max(1, int(rw * scale)),
            max(1, int(rh * scale)),
        )
        if self.start_point and self.end_point:
            painter.setPen(QPen(QColor(SUCCESS), 2, Qt.DashLine))
            painter.drawRect(
                int(self.start_point[0]),
                int(self.start_point[1]),
                int(self.end_point[0] - self.start_point[0]),
                int(self.end_point[1] - self.start_point[1]),
            )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PUBG Console")
        self.resize(1560, 980)
        self.setMinimumSize(1320, 860)
        self.roi = scan_red.Region(888, 759, 137, 33)
        self.gray_threshold = 180
        self.similarity_threshold = 85
        self.before_time = 10
        self.after_time = 10
        self.video_path = ""
        self.template_paths = ["", "", ""]
        self.templates: list[Optional[list[float]]] = [None, None, None]
        self.raw_markers = ""
        self.filtered_markers = ""
        self.grouped_markers = ""
        self.audio_streams: list[int] = []
        self.audio_checks: list[QCheckBox] = []
        self.audio_notes: list[QLineEdit] = []
        self.audio_container: Optional[QWidget] = None
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.current_frame: Optional[np.ndarray] = None
        self.current_worker: Optional[Worker] = None
        self.slice_process: Optional[QProcess] = None
        self.is_dark = True
        self.app_settings_path = os.path.join(
            os.path.dirname(__file__),
            "app_settings.json",
        )
        self.audio_config_path = os.path.join(
            os.path.dirname(__file__),
            "audio_track_config.json",
        )
        self._build_ui()
        self._load_notes()
        self._load_app_settings()

    def _build_ui(self):
        root = QWidget()
        root.setObjectName("appRoot")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        nav = QFrame()
        nav.setObjectName("nav")
        nav.setFixedWidth(220)
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(16, 24, 16, 20)
        title = QLabel("PUBG")
        title.setObjectName("navTitle")
        nav_layout.addWidget(title)
        nav_layout.addSpacing(36)
        self.nav_buttons = []
        for key, label in (("dash", "大屏"), ("settings", "参数设置"), ("notes", "优化记录")):
            button = QPushButton(label)
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, k=key: self.switch_page(k))
            nav_layout.addWidget(button)
            self.nav_buttons.append(button)
        nav_layout.addStretch(1)
        self.theme_btn = QPushButton("白昼模式")
        self.theme_btn.setObjectName("navButton")
        self.theme_btn.clicked.connect(self.toggle_theme)
        nav_layout.addWidget(self.theme_btn)
        version = QLabel("v0.3 similarity")
        version.setObjectName("navVersion")
        nav_layout.addWidget(version)
        layout.addWidget(nav)

        self.stack = QStackedWidget()
        layout.addWidget(self.stack, 1)
        self.dash_page = self._build_dash_page()
        self.settings_page = self._build_settings_page()
        self.notes_page = self._build_notes_page()
        for page, name in (
            (self.dash_page, "dashPage"),
            (self.settings_page, "settingsPage"),
            (self.notes_page, "notesPage"),
        ):
            page.setObjectName(name)
            page.setAttribute(Qt.WA_StyledBackground, True)
        self.stack.addWidget(self.dash_page)
        self.stack.addWidget(self.settings_page)
        self.stack.addWidget(self.notes_page)
        self._apply_styles()
        self.nav_buttons[0].setChecked(True)

    def toggle_theme(self):
        self.is_dark = not self.is_dark
        self._apply_styles()
        self._save_app_settings()

    def switch_page(self, key: str):
        index = {"dash": 0, "settings": 1, "notes": 2}[key]
        self.stack.setCurrentIndex(index)
        for button in self.nav_buttons:
            button.setChecked(False)
        self.nav_buttons[index].setChecked(True)

    def _build_dash_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        header = QLabel("大屏")
        header.setObjectName("pageTitle")
        layout.addWidget(header)
        grid = QGridLayout()
        grid.setSpacing(14)

        upload_card = Card("上传视频")
        self.choose_video_btn = QPushButton("上传视频")
        self.choose_video_btn.setObjectName("primaryBtn")
        self.choose_video_btn.clicked.connect(self.choose_video)
        upload_card.layout.addWidget(self.choose_video_btn)
        self.video_path_label = QLabel("未选择视频")
        self.video_path_label.setObjectName("pathLabel")
        self.video_path_label.setWordWrap(True)
        upload_card.layout.addWidget(self.video_path_label)
        tpl_title = QLabel("模板投影图（扫描前必传）")
        tpl_title.setObjectName("smallTitle")
        upload_card.layout.addWidget(tpl_title)
        self.template_buttons = []
        self.template_status = []
        for i in range(3):
            row = QHBoxLayout()
            button = QPushButton(f"模板{i + 1}")
            button.setObjectName("ghostBtn")
            button.clicked.connect(lambda _=False, n=i: self.choose_template(n))
            status = QLabel("未上传")
            status.setObjectName("muted")
            row.addWidget(button)
            row.addWidget(status, 1)
            upload_card.layout.addLayout(row)
            self.template_buttons.append(button)
            self.template_status.append(status)
        grid.addWidget(upload_card, 0, 0)

        preview_card = Card("视频预览")
        self.preview_label = QLabel("选择视频后显示预览")
        self.preview_label.setObjectName("previewLabel")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(680, 380)
        preview_card.layout.addWidget(self.preview_label, 1)
        frame_action_row = QHBoxLayout()
        send_frame_btn = QPushButton("当前帧设为框选预览")
        send_frame_btn.setObjectName("ghostBtn")
        send_frame_btn.clicked.connect(self.send_current_frame_to_settings)
        frame_action_row.addStretch(1)
        frame_action_row.addWidget(send_frame_btn)
        preview_card.layout.addLayout(frame_action_row)
        self.timeline = TimelineSlider()
        self.timeline.set_theme(self.is_dark)
        self.timeline.valueChanged.connect(self._on_timeline_changed)
        self.timeline.sliderReleased.connect(self.load_preview_at_time)
        preview_card.layout.addWidget(self.timeline)
        self.time_label = QLabel("00:00:00")
        self.time_label.setObjectName("muted")
        preview_card.layout.addWidget(self.time_label)
        grid.addWidget(preview_card, 0, 1)

        exec_card = Card("执行交互")
        self.scan_btn = QPushButton("开始扫描")
        self.scan_btn.setObjectName("primaryBtn")
        self.scan_btn.clicked.connect(self.start_scan)
        exec_card.layout.addWidget(self.scan_btn)
        self.merge_btn = QPushButton("归并时间戳")
        self.merge_btn.setObjectName("ghostBtn")
        self.merge_btn.clicked.connect(self.merge_markers)
        exec_card.layout.addWidget(self.merge_btn)
        self.slice_btn = QPushButton("开始切片")
        self.slice_btn.setObjectName("dangerBtn")
        self.slice_btn.clicked.connect(self.start_slice)
        exec_card.layout.addWidget(self.slice_btn)
        hint = QLabel("默认切片保存到视频同路径下，并创建同名文件夹进行保存")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        exec_card.layout.addWidget(hint)
        save_row = QHBoxLayout()
        save_label = QLabel("另存为")
        save_label.setObjectName("muted")
        save_row.addWidget(save_label)
        save_row.addStretch(1)
        self.save_as_btn = QPushButton("···")
        self.save_as_btn.setObjectName("roundBtn")
        self.save_as_btn.setFixedSize(60, 38)
        self.save_as_btn.clicked.connect(self.choose_output_dir)
        save_row.addWidget(self.save_as_btn)
        exec_card.layout.addLayout(save_row)
        self.output_path_label = QLabel("未选择输出目录")
        self.output_path_label.setObjectName("muted")
        exec_card.layout.addWidget(self.output_path_label)
        grid.addWidget(exec_card, 1, 0)

        log_card = Card("行为记录")
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setObjectName("logView")
        self.log_view.setMinimumHeight(180)
        log_card.layout.addWidget(self.log_view, 1)
        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setObjectName("progress")
        self.progress_label = QLabel("0%")
        self.progress_label.setObjectName("muted")
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.progress_label)
        log_card.layout.addLayout(progress_row)
        grid.addWidget(log_card, 1, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 5)
        grid.setRowStretch(1, 4)
        layout.addLayout(grid, 1)
        return page

    def _build_settings_page(self):
        page = QWidget()
        outer_layout = QVBoxLayout(page)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea{background:transparent;border:none;}")
        scroll.viewport().setAutoFillBackground(False)
        content = QWidget()
        content.setObjectName("settingsContent")
        content.setAttribute(Qt.WA_StyledBackground, True)
        content.setAutoFillBackground(False)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        outer_layout.addWidget(scroll)
        header = QLabel("参数设置")
        header.setObjectName("pageTitle")
        layout.addWidget(header)

        roi_card = Card("识别框位置框选")
        self.roi_selector = RoiSelector()
        self.roi_selector.roi_changed.connect(self._on_roi_changed)
        self.upload_preview_btn = QPushButton("上传预览图")
        self.upload_preview_btn.setObjectName("primaryBtn")
        self.upload_preview_btn.clicked.connect(self.upload_roi_preview)
        self.select_toggle = QPushButton("绘制矩形框")
        self.select_toggle.setObjectName("ghostBtn")
        self.select_toggle.setCheckable(True)
        self.select_toggle.clicked.connect(self._toggle_select)
        button_row = QHBoxLayout()
        button_row.addWidget(self.upload_preview_btn)
        button_row.addWidget(self.select_toggle)
        button_row.addStretch(1)
        roi_card.layout.addLayout(button_row)
        editor_row = QHBoxLayout()
        editor_row.addWidget(self.roi_selector, 3)
        info_card = QWidget()
        info_card.setObjectName("infoCard")
        info_card.setAttribute(Qt.WA_StyledBackground, True)
        info_layout = QVBoxLayout(info_card)
        info_layout.setContentsMargins(12, 12, 12, 12)
        info_layout.addWidget(QLabel("框选坐标", objectName="cardTitle"))
        self.roi_info_label = QLabel("888, 759, 137, 33")
        self.roi_info_label.setObjectName("pathLabel")
        info_layout.addWidget(self.roi_info_label)
        info_layout.addWidget(QLabel("手动输入坐标（1920x1080）", objectName="cardTitle"))
        coord_grid = QGridLayout()
        coord_grid.setHorizontalSpacing(8)
        coord_grid.setVerticalSpacing(6)
        self.roi_x_spin = QSpinBox()
        self.roi_y_spin = QSpinBox()
        self.roi_w_spin = QSpinBox()
        self.roi_h_spin = QSpinBox()
        self.roi_x_spin.setRange(0, 1920)
        self.roi_y_spin.setRange(0, 1080)
        self.roi_w_spin.setRange(1, 1920)
        self.roi_h_spin.setRange(1, 1080)
        self.roi_x_spin.setValue(888)
        self.roi_y_spin.setValue(759)
        self.roi_w_spin.setValue(137)
        self.roi_h_spin.setValue(33)
        coord_grid.addWidget(QLabel("x"), 0, 0)
        coord_grid.addWidget(self.roi_x_spin, 0, 1)
        coord_grid.addWidget(QLabel("y"), 1, 0)
        coord_grid.addWidget(self.roi_y_spin, 1, 1)
        coord_grid.addWidget(QLabel("w"), 2, 0)
        coord_grid.addWidget(self.roi_w_spin, 2, 1)
        coord_grid.addWidget(QLabel("h"), 3, 0)
        coord_grid.addWidget(self.roi_h_spin, 3, 1)
        info_layout.addLayout(coord_grid)
        self.apply_roi_btn = QPushButton("应用坐标")
        self.apply_roi_btn.setObjectName("ghostBtn")
        self.apply_roi_btn.clicked.connect(self._apply_manual_roi)
        info_layout.addWidget(self.apply_roi_btn)
        info_layout.addWidget(QLabel("放大预览", objectName="cardTitle"))
        self.zoom_label = QLabel("放大显示")
        self.zoom_label.setObjectName("previewLabel")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setMinimumSize(220, 140)
        info_layout.addWidget(self.zoom_label)
        info_layout.addStretch(1)
        editor_row.addWidget(info_card, 1)
        roi_card.layout.addLayout(editor_row, 1)
        layout.addWidget(roi_card)

        params_card = Card("参数滑轨")
        self.gray_control = self._make_slider("灰度阈值", 180, 0, 255)
        self.sim_control = self._make_slider("相似度阈值", 85, 0, 100)
        self.before_control = self._make_slider("时间戳前保留时间", 10, 0, 120)
        self.after_control = self._make_slider("时间戳后保留时间", 10, 0, 120)
        for control in (
            self.gray_control,
            self.sim_control,
            self.before_control,
            self.after_control,
        ):
            params_card.layout.addLayout(control[0])
        self.gray_control[2].valueChanged.connect(
            lambda v: self._set_gray(v)
        )
        self.sim_control[2].valueChanged.connect(
            lambda v: self._set_sim(v)
        )
        self.before_control[2].valueChanged.connect(
            lambda v: self._set_before(v)
        )
        self.after_control[2].valueChanged.connect(
            lambda v: self._set_after(v)
        )

        audio_title = QLabel("选择需要保存的音轨")
        audio_title.setObjectName("cardTitle")
        params_card.layout.addWidget(audio_title)
        self.audio_widget = QWidget()
        self.audio_layout = QGridLayout(self.audio_widget)
        self.audio_layout.setContentsMargins(0, 0, 0, 0)
        self.audio_layout.setHorizontalSpacing(18)
        self.audio_layout.setVerticalSpacing(4)
        params_card.layout.addWidget(self.audio_widget)
        save_row = QHBoxLayout()
        self.save_settings_btn = QPushButton("保存设置")
        self.save_settings_btn.setObjectName("primaryBtn")
        self.save_settings_btn.clicked.connect(self.save_all_settings)
        save_row.addWidget(self.save_settings_btn)
        save_row.addStretch(1)
        params_card.layout.addLayout(save_row)
        layout.addWidget(params_card)
        layout.addStretch(1)
        scroll.setWidget(content)
        return page

    def _make_slider(self, label, default, low, high):
        row = QHBoxLayout()
        text = QLabel(label)
        text.setObjectName("fieldLabel")
        text.setMinimumWidth(120)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(low, high)
        slider.setValue(default)
        slider.setObjectName("paramSlider")
        value = QLabel(str(default))
        value.setObjectName("sliderValue")
        row.addWidget(text)
        row.addWidget(slider, 1)
        row.addWidget(value)
        return row, text, slider, value

    def _build_notes_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(18, 18, 18, 18)
        header = QLabel("优化记录")
        header.setObjectName("pageTitle")
        layout.addWidget(header)
        notes_card = Card("随时记录可以优化的地方")
        self.notes_edit = QTextEdit()
        self.notes_edit.setObjectName("notesEdit")
        notes_card.layout.addWidget(self.notes_edit, 1)
        save = QPushButton("保存优化记录")
        save.setObjectName("primaryBtn")
        save.clicked.connect(self.save_notes)
        notes_card.layout.addWidget(save)
        layout.addWidget(notes_card, 1)
        return page

    def _apply_styles(self):
        dark = self.is_dark
        if dark:
            app_bg = "#0B0D17"
            nav_a, nav_b = "#171B2E", "#0E111F"
            nav_border = "rgba(255,255,255,0.08)"
            text_main = "#F8FAFC"
            text_second = "#CBD5E1"
            text_muted = "rgba(148,163,184,0.9)"
            card_bg = "rgba(255,255,255,0.055)"
            card_border = "rgba(255,255,255,0.10)"
            input_bg = "rgba(255,255,255,0.06)"
            input_border = "rgba(255,255,255,0.14)"
            log_bg = "rgba(0,0,0,0.24)"
            track = "rgba(255,255,255,0.14)"
            label_bg = "rgba(0,0,0,0.20)"
            ghost_bg = "rgba(255,255,255,0.07)"
            ghost_border = "rgba(255,255,255,0.14)"
            hint_color = "#FBBF24"
        else:
            app_bg = "#F2F5FA"
            nav_a, nav_b = "#FFFFFF", "#F8FAFC"
            nav_border = "#E2E8F0"
            text_main = "#0F172A"
            text_second = "#334155"
            text_muted = "#64748B"
            card_bg = "#FFFFFF"
            card_border = "#E2E8F0"
            input_bg = "#FFFFFF"
            input_border = "#CBD5E1"
            log_bg = "#F8FAFC"
            track = "#E2E8F0"
            label_bg = "#EFF6FF"
            ghost_bg = "#FFFFFF"
            ghost_border = "#CBD5E1"
            hint_color = "#B45309"

        qss = f"""
            QWidget#appRoot {{ background:{app_bg}; }}
            QStackedWidget {{ background:{app_bg}; }}
            QWidget#dashPage, QWidget#settingsPage, QWidget#notesPage {{
                background:transparent;
            }}
            QWidget#settingsContent {{ background:transparent; }}
            QFrame#nav {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                    stop:0 {nav_a}, stop:1 {nav_b});
                border-right: 1px solid {nav_border};
            }}
            QLabel#navTitle {{ color:{text_main}; font-size:30px; font-weight:900; }}
            QLabel#navVersion {{ color:{text_muted}; font-size:12px; }}
            QLabel#pageTitle {{ color:{text_main}; font-size:24px; font-weight:900; }}
            QPushButton#navButton {{
                text-align:left; padding:12px 14px; border-radius:12px;
                color:{text_second}; font-size:15px; font-weight:700;
                background:transparent; border:none;
            }}
            QPushButton#navButton:hover {{ background:rgba(249,115,22,0.08); }}
            QPushButton#navButton:checked {{
                background:rgba(249,115,22,0.15);
                color:#F97316;
                border:1px solid rgba(249,115,22,0.35);
            }}
            QFrame#card {{
                background:{card_bg};
                border:1px solid {card_border};
                border-radius:22px;
            }}
            QLabel#cardTitle {{ color:{text_main}; font-size:15px; font-weight:800; }}
            QLabel#smallTitle {{ color:{text_muted}; font-size:13px; font-weight:700; }}
            QLabel#muted {{ color:{text_muted}; font-size:13px; }}
            QLabel#fieldLabel {{ color:{text_second}; font-size:15px; font-weight:700; }}
            QLabel#sliderValue {{ color:#F97316; font-size:20px; font-weight:900; }}
            QLabel#hint {{ color:{hint_color}; font-size:12px; font-weight:600; }}
            QLabel#pathLabel {{
                color:{'#0C4A6E' if not dark else '#A5F3FC'};
                background:{label_bg};
                border-radius:8px; padding:8px; font-size:13px;
            }}
            QLabel#previewLabel {{
                background:{input_bg};
                color:{text_muted};
                border:1px solid {input_border};
                border-radius:14px; font-size:14px;
            }}
            QPushButton {{
                font-weight:800; font-size:15px; border-radius:14px;
            }}
            QPushButton#primaryBtn {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #EA580C, stop:1 #F97316);
                color:white; border:none;
                padding:12px 18px;
            }}
            QPushButton#primaryBtn:hover {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #D9560A, stop:1 #EA580C);
            }}
            QPushButton#primaryBtn:pressed {{
                padding:14px 18px 9px 18px;
                background:#C2410C;
            }}
            QPushButton#dangerBtn {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #DC2626, stop:1 #F43F5E);
                color:white; border:none; padding:12px 18px;
            }}
            QPushButton#dangerBtn:hover {{
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #C81E1E, stop:1 #E11D48);
            }}
            QPushButton#dangerBtn:pressed {{
                padding:14px 18px 9px 18px;
                background:#B91C1C;
            }}
            QPushButton#ghostBtn {{
                background:{ghost_bg};
                color:{text_main};
                border:1px solid {ghost_border};
                padding:10px 16px;
            }}
            QPushButton#ghostBtn:hover {{
                border:1px solid #F97316;
                color:#F97316;
            }}
            QPushButton#ghostBtn:pressed {{
                padding:12px 16px 7px 16px;
                background:rgba(249,115,22,0.12);
            }}
            QPushButton#roundBtn {{
                background:#F97316; color:white;
                border-radius:14px; font-size:20px; font-weight:900;
                border:none; padding:10px 18px;
            }}
            QPushButton#roundBtn:hover {{ background:#EA580C; }}
            QPushButton#roundBtn:pressed {{
                background:#C2410C;
                padding:12px 18px 7px 18px;
            }}
            QSlider#paramSlider::groove:horizontal {{
                height:8px; border-radius:4px; background:{track};
            }}
            QSlider#paramSlider::handle:horizontal {{
                width:22px; height:22px; margin:-7px 0; border-radius:11px;
                background:#F97316; border:2px solid {text_main};
            }}
            QTextEdit#logView {{
                background:{log_bg};
                border:1px solid {input_border};
                border-radius:12px; color:{text_second}; font-size:13px;
            }}
            QTextEdit#notesEdit {{
                background:{input_bg};
                border:1px solid {input_border};
                border-radius:12px; color:{text_main}; font-size:15px;
            }}
            QProgressBar#progress {{
                background:{track}; border:none;
                border-radius:9px; height:18px;
            }}
            QProgressBar#progress::chunk {{
                border-radius:9px;
                background:qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #F97316, stop:1 #34D399);
            }}
            QWidget#roiSelector, QWidget#infoCard {{
                background:{input_bg};
                border:1px solid {input_border};
                border-radius:14px;
            }}
            QLineEdit {{
                background:{input_bg};
                border:1px solid {input_border};
                border-radius:10px; padding:8px; color:{text_main};
            }}
            QCheckBox#audioCheck {{
                color:{text_second}; spacing:8px; font-size:14px;
            }}
        """
        self.setStyleSheet(qss)
        self.theme_btn.setText("黑夜模式" if not self.is_dark else "白昼模式")
        if hasattr(self, "timeline"):
            self.timeline.set_theme(self.is_dark)

    def switch_page_for_test(self, name: str):
        self.switch_page(name)

    def choose_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "视频文件 (*.mp4 *.mov *.mkv *.avi);;所有文件 (*.*)"
        )
        if not path:
            return
        self.video_path = path
        self.video_path_label.setText(path)
        self._log(f"视频上传成功：{path}", "success")
        duration, audio = red_slice.probe_media(path)
        self.audio_streams = audio
        self._refresh_audio_tracks()
        self._open_video_preview()
        self._auto_output()

    def choose_template(self, index: int):
        path, _ = QFileDialog.getOpenFileName(self, f"选择模板{index + 1}", "", "图片 (*.png *.jpg *.jpeg)")
        if not path:
            return
        image = similarity.load_image(path)
        projection = similarity.extract_template_projection(image)
        self.templates[index] = projection
        self.template_paths[index] = path
        self.template_status[index].setText("已上传")
        self._log(f"模板{index + 1}已提取曲线：{path}", "info")

    def _open_video_preview(self):
        if self.video_cap is not None:
            self.video_cap.release()
        self.video_cap = cv2.VideoCapture(self.video_path)
        duration = self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(1, self.video_cap.get(cv2.CAP_PROP_FPS))
        self.timeline.setRange(0, max(1, int(duration)))
        self.timeline.setValue(0)
        self.load_preview_at_time()

    def _on_timeline_changed(self):
        seconds = self.timeline.value()
        self.time_label.setText(self._format_seconds(seconds))

    def load_preview_at_time(self):
        if self.video_cap is None:
            return
        seconds = self.timeline.value()
        self.video_cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = self.video_cap.read()
        if not ok:
            return
        self.current_frame = frame
        height = 380
        scale = height / frame.shape[0]
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()
        self.preview_label.setPixmap(QPixmap.fromImage(qimg))

    def send_current_frame_to_settings(self):
        frame = getattr(self, "current_frame", None)
        if frame is None:
            QMessageBox.warning(self, "提示", "请先移动时间轴加载视频帧")
            return
        self.roi_selector.set_image(frame)
        self._update_zoom()
        self._log("当前视频帧已发送到参数设置预览", "success")
        self.switch_page("settings")

    def _auto_output(self):
        if not self.video_path:
            return
        stem = os.path.splitext(os.path.basename(self.video_path))[0]
        folder = os.path.join(os.path.dirname(self.video_path), stem)
        self.output_path_label.setText("默认输出: " + folder)

    def choose_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "选择切片保存目录")
        if directory:
            self.output_path_label.setText("输出目录: " + directory)

    def _refresh_audio_tracks(self):
        if self.audio_container is not None:
            self.audio_container.deleteLater()
        self.audio_container = QWidget()
        layout = QGridLayout(self.audio_container)
        layout.setSpacing(6)
        self.audio_checks = []
        self.audio_notes = []
        for index, stream in enumerate(self.audio_streams):
            note = QLineEdit()
            note.setPlaceholderText(f"音轨{index + 1}说明")
            check = QCheckBox(f"音轨{index + 1}")
            check.setObjectName("audioCheck")
            column = index % 3
            row = index // 3
            layout.addWidget(note, row * 2, column)
            layout.addWidget(check, row * 2 + 1, column)
            self.audio_notes.append(note)
            self.audio_checks.append(check)
            note.textChanged.connect(lambda *_: self._save_audio_config())
            check.toggled.connect(lambda *_: self._save_audio_config())
        if self.audio_layout.count():
            item = self.audio_layout.itemAt(0)
            if item is not None and item.widget() is not None:
                item.widget().deleteLater()
        self.audio_layout.addWidget(self.audio_container, 0, 0)
        self._restore_audio_config()

    def _save_audio_config(self):
        if not self.video_path:
            return
        data = {
            "video_path": self.video_path,
            "tracks": [
                {
                    "label": note.text(),
                    "checked": check.isChecked(),
                }
                for note, check in zip(self.audio_notes, self.audio_checks)
            ],
        }
        with open(self.audio_config_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    def _restore_audio_config(self):
        if not self.video_path or not os.path.exists(self.audio_config_path):
            return
        try:
            with open(self.audio_config_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        if data.get("video_path") != self.video_path:
            return
        tracks = data.get("tracks") or []
        for index, note in enumerate(self.audio_notes):
            if index < len(tracks):
                note.setText(tracks[index].get("label", ""))
        for index, check in enumerate(self.audio_checks):
            if index < len(tracks):
                check.setChecked(bool(tracks[index].get("checked", False)))

    def upload_roi_preview(self):
        path, _ = QFileDialog.getOpenFileName(self, "上传预览图", "", "图片 (*.png *.jpg *.jpeg)")
        if not path:
            return
        data = np.fromfile(path, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            QMessageBox.warning(self, "提示", "无法读取图片，请确认文件格式为 PNG/JPG/JPEG")
            return
        self.roi_selector.set_image(frame)
        self.roi_selector.set_roi(self.roi.x, self.roi.y, self.roi.width, self.roi.height)
        self._update_zoom()

    def _toggle_select(self):
        self.roi_selector.selecting = self.select_toggle.isChecked()
        self.select_toggle.setText("停止框选" if self.select_toggle.isChecked() else "开始框选")

    def _on_roi_changed(self, x, y, w, h):
        self.roi = scan_red.Region(x, y, w, h)
        self.roi_info_label.setText(f"{x}, {y}, {w}, {h}")
        self.roi_x_spin.setValue(x)
        self.roi_y_spin.setValue(y)
        self.roi_w_spin.setValue(w)
        self.roi_h_spin.setValue(h)
        self._update_zoom()

    def _apply_manual_roi(self):
        x = self.roi_x_spin.value()
        y = self.roi_y_spin.value()
        w = self.roi_w_spin.value()
        h = self.roi_h_spin.value()
        if x + w > 1920:
            w = 1920 - x
        if y + h > 1080:
            h = 1080 - y
        self.roi = scan_red.Region(x, y, max(1, w), max(1, h))
        self.roi_selector.set_roi(x, y, max(1, w), max(1, h))
        self._on_roi_changed(x, y, max(1, w), max(1, h))
        self._save_app_settings()

    def _update_zoom(self):
        image = self.roi_selector.pixmap
        if image is None:
            return
        x, y, w, h = self.roi_selector.roi
        width, height = self.roi_selector.orig_size
        if w <= 0 or h <= 0 or x + w > width or y + h > height:
            return
        crop = image.copy(x, y, w, h)
        target = max(180, crop.width())
        factor = target / crop.width()
        crop = crop.scaled(target, max(1, int(crop.height() * factor)), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.zoom_label.setPixmap(QPixmap.fromImage(crop.toImage()))

    def _set_gray(self, value):
        self.gray_threshold = value
        self.gray_control[3].setText(str(value))
        self._save_app_settings()

    def _set_sim(self, value):
        self.similarity_threshold = value
        self.sim_control[3].setText(str(value))
        self._save_app_settings()

    def _set_before(self, value):
        self.before_time = value
        self.before_control[3].setText(str(value))
        self._save_app_settings()

    def _set_after(self, value):
        self.after_time = value
        self.after_control[3].setText(str(value))
        self._save_app_settings()

    def _log(self, text, kind="info"):
        if self.is_dark:
            colors = {"info": "#CBD5E1", "start": "#FBBF24", "success": "#34D399", "error": "#F87171"}
        else:
            colors = {"info": "#475569", "start": "#B45309", "success": "#15803D", "error": "#DC2626"}
        color = colors.get(kind, colors["info"])
        self.log_view.append(f'<span style="color:{color}">{html.escape(text)}</span>')

    def _set_busy(self, busy):
        self.scan_btn.setEnabled(not busy)
        self.merge_btn.setEnabled(not busy)
        self.slice_btn.setEnabled(not busy)

    def _set_progress(self, percent):
        self.progress.setValue(percent)
        self.progress_label.setText(f"{percent}%")

    def _worker_progress(self, done, total, _label=""):
        self._set_progress(int(done / max(1, total) * 100))

    def start_scan(self):
        if not self.video_path:
            QMessageBox.warning(self, "提示", "请先上传视频")
            return
        if any(t is None for t in self.templates):
            QMessageBox.warning(self, "提示", "请先上传 3 张模板投影图")
            return
        thresholds = {
            "gray": self.gray_threshold,
            "sim": self.similarity_threshold / 100.0,
            "before": self.before_time,
            "after": self.after_time,
        }
        roi = self.roi
        video = self.video_path
        templates = list(self.templates)
        stem = os.path.splitext(os.path.basename(video))[0]
        raw_output = os.path.join(os.path.dirname(video), f"{stem}.red_timestamps.txt")
        filtered_output = os.path.join(os.path.dirname(video), f"{stem}.similarity_filtered.txt")
        self.raw_markers = raw_output
        self.filtered_markers = filtered_output
        self._set_busy(True)
        self._set_progress(0)
        self._log("正在扫描...", "start")

        def target(worker):
            snapshots: list[tuple[float, bytes]] = []

            def keep_snapshot(event_time: float, frame_bytes: bytes) -> None:
                snapshots.append((event_time, frame_bytes))

            event_count = scan_red.scan_video(
                video,
                raw_output,
                roi,
                sample_every=30,
                red_gray_min=thresholds["gray"],
                hue_window=10,
                sat_min=60,
                min_pixels=400,
                hardware=True,
                progress_callback=lambda done, total: worker.progress.emit(
                    done,
                    max(1, total * 2),
                    "",
                ),
                snapshot_callback=keep_snapshot,
            )
            worker.log.emit(f"扫描结束：{event_count} 个候选时间戳", "success")
            worker.progress.emit(50, 100, "")
            worker.log.emit(
                f"正在对 {len(snapshots)} 个时间戳做模板相似度比对",
                "start",
            )
            filtered = self._similarity_filter(
                snapshots,
                roi,
                templates,
                thresholds,
                worker,
            )
            worker.progress.emit(100, 100, "")
            return filtered

        worker = Worker(target, self)
        worker.progress.connect(self._worker_progress)
        worker.log.connect(self._log)
        worker.finished_ok.connect(self._on_scan_finished)
        worker.failed.connect(self._on_worker_error)
        self.current_worker = worker
        worker.start()

    def _similarity_filter(self, snapshots, roi, templates, thresholds, worker):
        kept = []
        total = len(snapshots)
        for index, (marker, frame_bytes) in enumerate(snapshots):
            if total:
                progress = 50 + int((index + 1) / total * 50)
                worker.progress.emit(progress, 100, "")
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(
                roi.height,
                roi.width,
                3,
            )
            red_mask = frame[..., 0] > thresholds["gray"]
            projection = list(red_mask.sum(axis=0))
            scores, best = similarity.compare_to_templates(
                projection,
                templates,
                roi.width,
            )
            best_score = scores[best] if best >= 0 else 0.0
            if best_score >= thresholds["sim"]:
                kept.append(marker)
                worker.log.emit(
                    f"相似度通过：{self._format_seconds(marker)}，模板{best + 1}，{best_score * 100:.1f}%",
                    "success",
                )
        with open(self.filtered_markers, "w", encoding="utf-8") as handle:
            for marker in kept:
                handle.write(self._format_seconds(marker) + "\n")
        worker.log.emit(f"相似度过滤完成：保留 {len(kept)} 个时间戳", "info")
        return len(kept)

    def _on_scan_finished(self, count):
        self._set_busy(False)
        self._set_progress(100)
        self._log(f"扫描与相似度过滤完成，可切片时间戳：{count}", "success")

    def merge_markers(self):
        source = self.filtered_markers if os.path.exists(self.filtered_markers) else self.raw_markers
        if not source or not os.path.exists(source):
            QMessageBox.warning(self, "提示", "请先完成扫描")
            return
        stem = os.path.splitext(os.path.basename(source))[0]
        output = os.path.join(os.path.dirname(source), f"{stem}.grouped_10s.txt")
        self.grouped_markers = output
        script = os.path.join(os.path.dirname(__file__), "merge_timestamps.py")
        self._set_busy(True)
        self._log("正在归并...", "start")
        self._run_script([sys.executable, script, source, "--gap", "10", "-o", output], self._on_merge_done)

    def _on_merge_done(self, _result=None):
        self._set_busy(False)
        self._set_progress(100)
        self._log(f"归并结束：{self.grouped_markers}", "success")

    def start_slice(self):
        if not self.video_path:
            QMessageBox.warning(self, "提示", "请先上传视频")
            return
        selected = [
            index + 1
            for index, check in enumerate(self.audio_checks)
            if check.isChecked()
        ]
        if not selected:
            QMessageBox.warning(self, "提示", "请进行音轨勾选")
            return
        markers = self.grouped_markers if os.path.exists(self.grouped_markers) else self.filtered_markers
        if not markers or not os.path.exists(markers):
            QMessageBox.warning(self, "提示", "请先完成扫描与归并")
            return
        output = self.output_path_label.text()
        if output.startswith("默认输出: "):
            output = output[len("默认输出: "):]
        elif output.startswith("输出目录: "):
            output = output[len("输出目录: "):]
        if not output or output == "未选择输出目录":
            stem = os.path.splitext(os.path.basename(self.video_path))[0]
            output = os.path.join(os.path.dirname(self.video_path), stem)
        script = os.path.join(os.path.dirname(__file__), "red_slice.py")
        args = [
            sys.executable,
            script,
            self.video_path,
            "--markers",
            markers,
            "--output-dir",
            output,
            "--audio-tracks",
            ",".join(str(item) for item in selected),
        ]
        self._log(f"正在切片...保存到 {output}", "start")
        self._set_busy(True)
        self.slice_process = QProcess(self)
        self.slice_process.setProcessChannelMode(QProcess.SeparateChannels)
        self.slice_process.readyReadStandardError.connect(
            lambda: self._read_slice_process()
        )
        self.slice_process.finished.connect(self._on_slice_process_finished)
        self.slice_process.errorOccurred.connect(self._on_slice_process_error)
        self.slice_process.start(args[0], args[1:])

    def _run_script(self, args, on_done):
        def target(worker):
            result = subprocess.run(args, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(result.stderr or "执行失败")
            return result.stdout

        worker = Worker(target, self)
        worker.finished_ok.connect(on_done)
        worker.failed.connect(self._on_worker_error)
        self.current_worker = worker
        worker.start()

    def _read_slice_process(self):
        if self.slice_process is None:
            return
        raw = bytes(self.slice_process.readAllStandardError()).decode("utf-8", errors="replace")
        for line in raw.splitlines():
            match = re.search(r"Exporting segment (\d+)/(\d+)", line)
            if match:
                done = int(match.group(1))
                total = int(match.group(2))
                percent = int(done / total * 100)
                self._set_progress(percent)
                self._log(f"切片进度：{done}/{total}", "start")

    def _on_slice_process_finished(self, exit_code, _status):
        self._set_busy(False)
        if exit_code != 0:
            self._log("切片进程失败", "error")
            QMessageBox.critical(self, "切片失败", "切片进程异常退出")
            return
        self._set_progress(100)
        output = self.output_path_label.text()
        self._log(f"切片结束，保存在 {output}", "success")

    def _on_slice_process_error(self, error):
        self._set_busy(False)
        self._log(f"切片进程错误：{error}", "error")

    def _on_worker_error(self, error):
        self._set_busy(False)
        self._log(f"错误：{error}", "error")
        QMessageBox.critical(self, "错误", error)

    def _parse_hms(self, text):
        parts = text.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(text)

    def _format_seconds(self, seconds):
        total = int(seconds)
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _save_app_settings(self):
        data = {
            "theme": "dark" if self.is_dark else "day",
            "sliders": {
                "gray": self.gray_threshold,
                "sim": self.similarity_threshold,
                "before": self.before_time,
                "after": self.after_time,
            },
        }
        if hasattr(self, "roi_selector"):
            data["roi"] = [self.roi.x, self.roi.y, self.roi.width, self.roi.height]
        with open(self.app_settings_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

    def save_all_settings(self):
        self._save_app_settings()
        self._save_audio_config()
        QMessageBox.information(self, "已保存", "参数、ROI 与音轨设置已保存")

    def _load_app_settings(self):
        if not os.path.exists(self.app_settings_path):
            return
        try:
            with open(self.app_settings_path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return
        self.is_dark = data.get("theme", "dark") == "dark"
        self._apply_styles()
        sliders = data.get("sliders") or {}
        if "gray" in sliders:
            self.gray_threshold = int(sliders["gray"])
            self.gray_control[2].setValue(self.gray_threshold)
        if "sim" in sliders:
            self.similarity_threshold = int(sliders["sim"])
            self.sim_control[2].setValue(self.similarity_threshold)
        if "before" in sliders:
            self.before_time = int(sliders["before"])
            self.before_control[2].setValue(self.before_time)
        if "after" in sliders:
            self.after_time = int(sliders["after"])
            self.after_control[2].setValue(self.after_time)
        roi = data.get("roi")
        if roi and len(roi) == 4 and hasattr(self, "roi_x_spin"):
            x, y, w, h = roi
            self.roi_x_spin.setValue(max(0, min(1920, int(x))))
            self.roi_y_spin.setValue(max(0, min(1080, int(y))))
            self.roi_w_spin.setValue(max(1, min(1920, int(w))))
            self.roi_h_spin.setValue(max(1, min(1080, int(h))))
            self._apply_manual_roi()

    def save_notes(self):
        path = os.path.join(os.path.dirname(__file__), "optimization_notes.md")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.notes_edit.toPlainText())
        QMessageBox.information(self, "已保存", f"优化记录已保存到 {path}")

    def _load_notes(self):
        path = os.path.join(os.path.dirname(__file__), "optimization_notes.md")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as handle:
                self.notes_edit.setPlainText(handle.read())


def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
