import io
import os
import sys
import time
import zipfile
from collections import deque

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtGui import QAction

from . import APP_VERSION
from .fragmentation import (
    apply_mask_alpha,
    build_mask_workflow_stages,
    crop_to_primary_mask,
    ellipsis_middle,
    np2pil,
    numpy_alpha_composite,
    pil2np,
)
from .interference import apply_interference_masked as apply_interfere_masked
from .i18n import (
    LANG_EN,
    LANG_ZH_TW,
    UiLanguageFilter,
    current_language,
    set_language,
    tr,
)
from .psd_export import PsdExportDialog, PsdExportWorker
from .workers import DegradePreviewThread, InterfereGenThread, OverlapThread, SplitThread


BG_OPTIONS = [
    ('#222222', '深灰'),
    ('#FFFFFF', '白'),
    ('#888888', '50%灰'),
    ('check', '透明網格'),
]
BG_EN_LABELS = {
    "深灰": "Dark",
    "白": "W",
    "50%灰": "50%",
    "透明網格": "Grid",
}


def application_dir():
    """Return the editable application directory in source and frozen builds."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECYCLE_BIN_MAX = 99
FRAGMENT_LIST_ROW_HEIGHT = 30


def freeze_form_label_column(form_layout):
    """Keep language changes from resizing a form's label column."""
    labels = []
    for row in range(form_layout.rowCount()):
        item = form_layout.itemAt(row, QtWidgets.QFormLayout.LabelRole)
        widget = item.widget() if item is not None else None
        if isinstance(widget, QtWidgets.QLabel):
            labels.append(widget)
    if not labels:
        return
    width = max(label.sizeHint().width() for label in labels)
    for label in labels:
        label.setFixedWidth(width)


def pil2qpixmap(image):
    if isinstance(image, np.ndarray):
        array = image.astype(np.uint8)
        height, width = array.shape[:2]
        qimage = QtGui.QImage(
            array.tobytes(), width, height, QtGui.QImage.Format_RGBA8888
        )
        return QtGui.QPixmap.fromImage(qimage)
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    qimage = QtGui.QImage(
        image.tobytes("raw", "RGBA"),
        image.width,
        image.height,
        QtGui.QImage.Format_RGBA8888,
    )
    return QtGui.QPixmap.fromImage(qimage)


def np2qpixmap(array):
    return pil2qpixmap(array)


class QHelpButton(QtWidgets.QLabel):
    def __init__(self, text):
        super().__init__(" ? ")
        self.setFixedSize(24, 24)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("""
            background: #555; color:#fff; font-weight:bold; font-size:16px;
            border-radius:7px; border:1px solid #999; margin-left:3px; margin-right:3px;""")
        self.tip = text
        self.tipBox = None
    def showTip(self):
        if self.tipBox: self.tipBox.close()
        self.tipBox = QtWidgets.QLabel(tr(self.tip), None, QtCore.Qt.ToolTip)
        self.tipBox.setStyleSheet("""
            background: #222; color:#fff; border-radius:8px; border:1.5px solid #999;
            font-size:14px; padding:10px 18px; min-width:210px; max-width:350px;""")
        self.tipBox.setWordWrap(True)
        pos = self.mapToGlobal(self.rect().bottomRight())
        screen = QtGui.QGuiApplication.screenAt(pos)
        if not screen: screen = QtWidgets.QApplication.primaryScreen()
        scr_geo = screen.geometry()
        self.tipBox.adjustSize()
        w, h = self.tipBox.width(), self.tipBox.height()
        x = min(pos.x()+12, scr_geo.right()-w-16)
        y = min(pos.y()-12, scr_geo.bottom()-h-16)
        x = max(scr_geo.left()+8, x)
        y = max(scr_geo.top()+8, y)
        self.tipBox.move(x, y)
        self.tipBox.show()
    def enterEvent(self, e): self.showTip()
    def leaveEvent(self, e): self.hideTip()
    def mousePressEvent(self, e):
        if self.tipBox and self.tipBox.isVisible(): self.hideTip()
        else: self.showTip()
    def hideTip(self):
        if self.tipBox:
            self.tipBox.close()
            self.tipBox = None

class ClickableFileLabel(QtWidgets.QLabel):
    def __init__(self, parent, kind):
        super().__init__(parent)
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.kind = kind
        self.setProperty("i18n_skip", True)
        self.setStyleSheet("QLabel { color: #ffc; font-size:13px; min-height:18px; }")
        self.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
    def enterEvent(self, event):
        self.setStyleSheet("QLabel { color: #ffc; font-size:13px; min-height:18px; text-decoration: underline; }")
        super().enterEvent(event)
    def leaveEvent(self, event):
        self.setStyleSheet("QLabel { color: #ffc; font-size:13px; min-height:18px; }")
        super().leaveEvent(event)
    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            mw = self.parent()
            if self.kind == 'main' and mw.main_img is not None:
                mw.img_wrap.preview.set_image(mw.main_img)
                mw.set_status("主圖預覽", True)
            elif self.kind == 'mask' and mw.mask_img is not None:
                mw.preview_mask_grayalpha(mw.mask_img)
                mw.set_status("主體遮罩預覽", True)
            elif self.kind == 'secondary_mask' and mw.secondary_mask_img is not None:
                mw.preview_mask_grayalpha(mw.secondary_mask_img)
                mw.set_status("次要遮罩預覽", True)
        super().mouseReleaseEvent(event)

class ImagePreview(QtWidgets.QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._parent = parent
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setBackgroundRole(QtGui.QPalette.Base)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumWidth(360)

        self.img = None
        self.qimg = None
        self.bg_idx = 0

        self._scale = 1.0
        self._max_scale = 2.0
        self.offset = QtCore.QPoint(0, 0)   # 平移偏移（widget 座標）
        self._drag_start = None             # 左鍵拖曳用

        self.trash_highlight = False
        self.overlay_mode = False
        self.overlay_base = None

        # —— 選框（影像座標；可越界；持久）——
        self._sel_img_rect = None           # (x0, y0, x1, y1) floats
        self._sel_dragging = False          # 右鍵是否正在拉框

        self.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #222222;")

    # ===== 基本設定 =====
    def set_bg(self, idx):
        self.bg_idx = idx
        self.update()

    def set_image(self, img, trash_highlight=False):
        self.img = img
        self.trash_highlight = trash_highlight
        self.overlay_mode = False
        if img is not None:
            if isinstance(img, np.ndarray):
                self.qimg = np2qpixmap(img)
            else:
                self.qimg = pil2qpixmap(img)
        else:
            self.qimg = None
        self.update()

    def set_overlay(self, overlay_img):
        self.overlay_mode = True
        self.overlay_base = overlay_img
        if overlay_img is not None:
            if isinstance(overlay_img, np.ndarray):
                self.qimg = np2qpixmap(overlay_img)
            else:
                self.qimg = pil2qpixmap(overlay_img)
        else:
            self.qimg = None
        self.update()

    # ===== 座標轉換 =====
    def _image_widget_rect(self) -> QtCore.QRectF:
        """當前縮放與平移下，影像在 widget 上的矩形（widget 座標）。"""
        if not self.qimg:
            return QtCore.QRectF()
        scale = self._scale
        img_w = float(self.qimg.width())  * scale
        img_h = float(self.qimg.height()) * scale
        ox = float((self.width()  - img_w) / 2.0 + self.offset.x())
        oy = float((self.height() - img_h) / 2.0 + self.offset.y())
        return QtCore.QRectF(ox, oy, img_w, img_h)

    def widget_to_img_unbounded(self, p: QtCore.QPointF):
        """widget -> image（不夾限，可為負或超過大小）。"""
        if not self.qimg:
            return None
        r = self._image_widget_rect()
        scale = self._scale
        x = (p.x() - r.left()) / scale
        y = (p.y() - r.top())  / scale
        return QtCore.QPointF(float(x), float(y))

    def img_to_widget(self, p: QtCore.QPointF):
        """image -> widget（不夾限）。"""
        if not self.qimg:
            return None
        r = self._image_widget_rect()
        scale = self._scale
        x = r.left() + p.x() * scale
        y = r.top()  + p.y() * scale
        return QtCore.QPointF(float(x), float(y))

    # ===== 選框 API =====
    def clear_selection(self):
        self._sel_img_rect = None
        self._sel_dragging = False
        self.update()

    def get_selection_rect_img(self):
        """回傳與影像邊界相交後的 (x,y,w,h)（整數像素）；完全無交集則回 None。"""
        if not (self.qimg and self._sel_img_rect):
            return None
        x0, y0, x1, y1 = self._sel_img_rect
        if x0 > x1: x0, x1 = x1, x0
        if y0 > y1: y0, y1 = y1, y0

        W = self.qimg.width()
        H = self.qimg.height()

        # 與影像做交集
        ix0 = max(0, min(int(x0), W - 1))
        iy0 = max(0, min(int(y0), H - 1))
        ix1 = max(0, min(int(x1), W - 1))
        iy1 = max(0, min(int(y1), H - 1))

        w = ix1 - ix0 + 1
        h = iy1 - iy0 + 1
        return (ix0, iy0, w, h) if (w > 0 and h > 0) else None

    # ===== 繪製 =====
    def paintEvent(self, ev):
        super().paintEvent(ev)
        painter = QtGui.QPainter(self)
        try:
            # 背景
            bg_key = BG_OPTIONS[self.bg_idx][0]
            if bg_key == "check":
                self._draw_checkerboard(painter)
            else:
                painter.fillRect(self.rect(), QtGui.QColor(bg_key))

            # 主圖
            if self.qimg is not None:
                scale = min(self._scale, self._max_scale)
                tgt_w = int(self.qimg.width()  * scale)
                tgt_h = int(self.qimg.height() * scale)
                pt = QtCore.QPoint(
                    (self.width()  - tgt_w) // 2 + self.offset.x(),
                    (self.height() - tgt_h) // 2 + self.offset.y()
                )

                if isinstance(self.qimg, QtGui.QPixmap):
                    scaled = self.qimg.scaled(tgt_w, tgt_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
                    painter.drawPixmap(pt, scaled)
                else:
                    scaled = self.qimg.scaled(tgt_w, tgt_h, QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
                    painter.drawImage(pt, scaled)

                # 垃圾桶高亮框（安全防護）
                try:
                    parent_widget = self._parent.parent()
                    is_trash_tab = (
                        hasattr(parent_widget, "tabs")
                        and parent_widget.tabs.currentWidget() == parent_widget.trash_tab
                    )
                except Exception:
                    is_trash_tab = False

                if is_trash_tab and getattr(self, "trash_highlight", False):
                    pen = QtGui.QPen(QtGui.QColor(255, 0, 0), 3)
                    painter.setPen(pen)
                    painter.drawRect(self.rect().adjusted(2, 2, -4, -4))
            else:
                # 無主圖置中提示
                font = painter.font()
                font.setPointSize(22)
                font.setBold(True)
                painter.setFont(font)
                if isinstance(bg_key, str) and (bg_key == "check" or bg_key.lower() in ("#ffffff", "white")):
                    pen_color = QtGui.QColor(0, 0, 0)
                else:
                    pen_color = QtGui.QColor(255, 255, 255)
                painter.setPen(QtGui.QPen(pen_color))
                painter.drawText(
                    self.rect(), QtCore.Qt.AlignCenter, tr("當前無預覽圖")
                )

            # 選框（影像座標 -> widget 座標，會隨縮放/平移正確對應）
            if self._sel_img_rect and self.qimg is not None:
                x0, y0, x1, y1 = self._sel_img_rect
                p0 = self.img_to_widget(QtCore.QPointF(x0, y0))
                p1 = self.img_to_widget(QtCore.QPointF(x1, y1))
                if p0 and p1:
                    rect = QtCore.QRectF(p0, p1).normalized().toRect()
                    pen = QtGui.QPen(QtGui.QColor(0, 200, 255), 2, QtCore.Qt.DashLine)
                    painter.setPen(pen)
                    painter.drawRect(rect.adjusted(0, 0, -1, -1))
                    painter.fillRect(rect, QtGui.QColor(0, 200, 255, 40))
        finally:
            painter.end()

    # 透明棋盤
    def _draw_checkerboard(self, painter):
        tile = 16
        cols = self.width() // tile + 2
        rows = self.height() // tile + 2
        for y in range(rows):
            for x in range(cols):
                color = QtGui.QColor(220, 220, 220) if (x + y) % 2 == 0 else QtGui.QColor(160, 160, 160)
                painter.fillRect(x * tile, y * tile, tile, tile, color)

    # 縮放（以滑鼠位置為中心），選框會自然跟著走（因為選框存在影像座標）
    def wheelEvent(self, ev):
        if not self.qimg:
            return
        old_scale = self._scale
        pos = ev.position() if hasattr(ev, "position") else QtCore.QPointF(ev.pos())
        img_rect = self._image_widget_rect()

        # 滑鼠對應的影像座標
        img_px = (pos.x() - img_rect.left()) / old_scale
        img_py = (pos.y() - img_rect.top())  / old_scale

        delta = ev.angleDelta().y() / 120.0
        factor = 1.15 ** delta
        self._scale = max(0.05, min(self._max_scale, self._scale * factor))

        # 讓滑鼠所在像素在縮放後仍停在滑鼠處
        new_img_rect = self._image_widget_rect()
        new_left = pos.x() - img_px * self._scale
        new_top  = pos.y() - img_py * self._scale
        self.offset = QtCore.QPoint(
            int(new_left - (self.width()  - new_img_rect.width())  / 2.0),
            int(new_top  - (self.height() - new_img_rect.height()) / 2.0)
        )
        self.update()
        if self._parent:
            self._parent.update_zoom_display()

    # 拖曳平移（左鍵）
    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self._drag_start = (ev.pos(), QtCore.QPoint(self.offset))
            self.setCursor(QtCore.Qt.ClosedHandCursor)
        elif ev.button() == QtCore.Qt.RightButton:
            # 任意處開始框選：轉成影像座標（不夾限）
            self._sel_dragging = True
            p_img = self.widget_to_img_unbounded(ev.position() if hasattr(ev, "position") else QtCore.QPointF(ev.pos()))
            if p_img is not None:
                self._sel_img_rect = (p_img.x(), p_img.y(), p_img.x(), p_img.y())
            self.grabMouse()
            self.update()

    def mouseMoveEvent(self, ev):
        if self._drag_start:
            delta = ev.pos() - self._drag_start[0]
            self.offset = self._drag_start[1] + delta
            self.update()
        elif self._sel_dragging:
            p_img = self.widget_to_img_unbounded(ev.position() if hasattr(ev, "position") else QtCore.QPointF(ev.pos()))
            if p_img is not None and self._sel_img_rect:
                x0, y0, _, _ = self._sel_img_rect
                self._sel_img_rect = (x0, y0, p_img.x(), p_img.y())
                self.update()

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self._drag_start = None
            self.setCursor(QtCore.Qt.ArrowCursor)
        elif ev.button() == QtCore.Qt.RightButton:
            try:
                self.releaseMouse()
            except Exception:
                pass
            # 單點不清除選框；如要單點清除，可取消下一行註解
            # if self._sel_img_rect and abs(self._sel_img_rect[0]-self._sel_img_rect[2])<2 and abs(self._sel_img_rect[1]-self._sel_img_rect[3])<2:
            #     self.clear_selection()
            self._sel_dragging = False
            self.update()

    def mouseDoubleClickEvent(self, ev):
        # 雙擊復位（保留選框）
        self._scale = 1.0
        self.offset = QtCore.QPoint(0, 0)
        self.update()
        if self._parent:
            self._parent.update_zoom_display()

class ImagePreviewWrap(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.preview = ImagePreview(self)
        self.zoom_lbl = QtWidgets.QLabel("100%")
        self.zoom_lbl.setStyleSheet("font-size:16px;font-weight:bold;min-width:55px;max-width:65px;")
        self.zoom_lbl.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignVCenter)
        self.zoom_down = QtWidgets.QPushButton("-")
        self.zoom_down.setFixedWidth(38)
        self.zoom_up = QtWidgets.QPushButton("+")
        self.zoom_up.setFixedWidth(38)
        self.zoom_down.clicked.connect(self.zoom_minus)
        self.zoom_up.clicked.connect(self.zoom_plus)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.preview, stretch=1)
        hz = QtWidgets.QHBoxLayout()
        hz.addWidget(self.zoom_down)
        hz.addWidget(self.zoom_lbl)
        hz.addWidget(self.zoom_up)
        self.language_btn = QtWidgets.QPushButton("中 / EN")
        self.language_btn.setFixedWidth(64)
        self.language_btn.setToolTip("切換繁體中文／English；不改變目前介面尺寸")
        self.language_btn.clicked.connect(parent.toggle_language)
        hz.addWidget(self.language_btn)
        self.bg_combo = QtWidgets.QComboBox()
        for index, (_color, name) in enumerate(BG_OPTIONS):
            self.bg_combo.addItem(name, index)
        metrics = self.bg_combo.fontMetrics()
        longest_text = max(
            [metrics.horizontalAdvance(name) for _color, name in BG_OPTIONS]
            + [metrics.horizontalAdvance(name) for name in BG_EN_LABELS.values()]
        )
        self.bg_combo.setFixedWidth(longest_text + 52)
        self.bg_combo.setToolTip("預覽背景類型。僅影響預覽，不影響輸出。")
        hz.addWidget(self.bg_combo)
        self.overlay_btn = QtWidgets.QPushButton("重疊預覽")
        self.overlay_btn.setCheckable(True)
        self.overlay_btn.setStyleSheet("background:#444; color:#fff; padding:2px 12px;")
        self.overlay_btn.setVisible(False)
        self.overlay_btn.clicked.connect(self.toggle_overlay)
        hz.addWidget(self.overlay_btn, alignment=QtCore.Qt.AlignLeft)
        self.bg_combo.currentIndexChanged.connect(self.change_background)
        hz.addStretch()
        self.status_lbl = StatusLabel("")
        self.status_lbl.setStyleSheet("color:#0f0;font-size:15px;min-width:140px;max-width:260px;")
        hz.addWidget(self.status_lbl)
        lay.addLayout(hz)
        self.preview._parent = self
        self.update_zoom_display()
        self.previewing_fragment_name = None
    def change_background(self, index):
        self.preview.set_bg(index)
        if self.overlay_btn.isChecked():
            self.overlay_btn.setChecked(False)
            if hasattr(self.parent(), "restore_overlay_off"):
                self.parent().restore_overlay_off()
    def zoom_minus(self):
        self.preview._scale = max(0.05, self.preview._scale-0.05)
        self.preview.repaint()
        self.update_zoom_display()
    def zoom_plus(self):
        self.preview._scale = min(2.0, self.preview._scale+0.05)
        self.preview.repaint()
        self.update_zoom_display()
    def update_zoom_display(self):
        val = int(self.preview._scale*100)
        self.zoom_lbl.setText(f"{val}%")
    def toggle_overlay(self, checked):
        if checked:
            if hasattr(self.parent(), "generate_overlay_preview"):
                overlay_img = self.parent().generate_overlay_preview()
                if overlay_img is not None:
                    self.preview.set_overlay(overlay_img)
                    self.parent().overlay_active = True
                else:
                    self.overlay_btn.setChecked(False)
                    QtWidgets.QMessageBox.warning(self, "錯誤", "請先載入主圖，並有碎片可預覽")
        else:
            if hasattr(self.parent(), "restore_overlay_off"):
                self.parent().restore_overlay_off()

class InterferePanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QFormLayout(self)

        self.block_size = QtWidgets.QSpinBox()
        self.block_size.setRange(1, 30)
        self.block_size.setValue(1)
        self.block_size.setFixedWidth(70)
        bs_tip = QHelpButton(
            "設定每一個干擾像素塊的基本邊長(px)，越大則每塊越大。\n\n"
            "優點：大尺寸提升覆蓋速度。\n\n"
            "缺點：塊太大時，干擾效果會不自然且容易被辨識。"
        )
        bs_tip.setFixedHeight(24)
        block_row = QtWidgets.QHBoxLayout()
        block_row.addWidget(self.block_size)
        block_row.addWidget(bs_tip)
        block_row.addStretch()
        lay.addRow("干擾像素尺寸(1~30)：", block_row)

        self.random_range = QtWidgets.QSpinBox()
        self.random_range.setRange(1, 100)
        self.random_range.setValue(6)
        self.random_range.setFixedWidth(70)
        rr_tip = QHelpButton(
            "決定干擾像素塊的尺寸隨機變動範圍，1為固定，數字越大越亂。\n\n"
            "優點：隨機性高提升防還原性。\n\n"
            "缺點：數值過大會產生極端尺寸、不均勻塊。"
        )
        rr_tip.setFixedHeight(24)
        rand_row = QtWidgets.QHBoxLayout()
        rand_row.addWidget(self.random_range)
        rand_row.addWidget(rr_tip)
        rand_row.addStretch()
        lay.addRow("尺寸隨機度(1~100)：", rand_row)

        self.density = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.density.setRange(1, 100)
        self.density.setValue(5)
        self.density_lbl = QtWidgets.QLabel("干擾密度：5%")
        density_tip = QHelpButton(
            "決定干擾像素填滿目標區域的比例，數字越高，干擾覆蓋越密集，100%不代表全填滿，實際上會受到區塊尺寸影響。\n\n"
            "優點：密度高可大幅阻礙還原。\n\n"
            "缺點：太高會讓檔案龐大且難以正常辨識。"
        )
        density_row = QtWidgets.QHBoxLayout()
        density_row.addWidget(self.density_lbl)
        density_row.addWidget(density_tip)
        density_row.addStretch()
        lay.addRow(density_row)
        lay.addRow(self.density)
        self.density.valueChanged.connect(lambda v: self.density_lbl.setText(f"干擾密度：{v}%"))

        self.alpha_min = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha_min.setRange(1, 100)
        self.alpha_min.setValue(1)
        self.alpha_min_lbl = QtWidgets.QLabel(f"取樣不透明度下限：{self.alpha_min.value()}%")
        amn_tip = QHelpButton(
            "設定可被選入干擾素材池的像素塊，必須覆蓋的最小不透明比例，避免選到太透明的雜訊。\n\n"
            "優點：濾除雜訊，保證干擾有效。\n\n"
            "缺點：設定過高會排除大部分素材，干擾池不足。"
        )
        amn_row = QtWidgets.QHBoxLayout()
        amn_row.addWidget(self.alpha_min_lbl)
        amn_row.addWidget(amn_tip)
        amn_row.addStretch()
        lay.addRow(amn_row)
        lay.addRow(self.alpha_min)
        self.alpha_min.valueChanged.connect(
            lambda v: self.alpha_min_lbl.setText(f"取樣不透明度下限：{v}%"))

        self.alpha_max = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha_max.setRange(1, 100)
        self.alpha_max.setValue(100)
        self.alpha_max_lbl = QtWidgets.QLabel("取樣不透明度上限：100%")
        amx_tip = QHelpButton(
            "設定可被選入干擾素材池的像素塊，必須覆蓋的最大不透明比例。可用來排除太實心的大片塊。\n\n"
            "優點：排除過大塊避免影響外觀。\n\n"
            "缺點：過小則素材有限，干擾效果變差。"
        )
        amx_row = QtWidgets.QHBoxLayout()
        amx_row.addWidget(self.alpha_max_lbl)
        amx_row.addWidget(amx_tip)
        amx_row.addStretch()
        lay.addRow(amx_row)
        lay.addRow(self.alpha_max)
        self.alpha_max.valueChanged.connect(
            lambda v: self.alpha_max_lbl.setText(f"取樣不透明度上限：{v}%"))

        self.allow_overlap_cb = QtWidgets.QCheckBox("允許干擾像素重疊")
        self.allow_overlap_cb.setChecked(True)
        overlap_tip = QHelpButton(
            "允許多個干擾像素塊彼此重疊。若關閉，干擾像素會盡量不交錯，但可能會減少填充面積。\n\n"
            "優點：允許重疊可提升覆蓋效率與密度。\n\n"
            "缺點：重疊過多時，部分區塊可能異常突出。"
        )
        overlap_row = QtWidgets.QHBoxLayout()
        overlap_row.addWidget(self.allow_overlap_cb)
        overlap_row.addWidget(overlap_tip)
        overlap_row.addStretch()
        lay.addRow(overlap_row)

        self.random_scope_cb = QtWidgets.QCheckBox("第 3 片起隨機使用前方碎片範圍")
        self.random_scope_cb.setChecked(True)
        progressive_tip = QHelpButton(
            "干擾像素取自目前主圖；若劣化處理頁已掛載劣化取樣圖，則優先取自該劣化圖。"
            "清單第一張不加入干擾，也不會被拿來當生成範圍；第二張沒有前方可用範圍，保持原樣。"
            "從清單第三張開始，只會在第二張到目前碎片前一張之間隨機選擇一片或多片，"
            "將其 alpha 聯集作為干擾生成範圍。"
        )
        progressive_row = QtWidgets.QHBoxLayout()
        progressive_row.addWidget(self.random_scope_cb)
        progressive_row.addWidget(progressive_tip)
        progressive_row.addStretch()
        lay.addRow(progressive_row)

        self.ignore_semitrans_cb = QtWidgets.QCheckBox("忽略半透明區域")
        self.ignore_semitrans_cb.setChecked(True)
        ignore_tip = QHelpButton(
            "勾選後，只把所選前方碎片完全不透明的區域加入生成範圍。\n"

            "取消勾選時，半透明像素也會加入所選碎片的聯集範圍。\n"

            "建議開啟，能避免在主圖透明邊緣產生髒點"
        )
        ignore_row = QtWidgets.QHBoxLayout()
        ignore_row.addWidget(self.ignore_semitrans_cb)
        ignore_row.addWidget(ignore_tip)
        ignore_row.addStretch()
        lay.addRow(ignore_row)


        automatic_note = QtWidgets.QLabel(
            "以上參數會在「執行拆解」與局部分割時自動套用；"
            "清單前兩張保持原樣，干擾由清單第三張開始。"
        )
        automatic_note.setWordWrap(True)
        automatic_note.setStyleSheet(
            "color:#8fd3ff; padding:8px; background:#1d303b; border-radius:4px;"
        )
        lay.addRow(automatic_note)

        # 保留屬性供舊工作階段安全關閉，但不再提供手動產生／合成按鈕。
        self.gen_btn = QtWidgets.QPushButton(self)
        self.apply_btn = QtWidgets.QPushButton(self)
        self.gen_btn.hide()
        self.apply_btn.hide()
        freeze_form_label_column(lay)

    def get_settings(self):
        minv = min(self.alpha_min.value(), self.alpha_max.value())
        maxv = max(self.alpha_min.value(), self.alpha_max.value())
        return dict(
            block_size=self.block_size.value(),
            random_range=self.random_range.value(),
            density=self.density.value() / 100,
            alpha_min=minv,
            alpha_max=maxv,
            allow_overlap=self.allow_overlap_cb.isChecked(),
            random_previous_scopes=self.random_scope_cb.isChecked(),
            ignore_semitrans=self.ignore_semitrans_cb.isChecked()
        )

class DegradePanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QFormLayout(self)

        # --- 上方：匯入來源圖 + tip ---
        top_row = QtWidgets.QHBoxLayout()
        self.import_source_btn = QtWidgets.QPushButton("匯入來源圖")
        self.import_source_btn.setFixedHeight(28)
        self.import_source_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        top_row.addWidget(self.import_source_btn)
        import_tip = QHelpButton(
            "此區用來製作干擾像素的劣化取樣圖。\n\n"
            "先匯入來源圖並調整下方劣化參數，再按「產生劣化預覽」；確認效果後按「掛載干擾像素」，"
            "全圖拆解與局部分割便會自動改用這張劣化圖取樣。\n\n"
            "「還原原圖」只把左側預覽切回匯入的原始來源，不會解除已掛載的干擾像素；"
            "載入新主圖、匯入新來源或重新產生劣化預覽時，掛載狀態會自動重設。"
        )
        top_row.addWidget(import_tip)
        lay.addRow(top_row)

        # 檔名顯示
        self.imported_filename_lbl = QtWidgets.QLabel("尚未載入任何圖")
        self.imported_filename_lbl.setStyleSheet("color:#ccc; font-size:12px;")
        self.imported_filename_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addRow("目前來源：", self.imported_filename_lbl)

        # --- 各種參數滑桿 ---
        # 方塊尺寸
        self.block_size = QtWidgets.QSpinBox()
        self.block_size.setRange(1, 60)
        self.block_size.setValue(5)
        bs_tip = QHelpButton("劣化方塊的基本尺寸（px），整張圖會以變動大小的方塊切割後個別劣化。")
        h1 = QtWidgets.QHBoxLayout()
        h1.addWidget(self.block_size)
        h1.addWidget(bs_tip)
        h1.addStretch()
        lay.addRow("方塊尺寸：", h1)

        # 尺寸隨機度
        self.rand_range = QtWidgets.QSpinBox()
        self.rand_range.setRange(1, 10)
        self.rand_range.setValue(2)
        rr_tip = QHelpButton("劣化方塊尺寸的隨機倍率範圍，1 代表所有區塊尺寸固定，2 代表區塊尺寸會隨機在設定值的 1~2 倍間變化。")
        h2 = QtWidgets.QHBoxLayout()
        h2.addWidget(self.rand_range)
        h2.addWidget(rr_tip)
        h2.addStretch()
        lay.addRow("尺寸隨機度：", h2)

        # 劣化密度
        self.density_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.density_slider.setRange(1, 100)
        self.density_slider.setValue(70)
        self.density_label = QtWidgets.QLabel("劣化密度：70%")
        self.density_slider.valueChanged.connect(lambda v: self.density_label.setText(f"劣化密度：{v}%"))
        density_tip = QHelpButton("控制整張圖中要放多少塊進行劣化（影響劣化區塊數量）。")
        h_density = QtWidgets.QHBoxLayout()
        h_density.addWidget(self.density_label)
        h_density.addWidget(density_tip)
        h_density.addStretch()
        lay.addRow(h_density)
        lay.addRow(self.density_slider)

        # 噪點強度
        self.noise_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.noise_slider.setRange(0, 100)
        self.noise_slider.setValue(10)
        self.noise_label = QtWidgets.QLabel("噪點強度：10%")
        self.noise_slider.valueChanged.connect(lambda v: self.noise_label.setText(f"噪點強度：{v}%"))
        noise_tip = QHelpButton("每個方塊中加入的隨機雜訊強度。")
        h3 = QtWidgets.QHBoxLayout()
        h3.addWidget(self.noise_label)
        h3.addWidget(noise_tip)
        h3.addStretch()
        lay.addRow(h3)
        lay.addRow(self.noise_slider)

        # 隨機明暗
        self.bright_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.bright_slider.setRange(0, 100)
        self.bright_slider.setValue(10)
        self.bright_label = QtWidgets.QLabel("隨機明暗：10%")
        self.bright_slider.valueChanged.connect(lambda v: self.bright_label.setText(f"隨機明暗：{v}%"))
        bright_tip = QHelpButton("每個方塊會有明暗偏移。")
        h4 = QtWidgets.QHBoxLayout()
        h4.addWidget(self.bright_label)
        h4.addWidget(bright_tip)
        h4.addStretch()
        lay.addRow(h4)
        lay.addRow(self.bright_slider)

        # 色偏
        self.color_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.color_slider.setRange(0, 100)
        self.color_slider.setValue(10)
        self.color_label = QtWidgets.QLabel("色偏強度：10%")
        self.color_slider.valueChanged.connect(lambda v: self.color_label.setText(f"色偏強度：{v}%"))
        color_tip = QHelpButton("每個方塊加入隨機 RGB 色偏。")
        h5 = QtWidgets.QHBoxLayout()
        h5.addWidget(self.color_label)
        h5.addWidget(color_tip)
        h5.addStretch()
        lay.addRow(h5)
        lay.addRow(self.color_slider)

        # --- 最下方：三顆按鈕一排（等寬） ---
        btn_row = QtWidgets.QHBoxLayout()
        self.gen_preview_btn = QtWidgets.QPushButton("產生劣化預覽")
        self.restore_source_btn = QtWidgets.QPushButton("還原原圖")
        self.apply_export_btn = QtWidgets.QPushButton("掛載干擾像素")
        self.apply_export_btn.setEnabled(False)
        for btn in (self.gen_preview_btn, self.restore_source_btn, self.apply_export_btn):
            btn.setFixedHeight(28)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.gen_preview_btn.setFixedWidth(120)
        self.restore_source_btn.setFixedWidth(100)
        self.apply_export_btn.setFixedWidth(150)
        btn_row.addWidget(self.gen_preview_btn)
        btn_row.addWidget(self.restore_source_btn)
        btn_row.addWidget(self.apply_export_btn)
        lay.addRow(btn_row)
        freeze_form_label_column(lay)

    def get_settings(self):
        return {
            "block_size": self.block_size.value(),
            "rand_range": self.rand_range.value(),
            "density": self.density_slider.value() / 100.0,
            "noise_strength": self.noise_slider.value(),
            "brightness_strength": self.bright_slider.value(),
            "color_strength": self.color_slider.value(),
        }

    def set_imported_filename(self, path):
        if not path:
            self.imported_filename_lbl.setProperty("i18n_skip", False)
            self.imported_filename_lbl.setText("尚未載入任何圖")
            self.imported_filename_lbl.setToolTip("")
            return
        base = os.path.basename(path)
        self.imported_filename_lbl.setProperty("i18n_skip", True)
        fm = self.imported_filename_lbl.fontMetrics()
        max_width = self.imported_filename_lbl.width() if self.imported_filename_lbl.width() > 0 else 220
        elided = fm.elidedText(base, QtCore.Qt.ElideMiddle, max_width)
        self.imported_filename_lbl.setText(elided)
        self.imported_filename_lbl.setToolTip(path)

class StatusLabel(QtWidgets.QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.setToolTip("")

    def enterEvent(self, event):
        if self._isTextElided():
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), self.text(), self)
        else:
            QtWidgets.QToolTip.hideText()
        super().enterEvent(event)

    def leaveEvent(self, event):
        QtWidgets.QToolTip.hideText()
        super().leaveEvent(event)

    def _isTextElided(self):
        metrics = self.fontMetrics()
        rect = self.contentsRect()
        return metrics.horizontalAdvance(self.text()) > rect.width()

class TrashCanWidget(QtWidgets.QWidget):
    def __init__(self, parent, recycle_bin):
        super().__init__(parent)
        self.parent = parent
        self.recycle_bin = recycle_bin
        self.current_highlight_img = None
        self.initUI()
    def initUI(self):
        vbox = QtWidgets.QVBoxLayout(self)
        self.info_lbl = QtWidgets.QLabel("垃圾桶 (可復原, 最多99項)")
        vbox.addWidget(self.info_lbl)
        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.list.setStyleSheet("""
            QListWidget::item { padding: 3px; }
            QListWidget::item:hover { background: #343434; }
            QListWidget::item:selected { background: #245a82; color: white; }
        """)
        vbox.addWidget(self.list, stretch=1)
        btnrow = QtWidgets.QHBoxLayout()
        self.restore_btn = QtWidgets.QPushButton("復原選擇碎片")
        self.restore_btn.clicked.connect(self.restore_selected)
        self.clear_btn = QtWidgets.QPushButton("清空垃圾桶")
        self.clear_btn.clicked.connect(self.clear_trash)
        btnrow.addWidget(self.restore_btn)
        btnrow.addWidget(self.clear_btn)
        vbox.addLayout(btnrow)
        self.list.itemSelectionChanged.connect(self.update_preview)
        self.list.customContextMenuRequested.connect(self.show_context_menu)
        self.refresh()

    def refresh(self):
        self.info_lbl.setText(f"垃圾碎片 {len(self.recycle_bin)}個")
        with QtCore.QSignalBlocker(self.list):
            self.list.clear()
            for name, img in self.recycle_bin:
                lw = QtWidgets.QListWidgetItem(name)
                lw.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                lw.setSizeHint(QtCore.QSize(0, FRAGMENT_LIST_ROW_HEIGHT))
                self.list.addItem(lw)

        # ==== 新增：自動選中最後一個並預覽 ====
        count = self.list.count()
        if count > 0:
            self.list.setCurrentRow(count - 1)
            item = self.list.item(count - 1)
            if item:
                # 保證 UI 顯示到底部
                row = self.list.row(item)
                QtCore.QTimer.singleShot(0, lambda r=row: self._scroll_to_row(r))
        else:
            # 沒有碎片就清除預覽
            self.clear_highlight()
        self.update_preview()

    def selected_rows(self):
        return sorted(self.list.row(item) for item in self.list.selectedItems())

    @staticmethod
    def _as_rgba_image(img):
        if isinstance(img, Image.Image):
            return img.convert("RGBA")
        array = np.asarray(img).astype(np.uint8)
        return Image.fromarray(array).convert("RGBA")

    def update_preview(self):
        # 垃圾桶在其他操作中也會被刷新；只有頁籤正在顯示時才接管左側預覽。
        if (
            hasattr(self.parent, "tabs")
            and self.parent.tabs.currentWidget() is not self
        ):
            return

        rows = [row for row in self.selected_rows() if 0 <= row < len(self.recycle_bin)]
        if not rows:
            self.clear_highlight()
            return

        selected_images = [self._as_rgba_image(self.recycle_bin[row][1]) for row in rows]
        if len(selected_images) == 1:
            preview_img = selected_images[0]
        else:
            width = max(img.width for img in selected_images)
            height = max(img.height for img in selected_images)
            preview_img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            # 與進階管理一致：清單越上方的碎片，預覽時位於越上層。
            for img in reversed(selected_images):
                layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
                layer.alpha_composite(img, (0, 0))
                preview_img = Image.alpha_composite(preview_img, layer)

        self.current_highlight_img = preview_img
        self.parent.img_wrap.preview.set_image(preview_img, trash_highlight=True)
        self.parent.set_status(f"垃圾桶預覽：已選擇 {len(rows)} 個碎片", True)

    def show_context_menu(self, pos):
        item = self.list.itemAt(pos)
        if item is not None and not item.isSelected():
            self.list.clearSelection()
            item.setSelected(True)
            self.list.setCurrentItem(item, QtCore.QItemSelectionModel.NoUpdate)

        menu = QtWidgets.QMenu(self.list)
        restore_action = menu.addAction("復原選擇碎片")
        restore_action.setEnabled(bool(self.list.selectedItems()))
        restore_action.triggered.connect(self.restore_selected)
        menu.exec(self.list.viewport().mapToGlobal(pos))

    def clear_highlight(self):
        self.current_highlight_img = None
        self.parent.img_wrap.preview.set_image(None, trash_highlight=False)

    def restore_selected(self):
        selected_rows = self.selected_rows()
        if not selected_rows:
            return
        items_to_restore = [
            (idx, self.recycle_bin[idx])
            for idx in selected_rows
            if 0 <= idx < len(self.recycle_bin)
        ]
        for idx, (name, img) in sorted(items_to_restore, key=lambda x: -x[0]):
            self.parent.restore_from_trash(name, img)
            del self.recycle_bin[idx]
        self.parent.refresh_fragment_list()
        self.refresh()

    def clear_trash(self):
        self.recycle_bin.clear()
        self.refresh()
        self.clear_highlight()

    def _scroll_to_row(self, r):
        it = self.list.item(r)
        if it is not None:
            self.list.scrollToItem(it, QtWidgets.QAbstractItemView.PositionAtBottom)


class FragmentListWidget(QtWidgets.QListWidget):
    """一般模式單選；進階模式的眼睛區域獨立控制顯示。"""

    visibilityToggleRequested = QtCore.Signal(object)
    VISIBILITY_ROLE = QtCore.Qt.UserRole + 10
    ROW_HEIGHT = FRAGMENT_LIST_ROW_HEIGHT

    def __init__(self, parent=None):
        super().__init__(parent)
        self.advanced_mode = False
        self.setIconSize(QtCore.QSize(22, 22))
        self._visible_icon = self._make_eye_icon(True)
        self._hidden_icon = self._make_eye_icon(False)

    @staticmethod
    def _make_eye_icon(visible):
        pixmap = QtGui.QPixmap(22, 22)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        color = QtGui.QColor("#8fd3ff" if visible else "#777777")
        painter.setPen(QtGui.QPen(color, 1.8))
        painter.drawEllipse(QtCore.QRectF(3, 7, 16, 9))
        if visible:
            painter.setBrush(color)
            painter.drawEllipse(QtCore.QRectF(9, 10, 4, 4))
        else:
            painter.drawLine(4, 17, 18, 5)
        painter.end()
        return QtGui.QIcon(pixmap)

    def set_advanced_mode(self, enabled):
        self.advanced_mode = bool(enabled)
        self.setSelectionMode(
            QtWidgets.QAbstractItemView.MultiSelection
            if self.advanced_mode
            else QtWidgets.QAbstractItemView.SingleSelection
        )

    def set_item_visible(self, item, visible):
        item.setData(self.VISIBILITY_ROLE, bool(visible))
        item.setIcon(self._visible_icon if visible else self._hidden_icon)
        item.setToolTip(
            tr("點擊眼睛隱藏此碎片" if visible else "點擊眼睛顯示此碎片")
        )

    def mousePressEvent(self, event):
        point = event.position().toPoint()
        item = self.itemAt(point)
        if (
            self.advanced_mode
            and item is not None
            and event.button() == QtCore.Qt.LeftButton
            and point.x() <= self.visualItemRect(item).left() + 32
        ):
            self.visibilityToggleRequested.emit(item)
            event.accept()
            return
        super().mousePressEvent(event)

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        saved_language = QtCore.QSettings(
            "DuoDuo", "MSW Skin Fragmenter Pro"
        ).value("language", LANG_ZH_TW)
        set_language(saved_language)
        self._language_filter = UiLanguageFilter(self)
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self._language_filter)
        self.setWindowTitle(f"MSW造型防盜拆解工具 專業版 MSW Skin Fragmenter Pro v{APP_VERSION}")
        self.setMinimumSize(1280,800)
        self.setStyleSheet("""
            QWidget {
                background: #232323;
                color: #eeeeee;
                font-size: 15px;
            }
            QMenu {
                background: #2b2b2b;
                color: #eeeeee;
                border: 1px solid #5a5a5a;
                padding: 4px;
            }
            QMenu::item {
                background: transparent;
                padding: 6px 30px 6px 12px;
                border-radius: 3px;
            }
            QMenu::item:selected:enabled {
                background: #287fbd;
                color: #ffffff;
            }
            QMenu::item:disabled {
                color: #777777;
            }
            QMenu::separator {
                height: 1px;
                background: #555555;
                margin: 4px 8px;
            }
        """)
        self.img_wrap = ImagePreviewWrap(self)
        self._degrade_warning_dialog = None
        self._last_degrade_warning = 0.0  # 用來簡單 debounce（秒）

        # 圖片 / 碎片 / 狀態
        self.main_img = None
        self.main_img_path = ""
        self.mask_img = None
        self.mask_img_path = ""
        self.secondary_mask_img = None
        self.secondary_mask_img_path = ""
        self.split_result = []
        self.fragment_data = {}
        self.fragment_order = []
        self.fragment_visibility = {}
        self.restore_mode = False
        self.recycle_bin = deque(maxlen=RECYCLE_BIN_MAX)
        self._initial_snapshot = None
        self.interfere_images_dict = {}

        # 新增：劣化來源與 preview 暫存（共用 preview）
        self.degrade_source_img = None
        self.degrade_preview_pending = None
        self.interference_source_img = None
        self.initUI()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.fragment_list.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)

        # 干擾像素
        self.interfere_panel.gen_btn.clicked.connect(self.on_gen_interfere_img)
        self.interfere_panel.apply_btn.clicked.connect(self.apply_interfere_to_fragments)

        # 劣化處理（新的 workflow handlers）
        self.degrade_panel.import_source_btn.clicked.connect(self.on_import_degrade_source)
        self.degrade_panel.gen_preview_btn.clicked.connect(self.on_generate_degrade_preview_shared)
        self.degrade_panel.apply_export_btn.clicked.connect(self.on_apply_degrade_source)
        self.degrade_panel.restore_source_btn.clicked.connect(self.on_restore_degrade_source)

        # 新增：監聽劣化參數改變以提示匯出未套用的預覽
        self.trash_tab.refresh()
        self._language_filter.retranslate_object(self, recursive=True)

    def toggle_language(self):
        language = LANG_ZH_TW if current_language() == LANG_EN else LANG_EN
        set_language(language)
        QtCore.QSettings("DuoDuo", "MSW Skin Fragmenter Pro").setValue(
            "language", language
        )
        self._language_filter.retranslate_object(self, recursive=True)
        self.img_wrap.language_btn.setToolTip(
            tr("切換繁體中文／English；不改變目前介面尺寸")
        )
        self.set_status(
            "Language: English" if language == LANG_EN else "語言：繁體中文",
            True,
        )

    def progress_step(self, step, total, msg):
        if msg:
            self.set_status(f"{msg} ({step}/{total})", True)
        else:
            self.set_status(f"拆解中... ({step}/{total})", True)

    def initUI(self):
        main = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.img_wrap, stretch=1)
        right = QtWidgets.QVBoxLayout()

        # 主圖 / 遮罩區
        ff = QtWidgets.QFormLayout()
        self.main_btn = QtWidgets.QPushButton("選擇主圖")
        self.main_btn.clicked.connect(self.load_main)
        self.main_file_lbl = ClickableFileLabel(self, 'main')
        main_row = QtWidgets.QHBoxLayout()
        main_row.addWidget(self.main_btn)
        main_row.addWidget(QHelpButton("請上傳含有透明區的 PNG 檔案作為主圖進行切割。透明像素將不會參與分割。"))
        ff.addRow("主圖：", main_row)
        ff.addRow("", self.main_file_lbl)

        self.mask_btn = QtWidgets.QPushButton("載入主體遮罩")
        self.mask_btn.clicked.connect(self.load_mask)
        self.mask_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.mask_btn.setMinimumWidth(150)
        self.del_mask_btn = QtWidgets.QPushButton("移除")
        self.del_mask_btn.clicked.connect(self.del_mask)
        self.del_mask_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        # 舊的非自動分割收尾仍讀取此狀態；新的遮罩流程使用各自的溢出開關。
        self.mask_crop_cb = QtWidgets.QCheckBox("不溢出")
        self.mask_crop_cb.setChecked(True)
        self.mask_crop_cb.hide()

        self.primary_overflow_cb = QtWidgets.QCheckBox("溢出")
        self.primary_overflow_cb.setChecked(False)
        self.primary_overflow_cb.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self.invert_mask_cb = QtWidgets.QCheckBox("反轉")
        self.invert_mask_cb.setChecked(False)
        self.invert_mask_cb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.invert_mask_cb.stateChanged.connect(self.reload_mask_with_invert)

        help_btn = QHelpButton(
            "遮罩流程會依已載入的遮罩執行：只載入主體遮罩時，第一張為主體外框，主體內部拆成後續碎片；"
            "只載入次要遮罩時，第一張為次要遮罩內部，次要外框拆成後續碎片；兩張都有時，依「主要內／外 → "
            "次要內／外」順序處理。"
            "\n\n"
            "主體「溢出」只讓主體外框在主要內／外分離時向內部延伸，預設關閉。兩種溢出都不會延後到最終碎片拆分。"
            "\n\n"
            "遮罩必須是 PNG，且大小與主圖完全一致；alpha 大於 0 的像素視為遮罩範圍。"
            "\n\n"
            "若兩張遮罩都不載入，則執行基本拆分。最終名稱仍為碎片 1～N，共 N 張。"
        )
        help_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.primary_mask_help_btn = help_btn

        mask_row = QtWidgets.QHBoxLayout()
        mask_row.addWidget(self.mask_btn, 2)
        mask_row.addWidget(self.del_mask_btn, 1)
        mask_row.addWidget(self.primary_overflow_cb, 1)
        mask_row.addWidget(self.invert_mask_cb, 1)
        mask_row.addWidget(help_btn)
        ff.addRow("主體遮罩：", mask_row)
        self.mask_file_lbl = ClickableFileLabel(self, 'mask')
        ff.addRow("", self.mask_file_lbl)

        self.secondary_mask_btn = QtWidgets.QPushButton("載入次要遮罩")
        self.secondary_mask_btn.clicked.connect(self.load_secondary_mask)
        self.secondary_mask_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.secondary_mask_btn.setMinimumWidth(150)
        self.del_secondary_mask_btn = QtWidgets.QPushButton("移除")
        self.del_secondary_mask_btn.clicked.connect(self.del_secondary_mask)
        self.del_secondary_mask_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.secondary_overflow_cb = QtWidgets.QCheckBox("溢出")
        self.secondary_overflow_cb.setChecked(True)
        self.secondary_overflow_cb.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self.secondary_invert_mask_cb = QtWidgets.QCheckBox("反轉")
        self.secondary_invert_mask_cb.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self.secondary_invert_mask_cb.stateChanged.connect(self.reload_secondary_mask_with_invert)
        secondary_help_btn = QHelpButton(
            "次要遮罩可單獨使用；有主體遮罩時只處理主體內部，沒有主體遮罩時直接處理主圖。"
            "次要內部會保留到第一張；次要外框拆成後續碎片。勾選「溢出」時，次要外框只在這次"
            "內／外分離時溢進次要內部；預設開啟，並使用方塊尺寸與隨機度各一半的參數（最低為 1）。"
            "關閉後會嚴格限制在次要外框範圍。"
            "完整次要外框產生後，才用原始參數拆成後續碎片；後續拆片不再溢出。\n\n"
            "勾選「反轉」可交換透明與不透明範圍。"
        )
        secondary_help_btn.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred
        )
        self.secondary_mask_help_btn = secondary_help_btn

        secondary_mask_row = QtWidgets.QHBoxLayout()
        secondary_mask_row.addWidget(self.secondary_mask_btn, 2)
        secondary_mask_row.addWidget(self.del_secondary_mask_btn, 1)
        secondary_mask_row.addWidget(self.secondary_overflow_cb, 1)
        secondary_mask_row.addWidget(self.secondary_invert_mask_cb, 1)
        secondary_mask_row.addWidget(secondary_help_btn)
        ff.addRow("次要遮罩：", secondary_mask_row)
        self.secondary_mask_file_lbl = ClickableFileLabel(self, 'secondary_mask')
        ff.addRow("", self.secondary_mask_file_lbl)
        freeze_form_label_column(ff)
        right.addLayout(ff)

        # 拆解參數
        self.num_input = QtWidgets.QSpinBox(); self.num_input.setRange(1,10); self.num_input.setValue(7)
        self.block_input = QtWidgets.QSpinBox(); self.block_input.setRange(1,30); self.block_input.setValue(3)
        self.rand_input = QtWidgets.QSpinBox(); self.rand_input.setRange(1,100); self.rand_input.setValue(6)
        ff2 = QtWidgets.QFormLayout()
        row1 = QtWidgets.QHBoxLayout(); row1.addWidget(self.num_input); row1.addWidget(QHelpButton(
            "此數值就是最終碎片總數。設定 6 時，清單與匯出結果為碎片 1、2、3、4、5、6。"
        ))
        ff2.addRow("分片數量(1~10)：", row1)
        row2 = QtWidgets.QHBoxLayout(); row2.addWidget(self.block_input); row2.addWidget(QHelpButton(
            "定義分割的最小區塊（鏤空最小洞）的尺寸。數字越大，每個分割塊越大。單位：px\n\n優點：區塊大可提升運算速度、減少碎片數。\n\n缺點：太大會降低隱蔽度，過小可能造成卡頓。"
        ))
        ff2.addRow("方塊尺寸(1~30)：", row2)
        row3 = QtWidgets.QHBoxLayout(); row3.addWidget(self.rand_input); row3.addWidget(QHelpButton(
            "區塊尺寸的隨機倍率範圍，1 代表所有區塊尺寸固定，2 代表區塊尺寸會隨機在設定值的 1~2 倍間變化。\n\n優點：提高碎片形狀隨機性，難以預測與還原。\n\n缺點：過高會造成計算量大增與碎片難以辨認。"
        ))
        ff2.addRow("尺寸隨機度(1~100)：", row3)
        self.overlap_pixel_input = QtWidgets.QSpinBox()
        self.overlap_pixel_input.setRange(0, 100)
        self.overlap_pixel_input.setValue(1)
        row5 = QtWidgets.QHBoxLayout()
        row5.addWidget(self.overlap_pixel_input)
        row5.addWidget(QHelpButton(
            "拆解後於鏤空區補原圖像素作為重疊像素。\n數值為聯集不透明像素的比例，依各碎片可填補區域分別回補。\n\n優點：增加還原難度，讓每片有干擾。\n\n缺點：比例過高會導致效能大幅下降、檔案變大。"
        ))
        ff2.addRow("重疊像素比(0~100%)：", row5)
        self.aggregation_input = QtWidgets.QSpinBox()
        self.aggregation_input.setRange(1, 10)
        self.aggregation_input.setValue(5)
        row6 = QtWidgets.QHBoxLayout()
        row6.addWidget(self.aggregation_input)
        row6.addWidget(QHelpButton(
            "調整回補的重疊像素聚集程度。1=最分散，10=最密集，預設5。\n\n優點：可調整碎片間重疊區域型態，提升反逆向性。\n\n缺點：極端值可能造成運算異常或不自然分佈。"
        ))
        ff2.addRow("重疊像素聚合(1~10)：", row6)
        freeze_form_label_column(ff2)
        right.addLayout(ff2)

        # 操作按鈕
        cth = QtWidgets.QHBoxLayout()
        self.split_btn = QtWidgets.QPushButton("執行拆解")
        self.split_btn.clicked.connect(self.split)
        cth.addWidget(self.split_btn)

        self.partial_btn = QtWidgets.QPushButton("局部分割")
        self.partial_btn.setToolTip("先在左側預覽用『右鍵拖曳』框選區域，再按此鍵只重分割該區域")
        self.partial_btn.clicked.connect(self.partial_split)
        cth.addWidget(self.partial_btn)

        self.save_btn = QtWidgets.QPushButton("還原初始分割")
        self.save_btn.clicked.connect(self.restore_initial_state)
        cth.addWidget(self.save_btn)
        right.addLayout(cth)

        # 分頁
        self.tabs = QtWidgets.QTabWidget()
        self.fragment_list = FragmentListWidget(self)
        self.fragment_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.fragment_list.customContextMenuRequested.connect(self.show_fragment_context_menu)
        self.fragment_list.visibilityToggleRequested.connect(self.toggle_fragment_visibility)
        self.fragment_list.itemClicked.connect(self.fragment_clicked)
        self.fragment_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        fragment_page = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(fragment_page)

        # 碎片管理區
        self.normal_panel = QtWidgets.QWidget()
        normal_lay = QtWidgets.QHBoxLayout(self.normal_panel)
        self.up_btn = QtWidgets.QPushButton("↑ 上移")
        self.up_btn.clicked.connect(self.move_fragment_up)
        normal_lay.addWidget(self.up_btn)
        self.down_btn = QtWidgets.QPushButton("↓ 下移")
        self.down_btn.clicked.connect(self.move_fragment_down)
        normal_lay.addWidget(self.down_btn)
        self.rename_btn = QtWidgets.QPushButton("重新命名")
        self.rename_btn.clicked.connect(self.rename_selected_fragment)
        normal_lay.addWidget(self.rename_btn)
        self.export_zip_btn = QtWidgets.QPushButton("全部匯出")
        self.export_zip_btn.clicked.connect(self.export_all_fragments_zip)
        normal_lay.addWidget(self.export_zip_btn)
        vbox.addWidget(self.fragment_list)
        vbox.addWidget(self.normal_panel)

        # 進階 panel
        self.adv_panel = QtWidgets.QWidget()
        adv_grid = QtWidgets.QGridLayout(self.adv_panel)
        self.merge_btn = QtWidgets.QPushButton("合併碎片")
        self.merge_btn.clicked.connect(
            lambda: self.merge_checked_restore_fragments(self.get_selected_fragments())
        )
        adv_grid.addWidget(self.merge_btn, 0, 0)
        self.copy_btn = QtWidgets.QPushButton("複製碎片")
        self.copy_btn.clicked.connect(
            lambda: self.copy_checked_restore_fragments(self.get_selected_fragments())
        )
        adv_grid.addWidget(self.copy_btn, 0, 1)
        self.delete_btn = QtWidgets.QPushButton("刪除碎片")
        self.delete_btn.setStyleSheet("background:#b33; color:#fff; font-weight:bold;")
        self.delete_btn.clicked.connect(
            lambda: self.delete_checked_restore_fragments(self.get_selected_fragments())
        )
        adv_grid.addWidget(self.delete_btn, 0, 2)
        self.rename_adv_btn = QtWidgets.QPushButton("重新命名")
        self.rename_adv_btn.clicked.connect(self.rename_selected_fragments)
        adv_grid.addWidget(self.rename_adv_btn, 1, 0)
        self.import_btn = QtWidgets.QPushButton("匯入碎片")
        self.import_btn.clicked.connect(self.import_fragments_btn)
        adv_grid.addWidget(self.import_btn, 1, 1)
        self.export_menu = self._make_export_menu()
        self.export_btn = QtWidgets.QPushButton("匯出碎片")
        self.export_btn.setMenu(self.export_menu)
        adv_grid.addWidget(self.export_btn, 1, 2)
        self.adv_panel.setVisible(False)
        vbox.addWidget(self.adv_panel)

        h_adv = QtWidgets.QHBoxLayout()
        self.restore_btn = QtWidgets.QPushButton("進階管理 / 還原預覽")
        self.restore_btn.clicked.connect(self.restore_preview)
        self.restore_btn.setStyleSheet("background:#444; color:#fff; font-weight:bold;")
        h_adv.addWidget(self.restore_btn)
        vbox.addLayout(h_adv)

        self.tabs.addTab(fragment_page, "碎片管理")

        # 干擾像素 tab
        self.interfere_panel = InterferePanel(self)
        interfere_tab = QtWidgets.QWidget()
        interfere_layout = QtWidgets.QVBoxLayout(interfere_tab)
        interfere_layout.addWidget(self.interfere_panel)
        self.tabs.addTab(interfere_tab, "干擾像素")

        # 劣化處理 tab（用已定義的 DegradePanel）
        self.degrade_panel = DegradePanel(self)
        degrade_proc_tab = QtWidgets.QWidget()
        degrade_proc_layout = QtWidgets.QVBoxLayout(degrade_proc_tab)
        degrade_proc_layout.addWidget(self.degrade_panel)
        self.tabs.addTab(degrade_proc_tab, "劣化處理")
        self.degrade_tab = degrade_proc_tab  # 記住劣化處理的 tab 方便比對

        # 垃圾桶
        self.trash_tab = TrashCanWidget(self, self.recycle_bin)
        self.tabs.addTab(self.trash_tab, "垃圾桶")

        # 關於（固定放在垃圾桶右側）
        self.about_tab = QtWidgets.QWidget()
        about_layout = QtWidgets.QVBoxLayout(self.about_tab)
        about_layout.setContentsMargins(24, 24, 24, 24)
        about_layout.setSpacing(16)

        about_title = QtWidgets.QLabel(f"MSW Skin Fragmenter Pro v{APP_VERSION}")
        about_title.setAlignment(QtCore.Qt.AlignCenter)
        about_title.setStyleSheet("font-size:20px; font-weight:bold; color:#fff;")
        about_layout.addWidget(about_title)

        about_summary = QtWidgets.QLabel("圖片碎片拆解、干擾像素與劣化處理工具")
        about_summary.setAlignment(QtCore.Qt.AlignCenter)
        about_summary.setStyleSheet("color:#bbb; font-size:13px;")
        about_layout.addWidget(about_summary)

        risk_group = QtWidgets.QGroupBox("用途與風險")
        risk_layout = QtWidgets.QVBoxLayout(risk_group)
        risk_text = QtWidgets.QLabel(
            "本工具僅供技術交流與學術用途，不保證碎片不可被還原。\n\n"
            "使用者需自行評估並承擔使用本工具所產生的所有風險。"
        )
        risk_text.setWordWrap(True)
        risk_text.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        risk_text.setStyleSheet("color:#ccc;")
        risk_layout.addWidget(risk_text)
        about_layout.addWidget(risk_group)

        license_group = QtWidgets.QGroupBox("版本與授權")
        license_layout = QtWidgets.QFormLayout(license_group)
        license_layout.addRow("版本：", QtWidgets.QLabel(APP_VERSION))
        license_layout.addRow("作者：", QtWidgets.QLabel("DuoDuo"))
        license_layout.addRow("開源授權：", QtWidgets.QLabel("MIT License"))
        license_layout.addRow("著作權：", QtWidgets.QLabel("© 2025 DuoDuo"))
        about_layout.addWidget(license_group)
        about_layout.addStretch()
        self.tabs.addTab(self.about_tab, "關於")

        right.addWidget(self.tabs, stretch=1)
        self._suppress_partial_warn = False

        main.addLayout(left, stretch=2)
        main.addLayout(right, stretch=1)
        self.switch_panel(False)

    def _next_fragment_name(self):
        base = tr("碎片") + " "
        i = 1
        while True:
            name = f"{base}{i}"
            if name not in self.fragment_data:
                return name
            i += 1

    def _confirm_partial_split_warning(self) -> bool:
        """顯示局部分割警告，回傳是否繼續。尊重「下次重新啟動前不再顯示」"""
        if getattr(self, "_suppress_partial_warn", False):
            return True

        box = QtWidgets.QMessageBox(self)
        box.setIcon(QtWidgets.QMessageBox.Warning)
        box.setWindowTitle("局部分割將覆蓋既有干擾像素")
        box.setText(
            "局部分割會先清除框選範圍內既有碎片內容，再依目前分片數量重新分配。\n\n"
            "重新分割完成後，程式會只在框選範圍內自動套用目前的干擾像素參數；"
            "框選範圍外的既有內容與干擾不會改變。"
        )
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        box.setDefaultButton(QtWidgets.QMessageBox.No)

        cb = QtWidgets.QCheckBox("下次重新啟動前不再顯示")
        box.setCheckBox(cb)

        ret = box.exec()
        if cb.isChecked():
            self._suppress_partial_warn = True
        return ret == QtWidgets.QMessageBox.Yes


    def partial_split(self):
        # 1) 前置檢查
        if self.main_img is None:
            QtWidgets.QMessageBox.information(self, "缺少主圖", "請先載入主圖")
            return
        sel = self.img_wrap.preview.get_selection_rect_img()
        if not sel:
            QtWidgets.QMessageBox.information(self, "未框選區域", "請在左側預覽用滑鼠右鍵拖曳框出一個區域後再按此鍵")
            return

        # === 新增：干擾像素覆蓋風險提示（可勾選本次不再顯示） ===
        if not self._confirm_partial_split_warning():
            return
        # === 新增結束 ===

        x, y, w, h = sel
        H, W = self.main_img.shape[:2]
        # 保守夾在邊界
        x = max(0, min(x, W-1)); y = max(0, min(y, H-1))
        w = max(1, min(w, W-x)); h = max(1, min(h, H-y))

        # 2) 依全圖流程建立局部來源；兩種溢出只在各自的內／外分離時建立。
        primary_outer_crop = None
        secondary_inner_crop = None
        mask_crop = None
        strict_mask = True
        blocksize = self.block_input.value()
        rand_factor = self.rand_input.value()
        if self.mask_img is not None or self.secondary_mask_img is not None:
            try:
                stages = build_mask_workflow_stages(
                    self.main_img,
                    self.mask_img,
                    self.secondary_mask_img,
                    primary_overflow=self.primary_overflow_cb.isChecked(),
                    block_size=blocksize,
                    random_factor=rand_factor,
                )
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "遮罩無法使用", str(exc))
                return
            primary_outer_crop = stages.primary_outer[y:y+h, x:x+w]
            secondary_inner_crop = stages.secondary_inner[y:y+h, x:x+w]
            main_crop = stages.secondary_outer_source[y:y+h, x:x+w]
            if stages.secondary_outer_mask is not None:
                mask_crop = stages.secondary_outer_mask[y:y+h, x:x+w]
        else:
            main_crop = self.main_img[y:y+h, x:x+w]

        # 3) 組合 SplitThread 參數（沿用目前 UI）
        final_count = self.num_input.value()
        has_mask_workflow = primary_outer_crop is not None
        split_piece_count = (
            max(1, final_count - 1) if has_mask_workflow else final_count
        )

        self._begin_fragment_progress(
            "局部分割進度", "局部分割：正在建立外框與碎片..."
        )
        self._set_fragment_progress_stage(0, 65)
        self.partial_btn.setEnabled(False)
        self.split_btn.setEnabled(False)

        # 4) 在背景執行緒分割；等待期間仍持續處理 UI 與進度事件。
        coverage = (
            mask_crop
            if mask_crop is not None
            else main_crop[..., 3] > 0
        )
        if np.any(coverage):
            th = SplitThread(
                main_crop,
                mask_crop,
                split_piece_count,
                blocksize,
                rand_factor,
                strict_mask=strict_mask,
                secondary_overflow_mask=(
                    mask_crop
                    if mask_crop is not None
                    and self.secondary_overflow_cb.isChecked()
                    else None
                ),
            )
            result_holder = {"imgs": None}
            wait_loop = QtCore.QEventLoop(self)
            def _cap(imgs):
                result_holder["imgs"] = imgs
                wait_loop.quit()
            th.result.connect(_cap)
            th.update_progress.connect(self._fragment_progress_update)
            th.finished.connect(wait_loop.quit)
            self.partial_split_thread = th
            th.start()
            wait_loop.exec()
            split_small_imgs = result_holder["imgs"]
        else:
            split_small_imgs = [
                np.zeros_like(main_crop) for _ in range(split_piece_count)
            ]
            self._fragment_progress_update(
                split_piece_count,
                split_piece_count,
                "局部範圍沒有外框像素，正在建立透明分片...",
            )

        if split_small_imgs is None or len(split_small_imgs) != split_piece_count:
            self.partial_btn.setEnabled(True)
            self.split_btn.setEnabled(True)
            self._finish_fragment_progress()
            QtWidgets.QMessageBox.warning(
                self, "局部分割失敗", "分片產生失敗，請檢查遮罩與參數"
            )
            return

        partial_final_base = None
        if has_mask_workflow:
            # 先保留階段 1 與階段 2 的結果；完成干擾後才做最終合成。
            if final_count == 1:
                new_small_imgs = [split_small_imgs[0]]
            else:
                new_small_imgs = [np.zeros_like(primary_outer_crop)] + list(split_small_imgs)
            primary_outer_full = np.zeros((H, W, 4), dtype=np.uint8)
            secondary_inner_full = np.zeros((H, W, 4), dtype=np.uint8)
            primary_outer_full[y:y+h, x:x+w] = primary_outer_crop
            secondary_inner_full[y:y+h, x:x+w] = secondary_inner_crop
            partial_final_base = (primary_outer_full, secondary_inner_full)
        else:
            new_small_imgs = list(split_small_imgs)
        if not new_small_imgs or not any(np.any(img[..., 3] > 0) for img in new_small_imgs):
            self.partial_btn.setEnabled(True)
            self.split_btn.setEnabled(True)
            self._finish_fragment_progress()
            QtWidgets.QMessageBox.warning(self, "局部分割失敗", "選定區域內沒有可分割的像素")
            return

        # 5) 貼回完整畫布；結果順序即清單由上往下的分配順序。
        new_imgs_full = []
        for arr in new_small_imgs:
            big = np.zeros((H, W, 4), dtype=np.uint8)
            big[y:y+h, x:x+w] = arr
            new_imgs_full.append(big)

        # 6) 先把「選取區域」從所有既有碎片中鏤空（清成透明）
        for name in list(self.fragment_order):
            img = self.fragment_data.get(name)
            if img is None:
                continue
            img[y:y+h, x:x+w] = 0
            self.fragment_data[name] = img

        # 7) 將新的局部分割結果灌回碎片
        existed = len(self.fragment_order)
        newly = len(new_imgs_full)
        fill_n = min(existed, newly)

        # 7a) 覆蓋到前 fill_n 個既有碎片（只覆蓋選框範圍，其它區域保持原樣）
        for i in range(fill_n):
            name = self.fragment_order[i]
            img = self.fragment_data[name]
            patch = new_imgs_full[i]
            sel_alpha = patch[...,3] > 0
            img[sel_alpha] = patch[sel_alpha]
            self.fragment_data[name] = img

        # 7b) 如果新結果比原本多：建立新碎片
        if newly > existed:
            for i in range(existed, newly):
                name = self._next_fragment_name()
                self.fragment_data[name] = new_imgs_full[i]
                self.fragment_order.append(name)

        # 8) 重新整理 UI 與預覽
        try:
            self.normalize_fragment_list_order()
        except Exception as e:
            print("局部分割後同步失敗", e)
        selection_mask = np.zeros((H, W), dtype=bool)
        selection_mask[y:y+h, x:x+w] = True
        target_names = self.fragment_order[:newly]
        self._start_partial_interference(
            target_names,
            selection_mask,
            added_count=max(0, newly - existed),
            final_base=partial_final_base,
        )

    def _start_partial_interference(
        self, target_names, selection_mask, added_count=0, final_base=None
    ):
        target_names = [name for name in target_names if name in self.fragment_data]
        if not target_names:
            self.set_status("局部分割完成，但沒有可處理的碎片", False)
            self.partial_btn.setEnabled(True)
            self.split_btn.setEnabled(True)
            self._finish_fragment_progress()
            return

        worker_names = target_names
        worker_data = {name: self.fragment_data[name] for name in target_names}
        expected_interference = max(0, len(target_names) - 2)

        settings = self.interfere_panel.get_settings()
        settings["generation_limit_mask"] = selection_mask
        source = (
            self.interference_source_img
            if self.interference_source_img is not None
            else self.main_img
        )
        self._partial_interference_targets = target_names
        self._partial_interference_mask = selection_mask
        self._partial_expected_interference = expected_interference
        self._partial_added_count = added_count
        self._partial_final_base = final_base
        self._set_fragment_progress_stage(
            65, 100, "局部分割：正在產生干擾像素..."
        )
        self.partial_btn.setEnabled(False)
        self.set_status(
            f"局部分割完成，正在為最上方 {len(target_names)} 張套用局部干擾...",
            True,
        )
        self.gen_thread = InterfereGenThread(
            worker_data, settings, worker_names, source
        )
        self.gen_thread.progress.connect(self._fragment_progress_update)
        self.gen_thread.result.connect(self._partial_interference_done)
        self.gen_thread.start()

    def _partial_interference_done(self, interference_by_name):
        sender = self.sender()
        if sender is not None and sender is not self.gen_thread:
            return
        applied_count = 0
        for name in self._partial_interference_targets:
            original = self.fragment_data.get(name)
            interference = interference_by_name.get(name)
            if (
                original is not None
                and interference is not None
                and interference.shape == original.shape
            ):
                self.fragment_data[name] = apply_interfere_masked(
                    original, interference, self._partial_interference_mask
                )
                applied_count += 1

        # 最後一步：干擾完成後才把主體外框＋次要內部放入清單第一張。
        if (
            self._partial_final_base is not None
            and self._partial_interference_targets
        ):
            first_name = self._partial_interference_targets[0]
            first_image = self.fragment_data.get(first_name)
            if first_image is not None:
                primary_outer, secondary_inner = self._partial_final_base
                final_base = numpy_alpha_composite(
                    primary_outer, secondary_inner
                )
                self.fragment_data[first_name] = numpy_alpha_composite(
                    first_image, final_base
                )
        self._partial_final_base = None

        self.split_result = [
            self.fragment_data[name]
            for name in self.fragment_order
            if name in self.fragment_data
        ]
        self.normalize_fragment_list_order()
        self.partial_btn.setEnabled(True)
        self.split_btn.setEnabled(True)
        self._finish_fragment_progress()
        if self.fragment_order:
            self.fragment_list.setCurrentRow(0)
            self.img_wrap.preview.set_image(self.fragment_data[self.fragment_order[0]])

        target_count = len(self._partial_interference_targets)
        expected_count = self._partial_expected_interference
        if applied_count == expected_count:
            self.set_status(
                f"局部分割完成：框選內容位於最上方 {target_count} 張，局部干擾已套用",
                True,
            )
        else:
            self.set_status(
                f"局部分割完成，但局部干擾僅成功 {applied_count}/{expected_count} 張",
                False,
            )


    def show_progress(self, done, total, msg):
        if f"({done}/{total})" not in msg and total > 1:
            msg = f"{msg} ({done}/{total})"
        self.set_status(msg, True)

    def preview_mask_grayalpha(self, arr):
        if arr is None:
            self.img_wrap.preview.set_image(None)
            return
        alpha = arr[..., 3]
        gray = np.stack([alpha, alpha, alpha, np.full_like(alpha, 255)], axis=-1)
        self.img_wrap.preview.set_image(gray)

    def reload_mask_with_invert(self):
        if not self.mask_img_path:
            return
        try:
            im, arr = self._read_mask_file(
                self.mask_img_path, self.invert_mask_cb.isChecked()
            )
            self.mask_img = arr
            self.set_status("主體遮罩已重新載入", True)
            self.preview_mask_grayalpha(self.mask_img)
            self.set_file_label(self.mask_file_lbl, self.mask_img_path, im)
        except Exception as e:
            self.set_status(f"主體遮罩重載失敗: {e}", False)

    def reload_secondary_mask_with_invert(self):
        if not self.secondary_mask_img_path:
            return
        try:
            im, arr = self._read_mask_file(
                self.secondary_mask_img_path,
                self.secondary_invert_mask_cb.isChecked(),
            )
            self.secondary_mask_img = arr
            self.set_status("次要遮罩已重新載入", True)
            self.preview_mask_grayalpha(self.secondary_mask_img)
            self.set_file_label(
                self.secondary_mask_file_lbl, self.secondary_mask_img_path, im
            )
        except Exception as e:
            self.set_status(f"次要遮罩重載失敗: {e}", False)

    def _make_export_menu(self):
        menu = QtWidgets.QMenu(self)
        action_sel = QAction("匯出選擇碎片", self)
        action_sel.triggered.connect(self.export_selected_fragments)
        action_all = QAction("匯出全部碎片", self)
        action_all.triggered.connect(self.export_all_fragments_zip)
        action_psd = QAction("匯出 .psd", self)
        action_psd.triggered.connect(self.export_selected_fragments_psd)
        menu.addAction(action_sel)
        menu.addAction(action_all)
        menu.addSeparator()
        menu.addAction(action_psd)
        return menu

    def switch_panel(self, adv):
        self.normal_panel.setVisible(not adv)
        self.adv_panel.setVisible(adv)

    def get_selected_fragments(self):
        """依目前清單順序回傳選取項目，與眼睛顯示狀態無關。"""
        return [
            self.fragment_list.item(i).text()
            for i in range(self.fragment_list.count())
            if self.fragment_list.item(i).isSelected()
            and self.fragment_list.item(i).text() in self.fragment_data
        ]

    def show_fragment_context_menu(self, position):
        item = self.fragment_list.itemAt(position)
        if item is None:
            return

        if not item.isSelected():
            self.fragment_list.clearSelection()
            item.setSelected(True)
        self.fragment_list.setCurrentItem(item, QtCore.QItemSelectionModel.NoUpdate)
        selected = self.get_selected_fragments()
        menu = QtWidgets.QMenu(self.fragment_list)

        if self.restore_mode:
            show_action = menu.addAction("顯示選取碎片")
            show_action.triggered.connect(lambda: self.set_selected_fragment_visibility(True))
            hide_action = menu.addAction("隱藏選取碎片")
            hide_action.triggered.connect(lambda: self.set_selected_fragment_visibility(False))
            menu.addSeparator()

            merge_action = menu.addAction("合併選取碎片")
            merge_action.setEnabled(len(selected) >= 2)
            merge_action.triggered.connect(
                lambda: self.merge_checked_restore_fragments(self.get_selected_fragments())
            )
            copy_action = menu.addAction("複製選取碎片")
            copy_action.triggered.connect(
                lambda: self.copy_checked_restore_fragments(self.get_selected_fragments())
            )
            rename_action = menu.addAction("重新命名選取碎片")
            rename_action.triggered.connect(self.rename_selected_fragments)
            delete_action = menu.addAction("刪除選取碎片")
            delete_action.triggered.connect(
                lambda: self.delete_checked_restore_fragments(self.get_selected_fragments())
            )
            menu.addSeparator()
            export_action = menu.addAction("匯出選取碎片")
            export_action.triggered.connect(self.export_selected_fragments)
            export_psd_action = menu.addAction("匯出選取碎片為 .psd")
            export_psd_action.triggered.connect(self.export_selected_fragments_psd)
            import_action = menu.addAction("匯入碎片")
            import_action.triggered.connect(self.import_fragments_btn)
        else:
            row = self.fragment_list.row(item)
            up_action = menu.addAction("上移")
            up_action.setEnabled(row > 0)
            up_action.triggered.connect(self.move_fragment_up)
            down_action = menu.addAction("下移")
            down_action.setEnabled(row < self.fragment_list.count() - 1)
            down_action.triggered.connect(self.move_fragment_down)
            menu.addSeparator()
            rename_action = menu.addAction("重新命名")
            rename_action.triggered.connect(self.rename_selected_fragment)
            export_action = menu.addAction("匯出此碎片")
            export_action.triggered.connect(
                lambda: self.export_single_fragment_by_name(item.text())
            )
            export_all_action = menu.addAction("全部匯出 ZIP")
            export_all_action.triggered.connect(self.export_all_fragments_zip)

        menu.exec(self.fragment_list.mapToGlobal(position))

    def toggle_fragment_visibility(self, item):
        name = item.text()
        if name not in self.fragment_data:
            return
        visible = not self.fragment_visibility.get(name, True)
        self.fragment_visibility[name] = visible
        self.fragment_list.set_item_visible(item, visible)
        self.update_restore_preview()

    def set_selected_fragment_visibility(self, visible):
        names = self.get_selected_fragments()
        for name in names:
            self.fragment_visibility[name] = bool(visible)
        for index in range(self.fragment_list.count()):
            item = self.fragment_list.item(index)
            if item.text() in names:
                self.fragment_list.set_item_visible(item, visible)
        self.update_restore_preview()

    def restore_preview(self):
        # 進入進階管理
        self.restore_mode = True
        self.restore_btn.setText("結束進階管理")
        self.restore_btn.setStyleSheet("background:#007aff; color:#fff; font-weight:bold;")
        self.restore_btn.clicked.disconnect()
        self.restore_btn.clicked.connect(self.cancel_restore_preview)

        # 先清掉舊的高亮/疊圖狀態，避免看到舊快取
        try:
            self.trash_tab.clear_highlight()
        except Exception:
            pass
        self.img_wrap.preview.trash_highlight = False
        self.img_wrap.preview.overlay_mode = False

        self.populate_fragment_list_advanced()
        self.update_restore_preview()

        self.img_wrap.overlay_btn.setVisible(True)
        self.img_wrap.overlay_btn.setChecked(False)
        self.overlay_active = False
        self.switch_panel(True)

    def cancel_restore_preview(self, silent=False):
        # 離開進階管理，回到單片預覽（去除重複與矛盾動作）
        self.restore_mode = False
        self.restore_btn.setText("進階管理 / 還原預覽")
        self.restore_btn.setStyleSheet("background:#444; color:#fff; font-weight:bold;")
        self.restore_btn.clicked.disconnect()
        self.restore_btn.clicked.connect(self.restore_preview)

        # 清除任何殘留的合成/疊圖狀態
        self.img_wrap.preview.trash_highlight = False
        self.img_wrap.preview.overlay_mode = False

        self.populate_fragment_list_no_checkbox()

        # 回到目前選取的碎片（或第一片）
        cur = self.fragment_list.currentItem()
        if cur:
            self.fragment_clicked(cur)
        elif self.fragment_list.count() > 0:
            self.fragment_list.setCurrentRow(0)
            self.fragment_clicked(self.fragment_list.item(0))

        self.img_wrap.overlay_btn.setVisible(False)
        self.img_wrap.overlay_btn.setChecked(False)
        self.overlay_active = False
        self.switch_panel(False)
        if not silent:
            self.set_status("已退出進階管理", True)

    def merge_checked_btn(self):
        self.merge_checked_restore_fragments(self.get_selected_fragments())

    def copy_checked_btn(self):
        self.copy_checked_restore_fragments(self.get_selected_fragments())

    def delete_checked_btn(self):
        self.delete_checked_restore_fragments(self.get_selected_fragments())

    def rename_checked_btn(self):
        self.rename_selected_fragments()

    def import_fragments_btn(self):
        fns, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "匯入碎片", "", "PNG圖檔 (*.png)")
        for fn in fns:
            try:
                im = Image.open(fn).convert("RGBA")
                arr = pil2np(im)
                h, w = arr.shape[:2]
                if self.main_img is not None:
                    main_h, main_w = self.main_img.shape[:2]
                    pad_arr = np.zeros((main_h, main_w, 4), dtype=np.uint8)
                    pad_arr[:min(h,main_h), :min(w,main_w), :] = arr[:min(h,main_h), :min(w,main_w), :]
                else:
                    pad_arr = arr.copy()
                base_name = os.path.splitext(os.path.basename(fn))[0]
                new_name = base_name
                idx = 1
                while new_name in self.fragment_data:
                    new_name = f"{base_name}_{idx}"
                    idx += 1
                self.fragment_data[new_name] = pad_arr
                self.fragment_order.append(new_name)
                self.fragment_visibility[new_name] = True
            except Exception as e:
                self.set_status(f"匯入失敗: {e}", False)
        if self.restore_mode:
            self.populate_fragment_list_advanced()
            self.update_restore_preview()
        else:
            self.populate_fragment_list_no_checkbox()
        self.set_status("已匯入碎片", True)
        self.normalize_fragment_list_order()
        self.split_result = [self.fragment_data[name] for name in self.fragment_order]

    def import_fragments_from_files(self, file_list):
        for fn in file_list:
            try:
                im = Image.open(fn).convert("RGBA")
                arr = pil2np(im)
                h, w = arr.shape[:2]
                if self.main_img is not None:
                    main_h, main_w = self.main_img.shape[:2]
                    pad_arr = np.zeros((main_h, main_w, 4), dtype=np.uint8)
                    pad_arr[:min(h,main_h), :min(w,main_w), :] = arr[:min(h,main_h), :min(w,main_w), :]
                else:
                    pad_arr = arr.copy()
                base_name = os.path.splitext(os.path.basename(fn))[0]
                new_name = base_name
                idx = 1
                while new_name in self.fragment_data:
                    new_name = f"{base_name}_{idx}"
                    idx += 1
                self.fragment_data[new_name] = pad_arr
                self.fragment_order.append(new_name)
                self.fragment_visibility[new_name] = True
            except Exception as e:
                self.set_status(f"匯入失敗: {e}", False)
        if self.restore_mode:
            self.populate_fragment_list_advanced()
            self.update_restore_preview()
        else:
            self.populate_fragment_list_no_checkbox()
        self.set_status("已匯入碎片", True)
        self.split_result = [self.fragment_data[name] for name in self.fragment_order]

    def export_selected_fragments(self):
        selected = self.get_selected_fragments()
        if not selected:
            QtWidgets.QMessageBox.information(self, "匯出失敗", "請先選取要匯出的碎片！")
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇匯出資料夾")
        if not folder: return
        try:
            for name in selected:
                img = self.fragment_data.get(name)
                if img is not None:
                    np2pil(img).save(os.path.join(folder, f"{name}.png"))
            self.set_status("已匯出選擇碎片", True)
        except Exception as e:
            self.set_status(f"匯出失敗: {e}", False)

    def export_selected_fragments_psd(self):
        selected = self.get_selected_fragments()
        if not selected:
            QtWidgets.QMessageBox.information(
                self, "PSD 匯出失敗", "請先在進階管理選取要匯出的碎片！"
            )
            return
        template_dir = os.path.join(application_dir(), "psd_templates")
        default_prefix = (
            os.path.splitext(os.path.basename(self.main_img_path))[0] + "_"
            if self.main_img_path
            else ""
        )
        dialog = PsdExportDialog(
            selected,
            template_dir,
            default_prefix,
            self,
        )
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇 PSD 匯出資料夾")
        if not folder:
            return
        self._psd_export_progress = QtWidgets.QProgressDialog(
            "正在準備 PSD 匯出...", "", 0, 0, self
        )
        self._psd_export_progress.setWindowTitle("PSD 匯出進度")
        self._psd_export_progress.setWindowModality(QtCore.Qt.WindowModal)
        self._psd_export_progress.setCancelButton(None)
        self._psd_export_progress.setMinimumDuration(0)
        self._psd_export_progress.setAutoClose(False)
        self._psd_export_progress.setAutoReset(False)
        self._psd_export_progress.show()

        self._psd_export_worker = PsdExportWorker(
            dialog.assignments(),
            self.fragment_data,
            folder,
            dialog.output_prefix(),
            self,
        )
        self._psd_export_worker.progress.connect(self._update_psd_export_progress)
        self._psd_export_worker.result.connect(self._psd_export_succeeded)
        self._psd_export_worker.error.connect(self._psd_export_failed)
        self._psd_export_worker.finished.connect(self._psd_export_finished)
        self.set_status("正在背景匯出 PSD...", True)
        self._psd_export_worker.start()

    def _update_psd_export_progress(self, current, total, message):
        progress = getattr(self, "_psd_export_progress", None)
        if progress is None:
            return
        progress.setRange(0, max(1, total))
        progress.setValue(current)
        progress.setLabelText(message)
        self.set_status(message, True)

    def _psd_export_succeeded(self, output_paths):
        progress = getattr(self, "_psd_export_progress", None)
        if progress is not None:
            progress.setValue(progress.maximum())
            progress.close()
        names = "、".join(path.name for path in output_paths)
        self.set_status(f"PSD 匯出完成：{names}", True)
        QtWidgets.QMessageBox.information(
            self,
            "PSD 匯出完成",
            f"已輸出 {len(output_paths)} 個 PSD 檔案：\n{names}",
        )

    def _psd_export_failed(self, message):
        progress = getattr(self, "_psd_export_progress", None)
        if progress is not None:
            progress.close()
        self.set_status(f"PSD 匯出失敗：{message}", False)
        QtWidgets.QMessageBox.warning(self, "PSD 匯出失敗", message)

    def _psd_export_finished(self):
        worker = getattr(self, "_psd_export_worker", None)
        if worker is not None:
            worker.deleteLater()
        self._psd_export_worker = None
        self._psd_export_progress = None

    def batch_rename_fragments(self, names=None):
        names = [name for name in (names or []) if name in self.fragment_data]
        if not names:
            return
        count = len(names)
        prefix, ok = QtWidgets.QInputDialog.getText(
            self,
            "批次命名",
            "請輸入前綴（例如：碎片）",
            text=tr("碎片"),
        )
        if not ok or not prefix: return
        digits = len(str(count))
        reserved = set(self.fragment_data) - set(names)
        rename_map = {}
        for index, old_name in enumerate(names, 1):
            base = f"{prefix}_{str(index).zfill(digits)}"
            new_name = base
            suffix = 2
            while new_name in reserved or new_name in rename_map.values():
                new_name = f"{base}_{suffix}"
                suffix += 1
            rename_map[old_name] = new_name

        for old_name, new_name in rename_map.items():
            self.fragment_data[new_name] = self.fragment_data.pop(old_name)
            self.fragment_visibility[new_name] = self.fragment_visibility.pop(old_name, True)
        self.fragment_order = [rename_map.get(name, name) for name in self.fragment_order]
        self.normalize_fragment_list_order()
        if self.restore_mode:
            self.populate_fragment_list_advanced(rename_map.values())
            self.update_restore_preview()
        self.set_status(f"已批次命名為 {prefix}_***", True)
        self.refresh_fragment_order()

    def rename_selected_fragments(self):
        selected = self.get_selected_fragments()
        if len(selected) == 1:
            self.rename_fragment_by_name(selected[0])
        elif len(selected) > 1:
            self.batch_rename_fragments(selected)
        else:
            QtWidgets.QMessageBox.information(self, "請選擇碎片", "請先選取要重新命名的碎片")

    def get_current_fragment_order(self):
        names = [self.fragment_list.item(i).text() for i in range(self.fragment_list.count())]
        return [n for n in names if n in self.fragment_data]

    def set_status(self, msg, ok=True):
        if len(msg) > 38:
            msg = ellipsis_middle(msg, 38)
        if msg and (
            "未套用劣化" in msg or
            "劣化預覽中 尚未套用" in msg or
            "尚未掛載" in msg or
            "干擾像素預覽中" in msg
        ):
            color = "#FFD600"
        else:
            color = "#0f0" if ok else "#f55"
        display_msg = tr(msg)
        self.img_wrap.status_lbl.setText(display_msg)
        self.img_wrap.status_lbl.setStyleSheet(f"color:{color}; font-weight:bold;")
        self.img_wrap.status_lbl.setToolTip(display_msg)

    def set_file_label(self, label, path, img):
        if not path or img is None:
            label.setText("")
            label.setToolTip("")
            return
        base = os.path.basename(path)
        if isinstance(img, np.ndarray):
            shape = f"{img.shape[1]}x{img.shape[0]}"
        elif hasattr(img, "width") and hasattr(img, "height"):
            shape = f"{img.width}x{img.height}"
        else:
            shape = "未知"
        txt = f"{ellipsis_middle(base)} ({shape})"
        label.setText(txt)
        label.setToolTip(base + " 解析度：" + shape)

    def get_unique_name(self, base):
        name = base
        num = 2
        while name in self.fragment_data or name in self.fragment_order:
            name = f"{base}{num}"
            num += 1
        return name

    def load_main(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "選擇主圖", "", "PNG圖檔 (*.png)")
        im = None
        if fname:
            try:
                im = Image.open(fname)
                self.main_img = pil2np(im)
                self.main_img_path = fname
                # 掛載來源屬於上一張主圖；更換主圖時避免誤用舊劣化圖。
                self._set_interference_source_mounted(False)
                self.img_wrap.preview.set_image(self.main_img)
                self.set_status("主圖載入成功", True)
            except Exception as e:
                self.set_status(f"主圖載入失敗: {e}", False)
                self.main_img = None
                self.main_img_path = ""
                im = None
            self.set_file_label(self.main_file_lbl, self.main_img_path, im)

    def _read_mask_file(self, path, inverted=False):
        im = Image.open(path).convert("RGBA")
        arr = pil2np(im)
        if self.main_img is not None and arr.shape[:2] != self.main_img.shape[:2]:
            raise ValueError("遮罩尺寸必須與主圖完全一致")
        if inverted:
            arr = arr.copy()
            arr[..., 3] = 255 - arr[..., 3]
        return im, arr

    def load_mask(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "選擇主體遮罩", "", "PNG圖檔 (*.png)")
        im = None
        if fname:
            try:
                im, arr = self._read_mask_file(fname, self.invert_mask_cb.isChecked())
                self.mask_img = arr
                self.mask_img_path = fname
                self.set_status("主體遮罩載入成功", True)
                self.preview_mask_grayalpha(self.mask_img)
                self.set_file_label(self.mask_file_lbl, self.mask_img_path, im)
            except Exception as e:
                self.set_status(f"主體遮罩載入失敗: {e}", False)
                self.mask_img = None
                self.mask_img_path = ""
                self.mask_file_lbl.setText("")

    def load_secondary_mask(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "選擇次要遮罩", "", "PNG圖檔 (*.png)")
        im = None
        if fname:
            try:
                im, arr = self._read_mask_file(
                    fname, self.secondary_invert_mask_cb.isChecked()
                )
                self.secondary_mask_img = arr
                self.secondary_mask_img_path = fname
                self.set_status("次要遮罩載入成功", True)
                self.preview_mask_grayalpha(self.secondary_mask_img)
                self.set_file_label(
                    self.secondary_mask_file_lbl, self.secondary_mask_img_path, im
                )
            except Exception as e:
                self.set_status(f"次要遮罩載入失敗: {e}", False)
                self.secondary_mask_img = None
                self.secondary_mask_img_path = ""
                self.secondary_mask_file_lbl.setText("")

    def del_mask(self):
        self.mask_img = None
        self.mask_img_path = ""
        self.mask_file_lbl.setText("")
        self.set_status("已移除主體遮罩", True)

    def del_secondary_mask(self):
        self.secondary_mask_img = None
        self.secondary_mask_img_path = ""
        self.secondary_mask_file_lbl.setText("")
        self.set_status("已移除次要遮罩", True)

    def split(self):
        danger_msgs = []
        # 方塊尺寸極小且重疊像素比例高（最危險組合）
        if self.block_input.value() <= 2 and self.rand_input.value() == 1 and self.overlap_pixel_input.value() > 2:
            danger_msgs.append("方塊尺寸極小且重疊像素比例高，會嚴重卡頓甚至當機！")

        # 方塊極小（容易產生大量碎片）
        if self.block_input.value() <= 2:
            danger_msgs.append("方塊尺寸小於等於2，會產生極大量碎片，容易造成當機。")

        # 碎片數過多（記憶體警告）
        if self.num_input.value() > 20:
            danger_msgs.append("碎片數量超過20，極易造成記憶體暴增與當機。")

        # 尺寸隨機度高+方塊小
        if self.rand_input.value() > 20 and self.block_input.value() <= 4:
            danger_msgs.append("尺寸隨機度過高且方塊太小，碎片組合將暴增，容易當機。")

        # 重疊像素高
        if self.overlap_pixel_input.value() > 20:
            danger_msgs.append("重疊像素比例超過20%，處理大圖或高分割時可能造成介面無回應或記憶體不足。")

        # 聚合度＋重疊像素高
        if self.aggregation_input.value() >= 8 and self.overlap_pixel_input.value() > 5:
            danger_msgs.append("重疊像素聚合度高且比例大於5%，會讓補丁集中、容易卡死。")

        # 方塊尺寸與隨機度乘積過大
        if self.block_input.value() * self.rand_input.value() > 300:
            danger_msgs.append("方塊尺寸與隨機度相乘過大，將產生異常碎片，容易當機。")

        if danger_msgs:
            reply = QtWidgets.QMessageBox.warning(
                self, "高風險參數警告",
                "\n".join(danger_msgs) + "\n\n確定要繼續執行嗎？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                self.set_status("已取消執行", False)
                return

        if self.main_img is None:
            self.set_status("請先載入主圖", False)
            return
        has_primary_mask = self.mask_img is not None
        has_secondary_mask = self.secondary_mask_img is not None

        block_sz = self.block_input.value()
        rand_sz = self.rand_input.value()
        final_count = self.num_input.value()
        self._auto_mask_workflow = has_primary_mask or has_secondary_mask
        if self._auto_mask_workflow:
            try:
                stages = build_mask_workflow_stages(
                    self.main_img,
                    self.mask_img,
                    self.secondary_mask_img if has_secondary_mask else None,
                    primary_overflow=self.primary_overflow_cb.isChecked(),
                    block_size=block_sz,
                    random_factor=rand_sz,
                )
            except ValueError as exc:
                self.set_status(str(exc), False)
                QtWidgets.QMessageBox.warning(self, "遮罩無法使用", str(exc))
                return
        else:
            stages = None
            inner_frame_source = self.main_img.copy()
            split_mask = None

        if stages is not None:
            inner_frame_source = stages.secondary_outer_source
            split_mask = stages.secondary_outer_mask

        if self.fragment_data:
            reply = QtWidgets.QMessageBox.question(
                self, "警告",
                "當前碎片管理中還有碎片。\n\n執行拆解會把這些碎片全數移到垃圾桶，確定要繼續嗎？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No
            )
            if reply != QtWidgets.QMessageBox.Yes:
                self.set_status("已取消拆解", False)
                return
            for name, img in self.fragment_data.items():
                self.recycle_bin.append((name, img))
            self.fragment_list.clear()
            self.fragment_data.clear()
            self.fragment_order.clear()
            self.fragment_visibility.clear()
            self.trash_tab.refresh()

        split_piece_count = (
            max(1, final_count - 1) if self._auto_mask_workflow else final_count
        )
        self._auto_primary_outer = (
            stages.primary_outer if stages is not None else None
        )
        self._auto_secondary_inner = (
            stages.secondary_inner if stages is not None else None
        )
        self._auto_fragment_zero = None
        self._auto_inner_frame_source = inner_frame_source
        self._auto_split_mask = split_mask
        self._auto_final_fragment_count = final_count
        self._auto_inner_fragment_count = split_piece_count
        if self._auto_mask_workflow and final_count == 1:
            split_status = "遮罩拆解：正在建立唯一碎片..."
        elif self._auto_mask_workflow:
            split_status = f"遮罩拆解：正在產生碎片 2～{final_count}..."
        else:
            split_status = f"基本拆分：正在產生 {final_count} 張碎片..."
        self.set_status(split_status, True)
        self._begin_fragment_progress("執行拆解進度", split_status)
        self._set_fragment_progress_stage(0, 55)
        self.fragment_list.clear()
        self.fragment_data.clear()
        self.split_btn.setEnabled(False)
        self.partial_btn.setEnabled(False)
        self.split_thread = SplitThread(
            inner_frame_source,
            split_mask,
            split_piece_count,
            block_sz,
            rand_sz,
            strict_mask=True,
            secondary_overflow_mask=(
                split_mask
                if split_mask is not None
                and self.secondary_overflow_cb.isChecked()
                else None
            ),
        )
        self.split_thread.update_progress.connect(self._fragment_progress_update)
        self.split_thread.result.connect(self._auto_split_done)
        self._split_start_time = time.time()
        self.split_thread.start()

    def progress(self, done, total, msg):
        if msg:
            self.set_status(f"{msg} ({done}/{total})", True)
        else:
            self.set_status(f"拆解中... ({done}/{total})", True)

    def _begin_fragment_progress(self, title, message):
        previous = getattr(self, "_fragment_progress_dialog", None)
        if previous is not None:
            previous.close()
        dialog = QtWidgets.QProgressDialog(message, "", 0, 100, self)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(QtCore.Qt.WindowModal)
        dialog.setCancelButton(None)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setMinimumWidth(460)
        dialog.setValue(0)
        dialog.show()
        self._fragment_progress_dialog = dialog
        self._fragment_progress_stage = (0, 100)

    def _set_fragment_progress_stage(self, start, end, message=None):
        self._fragment_progress_stage = (int(start), int(end))
        dialog = getattr(self, "_fragment_progress_dialog", None)
        if dialog is not None:
            dialog.setValue(int(start))
            if message:
                dialog.setLabelText(message)

    def _fragment_progress_update(self, done, total, message):
        start, end = getattr(self, "_fragment_progress_stage", (0, 100))
        fraction = min(1.0, max(0.0, done / total)) if total else 0.0
        value = int(round(start + (end - start) * fraction))
        dialog = getattr(self, "_fragment_progress_dialog", None)
        if dialog is not None:
            dialog.setValue(value)
            dialog.setLabelText(message or "處理中...")
        status = (
            f"{message} ({done}/{total})"
            if message and total
            else (message or "處理中...")
        )
        self.set_status(status, True)

    def _finish_fragment_progress(self):
        dialog = getattr(self, "_fragment_progress_dialog", None)
        if dialog is not None:
            dialog.setValue(100)
            dialog.close()
            dialog.deleteLater()
        self._fragment_progress_dialog = None

    def _auto_split_done(self, images):
        sender = self.sender()
        if sender is not None and sender is not self.split_thread:
            return
        expected_count = self._auto_inner_fragment_count
        if len(images) != expected_count:
            self.split_btn.setEnabled(True)
            self.partial_btn.setEnabled(True)
            self.set_status("分片產生失敗，請檢查遮罩與拆解參數", False)
            self._finish_fragment_progress()
            return

        fill_pct = self.overlap_pixel_input.value()
        if fill_pct <= 0:
            self._start_auto_interference(images)
            return

        self.set_status("一鍵拆解：正在填充重疊像素...", True)
        self._set_fragment_progress_stage(
            55, 75, "執行拆解：正在填充重疊像素..."
        )
        self.overlap_thread = OverlapThread(
            images,
            self._auto_inner_frame_source,
            self._auto_split_mask
            if self._auto_split_mask is not None
            else self._auto_inner_frame_source,
            fill_pct,
            self.block_input.value(),
            self.rand_input.value(),
            self.aggregation_input.value(),
            limit_to_mask=True,
        )
        self.overlap_thread.progress.connect(self._fragment_progress_update)

        def finish_overlap(result_images):
            sender = self.sender()
            if sender is not None and sender is not self.overlap_thread:
                return
            if len(result_images) != expected_count:
                self.set_status("重疊像素處理失敗，改用原始分割結果", False)
                result_images = images
            self._start_auto_interference(result_images)

        self.overlap_thread.result.connect(finish_overlap)
        self.overlap_thread.start()

    def _start_auto_interference(self, inner_fragments):
        self._set_fragment_progress_stage(
            75, 100, "執行拆解：正在產生干擾像素..."
        )
        if self._auto_mask_workflow:
            output_names = [
                tr(f"碎片 {index}")
                for index in range(1, self._auto_final_fragment_count + 1)
            ]
            if self._auto_final_fragment_count == 1:
                worker_fragments = [inner_fragments[0]]
            else:
                # 干擾階段用透明占位保留清單索引；第一張不參與干擾或範圍。
                worker_fragments = [
                    np.zeros_like(self._auto_primary_outer)
                ] + list(inner_fragments)
        else:
            output_names = [
                tr(f"碎片 {index}")
                for index in range(1, len(inner_fragments) + 1)
            ]
            worker_fragments = list(inner_fragments)
        worker_names = output_names
        snapshot_fragments = [image.copy() for image in worker_fragments]
        fragment_data = dict(zip(worker_names, worker_fragments))
        source = (
            self.interference_source_img
            if self.interference_source_img is not None
            else self.main_img
        )

        self._auto_output_names = output_names
        self._auto_worker_names = worker_names
        self._auto_worker_fragments = worker_fragments
        self._auto_snapshot_fragments = snapshot_fragments
        self.set_status(
            "一鍵拆解：正在以掛載劣化圖產生干擾..."
            if self.interference_source_img is not None
            else "一鍵拆解：正在以主圖產生干擾...",
            True,
        )
        self.gen_thread = InterfereGenThread(
            fragment_data,
            self.interfere_panel.get_settings(),
            worker_names,
            source,
        )
        self.gen_thread.progress.connect(self._fragment_progress_update)
        self.gen_thread.result.connect(self._auto_interference_done)
        self.gen_thread.start()

    def _auto_interference_done(self, interference_by_name):
        sender = self.sender()
        if sender is not None and sender is not self.gen_thread:
            return
        names = self._auto_output_names
        applied_count = 0
        base_fragments = self._auto_worker_fragments
        snapshot_fragments = [
            image.copy() for image in self._auto_snapshot_fragments
        ]
        if self._auto_mask_workflow:
            # 最後一步才合成主體外框與次要內部。
            fragment_zero = numpy_alpha_composite(
                self._auto_primary_outer, self._auto_secondary_inner
            )
            self._auto_fragment_zero = fragment_zero
            if self._auto_final_fragment_count == 1:
                final_fragments = [
                    numpy_alpha_composite(fragment_zero, base_fragments[0])
                ]
                snapshot_fragments = [
                    numpy_alpha_composite(fragment_zero, snapshot_fragments[0])
                ]
            else:
                final_fragments = [fragment_zero]
                snapshot_fragments[0] = fragment_zero.copy()
        else:
            final_fragments = [base_fragments[0]]
        for index, original in enumerate(base_fragments[1:], 1):
            interference = interference_by_name.get(names[index])
            if interference is not None and interference.shape == original.shape:
                final_fragments.append(apply_interfere_masked(original, interference))
                applied_count += 1
            else:
                final_fragments.append(original)
        interference_target_count = max(0, len(final_fragments) - 2)

        self.fragment_data = dict(zip(names, final_fragments))
        self.fragment_order = names
        self.fragment_visibility = {name: True for name in names}
        self.split_result = final_fragments
        self.interfere_images_dict.clear()
        self.normalize_fragment_list_order()
        self.cancel_restore_preview(silent=True)
        self.fragment_list.setCurrentRow(0)
        self.img_wrap.preview.set_image(final_fragments[0])
        self._initial_snapshot = {
            "fragment_names": names,
            "fragment_imgs": [image.copy() for image in snapshot_fragments],
        }
        self.split_btn.setEnabled(True)
        self.partial_btn.setEnabled(True)
        self._finish_fragment_progress()

        elapsed = int(time.time() - getattr(self, "_split_start_time", time.time()))
        source_text = "掛載劣化圖" if self.interference_source_img is not None else "主圖"
        total_count = len(final_fragments)
        range_text = f"碎片 1～{total_count}"
        if applied_count == interference_target_count:
            self.set_status(
                f"拆解完成：{range_text}，共 {total_count} 張；干擾來源為{source_text}（{elapsed}秒）",
                True,
            )
        else:
            self.set_status(
                f"已產生 {total_count} 張碎片，但只有 {applied_count}/{interference_target_count} 張成功套用干擾",
                False,
            )

    def split_done(self, images):
        if hasattr(self, "mask_crop_cb") and self.mask_crop_cb.isChecked():
            # 改：用 α>0 作為保留區
            if self.mask_img is not None:
                cover_mask = (self.mask_img[..., 3] > 0)
            else:
                cover_mask = (self.main_img[..., 3] > 0)
            images = [crop_to_primary_mask(img, cover_mask) for img in images]

        if hasattr(self, "mask_crop_cb") and self.mask_crop_cb.isChecked() and self.mask_img is not None:
            images = [apply_mask_alpha(img, self.mask_img) for img in images]

        fill_pct = self.overlap_pixel_input.value()
        block_size = self.block_input.value()
        rand_range = self.rand_input.value()
        agg = self.aggregation_input.value()

        if fill_pct > 0:
            self.set_status("開始進行重疊像素填充...", True)
            self.overlap_thread = OverlapThread(
                images, self.main_img, self.mask_img,
                fill_pct, block_size, rand_range, agg,
                limit_to_mask=(hasattr(self, "mask_crop_cb") and self.mask_crop_cb.isChecked())
            )
            self.overlap_thread.progress.connect(self.progress)
            def finish_overlap(result_images):
                self.set_status("重疊像素填充完成", True)
                # 如勾選「不溢出」，重疊填充後再裁一次，使用 α>0
                if hasattr(self, "mask_crop_cb") and self.mask_crop_cb.isChecked():
                    if self.mask_img is not None:
                        cover_mask = (self.mask_img[..., 3] > 0)
                    else:
                        cover_mask = (self.main_img[..., 3] > 0)
                    result_images = [crop_to_primary_mask(img, cover_mask) for img in result_images]
                self._finish_split(result_images)
            self.overlap_thread.result.connect(finish_overlap)
            self.overlap_thread.start()
        else:
            self._finish_split(images)


    def _finish_split(self, images):
        self.fragment_data.clear()
        self.fragment_order.clear()
        self.fragment_visibility.clear()
        for idx, arr in enumerate(images):
            name = tr(f"碎片 {idx+1}")
            self.fragment_data[name] = arr
            self.fragment_order.append(name)
        self.normalize_fragment_list_order()
        tcost = int(time.time() - getattr(self, "_split_start_time", time.time()))
        h, m, s = tcost//3600, (tcost%3600)//60, tcost%60
        if h>0: st = f"{h}小時{m}分{s}秒"
        elif m>0: st = f"{m}分{s}秒"
        else: st = f"{s}秒"
        self.set_status(f"拆解完成，花費{st}", True)
        if images:
            self.img_wrap.preview.set_image(images[0])
            self.fragment_list.setCurrentRow(0)
        self.split_btn.setEnabled(True)
        self.cancel_restore_preview(silent=True)
        self._initial_snapshot = {
            'fragment_names': [tr(f"碎片 {i+1}") for i in range(len(images))],
            'fragment_imgs': images[:]
        }


    def restore_initial_state(self):
        if not self._initial_snapshot:
            QtWidgets.QMessageBox.information(self, "還原初始設定", "尚未有分割過的結果可還原！")
            return
        self.fragment_data.clear()
        self.fragment_order.clear()
        self.fragment_visibility.clear()
        for name, img in zip(self._initial_snapshot['fragment_names'], self._initial_snapshot['fragment_imgs']):
            self.fragment_data[name] = img
            self.fragment_order.append(name)
        self.split_result = self._initial_snapshot['fragment_imgs'][:]
        self.normalize_fragment_list_order()
        self.img_wrap.preview.set_image(self.split_result[0])
        self.set_status("已還原初始分割狀態", True)
        self.cancel_restore_preview(silent=True)

    def normalize_fragment_list_order(self):
        try:
            selected_before = set(self.get_selected_fragments())
            current_item = self.fragment_list.currentItem()
            current_name = current_item.text() if current_item else None
            # 過濾 fragment_order，只保留 fragment_data 有的
            self.fragment_order = [name for name in self.fragment_order if name in self.fragment_data]
            # 移除 fragment_data 裡沒有在 order 裡的殘影
            keys_to_del = [k for k in self.fragment_data if k not in self.fragment_order]
            for k in keys_to_del:
                del self.fragment_data[k]
            self.fragment_visibility = {
                name: self.fragment_visibility.get(name, True)
                for name in self.fragment_order
            }

            if self.restore_mode:
                self.populate_fragment_list_advanced(selected_before)
            else:
                self.populate_fragment_list_no_checkbox(current_name)

            self.overlay_active = False

        except Exception as e:
            print("normalize_fragment_list_order 錯誤：", e)
            # 可選：這裡可以加 QMessageBox 警告或 log 紀錄

    def populate_fragment_list_no_checkbox(self, preferred_name=None):
        self.fragment_list.clear()
        self.fragment_list.set_advanced_mode(False)
        self.fragment_list.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        for name in self.fragment_order:
            if name in self.fragment_data:
                item = QtWidgets.QListWidgetItem(name)
                item.setFlags(
                    QtCore.Qt.ItemIsEnabled |
                    QtCore.Qt.ItemIsSelectable
                )
                item.setSizeHint(QtCore.QSize(0, self.fragment_list.ROW_HEIGHT))
                self.fragment_list.addItem(item)
        self.fragment_list.setStyleSheet("QListWidget::item { padding: 3px; }")
        self.fragment_list.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.fragment_list.setFocus(QtCore.Qt.OtherFocusReason)
        target_row = 0
        if preferred_name in self.fragment_order:
            target_row = self.fragment_order.index(preferred_name)
        if self.fragment_list.count() > 0:
            self.fragment_list.setCurrentRow(target_row)
            self.fragment_clicked(self.fragment_list.item(target_row))
        else:
            self.fragment_list.setCurrentRow(-1)

    def populate_fragment_list_advanced(self, selected_names=None):
        if selected_names is None:
            selected_names = self.get_selected_fragments()
        selected_names = set(selected_names)
        self.fragment_list.clear()
        self.fragment_list.set_advanced_mode(True)
        self.fragment_list.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.fragment_list.setStyleSheet("""
            QListWidget::item { padding: 3px; }
            QListWidget::item:hover { background: #343434; }
            QListWidget::item:selected { background: #245a82; color: white; }
        """)
        self.fragment_list.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        for name in self.fragment_order:
            if name in self.fragment_data:
                item = QtWidgets.QListWidgetItem(name)
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                item.setSizeHint(QtCore.QSize(0, self.fragment_list.ROW_HEIGHT))
                self.fragment_list.set_item_visible(
                    item, self.fragment_visibility.get(name, True)
                )
                self.fragment_list.addItem(item)
                item.setSelected(name in selected_names)

    def _on_fragment_list_rows_moved(self, parent, start, end, dest, row):
        self.fragment_order = [
            self.fragment_list.item(i).text()
            for i in range(self.fragment_list.count())
            if self.fragment_list.item(i).text() in self.fragment_data
        ]
    def force_normal_preview(self):
        self.restore_mode = False
        self.populate_fragment_list_no_checkbox()
        self.img_wrap.preview.set_image(None)
        self.set_status("", True)
        if self.fragment_list.count() > 0:
            self.fragment_list.setCurrentRow(0)

    def update_restore_preview(self):
        # 進階模式下，用清單順序決定疊圖層級：清單最上面＝最上層（最後疊）
        if not self.restore_mode:
            return
        if not self.fragment_data or self.fragment_list.count() == 0:
            self.img_wrap.preview.set_image(None, trash_highlight=False)
            return

        # 1) 眼睛狀態只控制預覽可見性，與目前選取項目完全分離。
        names_in_list_order = [
            name for name in self.fragment_order
            if name in self.fragment_data and self.fragment_visibility.get(name, True)
        ]

        if not names_in_list_order:
            self.set_status("所有碎片皆已隱藏", False)
            self.img_wrap.preview.set_image(None, trash_highlight=False)
            return

        # 2) 以第一個可用碎片取得尺寸
        base_size = None
        for name in names_in_list_order:
            frag = self.fragment_data.get(name)
            if frag is not None:
                h, w = frag.shape[:2]
                base_size = (w, h)
                break
        if base_size is None:
            self.img_wrap.preview.set_image(None, trash_highlight=False)
            return

        # 3) 疊圖順序：由下到上 → 反轉清單（讓清單最上面最後疊）
        base = Image.new("RGBA", base_size, (0, 0, 0, 0))
        for name in reversed(names_in_list_order):  # bottom → top
            frag = self.fragment_data.get(name)
            if frag is None or frag.shape[0] == 0 or frag.shape[1] == 0:
                continue
            # 全透明直接略過
            if not np.any(frag[..., 3]):
                continue
            overlay = Image.fromarray(frag, mode="RGBA")
            base = Image.alpha_composite(base, overlay)

        # 4) 顯示（清掉任何殘留高亮/疊圖狀態）
        self.img_wrap.preview.trash_highlight = False
        self.img_wrap.preview.overlay_mode = False
        self.img_wrap.preview.set_image(base, trash_highlight=False)
        self.set_status(f"進階管理預覽：{len(names_in_list_order)} 片（清單最上層優先）", True)

    def merge_checked_restore_fragments(self, checked_names):
        if not self.restore_mode:
            return
        if len(checked_names) < 2:
            QtWidgets.QMessageBox.information(self, "請選擇碎片", "合併至少需要選取兩個碎片")
            return
        imgs = [self.fragment_data[name] for name in checked_names if name in self.fragment_data]
        if len(imgs) < 2:
            return
        base = imgs[0].copy()
        for img in imgs[1:]:
            base = numpy_alpha_composite(base, img)
        new_name = f"{tr('合併碎片_')}{len(self.fragment_data)+1}"
        idx = 1
        while new_name in self.fragment_data:
            idx += 1
            new_name = f"{tr('合併碎片_')}{len(self.fragment_data)+idx}"
        for name in checked_names:
            img = self.fragment_data.pop(name)
            self.fragment_visibility.pop(name, None)
            if name in self.fragment_order:
                self.fragment_order.remove(name)
            self.recycle_bin.append((name, img))
        self.fragment_data[new_name] = base
        self.fragment_order.append(new_name)
        self.fragment_visibility[new_name] = True
        self.populate_fragment_list_advanced([new_name])
        self.update_restore_preview()
        self.trash_tab.refresh()
        self.set_status(f"已合併並移除原有碎片，共{len(checked_names)}個->1", True)
        self.refresh_fragment_order()

        try:
            self.normalize_fragment_list_order()
        except Exception as e:
            print("合併後同步失敗", e)

    def copy_checked_restore_fragments(self, checked_names):
        to_copy = []
        for name in checked_names:
            img = self.fragment_data.get(name)
            if img is not None:
                to_copy.append((name, img))
        if not to_copy:
            QtWidgets.QMessageBox.information(self, "請選擇碎片", "請先選取要複製的碎片")
            return
        new_items = []
        for name, img in to_copy:
            new_name = name + tr("_複製")
            idx = 1
            while new_name in self.fragment_data:
                new_name = f"{name}{tr('_複製')}{idx}"
                idx += 1
            self.fragment_data[new_name] = img.copy()
            self.fragment_order.append(new_name)
            self.fragment_visibility[new_name] = self.fragment_visibility.get(name, True)
            new_items.append(new_name)
        self.populate_fragment_list_advanced(new_items)
        self.set_status(f"已複製 {len(new_items)} 個碎片", True)
        self.refresh_fragment_order()

    def delete_checked_restore_fragments(self, checked_names):
        to_delete = []
        for name in checked_names:
            img = self.fragment_data.get(name)
            if img is not None:
                to_delete.append((name, img))
        if not to_delete:
            QtWidgets.QMessageBox.information(self, "請選擇碎片", "請先選取要刪除的碎片")
            return
        for name, img in to_delete:
            if name in self.fragment_data:
                del self.fragment_data[name]
            self.fragment_visibility.pop(name, None)
            if name in self.fragment_order:
                self.fragment_order.remove(name)
            self.recycle_bin.append((name, img))
        self.populate_fragment_list_advanced()
        self.update_restore_preview()
        self.trash_tab.refresh()
        self.set_status(f"已刪除 {len(to_delete)} 個碎片", True)
        self.refresh_fragment_order()

        try:
            self.normalize_fragment_list_order()
        except Exception as e:
            print("刪除後同步失敗", e)

    def export_single_fragment_by_name(self, name):
        img = self.fragment_data.get(name)
        if img is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "儲存碎片", name + ".png", "PNG圖檔 (*.png)"
        )
        if not fn:
            return
        try:
            # 自動遮罩流程的清單最上方碎片包含主體遮罩外框，匯出時不得再次套主體遮罩。
            np2pil(img).save(fn)
            self.set_status(f"已儲存: {os.path.basename(fn)}", True)
        except Exception as e:
            self.set_status(f"儲存失敗: {e}", False)

    def restore_from_trash(self, name, img):
        base = name
        idx = 1
        while base in self.fragment_data:
            base = f"{name}{tr('_復原')}{idx}"
            idx += 1
        self.fragment_data[base] = img
        self.fragment_order.append(base)
        self.fragment_visibility[base] = True
        try:
            self.normalize_fragment_list_order()
        except Exception as e:
            print("復原後同步失敗", e)
        self.update_restore_preview()
        self.trash_tab.refresh()
        self.set_status(f"已復原碎片: {base}", True)

    def refresh_fragment_list(self):
        try:
            self.normalize_fragment_list_order()
        except Exception as e:
            print("刷新碎片清單時同步失敗", e)
        if self.restore_mode:
            self.update_restore_preview()
        self.img_wrap.preview.set_image(None, trash_highlight=False)

    def export_all_fragments_zip(self):
        if not self.fragment_data:
            QtWidgets.QMessageBox.information(self, "匯出失敗", "沒有任何碎片可匯出！")
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "儲存所有碎片",
            tr("所有碎片.zip"),
            "ZIP 壓縮檔 (*.zip)",
        )
        if not fn:
            return
        # 強制副檔名
        if not fn.lower().endswith('.zip'):
            fn += '.zip'
        try:
            with zipfile.ZipFile(fn, 'w', zipfile.ZIP_DEFLATED) as zf:
                for name, img in self.fragment_data.items():
                    if img is None:
                        continue
                    try:
                        img_bytes = io.BytesIO()
                        np2pil(img).save(img_bytes, format='PNG')
                        img_bytes.seek(0)
                        # 避免非法字元/重複
                        safe_name = "".join(c if c.isalnum() or c in ' ._-' else '_' for c in name)
                        zf.writestr(f"{safe_name}.png", img_bytes.read())
                    except Exception as e:
                        print(f"碎片 {name} 匯出失敗: {e}")
            self.set_status(f"已匯出全部碎片到: {os.path.basename(fn)}", True)
        except Exception as e:
            self.set_status(f"壓縮匯出失敗: {e}", False)

    def rename_fragment(self, item):
        old_name = item.text()
        new_name, ok = QtWidgets.QInputDialog.getText(self, "重新命名", "輸入新名稱", text=old_name)
        if not ok or not new_name or new_name == old_name:
            return
        if new_name in self.fragment_data:
            QtWidgets.QMessageBox.warning(self, "名稱重複", "此名稱已存在，請選擇其他名稱。")
            return
        img = self.fragment_data.pop(old_name)
        self.fragment_data[new_name] = img
        self.fragment_visibility[new_name] = self.fragment_visibility.pop(old_name, True)
        if old_name in self.fragment_order:
            idx = self.fragment_order.index(old_name)
            self.fragment_order[idx] = new_name
        item.setText(new_name)
        self.set_status(f"已重新命名為 {new_name}", True)
        # --- 新增同步 ---
        try:
            self.normalize_fragment_list_order()
        except Exception as e:
            print("改名後同步失敗", e)

    def rename_fragment_by_name(self, name):
        for i in range(self.fragment_list.count()):
            item = self.fragment_list.item(i)
            if item.text() == name:
                self.rename_fragment(item)
                return
        QtWidgets.QMessageBox.warning(self, "找不到碎片", f"找不到名為「{name}」的碎片。")

    def rename_selected_fragment(self):
        selected = self.fragment_list.currentItem()
        if not selected:
            QtWidgets.QMessageBox.information(self, "請選擇碎片", "請先選取要重新命名的碎片")
            return
        self.rename_fragment(selected)

    def on_tab_changed(self, idx):
        current = self.tabs.currentWidget()

        # 離開劣化頁籤才取消正在跑的 preview
        if hasattr(self, "degrade_tab") and current is not self.degrade_tab:
            self._cancel_degrade_preview_if_running()

        if current == self.trash_tab:
            # 垃圾桶邏輯
            self.trash_tab.update_preview()
            if not self.trash_tab.list.selectedItems():
                self.img_wrap.status_lbl.setText("翻找垃圾桶中")
                self.img_wrap.status_lbl.setStyleSheet("color:#f55; font-weight:bold;")
            return

        # 非垃圾桶頁籤先清除 highlight
        self.trash_tab.clear_highlight()

        if current == self.degrade_tab:
            # 劣化頁籤：優先顯示預覽
            if getattr(self, 'degrade_preview_pending', None):
                self.img_wrap.preview.set_image(self.degrade_preview_pending['degraded'])
                if self.interference_source_img is not None:
                    self.set_status("劣化圖已掛載為干擾像素主圖", True)
                else:
                    self.set_status("劣化預覽（尚未掛載）", True)
            elif getattr(self, 'degrade_source_img', None) is not None:
                self.img_wrap.preview.set_image(self.degrade_source_img)
                self.set_status("已載入劣化來源圖", True)
            else:
                # 無劣化資料 fallback 原本行為
                if self.restore_mode:
                    self.populate_fragment_list_advanced()
                    self.update_restore_preview()
                else:
                    if self.fragment_list.count() > 0:
                        first_name = self.fragment_list.item(0).text()
                        img = self.fragment_data.get(first_name)
                        if img is not None:
                            self.img_wrap.preview.set_image(img)
                        else:
                            self.img_wrap.preview.set_image(None)
                    else:
                        self.img_wrap.preview.set_image(None)
            return

        # 其他非垃圾桶、非劣化的分頁（例如碎片管理/干擾）
        if self.restore_mode:
            self.populate_fragment_list_advanced()
            self.update_restore_preview()
        else:
            if self.fragment_list.count() > 0:
                first_name = self.fragment_list.item(0).text()
                img = self.fragment_data.get(first_name)
                if img is not None:
                    self.img_wrap.preview.set_image(img)
                else:
                    self.img_wrap.preview.set_image(None)
            else:
                self.img_wrap.preview.set_image(None)
        self.set_status("", True)

    def closeEvent(self, event):
        psd_worker = getattr(self, "_psd_export_worker", None)
        if psd_worker is not None and psd_worker.isRunning():
            QtWidgets.QMessageBox.information(
                self,
                "PSD 正在匯出",
                "PSD 檔案仍在背景儲存，請等待進度完成後再關閉程式。",
            )
            event.ignore()
            return
        active_threads = []
        for attr in (
            "split_thread",
            "partial_split_thread",
            "overlap_thread",
            "gen_thread",
            "degrade_thread",
        ):
            th = getattr(self, attr, None)
            if th is not None and hasattr(th, "isRunning") and th.isRunning():
                try:
                    th.abort()
                    active_threads.append(th)
                except Exception:
                    pass
        still_running = []
        for th in active_threads:
            try:
                if not th.wait(2000):
                    still_running.append(th)
            except Exception:
                still_running.append(th)
        if still_running:
            QtWidgets.QMessageBox.information(
                self,
                "背景處理正在停止",
                "背景處理尚未完全停止，請稍候再關閉程式。",
            )
            event.ignore()
            return
        self._finish_fragment_progress()
        super().closeEvent(event)

    def generate_overlay_preview(self):
        if self.main_img is None:
            return None
        arr = self.main_img.copy()
        h, w = arr.shape[:2]
        alpha = arr[..., 3]
        bw_arr = np.stack([alpha, alpha, alpha, np.full_like(alpha, 255)], axis=-1)
        mask_count = np.zeros((h, w), dtype=np.uint8)
        for name in self.fragment_order:
            if self.fragment_visibility.get(name, True) and name in self.fragment_data:
                frag = self.fragment_data[name]
                fa = frag[..., 3]
                mask_count += (fa > 0).astype(np.uint8)
        overlap_mask = mask_count >= 2
        bw_arr[overlap_mask] = [255, 0, 0, 255]
        return bw_arr

    def restore_overlay_off(self):
        self.overlay_active = False
        self.img_wrap.preview.overlay_mode = False
        if self.restore_mode:
            self.update_restore_preview()
        elif self.split_result:
            self.img_wrap.preview.set_image(self.split_result[0], trash_highlight=False)
        else:
            self.img_wrap.preview.set_image(None, trash_highlight=False)

    def move_fragment_up(self):
        row = self.fragment_list.currentRow()
        if row > 0:
            item = self.fragment_list.takeItem(row)
            self.fragment_list.insertItem(row - 1, item)
            self.fragment_list.setCurrentRow(row - 1)
            self.refresh_fragment_order()

    def move_fragment_down(self):
        row = self.fragment_list.currentRow()
        if 0 <= row < self.fragment_list.count() - 1:
            item = self.fragment_list.takeItem(row)
            self.fragment_list.insertItem(row + 1, item)
            self.fragment_list.setCurrentRow(row + 1)
            self.refresh_fragment_order()

    def refresh_fragment_order(self):
        self.fragment_order = [
            self.fragment_list.item(i).text()
            for i in range(self.fragment_list.count())
            if self.fragment_list.item(i).text() in self.fragment_data
        ]

    def on_gen_interfere_img(self):
        if len(self.fragment_order) < 2:
            QtWidgets.QMessageBox.warning(self, "錯誤", "至少需要兩片碎片才能產生干擾")
            return

        kw = self.interfere_panel.get_settings()
        first_fragment = self.fragment_data.get(self.fragment_order[0])
        if first_fragment is None:
            QtWidgets.QMessageBox.warning(self, "錯誤", "找不到第一片遮罩碎片")
            return
        reference_shape = first_fragment.shape
        mismatched = [
            name for name in self.fragment_order
            if name not in self.fragment_data
            or self.fragment_data[name].shape != reference_shape
        ]
        if mismatched:
            QtWidgets.QMessageBox.warning(
                self, "碎片尺寸不一致",
                "所有碎片必須使用相同畫布尺寸才能產生干擾。\n"
                f"請先處理以下碎片：{', '.join(mismatched[:5])}"
            )
            return

        base_source = self.main_img
        if base_source is None:
            QtWidgets.QMessageBox.warning(
                self, "尚未載入主圖",
                "請先載入要作為干擾像素來源的主圖（可使用劣化處理後的圖片）。"
            )
            return
        self.interfere_images_dict.clear()
        self.set_status("開始產生干擾像素...", True)
        self.interfere_panel.gen_btn.setEnabled(False)
        self.interfere_panel.apply_btn.setEnabled(False)
        QtWidgets.QApplication.processEvents()

        self.gen_thread = InterfereGenThread(
            self.fragment_data,
            kw,
            self.fragment_order,
            base_source
        )
        active_gen_thread = self.gen_thread
        self.gen_thread.progress.connect(
            lambda cur, tot, msg: self.set_status(f"{msg} ({cur}/{tot})", True)
        )

        def done(result_dict):
            if active_gen_thread is not self.gen_thread:
                return
            self.interfere_images_dict = result_dict
            self.interfere_panel.gen_btn.setEnabled(True)
            self.interfere_panel.apply_btn.setEnabled(bool(result_dict))

            if not result_dict:
                self.set_status("未產生任何干擾像素", False)
                QtWidgets.QMessageBox.warning(
                    self, "產生失敗", "沒有產生任何干擾像素，請調整參數後再試。"
                )
                return

            self.force_normal_preview()   # 直接呼叫新 function

            # 選第一個碎片並預覽
            if self.fragment_list.count() > 0:
                self.fragment_list.setCurrentRow(0)
                current = self.fragment_list.currentItem()
                if current:
                    self.fragment_clicked(current)

            QtWidgets.QMessageBox.information(
                self, "產生完成",
                f"已為 {len(result_dict)} 片碎片產生干擾像素。\n"
                "可於「碎片管理」預覽，確認後再到「干擾像素」執行「合成到碎片」。"
            )
            self.tabs.setCurrentIndex(0)

        self.gen_thread.result.connect(done)
        self.gen_thread.start()

    def interfere_progress(self, done, total, msg):
        self.set_status(f"{msg} ({done}/{total})", True)

    def interfere_done(self, result):
        self.interfere_images_dict = result
        self.set_status("干擾像素產生完成", True)
        self.interfere_panel.gen_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "產生完成", "已為每片碎片產生干擾像素，可執行「合成到碎片」或重複執行疊加。")

    def apply_interfere_to_fragments(self):
        if not self.interfere_images_dict:
            QtWidgets.QMessageBox.information(self, "尚未產生", "請先產生干擾像素")
            return
        if not self.fragment_data or not self.fragment_order:
            QtWidgets.QMessageBox.warning(self, "錯誤", "找不到碎片")
            return

        new_fragment_data = {}
        new_fragment_order = []
        new_fragment_visibility = {}

        # 第一片（mask）保留原樣
        first = self.fragment_order[0]
        if first in self.fragment_data:
            new_fragment_data[first] = self.fragment_data[first]
            new_fragment_order.append(first)
            new_fragment_visibility[first] = self.fragment_visibility.get(first, True)

        cnt = 0
        for name in self.fragment_order[1:]:
            interfere = self.interfere_images_dict.get(name)
            orig = self.fragment_data.get(name)
            if orig is not None and interfere is not None and interfere.shape == orig.shape:
                # 干擾圖已帶有該片隨機選出的前方碎片聯集範圍。
                merged = apply_interfere_masked(orig, interfere)
                new_name = self.get_unique_name(name + tr("_干擾"))
                new_fragment_data[new_name] = merged
                new_fragment_order.append(new_name)
                new_fragment_visibility[new_name] = self.fragment_visibility.get(name, True)
                self.recycle_bin.append((name, orig))
                cnt += 1
            else:
                new_fragment_data[name] = self.fragment_data.get(name)
                new_fragment_order.append(name)
                new_fragment_visibility[name] = self.fragment_visibility.get(name, True)

        self.fragment_data = new_fragment_data
        self.fragment_order = new_fragment_order
        self.fragment_visibility = new_fragment_visibility
        self.interfere_images_dict.clear()
        self.populate_fragment_list_no_checkbox()
        self.trash_tab.refresh()
        QtWidgets.QMessageBox.information(self, "合成完成", f"已合成到 {cnt} 個碎片（不包含遮罩碎片），原碎片移入垃圾桶")
        self.tabs.setCurrentIndex(0)
        self.cancel_restore_preview(silent=True)
    def on_import_degrade_source(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "匯入劣化來源圖", "", "PNG圖檔 (*.png)")
        if not fname:
            return
        try:
            im = Image.open(fname).convert("RGBA")
            arr = pil2np(im)
            self.degrade_source_img = arr
            self.degrade_preview_pending = None
            self._set_interference_source_mounted(False)
            self.degrade_panel.apply_export_btn.setEnabled(False)
            self.img_wrap.preview.set_image(arr)
            self.set_status("已載入劣化來源圖", True)
            self.degrade_panel.set_imported_filename(fname)
        except Exception as e:
            self.set_status(f"匯入失敗: {e}", False)

    def on_generate_degrade_preview_shared(self):
        if self.degrade_source_img is None:
            self.set_status("請先匯入劣化來源圖", False)
            return

        # 重新產生會捨棄尚未掛載的預覽；已掛載來源也會先解除。
        if self.degrade_preview_pending:
            reply = QtWidgets.QMessageBox.question(
                self,
                "重新產生劣化預覽",
                "目前已有劣化預覽。重新產生會解除現有掛載並取代預覽，確定要繼續嗎？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return
            self.degrade_preview_pending = None

        self._set_interference_source_mounted(False)

        settings = self.degrade_panel.get_settings()

        # 如果上一個還在跑，就先中斷
        if hasattr(self, "degrade_thread") and getattr(self, "degrade_thread", None) and self.degrade_thread.isRunning():
            self.degrade_thread.abort()
            self.degrade_thread.wait()

        self.degrade_panel.gen_preview_btn.setEnabled(False)
        self.degrade_panel.apply_export_btn.setEnabled(False)
        self.set_status("正在產生劣化預覽...", True)

        self.degrade_thread = DegradePreviewThread(self.degrade_source_img, settings)
        self.degrade_thread.progress.connect(lambda cur, tot, msg: self.set_status(f"{msg} ({cur}/{tot})", True))

        def finish(degraded):
            self.degrade_preview_pending = {'orig': self.degrade_source_img.copy(), 'degraded': degraded}
            self.img_wrap.preview.set_image(degraded)
            self.set_status("劣化預覽（尚未掛載）", True)
            self.degrade_panel.gen_preview_btn.setEnabled(True)
            self.degrade_panel.apply_export_btn.setEnabled(True)

        self.degrade_thread.result.connect(finish)
        self.degrade_thread.start()

    def on_restore_degrade_source(self):
        if getattr(self, 'degrade_source_img', None) is not None:
            self.img_wrap.preview.set_image(self.degrade_source_img)
            self.set_status("顯示原始劣化來源圖", True)

    def _set_interference_source_mounted(self, mounted):
        button = self.degrade_panel.apply_export_btn
        if mounted and self.degrade_preview_pending:
            self.interference_source_img = self.degrade_preview_pending['degraded'].copy()
        elif not mounted:
            self.interference_source_img = None

        if self.interference_source_img is not None:
            button.setText("已掛載干擾像素")
            button.setStyleSheet("background:#176b3a; color:white; font-weight:bold;")
        else:
            button.setText("掛載干擾像素")
            button.setStyleSheet("")

    def on_apply_degrade_source(self, *_):
        if not getattr(self, 'degrade_preview_pending', None):
            self._set_interference_source_mounted(False)
            QtWidgets.QMessageBox.information(self, "無預覽", "請先產生劣化預覽")
            return
        self._set_interference_source_mounted(True)
        self.set_status("已掛載劣化圖；後續拆解將用它產生干擾像素", True)

    def _cancel_degrade_preview_if_running(self):
        if getattr(self, "degrade_thread", None) and self.degrade_thread.isRunning():
            self.degrade_thread.abort()
            self.degrade_thread.wait()
            self.degrade_preview_pending = None
            self.set_status("已取消劣化預覽", False)
            self.degrade_panel.gen_preview_btn.setEnabled(True)
            self.degrade_panel.apply_export_btn.setEnabled(False)

    def fragment_clicked(self, item):
        name = item.text()
        self.img_wrap.previewing_fragment_name = name

        if self.restore_mode:
            # 進階模式顯示所有眼睛開啟的碎片，不以操作選取決定預覽。
            self.update_restore_preview()
            return

        if hasattr(self, 'interfere_images_dict') and self.interfere_images_dict and name in self.interfere_images_dict:
            img = self.interfere_images_dict[name]
            self.img_wrap.preview.set_image(img)
            self.set_status(f"{name} 干擾像素預覽中 尚未合成", ok=True)
            return

        img = self.fragment_data.get(name)
        if img is not None:
            self.img_wrap.preview.set_image(img)
            self.set_status(f"{name} 碎片預覽", ok=True)
        else:
            self.img_wrap.preview.set_image(None)
            self.set_status(f"{name} 無法預覽", ok=False)
