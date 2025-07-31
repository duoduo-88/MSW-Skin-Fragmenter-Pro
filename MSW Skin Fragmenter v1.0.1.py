import sys, os, time, io, random, zipfile
from collections import deque
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtGui import QShortcut, QKeySequence, QAction
from PIL import Image
import multiprocessing as mp
import numpy as np
import concurrent.futures
from PySide6.QtCore import QThread, Signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from PySide6.QtWidgets import QApplication

BG_OPTIONS = [
    ('#222222', '深灰'),
    ('#FFFFFF', '白'),
    ('#888888', '50%灰'),
    ('check', '透明網格')
]
RECYCLE_BIN_MAX = 99

def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)

def pil2np(im):
    if im.mode != "RGBA": im = im.convert("RGBA")
    return np.array(im)

def np2pil(arr):
    arr = arr.astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")

def pil2qpixmap(im):
    if isinstance(im, np.ndarray):
        arr = im.astype(np.uint8)
        h, w = arr.shape[:2]
        data = arr.tobytes()
        qimg = QtGui.QImage(data, w, h, QtGui.QImage.Format_RGBA8888)
        return QtGui.QPixmap.fromImage(qimg)
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    data = im.tobytes("raw", "RGBA")
    qimg = QtGui.QImage(data, im.width, im.height, QtGui.QImage.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(qimg)

def overlap_fill_density_blocks(
    fragments, main_img, mask_img, fill_percent, block_size, random_range, progress_cb=None
):
    if fill_percent <= 0 or not fragments or main_img is None:
        return fragments

    arr = main_img.copy()
    h, w = arr.shape[:2]
    if mask_img is not None:
        main_alpha255_mask = ((arr[...,3]==255) | (mask_img[...,3]==255))
    else:
        main_alpha255_mask = (arr[...,3]==255)
    coords = np.argwhere(main_alpha255_mask)
    total = len(coords)
    if total == 0:
        return fragments

    target_fill = int(total * fill_percent / 100)
    N = len(fragments)
    for i, frag in enumerate(fragments):
        filled = np.zeros((h, w), dtype=bool)
        fill_cnt = 0
        tries = 0
        while fill_cnt < target_fill and tries < target_fill * 10:
            y, x = coords[random.randrange(len(coords))]
            sz = random.randint(block_size, block_size * random_range)
            if x+sz > w or y+sz > h:
                tries += 1
                continue
            patch = arr[y:y+sz, x:x+sz]
            mask_patch = patch[...,3]==255
            frag_mask = ~filled[y:y+sz, x:x+sz] & mask_patch
            frag[y:y+sz, x:x+sz][frag_mask] = patch[frag_mask]
            filled[y:y+sz, x:x+sz][frag_mask] = True
            fill_cnt += np.sum(frag_mask)
            tries += 1
        frag[~main_alpha255_mask] = 0
        if progress_cb:
            progress_cb(i+1, N, f"重疊像素填充 {i+1}/{N}")
    return fragments
    
def overlap_fill_density_blocks_agg(
    fragments, main_img, mask_img, fill_percent, block_size, random_range, aggregation=1
):
    """
    補重疊像素（可控聚合度）：每片碎片隨機用方塊擴張原圖像素，分散or聚集。聚合度越高越集中。
    """
    if fill_percent <= 0 or not fragments or main_img is None:
        return fragments

    arr = main_img.copy()
    h, w = arr.shape[:2]
    if mask_img is not None:
        main_alpha255_mask = ((arr[...,3]==255) | (mask_img[...,3]==255))
    else:
        main_alpha255_mask = (arr[...,3]==255)
    coords = np.argwhere(main_alpha255_mask)
    total = len(coords)
    if total == 0:
        return fragments

    target_fill = int(total * fill_percent / 100)
    agg_ratio = min(max(aggregation, 1), 10) / 10.0

    for frag in fragments:
        filled = np.zeros((h, w), dtype=bool)
        fill_cnt = 0
        tries = 0
        cluster_centers = []
        while fill_cnt < target_fill and tries < target_fill * 10:
            use_cluster = cluster_centers and (random.random() < agg_ratio)
            if use_cluster:
                base_yx = random.choice(cluster_centers)
                dy, dx = np.random.randint(-block_size, block_size+1, 2)
                y, x = base_yx[0]+dy, base_yx[1]+dx
                if not (0 <= y < h and 0 <= x < w and main_alpha255_mask[y,x]):
                    y, x = coords[random.randrange(len(coords))]
            else:
                y, x = coords[random.randrange(len(coords))]

            sz = random.randint(block_size, block_size * random_range)
            if x+sz > w or y+sz > h or y<0 or x<0:
                tries += 1
                continue
            patch = arr[y:y+sz, x:x+sz]
            mask_patch = patch[...,3]==255
            frag_mask = ~filled[y:y+sz, x:x+sz] & mask_patch
            if np.sum(frag_mask) == 0:
                tries += 1
                continue
            frag[y:y+sz, x:x+sz][frag_mask] = patch[frag_mask]
            filled[y:y+sz, x:x+sz][frag_mask] = True
            fill_cnt += np.sum(frag_mask)
            for yy in range(y, y+sz):
                for xx in range(x, x+sz):
                    if frag_mask[yy-y, xx-x]:
                        cluster_centers.append((yy, xx))
            tries += 1
        frag[~main_alpha255_mask] = 0
    return fragments


def np2qpixmap(arr):
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    h, w = arr.shape[:2]
    data = arr.tobytes()
    qimg = QtGui.QImage(data, w, h, QtGui.QImage.Format_RGBA8888)
    return QtGui.QPixmap.fromImage(qimg)

def numpy_alpha_composite(base, img):
    base = base.astype(np.float32)
    img = img.astype(np.float32)
    alpha_top = img[...,3:4]/255.0
    alpha_base = base[...,3:4]/255.0
    out_rgb = img[...,:3]*alpha_top + base[...,:3]*alpha_base*(1-alpha_top)
    out_alpha = alpha_top + alpha_base*(1-alpha_top)
    out = np.empty_like(base)
    out[...,:3] = out_rgb / (out_alpha + 1e-6)
    out[...,3] = out_alpha[...,0]*255
    return out.astype(np.uint8)
    
def apply_interfere_masked(orig, interfere, main_mask):
    out = orig.copy()
    valid = (main_mask) & (interfere[...,3] > 0)
    out[valid] = numpy_alpha_composite(orig[valid], interfere[valid])
    return out    

def apply_mask_alpha(frag_img, mask_img):
    if mask_img is None:
        return frag_img
    frag = frag_img.copy()
    frag[..., 3] = np.minimum(frag[..., 3], mask_img[..., 3])
    return frag

def ellipsis_middle(text, maxlen=28):
    if len(text) <= maxlen:
        return text
    name, ext = os.path.splitext(text)
    if len(ext) > 5: ext = ext[:5] + "..."
    keep = maxlen - len(ext) - 3
    if keep < 8:
        return text[:maxlen-3] + "..."
    return name[:keep//2] + "..." + name[-keep//2:] + ext

def build_interfere_block_pool(src_arr, block_size=10, random_range=2, pool_size=400):
    h, w = src_arr.shape[:2]
    alpha = src_arr[..., 3]
    valid_coords = np.argwhere(alpha > 0)
    blocks = []
    for _ in range(pool_size * 2):
        y, x = valid_coords[random.randrange(len(valid_coords))]
        sz = random.randint(block_size, block_size * random_range)
        if x+sz > w or y+sz > h:
            continue
        patch = src_arr[y:y+sz, x:x+sz]
        if np.sum(patch[..., 3] > 0) > 0.01 * sz * sz:
            blocks.append(patch.copy())
        if len(blocks) >= pool_size:
            break
    return blocks

def gen_multi_overlap_interfere(
    frag_img,
    block_pool,
    coverage=1.3,
    max_try=8,
    allow_overlap=False,
    main_alpha255_mask=None
):
    h, w = frag_img.shape[:2]
    mask = (frag_img[..., 3] == 255)
    valid_yx = np.argwhere(mask)
    total_valid = len(valid_yx)
    if total_valid == 0 or not block_pool:
        return np.zeros_like(frag_img)

    if main_alpha255_mask is None:
        raise ValueError("main_alpha255_mask is required for constrained interference")

    ys, xs = np.where(main_alpha255_mask)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros_like(frag_img)
    min_x, max_x = xs.min(), xs.max()
    min_y, max_y = ys.min(), ys.max()
    required_box = (min_x, min_y, max_x, max_y)

    target_pixels = int(total_valid * coverage)
    out = np.zeros_like(frag_img)
    pasted = 0
    tries = 0
    pasted_mask = np.zeros((h, w), dtype=bool)

    while pasted < target_pixels and tries < target_pixels * max_try:
        patch = random.choice(block_pool)
        bh, bw = patch.shape[:2]
        y0, x0 = valid_yx[random.randrange(len(valid_yx))]
        y = y0 - random.randint(0, bh - 1)
        x = x0 - random.randint(0, bw - 1)

        if y < 0 or x < 0 or y + bh > h or x + bw > w:
            tries += 1
            continue

        patch_rect = (x, y, x + bw - 1, y + bh - 1)
        sub_mask = main_alpha255_mask[y:y+bh, x:x+bw]
        if np.sum(sub_mask) < int(bh * bw * 0.2):
            tries += 1
            continue

        if not allow_overlap and np.any(pasted_mask[y:y + bh, x:x + bw] & (patch[..., 3] > 0)):
            tries += 1
            continue

        patch_alpha = patch[..., 3:4] / 255.0
        out[y:y + bh, x:x + bw, :3] = (
            patch[..., :3] * patch_alpha + out[y:y + bh, x:x + bw, :3] * (1 - patch_alpha)
        ).astype(np.uint8)
        out[y:y + bh, x:x + bw, 3] = np.maximum(
            out[y:y + bh, x:x + bw, 3], patch[..., 3]
        )
        pasted += np.sum(patch[..., 3] > 0)
        pasted_mask[y:y + bh, x:x + bw] |= (patch[..., 3] > 0)
        tries += 1

    return out
    
def gen_interfere_worker(args):
    frag_name, frag_arr, main_img_arr, settings, split_result0 = args
    block_pool = build_interfere_block_pool(
        main_img_arr,
        settings['block_size'],
        settings['random_range'],
        pool_size=300
    )
    main_mask = (split_result0[..., 3] == 255)
    interfere = gen_multi_overlap_interfere(
        frag_arr, block_pool,
        coverage=settings['density'] * 1.8,
        allow_overlap=settings.get('allow_overlap', False),
        main_alpha255_mask=main_mask
    )
    return frag_name, interfere

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
        self.tipBox = QtWidgets.QLabel(self.tip, None, QtCore.Qt.ToolTip)
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
                mw.img_wrap.preview.set_image(mw.mask_img)
                mw.set_status("遮罩預覽", True)
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
        self.offset = QtCore.QPoint(0,0)
        self._drag_start = None
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #222222;")
        self._max_scale = 2.0
        self.trash_highlight = False
        self.overlay_mode = False
        self.overlay_base = None
    def set_bg(self, idx): self.bg_idx = idx; self.repaint()
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
        self.repaint()
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
        self.repaint()
    def paintEvent(self, ev):
        super().paintEvent(ev)
        painter = QtGui.QPainter(self)
        if BG_OPTIONS[self.bg_idx][0] == "check":
            self._draw_checkerboard(painter)
        else:
            painter.fillRect(self.rect(), QtGui.QColor(BG_OPTIONS[self.bg_idx][0]))
        if self.qimg:
            scale = min(self._scale, 2.0)
            scaled = self.qimg.scaled(
                self.qimg.width() * scale, self.qimg.height() * scale,
                QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
            pt = QtCore.QPoint(
                (self.width() - scaled.width()) // 2 + self.offset.x(),
                (self.height() - scaled.height()) // 2 + self.offset.y())
            painter.drawPixmap(pt, scaled)
        p = self._parent.parent()
        is_trash_tab = hasattr(p, 'tabs') and p.tabs.currentWidget() == p.trash_tab
        if is_trash_tab:
            if self.trash_highlight:
                pen = QtGui.QPen(QtGui.QColor(255,0,0), 3)
                pen.setJoinStyle(QtCore.Qt.MiterJoin)
                painter.setPen(pen)
                painter.drawRect(self.rect().adjusted(2,2,-3,-3))
        else:
            if self.qimg is None:
                font = painter.font()
                font.setPointSize(22)
                font.setBold(True)
                painter.setFont(font)
                if BG_OPTIONS[self.bg_idx][0] == "check":
                    pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
                else:
                    pen = QtGui.QPen(QtGui.QColor(255, 255, 255))
                painter.setPen(pen)
                painter.drawText(self.rect(), QtCore.Qt.AlignCenter, "當前無預覽圖")
    def _draw_checkerboard(self, painter):
        tile = 16
        cols = self.width() // tile + 2
        rows = self.height() // tile + 2
        for y in range(rows):
            for x in range(cols):
                color = QtGui.QColor(220,220,220) if (x+y)%2==0 else QtGui.QColor(160,160,160)
                painter.fillRect(x*tile, y*tile, tile, tile, color)
    def wheelEvent(self, ev):
        if not self.qimg:
            return

        old_scale = self._scale
        mouse_pos = ev.position() if hasattr(ev, 'position') else ev.pos()
        mouse_pos = QtCore.QPointF(mouse_pos)

        img_w = self.qimg.width() * old_scale
        img_h = self.qimg.height() * old_scale
        img_offset_x = (self.width() - img_w) // 2 + self.offset.x()
        img_offset_y = (self.height() - img_h) // 2 + self.offset.y()

        img_px = (mouse_pos.x() - img_offset_x) / old_scale
        img_py = (mouse_pos.y() - img_offset_y) / old_scale

        delta = ev.angleDelta().y() / 120.0
        scale_factor = 1.15 ** delta
        self._scale = max(0.05, min(2.0, self._scale * scale_factor))

        new_img_w = self.qimg.width() * self._scale
        new_img_h = self.qimg.height() * self._scale
        new_img_offset_x = mouse_pos.x() - img_px * self._scale
        new_img_offset_y = mouse_pos.y() - img_py * self._scale

        self.offset = QtCore.QPoint(int(new_img_offset_x - (self.width() - new_img_w) // 2),
                                    int(new_img_offset_y - (self.height() - new_img_h) // 2))
        self.repaint()
        if self._parent:
            self._parent.update_zoom_display()

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self._drag_start = (ev.pos(), QtCore.QPoint(self.offset))
            self.setCursor(QtCore.Qt.ClosedHandCursor)
    def mouseMoveEvent(self, ev):
        if self._drag_start:
            delta = ev.pos() - self._drag_start[0]
            self.offset = self._drag_start[1] + delta
            self.repaint()
    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self._drag_start = None
            self.setCursor(QtCore.Qt.ArrowCursor)
    def mouseDoubleClickEvent(self, ev):
        self._scale = 1.0
        self.offset = QtCore.QPoint(0,0)
        self.repaint()
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
        for i, (c, name) in enumerate(BG_OPTIONS):
            btn = QtWidgets.QPushButton(name)
            btn.setStyleSheet("background:#444; color:#fff; padding:2px 12px;")
            def make_bg_handler(idx):
                def handler():
                    self.preview.set_bg(idx)
                    if self.overlay_btn.isChecked():
                        self.overlay_btn.setChecked(False)
                        if hasattr(self.parent(), "restore_overlay_off"):
                            self.parent().restore_overlay_off()
                return handler
            btn.clicked.connect(make_bg_handler(i))
            hz.addWidget(btn)
        self.overlay_btn = QtWidgets.QPushButton("重疊預覽")
        self.overlay_btn.setCheckable(True)
        self.overlay_btn.setStyleSheet("background:#444; color:#fff; padding:2px 12px;")
        self.overlay_btn.setVisible(False)
        self.overlay_btn.clicked.connect(self.toggle_overlay)
        hz.addWidget(self.overlay_btn, alignment=QtCore.Qt.AlignLeft)
        hz.addStretch()
        self.status_lbl = QtWidgets.QLabel("")
        self.status_lbl.setStyleSheet("color:#0f0;font-size:15px;min-width:140px;max-width:260px;")
        hz.addWidget(self.status_lbl)
        lay.addLayout(hz)
        self.preview._parent = self
        self.update_zoom_display()
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
class SplitThread(QtCore.QThread):
    update_progress = QtCore.Signal(int, int, str)
    result = QtCore.Signal(list)
    def __init__(self, main_img, mask_img, count, blocksize, rand_factor):
        super().__init__()
        self.main_img = main_img
        self.mask_img = mask_img
        self.count = count
        self.blocksize = blocksize
        self.rand_factor = rand_factor
        self._abort = False
    def abort(self): self._abort = True
    def run(self):
        try:
            self.update_progress.emit(0, 7, "計算分割區塊...")
            t0 = time.time()
            rgba = pil2np(self.main_img)
            w, h = rgba.shape[1], rgba.shape[0]
            valid_coords = set()
            if self.mask_img is not None:
                mask_rgba = pil2np(self.mask_img)
                if mask_rgba.shape[:2] != (h, w):
                    self.result.emit([])
                    return
                for y in range(h):
                    for x in range(w):
                        if mask_rgba[y, x, 3] > 0:
                            valid_coords.add((x, y))
            else:
                for y in range(h):
                    for x in range(w):
                        if rgba[y, x, 3] > 0:
                            valid_coords.add((x, y))
            self.update_progress.emit(1, 7, "開始分割...")
            if not valid_coords:
                self.result.emit([])
                return
            sx, sy, ex, ey = 0, 0, w, h
            self.update_progress.emit(2, 7, "正在產生分割區塊...")
            blocks = []
            y = sy
            while y < ey:
                min_h = self.blocksize
                max_h = self.blocksize * self.rand_factor
                bh = random.randint(min_h, max_h)
                if y + bh > ey or ey - y < min_h:
                    bh = ey - y
                x = sx
                while x < ex:
                    min_w = self.blocksize
                    max_w = self.blocksize * self.rand_factor
                    bw = random.randint(min_w, max_w)
                    if x + bw > ex or ex - x < min_w:
                        bw = ex - x
                    has_overlap = False
                    for yy in range(y, y + bh):
                        for xx in range(x, x + bw):
                            if (xx, yy) in valid_coords:
                                has_overlap = True
                                break
                        if has_overlap:
                            break
                    if has_overlap:
                        blocks.append((x, y, bw, bh))
                    x += bw
                y += bh
            self.update_progress.emit(3, 7, "正在產生碎片（第1步）...")
            fragment_blocks = [[] for _ in range(self.count)]
            for block in blocks:
                fid = random.randint(0, self.count - 1)
                fragment_blocks[fid].append(block)
            self.update_progress.emit(4, 7, "正在分配像素到碎片...")
            pixel_fragments = [[set() for _ in range(w)] for _ in range(h)]
            for fid, blocks_for_fid in enumerate(fragment_blocks):
                for block in blocks_for_fid:
                    x, y, bw, bh = block
                    for yy in range(y, y + bh):
                        for xx in range(x, x + bw):
                            if rgba[yy, xx, 3] > 0:
                                pixel_fragments[yy][xx].add(fid)
            self.update_progress.emit(5, 7, "處理孤立像素...")
            for x, y in valid_coords:
                if len(pixel_fragments[y][x]) == 0:
                    fid = random.randint(0, self.count - 1)
                    pixel_fragments[y][x].add(fid)
            self.update_progress.emit(6, 7, "產生碎片圖像...")
            fragment_imgs = [np.zeros((h, w, 4), dtype=np.uint8) for _ in range(self.count)]
            for y in range(h):
                for x in range(w):
                    pix = rgba[y, x]
                    for fid in pixel_fragments[y][x]:
                        fragment_imgs[fid][y, x] = pix
            self._last_time = time.time() - t0
            self.result.emit(fragment_imgs)
        except Exception as e:
            import traceback
            print(f"[SplitThread Exception]: {e}")
            traceback.print_exc()
            self.result.emit([])   

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
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        vbox.addWidget(self.list, stretch=1)
        btnrow = QtWidgets.QHBoxLayout()
        self.restore_btn = QtWidgets.QPushButton("復原勾選的碎片")
        self.restore_btn.clicked.connect(self.restore_selected)
        self.clear_btn = QtWidgets.QPushButton("清空垃圾桶")
        self.clear_btn.clicked.connect(self.clear_trash)
        btnrow.addWidget(self.restore_btn)
        btnrow.addWidget(self.clear_btn)
        vbox.addLayout(btnrow)
        self.list.itemClicked.connect(self.on_trash_item_clicked)
        self.refresh()
    def refresh(self):
        self.info_lbl.setText(f"垃圾碎片 {len(self.recycle_bin)}個")
        self.list.clear()
        for name, img in self.recycle_bin:
            lw = QtWidgets.QListWidgetItem()
            lw.setText(name)
            lw.setFlags(lw.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            lw.setCheckState(QtCore.Qt.Unchecked)
            self.list.addItem(lw)
    def on_trash_item_clicked(self, item):
        idx = self.list.row(item)
        if 0 <= idx < len(self.recycle_bin):
            name, img = self.recycle_bin[idx]
            self.current_highlight_img = img
            self.parent.img_wrap.preview.set_image(img, trash_highlight=True)
        else:
            self.current_highlight_img = None
            self.parent.img_wrap.preview.set_image(None, trash_highlight=False)
    def clear_highlight(self):
        self.current_highlight_img = None
        self.parent.img_wrap.preview.set_image(None, trash_highlight=False)
    def restore_selected(self):
        checked_items = []
        for i in range(self.list.count()):
            item = self.list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                checked_items.append((i, item))
        if not checked_items: return
        items_to_restore = []
        for idx, item in checked_items:
            items_to_restore.append((idx, self.recycle_bin[idx]))
        for idx, (name, img) in sorted(items_to_restore, key=lambda x: -x[0]):
            self.parent.restore_from_trash(name, img)
            del self.recycle_bin[idx]
        self.refresh()
        self.parent.refresh_fragment_list()
        self.clear_highlight()
    def clear_trash(self):
        self.recycle_bin.clear()
        self.refresh()
        self.clear_highlight()

class OverlapThread(QThread):
    progress = Signal(int, int, str)
    result = Signal(list)
    def __init__(self, images, main_img, mask_img, fill_pct, block_size, rand_range, aggregation=1):
        super().__init__()
        self.images = [img.copy() for img in images]
        self.main_img = main_img.copy() if main_img is not None else None
        self.mask_img = mask_img.copy() if mask_img is not None else None
        self.fill_pct = fill_pct
        self.block_size = block_size
        self.rand_range = rand_range
        self.aggregation = aggregation
        self._abort = False
    def abort(self):
        self._abort = True
        
    def run(self):
        def progress_cb(cur, total, msg):
            self.progress.emit(cur, total, msg)
        def abort_cb():
            return getattr(self, '_abort', False)
        try:
            result = apply_overlap_to_all_fragments_mp(
                self.images, self.main_img, self.mask_img,
                self.fill_pct, self.block_size, self.rand_range,
                progress_cb=progress_cb, abort_cb=abort_cb
            )
            self.result.emit(result)
        except Exception as e:
            self.result.emit([])   

class FragmentListWidget(QtWidgets.QListWidget):
    def __init__(self, mainwindow, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mainwindow = mainwindow
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            super().dragEnterEvent(event)
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            super().dragMoveEvent(event)
    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            files = [url.toLocalFile() for url in event.mimeData().urls() if url.toLocalFile().lower().endswith('.png')]
            if files:
                self.mainwindow.import_fragments_from_files(files)
                event.accept()
                return
        super().dropEvent(event)

class InterferePanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QFormLayout(self)

        self.block_size = QtWidgets.QSpinBox()
        self.block_size.setRange(1, 30)
        self.block_size.setValue(3)
        self.block_size.setFixedWidth(70)
        bs_tip = QHelpButton(
    "設定每一個干擾像素塊的基本邊長(px)，越大則每塊越大。\n"
    "\n"
    "優點：大尺寸提升覆蓋速度。\n"
    "\n"
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
        self.random_range.setValue(1)
        self.random_range.setFixedWidth(70)
        rr_tip = QHelpButton(
    "決定干擾像素塊的尺寸隨機變動範圍，1為固定，數字越大越亂。\n"
    "\n"
    "優點：隨機性高提升防還原性。\n"
    "\n"
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
        self.density.setValue(70)
        self.density_lbl = QtWidgets.QLabel("干擾密度: 70%")
        density_tip = QHelpButton(
    "決定干擾像素填滿目標區域的比例，數字越高，干擾覆蓋越密集。\n"
    "\n"
    "優點：密度高可大幅阻礙還原。\n"
    "\n"
    "缺點：太高會讓檔案龐大且難以正常辨識。"
)
        density_row = QtWidgets.QHBoxLayout()
        density_row.addWidget(self.density_lbl)
        density_row.addWidget(density_tip)
        density_row.addStretch()
        lay.addRow(density_row)
        lay.addRow(self.density)
        self.density.valueChanged.connect(lambda v: self.density_lbl.setText(f"干擾密度: {v}%"))

        self.alpha_min = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha_min.setRange(1, 100)
        self.alpha_min.setValue(5)
        self.alpha_min_lbl = QtWidgets.QLabel(f"干擾像素的不透明面積最小數值: {self.alpha_min.value()}%")
        amn_tip = QHelpButton(
    "設定可被選入干擾素材池的像素塊，必須覆蓋的最小不透明比例，避免選到太透明的雜訊。\n"
    "\n"
    "優點：濾除雜訊，保證干擾有效。\n"
    "\n"
    "缺點：設定過高會排除大部分素材，干擾池不足。"
)
        amn_row = QtWidgets.QHBoxLayout()
        amn_row.addWidget(self.alpha_min_lbl)
        amn_row.addWidget(amn_tip)
        amn_row.addStretch()
        lay.addRow(amn_row)
        lay.addRow(self.alpha_min)
        self.alpha_min.valueChanged.connect(
            lambda v: self.alpha_min_lbl.setText(f"干擾像素的不透明面積最小數值: {v}%"))

        self.alpha_max = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha_max.setRange(1, 100)
        self.alpha_max.setValue(100)
        self.alpha_max_lbl = QtWidgets.QLabel("干擾像素的不透明面積最大數值: 100%")
        amx_tip = QHelpButton(
    "設定可被選入干擾素材池的像素塊，必須覆蓋的最大不透明比例。可用來排除太實心的大片塊。\n"
    "\n"
    "優點：排除過大塊避免影響外觀。\n"
    "\n"
    "缺點：過小則素材有限，干擾效果變差。"
)
        amx_row = QtWidgets.QHBoxLayout()
        amx_row.addWidget(self.alpha_max_lbl)
        amx_row.addWidget(amx_tip)
        amx_row.addStretch()
        lay.addRow(amx_row)
        lay.addRow(self.alpha_max)
        self.alpha_max.valueChanged.connect(
            lambda v: self.alpha_max_lbl.setText(f"干擾像素的不透明面積最大數值: {v}%"))

        self.allow_overlap_cb = QtWidgets.QCheckBox("允許干擾像素重疊")
        self.allow_overlap_cb.setChecked(True)
        overlap_tip = QHelpButton(
    "允許多個干擾像素塊彼此重疊。若關閉，干擾像素會盡量不交錯，但可能會減少填充面積。\n"
    "\n"
    "優點：允許重疊可提升覆蓋效率與密度。\n"
    "\n"
    "缺點：重疊過多時，部分區塊可能異常突出。"
)
        overlap_row = QtWidgets.QHBoxLayout()
        overlap_row.addWidget(self.allow_overlap_cb)
        overlap_row.addWidget(overlap_tip)
        overlap_row.addStretch()
        lay.addRow(overlap_row)

        self.gen_btn = QtWidgets.QPushButton("產生干擾像素圖")
        self.apply_btn = QtWidgets.QPushButton("合成到碎片")
        btnrow = QtWidgets.QHBoxLayout()
        btnrow.addWidget(self.apply_btn)
        lay.addRow(self.gen_btn)
        lay.addRow(btnrow)

    def get_settings(self):
        minv = min(self.alpha_min.value(), self.alpha_max.value())
        maxv = max(self.alpha_min.value(), self.alpha_max.value())
        return dict(
            block_size=self.block_size.value(),
            random_range=self.random_range.value(),
            density=self.density.value()/100,
            src_alpha_min_pct=minv,
            src_alpha_max_pct=maxv,
            allow_overlap=self.allow_overlap_cb.isChecked()
        )
class InterfereThread(QtCore.QThread):
    update_progress = QtCore.Signal(int, int, str)
    result = QtCore.Signal(dict)

    def __init__(self, fragment_data, main_img, split_result, settings):
        super().__init__()
        self.fragment_data = fragment_data
        self.main_img = main_img
        self.split_result = split_result
        self.settings = settings

    def run(self):
        try:
            result = {}
            frag_names = list(self.fragment_data.keys())
            N = len(frag_names)
            for i, name in enumerate(frag_names):
                frag = self.fragment_data[name]
                block_pool = build_interfere_block_pool(
                    self.main_img,
                    self.settings['block_size'],
                    self.settings['random_range'],
                    pool_size=300
                )
                main_mask = (self.split_result[0][..., 3] == 255)
                self.update_progress.emit(i+1, N, f"產生干擾像素({name})")
                result[name] = gen_multi_overlap_interfere(
                    frag, block_pool,
                    coverage=self.settings['density'] * 1.8,
                    allow_overlap=self.settings.get('allow_overlap', False),
                    main_alpha255_mask=main_mask
                )
            self.result.emit(result)
        except Exception as e:
            import traceback
            print(f"[InterfereThread Exception]: {e}")
            traceback.print_exc()
            self.result.emit({})        

class InterfereGenThread(QThread):
    progress = Signal(int, int, str)
    result = Signal(dict)
    def __init__(self, fragment_data, main_img, split_result, settings):
        super().__init__()
        self.fragment_data = fragment_data
        self.main_img = main_img
        self.split_result = split_result
        self.settings = settings
    def run(self):
        result = {}
        args_list = []
        for name, frag in self.fragment_data.items():
            args_list.append((name, frag, self.main_img, self.settings, self.split_result[0]))
        total = len(args_list)
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(gen_interfere_worker, arg): arg[0] for arg in args_list}
            done_cnt = 0
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    r_name, interfere = fut.result()
                except Exception as e:
                    print(f"產生 {name} 干擾像素失敗: {e}")
                    continue
                result[r_name] = interfere
                done_cnt += 1
                self.progress.emit(done_cnt, total, f"干擾像素完成({done_cnt}/{total})")
        self.result.emit(result)
        
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.checkbox_items = []
        self.setWindowTitle("MSW造型防盜拆解工具 MSW Skin Fragmenter 高性能版 v1.0.1")
        self.setMinimumSize(1200, 700)
        self.setStyleSheet("background:#232323; color:#eee; font-size:15px;")
        self.img_wrap = ImagePreviewWrap(self)
        self.main_img = None
        self.main_img_path = ""
        self.mask_img = None
        self.mask_img_path = ""
        self.split_result = []
        self.fragment_data = {}
        self.fragment_order = []
        self.restore_mode = False
        self.recycle_bin = deque(maxlen=RECYCLE_BIN_MAX)
        self._initial_snapshot = None
        self.interfere_images_dict = {} 
        self.initUI()
        self.setup_shortcuts()
        self.tabs.currentChanged.connect(self.on_tab_changed)
        self.fragment_list.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        self.interfere_panel.gen_btn.clicked.connect(self.on_gen_interfere_img)
        self.interfere_panel.apply_btn.clicked.connect(self.apply_interfere_to_fragments)
      
    def progress_step(self, step, total, msg):
        pct = int(step / total * 100) if total else 0
        if msg:
            self.set_status(f"{msg} ({pct}%)", True)
        else:
            self.set_status(f"拆解中... {pct}%", True)
    def split(self):
        self.progress_step("載入主圖...")
        self.progress_step("檢查參數...")
        self.progress_step("開始分割...")
        self.progress_step("計算分割區塊...")
        self.progress_step(f"正在產生碎片（第1步）...")
        self.progress_step("分配重疊像素（聚合度: %d）..." % self.aggregation_input.value())
        self.progress_step("合併碎片...")
        self.progress_step(f"拆解完成，花費 {sec}秒", ok=True)        

    def initUI(self):
        main = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.img_wrap, stretch=1)
        right = QtWidgets.QVBoxLayout()
        ff = QtWidgets.QFormLayout()
        self.main_btn = QtWidgets.QPushButton("選擇主圖")
        self.main_btn.clicked.connect(self.load_main)
        self.main_file_lbl = ClickableFileLabel(self, 'main')
        main_row = QtWidgets.QHBoxLayout()
        main_row.addWidget(self.main_btn)
        main_row.addWidget(QHelpButton("請上傳含有透明區的 PNG 檔案作為主圖進行切割。透明像素將不會參與分割。"))
        ff.addRow("主圖：", main_row)
        ff.addRow("", self.main_file_lbl)
        self.mask_btn = QtWidgets.QPushButton("選擇遮罩 (可選)")
        self.mask_btn.clicked.connect(self.load_mask)
        self.mask_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        self.del_mask_btn = QtWidgets.QPushButton("移除遮罩")
        self.del_mask_btn.clicked.connect(self.del_mask)
        self.del_mask_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        self.mask_crop_cb = QtWidgets.QCheckBox("不溢出遮罩範圍")
        self.mask_crop_cb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.mask_crop_cb.setChecked(True)
        help_btn = QHelpButton(
            "若有載入遮罩圖，僅遮罩圖中不透明的區域會參與碎片分割，碎片內容可以超出遮罩範圍，但每片至少有部分覆蓋遮罩。"
            "\n"
            "\n勾選「不溢出遮罩範圍」後，拆解完成時會立刻將所有碎片以遮罩圖透明度再次裁切，僅保留與遮罩重疊的像素，其他區域會變成透明。"
            "\n溢出遮罩的情況有可能會導致半透明圖像還原時異常，除非使用者確定無遮罩裁切溢出的範圍無使用半透明像素，否則不建議取消勾選。"
            "\n"
            "\n遮罩圖需為 PNG，且大小須與主圖完全一致。"
        )
        help_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        mask_row = QtWidgets.QHBoxLayout()
        mask_row.addWidget(self.mask_btn)
        mask_row.addWidget(self.del_mask_btn)
        mask_row.addWidget(self.mask_crop_cb)
        mask_row.addWidget(help_btn)

        ff.addRow("遮罩：", mask_row)
        self.mask_file_lbl = ClickableFileLabel(self, 'mask')
        ff.addRow("", self.mask_file_lbl)
        right.addLayout(ff)
        self.num_input = QtWidgets.QSpinBox(); self.num_input.setRange(1,10); self.num_input.setValue(3)
        self.block_input = QtWidgets.QSpinBox(); self.block_input.setRange(1,30); self.block_input.setValue(5)
        self.rand_input = QtWidgets.QSpinBox(); self.rand_input.setRange(1,100); self.rand_input.setValue(1)
        ff2 = QtWidgets.QFormLayout()
        row1 = QtWidgets.QHBoxLayout(); row1.addWidget(self.num_input); row1.addWidget(QHelpButton(
    "決定要將圖片分割成幾個碎片，通常越多越難重組與逆向。\n"
    "\n"
    "優點：數量越多，安全性提升、盜用困難度增加。\n"
    "\n"
    "缺點：碎片數過多會導致管理困難、記憶體消耗變高。"
))
        ff2.addRow("拆分張數(1~10)：", row1)
        row2 = QtWidgets.QHBoxLayout(); row2.addWidget(self.block_input); row2.addWidget(QHelpButton(
    "定義分割的最小區塊（鏤空最小洞）的尺寸。數字越大，每個分割塊越大。單位：px\n"
    "\n"
    "優點：區塊大可提升運算速度、減少碎片數。\n"
    "\n"
    "缺點：太大會降低隱蔽度，過小可能造成卡頓。"
))
        ff2.addRow("方塊尺寸(1~30)：", row2)
        row3 = QtWidgets.QHBoxLayout(); row3.addWidget(self.rand_input); row3.addWidget(QHelpButton(
    "區塊尺寸的隨機倍率範圍，1 代表所有區塊尺寸固定，2 代表區塊尺寸會隨機在設定值的 1~2 倍間變化。\n"
    "\n"
    "優點：提高碎片形狀隨機性，難以預測與還原。\n"
    "\n"
    "缺點：過高會造成計算量大增與碎片難以辨認。"
))
        ff2.addRow("尺寸隨機度(1~100)：", row3)
        self.overlap_pixel_input = QtWidgets.QSpinBox()
        self.overlap_pixel_input.setRange(0, 100)
        self.overlap_pixel_input.setValue(0)
        row5 = QtWidgets.QHBoxLayout()
        row5.addWidget(self.overlap_pixel_input)
        row5.addWidget(QHelpButton(
    "拆解後於鏤空區補原圖像素作為重疊像素。\n數值為聯集不透明像素的比例，填補到2~N片隨機碎片。\n"
    "\n"
    "優點：增加還原難度，讓每片有干擾。\n"
    "\n"
    "缺點：比例過高會導致效能大幅下降、檔案變大。"
))
        ff2.addRow("重疊像素比(0~100%)：", row5)
        self.aggregation_input = QtWidgets.QSpinBox()
        self.aggregation_input.setRange(1, 10)
        self.aggregation_input.setValue(1)
        row6 = QtWidgets.QHBoxLayout()
        row6.addWidget(self.aggregation_input)
        row6.addWidget(QHelpButton(
    "調整回補的重疊像素聚集程度。1=最分散，10=最密集，預設1\n"
    "\n"
    "優點：可調整碎片間重疊區域型態，提升反逆向性。\n"
    "\n"
    "缺點：極端值可能造成運算異常或不自然分佈。"
))
        ff2.addRow("重疊像素聚合(1~10)：", row6)
        right.addLayout(ff2)
        cth = QtWidgets.QHBoxLayout()
        self.split_btn = QtWidgets.QPushButton("執行拆解")
        self.split_btn.clicked.connect(self.split)
        cth.addWidget(self.split_btn)
        self.abort_btn = QtWidgets.QPushButton("終止並重啓")
        self.abort_btn.clicked.connect(self.abort_split)
        cth.addWidget(self.abort_btn)
        self.save_btn = QtWidgets.QPushButton("還原初始分割")
        self.save_btn.clicked.connect(self.restore_initial_state)
        cth.addWidget(self.save_btn)
        right.addLayout(cth)
        self.tabs = QtWidgets.QTabWidget()
        self.fragment_list = FragmentListWidget(self)
        self.fragment_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        fragment_page = QtWidgets.QWidget()
        vbox = QtWidgets.QVBoxLayout(fragment_page)
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
        self.adv_panel = QtWidgets.QWidget()
        adv_grid = QtWidgets.QGridLayout(self.adv_panel)
        self.merge_btn = QtWidgets.QPushButton("合併碎片")
        self.merge_btn.clicked.connect(lambda: self.merge_checked_restore_fragments(self.get_checked_fragments()))
        adv_grid.addWidget(self.merge_btn, 0, 0)
        self.copy_btn = QtWidgets.QPushButton("複製碎片")
        self.copy_btn.clicked.connect(lambda: self.copy_checked_restore_fragments(self.get_checked_fragments()))
        adv_grid.addWidget(self.copy_btn, 0, 1)
        self.delete_btn = QtWidgets.QPushButton("刪除碎片")
        self.delete_btn.setStyleSheet("background:#b33; color:#fff; font-weight:bold;")
        self.delete_btn.clicked.connect(lambda: self.delete_checked_restore_fragments(self.get_checked_fragments()))
        adv_grid.addWidget(self.delete_btn, 0, 2)
        self.rename_adv_btn = QtWidgets.QPushButton("重新命名")
        self.rename_adv_btn.clicked.connect(lambda: self.rename_fragment_by_name(self.get_checked_fragments()[0]) if len(self.get_checked_fragments()) == 1 else self.batch_rename_fragments())
        adv_grid.addWidget(self.rename_adv_btn, 1, 0)
        self.import_btn = QtWidgets.QPushButton("匯入碎片")
        self.import_btn.clicked.connect(self.import_fragments_btn)
        adv_grid.addWidget(self.import_btn, 1, 1)
        self.export_menu = self._make_export_menu() 
        self.export_btn = QtWidgets.QPushButton("匯出碎片")
        self.export_btn.setMenu(self.export_menu)
        adv_grid.addWidget(self.export_btn, 1, 2)
        self.export_btn.setMenu(self._make_export_menu())
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
        self.interfere_panel = InterferePanel(self)
        interfere_tab = QtWidgets.QWidget()
        interfere_layout = QtWidgets.QVBoxLayout(interfere_tab)
        interfere_layout.addWidget(self.interfere_panel)
        self.tabs.addTab(interfere_tab, "干擾像素")
        self.trash_tab = TrashCanWidget(self, self.recycle_bin)
        self.tabs.addTab(self.trash_tab, "垃圾桶")
        right.addWidget(self.tabs, stretch=1)
        self.disclaimer_lbl1 = QtWidgets.QLabel("本工具僅供技術交流與學術用途，不保證碎片不可被還原。")
        self.disclaimer_lbl2 = QtWidgets.QLabel("使用者需自行承擔所有風險。")
        self.disclaimer_lbl3 = QtWidgets.QLabel("© 2025 DuoDuo. 開源授權：MIT License")

        for lbl in [self.disclaimer_lbl1, self.disclaimer_lbl2, self.disclaimer_lbl3]:
            lbl.setStyleSheet("color:#aaa; font-size:11px; margin:1px 0;")
            lbl.setAlignment(QtCore.Qt.AlignCenter)

        right.addWidget(self.disclaimer_lbl1)
        right.addWidget(self.disclaimer_lbl2)
        right.addWidget(self.disclaimer_lbl3)
        main.addLayout(left, stretch=2)
        main.addLayout(right, stretch=1)
        self.switch_panel(False)
        
    def _make_export_menu(self):
        menu = QtWidgets.QMenu(self)
        action_sel = QAction("匯出選擇碎片", self)
        action_sel.triggered.connect(self.export_selected_fragments)
        action_all = QAction("匯出全部碎片", self)
        action_all.triggered.connect(self.export_all_fragments_zip)
        menu.addAction(action_sel)
        menu.addAction(action_all)
        return menu

    def switch_panel(self, adv):
        self.normal_panel.setVisible(not adv)
        self.adv_panel.setVisible(adv)

    def get_checked_fragments(self):
        return [self.fragment_list.item(i).text()
                for i in range(self.fragment_list.count())
                if self.fragment_list.item(i).checkState() == QtCore.Qt.Checked]

    def restore_preview(self):
        self.restore_mode = True
        self.restore_btn.setText("結束進階管理")
        self.restore_btn.setStyleSheet("background:#007aff; color:#fff; font-weight:bold;")
        self.restore_btn.clicked.disconnect()
        self.restore_btn.clicked.connect(self.cancel_restore_preview)
        self.populate_fragment_list_with_checkboxes()
        self.update_restore_preview()
        self.img_wrap.overlay_btn.setVisible(True)
        self.img_wrap.overlay_btn.setChecked(False)
        self.overlay_active = False
        self.switch_panel(True)

    def cancel_restore_preview(self):
        self.restore_mode = False
        self.restore_btn.setText("進階管理 / 還原預覽")
        self.restore_btn.setStyleSheet("background:#444; color:#fff; font-weight:bold;")
        self.restore_btn.clicked.disconnect()
        self.restore_btn.clicked.connect(self.restore_preview)
        self.populate_fragment_list_no_checkbox()
        if self.split_result:
            self.img_wrap.preview.set_image(self.split_result[0], trash_highlight=False)
        self.img_wrap.overlay_btn.setVisible(False)
        self.img_wrap.overlay_btn.setChecked(False)
        self.overlay_active = False
        self.switch_panel(False)

    def merge_checked_btn(self):
        checked = self.get_checked_fragments()
        self.merge_checked_restore_fragments(checked)

    def copy_checked_btn(self):
        checked = self.get_checked_fragments()
        self.copy_checked_restore_fragments(checked)

    def delete_checked_btn(self):
        checked = self.get_checked_fragments()
        self.delete_checked_restore_fragments(checked)

    def rename_checked_btn(self):
        checked = self.get_checked_fragments()
        if len(checked) == 1:
            self.rename_fragment_by_name(checked[0])
        elif len(checked) > 1:
            self.batch_rename_fragments()
        else:
            QtWidgets.QMessageBox.information(self, "請選擇碎片", "請先勾選要重新命名的碎片")

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
            except Exception as e:
                self.set_status(f"匯入失敗: {e}", False)
        if self.restore_mode:
            self.populate_fragment_list_with_checkboxes()
            self.update_restore_preview()
        else:
            self.populate_fragment_list_no_checkbox()
        self.set_status("已匯入碎片", True)

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
            except Exception as e:
                self.set_status(f"匯入失敗: {e}", False)
        if self.restore_mode:
            self.populate_fragment_list_with_checkboxes()
            self.update_restore_preview()
        else:
            self.populate_fragment_list_no_checkbox()
        self.set_status("已匯入碎片", True)

    def export_selected_fragments(self):
        checked = self.get_checked_fragments()
        if not checked:
            QtWidgets.QMessageBox.information(self, "匯出失敗", "請先勾選要匯出的碎片！")
            return
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "選擇匯出資料夾")
        if not folder: return
        try:
            for name in checked:
                img = self.fragment_data.get(name)
                if img is not None:
                    np2pil(img).save(os.path.join(folder, f"{name}.png"))
            self.set_status("已匯出選擇碎片", True)
        except Exception as e:
            self.set_status(f"匯出失敗: {e}", False)
    def on_fragment_list_context(self, pos):
        if not self.restore_mode:
            return

        item = self.fragment_list.itemAt(pos)
        if item is None:
            return
            
        checked = []
        for i in range(self.fragment_list.count()):
            it = self.fragment_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                checked.append(it.text())

        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #232323;
                color: #fff;
                border: 1.5px solid #0099cc;
                padding: 2px 3px;
                margin: 0px;
                border-radius: 4px;
                min-width: 1em;
            }
            QMenu::item {
                padding: 3px 12px;
                min-width: 1em;
            }
            QMenu::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0077ff, stop:1 #00eaff);
                color: #fff;
                margin: 0px;
            }
        """)

        if len(checked) >= 2:
            menu.addAction(
                "合併勾選碎片",
                lambda m=menu: (m.close(), self.merge_checked_restore_fragments(checked))
            )
        if len(checked) == 1:
            name = checked[0]
            menu.addAction(
                "重新命名",
                lambda m=menu: (m.close(), self.rename_fragment_by_name(name))
            )
            menu.addAction(
                "匯出單一碎片",
                lambda m=menu: (m.close(), self.export_single_fragment_by_name(name))
            )
        if checked:
            menu.addAction(
                "複製勾選碎片",
                lambda m=menu: (m.close(), self.copy_checked_restore_fragments(checked))
            )
            menu.addAction(
                "刪除勾選碎片",
                lambda m=menu: (m.close(), self.delete_checked_restore_fragments(checked))
            )

        menu.exec(self.fragment_list.mapToGlobal(pos))

    def batch_rename_fragments(self):
        count = self.fragment_list.count()
        if count == 0: return
        prefix, ok = QtWidgets.QInputDialog.getText(self, "批次命名", "請輸入前綴（例如：碎片）", text="碎片")
        if not ok or not prefix: return
        digits = len(str(count))
        old_names = [self.fragment_list.item(i).text() for i in range(count)]
        imgs = [self.fragment_data[name] for name in old_names if name in self.fragment_data]
        self.fragment_list.clear()
        self.fragment_data.clear()
        self.fragment_order.clear()
        for i, img in enumerate(imgs):
            new_name = f"{prefix}_{str(i+1).zfill(digits)}"
            item = QtWidgets.QListWidgetItem(new_name)
            self.fragment_list.addItem(item)
            self.fragment_data[new_name] = img
            self.fragment_order.append(new_name)
        self.set_status(f"已批次命名為 {prefix}_***", True)
        self.refresh_fragment_order()

    def get_current_fragment_order(self):
        names = [self.fragment_list.item(i).text() for i in range(self.fragment_list.count())]
        return [n for n in names if n in self.fragment_data]

    def set_status(self, msg, ok=True):
        if len(msg) > 38:
            msg = ellipsis_middle(msg, 38)
        color = "#0f0" if ok else "#f55"
        self.img_wrap.status_lbl.setText(msg)
        self.img_wrap.status_lbl.setStyleSheet(f"color:{color}; font-weight:bold;")

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
        
    def load_main(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "選擇主圖", "", "PNG圖檔 (*.png)")
        im = None
        if fname:
            try:
                im = Image.open(fname)
                self.main_img = pil2np(im)
                self.main_img_path = fname
                self.img_wrap.preview.set_image(self.main_img)
                self.set_status("主圖載入成功", True)
            except Exception as e:
                self.set_status(f"主圖載入失敗: {e}", False)
                self.main_img = None
                self.main_img_path = ""
                im = None
            self.set_file_label(self.main_file_lbl, self.main_img_path, im)
        
    def load_mask(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "選擇遮罩圖", "", "PNG圖檔 (*.png)")
        if fname:
            try:
                im = Image.open(fname)
                self.mask_img = pil2np(im)
                self.mask_img_path = fname
                self.set_status("遮罩載入成功", True)
                self.img_wrap.preview.set_image(self.mask_img)
            except Exception as e:
                self.set_status(f"遮罩載入失敗: {e}", False)
                self.mask_img = None
                self.mask_img_path = ""
            self.set_file_label(self.mask_file_lbl, self.mask_img_path, im if self.mask_img is not None else None)

    def del_mask(self):
        self.mask_img = None
        self.mask_img_path = ""
        self.mask_file_lbl.setText("")
        self.set_status("已移除遮罩", True)

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
            danger_msgs.append("碎片數量超過7，極易造成記憶體暴增與當機。")

        # 尺寸隨機度高+方塊小
        if self.rand_input.value() > 20 and self.block_input.value() <= 4:
            danger_msgs.append("尺寸隨機度過高且方塊太小，碎片組合將暴增，容易當機。")

        # 重疊像素高
        if self.overlap_pixel_input.value() > 10:
            danger_msgs.append("重疊像素比例超過10%，生成時間會較久。")
        if self.overlap_pixel_input.value() > 20:
            danger_msgs.append("重疊像素比例超過20%，大圖或高分割時容易卡死或爆RAM。")

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
            self.trash_tab.refresh()

        if self.main_img is None:
            self.set_status("請先載入主圖", False)
            return
        block_sz = self.block_input.value()
        rand_sz = self.rand_input.value()
        self.set_status("拆解中... 0%", True)
        self.fragment_list.clear()
        self.fragment_data.clear()
        self.split_btn.setEnabled(False)
        self.abort_btn.setEnabled(True)
        n = self.num_input.value()
        b = block_sz
        r = rand_sz
        mask = self.mask_img if self.mask_img is not None else None
        pil_main = np2pil(self.main_img)
        pil_mask = np2pil(mask) if mask is not None else None
        self.split_thread = SplitThread(pil_main, pil_mask, n, b, r)
        self.split_thread.update_progress.connect(self.progress)
        self.split_thread.result.connect(self.split_done)
        self._split_start_time = time.time()
        self.split_thread.start()
    def progress(self, done, total, msg):
        pct = int(done / total * 100) if total else 0
        if msg:
            self.set_status(f"{msg} ({pct}%)", True)
        else:
            self.set_status(f"拆解中... {pct}%", True)

    def split_done(self, images):
        if hasattr(self, "mask_crop_cb") and self.mask_crop_cb.isChecked() and self.mask_img is not None:
            images = [apply_mask_alpha(img, self.mask_img) for img in images]

        fill_pct = self.overlap_pixel_input.value()
        block_size = self.block_input.value()
        rand_range = self.rand_input.value()
        agg = self.aggregation_input.value()

        if fill_pct > 0:
            self.set_status("開始進行重疊像素填充...", True)
            self.overlap_thread = OverlapThread(
                images, self.main_img, self.mask_img, fill_pct, block_size, rand_range, agg
            )
            self.overlap_thread.progress.connect(self.progress)
            def finish_overlap(result_images):
                self.set_status("重疊像素填充完成", True)
                self._finish_split(result_images)
            self.overlap_thread.result.connect(finish_overlap)
            self.overlap_thread.start()
        else:
            self._finish_split(images)


    def _finish_split(self, images):
        self.split_result = images
        self.fragment_list.clear()
        self.fragment_data.clear()
        self.fragment_order.clear() 
        for idx, arr in enumerate(images):
            name = f"碎片 {idx+1}"
            item = QtWidgets.QListWidgetItem(name)
            self.fragment_list.addItem(item)
            self.fragment_data[name] = arr
            self.fragment_order.append(name)
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
        self.abort_btn.setEnabled(False)
        self.cancel_restore_preview()
        self._initial_snapshot = {
            'fragment_names': [f"碎片 {i+1}" for i in range(len(images))],
            'fragment_imgs': images[:]
        }

            
    def restore_initial_state(self):
        if not self._initial_snapshot:
            QtWidgets.QMessageBox.information(self, "還原初始設定", "尚未有分割過的結果可還原！")
            return
        self.fragment_list.clear()
        self.fragment_data.clear()
        self.fragment_order.clear()
        for name, img in zip(self._initial_snapshot['fragment_names'], self._initial_snapshot['fragment_imgs']):
            item = QtWidgets.QListWidgetItem(name)
            self.fragment_list.addItem(item)
            self.fragment_data[name] = img
            self.fragment_order.append(name)
        self.split_result = self._initial_snapshot['fragment_imgs'][:]
        self.img_wrap.preview.set_image(self.split_result[0])
        self.set_status("已還原初始分割狀態", True)
        self.cancel_restore_preview()

    def fragment_clicked(self, item):
        if self.restore_mode:
            return
        img = self.fragment_data.get(item.text())
        if img is not None:
            self.img_wrap.preview.set_image(img, trash_highlight=False)

    def populate_fragment_list_no_checkbox(self):
        self.fragment_list.clear()
        self.fragment_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.fragment_list.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        try:
            self.fragment_list.itemClicked.disconnect(self.fragment_clicked)
        except:
            pass
        for name in self.fragment_order:
            if name in self.fragment_data:
                item = QtWidgets.QListWidgetItem(name)
                item.setFlags(
                    QtCore.Qt.ItemIsEnabled |
                    QtCore.Qt.ItemIsSelectable |
                    QtCore.Qt.ItemIsDragEnabled |
                    QtCore.Qt.ItemIsDropEnabled
                )
                self.fragment_list.addItem(item)
        self.fragment_list.itemClicked.connect(self.fragment_clicked)

    def populate_fragment_list_with_checkboxes(self):
        self.fragment_list.clear()
        self.fragment_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.fragment_list.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        try:
            self.fragment_list.itemClicked.disconnect(self.fragment_clicked)
        except:
            pass
        for name in self.fragment_order:
            if name in self.fragment_data:
                item = QtWidgets.QListWidgetItem(name)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                item.setCheckState(QtCore.Qt.Checked)
                self.fragment_list.addItem(item)
        # 刪掉 contextMenuPolicy 和 customContextMenuRequested
        self.fragment_list.itemClicked.connect(self.fragment_clicked)
        self.fragment_list.itemChanged.connect(self.on_restore_item_changed)

    def _on_fragment_list_rows_moved(self, parent, start, end, dest, row):
        self.fragment_order = [
            self.fragment_list.item(i).text()
            for i in range(self.fragment_list.count())
            if self.fragment_list.item(i).text() in self.fragment_data
        ]
           
    def on_restore_item_changed(self, item):
        self.update_restore_preview()        

    def update_restore_preview(self):
        if not self.restore_mode: return
        imgs = []
        for i in range(self.fragment_list.count()):
            item = self.fragment_list.item(i)
            if item.checkState() == QtCore.Qt.Checked and item.text() in self.fragment_data:
                imgs.append(self.fragment_data[item.text()])
        if not imgs:
            self.set_status("請至少勾選一個碎片", False)
            self.img_wrap.preview.set_image(None, trash_highlight=False)
            return
        self.set_status("", True)
        h, w = imgs[0].shape[:2]
        base = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        for arr in reversed(imgs):
            overlay = np2pil(arr)
            base = Image.alpha_composite(base, overlay)
        self.img_wrap.preview.set_image(base, trash_highlight=False)
        
    def show_restore_context_menu(self, pos):
        if not self.restore_mode:
            return
        checked = []
        for i in range(self.fragment_list.count()):
            item = self.fragment_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                checked.append(item.text())

        menu = QtWidgets.QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #232323;
                color: #fff;
                border: 1.5px solid #0099cc;
                padding: 2px 3px;
                margin: 0px;
                border-radius: 4px;
                min-width: 1em;
            }
            QMenu::item {
                padding: 3px 12px;
                min-width: 1em;
            }
            QMenu::item:selected {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0077ff, stop:1 #00eaff);
                color: #fff;
                margin: 0px;
            }
        """)

        if len(checked) >= 2:
            menu.addAction("合併勾選碎片", lambda m=menu: (m.close(), self.merge_checked_restore_fragments(checked)))
        if len(checked) == 1:
            name = checked[0]
            menu.addAction("重新命名", lambda m=menu: (m.close(), self.rename_fragment_by_name(name)))
            menu.addAction("匯出單一碎片", lambda m=menu: (m.close(), self.export_single_fragment_by_name(name)))
        if checked:
            menu.addAction("複製勾選碎片", lambda m=menu: (m.close(), self.copy_checked_restore_fragments(checked)))
            menu.addAction("刪除勾選碎片", lambda m=menu: (m.close(), self.delete_checked_restore_fragments(checked)))

        menu.exec(self.fragment_list.mapToGlobal(pos))

    def merge_checked_restore_fragments(self, checked_names):
        if not self.restore_mode or len(checked_names) < 2:
            return
        imgs = [self.fragment_data[name] for name in checked_names if name in self.fragment_data]
        if len(imgs) < 2:
            return
        base = imgs[0].copy()
        for img in imgs[1:]:
            base = numpy_alpha_composite(base, img)
        new_name = f"合併碎片_{len(self.fragment_data)+1}"
        idx = 1
        while new_name in self.fragment_data:
            idx += 1
            new_name = f"合併碎片_{len(self.fragment_data)+idx}"
        for name in checked_names:
            img = self.fragment_data.pop(name)
            if name in self.fragment_order:
                self.fragment_order.remove(name)
            self.recycle_bin.append((name, img))
        self.fragment_data[new_name] = base
        self.fragment_order.append(new_name)
        self.populate_fragment_list_with_checkboxes()
        self.update_restore_preview()
        self.trash_tab.refresh()
        self.set_status(f"已合併並移除原有碎片，共{len(checked_names)}個->1", True)
        self.refresh_fragment_order()

    def copy_checked_restore_fragments(self, checked_names):
        to_copy = []
        for name in checked_names:
            img = self.fragment_data.get(name)
            if img is not None:
                to_copy.append((name, img))
        if not to_copy:
            return
        new_items = []
        for name, img in to_copy:
            new_name = name + "_複製"
            idx = 1
            while new_name in self.fragment_data:
                new_name = f"{name}_複製{idx}"
                idx += 1
            self.fragment_data[new_name] = img.copy()
            self.fragment_order.append(new_name)
            new_items.append(new_name)
        self.populate_fragment_list_with_checkboxes()
        self.set_status(f"已複製 {len(new_items)} 個碎片", True)
        self.refresh_fragment_order()

    def delete_checked_restore_fragments(self, checked_names):
        to_delete = []
        for name in checked_names:
            img = self.fragment_data.get(name)
            if img is not None:
                to_delete.append((name, img))
        if not to_delete:
            return
        for name, img in to_delete:
            if name in self.fragment_data:
                del self.fragment_data[name]
            if name in self.fragment_order:
                self.fragment_order.remove(name)
            self.recycle_bin.append((name, img))
        self.populate_fragment_list_with_checkboxes()
        self.update_restore_preview()
        self.trash_tab.refresh()
        self.set_status(f"已刪除 {len(to_delete)} 個碎片", True)
        self.refresh_fragment_order()

    def refresh_fragment_order(self):
        self.fragment_order = [
            self.fragment_list.item(i).text()
            for i in range(self.fragment_list.count())
            if self.fragment_list.item(i).text() in self.fragment_data
        ]

    def rename_fragment_by_name(self, name):
        for i in range(self.fragment_list.count()):
            item = self.fragment_list.item(i)
            if item.text() == name:
                self.rename_fragment(item)
                break

    def export_single_fragment_by_name(self, name):
        img = self.fragment_data.get(name)
        if img is None:
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "儲存碎片", name + ".png", "PNG圖檔 (*.png)")
        if not fn:
            return
        try:
            export_img = img
            if hasattr(self, "mask_crop_cb") and self.mask_crop_cb.isChecked():
                export_img = apply_mask_alpha(img, self.mask_img)
            np2pil(export_img).save(fn)
            self.set_status(f"已儲存: {os.path.basename(fn)}", True)
        except Exception as e:
            self.set_status(f"儲存失敗: {e}", False)

    def restore_from_trash(self, name, img):
        base = name
        idx = 1
        while base in self.fragment_data:
            base = f"{name}_復原{idx}"
            idx += 1
        self.fragment_data[base] = img
        self.fragment_order.append(base)
        if self.restore_mode:
            self.populate_fragment_list_with_checkboxes()
            self.update_restore_preview()
        else:
            self.populate_fragment_list_no_checkbox()

    def refresh_fragment_list(self):
        if self.restore_mode:
            self.populate_fragment_list_with_checkboxes()
            self.update_restore_preview()
        else:
            self.populate_fragment_list_no_checkbox()
        self.img_wrap.preview.set_image(None, trash_highlight=False)

    def abort_split(self):
        reply = QtWidgets.QMessageBox.question(
            self,
            "終止並重啟",
            "這會強制結束並重啟程式，所有未儲存資料都會消失！\n\n確定要終止並重啟？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.set_status("正在重啟...", False)
            QtWidgets.QApplication.processEvents()
            python = sys.executable
            os.execl(python, python, *sys.argv)

    def export_all_fragments_zip(self):
        if not self.fragment_data:
            QtWidgets.QMessageBox.information(self, "匯出失敗", "沒有任何碎片可匯出！")
            return
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "儲存所有碎片", "所有碎片.zip", "ZIP 壓縮檔 (*.zip)")
        if not fn:
            return
        try:
            with zipfile.ZipFile(fn, 'w', zipfile.ZIP_DEFLATED) as zf:
                for name, img in self.fragment_data.items():
                    img_bytes = io.BytesIO()
                    np2pil(img).save(img_bytes, format='PNG')
                    img_bytes.seek(0)
                    safe_name = name.replace('/', '_').replace('\\', '_')
                    zf.writestr(f"{safe_name}.png", img_bytes.read())
            self.set_status(f"已匯出全部碎片到: {os.path.basename(fn)}", True)
        except Exception as e:
            self.set_status(f"壓縮匯出失敗: {e}", False)

    def setup_shortcuts(self):
        shortcut_all  = QShortcut(QKeySequence("Ctrl+A"),       self)
        shortcut_none = QShortcut(QKeySequence("Ctrl+D"),       self)
        shortcut_inv  = QShortcut(QKeySequence("Ctrl+Shift+A"), self)
        shortcut_all.activated.connect(lambda: self._batch_checkbox_action('all'))
        shortcut_none.activated.connect(lambda: self._batch_checkbox_action('none'))
        shortcut_inv.activated.connect(lambda: self._batch_checkbox_action('invert'))

    def _batch_checkbox_action(self, mode):
        if not getattr(self, 'restore_mode', False):
            return
        for i in range(self.fragment_list.count()):
            item = self.fragment_list.item(i)
            if mode == 'all':
                item.setCheckState(QtCore.Qt.Checked)
            elif mode == 'none':
                item.setCheckState(QtCore.Qt.Unchecked)
            elif mode == 'invert':
                item.setCheckState(QtCore.Qt.Unchecked if item.checkState()==QtCore.Qt.Checked else QtCore.Qt.Checked)
        self.update_restore_preview()
        
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
        if old_name in self.fragment_order:
            idx = self.fragment_order.index(old_name)
            self.fragment_order[idx] = new_name
        item.setText(new_name)
        self.set_status(f"已重新命名為 {new_name}", True)

    def rename_selected_fragment(self):
        selected = self.fragment_list.currentItem()
        if not selected:
            QtWidgets.QMessageBox.information(self, "請選擇碎片", "請先選取要重新命名的碎片")
            return
        self.rename_fragment(selected)

    def on_tab_changed(self, idx):
        if self.tabs.currentWidget() != self.trash_tab:
            self.trash_tab.clear_highlight()
            if self.restore_mode:
                self.populate_fragment_list_with_checkboxes()
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
        else:
            if not self.restore_mode:
                if len(self.recycle_bin) > 0:
                    last_name, last_img = self.recycle_bin[-1]
                    self.img_wrap.preview.set_image(last_img, trash_highlight=True)
                else:
                    self.img_wrap.preview.set_image(None, trash_highlight=True)
        
    def generate_overlay_preview(self):
        if self.main_img is None:
            return None
        arr = self.main_img.copy()
        h, w = arr.shape[:2]
        alpha = arr[..., 3]
        bw_arr = np.stack([alpha, alpha, alpha, np.full_like(alpha, 255)], axis=-1)
        mask_count = np.zeros((h, w), dtype=np.uint8)
        for i in range(self.fragment_list.count()):
            item = self.fragment_list.item(i)
            if item.checkState() == QtCore.Qt.Checked and item.text() in self.fragment_data:
                frag = self.fragment_data[item.text()]
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
        if not self.fragment_data:
            QtWidgets.QMessageBox.information(self, "無碎片", "請先進行拆解")
            return
        kw = self.interfere_panel.get_settings()
        self.interfere_images_dict.clear()
        self.gen_thread = InterfereGenThread(
            self.fragment_data.copy(),
            self.main_img.copy(),
            self.split_result[:],
            kw
        )
        self.gen_thread.progress.connect(
            lambda cur, tot, msg: self.set_status(f"{msg}", True)
        )
        def done(result_dict):
            self.interfere_images_dict = result_dict
            QtWidgets.QMessageBox.information(self, "產生完成", "已為每片碎片產生干擾像素，可切換預覽或合成。")
            self.set_status("干擾像素產生完成", True)
        self.gen_thread.result.connect(done)
        self.gen_thread.start()
        self.set_status("開始產生干擾像素...", True)
        
    def interfere_progress(self, done, total, msg):
        self.set_status(f"{msg} ({done}/{total})", True)

    def interfere_done(self, result):
        self.interfere_images_dict = result
        self.set_status("干擾像素產生完成", True)
        self.interfere_panel.gen_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(self, "產生完成", "已為每片碎片產生干擾像素，可合成到碎片或重複執行疊加。")

    def apply_interfere_to_fragments(self):
        if not self.interfere_images_dict:
            QtWidgets.QMessageBox.information(self, "尚未產生", "請先產生干擾像素")
            return
        cnt = 0
        if not self.split_result or len(self.split_result) < 1:
            QtWidgets.QMessageBox.warning(self, "錯誤", "找不到第一片碎片作為遮罩，請先拆解圖片")
            return
        main_alpha255_mask = (self.split_result[0][..., 3] == 255)
        for idx, name in enumerate(self.fragment_order):
            if idx == 0:
                continue
            interfere = self.interfere_images_dict.get(name)
            orig = self.fragment_data.get(name)
            if orig is not None and interfere is not None and interfere.shape == orig.shape:
                merged = apply_interfere_masked(orig, interfere, main_alpha255_mask)
                self.recycle_bin.append((name, orig))
                self.fragment_data[name] = merged
                cnt += 1
        self.interfere_images_dict.clear()
        self.populate_fragment_list_no_checkbox()
        QtWidgets.QMessageBox.information(self, "合成完成", f"已合成到 {cnt} 個碎片（不包含第一片），原碎片移入垃圾桶")

def gen_overlap_pixels(main_img, mask_img, fill_percent, block_size, random_range):
    arr = main_img.copy()
    h, w = arr.shape[:2]
    if mask_img is not None:
        main_alpha255_mask = ((arr[...,3]==255) | (mask_img[...,3]==255))
    else:
        main_alpha255_mask = (arr[...,3]==255)
    coords = np.argwhere(main_alpha255_mask)
    total = len(coords)
    if total == 0:
        return np.zeros((h, w, 4), dtype=np.uint8)

    target_fill = int(total * fill_percent / 100)
    filled = np.zeros((h, w), dtype=bool)
    out = np.zeros((h, w, 4), dtype=np.uint8)
    fill_cnt = 0
    tries = 0
    while fill_cnt < target_fill and tries < target_fill * 10:
        y, x = coords[random.randrange(len(coords))]
        sz = random.randint(block_size, block_size * random_range)
        if x+sz > w or y+sz > h:
            tries += 1
            continue
        patch = arr[y:y+sz, x:x+sz]
        mask_patch = (patch[...,3] == 255)
        target = ~filled[y:y+sz, x:x+sz] & mask_patch
        out[y:y+sz, x:x+sz][target] = patch[target]
        filled[y:y+sz, x:x+sz][target] = True
        fill_cnt += np.sum(target)
        tries += 1
    out[~main_alpha255_mask] = 0
    return out

def apply_overlap_to_fragment(frag, overlap_img):
    mask = (frag[...,3] == 0) & (overlap_img[...,3] > 0)
    out = frag.copy()
    out[mask] = overlap_img[mask]
    return out

def overlap_merge_worker(args):
    frag, overlap_img = args
    return apply_overlap_to_fragment(frag, overlap_img)

def apply_overlap_to_all_fragments_mp(fragments, main_img, mask_img, fill_percent, block_size, random_range, progress_cb=None, abort_cb=None):
    overlap_img = gen_overlap_pixels(main_img, mask_img, fill_percent, block_size, random_range)
    N = len(fragments)
    args_list = [(frag, overlap_img) for frag in fragments]
    results = [None] * N
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(overlap_merge_worker, arg): idx for idx, arg in enumerate(args_list)}
        done_cnt = 0
        for fut in concurrent.futures.as_completed(futures):
            idx = futures[fut]
            if abort_cb is not None and abort_cb():
                executor.shutdown(wait=False, cancel_futures=True)
                raise Exception("Aborted by user")
            results[idx] = fut.result()
            done_cnt += 1
            if progress_cb:
                progress_cb(done_cnt, N, f"重疊像素合成 {done_cnt}/{N}")
    return results
if __name__ == "__main__":
    mp.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
