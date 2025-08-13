APP_VERSION = "1.1.5"
import sys
import os
import time
import io
import random
import zipfile
from collections import deque

import numpy as np
from PIL import Image

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QShortcut, QKeySequence, QAction

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
        primary_mask = (mask_img[...,3]==255)
    else:
        primary_mask = (arr[...,3]==255)
    coords = np.argwhere(primary_mask)
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
            sz = block_size
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
        frag[~primary_mask] = 0
        if progress_cb:
            progress_cb(i+1, N, f"重疊像素填充...")
    return fragments
    
def overlap_fill_density_blocks_agg(
    fragments, main_img, mask_img, fill_percent, block_size, random_range, aggregation=1
):
    if fill_percent <= 0 or not fragments or main_img is None:
        return fragments

    arr = main_img.copy()
    h, w = arr.shape[:2]
    if mask_img is not None:
        primary_mask = (mask_img[...,3]==255)
    else:
        primary_mask = (arr[...,3]==255)
    coords = np.argwhere(primary_mask)
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
                if not (0 <= y < h and 0 <= x < w and primary_mask[y,x]):
                    y, x = coords[random.randrange(len(coords))]
            else:
                y, x = coords[random.randrange(len(coords))]

            sz = block_size
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
        frag[~primary_mask] = 0
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
        
def simple_block_degrade(src_img, block_size, rand_range, density, noise_strength, brightness_strength, color_strength):
    """
    用設定產生塊狀劣化：明暗、色偏、雜訊（僅作用於 alpha>0 區域）。
    src_img: RGBA uint8 numpy array
    brightness_strength, color_strength, noise_strength: 0~100 百分比
    density: 覆蓋強度 (1.0 = 100%)
    """
    h, w = src_img.shape[:2]
    out = src_img.copy().astype(np.float32)

    # 只在可見區運算
    alpha_mask = src_img[..., 3] > 0
    valid_coords = np.argwhere(alpha_mask)
    valid_count = int(valid_coords.shape[0])

    # 無有效像素或密度為 0：直接回傳（alpha 保持原樣）
    if valid_count == 0 or density <= 0:
        out[..., 3] = src_img[..., 3]
        return np.clip(out, 0, 255).astype(np.uint8)

    # 以「有效像素」估算區塊數，維持視覺密度一致
    base_area = max(1, block_size * block_size)
    n_blocks = max(1, int(valid_count * float(density) / base_area))

    for _ in range(n_blocks):
        # 區塊尺寸：與原本一致
        sz = random.randint(block_size, max(block_size, block_size * rand_range))

        # 從有效座標池挑一點，作為區塊中心，並夾在邊界內
        yx = valid_coords[random.randint(0, valid_count - 1)]
        y0 = int(min(max(0, yx[0] - sz // 2), max(0, h - sz)))
        x0 = int(min(max(0, yx[1] - sz // 2), max(0, w - sz)))
        if y0 >= h or x0 >= w or sz <= 0:
            continue

        region = out[y0:y0+sz, x0:x0+sz, :3]
        if region.size == 0:
            continue

        # === 原本的劣化處理 ===
        # 隨機明暗
        b_factor = 1 + random.uniform(-brightness_strength, brightness_strength) / 100.0
        patch = region * b_factor

        # 隨機色偏（每 channel）
        color_offsets = np.array([
            random.uniform(-color_strength, color_strength),
            random.uniform(-color_strength, color_strength),
            random.uniform(-color_strength, color_strength),
        ]) / 100.0
        patch = patch + patch * color_offsets

        # 加雜訊（高斯）
        sigma = (noise_strength / 100.0) * 30
        if sigma > 0:
            noise = np.random.normal(0, sigma, patch.shape)
            patch = patch + noise

        patch = np.clip(patch, 0, 255)

        # 只寫回可見區（alpha>0）
        m = alpha_mask[y0:y0+sz, x0:x0+sz]
        if np.any(m):
            region[m] = patch[m]

    # alpha 保持原本
    out[..., 3] = src_img[..., 3]
    return np.clip(out, 0, 255).astype(np.uint8)
 
def _degrade_chunk(args):
    chunk, settings = args
    # 呼叫原本的 single-thread 劣化函式（simple_block_degrade 會保留 alpha）
    return simple_block_degrade(
        chunk,
        settings["block_size"],
        settings["rand_range"],
        settings["density"],
        settings["noise_strength"],
        settings["brightness_strength"],
        settings["color_strength"],
    )

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

def crop_to_primary_mask(img, primary_mask):
    if primary_mask is None:
        return img.copy()
    out = img.copy()
    if primary_mask.dtype != bool:
        primary_mask = primary_mask.astype(bool)
    if primary_mask.shape != out.shape[:2]:
        raise ValueError(f"primary_mask shape {primary_mask.shape} doesn't match image shape {out.shape[:2]}")
    inverse = ~primary_mask
    out[inverse] = 0
    return out

def crop_to_mask_alpha(img, mask_alpha):
    out = img.copy()
    out_alpha = out[..., 3].astype(np.float32) * (mask_alpha.astype(np.float32) / 255.0)
    out[..., 3] = np.clip(out_alpha, 0, 255).astype(np.uint8)
    return out

def ellipsis_middle(text, maxlen=28):
    if len(text) <= maxlen:
        return text
    name, ext = os.path.splitext(text)
    if len(ext) > 5: ext = ext[:5] + "..."
    keep = maxlen - len(ext) - 3
    if keep < 8:
        return text[:maxlen-3] + "..."
    return name[:keep//2] + "..." + name[-keep//2:] + ext

def build_interfere_block_pool(
    src_arr, block_size=10, random_range=2, pool_size=400, alpha_min=1, alpha_max=100, restrict_mask=None
):
    h, w = src_arr.shape[:2]
    alpha = src_arr[..., 3]
    if restrict_mask is not None:
        valid_coords = np.argwhere(restrict_mask)
    else:
        valid_coords = np.argwhere(alpha > 0)
    blocks = []
    for _ in range(pool_size * 2):
        y, x = valid_coords[random.randrange(len(valid_coords))]
        sz = random.randint(block_size, block_size * random_range)
        if x+sz > w or y+sz > h:
            continue
        patch = src_arr[y:y+sz, x:x+sz]
        alpha_area = np.sum(patch[..., 3] > 0) / (sz * sz) * 100
        if alpha_min <= alpha_area <= alpha_max:
            blocks.append(patch.copy())
        if len(blocks) >= pool_size:
            break
    return blocks

def gen_multi_overlap_interfere(
    frag_img, block_pool,
    coverage=0.7, max_try=10, allow_overlap=False,
    primary_mask=None, main_img_arr=None,
    block_size=10, random_range=2,
    fill_stages=None, patch_min_size_ratio=0.4
):
    h, w = frag_img.shape[:2]

    # ① 用有效區域計數
    valid_mask = primary_mask if primary_mask is not None else (frag_img[..., 3] > 0)
    coords = np.argwhere(valid_mask)
    total_valid = coords.shape[0]
    if total_valid == 0 or not block_pool:
        return np.zeros_like(frag_img)

    out = np.zeros_like(frag_img)
    pasted_mask = np.zeros((h, w), dtype=bool)
    pasted = 0
    tries = 0

    # ② 用密度決定填充目標
    coverage = float(np.clip(coverage, 0.05, 0.98))
    if fill_stages is None:
        fill_main_ratio = coverage * 0.9
        fill_final_ratio = coverage
    else:
        fill_main_ratio, fill_final_ratio = fill_stages

    max_fill_main = int(total_valid * fill_main_ratio)
    max_fill_final = int(total_valid * fill_final_ratio)

    # ③ 主階段：只在有效區域貼、只計有效像素
    while pasted < max_fill_main and tries < max_fill_final * max_try:
        y, x = coords[np.random.randint(len(coords))]
        patch = random.choice(block_pool)
        bh, bw = patch.shape[:2]
        if y+bh > h or x+bw > w: 
            tries += 1; 
            continue
        mask_patch = (patch[..., 3] > 0)

        # 只填到有效區域且未貼過的位置
        valid_patch_mask = (~pasted_mask[y:y+bh, x:x+bw]) & mask_patch & valid_mask[y:y+bh, x:x+bw]
        if not np.any(valid_patch_mask):
            tries += 1
            continue

        pa = patch[..., 3:4] / 255.0
        out[y:y+bh, x:x+bw, :3] = (
            patch[..., :3] * pa + out[y:y+bh, x:x+bw, :3] * (1 - pa)
        ).astype(np.uint8)
        out[y:y+bh, x:x+bw, 3] = np.maximum(out[y:y+bh, x:x+bw, 3], patch[..., 3])

        pasted += int(np.count_nonzero(valid_patch_mask))
        pasted_mask[y:y+bh, x:x+bw] |= valid_patch_mask
        tries += 1

    return out
    
def gen_interfere_worker(args):
    frag_name, frag_arr, block_pool, settings, primary_mask = args

    # block_pool 已經是主線程共用，這裡不再 build

    interfere = gen_multi_overlap_interfere(
        frag_arr, block_pool,
        coverage=settings['density'],                 # 直接用 0~1
        allow_overlap=settings.get('allow_overlap', False),
        primary_mask=primary_mask,
        main_img_arr=frag_arr,                        # 可選：開啟細補階段
        block_size=settings['block_size'],
        random_range=settings['random_range'],
    )

    if settings.get('ignore_semitrans', True):
        interfere = crop_to_primary_mask(interfere, primary_mask)
    else:
        interfere = crop_to_mask_alpha(interfere, frag_arr[..., 3])

    return frag_name, interfere

def compute_fragment_bbox(frag):
    alpha = frag[..., 3]
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0 or len(ys) == 0:
        return (0, 0, 0, 0)
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return (x0, y0, x1, y1)

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
                mw.preview_mask_grayalpha(mw.mask_img)
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

            mw = getattr(self._parent, "_parent", None)
            parent_widget = self._parent.parent()
            is_trash_tab = hasattr(parent_widget, 'tabs') and parent_widget.tabs.currentWidget() == parent_widget.trash_tab

            if is_trash_tab and self.trash_highlight:
                frame_color = QtGui.QColor(255, 0, 0)
                frame_width = 3
                pen = QtGui.QPen(frame_color, frame_width)
                painter.setPen(pen)
                painter.drawRect(self.rect().adjusted(2, 2, -4, -4))

        else:
            font = painter.font()
            font.setPointSize(22)
            font.setBold(True)
            painter.setFont(font)

            bg_key = BG_OPTIONS[self.bg_idx][0]
            # 白色背景 -> 黑字；透明網格 -> 黑字；其他(深色) -> 白字
            if isinstance(bg_key, str) and (bg_key == "check" or bg_key.lower() in ("#ffffff", "white")):
                pen_color = QtGui.QColor(0, 0, 0)     # 黑字
            else:
                pen_color = QtGui.QColor(255, 255, 255)  # 白字

            painter.setPen(QtGui.QPen(pen_color))
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
        self.status_lbl = StatusLabel("")
        self.status_lbl.setStyleSheet("color:#0f0;font-size:15px;min-width:140px;max-width:260px;")
        hz.addWidget(self.status_lbl)
        lay.addLayout(hz)
        self.preview._parent = self
        self.update_zoom_display()
        self.previewing_fragment_name = None
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
    def __init__(self, main_img, mask_img, count, blocksize, rand_factor, strict_mask=True):
        super().__init__()
        self.main_img = main_img
        self.mask_img = mask_img
        self.count = count
        self.blocksize = blocksize
        self.rand_factor = rand_factor
        self.strict_mask = strict_mask
        self._abort = False
    def abort(self): self._abort = True
    def run(self):
        try:
            self.update_progress.emit(0, 7, "計算分割區塊...")
            t0 = time.time()
            rgba = pil2np(self.main_img)
            h, w = rgba.shape[0], rgba.shape[1]

            # 以 numpy 快速取得 valid coords（有 alpha 的 pixel 或遮罩指定）
            if self.mask_img is not None:
                mask_rgba = pil2np(self.mask_img)
                if mask_rgba.shape[:2] != (h, w):
                    self.result.emit([])
                    return
                primary_mask = (mask_rgba[..., 3] == 255)
                coverage_mask = (mask_rgba[..., 3] > 0)
            else:
                primary_mask = (rgba[..., 3] == 255)
                coverage_mask = (rgba[..., 3] > 0)

            cov_coords_arr = np.argwhere(coverage_mask)  # for block覆蓋判斷（alpha>0 全覆蓋要求）
            if cov_coords_arr.size == 0:
                self.result.emit([])
                return
            # 用 set 方便 block 生成時檢查 overlap（格式為 (x,y)）
            cov_coords_set = set((int(x), int(y)) for y, x in cov_coords_arr)
            primary_coords_arr = np.argwhere(primary_mask)
            cov_coords_set = set((int(x), int(y)) for y, x in cov_coords_arr)
            primary_coords_arr = np.argwhere(primary_mask)

            self.update_progress.emit(1, 7, "開始分割...")

            # 分割區塊（tile-like，只有包含有效 pixel 才收）
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
                            if (xx, yy) in cov_coords_set:
                                has_overlap = True
                                break
                        if has_overlap:
                            break
                    if has_overlap:
                        blocks.append((x, y, bw, bh))
                    x += bw
                y += bh

            # 將 blocks 分派到各碎片
            self.update_progress.emit(3, 7, "正在產生碎片（第1步）...")
            fragment_blocks = [[] for _ in range(self.count)]
            for block in blocks:
                fid = random.randint(0, self.count - 1)
                fragment_blocks[fid].append(block)

            # 建立每 pixel 的 fragment 指派表（-1 表示尚未指派）
            self.update_progress.emit(4, 7, "正在分配像素到碎片...")
            fragment_id_map = np.full((h, w), -1, dtype=int)
            for fid, blocks_for_fid in enumerate(fragment_blocks):
                for block in blocks_for_fid:
                    x, y, bw, bh = block
                    for yy in range(y, y + bh):
                        for xx in range(x, x + bw):
                            # 勾選「不溢出」→ 嚴格限制在 α>0；未勾選→ 整塊都可指派
                            if (not self.strict_mask) or coverage_mask[yy, xx]:
                                fragment_id_map[yy, xx] = fid

            # 處理孤立 pixel（還沒被任何 block 覆蓋的 valid pixel）
            self.update_progress.emit(5, 7, "處理孤立像素...")
            for yx in cov_coords_arr:
                y, x = int(yx[0]), int(yx[1])
                if fragment_id_map[y, x] == -1:
                    fragment_id_map[y, x] = random.randint(0, self.count - 1)

            # 生成碎片圖像
            self.update_progress.emit(6, 7, "產生碎片圖像...")
            fragment_imgs = [np.zeros((h, w, 4), dtype=np.uint8) for _ in range(self.count)]
            for fid in range(self.count):
                mask = fragment_id_map == fid
                fragment_imgs[fid][mask] = rgba[mask]

            self._last_time = time.time() - t0
            self.result.emit(fragment_imgs)
        except Exception as e:
            import traceback
            print(f"[SplitThread Exception]: {e}")
            traceback.print_exc()
            self.result.emit([])

class OverlapThread(QThread):
    progress = Signal(int, int, str)
    result = Signal(list)

    def __init__(self, images, main_img, mask_img, fill_pct, block_size, rand_range, aggregation=1, limit_to_mask=False):
        super().__init__()
        self.images = [img.copy() for img in images]
        self.main_img = main_img.copy() if main_img is not None else None
        self.mask_img = mask_img.copy() if mask_img is not None else None
        self.fill_pct = fill_pct
        self.block_size = block_size
        self.rand_range = rand_range
        self.aggregation = aggregation
        self.limit_to_mask = limit_to_mask
        self._abort = False
        self._executor = None

    def abort(self):
        self._abort = True
        if self._executor:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._executor = None

    def run(self):
        def progress_cb(cur, total, msg):
            self.progress.emit(cur, total, msg)
        def abort_cb():
            return getattr(self, '_abort', False)
        self._executor = None
        try:
            result = apply_overlap_to_all_fragments_mp(
                self.images, self.main_img, self.mask_img,
                self.fill_pct, self.block_size, self.rand_range,
                limit_to_mask=True,                 # 強制只在遮罩內
                aggregation=self.aggregation,
                progress_cb=progress_cb, abort_cb=abort_cb
            )
            self.result.emit(result)
        except Exception as e:
            self.result.emit([])
        finally:
            if self._executor:
                try:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                self._executor = None 

class InterferePanel(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QtWidgets.QFormLayout(self)

        self.block_size = QtWidgets.QSpinBox()
        self.block_size.setRange(1, 30)
        self.block_size.setValue(3)
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
        self.random_range.setValue(10)
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
        self.density.setValue(70)
        self.density_lbl = QtWidgets.QLabel("干擾密度: 70%")
        density_tip = QHelpButton(
            "決定干擾像素填滿目標區域的比例，數字越高，干擾覆蓋越密集，100%不代表全填滿，實際上會受到區塊尺寸影影響。\n\n"
            "優點：密度高可大幅阻礙還原。\n\n"
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
            lambda v: self.alpha_min_lbl.setText(f"干擾像素的不透明面積最小數值: {v}%"))

        self.alpha_max = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.alpha_max.setRange(1, 100)
        self.alpha_max.setValue(100)
        self.alpha_max_lbl = QtWidgets.QLabel("干擾像素的不透明面積最大數值: 100%")
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
            lambda v: self.alpha_max_lbl.setText(f"干擾像素的不透明面積最大數值: {v}%"))

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

        self.ignore_semitrans_cb = QtWidgets.QCheckBox("忽略半透明區域")
        self.ignore_semitrans_cb.setChecked(True)
        ignore_tip = QHelpButton(
            "勾選後，僅在第一片碎片完全不透明的區域產生干擾像素。\n"
            
            "取消勾選時，會在第一片碎片涵蓋的區域都產生干擾像素。\n"
            
            "建議開啟，能避免在主圖透明邊緣產生髒點"
        )
        ignore_row = QtWidgets.QHBoxLayout()
        ignore_row.addWidget(self.ignore_semitrans_cb)
        ignore_row.addWidget(ignore_tip)
        ignore_row.addStretch()
        lay.addRow(ignore_row)


        self.gen_btn = QtWidgets.QPushButton("產生干擾像素圖")
        self.apply_btn = QtWidgets.QPushButton("合成到碎片")
        self.gen_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.apply_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.gen_btn.setFixedHeight(28)
        self.apply_btn.setFixedHeight(28)
        btnrow = QtWidgets.QHBoxLayout()
        btnrow.addWidget(self.gen_btn)
        btnrow.addWidget(self.apply_btn)
        lay.addRow(btnrow)

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
            ignore_semitrans=self.ignore_semitrans_cb.isChecked()
        )

class InterfereGenThread(QtCore.QThread):
    progress = QtCore.Signal(int, int, str)
    result = QtCore.Signal(dict)

    def __init__(self, fragment_data, settings, fragment_order, primary_mask, block_pool):
        super().__init__()
        self.fragment_data = fragment_data
        self.settings = settings
        self.fragment_order = fragment_order
        self.primary_mask = primary_mask
        self.block_pool = block_pool
        self._abort = False
        self._executor = None

    def abort(self):
        self._abort = True
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._executor = None

    def run(self):
        start_time = time.time()
        result = {}
        self._executor = None

        try:
            if not self.fragment_order or self.primary_mask is None or self.block_pool is None:
                self.result.emit({})
                return

            if isinstance(self.primary_mask, np.ndarray) and self.primary_mask.dtype == bool:
                primary_mask_bool = self.primary_mask
            else:
                primary_mask_bool = (self.primary_mask > 0)

            args_list = []
            for name in self.fragment_order:
                frag = self.fragment_data.get(name)
                if frag is None:
                    continue
                # block_pool 改成傳入，不再自己建
                args_list.append((
                    name,
                    frag,
                    self.block_pool,
                    self.settings,
                    primary_mask_bool
                ))

            total = len(args_list)
            self.progress.emit(0, total, "產生干擾像素中...")

            self._executor = ProcessPoolExecutor()
            futures = {self._executor.submit(gen_interfere_worker, arg): arg[0] for arg in args_list}
            done_cnt = 0

            for fut in as_completed(futures):
                if self._abort:
                    return  # 中斷不 emit
                name = futures[fut]
                try:
                    r_name, interfere = fut.result()
                    result[r_name] = interfere
                except Exception as e:
                    print(f"產生 {name} 干擾像素失敗: {e}")
                done_cnt += 1
                elapsed = time.time() - start_time
                self.progress.emit(done_cnt, total, f"產生干擾像素完成，花費 {elapsed:.1f} 秒")

            self.result.emit(result)

        except Exception as e:
            print("InterfereGenThread exception:", e)
            self.result.emit({})
        finally:
            if self._executor is not None:
                try:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                self._executor = None

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
        import_tip = QHelpButton("載入一張圖作為劣化來源，會在主預覽顯示原圖與後續劣化。")
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
        self.density_label = QtWidgets.QLabel("劣化密度: 70%")
        self.density_slider.valueChanged.connect(lambda v: self.density_label.setText(f"劣化密度: {v}%"))
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
        self.noise_label = QtWidgets.QLabel("噪點強度: 10%")
        self.noise_slider.valueChanged.connect(lambda v: self.noise_label.setText(f"噪點強度: {v}%"))
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
        self.bright_label = QtWidgets.QLabel("隨機明暗: 10%")
        self.bright_slider.valueChanged.connect(lambda v: self.bright_label.setText(f"隨機明暗: {v}%"))
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
        self.color_label = QtWidgets.QLabel("色偏強度: 10%")
        self.color_slider.valueChanged.connect(lambda v: self.color_label.setText(f"色偏強度: {v}%"))
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
        self.apply_export_btn = QtWidgets.QPushButton("套用並匯出")
        for btn in (self.gen_preview_btn, self.restore_source_btn, self.apply_export_btn):
            btn.setFixedHeight(28)
            btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        btn_row.addWidget(self.gen_preview_btn)
        btn_row.addWidget(self.restore_source_btn)
        btn_row.addWidget(self.apply_export_btn)
        lay.addRow(btn_row)

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
            self.imported_filename_lbl.setText("尚未載入任何圖")
            self.imported_filename_lbl.setToolTip("")
            return
        base = os.path.basename(path)
        fm = self.imported_filename_lbl.fontMetrics()
        max_width = self.imported_filename_lbl.width() if self.imported_filename_lbl.width() > 0 else 220
        elided = fm.elidedText(base, QtCore.Qt.ElideMiddle, max_width)
        self.imported_filename_lbl.setText(elided)
        self.imported_filename_lbl.setToolTip(path)




class DegradePreviewThread(QtCore.QThread):
    progress = Signal(int, int, str)  # 回報進度
    result = Signal(np.ndarray)

    def __init__(self, src_img, settings):
        super().__init__()
        self.src_img = src_img.copy()
        self.settings = settings
        self._abort = False
        self._executor = None

    def abort(self):
        self._abort = True
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._executor = None  # 防止重複 shutdown

    def run(self):
        self._executor = None
        try:
            block_size = self.settings["block_size"]
            rand_range = self.settings["rand_range"]
            density = self.settings["density"]
            noise_strength = self.settings["noise_strength"]
            brightness_strength = self.settings["brightness_strength"]
            color_strength = self.settings["color_strength"]

            h, w = self.src_img.shape[:2]
            n_blocks_estimate = h * w * density / max(1, (block_size * block_size))
            max_workers = min(mp.cpu_count(), 4)
            use_parallel = n_blocks_estimate >= 50 and max_workers > 1

            if not use_parallel:
                degraded = simple_block_degrade(
                    self.src_img,
                    block_size,
                    rand_range,
                    density,
                    noise_strength,
                    brightness_strength,
                    color_strength
                )
                degraded[..., 3] = self.src_img[..., 3]
                self.result.emit(degraded)
                return

            xs = np.linspace(0, w, max_workers + 1, dtype=int)
            tasks = []
            for i in range(max_workers):
                x0, x1 = xs[i], xs[i+1]
                chunk = self.src_img[:, x0:x1].copy()
                tasks.append(((chunk, self.settings), x0, x1))

            out = np.zeros_like(self.src_img, dtype=np.float32)
            total = len(tasks)
            done_cnt = 0

            self._executor = ProcessPoolExecutor(max_workers=max_workers)
            futures = {
                self._executor.submit(_degrade_chunk, chunk_args): (x0, x1)
                for (chunk_args, x0, x1) in tasks
            }

            for fut in as_completed(futures):
                if self._abort:
                    # 提前跳出，直接關閉 executor
                    return
                x0, x1 = futures[fut]
                try:
                    degraded_chunk = fut.result()
                    out[:, x0:x1, :3] = degraded_chunk[:, :, :3].astype(np.float32)
                except Exception:
                    out[:, x0:x1, :3] = self.src_img[:, x0:x1, :3].astype(np.float32)
                done_cnt += 1
                self.progress.emit(done_cnt, total, "劣化中...")

            out[..., 3] = self.src_img[..., 3]
            degraded = np.clip(out, 0, 255).astype(np.uint8)
            self.result.emit(degraded)
        except Exception as e:
            print("DegradePreviewThread exception:", e)
            self.result.emit(self.src_img)
        finally:
            # shutdown 無論如何都做
            if self._executor is not None:
                try:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
                self._executor = None  # 釋放引用，利於 gc

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

        # ==== 新增：自動選中最後一個並預覽 ====
        count = self.list.count()
        if count > 0:
            self.list.setCurrentRow(count - 1)
            item = self.list.item(count - 1)
            # 預覽圖片（直接呼叫點擊事件）
            if item:
                self.on_trash_item_clicked(item)
                # 保證 UI 顯示到底部
                row = self.list.row(item)
                QtCore.QTimer.singleShot(0, lambda r=row: self._scroll_to_row(r))
        else:
            # 沒有碎片就清除預覽
            self.clear_highlight()
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
   
    def _scroll_to_row(self, r):
        it = self.list.item(r)
        if it is not None:
            self.list.scrollToItem(it, QtWidgets.QAbstractItemView.PositionAtBottom)

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.checkbox_items = []
        self.setWindowTitle(f"MSW造型防盜拆解工具 專業版 MSW Skin Fragmenter Pro v{APP_VERSION}")
        self.setMinimumSize(1280,800)
        self.setStyleSheet("background:#232323; color:#eee; font-size:15px;")
        self.img_wrap = ImagePreviewWrap(self)
        self._degrade_warning_dialog = None
        self._last_degrade_warning = 0.0  # 用來簡單 debounce（秒）

        # 圖片 / 碎片 / 狀態
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

        # 新增：劣化來源與 preview 暫存（共用 preview）
        self.degrade_source_img = None
        self.degrade_preview_pending = None
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
        self.fragment_as_mask_name = None
      
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

        self.mask_btn = QtWidgets.QPushButton("自定遮罩")
        self.mask_btn.clicked.connect(self.load_mask)
        self.mask_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.del_mask_btn = QtWidgets.QPushButton("移除遮罩")
        self.del_mask_btn.clicked.connect(self.del_mask)
        self.del_mask_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        self.mask_crop_cb = QtWidgets.QCheckBox("不溢出")
        self.mask_crop_cb.setChecked(True)
        self.mask_crop_cb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        self.invert_mask_cb = QtWidgets.QCheckBox("反轉")
        self.invert_mask_cb.setChecked(False)
        self.invert_mask_cb.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.invert_mask_cb.stateChanged.connect(self.reload_mask_with_invert)

        help_btn = QHelpButton(
            "若有載入遮罩圖，僅遮罩圖中不透明（預覽圖爲白色部分）的區域會參與碎片分割，碎片內容可以超出遮罩範圍，但每片至少有部分覆蓋遮罩。"
            "\n\n"
            "勾選「不溢出」後、拆解完成時會立刻將所有碎片以遮罩圖透明度再次裁切，僅保留與遮罩重疊的像素，其他區域會變成透明。"
            "\n\n"
            "勾選「反轉」後，遮罩透明/不透明區域將反過來（遮罩為0變255，255變0）。"
            "\n\n"
            "遮罩圖需為 PNG，且大小須與主圖完全一致。"
        )
        help_btn.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        mask_row = QtWidgets.QHBoxLayout()
        mask_row.addWidget(self.mask_btn)
        mask_row.addWidget(self.del_mask_btn)
        mask_row.addWidget(self.mask_crop_cb)
        mask_row.addWidget(self.invert_mask_cb)
        mask_row.addWidget(help_btn)
        ff.addRow("遮罩：", mask_row)
        self.mask_file_lbl = ClickableFileLabel(self, 'mask')
        ff.addRow("", self.mask_file_lbl)
        right.addLayout(ff)

        # 拆解參數
        self.num_input = QtWidgets.QSpinBox(); self.num_input.setRange(1,10); self.num_input.setValue(3)
        self.block_input = QtWidgets.QSpinBox(); self.block_input.setRange(1,30); self.block_input.setValue(5)
        self.rand_input = QtWidgets.QSpinBox(); self.rand_input.setRange(1,100); self.rand_input.setValue(1)
        ff2 = QtWidgets.QFormLayout()
        row1 = QtWidgets.QHBoxLayout(); row1.addWidget(self.num_input); row1.addWidget(QHelpButton(
            "決定要將圖片分割成幾個碎片，通常越多越難重組與逆向。\n\n優點：數量越多，安全性提升、盜用困難度增加。\n\n缺點：碎片數過多會導致管理困難、記憶體消耗變高。"
        ))
        ff2.addRow("拆分張數(1~10)：", row1)
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
        self.overlap_pixel_input.setValue(0)
        row5 = QtWidgets.QHBoxLayout()
        row5.addWidget(self.overlap_pixel_input)
        row5.addWidget(QHelpButton(
            "拆解後於鏤空區補原圖像素作為重疊像素。\n數值為聯集不透明像素的比例，填補到2~N片隨機碎片。\n\n優點：增加還原難度，讓每片有干擾。\n\n缺點：比例過高會導致效能大幅下降、檔案變大。"
        ))
        ff2.addRow("重疊像素比(0~100%)：", row5)
        self.aggregation_input = QtWidgets.QSpinBox()
        self.aggregation_input.setRange(1, 10)
        self.aggregation_input.setValue(1)
        row6 = QtWidgets.QHBoxLayout()
        row6.addWidget(self.aggregation_input)
        row6.addWidget(QHelpButton(
            "調整回補的重疊像素聚集程度。1=最分散，10=最密集，預設1\n\n優點：可調整碎片間重疊區域型態，提升反逆向性。\n\n缺點：極端值可能造成運算異常或不自然分佈。"
        ))
        ff2.addRow("重疊像素聚合(1~10)：", row6)
        right.addLayout(ff2)

        # 操作按鈕
        cth = QtWidgets.QHBoxLayout()
        self.split_btn = QtWidgets.QPushButton("執行拆解")
        self.split_btn.clicked.connect(self.split)
        cth.addWidget(self.split_btn)
        self.save_btn = QtWidgets.QPushButton("還原初始分割")
        self.save_btn.clicked.connect(self.restore_initial_state)
        cth.addWidget(self.save_btn)
        right.addLayout(cth)

        # 分頁
        self.tabs = QtWidgets.QTabWidget()
        self.fragment_list = QtWidgets.QListWidget()
        self.fragment_list.setContextMenuPolicy(QtCore.Qt.NoContextMenu)  # 關閉右鍵選單
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

        right.addWidget(self.tabs, stretch=1)

        # 版權 / 免責聲明
        self.disclaimer_lbl1 = QtWidgets.QLabel("本工具僅供技術交流與學術用途，不保證碎片不可被還原。")
        self.disclaimer_lbl2 = QtWidgets.QLabel("使用者需自行承擔所有風險。")
        self.disclaimer_lbl3 = QtWidgets.QLabel("© 2025 DuoDuo. 開源授權：MIT License")
        for lbl in [self.disclaimer_lbl1, self.disclaimer_lbl2, self.disclaimer_lbl3]:
            lbl.setStyleSheet("color:#aaa; font-size:11px; margin:1px 0;")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            right.addWidget(lbl)

        main.addLayout(left, stretch=2)
        main.addLayout(right, stretch=1)
        self.switch_panel(False)

    def set_fragment_mask(self, name):
        if name not in self.fragment_data:
            return
        self.fragment_as_mask_name = name
        self.set_status(f"已設定「{name}」為遮罩（本身不會被干擾/劣化）", True)
        self.refresh_fragment_list()

    def clear_fragment_mask(self):
        self.fragment_as_mask_name = None
        self.set_status("已清除碎片遮罩", True)
        self.refresh_fragment_list()

    def get_reference_mask(self, ignore_semitrans=True):
        """
        劣化 / 干擾用的 primary mask：固定用 fragment_order[0]（第一片碎片）。
        回傳 bool array 或 None。
        """
        if not self.fragment_order:
            return None
        first = self.fragment_order[0]
        arr = self.fragment_data.get(first)
        if arr is None:
            return None
        if ignore_semitrans:
            return (arr[..., 3] == 255)
        else:
            return (arr[..., 3] > 0)

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
            im = Image.open(self.mask_img_path)
            arr = pil2np(im)
            if self.invert_mask_cb.isChecked():
                arr = arr.copy()
                arr[..., 3] = 255 - arr[..., 3]
            self.mask_img = arr
            self.set_status("遮罩已重新載入（反轉遮罩變更）", True)
            self.preview_mask_grayalpha(self.mask_img)
            self.set_file_label(self.mask_file_lbl, self.mask_img_path, im)
        except Exception as e:
            self.set_status(f"遮罩重載失敗: {e}", False)
        
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

        self.populate_fragment_list_with_checkboxes()
        self.update_restore_preview()

        self.img_wrap.overlay_btn.setVisible(True)
        self.img_wrap.overlay_btn.setChecked(False)
        self.overlay_active = False
        self.switch_panel(True)

    def cancel_restore_preview(self):
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
        self.set_status("已退出進階管理", True)
        
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
            except Exception as e:
                self.set_status(f"匯入失敗: {e}", False)
        if self.restore_mode:
            self.populate_fragment_list_with_checkboxes()
            self.update_restore_preview()
        else:
            self.populate_fragment_list_no_checkbox()
        self.set_status("已匯入碎片", True)
        self.split_result = [self.fragment_data[name] for name in self.fragment_order]

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

    def batch_rename_fragments(self):
        count = self.fragment_list.count()
        if count == 0: return
        prefix, ok = QtWidgets.QInputDialog.getText(self, "批次命名", "請輸入前綴（例如：碎片）", text="碎片")
        if not ok or not prefix: return
        digits = len(str(count))
        old_names = [self.fragment_list.item(i).text() for i in range(count)]
        imgs = [self.fragment_data[name] for name in old_names if name in self.fragment_data]
        self.fragment_data.clear()
        self.fragment_order.clear()
        for i, img in enumerate(imgs):
            new_name = f"{prefix}_{str(i+1).zfill(digits)}"
            self.fragment_data[new_name] = img
            self.fragment_order.append(new_name)
        self.normalize_fragment_list_order()
        self.set_status(f"已批次命名為 {prefix}_***", True)
        self.refresh_fragment_order()

    def get_current_fragment_order(self):
        names = [self.fragment_list.item(i).text() for i in range(self.fragment_list.count())]
        return [n for n in names if n in self.fragment_data]

    def set_status(self, msg, ok=True):
        if len(msg) > 38:
            msg = ellipsis_middle(msg, 38)
        if msg and (
            "未套用劣化" in msg or
            "劣化預覽中 尚未套用" in msg or
            "干擾像素預覽中" in msg
        ):
            color = "#FFD600"
        else:
            color = "#0f0" if ok else "#f55"
        self.img_wrap.status_lbl.setText(msg)
        self.img_wrap.status_lbl.setStyleSheet(f"color:{color}; font-weight:bold;")
        self.img_wrap.status_lbl.setToolTip(msg)

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
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "選擇主圖 / 干擾素材", "", "PNG圖檔 (*.png)")
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
        im = None
        if fname:
            try:
                im = Image.open(fname)
                arr = pil2np(im)
                if self.invert_mask_cb.isChecked():
                    arr = arr.copy()
                    arr[..., 3] = 255 - arr[..., 3]
                self.mask_img = arr
                self.mask_img_path = fname
                self.set_status("遮罩載入成功", True)
                self.preview_mask_grayalpha(self.mask_img)
                self.set_file_label(self.mask_file_lbl, self.mask_img_path, im)
            except Exception as e:
                self.set_status(f"遮罩載入失敗: {e}", False)
                self.mask_img = None
                self.mask_img_path = ""
                self.mask_file_lbl.setText("")
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

        if self.main_img is None:
            self.set_status("請先載入主圖", False)
            return

        n = self.num_input.value()
        mask = self.mask_img if self.mask_img is not None else None

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

        if n == 1 and mask is None:
            arr = self.main_img.copy()
            self.split_result = [arr]
            self.fragment_list.clear()
            self.fragment_data.clear()
            self.fragment_order.clear()
            name = "碎片 1"
            item = QtWidgets.QListWidgetItem(name)
            self.fragment_list.addItem(item)
            self.fragment_data[name] = arr
            self.fragment_order.append(name)
            self.img_wrap.preview.set_image(arr)
            self.set_status("已直接建立唯一碎片", True)
            self._initial_snapshot = {
                'fragment_names': [name],
                'fragment_imgs': [arr.copy()]
            }
            return

        block_sz = self.block_input.value()
        rand_sz = self.rand_input.value()
        self.set_status("拆解中... 0%", True)
        self.fragment_list.clear()
        self.fragment_data.clear()
        self.split_btn.setEnabled(False)
        pil_main = np2pil(self.main_img)
        pil_mask = np2pil(mask) if mask is not None else None
        self.split_thread = SplitThread(
            pil_main, pil_mask, n, block_sz, rand_sz,
            strict_mask=(self.mask_crop_cb.isChecked() if hasattr(self, "mask_crop_cb") else True)
        )
        self.split_thread.update_progress.connect(self.progress)
        self.split_thread.result.connect(self.split_done)
        self._split_start_time = time.time()
        self.split_thread.start()

    def progress(self, done, total, msg):
        if msg:
            self.set_status(f"{msg} ({done}/{total})", True)
        else:
            self.set_status(f"拆解中... ({done}/{total})", True)

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
        for idx, arr in enumerate(images):
            name = f"碎片 {idx+1}"
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
        self.cancel_restore_preview()
        self._initial_snapshot = {
            'fragment_names': [f"碎片 {i+1}" for i in range(len(images))],
            'fragment_imgs': images[:]
        }

            
    def restore_initial_state(self):
        if not self._initial_snapshot:
            QtWidgets.QMessageBox.information(self, "還原初始設定", "尚未有分割過的結果可還原！")
            return
        self.fragment_data.clear()
        self.fragment_order.clear()
        for name, img in zip(self._initial_snapshot['fragment_names'], self._initial_snapshot['fragment_imgs']):
            self.fragment_data[name] = img
            self.fragment_order.append(name)
        self.split_result = self._initial_snapshot['fragment_imgs'][:]
        self.normalize_fragment_list_order()
        self.img_wrap.preview.set_image(self.split_result[0])
        self.set_status("已還原初始分割狀態", True)
        self.cancel_restore_preview()

    def normalize_fragment_list_order(self):
        try:
            # 過濾 fragment_order，只保留 fragment_data 有的
            self.fragment_order = [name for name in self.fragment_order if name in self.fragment_data]
            # 移除 fragment_data 裡沒有在 order 裡的殘影
            keys_to_del = [k for k in self.fragment_data if k not in self.fragment_order]
            for k in keys_to_del:
                del self.fragment_data[k]

            self.fragment_list.clear()

            if self.restore_mode:
                for name in self.fragment_order:
                    if name in self.fragment_data:
                        item = QtWidgets.QListWidgetItem(name)
                        item.setFlags(
                            item.flags() | QtCore.Qt.ItemIsUserCheckable |
                            QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled
                        )
                        item.setCheckState(QtCore.Qt.Checked)
                        self.fragment_list.addItem(item)
            else:
                for name in self.fragment_order:
                    if name in self.fragment_data:
                        item = QtWidgets.QListWidgetItem(name)
                        self.fragment_list.addItem(item)

            if self.fragment_list.count() > 0:
                self.fragment_list.setCurrentRow(0)
            else:
                self.fragment_list.setCurrentRow(-1)

            self.overlay_active = False

        except Exception as e:
            print("normalize_fragment_list_order 錯誤：", e)
            # 可選：這裡可以加 QMessageBox 警告或 log 紀錄

    def populate_fragment_list_no_checkbox(self):
        self.fragment_list.clear()
        self.fragment_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.fragment_list.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)
        try:
            self.fragment_list.itemClicked.disconnect(self.fragment_clicked)
        except Exception:
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
        # 恢復樣式與選取狀態
        self.fragment_list.setStyleSheet("")
        self.fragment_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.fragment_list.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.fragment_list.clearSelection()
        if self.fragment_list.count() > 0:
            self.fragment_list.setCurrentRow(0)
        self.fragment_list.setFocus(QtCore.Qt.OtherFocusReason)     
        if self.fragment_list.count() > 0:
            self.fragment_list.setCurrentRow(0)
            # 強制觸發點擊事件（如果你的預覽是靠 fragment_clicked 完成的）
            item = self.fragment_list.item(0)
            if item:
                self.fragment_clicked(item)
                
    def populate_fragment_list_with_checkboxes(self):
        self.fragment_list.clear()

        if self.restore_mode:
            # 進階還原模式：不顯示選取與 hover 高亮
            self.fragment_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            self.fragment_list.setFocusPolicy(QtCore.Qt.NoFocus)
            self.fragment_list.clearSelection()
            self.fragment_list.setStyleSheet("""
                QListWidget::item {
                    background: transparent;
                }
                QListWidget::item:hover {
                    background: transparent;
                }
                QListWidget::item:selected {
                    background: transparent;
                }
            """)
        else:
            self.fragment_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            # 恢復預設行為（如果之前有套 style 可能需要清掉）
            self.fragment_list.setFocusPolicy(QtCore.Qt.StrongFocus)
            self.fragment_list.setStyleSheet("")  # 恢復預設
            self.fragment_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            self.fragment_list.setFocusPolicy(QtCore.Qt.StrongFocus)

        self.fragment_list.setDragDropMode(QtWidgets.QAbstractItemView.NoDragDrop)

        for name in self.fragment_order:
            if name in self.fragment_data:
                item = QtWidgets.QListWidgetItem(name)
                # restore_mode 也要可打勾，所以保留 user-checkable
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                item.setCheckState(QtCore.Qt.Checked)
                self.fragment_list.addItem(item)

        try:
            self.fragment_list.itemChanged.disconnect(self.on_restore_item_changed)
        except Exception:
            pass
        self.fragment_list.itemChanged.connect(self.on_restore_item_changed)

        if self.restore_mode:
            self.fragment_list.clearSelection()

    def _on_fragment_list_rows_moved(self, parent, start, end, dest, row):
        self.fragment_order = [
            self.fragment_list.item(i).text()
            for i in range(self.fragment_list.count())
            if self.fragment_list.item(i).text() in self.fragment_data
        ]
    def on_restore_item_changed(self, item):
        self.update_restore_preview()     
        
    def force_normal_preview(self):
        self.restore_mode = False
        self.fragment_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
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

        # 1) 取得「清單順序（由上到下）」的已勾選碎片名
        names_in_list_order = []
        for i in range(self.fragment_list.count()):
            it = self.fragment_list.item(i)
            if it.checkState() == QtCore.Qt.Checked:
                names_in_list_order.append(it.text())

        if not names_in_list_order:
            self.set_status("未勾選碎片", False)
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
            export_img = img
            if (hasattr(self, "mask_crop_cb")
                    and self.mask_crop_cb.isChecked()
                    and getattr(self, "mask_img", None) is not None):
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
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "儲存所有碎片", "所有碎片.zip", "ZIP 壓縮檔 (*.zip)")
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
            if not self.restore_mode:
                if len(self.recycle_bin) > 0:
                    last_name, last_img = self.recycle_bin[-1]
                    self.img_wrap.preview.set_image(last_img, trash_highlight=True)
                else:
                    self.img_wrap.preview.set_image(None, trash_highlight=True)
            self.img_wrap.status_lbl.setText("翻找垃圾桶中")
            self.img_wrap.status_lbl.setStyleSheet("color:#f55; font-weight:bold;")
            return

        # 非垃圾桶頁籤先清除 highlight
        self.trash_tab.clear_highlight()

        if current == self.degrade_tab:
            # 劣化頁籤：優先顯示預覽
            if getattr(self, 'degrade_preview_pending', None):
                self.img_wrap.preview.set_image(self.degrade_preview_pending['degraded'])
                self.set_status("劣化預覽（尚未套用）", True)
            elif getattr(self, 'degrade_source_img', None) is not None:
                self.img_wrap.preview.set_image(self.degrade_source_img)
                self.set_status("已載入劣化來源圖", True)
            else:
                # 無劣化資料 fallback 原本行為
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
            return

        # 其他非垃圾桶、非劣化的分頁（例如碎片管理/干擾）
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
        self.set_status("", True)               

    def closeEvent(self, event):
        for attr in ("split_thread", "overlap_thread", "gen_thread", "degrade_thread"):
            th = getattr(self, attr, None)
            if th is not None and hasattr(th, "isRunning") and th.isRunning():
                try:
                    th.abort()
                    th.wait(2000)
                except Exception:
                    pass
        super().closeEvent(event)

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
        # ... 你原本的前置不動 ...

        # 取得 settings
        kw = self.interfere_panel.get_settings()
        primary_mask = self.get_reference_mask(kw.get('ignore_semitrans', True))
        if primary_mask is None:
            QtWidgets.QMessageBox.warning(self, "錯誤", "找不到用來當作遮罩的碎片（用於產生干擾）")
            return
        primary_mask = primary_mask.astype(bool)

        # 只產生一次 block_pool！重用
        block_pool = build_interfere_block_pool(
            self.main_img.copy(),
            kw['block_size'],
            kw['random_range'],
            pool_size=300,
            alpha_min=kw.get('alpha_min', 1),
            alpha_max=kw.get('alpha_max', 100),
            restrict_mask=primary_mask
        )

        # ... 你原本的 fragment 處理也不變 ...
        if not self.fragment_order:
            QtWidgets.QMessageBox.warning(self, "錯誤", "沒有碎片可做干擾")
            return
        mask_fragment = self.fragment_order[0]
        filtered_fragment_data = {
            name: img for name, img in self.fragment_data.items()
            if name != mask_fragment
        }
        filtered_fragment_order = [n for n in self.fragment_order if n != mask_fragment]
        if not filtered_fragment_order:
            QtWidgets.QMessageBox.warning(self, "錯誤", "沒有可做干擾的碎片（只有一片）")
            return
        if not filtered_fragment_order:
            QtWidgets.QMessageBox.warning(self, "錯誤", "沒有可做干擾的碎片（全部被設定為遮罩）")
            return

        self.interfere_images_dict.clear()
        self.set_status("開始產生干擾像素...", True)
        QtWidgets.QApplication.processEvents()

        # 傳 block_pool 進 thread
        self.gen_thread = InterfereGenThread(
            filtered_fragment_data,
            kw,
            filtered_fragment_order,
            primary_mask,
            block_pool     # 新增參數！
        )
        self.gen_thread.progress.connect(
            lambda cur, tot, msg: self.set_status(f"{msg} ({cur}/{tot})", True)
        )

        def done(result_dict):
            self.interfere_images_dict = result_dict

            self.force_normal_preview()   # 直接呼叫新 function

            # 選第一個碎片並預覽
            if self.fragment_list.count() > 0:
                self.fragment_list.setCurrentRow(0)
                current = self.fragment_list.currentItem()
                if current:
                    self.fragment_clicked(current)

            QtWidgets.QMessageBox.information(
                self, "產生完成",
                "已為每片碎片產生干擾像素，可於「碎片管理」預覽，檢視結果後再到「干擾像素」執行「合成到碎片」。"
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
        QtWidgets.QMessageBox.information(self, "產生完成", "已為每片碎片產生干擾像素，可執行執行「合成到碎片」或重複執行疊加。")

    def apply_interfere_to_fragments(self):
        if not self.interfere_images_dict:
            QtWidgets.QMessageBox.information(self, "尚未產生", "請先產生干擾像素")
            return
        if not self.fragment_data or not self.fragment_order:
            QtWidgets.QMessageBox.warning(self, "錯誤", "找不到碎片")
            return

        ignore_semitrans = self.interfere_panel.ignore_semitrans_cb.isChecked()
        primary_mask = self.get_reference_mask(ignore_semitrans)
        if primary_mask is None:
            QtWidgets.QMessageBox.warning(self, "錯誤", "找不到遮罩碎片")
            return

        new_fragment_data = {}
        new_fragment_order = []

        # 第一片（mask）保留原樣
        first = self.fragment_order[0]
        if first in self.fragment_data:
            new_fragment_data[first] = self.fragment_data[first]
            new_fragment_order.append(first)

        cnt = 0
        for name in self.fragment_order[1:]:
            interfere = self.interfere_images_dict.get(name)
            orig = self.fragment_data.get(name)
            if orig is not None and interfere is not None and interfere.shape == orig.shape:
                merged = apply_interfere_masked(orig, interfere, primary_mask)
                new_name = self.get_unique_name(name + "_干擾")
                new_fragment_data[new_name] = merged
                new_fragment_order.append(new_name)
                self.recycle_bin.append((name, orig))
                cnt += 1
            else:
                new_fragment_data[name] = self.fragment_data.get(name)
                new_fragment_order.append(name)

        self.fragment_data = new_fragment_data
        self.fragment_order = new_fragment_order
        self.interfere_images_dict.clear()
        self.populate_fragment_list_no_checkbox()
        self.trash_tab.refresh()
        QtWidgets.QMessageBox.information(self, "合成完成", f"已合成到 {cnt} 個碎片（不包含遮罩碎片），原碎片移入垃圾桶")
        self.tabs.setCurrentIndex(0)
        self.cancel_restore_preview()
    def on_import_degrade_source(self):
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(self, "匯入劣化來源圖", "", "PNG圖檔 (*.png)")
        if not fname:
            return
        try:
            im = Image.open(fname).convert("RGBA")
            arr = pil2np(im)
            self.degrade_source_img = arr
            self.degrade_preview_pending = None
            self.img_wrap.preview.set_image(arr)
            self.set_status("已載入劣化來源圖", True)
            self.degrade_panel.set_imported_filename(fname)
        except Exception as e:
            self.set_status(f"匯入失敗: {e}", False)

    def on_generate_degrade_preview_shared(self):
        if self.degrade_source_img is None:
            self.set_status("請先匯入劣化來源圖", False)
            return

        # 如果已有尚未套用的預覽，才跳提示
        if self.degrade_preview_pending:
            reply = QtWidgets.QMessageBox.question(
                self,
                "未套用的劣化預覽",
                "目前有尚未套用的劣化預覽。要先匯出（套用）它，捨棄，還是取消產生新預覽？\n\n"
                "是：先匯出並再產生新預覽。\n"
                "否：捨棄目前預覽並產生新預覽。\n"
                "取消：不做任何事。",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Yes
            )
            if reply == QtWidgets.QMessageBox.Yes:
                # 先套用（匯出），再繼續產生
                self.on_apply_degrade_source()
            elif reply == QtWidgets.QMessageBox.No:
                # 捨棄現有預覽
                self.degrade_preview_pending = None
            else:
                # 取消：不產生新的預覽
                return

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
            self.set_status("劣化預覽（尚未套用）", True)
            self.degrade_panel.gen_preview_btn.setEnabled(True)
            self.degrade_panel.apply_export_btn.setEnabled(True)

        self.degrade_thread.result.connect(finish)
        self.degrade_thread.start()

    def on_restore_degrade_source(self):
        if getattr(self, 'degrade_source_img', None) is not None:
            self.img_wrap.preview.set_image(self.degrade_source_img)
            self.set_status("顯示原始劣化來源圖", True)

    def on_apply_degrade_source(self):
        if not getattr(self, 'degrade_preview_pending', None):
            QtWidgets.QMessageBox.information(self, "無預覽", "請先產生劣化預覽")
            return
        degraded = self.degrade_preview_pending['degraded']
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self, "儲存劣化圖", "degraded.png", "PNG圖檔 (*.png)")
        if not fn:
            return
        try:
            np2pil(degraded).save(fn)
            self.set_status(f"已儲存劣化圖: {os.path.basename(fn)}", True)
            self.degrade_preview_pending = None
        except Exception as e:
            self.set_status(f"儲存失敗: {e}", False)

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
            # 進階恢復模式下不要顯示單一碎片，改用勾選的碎片合成
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

def apply_overlap_to_fragment(frag, overlap_img):
    mask = (frag[...,3] == 0) & (overlap_img[...,3] > 0)
    out = frag.copy()
    out[mask] = overlap_img[mask]
    return out

def overlap_merge_worker(args):
    frag, overlap_img = args
    return apply_overlap_to_fragment(frag, overlap_img)

def overlap_fill_fragment_agg_worker(args):
    import numpy as np, random
    # args: (frag, main_img, mask_img, fill_percent, block_size, aggregation, limit_to_mask)
    frag, main_img, mask_img, fill_percent, block_size, aggregation, limit_to_mask = args

    if main_img is None or fill_percent <= 0:
        return frag

    arr = main_img
    h, w = arr.shape[:2]

    # 只在「遮罩內 ∧ 主圖 alpha==255」視為有效區；沒遮罩就退回主圖 alpha==255
    if mask_img is not None:
        valid = (mask_img[..., 3] == 255) & (arr[..., 3] == 255)
    else:
        valid = (arr[..., 3] == 255)

    # 這片碎片「可填區」= 該片透明 ∧ valid
    frag_trans = (frag[..., 3] == 0)
    frag_valid = frag_trans & valid
    coords = np.argwhere(frag_valid)
    total = int(coords.shape[0])
    if total == 0:
        return frag

    # 目標量以「本片可填區」計算
    target_fill = int(total * float(fill_percent) / 100.0)
    if target_fill <= 0:
        return frag

    filled = np.zeros((h, w), dtype=bool)
    out = frag.copy()

    agg_ratio = min(max(int(aggregation), 1), 10) / 10.0  # 1~10 -> 0.1~1.0
    cluster_centers = []
    fill_cnt = 0
    tries = 0
    fail_streak = 0
    try_factor = 8
    fail_limit = 4000
    max_tries = max(1000, int(target_fill * try_factor))

    while fill_cnt < target_fill and tries < max_tries and fail_streak < fail_limit:
        # 聚合取樣（優先靠近已填中心），否則從本片可填池抽
        if cluster_centers and (random.random() < agg_ratio):
            base_y, base_x = random.choice(cluster_centers)
            dy = random.randint(-block_size, block_size)
            dx = random.randint(-block_size, block_size)
            y, x = base_y + dy, base_x + dx
            if not (0 <= y < h and 0 <= x < w and frag_valid[y, x]):
                y, x = coords[random.randrange(len(coords))]
        else:
            y, x = coords[random.randrange(len(coords))]

        sz = block_size  # 固定大小
        if x + sz > w or y + sz > h or y < 0 or x < 0:
            tries += 1; fail_streak += 1
            continue

        patch = arr[y:y+sz, x:x+sz]
        # 僅在「主圖非透明 ∧ 本片可填區 ∧ 尚未填過」填入
        mask_patch = (patch[..., 3] == 255)
        local_valid = frag_valid[y:y+sz, x:x+sz]
        target = (~filled[y:y+sz, x:x+sz]) & mask_patch & local_valid
        if not np.any(target):
            tries += 1; fail_streak += 1
            continue

        out_region = out[y:y+sz, x:x+sz]
        out_region[target] = patch[target]
        filled[y:y+sz, x:x+sz][target] = True

        inc = int(np.count_nonzero(target))
        fill_cnt += inc
        if inc > 0:
            fail_streak = 0
            ys, xs = np.where(target)
            k = min(5, ys.size)
            if k > 0:
                idxs = np.random.choice(ys.size, size=k, replace=False)
                for ii in idxs:
                    cluster_centers.append((y + int(ys[ii]), x + int(xs[ii])))

        tries += 1

    return out

def apply_overlap_to_all_fragments_mp(
    fragments, main_img, mask_img, fill_percent, block_size, random_range,
    progress_cb=None, abort_cb=None, limit_to_mask=True, aggregation=1
):
    # 每片碎片個別聚合（固定 block_size），嚴格限制在遮罩∧主圖可見，並以本片可填區為基準
    N = len(fragments)
    if fill_percent <= 0 or N == 0 or main_img is None:
        return fragments

    from concurrent.futures import ProcessPoolExecutor, as_completed
    args_list = [
        (frag, main_img, mask_img, fill_percent, block_size, aggregation, limit_to_mask)
        for frag in fragments
    ]
    results = [None] * N

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(overlap_fill_fragment_agg_worker, arg): idx for idx, arg in enumerate(args_list)}
        done_cnt = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            if abort_cb is not None and abort_cb():
                executor.shutdown(wait=False, cancel_futures=True)
                raise Exception("Aborted by user")
            results[idx] = fut.result()
            done_cnt += 1
            if progress_cb:
                progress_cb(done_cnt, N, "重疊像素合成.")
    return results

if __name__ == "__main__":
    mp.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())