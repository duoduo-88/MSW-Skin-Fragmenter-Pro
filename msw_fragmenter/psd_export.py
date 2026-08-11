import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

try:
    from psd_tools import PSDImage
    from psd_tools.api.layers import PixelLayer
except ImportError:  # 讓主程式在缺少選用套件時仍可啟動。
    PSDImage = None
    PixelLayer = None


INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*]')


@dataclass(frozen=True)
class PsdLayerTarget:
    key: str
    label: str
    index_path: tuple[int, ...]


def ensure_psd_support():
    if PSDImage is None or PixelLayer is None:
        raise RuntimeError(
            "尚未安裝 PSD 匯出套件，請先執行：pip install psd-tools>=1.10"
        )


def list_psd_templates(template_dir):
    folder = Path(template_dir)
    folder.mkdir(parents=True, exist_ok=True)
    return sorted(
        (path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == ".psd"),
        key=lambda path: path.name.casefold(),
    )


def clean_psd_layer_name(name):
    """移除 PSD 名稱中的 NUL 等不可顯示控制字元。"""
    return "".join(
        char
        for char in str(name or "")
        if unicodedata.category(char) not in {"Cc", "Cf"}
    ).strip()


def read_psd_layer_targets(psd_path):
    """依 Photoshop 面板由上到下列出圖層，索引路徑仍指向原始結構。"""
    ensure_psd_support()
    psd = PSDImage.open(psd_path)
    return _collect_psd_layer_targets(psd)


def _collect_psd_layer_targets(psd):
    targets = []

    def visit(container, parent_names=(), parent_path=()):
        # psd-tools 容器順序為底到頂；Photoshop 圖層面板顯示為頂到底。
        for index in range(len(container) - 1, -1, -1):
            layer = container[index]
            current_path = parent_path + (index,)
            display_name = clean_psd_layer_name(layer.name) or f"未命名圖層 {index + 1}"
            current_names = parent_names + (display_name,)
            if layer.is_group():
                visit(layer, current_names, current_path)
            else:
                targets.append(
                    PsdLayerTarget(
                        key="/".join(map(str, current_path)),
                        label=" / ".join(current_names),
                        index_path=current_path,
                    )
                )

    visit(psd)
    return targets


def _resolve_layer(psd, index_path):
    layer = psd
    for index in index_path:
        layer = layer[index]
    return layer


def _composite_fragments(fragment_names, fragment_data, size):
    width, height = size
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))
    # 清單最上方代表最上層，因此由清單底部往上合成。
    for name in reversed(fragment_names):
        array = np.asarray(fragment_data[name], dtype=np.uint8)
        if array.shape[:2] != (height, width):
            raise ValueError(
                f"碎片「{name}」尺寸為 {array.shape[1]}×{array.shape[0]}，"
                f"但 PSD 尺寸為 {width}×{height}"
            )
        canvas = Image.alpha_composite(canvas, Image.fromarray(array, mode="RGBA"))
    return canvas


def _safe_output_prefix(prefix):
    prefix = str(prefix or "").strip()
    if INVALID_FILENAME_CHARS.search(prefix):
        raise ValueError('檔名前綴不能包含 < > : " / \\ | ? *')
    if prefix.endswith("."):
        raise ValueError("檔名前綴不能以句點結尾")
    return prefix


class PsdExportCancelled(RuntimeError):
    pass


def export_psd_assignments(
    assignments,
    fragment_data,
    output_dir,
    output_prefix,
    progress_cb=None,
    abort_cb=None,
):
    """按範本分組匯出；同一範本的多個碎片只產生一份 PSD。"""
    ensure_psd_support()
    output_folder = Path(output_dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_prefix = _safe_output_prefix(output_prefix)

    by_template = {}
    for assignment in assignments:
        template = str(Path(assignment["template_path"]).resolve())
        by_template.setdefault(template, []).append(assignment)

    if not by_template:
        raise ValueError("沒有可匯出的 PSD 設定")

    total_steps = sum(
        2 + len({assignment["target_key"] for assignment in template_assignments})
        for template_assignments in by_template.values()
    )
    completed_steps = 0

    def check_abort():
        if abort_cb is not None and abort_cb():
            raise PsdExportCancelled("PSD 匯出已取消")

    def report(message):
        if progress_cb is not None:
            progress_cb(completed_steps, total_steps, message)

    used_output_names = set()
    exported_paths = []
    for template_path, template_assignments in by_template.items():
        check_abort()
        template_label = Path(template_path).name
        report(f"正在讀取 PSD 範本：{template_label}")
        psd = PSDImage.open(template_path)
        targets = {target.key: target for target in _collect_psd_layer_targets(psd)}
        completed_steps += 1
        report(f"已讀取 PSD 範本：{template_label}")
        by_target = {}
        for assignment in template_assignments:
            key = assignment["target_key"]
            if key not in targets:
                raise ValueError(
                    f"PSD「{Path(template_path).name}」找不到指定圖層，請重新選擇"
                )
            by_target.setdefault(key, []).append(assignment["fragment_name"])

        # 取代元素不會改變父容器長度與索引，因此其他索引路徑保持有效。
        for target_key, fragment_names in by_target.items():
            check_abort()
            target = targets[target_key]
            report(f"正在寫入圖層：{target.label}")
            old_layer = _resolve_layer(psd, target.index_path)
            parent = old_layer.parent
            layer_index = parent.index(old_layer)
            image = _composite_fragments(fragment_names, fragment_data, psd.size)
            new_layer = PixelLayer.frompil(
                image,
                psd_file=psd,
                layer_name=(
                    clean_psd_layer_name(old_layer.name)
                    or f"碎片圖層 {layer_index + 1}"
                ),
                top=0,
                left=0,
            )
            for attr in ("visible", "opacity", "blend_mode"):
                try:
                    setattr(new_layer, attr, getattr(old_layer, attr))
                except (AttributeError, ValueError):
                    pass
            parent.remove(old_layer)
            parent.insert(layer_index, new_layer)
            completed_steps += 1
            report(f"已寫入圖層：{target.label}")

        template_name = Path(template_path).name
        candidate_stem = f"{output_prefix}{Path(template_name).stem}"
        unique_stem = candidate_stem
        suffix = 2
        while unique_stem.casefold() in used_output_names:
            unique_stem = f"{candidate_stem}_{suffix}"
            suffix += 1
        used_output_names.add(unique_stem.casefold())
        output_path = output_folder / f"{unique_stem}.psd"
        if os.path.normcase(os.path.abspath(output_path)) == os.path.normcase(
            os.path.abspath(template_path)
        ):
            raise ValueError("匯出位置不能覆蓋專案內的 PSD 範本")
        check_abort()
        report(f"正在儲存：{output_path.name}")
        psd.save(output_path)
        exported_paths.append(output_path)
        completed_steps += 1
        report(f"已儲存：{output_path.name}")
    return exported_paths


class PsdExportWorker(QtCore.QThread):
    progress = QtCore.Signal(int, int, str)
    result = QtCore.Signal(object)
    error = QtCore.Signal(str)

    def __init__(
        self,
        assignments,
        fragment_data,
        output_dir,
        output_prefix,
        parent=None,
    ):
        super().__init__(parent)
        self.assignments_data = list(assignments)
        self.fragment_data = fragment_data
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            paths = export_psd_assignments(
                self.assignments_data,
                self.fragment_data,
                self.output_dir,
                self.output_prefix,
                progress_cb=self.progress.emit,
                abort_cb=lambda: self._abort,
            )
            if not self._abort:
                self.result.emit(paths)
        except PsdExportCancelled:
            self.error.emit("PSD 匯出已取消")
        except Exception as exc:
            self.error.emit(str(exc))


class PsdExportDialog(QtWidgets.QDialog):
    def __init__(self, fragment_names, template_dir, default_prefix, parent=None):
        super().__init__(parent)
        self.fragment_names = list(fragment_names)
        self.template_dir = Path(template_dir)
        self.layer_cache = {}
        self.setWindowTitle("匯出 .psd 設定")
        self.resize(760, 460)

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(
            "為每個碎片選擇 PSD 範本與要取代的圖層。相同 PSD 範本的碎片會合併到同一份輸出檔。"
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        folder_row = QtWidgets.QHBoxLayout()
        folder_row.addWidget(QtWidgets.QLabel("PSD 範本資料夾："))
        self.folder_label = QtWidgets.QLineEdit(str(self.template_dir))
        self.folder_label.setReadOnly(True)
        folder_row.addWidget(self.folder_label, 1)
        open_button = QtWidgets.QPushButton("開啟資料夾")
        open_button.clicked.connect(self.open_template_folder)
        folder_row.addWidget(open_button)
        refresh_button = QtWidgets.QPushButton("重新整理")
        refresh_button.clicked.connect(self.reload_templates)
        folder_row.addWidget(refresh_button)
        layout.addLayout(folder_row)

        self.table = QtWidgets.QTableWidget(len(self.fragment_names), 3)
        self.table.setHorizontalHeaderLabels(("碎片", "PSD 檔案", "PSD 圖層"))
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        for row, name in enumerate(self.fragment_names):
            item = QtWidgets.QTableWidgetItem(name)
            item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
            self.table.setItem(row, 0, item)
            template_combo = QtWidgets.QComboBox()
            layer_combo = QtWidgets.QComboBox()
            template_combo.currentIndexChanged.connect(
                lambda _index, current_row=row: self.reload_layers(current_row)
            )
            self.table.setCellWidget(row, 1, template_combo)
            self.table.setCellWidget(row, 2, layer_combo)
        layout.addWidget(self.table, 1)

        filename_row = QtWidgets.QHBoxLayout()
        filename_row.addWidget(QtWidgets.QLabel("匯出檔名前綴："))
        self.filename_edit = QtWidgets.QLineEdit(default_prefix)
        self.filename_edit.setPlaceholderText("例如：角色_（可留空）")
        filename_row.addWidget(self.filename_edit, 1)
        filename_row.addWidget(QtWidgets.QLabel("＋ 原始 PSD 檔名"))
        layout.addLayout(filename_row)

        self.message_label = QtWidgets.QLabel()
        self.message_label.setWordWrap(True)
        layout.addWidget(self.message_label)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok
        )
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText("匯出")
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        self.export_button = buttons.button(QtWidgets.QDialogButtonBox.Ok)
        layout.addWidget(buttons)
        self.reload_templates()

    def open_template_folder(self):
        self.template_dir.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(self.template_dir.resolve()))
        )

    def reload_templates(self):
        try:
            ensure_psd_support()
            templates = list_psd_templates(self.template_dir)
            error = ""
        except Exception as exc:
            templates = []
            error = str(exc)
        for row in range(self.table.rowCount()):
            combo = self.table.cellWidget(row, 1)
            previous = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            for path in templates:
                combo.addItem(path.name, str(path.resolve()))
            if previous:
                index = combo.findData(previous)
                if index >= 0:
                    combo.setCurrentIndex(index)
            combo.blockSignals(False)
            self.reload_layers(row)
        if error:
            self.message_label.setText(error)
        elif not templates:
            self.message_label.setText("請先把 .psd 檔案放進上方的 PSD 範本資料夾，再按「重新整理」。")
        else:
            self.message_label.setText(
                "選到同一範本與同一圖層的多個碎片，會依目前碎片清單順序疊合後寫入該圖層。"
            )
        self.export_button.setEnabled(bool(templates) and not error)

    def reload_layers(self, row):
        template_combo = self.table.cellWidget(row, 1)
        layer_combo = self.table.cellWidget(row, 2)
        template_path = template_combo.currentData()
        layer_combo.clear()
        if not template_path:
            return
        try:
            if template_path not in self.layer_cache:
                self.layer_cache[template_path] = read_psd_layer_targets(template_path)
            targets = self.layer_cache[template_path]
            for target in targets:
                layer_combo.addItem(target.label, target.key)
            if not targets:
                layer_combo.addItem("（PSD 沒有可取代的圖層）", None)
        except Exception as exc:
            layer_combo.addItem(f"（讀取失敗：{exc}）", None)

    def _validate_and_accept(self):
        try:
            _safe_output_prefix(self.filename_edit.text())
            for row in range(self.table.rowCount()):
                if not self.table.cellWidget(row, 1).currentData():
                    raise ValueError(f"「{self.fragment_names[row]}」尚未選擇 PSD 檔案")
                if self.table.cellWidget(row, 2).currentData() is None:
                    raise ValueError(f"「{self.fragment_names[row]}」尚未選擇 PSD 圖層")
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "PSD 匯出設定", str(exc))
            return
        self.accept()

    def assignments(self):
        result = []
        for row, fragment_name in enumerate(self.fragment_names):
            result.append(
                {
                    "fragment_name": fragment_name,
                    "template_path": self.table.cellWidget(row, 1).currentData(),
                    "target_key": self.table.cellWidget(row, 2).currentData(),
                }
            )
        return result

    def output_prefix(self):
        return self.filename_edit.text().strip()
