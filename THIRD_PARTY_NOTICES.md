# Third-Party Notices

The MIT License in this repository applies to MSW Skin Fragmenter Pro source code written for this project. Third-party packages and their transitive components retain their own copyright and license terms.

## Direct Python Dependencies

| Component | Use | License information |
| --- | --- | --- |
| [Qt for Python / PySide6](https://doc.qt.io/qtforpython-6/) | Desktop user interface | Available under LGPLv3/GPLv3 and the Qt commercial license. See the [official licensing documentation](https://doc.qt.io/qtforpython-6/licenses.html). |
| [NumPy](https://numpy.org/) | Array and pixel processing | [BSD 3-Clause License](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| [Pillow](https://python-pillow.github.io/) | Image loading, processing, and export | [MIT-CMU License](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |

The unpinned entries in `requirements.txt` do not select fixed dependency versions. The license files shipped with the versions actually installed or bundled are authoritative.

## Packaged Builds

Executable distributions may include PySide6/Qt, Shiboken, NumPy, Pillow, Python, packaging tools, and additional transitive libraries. Anyone creating or redistributing a packaged build must preserve the notices and license files supplied with those exact components and comply with the chosen Qt/PySide6 licensing terms. The project-level MIT License does not replace those obligations.
