import multiprocessing as mp
import sys

from PySide6 import QtWidgets

from msw_fragmenter.ui import MainWindow


def main():
    mp.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
