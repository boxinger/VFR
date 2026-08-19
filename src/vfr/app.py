from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QCoreApplication, QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

from vfr.ui.main_window import MainWindow


def build_application(argv: list[str] | None = None) -> QApplication:
    application = QApplication.instance()
    if application is None:
        application = QApplication(argv or sys.argv)
    QCoreApplication.setOrganizationName("VFR")
    QCoreApplication.setApplicationName("VFR Controller Designer")
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            application.setFont(QFont(families[0], 9))
    pg.setConfigOptions(antialias=True, background="#171a21", foreground="#d9dde7")
    return application


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VFR 控制器频响设计器")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="创建主窗口并立即退出，用于环境验证",
    )
    arguments, qt_arguments = parser.parse_known_args(argv)
    application = build_application([sys.argv[0], *qt_arguments])
    window = MainWindow()
    window.show()
    if arguments.smoke_test:
        QTimer.singleShot(50, application.quit)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
