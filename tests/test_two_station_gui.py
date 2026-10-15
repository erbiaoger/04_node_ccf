"""Offscreen Qt workflow: editing, gate, persistence, navigation and worker lifecycle."""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from dasqt.app.bridge.log_bus import BridgeLogBus
from dasqt.features.node_two_station.gui import NodeTwoStationWindow
from PyQt6.QtWidgets import QApplication
from test_two_station import make_task


def test_gui_pair_navigation_edit_undo_and_reference_gate(tmp_path):
    app = QApplication.instance() or QApplication([])
    task = make_task(tmp_path)
    task.set_reference([[1, 2000], [10, 2000]], True)
    window = NodeTwoStationWindow(bridge=BridgeLogBus(parent=app), options=task.p)
    window.loaded(task)
    window.show()
    app.processEvents()
    assert window.pair_list.count() == 3
    window.pair_list.setCurrentRow(2)
    app.processEvents()
    window.target.setCurrentText("当前配对")
    event = SimpleNamespace(inaxes=window.pair_axis, xdata=3.0, ydata=2100.0, button=1)
    window.click(event)
    assert task.pair(2)["manual_edited"]
    window.undo()
    assert not task.pair(2)["manual_edited"]
    window.full()
    assert (
        "样本已复核" in window.log.toPlainText()
        or "先确认参考" in window.log.toPlainText()
    )
    window.target.setCurrentText("参考")
    window.click(
        SimpleNamespace(inaxes=window.reference_axis, xdata=3.0, ydata=2100.0, button=1)
    )
    assert not task.confirmed
    window.undo()
    assert task.confirmed
    window.search.setText("1")
    assert window.pair_list.count() == 2
    window.search.clear()
    window.export_figure = lambda: None
    filename = tmp_path / "node_preview.png"
    window.figure.savefig(filename, dpi=100)
    assert filename.is_file()
    reopened = make_task(tmp_path)
    # fixture recreates same contents; source hash identity stays stable.
    assert reopened.confirmed
    window.close()


def test_background_cancel_keeps_ui_alive(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = NodeTwoStationWindow(bridge=BridgeLogBus(parent=app))
    done = []

    def work(cancel, log):
        log("running")
        while not cancel.wait(0.01):
            pass
        return "cancelled"

    window.execute(work, done.append)
    deadline = time.monotonic() + 2
    while not window.busy() and time.monotonic() < deadline:
        app.processEvents()
    window.cancel()
    while window.busy() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert done == ["cancelled"]
    assert all(button.isEnabled() for button in window.action_buttons)
    window.close()


def test_main_window_registers_and_opens_node_feature(tmp_path, monkeypatch):
    from dasqt.utils import config

    monkeypatch.setattr(config, "get_user_config_dir", lambda *args: str(tmp_path))
    monkeypatch.setattr(
        config, "migrate_legacy_user_config", lambda *args, **kwargs: {}
    )
    from dasqt.gui.main_window import DASMainWindow

    app = QApplication.instance() or QApplication([])
    main = DASMainWindow()
    try:
        assert "node_two_station_dispersion" in {
            r.feature_id for r in main.submain_registry.registered_features()
        }
        menu = next(
            action.menu()
            for action in main.menuBar().actions()
            if action.text() == "Dispersion"
        )
        assert main.node_two_station_action in menu.actions()
        assert "Two-station" in main.node_two_station_action.text()
        main.node_two_station_action.trigger()
        app.processEvents()
        children = main.submain_registry.instances("node_two_station_dispersion")
        assert len(children) == 1 and isinstance(children[0], NodeTwoStationWindow)
        children[0].close()
    finally:
        main.close()
        app.processEvents()


def test_overall_button_runs_cpu_and_populates_canvas(tmp_path):
    app = QApplication.instance() or QApplication([])
    task = make_task(tmp_path, df=0.5, dv=100.0)
    window = NodeTwoStationWindow(bridge=BridgeLogBus(parent=app), options=task.p)
    window.loaded(task)
    button = next(
        button for button in window.action_buttons if button.text() == "总体参考初稿"
    )
    button.click()
    deadline = time.monotonic() + 20
    while window.busy() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert not window.busy()
    assert "Computing overall park" in window.log.toPlainText()
    assert "Computing overall fj" in window.log.toPlainText()
    assert len(window.reference_axis.collections) > 0
    window.close()


def test_real_mouse_load_auto_computes_and_restores_controls_on_gui_thread(tmp_path):
    from PyQt6.QtCore import Qt, QThread
    from PyQt6.QtTest import QTest

    app = QApplication.instance() or QApplication([])
    task = make_task(tmp_path, df=0.5, dv=100.0)
    window = NodeTwoStationWindow(bridge=BridgeLogBus(parent=app), options=task.p)
    threads = []
    original = window.loaded

    def loaded(*args, **kwargs):
        threads.append(QThread.currentThread() == app.thread())
        return original(*args, **kwargs)

    window.loaded = loaded
    window.show()
    app.processEvents()
    QTest.mouseClick(window.primary_load_button, Qt.MouseButton.LeftButton)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(10)
        if "总体频散计算完成" in window.log.toPlainText() and not window.busy():
            break
    assert threads == [True]
    assert "开始计算总体频散参考" in window.log.toPlainText()
    assert "总体频散计算完成" in window.log.toPlainText()
    assert not window.busy()
    assert all(button.isEnabled() for button in window.action_buttons)
    assert len(window.reference_axis.collections) > 0
    window.close()


def test_loading_only_mode_leaves_visible_compute_button_enabled(tmp_path):
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest

    app = QApplication.instance() or QApplication([])
    task = make_task(tmp_path, df=0.5, dv=100.0)
    window = NodeTwoStationWindow(bridge=BridgeLogBus(parent=app), options=task.p)
    window.auto_overall.setChecked(False)
    window.show()
    app.processEvents()
    QTest.mouseClick(window.primary_load_button, Qt.MouseButton.LeftButton)
    deadline = time.monotonic() + 20
    while window.busy() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(10)
    assert window.task is not None
    assert (
        window.primary_overall_button.isEnabled()
        and window.primary_overall_button.isVisible()
    )
    QTest.mouseClick(window.primary_overall_button, Qt.MouseButton.LeftButton)
    while window.busy() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(10)
    assert "总体频散计算完成" in window.log.toPlainText()
    window.close()
