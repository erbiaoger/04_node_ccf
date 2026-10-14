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
        main.node_two_station_action.trigger()
        app.processEvents()
        children = main.submain_registry.instances("node_two_station_dispersion")
        assert len(children) == 1 and isinstance(children[0], NodeTwoStationWindow)
        children[0].close()
    finally:
        main.close()
        app.processEvents()
