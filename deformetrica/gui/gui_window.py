import json
import logging
import math
import os
import sys

import pkg_resources
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from deformetrica import default
from . import gui_api, gui_graph

logging.basicConfig(level=logging.DEBUG)


class Spoiler:
    def __init__(self, title, start, parent=None):
        self.widget = QWidget()
        self.toggleButton = QToolButton()
        self.toggleButton.setStyleSheet("QToolButton { border: none; }")
        self.toggleButton.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toggleButton.setArrowType(QtCore.Qt.RightArrow)
        self.toggleButton.setText(title)
        self.toggleButton.setCheckable(True)
        self.toggleButton.setChecked(start)

        self.headerLine = QFrame()
        self.headerLine.setFrameShape(QFrame.HLine)
        self.headerLine.setFrameShadow(QFrame.Sunken)
        self.headerLine.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        self.mainLayout = QGridLayout()
        self.mainLayout.setVerticalSpacing(0)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)

        self.mainLayout.addWidget(self.toggleButton, 0, 0, 1, 1, QtCore.Qt.AlignLeft)
        self.mainLayout.addWidget(self.headerLine, 0, 2, 1, 1)
        self.widget.setLayout(self.mainLayout)

        self.frame = QWidget()

        _this = self

        def on_click(checked):
            if not checked:
                _this.frame.show()
            else:
                _this.frame.hide()
            _this.toggleButton.setArrowType(QtCore.Qt.DownArrow if not checked else QtCore.Qt.RightArrow)

        self.toggleButton.clicked.connect(on_click)

        on_click(start)


class GroupTemplate:
    def __init__(self, _group, form):
        self.info = _group
        self.multiple = "multiple" in _group
        self.formIndex = form.count()

        if self.multiple:
            self.groups = []

            widget = QPushButton("Add new " + _group["name"])
            self.index = 0

            def open_new():
                group = Group(_group)

                delete_button = QPushButton("Remove this " + self.info["name"])

                def remove(rem=group):
                    self.groups.remove(group)
                    form.removeWidget(group.spoiler.frame)
                    form.removeWidget(group.spoiler.widget)

                delete_button.clicked.connect(remove)
                group.current.addWidget(delete_button)

                form.insertWidget(self.formIndex, group.spoiler.widget)
                form.insertWidget(self.formIndex + 1, group.spoiler.frame)

                self.groups.append(group)
                self.index += 1

            widget.clicked.connect(open_new)
            self.open_new = open_new

            open_new()

            form.addWidget(widget)

        else:
            self.group = Group(_group)
            form.addWidget(self.group.spoiler.widget)
            form.addWidget(self.group.spoiler.frame)

    def get_values(self, values):
        if self.multiple:
            vals = []
            for group in self.groups:
                val = {}
                group.get_values(val)
                vals.append(val)
            values[self.info["name"]] = vals
        else:
            val = {}
            self.group.get_values(val)
            values[self.info["name"]] = val

    def set_values(self, values):
        if self.multiple:
            for i in range(len(values[self.info["name"]])):
                if i >= len(self.groups):
                    self.open_new()
                self.groups[i].set_values(values[self.info["name"]][i])
        else:
            self.group.set_values(values[self.info["name"]])


class Group:
    def __init__(self, _group):
        self.group = _group
        self.params = {}

        if "label" not in self.group:
            self.group["label"] = self.group["name"].replace('_', ' ').title()

        self.current = QVBoxLayout()
        self.spoiler = Spoiler(self.group["label"], "optional" in self.group)
        self.spoiler.frame.setLayout(self.current)
        self.current.setContentsMargins(0, 0, 0, 30)

        for param in _group["parameters"]:
            self.params[param["name"]] = Param(param, self.current)

    def get_values(self, values):
        for param in self.params:
            self.params[param].get_value(values)

    def set_values(self, values):
        for param in self.params:
            self.params[param].set_value(values)


class Param:
    def __init__(self, param, current):
        self.param = param
        self.type = param["type"]
        self.value = None
        self.update = lambda x: x

        def update_value(x):
            self.value = x
            self.update(x)

        self.update_value = update_value

        if "default" not in param:
            try:
                param["default"] = getattr(default, param["name"])
            except:
                param["default"] = None

        if "label" not in param:
            param["label"] = param["name"].replace('_', ' ').title()

        if param["type"] == "slider":
            slider = Slider(param, update_value)
            current.addWidget(slider.label)
            current.addWidget(slider.slider)
            self.widget = slider.label
            self.update = lambda x: slider.set_value(x)

        elif param["type"] == "selector":
            selector = QComboBox()

            def update(x, func=update_value, values=param["values"]):
                func(values[x])

            selector.currentIndexChanged.connect(update)

            for i in range(len(param["values"])):
                selector.insertItem(i, str(param["values"][i]))

            self.update = lambda x: selector.setCurrentIndex(self.param["values"].index(x))

            selector.setCurrentIndex(param["values"].index(param["default"]))

            self.widget = QLabel(param["label"] + " : ")
            current.addWidget(self.widget)
            current.addWidget(selector)

        elif param["type"] == "toggle":
            # toggle = QCheckBox(param["label"])
            toggle = QPushButton(param["label"])
            toggle.setCheckable(True)
            toggle.toggled.connect(update_value)
            if bool(param["default"]):
                toggle.toggle()
            else:
                toggle.toggle()
                toggle.toggle()
            self.update = lambda x: toggle.setDown(x)
            toggle.setDown(param["default"])
            current.addWidget(toggle)
            self.widget = toggle

        elif param["type"] == "int":
            widget = QLineEdit()
            widget.setValidator(QIntValidator())
            widget.textChanged.connect(lambda x: update_value(int(x) if len(x) else 0))

            self.widget = QLabel(param["label"] + " : ")
            current.addWidget(self.widget)
            current.addWidget(widget)
            widget.setText(str(param["default"]))

        elif param["type"] == "float":
            widget = QLineEdit()
            widget.setValidator(QDoubleValidator())
            widget.textChanged.connect(lambda x: update_value(float(x) if len(x) else 0.0))

            self.widget = QLabel(param["label"] + " : ")
            current.addWidget(self.widget)
            current.addWidget(widget)
            widget.setText(str(param["default"]))

        elif param["type"] == "file":
            self.widget = QPushButton(param["label"])
            self.widget.clicked.connect(lambda: update_value(self.check_file(QFileDialog.getOpenFileName(None, "Choose File", ""))))
            current.addWidget(self.widget)

            label = QLabel("")
            current.addWidget(label)
            self.update = lambda x: label.setText(str(x).split('/')[-1])

            update_value(param["default"])

        elif param["type"] == "files":
            self.widget = QPushButton(param["label"])
            self.widget.clicked.connect(lambda: update_value(self.check_file(QFileDialog.getOpenFileNames(None, "Choose Files", ""))))
            current.addWidget(self.widget)

            update_value(param["default"])

        elif param["type"] == "directory":
            self.widget = QPushButton(param["label"])
            self.widget.clicked.connect(lambda: update_value(self.check_file(QFileDialog.getExistingDirectory(None, "Choose Directory"))))
            current.addWidget(self.widget)

            label = QLabel("")
            current.addWidget(label)
            self.update = lambda x: label.setText(str(x).split('/')[-1])

            update_value(param["default"])

        if "tooltip" in param:
            self.widget.setToolTip(param["tooltip"])

    def check_file(self, x):
        if isinstance(x, str):
            if len(x) > 0:
                return x
            else:
                return self.value
        elif isinstance(x, tuple):
            if len(x[0]) > 0:
                return x[0]
            else:
                return self.value

    def check(self):
        if "optional" not in self.param:
            has_error = False
            if self.value is None or self.value == "":
                has_error = True
            elif self.type == "file":
                file = QtCore.QFileInfo(self.value)
                if not file.exists() or not file.isFile():
                    has_error = True
            elif self.type == "files":
                for x in self.value:
                    file = QtCore.QFileInfo(x)
                    if not file.exists() or not file.isFile():
                        has_error = True
            elif self.type == "directory":
                file = QtCore.QFileInfo(self.value)
                if not file.exists() or not file.isDir():
                    has_error = True

            self.set_error(has_error)

    def get_value(self, values):
        self.check()
        values[self.param["name"]] = self.value

    def set_value(self, values):
        if self.param["name"] in values:
            self.update_value(values[self.param["name"]])
            self.check()
        else:
            self.set_error(True)

    def set_error(self, error):
        palette = self.widget.palette()
        palette.setColor(self.widget.foregroundRole(), QtCore.Qt.red if error else QtCore.Qt.black)
        # self.widget.setPalette(palette)


class Slider:
    def __init__(self, param, callback):
        self.name = param["label"]
        self.callback = callback
        self.width = 10.

        self.log = "scale" in param and param["scale"] == "logarithmic"
        if "scale" in param and param["scale"] == "int":
            self.width = 1

        self.slider = QSlider(QtCore.Qt.Horizontal)
        self.slider.setValue(1)
        self.slider.valueChanged.connect(self.on_value_change)

        self.label = QLabel()

        if self.log:
            self.slider.setRange(math.log10(param["min"]) * self.width, math.log10(param["max"]) * self.width)
            self.slider.setValue(math.log10(param["default"]) * self.width if ("default" in param) else (math.log10(param["min"]) + math.log10(param["max"])) * self.width / 2.0)
        else:
            self.slider.setRange(param["min"] * self.width, param["max"] * self.width)
            self.slider.setValue(param["default"] * self.width if ("default" in param) else (param["min"] + param["max"]) * self.width / 2.0)

    def on_value_change(self, x):
        if self.log:
            self.label.setText(self.name + " : " + str('{:0.1e}'.format(10 ** (x / self.width))).rjust(4))
            self.callback(10 ** (x / self.width))
        else:
            self.label.setText(self.name + " : " + str(x / self.width).rjust(4))
            self.callback(x / self.width if self.width != 1 else int(x / self.width))

    def set_value(self, x):
        self.slider.setValue((math.log10(x) if self.log else x) * self.width)


class Gui:
    def __init__(self, parent, function_name, change_function, config_path):
        self.parent = parent
        self.change_function = change_function
        self.config_path = config_path

        self.scroll = QScrollArea()
        self.parent.addWidget(self.scroll)
        self.scroll.setWidgetResizable(True)
        self.function_parameters = None
        self.groups = {}

        self.prepare_gui(function_name)

    def prepare_gui(self, function_name):
        inner_scroll = QWidget()
        policy = inner_scroll.sizePolicy()
        policy.setVerticalStretch(0)
        policy.setHorizontalStretch(1)
        policy.setVerticalPolicy(4)
        inner_scroll.setSizePolicy(policy)
        self.scroll.setWidget(inner_scroll)

        # Open config file
        with open(os.path.join(self.config_path, function_name + '.json')) as f:
            self.function_parameters = json.load(f)

        form = QVBoxLayout(inner_scroll)
        form.setContentsMargins(10, 10, 10, 10)
        inner_scroll.setLayout(form)

        title = QLabel("Parameters")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setFont(QFont("Arial", 15, QtGui.QFont.Bold))
        form.addWidget(title)

        current = form

        # Create Ui Widgets corresponding to json file contents
        for group in self.function_parameters["parameter_groups"]:
            self.groups[group["name"]] = GroupTemplate(group, current)

    def reload(self, function_name):
        # Delete all UI and recreate it
        self.prepare_gui(function_name)

    def get_values(self):
        values = {}

        for group in self.groups:
            self.groups[group].get_values(values)

        return values

    def save_values(self, file):
        values = self.get_values()

        obj = {"function_name": self.function_parameters["function_name"],
               "values": values}

        try:
            with open(file[0], 'w') as outfile:
                json.dump(obj, outfile)

        except Exception as e:
            if len(file[0]) > 0:
                print('Error loading file ' + file[0])

    def load_values(self, file):
        try:
            with open(file[0], 'r') as f:
                obj = json.load(f)
                self.change_function(obj["function_name"])
                for group in self.groups:
                    self.groups[group].set_values(obj["values"])

        except Exception as e:
            if len(file[0]) > 0:
                print('Error loading file ' + file[0])
            # else if canceled


class Stream(QtCore.QObject):
    on_new_text_signal = pyqtSignal(str)

    def __init__(self, parent=None, on_new_text=None):
        super().__init__(parent)
        if on_new_text is not None:
            self.on_new_text_signal.connect(on_new_text)

    def write(self, text):
        self.on_new_text_signal.emit(str(text))


class Console:
    def __init__(self, parent=None):
        self.process = QTextEdit(parent)
        # self.process.moveCursor(QtGui.QTextCursor.Start)
        self.process.setReadOnly(True)
        self.process.setStyleSheet("QTextEdit { padding-left:5; padding-top:5; padding-bottom:5; padding-right:5}")
        self.process.ensureCursorVisible()
        self.process.setLineWrapColumnOrWidth(1000)
        self.process.setLineWrapMode(QTextEdit.FixedPixelWidth)

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        # hook system standard output and standard error
        sys.stdout = Stream(on_new_text=self.on_text_update)
        sys.stderr = Stream(on_new_text=self.on_text_update)

    def __del__(self):
        # unhook
        # sys.stdout = sys.__stdout__
        # sys.stderr = sys.__stderr__
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

    def on_text_update(self, text):
        assert isinstance(text, str)
        cursor = self.process.textCursor()
        # cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.process.setTextCursor(cursor)
        self.process.ensureCursorVisible()


# Makes Buttons to call and
class RunButton:
    def __init__(self, parent, graph, function_name, gui):
        self.graph = graph
        self.function_name = function_name
        self.gui = gui

        self.down = False
        # Create Run Button on the bottom left
        self.run_button = QPushButton("Run")
        self.run_button.setEnabled(True)
        self.run_button.clicked.connect(self.on_run)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.on_stop)

        dbut = QHBoxLayout()
        dbut.addWidget(self.run_button)
        dbut.addWidget(self.stop_button)
        parent.addLayout(dbut)

    def on_run(self):
        self.down = True
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.graph.next_run()

        def callback(obj):
            self.graph.iteration(obj)
            return self.down

        def run_callback(deformetrica_method_result):
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)

        gui_api.call(self.function_name, self.gui.get_values(), estimator_callback=callback, callback=run_callback)

    def on_stop(self):
        self.down = False


class StartGui:
    def __init__(self):
        manager = pkg_resources.ResourceManager()
        self.images_path = manager.resource_filename('gui.resources.images', '')
        self.config_path = manager.resource_filename('gui.resources.config', '')
        self.launcher_config = None

    # Start point of the gui
    def start(self):
        app = QApplication(sys.argv)

        self.running = False
        # Choose function and open main window
        self.choose_function(self.function_window)

        app.exec_()

    # Create funtion choice dialog
    def choose_function(self, open_main_window):
        # Open config file
        with open(os.path.join(self.config_path, 'api.json')) as f:
            self.launcher_config = json.load(f)
            print(self.launcher_config)

        dialog = QDialog()
        dialog.setWindowTitle("Select Function")
        dialog.setWindowIcon(QIcon(os.path.join(self.images_path, self.launcher_config["logo"])))

        layout1 = QGridLayout(dialog)
        layout1.setHorizontalSpacing(20)
        dialog.setLayout(layout1)

        wid = QLabel()
        wid.setPixmap(QPixmap(os.path.join(self.images_path, self.launcher_config["logo"])))  # big logo
        layout1.addWidget(wid, 0, 0, 1, -1, QtCore.Qt.AlignCenter)  # span across whole row, align center

        # add widgets for each function
        i = 0
        for x in self.launcher_config["functions"]:
            def on_click(b, _function_name=x["file"]):
                try:
                    open_main_window(_function_name)  # callback with chosen function
                    dialog.done(20)
                except FileNotFoundError:
                    print('Not implemented yet')
                    pass

            button = QPushButton(x["name"], )  # primary button
            # button.setStyleSheet("border: 1px solid black; background: white")
            button.setFixedSize(200, 50)
            button.clicked.connect(on_click)
            button.setEnabled(False if "enabled" in x and x["enabled"].lower() == 'false' else True)
            layout1.addWidget(button, 1, i)

            img = QLabel()
            img.setPixmap(QPixmap(os.path.join(self.images_path, x["image"])))  # function image
            layout1.addWidget(img, 2, i)

            # label = QLabel(x["description"])  # Description text
            # label.setFixedSize(200, 50)
            # label.setWordWrap(True)
            # layout1.addWidget(label, 3, i)

            i += 1

        # add footer
        if "footer" in self.launcher_config:
            footer = QLabel(self.launcher_config["footer"])
            layout1.addWidget(footer, 4, 0, 1, -1, QtCore.Qt.AlignRight)

        # handle dialog close
        if dialog.exec() != 20 and not self.running:
            exit(0)

    # Prepare main window
    def function_window(self, function_name):
        self.running = True
        self.function_name = function_name

        self.window = QMainWindow()
        self.window.setWindowTitle("Deformetrica")
        self.window.setWindowIcon(QIcon(os.path.join(self.images_path, self.launcher_config["logo"])))

        # Split window horizontally
        split = QSplitter()
        self.window.setCentralWidget(split)

        # Lay left side vertically
        st = QWidget(split)
        prime = QVBoxLayout(st)
        st.setLayout(prime)
        prime.setContentsMargins(0, 0, 0, 0)
        st.resize(300, 800)

        # Attach gui to top left
        self.gui = Gui(prime, function_name, self.change_function, self.config_path)

        # Split right side vertically
        self.split2 = QSplitter(QtCore.Qt.Vertical, split)
        self.split2.resize(900, 800)

        # Attach graph top right
        self.graph = gui_graph.Graph(function_name)()
        self.split2.insertWidget(0, self.graph)

        # Attach console bottom right
        self.console = Console(self.split2)

        # Attach run buttons to bottom left
        self.run = RunButton(prime, self.graph, self.function_name, self.gui)

        self.make_menu_bar()

        self.window.show()
        self.window.resize(1200, 700)

    # Add all actions to the menu bar of the window
    def make_menu_bar(self):
        bar = self.window.menuBar()

        file_menu = bar.addMenu("File")

        # This collects the value of each parameter and stores it in a JSON file
        save = file_menu.addAction("Save")
        save.setShortcut("Ctrl+S")
        save.setStatusTip("Save current settings into a file")

        def on_save_file():
            self.gui.save_values(QFileDialog.getSaveFileName(None, "Choose File", ".", "JSON File (*.json)"))

        save.triggered.connect(on_save_file)

        # This uses the JSON file specified to fill the parameters. 
        load = file_menu.addAction("Open")
        load.setShortcut("Ctrl+O")
        load.setStatusTip("Load settings from a file")

        def on_load_file():
            self.gui.load_values(QFileDialog.getOpenFileName(None, "Choose File", ".", "JSON File (*.json)"))

        load.triggered.connect(on_load_file)

        # Shows the function selection dialog
        change = file_menu.addAction("Choose Function")
        change.setStatusTip("Return to function selection")

        def on_change_function():
            self.choose_function(self.change_function)

        change.triggered.connect(on_change_function)

        view_menu = bar.addMenu("View")

        # Saves matplotlib figure into the given file ; can be png or pdf for now
        save_graph = view_menu.addAction("Save Figure")
        save_graph.setStatusTip("Save figure into a file")

        def on_save_graph():
            url = QFileDialog.getSaveFileName(None, "Choose File", ".", "Portable Network Graphics (*.png);;Portable Document Format (*.pdf)")[0]
            if len(url) > 0:
                self.graph.save(url)

        save_graph.triggered.connect(on_save_graph)

        # removes all plots from graph (calls clear on graph class)
        clear_graph = view_menu.addAction("Clear Figure")
        clear_graph.setStatusTip("Remove all plots from the figure")

        def on_clear_graph():
            self.graph.clear()

        clear_graph.triggered.connect(on_clear_graph)

        # removes all text from console widget
        clear_console = view_menu.addAction("Clear Console")
        clear_console.setStatusTip("Remove all text from console")

        def on_clear_console():
            self.console.process.clear()

        clear_console.triggered.connect(on_clear_console)

    def change_function(self, function_name):
        self.function_name = function_name
        self.gui.reload(function_name)
        # self.graph = gui_graph.Graph(function_name)()
        # self.graph.clear()
        self.split2.replaceWidget(0, self.graph)
