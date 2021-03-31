import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QMenu, QMenuBar
from PyQt5.QtGui import QIcon


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._initialize_ui()

    def _initialize_ui(self):
        """Initialize the window and display its contents to the screen."""
        self.setMinimumSize(500, 400)
        self.setWindowTitle("Motorcycle Garage Door Opener 1.0")
        self.setWindowIcon(QIcon("images/ducati.png"))
        self._setup_gui()
        self.show()

    def _setup_gui(self):
        """Set up the toolbar, input video, and output video"""
        self._create_menu_bar()

    def _create_menu_bar(self):
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        # Creating menus using a QMenu object
        file_menu = QMenu("&File", self)
        tools_menu = QMenu("&Tools", self)
        help_menu = QMenu("&Help", self)

        menu_bar.addMenu(file_menu)
        menu_bar.addMenu(tools_menu)
        menu_bar.addMenu(help_menu)

# Run main event loop
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
