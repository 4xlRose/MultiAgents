import sys
import random
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QFileDialog, QMessageBox, QInputDialog, QHBoxLayout
from PyQt5.QtCore import Qt

# Maze dimensions
ROWS, COLS = 20, 20

# Define colors for different elements
COLORS = {
    'B': 'brown',
    'F': 'white',
    'G': 'green',
    '1': 'red',
    '2': 'blue',
    '3': 'orange',
    '4': 'purple'
}

class MazeButton(QPushButton):
    def __init__(self, row, col, parent=None):
        super().__init__('F', parent)
        self.row = row
        self.col = col
        self.setFixedSize(30, 30)
        self.setStyleSheet(f"background-color: {COLORS['F']};")

    def mousePressEvent(self, event):
        self.parent().handle_click(event, self.row, self.col)



class MazeEditor(QWidget):
    def __init__(self):
        super().__init__()

        self.grid = [['F' for _ in range(COLS)] for _ in range(ROWS)]  # Default all cells to Floor (F)
        self.mouse_pressed = False  # Track if the left mouse button is being held down
        self.current_action = None  # Track the current mouse action ('left_click' or 'shift_left_click')
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Maze Designer')

        layout = QGridLayout()
        self.setLayout(layout)

        # Create the grid of buttons
        self.buttons = [[None for _ in range(COLS)] for _ in range(ROWS)]
        for row in range(ROWS):
            for col in range(COLS):
                btn = MazeButton(row, col, self)
                layout.addWidget(btn, row, col)
                self.buttons[row][col] = btn

                # If buttons are in the border, set them to 'B'
                if row == 0 or col == 0 or row == ROWS - 1 or col == COLS - 1:
                    self.grid[row][col] = 'B'
                    btn.setText('B')
                    btn.setStyleSheet(f"background-color: {COLORS['B']};")
                    btn.setEnabled(False)

        self.randomize_maze()  # Randomize the maze at the start

        # Add save, random, and clean buttons at the bottom
        save_btn = QPushButton('Save Maze')
        save_btn.clicked.connect(self.save_maze)

        load_btn = QPushButton('Load Maze')
        load_btn.clicked.connect(self.load_maze)

        random_btn = QPushButton('Random')
        random_btn.clicked.connect(self.randomize_maze)

        clean_btn = QPushButton('Clean')
        clean_btn.clicked.connect(self.clean_maze)

        qhorizontal = QHBoxLayout()
        qhorizontal.addWidget(save_btn)
        qhorizontal.addWidget(load_btn)
        qhorizontal.addWidget(random_btn)
        qhorizontal.addWidget(clean_btn)

        layout.addLayout(qhorizontal, ROWS, 0, 1, COLS)

        self.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_pressed = True
            if event.modifiers() & Qt.ShiftModifier:
                self.current_action = 'shift_left_click'
            else:
                self.current_action = 'left_click'
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.mouse_pressed = False
        self.current_action = None
        super().mouseReleaseEvent(event)

    def handle_click(self, event, row, col):
        # Handle left click, Shift + left click, and right click
        if event.button() == Qt.LeftButton:
            if event.modifiers() & Qt.ShiftModifier:
                self.toggle_goal(row, col)  # Shift + left click toggles to/from 'G'
            else:
                self.toggle_box_floor(row, col)  # Left click toggles between 'F' and 'B'
        elif event.button() == Qt.RightButton:
            self.set_number(row, col)  # Right click sets a number

    def toggle_box_floor(self, row, col):
        # Toggle between 'F' and 'B'
        current_element = self.grid[row][col]
        next_element = 'B' if current_element == 'F' else 'F'
        self.grid[row][col] = next_element
        self.buttons[row][col].setText(next_element)
        self.buttons[row][col].setStyleSheet(f"background-color: {COLORS[next_element]};")

    def toggle_goal(self, row, col):
        # Toggle between whatever it is and 'G'
        current_element = self.grid[row][col]
        next_element = 'G' if current_element != 'G' else 'F'  # Revert to 'F' if toggling back from 'G'
        self.grid[row][col] = next_element
        self.buttons[row][col].setText(next_element)
        self.buttons[row][col].setStyleSheet(f"background-color: {COLORS.get(next_element, 'white')};")

    def set_number(self, row, col):
        # Open a popup to input a number and set it as the cell's text
        num, ok = QInputDialog.getText(self, 'Set Number', 'Enter a number:')
        if ok and num.isdigit():
            self.grid[row][col] = num
            self.buttons[row][col].setText(num)
            self.buttons[row][col].setStyleSheet(f"background-color: {COLORS.get(num, 'gray')};")  # Default to gray if no color defined

    def clean_maze(self):
        # Clean the entire grid with 'F'
        for row in range(ROWS):
            for col in range(COLS):
                self.grid[row][col] = 'F'
                self.buttons[row][col].setText('F')
                self.buttons[row][col].setStyleSheet(f"background-color: {COLORS['F']};")

        # Set the border to 'B' and disable the buttons
        for row in range(ROWS):
            for col in range(COLS):
                if row == 0 or col == 0 or row == ROWS - 1 or col == COLS - 1:
                    self.grid[row][col] = 'B'
                    self.buttons[row][col].setText('B')
                    self.buttons[row][col].setStyleSheet(f"background-color: {COLORS['B']};")
                    self.buttons[row][col].setEnabled(False)

    def randomize_maze(self):
        # Randomize the entire grid with 'F' and 'B'
        for row in range(1, ROWS - 1):
            for col in range(1, COLS - 1):
                random_value = 'F' if random.random() < 0.6 else 'B'
                self.grid[row][col] = random_value
                self.buttons[row][col].setText(random_value)
                self.buttons[row][col].setStyleSheet(f"background-color: {COLORS[random_value]};")

    def save_maze(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Maze", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, 'w') as file:
                    for row in self.grid:
                        file.write("".join(row) + "\n")
                QMessageBox.information(self, "Saved", f"Maze saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not save the file: {e}")

    def load_maze(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Maze", "", "Text Files (*.txt)")
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    for row, line in enumerate(file):
                        for col, char in enumerate(line.strip()):
                            self.grid[row][col] = char
                            self.buttons[row][col].setText(char)
                            self.buttons[row][col].setStyleSheet(f"background-color: {COLORS.get(char, 'gray')};")
                QMessageBox.information(self, "Loaded", f"Maze loaded from {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not load the file: {e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            self.save_maze()
        super().keyPressEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    editor = MazeEditor()
    sys.exit(app.exec_())