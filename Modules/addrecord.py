from PyQt6.QtWidgets import (QDialog, QLineEdit, QVBoxLayout, QPushButton,
                             QLabel, QGridLayout, QHBoxLayout, QSpacerItem, QSizePolicy, QMessageBox)
from PyQt6.QtCore import Qt,QSize
import unicodedata
class RecordDialog(QDialog):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.columns = columns
        self.inputs = {}

        # Translation
        self.setWindowTitle(self.translate("AddRecordDialogTitle"))

        mainLayout = QVBoxLayout(self)
        gridLayout = QGridLayout()
        gridLayout.setVerticalSpacing(10)

        line_edit_style = """QLineEdit { 
                                border: 1px solid gray; 
                                padding: 2px; 
                                border-radius: 5px; 
                             }"""

        for idx, col in enumerate(columns):
            lbl = QLabel(col, self)  # No translation for column headers
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            le = QLineEdit(self)
            le.setStyleSheet(line_edit_style)
            gridLayout.addWidget(lbl, 0, idx)
            gridLayout.addWidget(le, 1, idx)
            self.inputs[col] = le

        mainLayout.addLayout(gridLayout)
        spacer = QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        mainLayout.addItem(spacer)

        btnLayout = QHBoxLayout()
        btnAdd = QPushButton(self.translate("AddButtonText"), self)  # Translated button text
        btnAdd.clicked.connect(self.accept)
        btnCancel = QPushButton(self.translate("CancelButtonText"), self)  # Translated button text
        btnCancel.setObjectName("btnCancel")
        btnCancel.clicked.connect(self.reject)
        btnCancel.setStyleSheet("""
                             QPushButton#btnCancel {
                                 color: black;
                                 background-color: #ffffff;
                             }

                             QPushButton#btnCancel:hover {
                                 background-color: #CCDC3545;
                                 color: white;
                             }

                             QPushButton#btnCancel:pressed {
                                 background-color: #DC3545;
                                 color: white;
                             }
                         """)
        btnLayout.addWidget(btnCancel)
        btnLayout.addWidget(btnAdd)
        mainLayout.addLayout(btnLayout)

    def translate(self, key):
        return self.parent().translations[self.parent().current_language][key]




    def get_record(self):
        return {col: le.text() for col, le in self.inputs.items()}

def normalize_string(s):
    return unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode("utf-8").lower()

class DynamicPriceRecordDialog(QDialog):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.columns = columns
        self.inputs = {}

        self.translations = parent.translations
        self.current_language = parent.current_language

        self.setWindowTitle(self.translations[self.current_language]["AddPriceRecordTitle"])

        mainLayout = QVBoxLayout(self)
        gridLayout = QGridLayout()
        gridLayout.setVerticalSpacing(10)
        line_edit_style = """QLineEdit { 
                                border: 1px solid gray; 
                                padding: 2px; 
                                border-radius: 5px; 
                             }"""

        # Create an input field for each column

        for idx, col in enumerate(columns):
            translated_col = self.translations[self.current_language].get(col,
                                                                    col)
            lbl = QLabel(translated_col, self)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            le = QLineEdit(self)
            le.setStyleSheet(line_edit_style)
            gridLayout.addWidget(lbl, 0, idx)
            gridLayout.addWidget(le, 1, idx)
            self.inputs[col] = le
        max_button_size = QSize(150, 50)
        mainLayout.addLayout(gridLayout)
        spacer = QSpacerItem(20, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        mainLayout.addItem(spacer)
        btnLayout = QHBoxLayout()
        btnAdd = QPushButton(self.translations[self.current_language]["AddButton"], self)
        btnCancel = QPushButton(self.translations[self.current_language]["CancelButton"], self)
        btnCancel.setMaximumSize(max_button_size)
        btnAdd.setMaximumSize(max_button_size)
        btnAdd.clicked.connect(self.on_accept)
        btnCancel.setObjectName("btnCancel")
        btnCancel.setStyleSheet(

            '''
                QPushButton{
                color: black;
                background-color:  #ffffff;
                }
            
                QPushButton:hover{
                background-color:  #CCDC3545;
                color: white;
                }
            
                QPushButton:pressed{
                background-color:  #DC3545;
                color: white;
                }
            
            '''
        )

        btnCancel.clicked.connect(self.reject)
        btnLayout.addWidget(btnCancel)
        btnLayout.addWidget(btnAdd)

        mainLayout.addLayout(btnLayout)
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.on_accept)
        self.adjustSize()
        self.setFixedSize(self.size())



    def get_record(self):
        return {col: le.text() for col, le in self.inputs.items()}

    def validate_input(self):
        target_df = self.parent().pricing_alternate_df if self.parent().is_alternate_structure else self.parent().pricing_df

        for column, line_edit in self.inputs.items():
            value = line_edit.text().strip()
            normalized_value = normalize_string(value)
            if column in ["Names", "Location"]:
                if not value.isalpha():
                    return False, self.translations[self.current_language]["AlphaCheckError"].format(column)

                if normalized_value in target_df[column].apply(normalize_string).values:
                    return False, self.translations[self.current_language]["ExistsCheckError"].format(column, value)
            else:
                try:
                    int_val = int(value)
                except ValueError:
                    return False, self.translations[self.current_language]["IntCheckError"].format(column)

        return True, ""

    def on_accept(self):
        is_valid, error_message = self.validate_input()
        if not is_valid:
            QMessageBox.warning(
                self,
                self.translations[self.current_language]["InputErrorTitle"],
                error_message
            )
        else:
            super().accept()


