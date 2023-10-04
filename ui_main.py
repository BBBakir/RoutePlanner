# Form implementation generated from reading ui file 'lav.ui'
#
# Created by: PyQt6 UI code generator 6.4.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.
from Modules.GraphScreen import PannableView
from PyQt6 import QtCore, QtGui, QtWidgets
import os
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):


        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        MainWindow.resize(1096, 896)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        font.setBold(False)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(resource_path("Assets/Icons/icon.ico")), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setWindowOpacity(1.0)
        MainWindow.setStyleSheet("")
        MainWindow.setIconSize(QtCore.QSize(256, 256))
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        MainWindow.setDocumentMode(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.TabShape.Rounded)
        MainWindow.setDockNestingEnabled(False)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.centralwidget.setStyleSheet("* { text-transform: capitalize; }")
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(parent=self.centralwidget)
        self.tabWidget.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tabWidget.sizePolicy().hasHeightForWidth())
        self.tabWidget.setSizePolicy(sizePolicy)
        self.tabWidget.setMaximumSize(QtCore.QSize(1200, 1200))
        self.tabWidget.setBaseSize(QtCore.QSize(0, 0))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(11)
        font.setBold(False)
        self.tabWidget.setFont(font)
        self.tabWidget.setMouseTracking(False)
        self.tabWidget.setAutoFillBackground(False)
        self.tabWidget.setObjectName("tabWidget")
        self.Home = QtWidgets.QWidget()
        self.Home.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.Home.setObjectName("Home")
        self.gridLayout = QtWidgets.QGridLayout(self.Home)
        self.gridLayout.setObjectName("gridLayout")
        self.RecordTable = QtWidgets.QTableWidget(parent=self.Home)
        self.RecordTable.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.RecordTable.setObjectName("RecordTable")
        self.gridLayout.addWidget(self.RecordTable, 5, 1, 9, 6)
        spacerItem = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.gridLayout.addItem(spacerItem, 4, 2, 1, 1)
        self.ClearRecord = QtWidgets.QPushButton(parent=self.Home)
        self.ClearRecord.setMinimumSize(QtCore.QSize(200, 0))
        self.ClearRecord.setObjectName("ClearRecord")
        self.gridLayout.addWidget(self.ClearRecord, 18, 1, 1, 1)
        self.AddRecord = QtWidgets.QPushButton(parent=self.Home)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.AddRecord.sizePolicy().hasHeightForWidth())
        self.AddRecord.setSizePolicy(sizePolicy)
        self.AddRecord.setMinimumSize(QtCore.QSize(50, 0))
        self.AddRecord.setObjectName("AddRecord")
        self.gridLayout.addWidget(self.AddRecord, 8, 7, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.gridLayout.addItem(spacerItem1, 14, 4, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem2, 8, 6, 1, 1)
        self.ExcelExport = QtWidgets.QPushButton(parent=self.Home)
        self.ExcelExport.setMinimumSize(QtCore.QSize(150, 0))
        self.ExcelExport.setObjectName("ExcelExport")
        self.gridLayout.addWidget(self.ExcelExport, 18, 7, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(150, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem3, 18, 2, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(20, 100, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.gridLayout.addItem(spacerItem4, 12, 7, 1, 1)
        self.UploadExcel = QtWidgets.QPushButton(parent=self.Home)
        self.UploadExcel.setObjectName("UploadExcel")
        self.gridLayout.addWidget(self.UploadExcel, 3, 3, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(13, 30, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.gridLayout.addItem(spacerItem5, 2, 1, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.gridLayout.addItem(spacerItem6, 0, 1, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem7, 8, 8, 1, 1)
        spacerItem8 = QtWidgets.QSpacerItem(20, 100, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.gridLayout.addItem(spacerItem8, 6, 7, 1, 1)
        self.ExcelPath = QtWidgets.QLineEdit(parent=self.Home)
        self.ExcelPath.setObjectName("ExcelPath")
        self.gridLayout.addWidget(self.ExcelPath, 3, 1, 1, 2)
        spacerItem9 = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem9, 8, 0, 1, 1)
        spacerItem10 = QtWidgets.QSpacerItem(125, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacerItem10, 18, 3, 1, 1)
        self.RoutePlannerTitle = QtWidgets.QLabel(parent=self.Home)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(30)
        sizePolicy.setVerticalStretch(60)
        sizePolicy.setHeightForWidth(self.RoutePlannerTitle.sizePolicy().hasHeightForWidth())
        self.RoutePlannerTitle.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(26)
        font.setBold(True)
        self.RoutePlannerTitle.setFont(font)
        self.RoutePlannerTitle.setLayoutDirection(QtCore.Qt.LayoutDirection.LeftToRight)
        self.RoutePlannerTitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.RoutePlannerTitle.setObjectName("RoutePlannerTitle")
        self.gridLayout.addWidget(self.RoutePlannerTitle, 1, 1, 1, 7)
        self.DeleteRecord = QtWidgets.QPushButton(parent=self.Home)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.DeleteRecord.sizePolicy().hasHeightForWidth())
        self.DeleteRecord.setSizePolicy(sizePolicy)
        self.DeleteRecord.setMinimumSize(QtCore.QSize(50, 0))
        self.DeleteRecord.setObjectName("DeleteRecord")
        self.gridLayout.addWidget(self.DeleteRecord, 9, 7, 1, 1)
        self.Calculate = QtWidgets.QPushButton(parent=self.Home)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding, QtWidgets.QSizePolicy.Policy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Calculate.sizePolicy().hasHeightForWidth())
        self.Calculate.setSizePolicy(sizePolicy)
        self.Calculate.setMinimumSize(QtCore.QSize(50, 0))
        self.Calculate.setObjectName("Calculate")
        self.gridLayout.addWidget(self.Calculate, 10, 7, 1, 1)
        self.tabWidget.addTab(self.Home, "")
        self.AddRoute = QtWidgets.QWidget()
        self.AddRoute.setObjectName("AddRoute")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.AddRoute)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_2 = QtWidgets.QLabel(parent=self.AddRoute)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(25)
        font.setBold(True)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 1, 1, 1, 11)
        spacerItem11 = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_2.addItem(spacerItem11, 16, 12, 1, 1)

        spacerItem12 = QtWidgets.QSpacerItem(40, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout_2.addItem(spacerItem12, 4, 10, 1, 1)
        spacerItem13 = QtWidgets.QSpacerItem(20, 60, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_2.addItem(spacerItem13, 2, 7, 1, 1)
        spacerItem14 = QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_2.addItem(spacerItem14, 4, 2, 1, 1)
        spacerItem15 = QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_2.addItem(spacerItem15, 4, 7, 1, 1)
        self.saveRouteButton = QtWidgets.QPushButton(parent=self.AddRoute)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.saveRouteButton.sizePolicy().hasHeightForWidth())
        self.saveRouteButton.setSizePolicy(sizePolicy)
        self.saveRouteButton.setMinimumSize(QtCore.QSize(150, 50))
        self.saveRouteButton.setMinimumSize(QtCore.QSize(150, 50))
        self.saveRouteButton.setObjectName("saveRouteButton")
        self.gridLayout_2.addWidget(self.saveRouteButton, 16, 10, 1, 1)
        spacerItem16 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout_2.addItem(spacerItem16, 4, 1, 1, 1)
        spacerItem17 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_2.addItem(spacerItem17, 16, 5, 1, 1)
        spacerItem18 = QtWidgets.QSpacerItem(40, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout_2.addItem(spacerItem18, 4, 6, 1, 1)
        spacerItem19 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_2.addItem(spacerItem19, 4, 9, 1, 1)
        spacerItem20 = QtWidgets.QSpacerItem(20, 120, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Maximum)
        self.gridLayout_2.addItem(spacerItem20, 10, 5, 1, 1)
        spacerItem21 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout_2.addItem(spacerItem21, 6, 5, 1, 1)
        spacerItem22 = QtWidgets.QSpacerItem(30, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_2.addItem(spacerItem22, 4, 5, 1, 1)
        spacerItem23 = QtWidgets.QSpacerItem(40, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout_2.addItem(spacerItem23, 4, 4, 1, 1)
        spacerItem24 = QtWidgets.QSpacerItem(20, 250, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Maximum)
        self.gridLayout_2.addItem(spacerItem24, 8, 5, 1, 1)
        spacerItem25 = QtWidgets.QSpacerItem(20, 15, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.gridLayout_2.addItem(spacerItem25, 0, 1, 1, 1)
        spacerItem26 = QtWidgets.QSpacerItem(40, 40, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Expanding)
        self.gridLayout_2.addItem(spacerItem26, 4, 8, 1, 1)
        spacerItem27 = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_2.addItem(spacerItem27, 16, 7, 1, 1)
        spacerItem28 = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_2.addItem(spacerItem28, 4, 0, 1, 1)
        self.tabWidget.addTab(self.AddRoute, "")
        self.Pricing = QtWidgets.QWidget()
        self.Pricing.setObjectName("Pricing")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.Pricing)
        self.gridLayout_3.setObjectName("gridLayout_3")
        spacerItem29 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_3.addItem(spacerItem29, 7, 6, 1, 1)
        self.PricingExcelUpload = QtWidgets.QPushButton(parent=self.Pricing)
        self.PricingExcelUpload.setObjectName("PricingExcelUpload")
        self.gridLayout_3.addWidget(self.PricingExcelUpload, 3, 3, 1, 1)
        self.PricingExcelPath = QtWidgets.QLineEdit(parent=self.Pricing)
        self.PricingExcelPath.setObjectName("PricingExcelPath")
        self.gridLayout_3.addWidget(self.PricingExcelPath, 3, 1, 1, 2)
        self.PricingExcelTable = QtWidgets.QTableView(parent=self.Pricing)
        self.PricingExcelTable.setObjectName("PricingExcelTable")
        self.gridLayout_3.addWidget(self.PricingExcelTable, 5, 1, 1, 6)
        spacerItem30 = QtWidgets.QSpacerItem(125, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_3.addItem(spacerItem30, 7, 4, 1, 1)
        spacerItem31 = QtWidgets.QSpacerItem(20, 10, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.gridLayout_3.addItem(spacerItem31, 4, 1, 1, 1)
        spacerItem32 = QtWidgets.QSpacerItem(130, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_3.addItem(spacerItem32, 7, 3, 1, 1)
        spacerItem33 = QtWidgets.QSpacerItem(20, 20, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.gridLayout_3.addItem(spacerItem33, 7, 5, 1, 1)
        self.PricingTitle = QtWidgets.QLabel(parent=self.Pricing)
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(26)
        font.setBold(True)
        self.PricingTitle.setFont(font)
        self.PricingTitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.PricingTitle.setObjectName("PricingTitle")
        self.gridLayout_3.addWidget(self.PricingTitle, 1, 1, 1, 7)
        spacerItem34 = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_3.addItem(spacerItem34, 5, 0, 1, 1)
        spacerItem35 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.gridLayout_3.addItem(spacerItem35, 2, 1, 1, 1)
        spacerItem36 = QtWidgets.QSpacerItem(50, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_3.addItem(spacerItem36, 5, 8, 1, 1)
        spacerItem37 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Fixed)
        self.gridLayout_3.addItem(spacerItem37, 0, 1, 1, 1)
        self.SavePricing = QtWidgets.QPushButton(parent=self.Pricing)
        self.SavePricing.setObjectName("SavePricing")
        self.gridLayout_3.addWidget(self.SavePricing, 7, 7, 1, 1)
        spacerItem38 = QtWidgets.QSpacerItem(100, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_3.addItem(spacerItem38, 7, 2, 1, 1)
        self.PricingClear = QtWidgets.QPushButton(parent=self.Pricing)
        self.PricingClear.setObjectName("PricingClear")
        self.gridLayout_3.addWidget(self.PricingClear, 7, 1, 1, 1)
        spacerItem39 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Minimum)
        self.gridLayout_3.addItem(spacerItem39, 5, 7, 1, 1)
        self.ChangePricingStructure = QtWidgets.QPushButton(parent=self.Pricing)
        self.ChangePricingStructure.setMinimumSize(QtCore.QSize(150, 0))
        self.ChangePricingStructure.setObjectName("ChangePricingStructure")
        self.gridLayout_3.addWidget(self.ChangePricingStructure, 6, 7, 1, 1)
        self.tabWidget.addTab(self.Pricing, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        spacerItem40 = QtWidgets.QSpacerItem(20, 30, QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacerItem40)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1096, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.RecordTable.setShowGrid(True)
        self.PricingExcelTable.setShowGrid(True)
        self.AddPriceButton = QtWidgets.QPushButton(parent=self.Pricing)
        self.AddPriceButton.setObjectName("AddPriceButton")
        self.AddPriceButton.setText("Add Price")
        self.gridLayout_3.addWidget(self.AddPriceButton, 5, 7, 1,
                                    1)

        self.graphView = PannableView(self.AddRoute)
        self.graphView.setObjectName("graphView")
        self.gridLayout_2.addWidget(self.graphView, 2, 1, 14, 11)

        self.btnLanguage = QtWidgets.QPushButton('TR')
        self.btnLanguage.setFixedSize(30, 14)
        self.btnLanguage.setCheckable(True)
        self.btnLanguage.setObjectName("btnLanguage")
        self.btnLanguage.setChecked(True)
        self.btnLanguage.setStyleSheet('''
            QPushButton {
                background-color: white;
                border: 1px solid black;
                font-size: 10px;
                color: black;
                padding: 0px;
            }
            
            QPushButton:hover {
                background-color: #CC1F145D;
                color: white;
            }
            '''

        )

        # Create a layout for the button, to align it to the left
        self.button_layout = QtWidgets.QHBoxLayout()
        self.button_layout.addWidget(self.btnLanguage)
        self.button_layout.addStretch(1)

        # Add this layout to your existing vertical layout
        self.horizontalLayout.addLayout(self.button_layout)
        '''
        STYLE SHEETS    
        '''
        self.RecordTable.setStyleSheet("""
                QTableWidget {
                    border: 1px solid black;
                    background-color: #FFFFFFF;
                }
                QTableWidget::item:selected { background-color: #66FF0093; }
                QHeaderView::section {
                    background-color: #FFFFFF;
                    padding: 4px;
                    border: 1px solid #000000;
                    font-size: 12pt;
                }
    
                QHeaderView::section:vertical {
                    border-top: 1px solid black;
                    background-color: #fffffff;
                }
                
    
                QTableWidget::item {
                    background-color: #0D1F145D;
                    border: 1px solid #000000;
                }
    
                QTableWidget {
                    gridline-color: black;
                    font-size: 12pt;
                }
            """)
        self.PricingExcelTable.setStyleSheet("""
            QTableView {
                border: 1px solid black;
                background-color: #FFFFFF;  /* Fixed the extra F */
                gridline-color: black;
                font-size: 12pt;
            }
            QTableView::item:selected { background-color: #66FF0093; }

            QHeaderView::section {
                background-color: #FFFFFF;
                padding: 4px;
                border: 1px solid #000000;
                font-size: 12pt;
            }

            QHeaderView::section:vertical {
                border-top: 1px solid black;
                background-color: #FFFFFF;  /* Fixed the extra F */
            }

            QTableView::item {
                background-color: #0D1F145D;
                border: 1px solid #000000;
            }
        """)


    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "LAV Route Planner"))
        __sortingEnabled = self.RecordTable.isSortingEnabled()
        self.RecordTable.setSortingEnabled(False)
        self.RecordTable.setSortingEnabled(__sortingEnabled)
        self.ClearRecord.setText(_translate("MainWindow", "ClearRecord"))
        self.AddRecord.setText(_translate("MainWindow", "AddRecord"))
        self.ExcelExport.setText(_translate("MainWindow", "Export Excel"))
        self.UploadExcel.setText(_translate("MainWindow", "Upload Excel"))
        self.RoutePlannerTitle.setText(_translate("MainWindow", "Route Planner "))
        self.DeleteRecord.setText(_translate("MainWindow", "Delete"))
        self.Calculate.setText(_translate("MainWindow", "Calculate"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Home), _translate("MainWindow", "Home"))
        self.label_2.setText(_translate("MainWindow", "Add Delivery Route"))
        self.saveRouteButton.setText(_translate("MainWindow", "Save Route"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.AddRoute), _translate("MainWindow", "Add Route"))
        self.PricingExcelUpload.setText(_translate("MainWindow", "Upload Excel"))
        self.PricingTitle.setText(_translate("MainWindow", "Pricing Update"))
        self.SavePricing.setText(_translate("MainWindow", "Save Pricing"))
        self.PricingClear.setText(_translate("MainWindow", "Clear All"))
        self.ChangePricingStructure.setText(_translate("MainWindow", "Change Structure"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.Pricing), _translate("MainWindow", "Pricing"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
