from PyQt6 import QtWidgets
from PyQt6.QtCore import QEvent,QSettings
import networkx as nx
from qt_material import apply_stylesheet
import ui_main
import pandas as pd
from PyQt6.QtWidgets import QApplication,QTableWidgetItem, QDialog ,QFileDialog , QMessageBox, QGraphicsScene,QMenu,QInputDialog,QLineEdit
from PyQt6.QtGui import QStandardItemModel, QStandardItem,QShortcut,QKeySequence

from Modules.VRPModel import   Heuristic
from Modules.addrecord import RecordDialog,DynamicPriceRecordDialog
from Assets.Data.Pricing.pricing_structure import StructureChange , custom_sort, edges_calc,remove_accents
from Modules.route import *
from Modules.GraphScreen import *
import pickle
import json
from Modules.data_process import  create_distance_matrices_dict
import re
# from Modules.VRPModel import VRPSolver
import os
import subprocess
import tempfile
import numpy as np
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class MyApp(QtWidgets.QMainWindow, ui_main.Ui_MainWindow):
    def __init__(self):
        super().__init__()

        # Setup UI from the ui_main.py
        self.setupUi(self)
        self.UploadExcel.clicked.connect(self.openFileDialog)
        self.RecordTable.cellClicked.connect(self.handleCellClicked)
        self.RecordTable.itemChanged.connect(self.handle_item_changed)
        self.ClearRecord.clicked.connect(self.clear_records)
        self.PricingClear.clicked.connect(self.pricing_clear_records)
        self.DeleteRecord.clicked.connect(self.delete_selected_rows)

        self.df = pd.DataFrame()
        self.AddRecord.clicked.connect(self.on_add_record)
        self.tabWidget.setCurrentIndex(0)
        self.PricingExcelUpload.clicked.connect(self.upload_new_data)
        # self.PricingExcelTable.cellClicked.connect(self.handlePricingCellClicked)
        # self.PricingExcelTable.itemChanged.connect(self.handle_pricing_item_changed)
        self.pricing_df = pd.read_excel(resource_path("Assets/Data/Pricing/prices.xlsx"), header=None, names=['Names', 'Prices'])
        self.pricing_df['Prices'] = self.pricing_df['Prices'].apply(lambda x: int(str(x).replace('₺', '').replace('.', '').replace(',', '.')))
        self.pricing_alternate_df = pd.read_csv(resource_path("Assets/Data/Pricing/pricing.csv"))
        cols_to_convert = ['Truck', 'Lorry', 'Iveco', 'Bulk']
        for col in cols_to_convert:
            self.pricing_alternate_df[col] = self.pricing_alternate_df[col].astype(int)
        self.display_data(self.pricing_df)
        self.SavePricing.clicked.connect(self.save_pricing)
        self.is_alternate_structure = False
        self.ChangePricingStructure.clicked.connect(lambda: self.toggle_pricing_structure(True))
        self.AddPriceButton.clicked.connect(self.show_price_dialog)
        self.excel_uploaded = False

        self.scene = QGraphicsScene(self)
        self.graphView.setScene(self.scene)

        # Attributes to store created nodes and routes
        self.nodes = []
        self.routes = []
        self.graph = nx.DiGraph()  # Directed graph

        # Attributes to store created nodes and routes
        self.selected_nodes = []
        self.context_menu_triggered = False
        # Connecting signals to slots
        self.graphView.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.graphView.customContextMenuRequested.connect(self.showContextMenu)
        self.graphView.setDragMode(QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)

        self.graphView.viewport().installEventFilter(self)
        self.saveRouteButton.clicked.connect(self.route_save)

        self.settings = QSettings('LAV', 'Route Planner')
        self.current_language = 'TR'
        with open(resource_path("Assets/Data/language.json"), "r", encoding="utf-8") as f:
            self.translations = json.load(f)
        if self.current_language == 'TR':
            self.btnLanguage.setText('TR')
        else:
            self.btnLanguage.setText('ENG')
        self.setLanguage(self.current_language)
        self.btnLanguage.clicked.connect(self.toggle_language)

        self.load_graph()

        self.vehicle_cap = {
            'Iveco':  10000,
            'Truck':  33000,
            'Lorry':  20000,
            'Bulk': 20000
        }
        self.distance_matrices, self.city_to_index, self.index_to_city = create_distance_matrices_dict()

        self.copy_shortcut = QShortcut(QKeySequence("Ctrl+C"), self.RecordTable)
        self.copy_shortcut.activated.connect(self.copy_to_clipboard)
        self.Calculate.clicked.connect(self.optimize_route)
        self.ExcelExport.clicked.connect(self.process_and_export)
        self.temp_files = self.load_temp_files()

    def process_and_export(self):
        if self.df.empty:
            return None
        df = self.df.copy()
        df.sort_values(by=['arac id', 'arac tipi'], inplace=True)

        modified_df = pd.DataFrame(columns=df.columns)
        previous_vehicle_id = None
        previous_vehicle_type = None

        for index, row in df.iterrows():
            vehicle_id = row['arac id']
            vehicle_type = row['arac tipi']
            if (vehicle_id != previous_vehicle_id) or (vehicle_type != previous_vehicle_type):
                blank_row = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
                modified_df = pd.concat([modified_df, blank_row], ignore_index=True)
                previous_vehicle_id = vehicle_id
                previous_vehicle_type = vehicle_type
            modified_df = pd.concat([modified_df, pd.DataFrame([row.values], columns=df.columns)], ignore_index=True)

        modified_df.drop(columns=['arac id'], inplace=True)
        modified_df.reset_index(drop=True, inplace=True)

        # Create a temporary file to hold the Excel data
        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False, prefix='cozum_')
        temp_filename = temp_file.name
        self.temp_files.append(temp_filename)
        modified_df.to_excel(temp_filename, index=False)

        try:
            # Attempt to open the temporary Excel file with the default application
            if os.name == 'nt':  # For Windows
                os.startfile(temp_filename)
            else:  # For macOS and Linux
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, temp_filename])

        except Exception as e:
            print(f"Failed to open the file: {e}")



    def optimize_route(self):
        self.delete_temp_files()
        if self.df is None or self.df.empty:
            return None

        vehicle_types = ['Iveco', 'Truck', 'Bulk', 'Lorry']
        csv_files = [resource_path('Assets/Data/Pricing/Iveco_matrix.csv'), resource_path('Assets/Data/Pricing/Truck_matrix.csv'), resource_path('Assets/Data/Pricing/Bulk_matrix.csv'), resource_path('Assets/Data/Pricing/Lorry_matrix.csv')]
        self.customer_dict = self.aggregate_customer_data()
        cost_matrices = {
            vehicle_type: pd.read_csv(csv_file, index_col=0, header=0)
            for vehicle_type, csv_file in zip(vehicle_types, csv_files)
        }
        self.pricing_alternate_df['Bulk'] += self.pricing_alternate_df['Truck']
        VRP_model = Heuristic(cost_matrices,self.pricing_alternate_df, self.customer_dict, self.vehicle_cap,self.df)
        df = VRP_model.final_df
        self.df = df
        self.update_record_table()
        message = self.translations[self.current_language]["CalculationFinished"]
        QtWidgets.QMessageBox.information(self, "", message)

    def save_temp_files(self):
        with open('temp_files.json', 'w') as f:
            json.dump(self.temp_files, f)

    def load_temp_files(self):
        if os.path.exists('temp_files.json'):
            with open('temp_files.json', 'r') as f:
                return json.load(f)
        return []

    def delete_temp_files(self):
        try:
            for temp_filename in self.temp_files:
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            self.temp_files.clear()
            self.save_temp_files()
        except:
            pass

    def update_record_table(self):
        if self.df is None or self.df.empty:

            return

        self.RecordTable.setRowCount(self.df.shape[0])  # number of rows in the DataFrame
        self.RecordTable.setColumnCount(self.df.shape[1])  # number of columns

        self.RecordTable.setHorizontalHeaderLabels(self.df.columns)

        for row in range(self.df.shape[0]):
            # Setting row number as the vertical header
            header_item = QTableWidgetItem(str(row + 1))  # +1 so that the row numbering starts from 1, not 0
            header_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.RecordTable.setVerticalHeaderItem(row, header_item)

            for col in range(self.df.shape[1]):
                item = QTableWidgetItem(str(self.df.iloc[row, col]))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.RecordTable.setItem(row, col, item)

    def openFileDialog(self):
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Excel File", "",
                                                  "Excel Files (*.xls *.xlsx);;All Files (*)")

        if filePath:
            all_data = pd.read_excel(filePath, header=None)

            desired_columns = ['Müşteri Kodu', 'ŞİPARİŞ NO', 'SEVK TARİHİ', 'MÜŞTERİ', 'KOLİ', 'PALET', 'SEVK ADRESİ']
            desired_columns_processed = [remove_accents(col).lower() for col in desired_columns]

            header_row_index = None

            for index, row in all_data.iterrows():
                row_processed = [remove_accents(str(cell)).lower() for cell in row]
                if all(desired_col in row_processed for desired_col in desired_columns_processed):
                    header_row_index = index
                    break

            if header_row_index is not None:
                all_data = pd.read_excel(resource_path(filePath), header=header_row_index)
                all_data.columns = [remove_accents(col).lower() for col in all_data.columns]
                print(all_data.columns)
            else:
                print("Desired columns not found in the data.")

            self.df = all_data[desired_columns_processed].dropna(how='all')
            self.df['siparis no'] = pd.to_numeric(self.df['siparis no'], errors='coerce', downcast='integer')
            self.df['sevk tarihi'] = pd.to_datetime(self.df['sevk tarihi']).dt.date
            self.ExcelPath.setText(filePath)
            self.RecordTable.setRowCount(self.df.shape[0])  # number of rows in the DataFrame
            self.RecordTable.setColumnCount(self.df.shape[1])  # number of columns

            self.update_record_table()

            self.UploadExcel.clearFocus()

        self.customer_dict = self.aggregate_customer_data()
        print(self.customer_dict)



    def aggregate_customer_data(self):
        customer_dict = {}
        unmatched_entries = []
        for index, row in self.df.iterrows():
            customer_id = row['musteri kodu']
            city = self.clean_city(row['sevk adresi'])
            siparis_no = row['siparis no']
            musteri_ismi = row['musteri']
            koli = row['koli']
            demand = row['palet']
            demand = re.sub(r'[^\d.]', '', demand)
            demand = float(demand)

            if city is None:  # If city is not found
                unmatched_entry = {
                    'customer_id': customer_id,
                    'siparis_no': siparis_no,
                    'musteri_ismi': musteri_ismi,
                    'koli': koli,
                    'demand': demand
                }
                unmatched_entries.append(unmatched_entry)
            else:
                if customer_id not in customer_dict:
                    customer_dict[customer_id] = {
                        'musteri_ismi': musteri_ismi,
                        'cities': {}
                    }

                if city not in customer_dict[customer_id]['cities']:
                    customer_dict[customer_id]['cities'][city] = {
                        'siparis_no': {},
                        'total_demand': 0
                    }

                city_dict = customer_dict[customer_id]['cities'][city]
                if siparis_no not in city_dict['siparis_no']:
                    city_dict['siparis_no'][siparis_no] = {}

                # Assuming each siparis_no and koli combination is unique
                city_dict['siparis_no'][siparis_no][koli] = demand
                # Update total_demand for this city
                city_dict['total_demand'] += demand
        return customer_dict


    def clean_city(self,city):
        with open(resource_path('Assets/Data/city_mapping.json')) as f:
            city_mappings = json.load(f)


        # Normalize the city name (convert to lowercase, remove extra spaces, etc.)
        cleaned_city = city.lower().replace(" ", "")
        cleaned_city = remove_accents(cleaned_city)
        city_list = list(self.city_to_index.keys())
        # First, check if the cleaned city is in the city list
        for known_city in city_list:
            if cleaned_city == known_city:
                return known_city  # Return the known city name from city_list

        # Second, check the first three characters against your city list
        prefix = cleaned_city[:max(3,len(cleaned_city))]
        for known_city in city_list:
            if prefix == known_city[:max(3,len(cleaned_city))]:
                return known_city  # Return the known city name from city_list

        if cleaned_city in city_mappings:
            return city_mappings[cleaned_city]

        for district, mapped_city in city_mappings.items():
            if prefix in remove_accents(district.lower()):
                return mapped_city

        for known_city in city_list:
            if known_city in cleaned_city:
                return known_city  # Return the known city name from city_list

        return None  # Return None if no match found

    def copy_to_clipboard(self):
        selected = self.RecordTable.selectedRanges()
        if selected:
            rows = selected[0].rowCount()
            cols = selected[0].columnCount()
            row_idx = selected[0].topRow()
            col_idx = selected[0].leftColumn()

            clipboard_data = ""
            for r in range(row_idx, row_idx + rows):
                row_data = []
                for c in range(col_idx, col_idx + cols):
                    item = self.RecordTable.item(r, c)
                    if item is not None:
                        row_data.append(item.text())
                    else:
                        row_data.append("")
                clipboard_data += '\t'.join(row_data) + '\n'

            clipboard = QApplication.clipboard()
            clipboard.setText(clipboard_data)

    def toggle_language(self):
        if self.current_language == 'ENG':
            self.current_language = 'TR'
            self.btnLanguage.setText('TR')
            self.settings.setValue('language', 'TR')
        else:
            self.current_language = 'ENG'
            self.btnLanguage.setText('ENG')
            self.settings.setValue('language', 'ENG')
        self.setLanguage(self.current_language)

    def setLanguage(self, language):



        for widget_name, translation in self.translations[language].items():
            widget = self.centralwidget.findChild(QtWidgets.QWidget, widget_name)
            if isinstance(widget, QtWidgets.QLabel) or isinstance(widget, QtWidgets.QPushButton) :
                widget.setText(translation)
            elif isinstance(widget, QtWidgets.QWidget):
                tab_widget = self.centralwidget.findChild(QtWidgets.QTabWidget)  # Find the QTabWidget
                if tab_widget:
                    index = tab_widget.indexOf(widget)
                    if index != -1:
                        tab_widget.setTabText(index, translation)
        self.btnLanguage.setText('TR' if language == 'TR' else 'ENG')

    def edges_connected_to(self, node):
        edges = []
        for item in self.scene.items():
            if isinstance(item, GraphicsEdge) and (item.start_node == node or item.end_node == node):
                edges.append(item)
        return edges

    def save_graph(self):
        data = {
            'graph': self.graph,
            'nodes': [node.node_id for node in self.nodes],  # Assuming node.node_id represents the node name.
            'routes': self.routes
        }

        with open(resource_path('Assets/Data/graph_data.pkl'), 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=True)

    def load_graph(self):
        try:
            with open(resource_path('Assets/Data/graph_data.pkl'), 'rb') as f:
                data = pickle.load(f, fix_imports=True, encoding="utf-8")

                self.graph = data['graph']

                self.redraw()

                self.adjust_scene_after_loading()
        except FileNotFoundError:
            pass


    def route_save(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(self.translations[self.current_language]["SaveRoutesDialogText"])
        msg.setWindowTitle(self.translations[self.current_language]["SaveRoutesDialogTitle"])
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel)
        msg.button(QMessageBox.StandardButton.Yes).setText(
            self.translations[self.current_language]["SaveRoutesDialogYes"])
        msg.button(QMessageBox.StandardButton.Cancel).setText(
            self.translations[self.current_language]["SaveRoutesDialogCancel"])

        retval = msg.exec()


        if retval == QMessageBox.StandardButton.Yes:
            self.save_graph()

            save_msg = QMessageBox(self)
            save_msg.setIcon(QMessageBox.Icon.Information)
            save_msg.setText(self.translations[self.current_language]["RouteSavedText"])
            save_msg.setWindowTitle(self.translations[self.current_language]["RouteSavedTitle"])
            save_msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            save_msg.button(QMessageBox.StandardButton.Ok).setText(
                self.translations[self.current_language]["RouteSavedOk"])
            save_msg.exec()




    def eventFilter(self, source, event):
        if event.type() == QEvent.Type.MouseButtonPress and source == self.graphView.viewport():
            items = self.graphView.items(event.pos())
            if not any(isinstance(item, GraphicsNode) for item in items):
                self.selected_nodes.clear()
        return super().eventFilter(source, event)

    def redraw(self):
        self.scene.clear()

        # Redraw nodes based on self.graph data
        node_items = {}
        for node, attributes in self.graph.nodes(data=True):
            pos = attributes.get('pos', (0, 0))
            node_item = GraphicsNode(node, *pos, main_app=self)
            self.scene.addItem(node_item)
            node_items[node] = node_item

        for u, v in self.graph.edges():
            edge_item = GraphicsEdge(node_items[u], node_items[v], main_app=self)
            self.scene.addItem(edge_item)



    def adjust_scene_after_loading(self):
        padding = 200
        max_x = max_y = float('-inf')
        min_x = min_y = float('inf')

        # Find maximum and minimum coordinates among all nodes
        for node, attributes in self.graph.nodes(data=True):
            pos = attributes.get('pos', (0, 0))
            max_x = max(max_x, pos[0])
            min_x = min(min_x, pos[0])
            max_y = max(max_y, pos[1])
            min_y = min(min_y, pos[1])

        # Compute new bounds with padding
        new_left = min_x - padding
        new_right = max_x + padding
        new_top = min_y - padding
        new_bottom = max_y + padding

        current_rect = self.scene.sceneRect()

        # Update the scene rectangle only if computed bounds are larger than the current rectangle
        if (new_left < current_rect.left() or
                new_right > current_rect.right() or
                new_top < current_rect.top() or
                new_bottom > current_rect.bottom()):
            self.graphView.setSceneRect(QRectF(new_left, new_top, new_right - new_left, new_bottom - new_top))

    def adjust_scene_for_node(self, node_item):
        scene_rect = self.graphView.sceneRect()
        node_rect = node_item.sceneBoundingRect()

        if not scene_rect.contains(node_rect):
            new_left = min(scene_rect.left(), node_rect.left())
            new_right = max(scene_rect.right(), node_rect.right())
            new_top = min(scene_rect.top(), node_rect.top())
            new_bottom = max(scene_rect.bottom(), node_rect.bottom())

            self.graphView.setSceneRect(QRectF(new_left, new_top, new_right - new_left, new_bottom - new_top))

    def add_node(self, position):
        name, ok = QInputDialog.getText(self, 'Name Node', 'Enter node name:')
        if ok:
            node_id = name
            self.graph.add_node(node_id, pos=(position.x(), position.y()), label=name)
            self.redraw()
            # Fetch the node item after drawing and then adjust the scene
            for item in self.scene.items(position):
                if isinstance(item, GraphicsNode):
                    self.adjust_scene_for_node(item)
                    break


    def add_route(self):
        if len(self.selected_nodes) == 2:
            start_node, end_node = self.selected_nodes
            if not self.graph.has_edge(start_node.node_id, end_node.node_id):  # Check if edge already exists
                self.graph.add_edge(start_node.node_id, end_node.node_id)
            self.selected_nodes.clear()  # Clear selection
            self.redraw()

    def delete_node(self, node):
        self.graph.remove_node(node.node_id)
        self.redraw()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.delete_selected_node()
        super().keyPressEvent(event)

    def delete_selected_node(self):
        nodes_to_delete = list(self.selected_nodes)  # Copy the list of selected nodes

        for node in nodes_to_delete:
            self.graph.remove_node(node.node_id)
            self.selected_nodes.remove(node)
        # Redraw the scene to reflect changes
        self.redraw()

    def showContextMenu(self, position):
        self.context_menu_triggered = True


        global_position = self.graphView.mapToGlobal(position)  # position in global coordinates
        scene_position = self.graphView.mapToScene(position)  # position in scene coordinates

        contextMenu = QMenu(self)

        # Check if we clicked on a node
        clicked_items = self.graphView.items(position)
        clicked_node = None
        for item in clicked_items:
            if isinstance(item, GraphicsNode):
                clicked_node = item
                break


        deleteNodeAction = None
        lang = self.current_language
        addNodeAction = contextMenu.addAction(self.translations[lang]["Add Node"])
        addRouteAction = contextMenu.addAction(self.translations[lang]["Add Route Between Selected Nodes"])
        if clicked_node:
            deleteNodeAction = contextMenu.addAction(self.translations[lang]["Delete Node"])
            renameNodeAction = contextMenu.addAction(self.translations[lang]["Rename Node"])

        action = contextMenu.exec(global_position)
        try:
            if action == addNodeAction:
                self.add_node(scene_position)
            elif action == addRouteAction:
                self.add_route()
            elif action == deleteNodeAction and clicked_node:
                self.delete_node(clicked_node)
            elif action == renameNodeAction and clicked_node:
                self.rename_node(clicked_node)

            self.context_menu_triggered = False
        except:
            return None

    def rename_node(self, node):
        old_name = node.node_id
        new_name, ok = QInputDialog.getText(self, 'Rename Node', 'Enter new node name:', QLineEdit.EchoMode.Normal, old_name)
        if ok and new_name and new_name != old_name:
            # Update the graph structure with the new node name
            mapping = {old_name: new_name}
            nx.relabel_nodes(self.graph, mapping, copy=False)
            # Update node's label if it's being used as a display name
            self.graph.nodes[new_name]['label'] = new_name
            self.redraw()


    def display_data(self, df):
        model = QStandardItemModel()

        model.setHorizontalHeaderLabels(df.columns.tolist())

        for index, row in df.iterrows():
            formatted_row = []
            for value in row:
                if isinstance(value, float):
                    formatted_value = "{:.2f}".format(value)
                else:
                    formatted_value = str(value)
                formatted_row.append(QStandardItem(formatted_value))

            model.appendRow(formatted_row)


        self.PricingExcelTable.setModel(model)

    def toggle_pricing_structure(self,pref=True):
        if self.is_alternate_structure and pref:
            self.is_alternate_structure = False
            self.display_data(self.pricing_df)
        else:
            self.is_alternate_structure = True
            self.display_data(self.pricing_alternate_df)


    def upload_new_data(self):
        self.PricingExcelPath.clear()
        self.pricing_df = custom_sort(self.pricing_df)
        filePath, _ = QFileDialog.getOpenFileName(self, "Open Excel File", "", "Excel Files (*.xlsx);;All Files (*)")
        if filePath:
            self.PricingExcelPath.setText(filePath)
            self.toggle_pricing_structure()
            self.is_alternate_structure = False
            column_names = ['Names', 'Prices']
            new_df = pd.read_excel(filePath, header=None, names=column_names)
            new_df['Prices'] = new_df['Prices'].apply(lambda x: int(str(x).replace('₺', '').replace('.', '').replace(',', '.')))
            self.pricing_df['New Prices'] = new_df['Prices']  # Assuming 'Price' is your column name
            self.pricing_df['Difference'] = pd.to_numeric(self.pricing_df['New Prices']) - pd.to_numeric(self.pricing_df['Prices'])

            self.display_data_with_colors(self.pricing_df)
            try:
                self.ChangePricingStructure.clicked.disconnect(self.toggle_pricing_structure)
            except TypeError:
                pass
            self.excel_uploaded = True

        self.PricingExcelUpload.clearFocus()

    def show_price_dialog(self):

        model = self.PricingExcelTable.model()
        columns = [model.headerData(i, Qt.Orientation.Horizontal) for i in range(model.columnCount())]

        dialog = DynamicPriceRecordDialog(columns, self)# Apply the translation
        result = dialog.exec()

        if result == QDialog.DialogCode.Accepted.value:
            record = dialog.get_record()

            if self.is_alternate_structure:
                self.pricing_alternate_df = pd.concat([self.pricing_alternate_df, pd.DataFrame([record])],
                                                      ignore_index=True)
                self.display_data(self.pricing_alternate_df)
            else:
                self.pricing_df = pd.concat([self.pricing_df, pd.DataFrame([record])], ignore_index=True)
                self.pricing_df = custom_sort(self.pricing_df)
                self.display_data(self.pricing_df)

    def display_data_with_colors(self, df):
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(df.columns.to_list())

        for _, row in df.iterrows():
            items = []
            for col in df.columns:
                value = row[col]
                item = QStandardItem(str(value))
                if col == 'Difference':
                    if value > 0:
                        item.setBackground(Qt.GlobalColor.green)
                    elif value < 0:
                        item.setBackground(Qt.GlobalColor.red)
                items.append(item)
            model.appendRow(items)

        self.PricingExcelTable.setModel(model)

    def save_pricing(self):
        # Create an empty DataFrame
        changed=False
        df = pd.DataFrame()
        if  self.is_alternate_structure ==True:
            self.toggle_pricing_structure()
            self.is_alternate_structure = False
            changed= True
        # Access the model of the QTableView
        model = self.PricingExcelTable.model()
        rows, cols = model.rowCount(), model.columnCount()

        for row in range(rows):
            data = {}
            for col in range(cols):
                header = model.headerData(col, Qt.Orientation.Horizontal)
                item = model.item(row, col)
                if item:
                    data[header] = item.text()
                else:
                    data[header] = None
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

        if 'New Prices' in df.columns:
            df['Prices'] = df['New Prices']
        else:
            df['Prices'] = df['Prices']
        self.pricing_df = df[['Names', 'Prices']]
        df[['Names', 'Prices']].to_excel(resource_path("Assets/Data/Pricing/prices.xlsx"), index=False, header=False)
        self.pricing_alternate_df = StructureChange(df, resource_path('Assets/Data/Pricing/pricing.csv'))
        cols_to_convert = ['Truck', 'Lorry', 'Iveco', 'Bulk']
        for col in cols_to_convert:
            self.pricing_alternate_df[col] = self.pricing_alternate_df[col].astype(int)

        if changed ==True:
            self.toggle_pricing_structure(False)
            self.is_alternate_structure = True

        self.display_data(self.pricing_df)
        self.pricing_df = custom_sort(self.pricing_df)
        title_translation = self.translations[self.current_language]["SavePricingSuccessTitle"]
        message_translation = self.translations[self.current_language]["SavePricingSuccessMessage"]
        ok_button_translation = self.translations[self.current_language]["OKButtonText"]

        msg = QMessageBox(self)
        msg.setWindowTitle(title_translation)  # Set translated title
        msg.setText(message_translation)  # Set translated message
        msg.setIcon(QMessageBox.Icon.Information)
        ok_button = msg.addButton(ok_button_translation, QMessageBox.ButtonRole.AcceptRole)
        msg.exec()

        self.SavePricing.clearFocus()
        edges_calc(self.graph.edges)
        self.distance_matrices, self.city_to_index, self.index_to_city = create_distance_matrices_dict()
        print(self.city_to_index)


    def handle_item_changed(self, item):
        row, column = item.row(), item.column()
        # Update the value in the DataFrame
        self.df.iat[row, column] = item.text()

    def handleCellClicked(self, row, column):

        newColor = QColor("white")

        item = self.RecordTable.item(row, column)
        if item:
            item.setForeground(newColor)

    def clear_records(self):
        # Clear QTableWidget
        self.RecordTable.setRowCount(0)
        self.RecordTable.setColumnCount(0)
        self.ExcelPath.setText("")
        # Clear DataFrame
        self.df = pd.DataFrame()

    def pricing_clear_records(self):
        # Clear QTableWidget
        self.pricing_df = pd.read_excel(resource_path("Assets/Data/Pricing/prices.xlsx"), header=None, names=['Names', 'Prices'])
        self.pricing_df['Prices'] = self.pricing_df['Prices'].apply(lambda x: int(str(x).replace('₺', '').replace('.', '').replace(',', '.')))
        self.pricing_alternate_df = pd.read_csv(resource_path("Assets/Data/Pricing/pricing.csv"))
        cols_to_convert = ['Truck', 'Lorry', 'Iveco', 'Bulk']
        for col in cols_to_convert:
            self.pricing_alternate_df[col] = self.pricing_alternate_df[col].astype(int)
        self.is_alternate_structure = False
        self.display_data(self.pricing_df)

    def delete_selected_rows(self):
        selected_rows = set()  # To hold unique selected row numbers
        # Get all the selected items
        for item in self.RecordTable.selectedItems():
            selected_rows.add(item.row())
        # Delete rows in the DataFrame
        self.df.drop(self.df.index[list(selected_rows)], inplace=True)
        self.df = self.df.reset_index(drop=True)

        # Delete rows from the QTableWidget (from bottom to top)
        for row in sorted(selected_rows, reverse=True):
            self.RecordTable.removeRow(row)

    def on_add_record(self):
        # Assuming df and columns are defined or accessible here
        dialog = RecordDialog(self.df.columns, self)
        result = dialog.exec()
        self.AddRecord.clearFocus()
        if result == QDialog.DialogCode.Accepted:
            record = dialog.get_record()

            if all(value == "" for value in record.values()):
                return

            # Add record to df
            self.df = pd.concat([self.df, pd.DataFrame([record])], ignore_index=True)
            print(self.df)

            # Update table to reflect the new df
            row_position = self.RecordTable.rowCount()
            self.RecordTable.insertRow(row_position)

            for col, value in enumerate(record.values()):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.RecordTable.setItem(row_position, col, item)

if __name__ == '__main__':
    app = QtWidgets.QApplication([])

    extra = {

        # Font
        'font_family': 'Roboto',
        'text-transform': 'capitalize'
    }

    apply_stylesheet(app, theme=resource_path('Assets/AppStyle/colors.xml'), extra=extra, css_file=resource_path('./Assets/AppStyle/custom.css'))

    window = MyApp()
    global app_instance
    app_instance = window
    window.show()
    app.exec()
