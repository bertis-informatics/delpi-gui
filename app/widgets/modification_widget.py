from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QSpinBox, QPushButton, QTableWidget, QTableWidgetItem, 
                               QHeaderView, QComboBox, QCheckBox)
from PySide6.QtCore import Qt

class ModificationWidget(QWidget):
    def __init__(self):
        from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                               QSpinBox, QPushButton, QTableWidget, QTableWidgetItem, 
                               QHeaderView, QComboBox, QCheckBox)
from PySide6.QtCore import Qt

class ModificationWidget(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(10)
        self.setLayout(layout)

        # 1. Global Settings
        top_layout = QHBoxLayout()
        lbl_max = QLabel("Max Mods / Peptide:")
        self.spin_max_mods = QSpinBox()
        self.spin_max_mods.setRange(0, 10)
        self.spin_max_mods.setValue(2)
        top_layout.addWidget(lbl_max)
        top_layout.addWidget(self.spin_max_mods)
        top_layout.addStretch()
        layout.addLayout(top_layout)

        # 2. Modifications Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Name", "Residue", "Location", "Fixed", "Del"])
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(200)
        
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)       
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents) 
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents) 
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents) 
        header.setSectionResizeMode(4, QHeaderView.Fixed) 
        self.table.setColumnWidth(4, 40)

        # Hide the 4th index (Del column) by default
        self.table.setColumnHidden(4, True)

        layout.addWidget(self.table)

        # 4. Manual Add Button (Hidden by default)
        btn_manual_add = QPushButton("Add Custom Row")
        layout.addWidget(btn_manual_add)
        btn_manual_add.setVisible(False)

    def add_mod_row(self, name, residue, location, is_fixed):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(name))
        self.table.setItem(row, 1, QTableWidgetItem(residue))
        
        combo_loc = QComboBox()
        combo_loc.addItems(["Anywhere", "Protein N-term", "Protein C-term", "Peptide N-term", "Peptide C-term"])
        idx = combo_loc.findText(location, Qt.MatchContains)
        if idx >= 0: 
            combo_loc.setCurrentIndex(idx)
        self.table.setCellWidget(row, 2, combo_loc)
        
        chk_widget = QWidget() 
        chk_layout = QHBoxLayout(chk_widget)
        chk_layout.setAlignment(Qt.AlignCenter) 
        chk_layout.setContentsMargins(0, 0, 0, 0)

        chk_box = QCheckBox() 
        chk_box.setChecked(is_fixed)
        chk_box.setEnabled(False)  # Temporarily disabled
        chk_layout.addWidget(chk_box)
        self.table.setCellWidget(row, 3, chk_widget)
        
        btn_del = QPushButton("X") 
        btn_del.setStyleSheet("color: white; background-color: #d9534f; font-weight: bold; border-radius: 2px;")
        btn_del.clicked.connect(self.delete_current_row)
        self.table.setCellWidget(row, 4, btn_del)

    def delete_current_row(self):
        btn = self.sender()
        if not btn: 
            return
            
        for r in range(self.table.rowCount()):
            if self.table.cellWidget(r, 4) == btn:
                self.table.removeRow(r)
                return
            
    def clear_rows(self):
        self.table.setRowCount(0)
    
    def get_data(self):
        mods_list = []
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            res_item = self.table.item(row, 1)
            
            if not name_item or not res_item:
                continue
                
            name = name_item.text()
            residue = res_item.text()

            combo = self.table.cellWidget(row, 2)
            location_text = combo.currentText() if combo else "Anywhere"
            
            # Convert string to YAML format (e.g., Protein N-term -> protein_n_term)
            location_map = {
                "Anywhere": "anywhere",
                "Protein N-term": "protein_n_term",
                "Protein C-term": "protein_c_term",
                "Peptide N-term": "peptide_n_term",
                "Peptide C-term": "peptide_c_term"
            }
            location = location_map.get(location_text, "anywhere")

            # 3. Extract Fixed property (QCheckBox)
            # Structure: CellWidget -> Layout -> CheckBox
            chk_widget = self.table.cellWidget(row, 3)
            is_fixed = False
            
            if chk_widget:
                # Find the checkbox inside the widget layout using findChild
                chk_box = chk_widget.findChild(QCheckBox)
                if chk_box:
                    is_fixed = chk_box.isChecked()

            mods_list.append({
                "mod_name": name,
                "residue": residue,
                "location": location,
                "fixed": is_fixed
            })
            
        return mods_list
    
    