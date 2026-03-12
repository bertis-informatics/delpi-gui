import os
import sys
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                               QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox, 
                               QTableWidget, QTableWidgetItem, QHeaderView, 
                               QWidget, QGroupBox, QAbstractItemView, QSizePolicy)
from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon

# Material 라이브러리 임포트
from qt_material import apply_stylesheet

def resource_path(relative_path):
    """실행 환경에 따른 리소스 절대 경로 반환"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_path, relative_path)

FIXED_MODS = [
    ("Carbamidomethyl (C)", "C", "+43.005814", "Carbamidomethyl", "C", "anywhere"),
    ("TMT6plex", "N-term, K", "+229.2634", "TMT6plex", "K, Peptide N-term", "anywhere"),
    ("TMTpro", "N-term, K", "+304.2071", "TMTpro", "K, Peptide N-term", "anywhere")
]

VAR_MODS = [
    ("Acetyl", "Protein N-term", "+42.010565", "Acetyl", "*", "protein_n_term"),
    ("Oxidation", "M", "+15.994915", "Oxidation", "M", "anywhere"),
    ("Deamidation", "N, Q", "+0.984016", "Deamidation", "N, Q", "anywhere"),
    ("Phospho", "S, T, Y", "+79.966331", "Phospho", "S, T, Y", "anywhere"),
    ("Pyro-glu (Q -> pyro-Glu)", "N-term, Q", "-17.026549", "Pyro-glu from Q", "Q", "peptide_n_term"),
    ("Pyro-glu (E -> pyro-Glu)", "N-term, E", "-18.010565", "Pyro-glu from E", "E", "peptide_n_term")
]

class AdvancedSettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")

        icon_path = resource_path(os.path.join("app", "ui", "delpi_icon_260304.png"))
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # 창 크기 고정
        self.setFixedSize(1100, 850) 
        
        apply_stylesheet(self, theme='light_blue.xml', invert_secondary=True)
        
        # 커스텀 QSS 적용 (레이아웃 여백만 조정)
        self.apply_custom_qss()

        self.init_ui()

    def apply_custom_qss(self):
        """외부 qss 파일을 로드하여 Material 테마 위에 덧씌웁니다."""
        qss_path = resource_path(os.path.join("app", "ui", "advanced_settings.qss"))
        
        if os.path.exists(qss_path):
            try:
                with open(qss_path, "r", encoding="utf-8") as f:
                    custom_qss = f.read()
                self.setStyleSheet(self.styleSheet() + custom_qss)
            except Exception as e:
                print(f"QSS Load Error: {e}")
        else:
            # 외부 QSS를 못 찾을 경우를 대비한 기본 백업 스타일
            self.setStyleSheet(self.styleSheet() + """
                QDialog { background-color: #F8FAFC; }
                QGroupBox { border: 1px solid #CBD5E1; border-radius: 8px; margin-top: 20px; background-color: #FFFFFF; }
                QGroupBox::title { subcontrol-origin: margin; left: 15px; padding: 0 5px; font-weight: bold; font-size: 15px; }
                QTableWidget QWidget { background-color: transparent; }
                QSpinBox, QDoubleSpinBox, QComboBox { min-height: 22px; max-height: 22px; padding: 0px 5px; font-size: 12px; }
                QComboBox QAbstractItemView { border: 1px solid #2196F3; background-color: white; }
            """)

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(40)

        # --- LEFT COLUMN ---
        left_col = QVBoxLayout()
        left_col.setSpacing(15)
        
        # 1. Digestion
        group_dig = QGroupBox("") # 타이틀 비움
        lay_dig = QVBoxLayout(group_dig)
        lay_dig.setSpacing(15)
        
        # 💡 커스텀 타이틀 추가
        title_dig = QLabel("Digestion")
        title_dig.setObjectName("group_title")
        lay_dig.addWidget(title_dig)

        row_enz = QHBoxLayout()
        lbl_enz = QLabel("Enzyme:")
        lbl_enz.setMinimumWidth(120)
        row_enz.addWidget(lbl_enz)
        self.combo_enzyme = QComboBox()
        self.combo_enzyme.addItems(["Trypsin", "Lys-C", "Arg-C"])
        row_enz.addWidget(self.combo_enzyme)
        row_enz.addStretch()
        lay_dig.addLayout(row_enz)

        row_len = QHBoxLayout()
        lbl_len = QLabel("Peptide length:")
        lbl_len.setMinimumWidth(120)
        row_len.addWidget(lbl_len)
        self.spin_min_len = QSpinBox(); self.spin_min_len.setRange(1, 100); self.spin_min_len.setValue(7)
        self.spin_max_len = QSpinBox(); self.spin_max_len.setRange(1, 100); self.spin_max_len.setValue(30)
        row_len.addWidget(self.spin_min_len); row_len.addWidget(QLabel(" - ", alignment=Qt.AlignCenter)); row_len.addWidget(self.spin_max_len)
        row_len.addStretch()
        lay_dig.addLayout(row_len)

        row_miss = QHBoxLayout()
        lbl_miss = QLabel("Missed cleavages:")
        lbl_miss.setMinimumWidth(120)
        row_miss.addWidget(lbl_miss)
        self.spin_missed = QSpinBox(); self.spin_missed.setRange(0, 10); self.spin_missed.setValue(1)
        row_miss.addWidget(self.spin_missed)
        row_miss.addStretch()
        lay_dig.addLayout(row_miss)

        row_nterm = QHBoxLayout()
        row_nterm.addWidget(QLabel("N-terminal methionine excision:"))
        self.chk_nterm = QCheckBox("")  
        self.chk_nterm.setChecked(True)
        row_nterm.addWidget(self.chk_nterm)
        row_nterm.addStretch()  
        lay_dig.addLayout(row_nterm)

        left_col.addWidget(group_dig)

        # 2. Mass Tolerance
        group_tol = QGroupBox("")
        lay_tol = QVBoxLayout(group_tol)
        lay_tol.setSpacing(10)

        # 💡 커스텀 타이틀 추가
        title_tol = QLabel("Mass Tolerance")
        title_tol.setObjectName("group_title")
        lay_tol.addWidget(title_tol)
        
        row_ms1 = QHBoxLayout()
        lbl_ms1 = QLabel("MS1:")
        lbl_ms1.setMinimumWidth(50) 
        row_ms1.addWidget(lbl_ms1)
        self.spin_ms1_tol = QSpinBox(); self.spin_ms1_tol.setRange(1, 100); self.spin_ms1_tol.setValue(10)
        row_ms1.addWidget(self.spin_ms1_tol)
        row_ms1.addWidget(QLabel("ppm"))
        row_ms1.addStretch()
        lay_tol.addLayout(row_ms1)

        row_ms2 = QHBoxLayout()
        lbl_ms2 = QLabel("MS2:")
        lbl_ms2.setMinimumWidth(50) 
        row_ms2.addWidget(lbl_ms2)
        self.spin_ms2_tol = QSpinBox(); self.spin_ms2_tol.setRange(1, 100); self.spin_ms2_tol.setValue(10)
        row_ms2.addWidget(self.spin_ms2_tol)
        row_ms2.addWidget(QLabel("ppm"))
        row_ms2.addStretch()
        lay_tol.addLayout(row_ms2)

        left_col.addWidget(group_tol)

        # 3. Precursor Ion
        group_pre = QGroupBox("")
        lay_pre = QVBoxLayout(group_pre)
        lay_pre.setSpacing(10)

        # 💡 커스텀 타이틀 추가
        title_pre = QLabel("Precursor Ion")
        title_pre.setObjectName("group_title")
        lay_pre.addWidget(title_pre)
        
        row_pre_z = QHBoxLayout()
        lbl_pre_z = QLabel("Charge:")
        lbl_pre_z.setMinimumWidth(80)
        row_pre_z.addWidget(lbl_pre_z)
        self.spin_pre_min_z = QSpinBox(); self.spin_pre_min_z.setRange(1, 10); self.spin_pre_min_z.setValue(1)
        self.spin_pre_max_z = QSpinBox(); self.spin_pre_max_z.setRange(1, 10); self.spin_pre_max_z.setValue(4)
        row_pre_z.addWidget(self.spin_pre_min_z); row_pre_z.addWidget(QLabel(" - ", alignment=Qt.AlignCenter)); row_pre_z.addWidget(self.spin_pre_max_z)
        row_pre_z.addStretch()
        lay_pre.addLayout(row_pre_z)

        row_pre_mz = QHBoxLayout()
        lbl_pre_mz = QLabel("m/z:")
        lbl_pre_mz.setMinimumWidth(80)
        row_pre_mz.addWidget(lbl_pre_mz)
        self.spin_pre_min_mz = QDoubleSpinBox(); self.spin_pre_min_mz.setRange(0, 5000); self.spin_pre_min_mz.setValue(300.0)
        self.spin_pre_max_mz = QDoubleSpinBox(); self.spin_pre_max_mz.setRange(0, 5000); self.spin_pre_max_mz.setValue(1800.0)
        row_pre_mz.addWidget(self.spin_pre_min_mz); row_pre_mz.addWidget(QLabel(" - ", alignment=Qt.AlignCenter)); row_pre_mz.addWidget(self.spin_pre_max_mz)
        row_pre_mz.addStretch()
        lay_pre.addLayout(row_pre_mz)
        
        left_col.addWidget(group_pre)

        # 4. Fragment Ion
        group_frag = QGroupBox("")
        lay_frag = QVBoxLayout(group_frag)
        lay_frag.setSpacing(10)

        # 💡 커스텀 타이틀 추가
        title_frag = QLabel("Fragment Ion")
        title_frag.setObjectName("group_title")
        lay_frag.addWidget(title_frag)

        row_frag_z = QHBoxLayout()
        lbl_frag_z = QLabel("Charge:")
        lbl_frag_z.setMinimumWidth(80)
        row_frag_z.addWidget(lbl_frag_z)
        self.spin_frag_min_z = QSpinBox(); self.spin_frag_min_z.setRange(1, 10); self.spin_frag_min_z.setValue(1)
        self.spin_frag_max_z = QSpinBox(); self.spin_frag_max_z.setRange(1, 10); self.spin_frag_max_z.setValue(2)
        row_frag_z.addWidget(self.spin_frag_min_z); row_frag_z.addWidget(QLabel(" - ", alignment=Qt.AlignCenter)); row_frag_z.addWidget(self.spin_frag_max_z)
        row_frag_z.addStretch()
        lay_frag.addLayout(row_frag_z)

        row_frag_mz = QHBoxLayout()
        lbl_frag_mz = QLabel("m/z:")
        lbl_frag_mz.setMinimumWidth(80)
        row_frag_mz.addWidget(lbl_frag_mz)
        self.spin_frag_min_mz = QDoubleSpinBox(); self.spin_frag_min_mz.setRange(0, 5000); self.spin_frag_min_mz.setValue(200.0)
        self.spin_frag_max_mz = QDoubleSpinBox(); self.spin_frag_max_mz.setRange(0, 5000); self.spin_frag_max_mz.setValue(1800.0)
        row_frag_mz.addWidget(self.spin_frag_min_mz); row_frag_mz.addWidget(QLabel(" - ", alignment=Qt.AlignCenter)); row_frag_mz.addWidget(self.spin_frag_max_mz)
        row_frag_mz.addStretch()
        lay_frag.addLayout(row_frag_mz)

        left_col.addWidget(group_frag)
        left_col.addStretch()

        # --- RIGHT COLUMN ---
        right_col = QVBoxLayout()
        right_col.setSpacing(15)

        # 5. Modification
        group_mods = QGroupBox("")
        lay_mods = QVBoxLayout(group_mods)
        lay_mods.setSpacing(15)

        # 💡 커스텀 타이틀 추가
        title_mods = QLabel("Modifications")
        title_mods.setObjectName("group_title")
        lay_mods.addWidget(title_mods)
        
        lay_mods.addWidget(QLabel("Fixed Modifications"))
        self.table_fixed = self.create_table(FIXED_MODS)
        lay_mods.addWidget(self.table_fixed)

        row_max = QHBoxLayout()
        row_max.addWidget(QLabel("Max variable modifications per peptide:"))
        self.spin_max_mods = QSpinBox(); self.spin_max_mods.setRange(0, 10); self.spin_max_mods.setValue(2)
        row_max.addWidget(self.spin_max_mods)
        row_max.addStretch()
        lay_mods.addLayout(row_max)

        lay_mods.addWidget(QLabel("Variable Modifications"))
        self.table_var = self.create_table(VAR_MODS)
        lay_mods.addWidget(self.table_var)
        
        right_col.addWidget(group_mods)

        # 6. Q-value
        group_q = QGroupBox("")
        lay_q_outer = QVBoxLayout(group_q) # 헤더를 넣기 위해 VBox로 먼저 설정
        
        title_q = QLabel("False Discovery Rate (FDR) Control")
        title_q.setObjectName("group_title")
        lay_q_outer.addWidget(title_q)

        row_q_content = QHBoxLayout()
        row_q_content.addWidget(QLabel("Q-value threshold: "))
        self.spin_qvalue = QDoubleSpinBox(); self.spin_qvalue.setSuffix(" %"); self.spin_qvalue.setDecimals(2); self.spin_qvalue.setSingleStep(0.1); self.spin_qvalue.setValue(1.0)
        row_q_content.addWidget(self.spin_qvalue)
        row_q_content.addStretch()
        lay_q_outer.addLayout(row_q_content)
        
        right_col.addWidget(group_q)
        right_col.addStretch()

        main_layout.addLayout(left_col, 4)
        main_layout.addLayout(right_col, 6)

    def create_table(self, mods_data):
        tbl = QTableWidget()
        tbl.setColumnCount(4)
        tbl.setHorizontalHeaderLabels(["PTM", "Site", "ΔMass", "Use"])
        tbl.verticalHeader().setVisible(False)
        tbl.setFixedHeight(180)
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        tbl.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        tbl.setSelectionMode(QAbstractItemView.NoSelection) 

        header = tbl.horizontalHeader()
        header.setFixedHeight(30)
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        tbl.setColumnWidth(0, 200)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        tbl.setColumnWidth(2, 90)
        header.setSectionResizeMode(3, QHeaderView.Fixed) 
        tbl.setColumnWidth(3, 70)

        tbl.setRowCount(len(mods_data))
        for row, data in enumerate(mods_data):
            ui_ptm, ui_site, ui_mass, i_name, i_res, i_loc = data
            
            item_ptm = QTableWidgetItem(ui_ptm)
            item_ptm.setData(Qt.UserRole, i_name); item_ptm.setData(Qt.UserRole+1, i_res); item_ptm.setData(Qt.UserRole+2, i_loc)
            tbl.setItem(row, 0, item_ptm)
            tbl.setItem(row, 1, QTableWidgetItem(ui_site))
            
            # 💡 [핵심 해결 2] 소수점 3자리까지만 포맷팅하여 깔끔하게 표시
            try:
                formatted_mass = f"{float(ui_mass):+.3f}" # '+' 부호 유지, 소수점 3자리
            except ValueError:
                formatted_mass = ui_mass # 혹시 숫자가 아니면 원본 그대로 표시
                
            item_mass = QTableWidgetItem(formatted_mass)
            item_mass.setTextAlignment(Qt.AlignCenter)
            tbl.setItem(row, 2, item_mass)

            chk_container = QWidget()
            chk_lay = QHBoxLayout(chk_container); chk_lay.setContentsMargins(0,0,0,0); chk_lay.setAlignment(Qt.AlignCenter)
            chk = QCheckBox("")
            chk_lay.addWidget(chk)
            tbl.setCellWidget(row, 3, chk_container)
            
        return tbl

    def clear_rows(self):
        for tbl in (self.table_fixed, self.table_var):
            for row in range(tbl.rowCount()):
                chk = tbl.cellWidget(row, 3).findChild(QCheckBox)
                if chk: chk.setChecked(False)

    def add_mod_row(self, name, residue, location, is_fixed):
        for tbl in (self.table_fixed, self.table_var):
            for row in range(tbl.rowCount()):
                i_name = tbl.item(row, 0).data(Qt.UserRole)
                if i_name and name.lower() in i_name.lower():
                    chk = tbl.cellWidget(row, 3).findChild(QCheckBox)
                    if chk: chk.setChecked(True)

    def get_mods_data(self):
        mods_list = []
        def extract(tbl, is_fixed):
            for row in range(tbl.rowCount()):
                chk = tbl.cellWidget(row, 3).findChild(QCheckBox)
                if chk and chk.isChecked():
                    i_name = tbl.item(row, 0).data(Qt.UserRole)
                    i_res_str = tbl.item(row, 0).data(Qt.UserRole + 1)
                    i_loc = tbl.item(row, 0).data(Qt.UserRole + 2)
                    for res in [r.strip() for r in i_res_str.split(',')]:
                        if res: mods_list.append({"mod_name": i_name, "residue": res, "location": i_loc, "fixed": is_fixed})
        extract(self.table_fixed, True); extract(self.table_var, False)
        return mods_list