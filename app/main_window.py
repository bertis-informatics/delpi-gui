import sys
import os
import yaml
import time
import re
import subprocess
import urllib.request

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QSizePolicy, QMessageBox,
                               QFileDialog, QLabel, QButtonGroup, QListWidgetItem, QDialog, QGroupBox, QSizeGrip)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, QProcess, QTimer, Qt, QSize, QPoint
from PySide6.QtGui import QShortcut, QKeySequence, QPixmap, QMouseEvent, QIcon

from app.widgets.modification_widget import AdvancedSettingsDialog


# Utility Functions for Paths
def get_executable_dir():
    """Returns the true directory of the executable or script (Preserved)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        current_dir = os.path.abspath(os.path.dirname(__file__))
        if os.path.basename(current_dir) == "app":
            return os.path.dirname(current_dir)
        return current_dir

def resource_path(relative_path):
    """Returns the absolute path to resources in _MEIPASS or dev folder."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


class ProteomicsApp(QMainWindow):
    APP_VERSION = "0.1.0"

    def __init__(self):
        super().__init__()
        
        # 1. 윈도우 기본 틀 제거 및 배경 투명화
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # 2. UI 파일 로드 및 창 크기 조절 손잡이 설정
        self.load_ui()
        
        # 3. 고급 설정 팝업 초기화
        self.adv_dialog = AdvancedSettingsDialog(self)
        
        # 4. UI 커스터마이징 및 초기화
        self.init_ui_customization()
        self.apply_modern_theme()
        self.setup_interactions()
        self.setup_button_icons()

        self.ui.radio_dia.setChecked(True)
        self.update_analysis_mode()

        # 로그 버튼 숨김 처리
        self.ui.btn_clear_log.hide()
        self.ui.btn_save_log.hide()

        # 5. 백그라운드 프로세스 설정
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.process_finished)

        # 6. 타이틀바 설정
        self.ui.btn_close_window.clicked.connect(self.close)
        self.update_gpu_info()
        self.old_pos = None
    
    def update_gpu_info(self):
        """nvidia-smi를 사용하여 현재 CUDA GPU 이름과 VRAM 용량을 가져옵니다."""
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                encoding='utf-8', startupinfo=self._get_startupinfo()
            )
            parts = result.strip().split(', ')
            if len(parts) >= 2:
                gpu_name = parts[0].replace("NVIDIA GeForce ", "") 
                vram_mb = int(parts[1].replace(" MiB", ""))
                vram_gb = round(vram_mb / 1024)
                
                info_text = f"GPU : {gpu_name} ({vram_gb} GB)"
                self.ui.label_gpu_info.setText(info_text)
        except Exception as e:
            self.ui.label_gpu_info.setText("GPU : CPU / Not Detected")

    def _get_startupinfo(self):
        import os
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            return startupinfo
        return None
    
    # --- 타이틀바 드래그 창 이동 로직 ---
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            if event.position().y() <= 40:
                self.old_pos = event.globalPosition().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self.old_pos is not None:
            delta = event.globalPosition().toPoint() - self.old_pos
            self.move(self.pos() + delta)
            self.old_pos = event.globalPosition().toPoint()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self.old_pos = None

    def load_ui(self):
        loader = QUiLoader()
        path = resource_path(os.path.join("app", "ui", "mainwindow.ui"))

        ui_file = QFile(path)
        if not ui_file.open(QIODevice.ReadOnly): 
            print(f"Cannot open {path}: {ui_file.errorString()}")
            sys.exit(-1)
            
        self.ui = loader.load(ui_file)
        ui_file.close()

        self.setCentralWidget(self.ui.centralWidget())
        
        # 기본 및 최소 사이즈 설정
        self.setMinimumSize(1300, 950)
        self.resize(1300, 950)

        # 창 크기 조절 손잡이
        self.size_grip = QSizeGrip(self)
        self.size_grip.setFixedSize(20, 20) 
        self.size_grip.setStyleSheet("background-color: transparent;") 

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'size_grip'):
            self.size_grip.move(self.width() - self.size_grip.width(), self.height() - self.size_grip.height())

    def init_ui_customization(self):
        # 💡 좌상단 메인 앱 아이콘 설정
        icon_path = resource_path(os.path.join("app", "ui", "delpi_icon_260304.png"))

        self.setWindowTitle(f"DelPi v {self.APP_VERSION}")
        self.ui.label_app_name.setText(f"DelPi v {self.APP_VERSION}")
        
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            scaled_pixmap = pixmap.scaled(30, 30, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.label_app_icon.setPixmap(scaled_pixmap)
        else:
            self.ui.label_app_icon.setText("🔷")
            self.ui.label_app_icon.setStyleSheet("font-size: 20px; background: transparent;")
            self.log_message(">> Icon file not found. Using placeholder emoji (🔷).")

        self.ui.label_app_icon.setAlignment(Qt.AlignCenter)

        close_icon_path = resource_path(os.path.join("app", "ui", "delpi_x_icon.png"))
        
        if os.path.exists(close_icon_path):
            self.ui.btn_close_window.setText("") # 기존 "✕" 텍스트 제거
            
            # 아이콘 로드 및 크기 설정 (14~16px 정도가 가장 예쁩니다)
            close_pixmap = QPixmap(close_icon_path)
            self.ui.btn_close_window.setIcon(QIcon(close_pixmap))
            self.ui.btn_close_window.setIconSize(QSize(45, 45)) 
        else:
            # 이미지 파일이 없을 경우 대비 (안전장치)
            self.ui.btn_close_window.setText("✕")

        # 읽기 전용 텍스트 박스 설정
        self.ui.line_fasta.setReadOnly(True)
        self.ui.line_db.setReadOnly(True)
        self.ui.line_output.setReadOnly(True)

        if hasattr(self.ui, 'label_spinner'):
            self.ui.label_spinner.setText("")
            self.ui.label_spinner.setStyleSheet("font-weight: bold; font-size: 16px; color: #2196F3;")

        self.spinner_timer = QTimer(self)
        self.spinner_timer.timeout.connect(self.update_spinner)
        self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.spinner_idx = 0

        # 라디오 버튼 그룹 설정
        self.bg_acq = QButtonGroup(self)
        self.bg_acq.addButton(self.ui.radio_dia)
        self.bg_acq.addButton(self.ui.radio_dda)

        self.bg_preset = QButtonGroup(self)
        self.bg_preset.addButton(self.ui.radio_global)
        self.bg_preset.addButton(self.ui.radio_phospho)
        self.bg_preset.addButton(self.ui.radio_ubiquitin)
        self.bg_preset.addButton(self.ui.radio_custom)

        self.setup_groupbox_icons()

    def apply_modern_theme(self):
        qss_path = resource_path(os.path.join("app", "ui", "modern_theme.qss"))
        if os.path.exists(qss_path):
            try:
                with open(qss_path, "r", encoding="utf-8") as f:
                    modern_qss = f.read()
                self.setStyleSheet(modern_qss)
            except Exception as e:
                print(f"Error reading theme file: {e}")
        else:
            print(f"Theme file not found: {qss_path}")

    # =========================================================================
    # 💡 [깔끔하게 정리된 그룹박스 아이콘 세팅 함수]
    # UI 파일 내에 이미 존재하는 icon_files 등의 라벨에 이미지 파일만 매핑합니다.
    # =========================================================================
    def setup_groupbox_icons(self):
        icon_size = 32

        # 1. INPUT MS Files 아이콘
        files_icon_path = resource_path(os.path.join("app", "ui", "delpi_files_icon.png"))
        if os.path.exists(files_icon_path):
            pixmap = QPixmap(files_icon_path).scaled(icon_size, icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.icon_files.setAlignment(Qt.AlignCenter)
            self.ui.icon_files.setPixmap(pixmap)
            
        # 2. Database 아이콘
        db_icon_path = resource_path(os.path.join("app", "ui", "delpi_db_icon.png"))
        if os.path.exists(db_icon_path):
            pixmap = QPixmap(db_icon_path).scaled(icon_size, icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.icon_db.setAlignment(Qt.AlignCenter)
            self.ui.icon_db.setPixmap(pixmap)

        # 3. Output 아이콘
        output_icon_path = resource_path(os.path.join("app", "ui", "delpi_folder_icon.png"))
        if os.path.exists(output_icon_path):
            pixmap = QPixmap(output_icon_path).scaled(icon_size, icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.icon_output.setAlignment(Qt.AlignCenter)
            self.ui.icon_output.setPixmap(pixmap)

        # 4. Method 아이콘
        method_icon_path = resource_path(os.path.join("app", "ui", "delpi_method_icon.png"))
        if os.path.exists(method_icon_path):
            pixmap = QPixmap(method_icon_path).scaled(icon_size, icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.icon_method.setAlignment(Qt.AlignCenter)
            self.ui.icon_method.setPixmap(pixmap)

        # 5. Search Status (Log) 아이콘
        log_icon_path = resource_path(os.path.join("app", "ui", "delpi_engine_icon.png"))
        if os.path.exists(log_icon_path):
            pixmap = QPixmap(log_icon_path).scaled(icon_size, icon_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.ui.icon_log.setAlignment(Qt.AlignCenter)
            self.ui.icon_log.setPixmap(pixmap)

    def setup_button_icons(self):
        """파일/폴더 선택 버튼들에 아이콘을 입힙니다."""
        
        # 아이콘 파일 경로 매핑 (파일명은 가지고 계신 파일에 맞춰 수정하세요)
        buttons_map = {
            self.ui.btn_browse_fasta: "delpi_files_icon.png",  # FASTA 파일
            self.ui.btn_browse_db: "delpi_folder_icon.png",    # DB 폴더
            self.ui.btn_browse_output: "delpi_folder_icon.png" # 출력 폴더
        }
        
        for btn, filename in buttons_map.items():
            icon_path = resource_path(os.path.join("app", "ui", filename))
            
            if os.path.exists(icon_path):
                # 버튼 텍스트는 지우고 아이콘 세팅
                btn.setText("") 
                
                # 아이콘 크기는 16x16 또는 18x18이 적당합니다
                pixmap = QPixmap(icon_path).scaled(18, 18, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                btn.setIcon(QIcon(pixmap))
                btn.setIconSize(QSize(18, 18))
            else:
                # 파일 없을 때 대비 (기본 점 유지)
                btn.setText("...")
                
    def update_spinner(self):
        self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_frames)
        if hasattr(self.ui, 'label_spinner'):
            self.ui.label_spinner.setText(self.spinner_frames[self.spinner_idx])

    def setup_interactions(self):
        self.ui.btn_run.clicked.connect(self.run_search)
        self.ui.btn_clear_log.clicked.connect(self.ui.text_log.clear)
        self.ui.btn_save_log.clicked.connect(self.save_log)

        self.ui.btn_add.clicked.connect(self.browse_files)
        self.ui.btn_remove.clicked.connect(self.remove_files)
        self.ui.list_files.itemSelectionChanged.connect(self.update_list_styles)

        self.ui.btn_show_advanced.clicked.connect(self.toggle_advanced)

        self.ui.btn_browse_fasta.clicked.connect(self.browse_fasta)
        self.ui.btn_browse_db.clicked.connect(self.browse_db_dir)
        self.ui.btn_browse_output.clicked.connect(self.browse_output_dir)

        self.ui.radio_db_fasta.toggled.connect(self.toggle_database_mode)
        self.ui.radio_db_lib.toggled.connect(self.toggle_database_mode)
        self.toggle_database_mode()

        self.ui.radio_global.toggled.connect(self.update_analysis_mode)
        self.ui.radio_phospho.toggled.connect(self.update_analysis_mode)
        self.ui.radio_dda.toggled.connect(self.update_analysis_mode)
        self.ui.radio_dia.toggled.connect(self.update_analysis_mode)
        self.ui.radio_ubiquitin.toggled.connect(self.update_analysis_mode)
        self.ui.radio_custom.toggled.connect(self.update_analysis_mode)

        self.ui.radio_dia.toggled.connect(self.update_summary_label)
        self.ui.radio_dda.toggled.connect(self.update_summary_label)
        self.ui.radio_global.toggled.connect(self.update_summary_label)
        self.ui.radio_phospho.toggled.connect(self.update_summary_label)
        self.ui.radio_ubiquitin.toggled.connect(self.update_summary_label)
        self.ui.radio_custom.toggled.connect(self.update_summary_label)

        self.shortcut_cancel = QShortcut(QKeySequence("Ctrl+Q"), self)
        self.shortcut_cancel.activated.connect(self.confirm_cancel)
        self.shortcut_cancel.setEnabled(False) 

        self.update_summary_label()

        # 💡 [Validation] 내용이 바뀔 때마다 조건 검사 함수 호출
        # 1. 파일 리스트 변화 (항목 추가/삭제 시)
        self.ui.list_files.model().rowsInserted.connect(self.validate_run_condition)
        self.ui.list_files.model().rowsRemoved.connect(self.validate_run_condition)

        # 2. 텍스트 박스 변화 (FASTA, Output)
        self.ui.line_fasta.textChanged.connect(self.validate_run_condition)
        self.ui.line_db.textChanged.connect(self.validate_run_condition)
        self.ui.line_output.textChanged.connect(self.validate_run_condition)

        # 3. Database 라디오 버튼 변경
        # (ui 위젯 이름이 다르면 실제 이름으로 맞춰주세요)
        self.ui.radio_db_fasta.toggled.connect(self.on_db_mode_changed)
        self.ui.radio_db_lib.toggled.connect(self.on_db_mode_changed)

        # 프로그램 시작 시 1회 강제 검사 (초기 비활성화 상태 세팅)
        self.validate_run_condition()

    def update_analysis_mode(self):
        is_global = self.ui.radio_global.isChecked()
        is_dda = self.ui.radio_dda.isChecked()

        if getattr(self.ui, 'radio_phospho', None) and self.ui.radio_phospho.isChecked():
            mode_prefix = "Phosphoproteomics"
        elif getattr(self.ui, 'radio_ubiquitin', None) and self.ui.radio_ubiquitin.isChecked():
            mode_prefix = "Ubiquitin"
        elif getattr(self.ui, 'radio_custom', None) and self.ui.radio_custom.isChecked():
            mode_prefix = "Custom"
        else:
            mode_prefix = "Global Proteomics" 

        acq_suffix = "DDA" if is_dda else "DIA"
        target_filename = f"{mode_prefix}-{acq_suffix}.yaml"

        config_data = self.load_config_from_yaml(target_filename)
        self.adv_dialog.clear_rows()

        if not config_data:

            # [공통 기본 세팅]
            config_data = {
                'digest': {
                    'enzyme': 'trypsin',
                    'min_len': 7,
                    'max_len': 30,
                    'max_missed_cleavages': 1,
                    'n_term_methionine_excision': True
                },
                'ms1_mass_tol_in_ppm': 10,
                'ms2_mass_tol_in_ppm': 10 if is_dda else 15, 
                'q_value_cutoff': 0.01,
                'precursor': {
                    'min_charge': 2, 
                    'max_charge': 4,
                    'min_mz': 300.0,
                    'max_mz': 1800.0
                },
                'fragment': {
                    'min_charge': 1,
                    'max_charge': 2,
                    'min_mz': 200.0,
                    'max_mz': 1800.0
                },
                'modification': {
                    'max_mods': 2,
                    'mod_param_set': [
                        {'mod_name': 'Carbamidomethyl', 'residue': 'C', 'location': 'any', 'fixed': True},
                        {'mod_name': 'Oxidation', 'residue': 'M', 'location': 'any', 'fixed': False}
                    ]
                }
            }

            if mode_prefix == "Phosphoproteomics":
                config_data['modification']['mod_param_set'].append(
                    {'mod_name': 'Phospho', 'residue': 'S,T,Y', 'location': 'any', 'fixed': False}
                )
                config_data['modification']['max_mods'] = 3
                
            elif mode_prefix == "Ubiquitin":
                config_data['modification']['mod_param_set'].append(
                    {'mod_name': 'GlyGly', 'residue': 'K', 'location': 'any', 'fixed': False}
                )
                config_data['modification']['max_mods'] = 3

        if config_data:
            try:
                digest = config_data.get('digest', {})
                self.adv_dialog.combo_enzyme.setCurrentText(digest.get('enzyme', 'Trypsin').capitalize()) 
                self.adv_dialog.spin_min_len.setValue(digest.get('min_len', 7))
                self.adv_dialog.spin_max_len.setValue(digest.get('max_len', 30))
                self.adv_dialog.spin_missed.setValue(digest.get('max_missed_cleavages', 1))
                self.adv_dialog.chk_nterm.setChecked(digest.get('n_term_methionine_excision', True))

                self.adv_dialog.spin_ms1_tol.setValue(config_data.get('ms1_mass_tol_in_ppm', 10))
                self.adv_dialog.spin_ms2_tol.setValue(config_data.get('ms2_mass_tol_in_ppm', 10))
                self.adv_dialog.spin_qvalue.setValue(config_data.get('q_value_cutoff', 0.01) * 100.0)

                pre = config_data.get('precursor', {})
                self.adv_dialog.spin_pre_min_z.setValue(pre.get('min_charge', 1))
                self.adv_dialog.spin_pre_max_z.setValue(pre.get('max_charge', 4))
                self.adv_dialog.spin_pre_min_mz.setValue(pre.get('min_mz', 300.0))
                self.adv_dialog.spin_pre_max_mz.setValue(pre.get('max_mz', 1800.0))

                frag = config_data.get('fragment', {})
                self.adv_dialog.spin_frag_min_z.setValue(frag.get('min_charge', 1))
                self.adv_dialog.spin_frag_max_z.setValue(frag.get('max_charge', 2))
                self.adv_dialog.spin_frag_min_mz.setValue(frag.get('min_mz', 200.0))
                self.adv_dialog.spin_frag_max_mz.setValue(frag.get('max_mz', 1800.0))
                
                mod_config = config_data.get('modification', {})
                self.adv_dialog.spin_max_mods.setValue(mod_config.get('max_mods', 2))
                
                saved_mods = mod_config.get('mod_param_set', [])
                if saved_mods:
                    for m in saved_mods:
                        self.adv_dialog.add_mod_row(m.get('mod_name', ''), m.get('residue', ''), m.get('location', ''), m.get('fixed', False))
            except Exception as e:
                self.log_message(f"Error parsing YAML: {e}")

    def update_summary_label(self):
        file_count = self.ui.list_files.count()
        file_text = f"{file_count} RAW files" if file_count != 1 else "1 RAW file"
        
        acq_mode = "DIA" if self.ui.radio_dia.isChecked() else "DDA"
        
        preset = "Standard"
        if getattr(self.ui, 'radio_phospho', None) and self.ui.radio_phospho.isChecked():
            preset = "Phospho"
        elif getattr(self.ui, 'radio_ubiquitin', None) and self.ui.radio_ubiquitin.isChecked():
            preset = "Ubiquitin"
        elif getattr(self.ui, 'radio_custom', None) and self.ui.radio_custom.isChecked():
            preset = "Custom"
            
        summary_text = f"{file_text} · {acq_mode} · {preset} Proteome"
        if hasattr(self.ui, 'label_summary'):
            self.ui.label_summary.setText(summary_text)

    def toggle_advanced(self):
        self.adv_dialog.exec()

    def toggle_database_mode(self):
        is_existing = self.ui.radio_db_lib.isChecked()
        
        self.ui.label_fasta.setVisible(not is_existing)
        self.ui.line_fasta.setVisible(not is_existing)
        self.ui.btn_browse_fasta.setVisible(not is_existing)
        
        # self.ui.label_db_help.setVisible(not is_existing)
        
        if is_existing:
            self.ui.label_db.setText("Spectral library:")
        else:
            self.ui.label_db.setText("Output spectral library:")

    def on_db_mode_changed(self):
        """Database 라디오 버튼 상태에 따라 UI를 업데이트합니다."""
        # 'Existing spectral library'가 선택되었을 때
        if self.ui.radio_db_lib.isChecked():
            # 💡 하단 안내문구를 경고 메시지로 변경 ("A spectral library..." 라벨 객체 이름)
            self.ui.label_db_help.setText("⚠️ This feature is not supported yet.")
            self.ui.label_db_help.setStyleSheet("color: #E53E3E; font-weight: bold;") # 빨간색 강조
        # 'Generate from FASTA'가 선택되었을 때
        else:
            self.ui.label_db_help.setText("A spectral library will be generated automatically.")
            self.ui.label_db_help.setStyleSheet("color: #64748B;") # 원래 회색으로 복구
            
        # 모드가 바뀌었으니 다시 실행 가능 여부 검사
        self.validate_run_condition()

    def validate_run_condition(self):
        """모든 필수 입력이 채워졌는지 검사하여 RUN 버튼 상태를 결정합니다."""
        is_ready = True

        # 1. Input MS file 확인 (리스트에 1개 이상의 파일이 있어야 함)
        if self.ui.list_files.count() == 0:
            is_ready = False

        # 2. Database 설정 확인
        if self.ui.radio_db_lib.isChecked():
            # 기능 미지원 상태이므로 무조건 실행 불가
            is_ready = False
        elif self.ui.radio_db_fasta.isChecked():
            # FASTA 파일 입력란이 비어있으면 실행 불가
            if not self.ui.line_fasta.text().strip() or not self.ui.line_db.text().strip():
                is_ready = False

        # 3. Output folder 확인
        if not self.ui.line_output.text().strip():
            is_ready = False

        # --- 버튼 상태 및 디자인 업데이트 ---
        self.ui.btn_run.setEnabled(is_ready)
        
        if is_ready:
            # 💡 준비 완료: QSS의 원래 세련된 청록색 디자인으로 복구
            self.ui.btn_run.setStyleSheet("") 
            self.ui.btn_run.setText("RUN SEARCH")
        else:
            # 💡 준비 미완료: 클릭 불가능해 보이도록 회색 처리
            self.ui.btn_run.setStyleSheet("""
                QPushButton#btn_run { 
                    background-color: #E2E8F0; 
                    color: #94A3B8; 
                    border: none; 
                    border-radius: 6px; 
                    padding: 15px; 
                    font-weight: bold; 
                    font-size: 18px; 
                }
            """)
            # (선택 사항) 버튼 텍스트도 바꿔줄 수 있습니다.
            # self.ui.btn_run.setText("MISSING SETTINGS")

    def collect_settings(self):
        config = {}
        config['acquisition_method'] = "DDA" if self.ui.radio_dda.isChecked() else "DIA"
        config['ms1_mass_tol_in_ppm'] = self.adv_dialog.spin_ms1_tol.value()
        config['ms2_mass_tol_in_ppm'] = self.adv_dialog.spin_ms2_tol.value()
        config['q_value_cutoff'] = round(self.adv_dialog.spin_qvalue.value() / 100.0, 4)

        config['digest'] = {
            'enzyme': self.adv_dialog.combo_enzyme.currentText().lower(),
            'min_len': self.adv_dialog.spin_min_len.value(),
            'max_len': self.adv_dialog.spin_max_len.value(),
            'max_missed_cleavages': self.adv_dialog.spin_missed.value(),
            'n_term_methionine_excision': self.adv_dialog.chk_nterm.isChecked()
        }

        config['modification'] = {
            'max_mods': self.adv_dialog.spin_max_mods.value(),
            'mod_param_set': self.adv_dialog.get_mods_data()
        }

        config['precursor'] = {
            'min_charge': self.adv_dialog.spin_pre_min_z.value(),
            'max_charge': self.adv_dialog.spin_pre_max_z.value(),
            'min_mz': int(self.adv_dialog.spin_pre_min_mz.value()),
            'max_mz': int(self.adv_dialog.spin_pre_max_mz.value())
        }

        config['fragment'] = {
            'min_charge': self.adv_dialog.spin_frag_min_z.value(),
            'max_charge': self.adv_dialog.spin_frag_max_z.value(),
            'min_mz': int(self.adv_dialog.spin_frag_min_mz.value()),
            'max_mz': int(self.adv_dialog.spin_frag_max_mz.value())
        }
        return config
    
    def collect_paths(self):
        paths = {}
        files = []
        for i in range(self.ui.list_files.count()):
            file_path = self.ui.list_files.item(i).data(Qt.UserRole)
            if file_path:
                files.append(file_path)
                
        paths['input_files'] = files
        paths['fasta_file'] = self.ui.line_fasta.text()
        paths['database_directory'] = self.ui.line_db.text()
        paths['output_directory'] = self.ui.line_output.text()
        return paths
    
    def save_yaml_file(self, config_data, filename):
        settings_dir = os.path.join(get_executable_dir(), "settings")
            
        if not os.path.exists(settings_dir):
            try:
                os.makedirs(settings_dir)
            except Exception as e:
                self.log_message(f"Error creating 'settings' directory: {e}")
                return False

        file_path = os.path.join(settings_dir, filename)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)
            self.log_message(f">> [Config Saved] {file_path}")
            return True
        except Exception as e:
            self.log_message(f"Error saving YAML: {e}")
            return False
        
    def load_config_from_yaml(self, filename):
        config_path = os.path.join(get_executable_dir(), "settings", filename)
    
        if not os.path.exists(config_path):
            return None 
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.log_message(f"Error loading {filename}: {e}")
            return None
        
    def browse_files(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles) 
        name_filter = "Mass Spec Files (*.raw *.mzML *.mzML.gz);;All Files (*)"
        
        filenames, _ = file_dialog.getOpenFileNames(self, "Select Input Files", "", name_filter)

        if filenames:
            current_files = []
            for i in range(self.ui.list_files.count()):
                file_path = self.ui.list_files.item(i).data(Qt.UserRole)
                if file_path:
                    current_files.append(file_path)
            
            count = 0
            for f in filenames:
                f = os.path.normpath(f) 
                
                if f not in current_files:
                    item = QListWidgetItem(self.ui.list_files)
                    item.setData(Qt.UserRole, f) 
                    
                    filename_only = os.path.basename(f)
                    try:
                        size_mb = os.path.getsize(f) / (1024 * 1024)
                        if size_mb >= 1024:
                            size_gb = size_mb / 1024
                            size_str = f"{size_gb:.1f} GB"
                        else:
                            size_str = f"{size_mb:.1f} MB"
                    except:
                        size_str = "Unknown"
                        
                    ext = os.path.splitext(f)[1][1:].upper()
                    if not ext: ext = "FILE"
                    
                    widget = QWidget()
                    widget.setObjectName("bg_widget")
                    widget.setStyleSheet("QWidget#bg_widget { background: transparent; }")
                    
                    layout = QHBoxLayout(widget)
                    layout.setContentsMargins(10, 0, 10, 0)
                    
                    lbl_name = QLabel(filename_only)
                    lbl_name.setObjectName("lbl_name")
                    lbl_name.setStyleSheet("color: #7F8C8D; font-weight: bold; background: transparent;")
                    
                    lbl_info = QLabel(f"{size_str}    {ext}")
                    lbl_info.setObjectName("lbl_info")
                    lbl_info.setStyleSheet("color: #B2BABB; font-weight: bold; font-size: 11px; background: transparent;")
                    lbl_info.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    
                    lbl_name.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
                    lbl_info.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
                    
                    layout.addWidget(lbl_name)
                    layout.addWidget(lbl_info)
                    
                    item.setSizeHint(QSize(100, 36)) 
                    self.ui.list_files.setItemWidget(item, widget)
                    
                    count += 1
                else:
                    self.log_message(f"Skipped duplicate: {os.path.basename(f)}")
            
            if count > 0:
                self.log_message(f"Added {count} files.")
            self.update_summary_label()

    def remove_files(self):
        selected_items = self.ui.list_files.selectedItems()
    
        if not selected_items:
            self.log_message("No files selected to remove.")
            return

        for item in selected_items:
            row = self.ui.list_files.row(item)
            self.ui.list_files.takeItem(row)
            
        self.log_message(f"Removed {len(selected_items)} files.")
        self.update_summary_label()

    def browse_fasta(self):
        fname, _ = QFileDialog.getOpenFileName(
            self, 
            "Select FASTA Database", 
            "", 
            "FASTA Files (*.fasta *.fa);;All Files (*)"
        )
        if fname:
            self.ui.line_fasta.setText(os.path.normpath(fname))

    def browse_db_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Database Directory", 
            ""
        )
        if directory:
            self.ui.line_db.setText(os.path.normpath(directory))

    def browse_output_dir(self):
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Output Directory", 
            ""
        )
        if directory:
            self.ui.line_output.setText(os.path.normpath(directory))

    def save_log(self):
        log_text = self.ui.text_log.toPlainText()
        if not log_text.strip():
            QMessageBox.warning(self, "Warning", "There is no log to save!")
            return

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        default_filename = f"DelPi_Log_{timestamp}.txt"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Log", 
            default_filename, 
            "Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(log_text)
                self.log_message(f">> [Log Saved] {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save log:\n{e}")

    def log_message(self, message):
        self.ui.text_log.append(message)
        scrollbar = self.ui.text_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def run_search(self):
        settings_config = self.collect_settings()
        path_config = self.collect_paths()
        
        if not path_config['input_files']:
            QMessageBox.warning(self, "Warning", "No input files selected!")
            self.log_message("No input files selected!")
            return
        if not path_config['fasta_file']:
            QMessageBox.warning(self, "Warning", "FASTA file not specified!")
            self.log_message("FASTA file not specified!")
            return
        
        output_dir = path_config.get('output_directory', '').strip()
        if not output_dir or not os.path.isdir(output_dir):
            QMessageBox.warning(self, "Warning", "Output Directory not specified!")
            self.log_message("Output Directory not specified!")
            return
        
        has_raw_file = any(f.lower().endswith('.raw') for f in path_config['input_files'])

        if has_raw_file:
            if not self.check_and_download_dlls():
                return

        full_config = { **path_config, **settings_config} 

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        snapshot_filename = f"Run_{timestamp}.yaml"

        full_config_path = os.path.join(output_dir, snapshot_filename)

        try:
            with open(full_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(full_config, f, sort_keys=False)
            self.log_message(f">> Run config snapshot saved: {snapshot_filename}")
        except Exception as e:
            self.log_message(f"Warning: Could not save snapshot: {e}")

        if self.ui.radio_global.isChecked():
            target_filename = "Global Proteomics-{}.yaml".format(settings_config['acquisition_method'])
        else:
            target_filename = "Phosphoproteomics-{}.yaml".format(settings_config['acquisition_method'])

        success = self.save_yaml_file(settings_config, target_filename)

        if not success:
            self.log_message(f"<span style='color:red;'><b>[ERROR]</b> Failed to save configuration file ({target_filename}). Analysis stopped.</span>")
            return 

        self.ui.btn_run.setEnabled(False)
        self.ui.btn_run.setText("RUNNING...\n(Press Ctrl+Q to cancel)")
        self.ui.btn_run.setStyleSheet("background-color: #FFA500; color: white; font-weight: bold; font-size: 16px;")
        self.shortcut_cancel.setEnabled(True)

        self.log_message("\n" + "="*30)
        self.log_message(f"   STARTING DELPI ENGINE   ")
        self.log_message("="*30)

        self.current_run_stage = "First Run"

        self.ui.progressBar_total.setValue(0)
        self.ui.label_total_progress.setText("Total Progress: First Run (0/0)")
        self.ui.progressBar_file.setValue(0)
        self.ui.label_file_progress.setText("Current File Progress: Ready...")

        if hasattr(self.ui, 'label_spinner'):
            self.ui.label_spinner.setStyleSheet("font-weight: bold; font-size: 18px; color: #2196F3;")
        self.spinner_timer.start(100)

        wrapper_script = resource_path("run_engine.py")
        
        python_exe = sys.executable
        args = ["-u", wrapper_script, full_config_path]

        self.log_message(f">> Executing background process...")
        self.process.start(python_exe, args)

    def confirm_cancel(self):
        from PySide6.QtCore import QProcess
        
        if hasattr(self, 'process') and self.process.state() == QProcess.ProcessState.Running:
            reply = QMessageBox.question(
                self, 
                "Cancel Analysis", 
                "Are you sure you want to cancel the running analysis?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.log_message("\n" + "!"*50)
                self.log_message("<span style='color:#f44336;'><b>[CANCEL] User requested cancellation. Stopping the engine...</b></span>")
                self.log_message("!"*50 + "\n")
                self.process.kill() 
            else:
                self.log_message(">> Cancellation aborted. Resuming GUI updates...")

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        stdout = bytes(data).decode("utf8", errors="ignore")
        
        lines = stdout.replace('\r', '\n').split('\n')
        for line in lines:
            self.process_log_line(line, is_stderr=False)

    def handle_stderr(self):
        data = self.process.readAllStandardError()
        stderr = bytes(data).decode("utf8", errors="ignore")
        
        ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
        stderr = ansi_escape.sub('', stderr)
        
        lines = stderr.replace('\r', '\n').split('\n')
        for line in lines:
            self.process_log_line(line, is_stderr=True)

    def process_log_line(self, text, is_stderr=False):
        text = text.replace('\x1b', '').replace('[A', '')
        text = text.strip()
        
        if text.endswith('[A'):
            text = text[:-2].strip()
            
        if not text:
            return

        ignore_keywords = [
            "GPU available:", "TPU available:", "LOCAL_RANK:", 
            "💡 Tip:", "is deprecated", "UserWarning:", "warnings.warn",
            "`Trainer.fit` stopped", "PerformanceWarning:", 
            "map directly to c-types", "inferred_type->mixed", 
            "dtype='object'", "df.to_pandas().to_hdf(", "items->Index",
        ]
        if any(kw in text for kw in ignore_keywords):
            return

        if "%|" in text or "it/s" in text or "Epoch" in text or "Validation" in text or "Predicting" in text:
            percent_match = re.search(r'(\d{1,3})%\|', text)
            if percent_match:
                self.ui.progressBar_file.setValue(int(percent_match.group(1)))
            
            task_match = re.search(r'^(.*?):\s*\d{1,3}%\|', text)
            if task_match:
                task_name = task_match.group(1).replace("[ERROR]", "").replace("[STDERR]", "").strip()
                self.ui.label_file_progress.setText(f"Current File: {task_name}")
            elif "Epoch" in text:
                epoch_match = re.search(r'(Epoch\s+\d+)', text)
                if epoch_match:
                    self.ui.label_file_progress.setText(f"Current File: Training {epoch_match.group(1)}")
            elif "Validation" in text:
                self.ui.label_file_progress.setText(f"Current File: Validation")
            
            return

        if "Second search after transfer learning" in text:
            self.current_run_stage = "Second Run"
            
        match_start = re.search(r'\[(\d+)/(\d+)\] Processing run', text)
        if match_start:
            curr = match_start.group(1)
            tot = match_start.group(2)
            self.ui.label_total_progress.setText(f"Total Progress: {self.current_run_stage} ({curr}/{tot})")

        match_done = re.search(r'\[(\d+)/(\d+)\] Completed processing', text)
        if match_done:
            curr = int(match_done.group(1))
            tot = int(match_done.group(2))
            base_percent = (curr / tot) * 50
            if self.current_run_stage == "First Run":
                self.ui.progressBar_total.setValue(int(base_percent))
            else:
                self.ui.progressBar_total.setValue(int(50 + base_percent))

        if is_stderr and ("Error" in text or "Exception" in text or "Traceback" in text):
            self.log_message(f"<span style='color:red;'><b>[ERROR]</b> {text}</span>")
        else:
            self.log_message(text)
    
    def process_finished(self, exit_code, exit_status):
        self.log_message(f"\n=== ANALYSIS FINISHED (Exit Code: {exit_code}) ===")

        if hasattr(self.ui, 'label_spinner'):
            if exit_status == QProcess.ExitStatus.CrashExit:
                self.log_message(f"\n=== ANALYSIS STOPPED BY USER ===")
                self.ui.label_spinner.setText("⏹")
                self.ui.label_spinner.setStyleSheet("font-weight: bold; font-size: 14px; color: #f44336; padding-right: 5px;")
            
            else:
                if exit_code == 0:
                    self.ui.label_spinner.setText("✔")
                    self.ui.label_spinner.setStyleSheet("font-weight: bold; font-size: 16px; color: #4CAF50; padding-right: 5px;")
                    self.ui.progressBar_total.setValue(100)
                else:
                    self.ui.label_spinner.setText("✖")
                    self.ui.label_spinner.setStyleSheet("font-weight: bold; font-size: 16px; color: #f44336; padding-right: 5px;")

        self.spinner_timer.stop()
        self.ui.btn_run.setEnabled(True)
        self.ui.btn_run.setText("RUN SEARCH")
        self.ui.btn_run.setStyleSheet("")

    def check_and_download_dlls(self):
        dll_names = [
            "ThermoFisher.CommonCore.Data.dll",
            "ThermoFisher.CommonCore.RawFileReader.dll"
        ]
        
        target_dir = os.path.join(get_executable_dir(), "dlls", "thermo_fisher")
        
        all_exist = all(os.path.exists(os.path.join(target_dir, dll)) for dll in dll_names)
        if all_exist:
            return True 

        msg = QMessageBox(self)
        msg.setWindowTitle("Thermo Fisher RawFileReader License")

        license_text = (
            "============================================================\n"
            "   Thermo RawFileReader License Agreement\n"
            "============================================================\n\n"
            "This program will download Thermo Fisher RawFileReader DLLs.\n"
            "These DLLs are Copyright (c) Thermo Fisher Scientific.\n\n"
            "By proceeding, you agree to the Thermo RawFileReader license.\n\n"
            "Full license: https://github.com/thermofisherlsms/RawFileReader/blob/main/License.doc\n\n"
            "Do you want to download these files now?"
        )
        msg.setText(license_text)

        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.button(QMessageBox.Yes).setText("Agree & Download")
        msg.button(QMessageBox.No).setText("Cancel")
        
        if msg.exec() == QMessageBox.Yes:
            os.makedirs(target_dir, exist_ok=True)
            self.log_message("\n>> Starting DLL download. Please wait...")
            
            base_url = "https://github.com/thermofisherlsms/RawFileReader/raw/main/Libs/Net471/"
            
            if hasattr(self.ui, 'label_spinner'):
                self.ui.label_spinner.setStyleSheet("font-weight: bold; font-size: 18px; color: #2196F3;")
            self.spinner_timer.start(100)
            
            try:
                for dll in dll_names:
                    url = base_url + dll
                    dest = os.path.join(target_dir, dll)
                    self.log_message(f"Downloading {dll}...")
                    
                    from PySide6.QtWidgets import QApplication
                    QApplication.processEvents() 
                    
                    urllib.request.urlretrieve(url, dest)
                    
                self.log_message(">> DLLs downloaded successfully!\n")
                self.spinner_timer.stop()
                if hasattr(self.ui, 'label_spinner'):
                    self.ui.label_spinner.setText("✔")
                    self.ui.label_spinner.setStyleSheet("font-weight: bold; font-size: 18px; color: #4CAF50;")
                return True
                
            except Exception as e:
                self.spinner_timer.stop()
                if hasattr(self.ui, 'label_spinner'):
                    self.ui.label_spinner.setText("✖")
                    self.ui.label_spinner.setStyleSheet("font-weight: bold; font-size: 18px; color: red;")
                self.log_message(f"<span style='color:red;'><b>[ERROR]</b> Download failed: {e}</span>")
                return False
        else:
            self.log_message("\n" + "-" * 50)
            self.log_message("<span style='color:#FF5722;'><b>[CANCELLED] Thermo RawFileReader License was declined.</b></span>")
            self.log_message("<span style='color:#FF5722;'>Cannot process '.raw' files without these required DLLs.</span>")
            self.log_message("<span style='color:#FF5722;'>▶ To proceed: Remove '.raw' files from the input list, or click 'RUN SEARCH' again to accept the license.</span>")
            self.log_message("-" * 50 + "\n")
            
            if hasattr(self.ui, 'label_spinner'):
                self.ui.label_spinner.setText("")
            return False
        
    def update_list_styles(self):
        for i in range(self.ui.list_files.count()):
            item = self.ui.list_files.item(i)
            widget = self.ui.list_files.itemWidget(item)
            
            if widget:
                lbl_name = widget.findChild(QLabel, "lbl_name")
                lbl_info = widget.findChild(QLabel, "lbl_info")
                
                if item.isSelected():
                    if lbl_name: lbl_name.setStyleSheet("background: transparent; color: #FFFFFF; font-weight: bold;")
                    if lbl_info: lbl_info.setStyleSheet("background: transparent; color: #E0E4E8; font-weight: bold; font-size: 11px;")
                else:
                    if lbl_name: lbl_name.setStyleSheet("background: transparent; color: #7F8C8D; font-weight: bold;")
                    if lbl_info: lbl_info.setStyleSheet("background: transparent; color: #B2BABB; font-weight: bold; font-size: 11px;")