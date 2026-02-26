import sys
import os
import yaml
import time
import re

import urllib.request

from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QSizePolicy, QMessageBox,
                               QFileDialog, QLabel)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtCore import QFile, QIODevice, QProcess, QTimer

from app.widgets.modification_widget import ModificationWidget


# Utility Functions for Paths (Crucial for PyInstaller compatibility)
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
        def __init__(self):
            super().__init__()
            self.load_ui()
            self.init_ui_customization()
            
            self.align_buttons_right()
            self.setup_interactions()

            # Initialize with Global Proteomics mode
            self.update_analysis_mode()
            
            self.toggle_advanced() 

            self.process = QProcess(self)
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stderr)
            self.process.finished.connect(self.process_finished)

        def load_ui(self):
            loader = QUiLoader()
            # Ensure UI loads correctly both in Dev and PyInstaller Build
            path = resource_path(os.path.join("app", "ui", "mainwindow.ui"))

            ui_file = QFile(path)
            if not ui_file.open(QIODevice.ReadOnly): 
                print(f"Cannot open {path}: {ui_file.errorString()}")
                sys.exit(-1)
            self.ui = loader.load(ui_file)
            ui_file.close()

            self.setCentralWidget(self.ui.centralWidget())
            self.resize(1300, 900)

        def init_ui_customization(self):
            self.mod_widget = ModificationWidget()
            if hasattr(self.ui, 'verticalLayout_mods'):
                self.ui.verticalLayout_mods.addWidget(self.mod_widget)
            
            self.ui.line_fasta.setReadOnly(True)
            self.ui.line_db.setReadOnly(True)
            self.ui.line_output.setReadOnly(True)

            # Initially hidden
            self.ui.btn_add_custom.setVisible(False)
            self.ui.progressBar_step.setVisible(False)
            self.ui.label_step.setVisible(False)

            if hasattr(self.ui, 'label_spinner'):
                self.ui.label_spinner.setText("")
                self.ui.label_spinner.setStyleSheet("font-weight: bold; font-size: 16px; color: #2196F3;")

            self.spinner_timer = QTimer(self)
            self.spinner_timer.timeout.connect(self.update_spinner)
            self.spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            self.spinner_idx = 0
            
        def update_spinner(self):
            self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner_frames)
            if hasattr(self.ui, 'label_spinner'):
                self.ui.label_spinner.setText(self.spinner_frames[self.spinner_idx])

        def align_buttons_right(self):
            container = self.ui.widget_action_container
            container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            
            self.ui.btn_clear_log.setParent(self.ui)
            self.ui.btn_save_log.setParent(self.ui)
            self.ui.btn_run.setParent(self.ui)
            
            old_layout = container.layout()
            if old_layout: QWidget().setLayout(old_layout)

            h_layout = QHBoxLayout(container)
            h_layout.setContentsMargins(0, 0, 0, 0)
            h_layout.setSpacing(10)

            h_layout.addStretch(1)

            v_layout = QVBoxLayout()
            v_layout.setSpacing(5)
            
            for btn in [self.ui.btn_clear_log, self.ui.btn_save_log]:
                btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
                btn.setMaximumWidth(300)
                btn.setFixedHeight(40)
                v_layout.addWidget(btn)
            
            h_layout.addLayout(v_layout, 10)

            self.ui.btn_run.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.ui.btn_run.setMaximumWidth(300)
            self.ui.btn_run.setFixedHeight(85)
            h_layout.addWidget(self.ui.btn_run, 10)

            self.ui.btn_clear_log.show()
            self.ui.btn_save_log.show()
            self.ui.btn_run.show()

        def setup_interactions(self):
            self.ui.btn_run.clicked.connect(self.run_search)
            self.ui.btn_clear_log.clicked.connect(self.ui.text_log.clear)
            self.ui.btn_save_log.clicked.connect(self.save_log)

            self.ui.btn_add.clicked.connect(self.browse_files)
            self.ui.btn_remove.clicked.connect(self.remove_files)

            self.ui.btn_show_advanced.clicked.connect(self.toggle_advanced)

            self.ui.btn_browse_fasta.clicked.connect(self.browse_fasta)
            self.ui.btn_browse_db.clicked.connect(self.browse_db_dir)
            self.ui.btn_browse_output.clicked.connect(self.browse_output_dir)

            self.ui.radio_global.toggled.connect(self.update_analysis_mode)
            self.ui.radio_phospho.toggled.connect(self.update_analysis_mode)

            self.ui.radio_dda.toggled.connect(self.update_analysis_mode)
            self.ui.radio_dia.toggled.connect(self.update_analysis_mode)

        def update_analysis_mode(self):
            is_global = self.ui.radio_global.isChecked()
            is_dda = self.ui.radio_dda.isChecked()

            mode_prefix = "Global Proteomics" if is_global else "Phosphoproteomics"
            acq_suffix = "DDA" if is_dda else "DIA"

            target_filename = f"{mode_prefix}-{acq_suffix}.yaml"

            # [Digest Settings]
            default_enzyme = "Trypsin"
            default_min_len = 7
            default_max_len = 30
            default_nterm_exc = True

            # Default values
            default_missed = 1
            default_max_mods = 2
            default_mods = []

            common_mods = [
                ("Carbamidomethyl", "C", "Anywhere", True),
                ("Oxidation", "M", "Anywhere", False),
                ("Acetyl", "*", "Protein N-term", False)
            ]

            phospho_mods = [
                ("Phospho", "S", "Anywhere", False),
                ("Phospho", "T", "Anywhere", False),
                ("Phospho", "Y", "Anywhere", False)
            ]

            if is_global:
                # === Global Proteomics ===
                default_missed = 1
                default_max_mods = 2
                default_mods = list(common_mods)

                if not is_dda: # DIA
                    default_missed = 1 
                    default_max_mods = 2
            else:
                # === Phosphoproteomics ===
                default_missed = 2
                default_max_mods = 3
                default_mods = list(common_mods) + list(phospho_mods)
                
                if not is_dda: # DIA
                    default_missed = 2
                    default_max_mods = 3

            # [Tolerances]
            default_ms1_tol = 10
            default_ms2_tol = 10
            default_qvalue = 0.01

            # [Precursor Ranges]
            default_pre_min_z = 1
            default_pre_max_z = 4
            default_pre_min_mz = 300.0
            default_pre_max_mz = 1800.0

            # [Fragment Ranges]
            default_frag_min_z = 1
            default_frag_max_z = 2
            default_frag_min_mz = 200.0
            default_frag_max_mz = 1800.0

            config_data = self.load_config_from_yaml(target_filename)

            self.mod_widget.clear_rows()

            if config_data:
                try:
                    # Digest
                    digest = config_data.get('digest', {})
                    enzyme_val = digest.get('enzyme', default_enzyme)
                    self.ui.combo_enzyme.setCurrentText(enzyme_val.capitalize()) 
                    self.ui.spin_min_len.setValue(digest.get('min_len', default_min_len))
                    self.ui.spin_max_len.setValue(digest.get('max_len', default_max_len))
                    self.ui.spin_missed.setValue(digest.get('max_missed_cleavages', default_missed))
                    self.ui.chk_nterm.setChecked(digest.get('n_term_methionine_excision', default_nterm_exc))

                    # Tolerance
                    self.ui.spin_ms1_tol.setValue(config_data.get('ms1_mass_tol_in_ppm', default_ms1_tol))
                    self.ui.spin_ms2_tol.setValue(config_data.get('ms2_mass_tol_in_ppm', default_ms2_tol))
                    self.ui.spin_qvalue.setValue(config_data.get('q_value_cutoff', default_qvalue))

                    # Precursor 
                    pre = config_data.get('precursor', {})
                    self.ui.spin_pre_min_z.setValue(pre.get('min_charge', default_pre_min_z))
                    self.ui.spin_pre_max_z.setValue(pre.get('max_charge', default_pre_max_z))
                    self.ui.spin_pre_min_mz.setValue(pre.get('min_mz', default_pre_min_mz))
                    self.ui.spin_pre_max_mz.setValue(pre.get('max_mz', default_pre_max_mz))

                    # Fragment
                    frag = config_data.get('fragment', {})
                    self.ui.spin_frag_min_z.setValue(frag.get('min_charge', default_frag_min_z))
                    self.ui.spin_frag_max_z.setValue(frag.get('max_charge', default_frag_max_z))
                    self.ui.spin_frag_min_mz.setValue(frag.get('min_mz', default_frag_min_mz))
                    self.ui.spin_frag_max_mz.setValue(frag.get('max_mz', default_frag_max_mz))
                    
                    # Modification
                    mod_config = config_data.get('modification', {})
                    self.mod_widget.spin_max_mods.setValue(mod_config.get('max_mods', default_max_mods))
                    
                    # Mod Rows
                    saved_mods = mod_config.get('mod_param_set', [])
                    if saved_mods:
                        for mod_dict in saved_mods:
                            name = mod_dict.get('mod_name', '')
                            res = mod_dict.get('residue', '')
                            loc = mod_dict.get('location', '')

                            if loc == 'protein_n_term':
                                loc = 'Protein N-term'
                            elif loc == 'anywhere':
                                loc = 'Anywhere'

                            fixed = mod_dict.get('fixed', False)
                    
                            self.mod_widget.add_mod_row(name, res, loc, fixed)
                    else:
                        for mod in default_mods:
                            self.mod_widget.add_mod_row(*mod)

                except Exception as e:
                    self.log_message(f"Error parsing YAML: {e}. Reverting to defaults.")
                    config_data = None
            
            if not config_data:
                # Digest
                self.ui.combo_enzyme.setCurrentText(default_enzyme)
                self.ui.spin_min_len.setValue(default_min_len)
                self.ui.spin_max_len.setValue(default_max_len)
                self.ui.spin_missed.setValue(default_missed)
                self.ui.chk_nterm.setChecked(default_nterm_exc)

                # Tolerance
                self.ui.spin_ms1_tol.setValue(default_ms1_tol)
                self.ui.spin_ms2_tol.setValue(default_ms2_tol)
                self.ui.spin_qvalue.setValue(default_qvalue)

                # Ranges
                self.ui.spin_pre_min_z.setValue(default_pre_min_z)
                self.ui.spin_pre_max_z.setValue(default_pre_max_z)
                self.ui.spin_pre_min_mz.setValue(default_pre_min_mz)
                self.ui.spin_pre_max_mz.setValue(default_pre_max_mz)
                
                self.ui.spin_frag_min_z.setValue(default_frag_min_z)
                self.ui.spin_frag_max_z.setValue(default_frag_max_z)
                self.ui.spin_frag_min_mz.setValue(default_frag_min_mz)
                self.ui.spin_frag_max_mz.setValue(default_frag_max_mz)

                # Mods
                self.mod_widget.spin_max_mods.setValue(default_max_mods)
                for mod in default_mods:
                    self.mod_widget.add_mod_row(*mod)

        def toggle_advanced(self):
            is_checked = self.ui.btn_show_advanced.isChecked()
            self.ui.group_advanced.setVisible(is_checked)
            if is_checked:
                self.ui.btn_show_advanced.setText("Hide Advanced <<")
            else:
                self.ui.btn_show_advanced.setText("Show Advanced >>")

        def collect_settings(self):
            config = {}
            config['acquisition_method'] = "DDA" if self.ui.radio_dda.isChecked() else "DIA"

            # Tolerances & Cutoff
            config['ms1_mass_tol_in_ppm'] = self.ui.spin_ms1_tol.value()
            config['ms2_mass_tol_in_ppm'] = self.ui.spin_ms2_tol.value()
            config['q_value_cutoff'] = self.ui.spin_qvalue.value()

            # Digest
            config['digest'] = {
                'enzyme': self.ui.combo_enzyme.currentText().lower(),
                'min_len': self.ui.spin_min_len.value(),
                'max_len': self.ui.spin_max_len.value(),
                'max_missed_cleavages': self.ui.spin_missed.value(),
                'n_term_methionine_excision': self.ui.chk_nterm.isChecked()
            }

            # Modifications
            config['modification'] = {
                'max_mods': self.mod_widget.spin_max_mods.value(),
                'mod_param_set': self.mod_widget.get_data()
            }

            # Precursor
            config['precursor'] = {
                'min_charge': self.ui.spin_pre_min_z.value(),
                'max_charge': self.ui.spin_pre_max_z.value(),
                'min_mz': int(self.ui.spin_pre_min_mz.value()),
                'max_mz': int(self.ui.spin_pre_max_mz.value())
            }

            # Fragment
            config['fragment'] = {
                'min_charge': self.ui.spin_frag_min_z.value(),
                'max_charge': self.ui.spin_frag_max_z.value(),
                'min_mz': int(self.ui.spin_frag_min_mz.value()),
                'max_mz': int(self.ui.spin_frag_max_mz.value())
            }
            
            return config
        
        def collect_paths(self):
            paths = {}
            # Fetch file list from the list widget
            files = [self.ui.list_files.item(i).text() for i in range(self.ui.list_files.count())]
            paths['input_files'] = files
            paths['fasta_file'] = self.ui.line_fasta.text()
            paths['database_directory'] = self.ui.line_db.text()
            paths['output_directory'] = self.ui.line_output.text()
            return paths
        
        def save_yaml_file(self, config_data, filename):
            # Always save in the real executable directory (not _MEIPASS)
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
            file_dialog.setFileMode(QFileDialog.ExistingFiles) # Allow multiple selections
            
            name_filter = "Mass Spec Files (*.raw *.mzML *.mzML.gz);;All Files (*)"
            
            filenames, _ = file_dialog.getOpenFileNames(
                self, 
                "Select Input Files", 
                "", 
                name_filter
            )

            if filenames:
                # Check current list to prevent duplicates
                current_files = [self.ui.list_files.item(i).text() for i in range(self.ui.list_files.count())]
                
                count = 0
                for f in filenames:
                    f = os.path.normpath(f) 
                    
                    if f not in current_files:
                        self.ui.list_files.addItem(f)
                        count += 1
                    else:
                        self.log_message(f"Skipped duplicate: {os.path.basename(f)}")
                
                if count > 0:
                    self.log_message(f"Added {count} files.")

        def remove_files(self):
            selected_items = self.ui.list_files.selectedItems()
        
            if not selected_items:
                self.log_message("No files selected to remove.")
                return

            for item in selected_items:
                # Find row index and remove item
                row = self.ui.list_files.row(item)
                self.ui.list_files.takeItem(row)
                
            self.log_message(f"Removed {len(selected_items)} files.")

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
            
            # Open file save dialog
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

            # Trigger DLL check and download only if '.raw' files are present.
            if has_raw_file:
                if not self.check_and_download_dlls():
                    return

            # Merge configs
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

            # Save file with the currently selected settings name
            if self.ui.radio_global.isChecked():
                target_filename = "Global Proteomics-{}.yaml".format(settings_config['acquisition_method'])
            else:
                target_filename = "Phosphoproteomics-{}.yaml".format(settings_config['acquisition_method'])

            success = self.save_yaml_file(settings_config, target_filename)

            if not success:
                self.log_message(f"<span style='color:red;'><b>[ERROR]</b> Failed to save configuration file ({target_filename}). Analysis stopped.</span>")
                return # Stop execution if saving fails

            # Disable button to prevent multiple executions
            self.ui.btn_run.setEnabled(False)
            self.ui.btn_run.setText("RUNNING...")
            self.ui.btn_run.setStyleSheet("background-color: #FFA500; color: white; font-weight: bold; font-size: 18px;")

            self.log_message("\n" + "="*30)
            self.log_message(f"   STARTING DELPI ENGINE   ")
            self.log_message("="*30)

            self.current_run_stage = "First Run"

            self.ui.progressBar_total.setValue(0)
            self.ui.label_total.setText("Total Progress: First Run (0/0)")
            self.ui.progressBar_file.setValue(0)
            self.ui.label_file.setText("Current File Progress: Ready...")

            if hasattr(self.ui, 'label_spinner'):
                self.ui.label_spinner.setStyleSheet("font-weight: bold; font-size: 18px; color: #2196F3;")
            self.spinner_timer.start(100)

            # Find the bundled run_engine.py using resource_path
            wrapper_script = resource_path("run_engine.py")
            
            python_exe = sys.executable
            args = ["-u", wrapper_script, full_config_path]

            self.log_message(f">> Executing background process...")
            self.process.start(python_exe, args)

        def handle_stdout(self):
            data = self.process.readAllStandardOutput()
            stdout = bytes(data).decode("utf8", errors="ignore")
            
            # Convert \r to \n to prevent tqdm logs from collapsing into a single line
            lines = stdout.replace('\r', '\n').split('\n')
            for line in lines:
                self.process_log_line(line, is_stderr=False)

        def handle_stderr(self):
            data = self.process.readAllStandardError()
            stderr = bytes(data).decode("utf8", errors="ignore")
            
            # Remove terminal color/control ANSI escape codes
            ansi_escape = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
            stderr = ansi_escape.sub('', stderr)
            
            # Convert \r to \n for proper splitting
            lines = stderr.replace('\r', '\n').split('\n')
            for line in lines:
                self.process_log_line(line, is_stderr=True)

        def process_log_line(self, text, is_stderr=False):
            text = text.replace('\x1b', '').replace('[A', '')
            text = text.strip()
            
            # Remove artifact '[A' left by PyTorch Lightning progress bars
            if text.endswith('[A'):
                text = text[:-2].strip()
                
            if not text:
                return

            # Hide messy warnings and informational messages
            ignore_keywords = [
                "GPU available:", "TPU available:", "LOCAL_RANK:", 
                "💡 Tip:", "is deprecated", "UserWarning:", "warnings.warn",
                "`Trainer.fit` stopped", "PerformanceWarning:", 
                "map directly to c-types", "inferred_type->mixed", 
                "dtype='object'", "df.to_pandas().to_hdf(", "items->Index",
                # "Sanity Checking", "Scoring test dataset"
            ]
            if any(kw in text for kw in ignore_keywords):
                return

            # Detect tqdm progress bars -> Block console output and update UI progress bar instead
            if "%|" in text or "it/s" in text or "Epoch" in text or "Validation" in text or "Predicting" in text:
                # Extract percentage digits
                percent_match = re.search(r'(\d{1,3})%\|', text)
                if percent_match:
                    self.ui.progressBar_file.setValue(int(percent_match.group(1)))
                
                # Update label (extract task name)
                task_match = re.search(r'^(.*?):\s*\d{1,3}%\|', text)
                if task_match:
                    task_name = task_match.group(1).replace("[ERROR]", "").replace("[STDERR]", "").strip()
                    self.ui.label_file.setText(f"Current Task: {task_name}")
                elif "Epoch" in text:
                    epoch_match = re.search(r'(Epoch\s+\d+)', text)
                    if epoch_match:
                        self.ui.label_file.setText(f"Current Task: Training {epoch_match.group(1)}")
                elif "Validation" in text:
                    self.ui.label_file.setText(f"Current Task: Validation")
                
                # [Core] Prevent progress strings from printing to the text log and return
                return

            # Detect DelPi analysis cycles (Update Total Progress)
            if "Second search after transfer learning" in text:
                self.current_run_stage = "Second Run"
                
            match_start = re.search(r'\[(\d+)/(\d+)\] Processing run', text)
            if match_start:
                curr = match_start.group(1)
                tot = match_start.group(2)
                self.ui.label_total.setText(f"Total Progress: {self.current_run_stage} ({curr}/{tot})")

            match_done = re.search(r'\[(\d+)/(\d+)\] Completed processing', text)
            if match_done:
                curr = int(match_done.group(1))
                tot = int(match_done.group(2))
                base_percent = (curr / tot) * 50
                if self.current_run_stage == "First Run":
                    self.ui.progressBar_total.setValue(int(base_percent))
                else:
                    self.ui.progressBar_total.setValue(int(50 + base_percent))

            # Print only the 'real logs' that pass the filters
            if is_stderr and ("Error" in text or "Exception" in text or "Traceback" in text):
                self.log_message(f"<span style='color:red;'><b>[ERROR]</b> {text}</span>")
            else:
                self.log_message(text)
        
        def process_finished(self, exit_code, exit_status):
            self.log_message(f"\n=== ANALYSIS FINISHED (Exit Code: {exit_code}) ===")

            # Stop spinner and show result icon when analysis finishes
            self.spinner_timer.stop()
            
            if hasattr(self.ui, 'label_spinner'):
                if exit_code == 0:
                    self.ui.label_spinner.setText("✔")
                    self.ui.label_spinner.setStyleSheet("font-weight: bold; font-size: 18px; color: #4CAF50;")
                else:
                    self.ui.label_spinner.setText("✖")
                    self.ui.label_spinner.setStyleSheet("font-weight: bold; font-size: 18px; color: red;")

            if exit_code == 0:
                self.ui.progressBar_total.setValue(100)
            
            self.ui.btn_run.setEnabled(True)
            self.ui.btn_run.setText("RUN SEARCH")
            self.ui.btn_run.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; font-size: 18px;")

        def check_and_download_dlls(self):
            """Check for required DLL files before execution, and download them after license agreement if missing."""
            dll_names = [
                "ThermoFisher.CommonCore.Data.dll",
                "ThermoFisher.CommonCore.RawFileReader.dll"
            ]
            
            # Store inside the executable's directory (the most reliable path).
            # Path: Delpi-GUI/dlls/thermo_fisher
            target_dir = os.path.join(get_executable_dir(), "dlls", "thermo_fisher")
            
            # Check if all required files exist
            all_exist = all(os.path.exists(os.path.join(target_dir, dll)) for dll in dll_names)
            if all_exist:
                return True # Pass if all files exist

            # Show license agreement popup if files are missing
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
            
            # If user accepts the agreement
            if msg.exec() == QMessageBox.Yes:
                os.makedirs(target_dir, exist_ok=True)
                self.log_message("\n>> Starting DLL download. Please wait...")
                
                # Correct Github Raw URL path (Libs/Net471)
                base_url = "https://github.com/thermofisherlsms/RawFileReader/raw/main/Libs/Net471/"
                
                # Start spinner
                if hasattr(self.ui, 'label_spinner'):
                    self.ui.label_spinner.setStyleSheet("font-weight: bold; font-size: 18px; color: #2196F3;")
                self.spinner_timer.start(100)
                
                try:
                    for dll in dll_names:
                        url = base_url + dll
                        dest = os.path.join(target_dir, dll)
                        self.log_message(f"Downloading {dll}...")
                        
                        # Prevent GUI freezing
                        from PySide6.QtWidgets import QApplication
                        QApplication.processEvents() 
                        
                        # Execute file download
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