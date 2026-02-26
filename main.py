import sys
import os

# =====================================================================
# [Required at top] Prevent multiprocessing conflicts with PyInstaller console=False
# A dummy writer (black hole) to prevent background workers from crashing 
# due to missing standard output (None).
# =====================================================================
class NullWriter:
    def write(self, text): pass
    def flush(self): pass

if sys.stdout is None:
    sys.stdout = NullWriter()
if sys.stderr is None:
    sys.stderr = NullWriter()

import traceback
import multiprocessing
import runpy
from PySide6.QtWidgets import QApplication

from app.main_window import ProteomicsApp

def log_error(msg):
    with open("error_log.txt", "w", encoding="utf-8") as f:
        f.write(msg)

def resource_path(relative_path):
    # Function to find the absolute path within the temporary folder after a PyInstaller build
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

if __name__ == "__main__":
    # Prevent infinite recursive spawning when using multiprocessing in a Windows environment
    multiprocessing.freeze_support()

    # Logic to intercept and execute the background engine
    engine_script_idx = -1
    for i, arg in enumerate(sys.argv):
        if "run_engine.py" in arg:
            engine_script_idx = i
            break

    if engine_script_idx != -1:
        try:
            # Remove preceding unnecessary options (like -u) and create a new list starting from the engine script
            script_path = sys.argv[engine_script_idx]
            sys.argv = sys.argv[engine_script_idx:] 
            
            # Correct to absolute path
            if not os.path.isabs(script_path):
                script_path = resource_path(script_path)
            
            # Silently execute the background engine without the GUI
            runpy.run_path(script_path, run_name="__main__")
            sys.exit(0)
            
        except Exception:
            error_msg = traceback.format_exc()
            log_error("--- [Background Engine Error] ---\n" + error_msg)
            sys.exit(1)

    try:
        app = QApplication(sys.argv)
        window = ProteomicsApp()
        window.show()
        sys.exit(app.exec())

    except Exception:
        error_msg = traceback.format_exc()
        log_error("--- [GUI Fatal Error] ---\n" + error_msg)
        sys.exit(1)