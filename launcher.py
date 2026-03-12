import os
import sys
import subprocess
import urllib.request
import zipfile
import tkinter as tk
from tkinter import messagebox

BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
ENV_DIR = os.path.join(BASE_DIR, "delpi_gui_env")
REQ_FILE = os.path.join(BASE_DIR, "requirements.txt")
MAIN_APP = os.path.join(BASE_DIR, "main.py")

def show_error(title, message):
    """Displays a graphical error message and exits the application."""
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(title, message)
    root.destroy()
    sys.exit(1)

def run_cmd(cmd, desc):
    """Executes a system command and handles potential errors."""
    print(f"\n[DelPi Launcher] {desc}...")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        show_error("Installation Error", f"An error occurred during: {desc}\n\nCommand: {' '.join(cmd)}\nExit Code: {e.returncode}")

def get_valid_system_python():
    """Checks if Python 3.12 (64-bit) is installed on the system."""
    try:
        result = subprocess.run(
            ["python", "-c", "import sys, platform; print(f'{sys.version_info.major},{sys.version_info.minor},{platform.architecture()[0]}')"],
            capture_output=True, text=True, check=True
        )
        major, minor, arch = result.stdout.strip().split(',')
        
        # Validation: Requires 64-bit and exactly Python 3.12
        if arch == "64bit" and int(major) == 3 and int(minor) == 12:
            return "python"
        return None
    except Exception:
        return None

def setup_portable_python(env_dir):
    """Downloads and configures a portable Python 3.12 environment if no suitable system Python is found."""
    print("\n[DelPi Launcher] Suitable Python not found. Downloading portable Python 3.12...")
    os.makedirs(env_dir, exist_ok=True)
    
    # 1. Download Python Embeddable package
    py_url = "https://www.python.org/ftp/python/3.12.2/python-3.12.2-embed-amd64.zip"
    zip_path = os.path.join(env_dir, "python.zip")
    print("  -> Downloading Python 3.12.2... (approx. 10MB)")
    urllib.request.urlretrieve(py_url, zip_path)
    
    # 2. Extract the archive
    print("  -> Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(env_dir)
    os.remove(zip_path)
    
    # 3. Modify ._pth file to enable site-packages (Critical for pip functionality)
    pth_file = os.path.join(env_dir, "python312._pth")
    if os.path.exists(pth_file):
        with open(pth_file, 'r') as f:
            lines = f.readlines()
        with open(pth_file, 'w') as f:
            for line in lines:
                if line.strip() == "#import site":
                    f.write("import site\n") 
                else:
                    f.write(line)
                    
    # 4. Download and install pip
    print("  -> Installing package manager (pip)...")
    get_pip_url = "https://bootstrap.pypa.io/get-pip.py"
    get_pip_path = os.path.join(env_dir, "get-pip.py")
    urllib.request.urlretrieve(get_pip_url, get_pip_path)
    
    py_exe = os.path.join(env_dir, "python.exe")
    subprocess.run([py_exe, get_pip_path, "--no-warn-script-location"], check=True)
    os.remove(get_pip_path)
    print("  -> Portable Python environment is ready! ✔️")

def main():
    print("="*50)
    print("   DelPi Search Engine - Initialization   ")
    print("="*50)

    # 1. Build Python environment (venv or Portable)
    if not os.path.exists(ENV_DIR):
        print("\n[DelPi Launcher] Checking system Python version...")
        sys_py = get_valid_system_python()
        if sys_py:
            print("[DelPi Launcher] System Python 3.12 (64-bit) verified. Creating virtual environment (venv).")
            run_cmd([sys_py, "-m", "venv", ENV_DIR], "Creating virtual environment")
        else:
            setup_portable_python(ENV_DIR)
    else:
        print("\n[DelPi Launcher] Python environment already exists.")

    # 2. Set Python executable path (Handling different structures for venv vs portable)
    if os.path.exists(os.path.join(ENV_DIR, "Scripts", "python.exe")):
        python_exe = os.path.join(ENV_DIR, "Scripts", "python.exe") # venv structure
    else:
        python_exe = os.path.join(ENV_DIR, "python.exe") # Portable structure

    # 3. Install packages (Using 'python -m pip' to avoid path conflicts)
    if os.path.exists(REQ_FILE):
        run_cmd([python_exe, "-m", "pip", "install", "--upgrade", "pip", "--no-warn-script-location"], "Upgrading pip")
        
        # Pre-install PyTorch with CUDA 12.8 support
        run_cmd(
            [python_exe, "-m", "pip", "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu128", "--no-warn-script-location"], 
            "Installing PyTorch (CUDA 12.8)"
        )
        
        # Install remaining dependencies from requirements.txt
        run_cmd([python_exe, "-m", "pip", "install", "-r", REQ_FILE, "--no-warn-script-location"], "Installing dependency packages")
    else:
        show_error("Missing File", f"Dependency list file not found:\n{REQ_FILE}")

    # 4. GPU Detection Check
    print("\n[DelPi Launcher] Verifying GPU (CUDA) status...")
    check_script = "import sys, torch; sys.exit(0) if torch.cuda.is_available() else sys.exit(1)"
    try:
        subprocess.run([python_exe, "-c", check_script], check=True)
        print("[DelPi Launcher] GPU (CUDA) recognized successfully! ✔️")
    except subprocess.CalledProcessError:
        show_error("GPU / CUDA Error", "No CUDA-compatible GPU found, or PyTorch was not installed with GPU support.")

    # 5. Launch the GUI Application
    if os.path.exists(MAIN_APP):
        print("\n[DelPi Launcher] Launching DelPi GUI...")
        
        # Inject project directory into PYTHONPATH so 'app' module can be found
        run_env = os.environ.copy()
        run_env["PYTHONPATH"] = BASE_DIR
        
        # Set working directory to project root and execute
        # subprocess.Popen(
        #     [python_exe, MAIN_APP], 
        #     cwd=BASE_DIR, 
        #     env=run_env
        # )
        CREATE_NO_WINDOW = 0x08000000 if os.name == 'nt' else 0
        
        subprocess.Popen(
            [python_exe, MAIN_APP], 
            cwd=BASE_DIR, 
            env=run_env,
            creationflags=CREATE_NO_WINDOW 
        )
    else:
        show_error("Missing File", f"Main execution file not found:\n{MAIN_APP}")

if __name__ == "__main__":
    main()