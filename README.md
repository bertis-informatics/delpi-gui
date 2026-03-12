# DelPi Search Engine GUI (v0.1.0)

DelPi GUI is a Graphical User Interface designed for the deep learning-based DelPi Search Engine. It provides a streamlined  workflow for Global Proteomics and Phosphoproteomics data analysis, supporting both DDA and DIA methods.

---

##  Key Features

* **Smart Environment Manager (Launcher)**: No manual Python setup required. The integrated launcher automatically detects system Python 3.12 or installs a portable version, setting up a dedicated virtual environment with **PyTorch (CUDA 12.8)** automatically.
* **Intuitive Configuration & Safety**: 
    * Fine-tune digestion rules, mass tolerances, and custom PTMs in the **Advanced Settings**.
    * **Safe-Run Validation**: The 'RUN SEARCH' button remains disabled until all required inputs (Files, FASTA, Output) are correctly configured.
* **Real-time Visualization**: High-performance backend terminal outputs are intercepted and visualized directly through a sleek, modern Teal-themed GUI.
* **Native Thermo '.raw' Support**: Seamlessly handles Thermo Fisher `.raw` files using the integrated `pymsio` library and pre-configured DLLs.

---

##  System Requirements

* **OS**: Windows 10 / 11 (64-bit)
* **GPU**: NVIDIA GPU with CUDA support (Recommended for deep learning performance)
* **Disk Space**: At least **10 GB** of free space (for Python environment, PyTorch, and database indexing).
* **Network**: Active internet connection required during the **initial launch** to download dependency packages.

---

## Installation Guide

1.  Visit the [Releases](../../releases) page and download `DelPi-GUI_Setup_v0.1.0.exe`.
2.  Run the installer to extract the application files to your desired location.
3.  **First Launch**: Open `DelPi-GUI.exe`. 
    * The launcher will initialize the environment and install **PyTorch (CUDA 12.8)** and other dependencies. 
    * *Please wait until the GUI appears automatically.*
4.  **Subsequent Launches**: The launcher skips the installation check and opens the GUI immediately.

---

## Quick Start

1.  **Add Files**: Click `+ Add Files` to import mass spectrometry data (`.raw`, `.mzML`, `.mzML.Gz`).
2.  **Database Setup**: 
    * Select `Generate from FASTA` and load your target protein database.
    * *(Note: 'Existing spectral library' is currently under development.)*
3.  **Set Output**: Choose a destination folder for analysis results, `.yaml` setting and logs.
4.  **Analysis Parameters**: 
    * Select your Acquisition Method (DDA/DIA)
    * Choose Search preset.
5.  **Run Search**: Once all fields are valid, click the **RUN SEARCH** button to start the engine.

---

© 2026 Bertis. All rights reserved.