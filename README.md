# DelPi Search Engine GUI

DelPi GUI is a user-friendly Graphical User Interface designed for the deep learning-based DelPi Search Engine. It streamlines the workflow for Global Proteomics and Phosphoproteomics data analysis, supporting both DDA (Data-Dependent Acquisition) and DIA (Data-Independent Acquisition) methods.

## Key Features

* **Intuitive Configuration**: Easily configure search parameters, including digestion rules, mass tolerances, and custom peptide modifications.
* **Real-time Progress Tracking**: The PyTorch-based search engine runs as a seamless background process. Terminal outputs and progress bars (tqdm) are intercepted and visualized directly within the GUI.
* **Native Thermo '.raw' Support**: Automatically detects Thermo Fisher `.raw` files and prompts the user to download the required `ThermoFisher.CommonCore` DLLs directly from the official repository.
* **Smart & Lightweight Installer**: The provided setup file is highly lightweight. It automatically downloads and extracts the heavy deep learning dependencies (`torch.7z`) during the installation process to save bandwidth and prevent corruption.

## System Requirements

* **OS**: Windows 10 / 11 (64-bit)
* **Architecture**: x64 compatible
* **Disk Space**: At least 5 GB of free space (required for PyTorch extraction and temporary download caches)
* **Network**: Active internet connection required during installation and for the initial `.raw` file DLL setup.

## Installation Guide

1. Navigate to the [Releases](../../releases) page.
2. Download the latest installer: `DelPi-GUI_Setup_vX.X.X.exe`.
3. Run the installer.
   * *Note: The installer will automatically download the required ~2GB `torch.7z` archive from the GitHub assets. Please ensure your internet connection is stable.*
4. Launch "DelPi Search Engine" from your Desktop or Start Menu.

## Quick Start

1. **Load Files**: Click `Add` to import your mass spectrometry data files (`.raw`, `.mzML`, `.mzML.gz`).
2. **Select FASTA**: Browse and select your target FASTA database.
3. **Set Output Directory**: Choose where the analysis results and YAML configurations will be saved.
4. **Configure Parameters**: 
   * Select the Analysis Type (Global Proteomics or Phosphoproteomics).
   * Select the Acquisition Method (DDA or DIA).
   * Expand the `Show Advanced >>` menu to fine-tune modifications and mass tolerances.
5. **Run**: Click `RUN SEARCH`. The configuration will be saved automatically, and the engine will begin processing.