# DoseCUDA

**NOT FOR CLINICAL USE**

**License**
This project is licensed under the GPL-2.0 License - see the [LICENSE](LICENSE) file for details.

**To use, please cite [our paper](http://doi.org/10.1002/acm2.70093):**
Bhattacharya M, Reamy C, Li H, Lee J, Hrinivich WT. A Python package for fast GPU‐based proton pencil beam dose calculation. Journal of Applied Clinical Medical Physics. 2025 Apr 11:e70093.

**DoseCUDA** is a Python package enabling GPU-based radiation dose calculation for research, development, and education. The package currently supports photon dose calculation using a collapsed cone convolution superposition algorithm and proton dose calculation using a double-Gaussian pencil beam algorithm. The default photon beam model corresponds to the 6 MV energy of a Varian Truebeam linear accelerator and the default proton beam model corresponds to a Hitachi Probeat synchrotron-based PBS delivery system with 98 discrete energies.

# Quickstart Guide

## Prerequisites
Before installing DoseCUDA, ensure you have the following dependencies installed:
- **Python 3.7+**
- **CMake 3.15+**
- **CUDA Toolkit** (Ensure you have a compatible version for your GPU)
- **Git** (for cloning the repository)

Ensure that your GPU and CUDA drivers are properly set up before proceeding.

## Installation on Linux

Linux builds were tested on Ubuntu 22.04.5 LTS and Debian 13.

1. **Install Python** (if not already installed):
   ```
   sudo apt update
   sudo apt install python3 python3-pip
   ```
   
2. **(Optional but recommended)**: Create a virtual environment to isolate DoseCUDA and its dependencies:
   ```
   python3 -m venv dosecuda-env
   source dosecuda-env/bin/activate
   ```

3. **Clone the DoseCUDA repository**:
   ```
   git clone https://github.com/jhu-som-radiation-oncology/DoseCUDA
   cd DoseCUDA
   ```

4. **Install using pip**:
   ```
   python -m pip install .
   ```
   You may have to [override the host compiler](#additional-notes-linux) if yours is not supported.

5. **Verify the installation**:
   Run the test script to verify that DoseCUDA is installed correctly and working:
   ```
   python tests/test_phantom_impt.py
   ```
   or
   ```
   python tests/test_phantom_imrt.py
   ```

   If everything is installed correctly, you should see the dose calculation output in your terminal and files saved into `./test_phantom_output`.


6. **Deactivating the virtual environment** (if used):
   After using the package, you can deactivate the virtual environment:
   ```
   deactivate
   ```

## Additional Notes (Linux):
- **CUDA version**: Ensure that your CUDA version is compatible with your GPU and the CUDA Toolkit installed on your system.
- **Virtual environments**: Virtual environments help avoid conflicts between dependencies required by DoseCUDA and other Python projects you may have on your machine.
- **NVCC host compiler**: You can supply a host compiler override to pip by setting the environment variable `CUDAHOSTCXX` to the compiler command (e.g. `CUDAHOSTCXX=clang++`). Use this if your system's default is not supported.

## Installation on Windows

#### Warning: This section is out of date

Windows builds have only been tested with Visual Studio 2022 on Windows 11 Enterprise Edition.

1. **Install Python** (if not already installed):
   - Download Python from the [official website](https://www.python.org/downloads/) and install it.
   - Make sure to check the option "Add Python to PATH" during the installation.

2. **Install MSVC**
   - Download Microsoft's Visual Studio installer from the [official website](https://visualstudio.microsoft.com/downloads/).
      - Install the x64/x86 build tools (C++ compiler and runtime libraries)
      - Install the CMake tools
      - (optional) Install Git for Windows

3. **Install the CUDA Toolkit**:
   - Download and install the appropriate version of the CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-toolkit).
   - Make sure the drivers for your GPU are compatible with this version.

4. **Install Git (if not installed through MSVC)**:
   - Download and install Git from the [official website](https://git-scm.com/download/win).
   - During the installation, choose "Git from the command line and also from 3rd-party software".

5. **Create a virtual environment** (optional but recommended):
   Open a command prompt (cmd) and run:
   ```cmd
   python -m venv dosecuda-env
   dosecuda-env\Scripts\activate
   ```

6. **Clone the DoseCUDA repository**:
   ```cmd
   git clone https://github.com/jhu-som-radiation-oncology/DoseCUDA
   cd DoseCUDA
   ```

7. **Install DoseCUDA using pip**:
   From a [**developer command prompt**](#additional-notes-windows):
   ```cmd
   python -m pip install .
   ```

8. **Verify the installation**:
   Run the test script to verify that DoseCUDA is installed correctly and working:
   ```cmd
   python tests\test_phantom_impt.py
   ```
   or
   ```
   python tests\test_phantom_imrt.py
   ```

   If everything is installed correctly, the dose calculation output should appear in the terminal, with files saved into `.\test_phantom_output`.

9. **Deactivate the virtual environment** (if used):
   After using the package, deactivate the virtual environment by typing:
   ```cmd
   deactivate
   ```

## Additional Notes (Windows):
- **PATH configuration**: Ensure Python, CMake, and CUDA are added to your system's PATH during their installations to avoid issues.
- **CUDA compatibility**: Ensure your GPU is compatible with the version of CUDA Toolkit you installed.
- **Developer command prompt**: Using a [developer command prompt](https://learn.microsoft.com/en-us/visualstudio/ide/reference/command-prompt-powershell) ensures that all MSVC build tools are present in the environment's PATH. Use the `x64 Native Tools Command Prompt` for best results.

# Contact

For any questions or support regarding DoseCUDA, please reach out via email:
* **DoseCUDA@gmail.com**

# Developers

* Tom Hrinivich
* Calin Reamy
* Mahasweta Bhattacharya

# Funding Support
* The Commonwealth Fund
