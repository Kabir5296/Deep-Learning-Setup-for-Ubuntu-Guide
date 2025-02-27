#!/bin/bash
echo "This script will install NVIDIA drivers, CUDA 12.8, and cuDNN 9.7.1"

# Remove existing installations of NVIDIA driver
echo "Removing existing NVIDIA installations..."
sudo apt purge nvidia* -y || true
sudo apt remove nvidia-* -y || true
sudo rm /etc/apt/sources.list.d/cuda* || true
sudo apt autoremove -y && sudo apt autoclean -y

## install NVIDIA driver (specific version). Here, I'm installing it for 550
echo "Installing NVIDIA driver 550..."
sudo apt update && sudo apt upgrade -y
sudo apt install gcc
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install libnvidia-common-550 libnvidia-gl-550 nvidia-driver-550 -y

# add the cuda repository pins
echo "Adding CUDA repository pins..."
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

# download cuda and cudnn
echo "Downloading CUDA and cuDNN..."
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb
wget https://developer.download.nvidia.com/compute/cudnn/9.7.1/local_installers/cudnn-local-repo-ubuntu2204-9.7.1_1.0-1_amd64.deb

# dpkg cuda and cudnn
echo "Installing CUDA and cuDNN packages..."
sudo dpkg -i cuda-repo-ubuntu2204-12-8-local_12.8.0-570.86.10-1_amd64.deb -y
sudo cp /var/cuda-repo-ubuntu2204-12-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo dpkg -i cudnn-local-repo-ubuntu2204-9.7.1_1.0-1_amd64.deb -y
sudo cp /var/cudnn-local-repo-ubuntu2204-9.7.1/cudnn-*-keyring.gpg /usr/share/keyrings/

# install cuda and cudnn
echo "Installing CUDA toolkit and cuDNN..."
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-8
sudo apt-get -y install cudnn-cuda-12

# add cuda env path to bashrc
echo "Adding CUDA environment paths to bashrc..."
echo 'export PATH="/usr/local/cuda-12.8/bin${PATH:+:${PATH}}"' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"' >> ~/.bashrc
source ~/.bashrc
sudo ldconfig

# check installation
echo "Checking installation..."
nvcc -V || echo "CUDA compilation tools not found!"
if [ -f /usr/include/x86_64-linux-gnu/cudnn_version_v9.h ]; then
    cat /usr/include/x86_64-linux-gnu/cudnn_version_v9.h
else
    echo "cuDNN version header not found!"
fi

echo "Installation complete. A system reboot is recommended."
echo "Please reboot your system with: sudo reboot, then check with 'nvidia-smi' if the driver is properly installed and found."