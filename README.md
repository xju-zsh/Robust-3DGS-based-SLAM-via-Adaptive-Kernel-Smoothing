<!-- PROJECT LOGO -->
<p align="center">
  <h1 align="center">Robust 3DGS-based SLAM via Adaptive Kernel Smoothing</h1>
  <p align="center">
    Shouhe Zhang, Dayong Ren∗, WEN JIE LI, Piaopiao Yu, Sensen Song∗,Kaikai Shao5, and Yurong Qian
  </p>
### Installation

Please follow the instructions below to install the repo and dependencies.

```bash
git clone https://github.com/xju-zsh/Robust-3DGS-based-SLAM-via-Adaptive-Kernel-Smoothing.git
cd Robust-3DGS-based-SLAM-via-Adaptive-Kernel-Smoothing

```bash
# Create conda environment
conda create -n rslam python=3.10
conda activate rslam

# Install the requirements
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

# Build extension 
cd diff-gaussian-rasterization
python setup.py install

cd pytorch3D
pip install iopath-0.1.5.tar.gz
pip install ./pytorch3d-0.7.8+pt2.3.1cu118-cp310-cp310-linux_x86_64.whl
