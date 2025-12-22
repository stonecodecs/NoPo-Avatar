apt update
apt install -y git wget curl unzip libgl1 libglib2.0-0 libsm6 libxrender1 libxext6
pip install -r requirements.txt
pip install setuptools wheel ninja trimesh
pip install ijson Pillow h5py
pip install datasets ipykernel
pip install --no-build-isolation git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation