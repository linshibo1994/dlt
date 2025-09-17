
# Create conda environment with Python 3.11 (recommended for GPU support)
conda create -n dlt_gpu python=3.11 -y
conda activate dlt_gpu

# Install CUDA toolkit
conda install -c conda-forge cudatoolkit=11.7 -y

# Install TensorFlow GPU
pip install tensorflow-gpu==2.12.0

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# Install other dependencies
pip install -r requirements.txt

# Verify installation
python -c "import tensorflow as tf; print('TF GPU:', len(tf.config.list_physical_devices('GPU')) > 0)"
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"
