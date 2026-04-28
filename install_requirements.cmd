conda install -c conda-forge openjdk=17 -y
conda install -c conda-forge jpype1 -y
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install gymnasium numpy scipy scikit-image scikit-learn trimesh mph
pip install stable-baselines3[extra]
