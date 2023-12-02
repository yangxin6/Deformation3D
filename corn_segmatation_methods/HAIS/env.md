
cuda 11.8

spconv 2

```bash
conda create -n hais python=3.8


#conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -nvidia
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

pip install -r requirements.txt 
conda install -c bioconda google-sparsehash 
conda install libboost
sudo apt-get install libsparsehash-dev


pip install spconv-cu117


cd HAIS/lib/hais_ops
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/include:$CPLUS_INCLUDE_PATH
#export CPLUS_INCLUDE_PATH=/home/keys/anaconda3/envs}/hais/include:$CPLUS_INCLUDE_PATH
#export CPLUS_INCLUDE_PATH=/home/yangxin/anaconda3/envs}/hais/include:$CPLUS_INCLUDE_PATH
python setup.py build_ext develop
```
