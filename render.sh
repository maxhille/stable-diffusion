#!/usr/bin/env sh

python3 -m venv venv
source venv/bin/activate
#pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# https://stackoverflow.com/a/73582465
export HSA_OVERRIDE_GFX_VERSION=10.3.0
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/rocm5.2
pip install git+https://github.com/openai/CLIP.git@d50d76daa670286dd6cacf3bcd80b5e4823fc8e1
pip install einops
pip install pytorch_lightning
pip install diffusers
pip install transformers
pip install omegaconf
pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers
pip install kornia

echo Running TXT2IMG

python txt2img.py --ckpt ~/v1-5-pruned-emaonly.ckpt --n_samples 1 --n_iter 1 --H 256 --W 256 --prompt "a turtle with a horse on the back, by greg ruthkowski" 
