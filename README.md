# Artist
## Motivation
- An easy to edit codebase for Progressive GAN originally published by this [research paper][Progressive GAN Research Paper] and other GANs
- Supplement my personal projects
- Resume builder
- For fun and to understand state-of-the-art AI

## Installation
#### CUDA Installation
Ensure you have a GPU if you want to train in any reasonable amount of time.
- [Install CUDA here][CUDA Install]
- [Install cuDNN here][cuDNN Install]
#### Project Installation
```sh
pip install Artist-Engineering-Geek
# Don't forget to install your specific pytorch and torchvision libraries for your gpu
# in my case, I have the NVIDIA RTX 3090 so this is my version
pip install --no-cache-dir --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu112/torch_nightly.html
```

## Running the program
To train the program, run the "Train.ipynb" notebook and alter your parameters at will on GitHub.
There should be sufficient in-code documentation for you to understand what the hell is going on

   [CUDA Install]: <https://developer.nvidia.com/cuda-downloads>
   [cuDNN Install]: <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html>
   [Progressive GAN Research Paper]: <https://arxiv.org/abs/1710.10196>