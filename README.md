
# Compatibility
![image](https://github.com/markovml/usage-examples/assets/82056365/73a319d8-e943-452d-a03a-7264e3a70f90)

# usage-examples
This repository has code samples to use MarkovML SDK. Please make sure you have Markov SDK installed.
The instructions to install the SDK are here: https://developer.markovml.com/docs/setup-your-machine

# Installation
We strongly recommend using a virtual environment or conda environment for MarkovML

# Virtual Environment

You can create a Virtual environment by


` python -m venv 'markovml' `

# Conda
You can create a conda environment by following the command:

`conda -n 'YOUR_ENVIRONMENT_NAME' -python=3.9`

Make sure you have Miniconda or anaconda installed. The instructions to install Minicoda are below.
The instructions to install anaconda are here: https://docs.anaconda.com/free/anaconda/install/index.html

### Miniconda Installation MacOS
`
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
`
### Miniconda Installation Linux
`
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
`

# Run all examples using tox
https://gist.github.com/kushagra7589/c361e8a96a2f4309fd0dfa4409a76263
