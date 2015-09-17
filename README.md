![Logo](https://github.com/edouardoyallon/scatwave/blob/master/logo.png)

# ScatWave

ScatWave is a Torch implementation of scattering and other mathematical representations using CUDA libraries, designed for images, sounds, natural language processing and video.

# Disclaimer 

CONFIDENTIAL: DO NOT SHARE THIS SOFTWARE

# How to install

Assuming Torch is already installed on your computer, simply cd in scatwave_pkg, then 'luarocks make'

# Few results...

This version gives comparable results relatively to the version of MATLAB. (78.5% accuracy with a Gaussian SVM, the difference with the CVPR paper comes from the fact the inputs are different(not resizing to 64x64 and renormalized) and that in this version there is no usage of more wavelet per octave) On MATLAB the computation of the feature takes around 30 minutes with the paralleltoolbox, one epoch with this version takes around 100s. Now 86.2% on CIFAR-10. Results on ImageNet soon.

# Usage

scatwave = require 'scatwave'
x=torch.FloatTensor(128,3,32,32)
scat = scatwave.network.new(3,x:size())
scat_coeff = scat(x) -- or scat(x,1)


# Contributors

Mathieux Andreux, Carmine Cella, Vincent Lostanlen, Edouard Oyallon. Contacts: surname.name@ens.fr

Team DATA - Ecole Normale Supérieure
