![Logo](https://github.com/edouardoyallon/scatwave/blob/master/logo.png)

# ScatWave

ScatWave is a Torch implementation of 2D scattering using CUDA libraries, designed for images.

# Disclaimer 

This software belongs to the team DATA @ ENS, its main author is Edouard Oyallon.

# How to install

Assuming Torch is already installed on your computer, simply cd in scatwave_pkg, then 'luarocks make'
Make sure you have FFTW and cuFFT installed and that the libraries are linked to the software.

# Few results...

ScatWave + one FC = on CIFAR 10<br/>
ScatWave + one FC = on CIFAR 100<br/>
ScatWave + Deepnet = 90.6% on CIFAR10<br/>
ScatWave + Deepnet = 90.6% on CIFAR100<br/>

# Usage

scatwave = require 'scatwave'<br/>
x=torch.FloatTensor(128,3,32,32)<br/>
scat = scatwave.network.new(3,x:size())<br/>
scat_coeff = scat(x) -- or scat(x,1)<br/>
<br/>
You can go to cuda via:<br/>
scat=scat:cuda()<br/>

# Reproducing the paper

- Data can be downloaded from this page: https://github.com/szagoruyko/wide-residual-networks/blob/master/README.md.<br/>
The whitened versions work quite better and are used in this work.

- training the network on cifar10:<br/>
th train_cifar10.lua

- training the network on cifar100:<br/>
th train_cifar100.lua

- transfering to matlab W1:<br/>
th get_W1.lua

- analysing the operator:<br/>
matlab sparsify_W1.m

- retraining the deepnet with a new W1:<br/>
th retrain_with_fix_W1_pretrained_end_cifar10.lua

- replace the scattering by a deepnet with a pretrained and fixed model:<br/>
th replace_scattering_fix_end_cifar10.lua

# Contributors

Carmine Cella, Edouard Oyallon. Contacts: surname.name@ens.fr

Team DATA - Ecole Normale Supérieure
