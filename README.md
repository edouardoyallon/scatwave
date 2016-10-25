![Logo](https://github.com/edouardoyallon/scatwave/blob/master/logo.png)

# ScatWave

ScatWave is a Torch implementation of 2D scattering using CUDA libraries, designed for images.

# Disclaimer 

This software belongs to the team DATA @ ENS, its main author is Edouard Oyallon.

# How to install

Assuming Torch is already installed on your computer, simply cd in scatwave_pkg, then 'luarocks make'

# Few results...

ScatWave + one FC = on CIFAR 10__
ScatWave + one FC = on CIFAR 100__
ScatWave + Deepnet = 90.6% on CIFAR10 with__
ScatWave + Deepnet = 90.6% on CIFAR100 with__

# Usage

scatwave = require 'scatwave'__
x=torch.FloatTensor(128,3,32,32)__
scat = scatwave.network.new(3,x:size())__
scat_coeff = scat(x) -- or scat(x,1)__

You can go to cuda via:__
scat=scat:cuda()__


# Contributors

Carmine Cella, Edouard Oyallon. Contacts: surname.name@ens.fr

Team DATA - Ecole Normale Supérieure
