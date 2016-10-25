![Logo](https://github.com/edouardoyallon/scatwave/blob/master/logo.png)

# ScatWave

ScatWave is a Torch implementation of 2D scattering using CUDA libraries, designed for images.

# Disclaimer 

This software belongs to the team DATA @ ENS, its main author is Edouard Oyallon.

# How to install

Assuming Torch is already installed on your computer, simply cd in scatwave_pkg, then 'luarocks make'

# Few results...

ScatWave + one FC = on CIFAR 10
ScatWave + one FC = on CIFAR 100
ScatWave + Deepnet = 90.6% on CIFAR10 with
ScatWave + Deepnet = 90.6% on CIFAR100 with

# Usage

scatwave = require 'scatwave'
x=torch.FloatTensor(128,3,32,32)
scat = scatwave.network.new(3,x:size())
scat_coeff = scat(x) -- or scat(x,1)

You can go to cuda via:
scat=scat:cuda()


# Contributors

Carmine Cella, Edouard Oyallon. Contacts: surname.name@ens.fr

Team DATA - Ecole Normale Supérieure
