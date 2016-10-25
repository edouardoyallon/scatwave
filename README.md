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
ScatWave + Deepnet = 90.6% on CIFAR10 with<br/>
ScatWave + Deepnet = 90.6% on CIFAR100 with<br/>

# Usage

scatwave = require 'scatwave'<br/>
x=torch.FloatTensor(128,3,32,32)<br/>
scat = scatwave.network.new(3,x:size())<br/>
scat_coeff = scat(x) -- or scat(x,1)<br/>
<br/>
You can go to cuda via:<br/>
scat=scat:cuda()<br/>


# Contributors

Carmine Cella, Edouard Oyallon. Contacts: surname.name@ens.fr

Team DATA - Ecole Normale Supérieure
