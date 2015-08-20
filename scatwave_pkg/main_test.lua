require 'torch'
require 'gnuplot'
require 'image'
require 'env'
require 'sys'
complex=require 'complex'


-- Let's launch ScatWave!
ScatWave = require 'scatwave'


sys.tic()
y=ScatWave.network.new(128,128,4,2)
print(sys.toc())

-- Load lena
x=image.lena()
x=image.scale(x,128,128)
x=x:float()


-- Let's try to get the scattering coefficients
sys.tic()
z=y:scat(x)
print(sys.toc())