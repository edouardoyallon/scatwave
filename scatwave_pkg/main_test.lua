require 'torch'
require 'gnuplot'
require 'image'
require 'env'
require 'sys'


-- Let's launch ScatWave!
ScatWave = require 'init'



sys.tic()
y=ScatWave.network.new(128,128,2,2) -- constructor to build a network for images of size
print(sys.toc())   -- 32 by 32 invariant up to 2^2



x=image.lena()
x=image.scale(x,128,128)
--x=torch.squeeze(torch.sum(x,1))

-- Let's try to get the scattering coefficients
   sys.tic()
for i=1,4 do
   z=y:scat(x)
end
   print(sys.toc()/4)


-- Printing
-- print(z)

