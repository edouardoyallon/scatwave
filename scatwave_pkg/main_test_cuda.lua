require 'cutorch'
require 'gnuplot'
require 'image'
require 'env'
require 'sys'


x=torch.randn(128,3,32,32)
x=x:cuda()

-- Let's launch ScatWave!
ScatWave = require 'init'


sys.tic()
y=ScatWave.network.new(32,32,2,3)
y:cuda()

-- Let's try to get the scattering coefficients
for i=1,10 do
   z=y:scat(x)
   if(i%2==0) then
      collectgarbage()   
   end
end

cutorch.synchronize()
print(sys.toc()/10)

