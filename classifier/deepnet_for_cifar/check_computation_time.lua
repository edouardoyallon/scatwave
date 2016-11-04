require 'cutorch'
require 'cunn'
sys=require 'sys'

scatwave = require 'scatwave'



-----------------------------------------------
---- SCATTERING
-----------------------------------------------


N={32,128,256};
J={2,4,5}


for n=3,3 do
for j=1,3 do



SCAT=scatwave.network.new(J[j],torch.LongStorage({128,3,N[n],N[n]}))
SCAT:cuda()
      
      x=torch.randn(128,3,N[n],N[n]):cuda()
      init_t=sys.clock()
sys.tic()
for t=1,10 do
         s=SCAT:scat(x)
      end
cutorch.synchronize()
      final_t=sys.toc()
    print('J:'..J[j]..' N:' .. N[n] .. ' time:'..(final_t)/10)
end
end
