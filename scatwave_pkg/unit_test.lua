local unit_test_scatnet={}

require 'torch'
tester = torch.Tester()
local complex=require 'complex'

function unit_test_scatnet.complex()
   
local playing = torch.Tensor(4,5,7,2)

   
   for i=1,4 do
      for j=1,5 do
         for k=1,7 do
            playing[i][j][k][1]=i*j*k
            playing[i][j][k][1]=torch.sqrt(i*j*k)
         end
      end
   end

   
   
   -- unit_complex
   local u_a=complex.unit_complex(0.23)
   tester:asserteq(u_a:nDimension(),2,'Unit complex number dimension isn\'t 2')
   tester:asserteq(u_a:size(2),2,'Not a complex')
   
   local norm_u_a=torch.sqrt(u_a[1][1]^2+u_a[1][2]^2)
   tester:asserteq(norm_u_a,1,'Not norm equal to 1')
   
   -- complex.abs_value
   tester:asserteq(complex.abs_value(u_a)[1],1,'Abs value not equal to 1')
   
   

end



tester:add(unit_test_scatnet)
tester:run()

return unit_test_scatnet