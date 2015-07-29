require 'cutorch'
require 'gnuplot'
require 'image'
require 'env'
require 'sys'



x=image.lena()
x=image.scale(x,32,32)
x=x:float()

x=x:cuda()
-- Let's launch ScatWave!
ScatWave = require 'init'



sys.tic()
y=ScatWave.network.new(32,32,2,2) -- constructor to build a network
   
y:cuda()  -- make the network using CUDA
   
-- for images of size
print(sys.toc())   -- 32 by 32 invariant up to 2^2



-- Let's try to get the scattering coefficients
   sys.tic()
for i=1,10 do
   z=y:scat(x)
   
end
   print(sys.toc())


-- Printing
-- print(z)

