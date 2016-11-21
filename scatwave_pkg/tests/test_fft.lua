require 'scatwave'
fft=require 'scatwave.wrapper_fft'
inp = torch.FloatTensor(1,1,4,4,2):zero()  
inp[1][1][1][2][1]=1 
print('fft', fft.my_2D_fft_complex_batch(inp,3,0))
print('ifft', fft.my_2D_fft_complex_batch(inp,3,1))
