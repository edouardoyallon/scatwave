--[[
     ScatWave implementation of Scattering Network
     Written by Edouard Oyallon
     Team DATA ENS
     Copyright 2015
]]

local ffi = require 'ffi'
local FFT = {}


local ok, err = pcall(function () FFT.C=ffi.load('fftw3f.so.3') end)

if(not ok) then
   print(err)
   error('library cufft not found...')
end

-- defines types
ffi.cdef[[
typedef struct _FILE FILE;
typedef long double __float128;
typedef struct fftwf_plan_s *fftwf_plan; 
typedef float fftwf_complex[2]; 
]]


-- defines structures
ffi.cdef[[
extern fftwf_plan fftwf_plan_many_dft(int rank, const int *n, int howmany, fftwf_complex *in, const int
                                     *inembed, int istride, int idist, fftwf_complex *out, const int
                                     *onembed, int ostride, int odist, int sign, unsigned flags); 
extern void fftwf_destroy_plan(fftwf_plan p); 
extern void fftwf_execute(const fftwf_plan p); 
extern fftwf_plan fftwf_plan_dft_2d(int n0, int n1, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags); 
extern fftwf_plan fftwf_plan_many_dft_r2c(int rank, const int *n, int howmany, float *in, const int *inembed, int istride, int idist, fftwf_complex *out, const int *onembed, int ostride, int odist, unsigned flags); 
extern fftwf_plan fftwf_plan_many_dft_c2r(int rank, const int *n, int howmany,
                                      fftwf_complex *in, const int *inembed,
                                      int istride, int idist,
                                      float *out, const int *onembed,
                                      int ostride, int odist,
                                      unsigned flags);

]]

-- defines constant 
FFT.FORWARD  = -1
FFT.BACKWARD =  1
FFT.ESTIMATE = 64




return FFT
