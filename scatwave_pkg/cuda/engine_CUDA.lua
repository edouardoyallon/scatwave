local ffi = require 'ffi'

local cuFFT = ffi.load('cufft')

ffi.cdef[[
struct fftw_iodim_do_not_use_me {
     int n;
     int is;
     int os;
};


fftw_plan CUFFTAPI fftw_plan_guru_dft_r2c(int rank, const fftw_iodim *dims,
                                          int batch_rank, const fftw_iodim *batch_dims,
                                          double *in, fftw_complex *out, 
                                          unsigned flags);


fftw_plan CUFFTAPI fftw_plan_guru_dft(int rank, const fftw_iodim *dims,
                                      int batch_rank, const fftw_iodim *batch_dims,
                                      fftw_complex *in, fftw_complex *out,
                                      int sign, unsigned flags);


]]


return cuFFT
