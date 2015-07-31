local ffi = require 'ffi'
local cuFFT = {}


local ok, cuFFT_lib = pcall(function () return ffi.load('cufftw') end)

if(not ok) then
   error('library cufft not found...')
end

-- defines types
ffi.cdef[[
typedef double fftw_complex[2];
typedef float fftwf_complex[2];

typedef struct {
    int n;
    int is;
    int os;
} fftw_iodim;

typedef fftw_iodim fftwf_iodim;

typedef void *fftwf_plan;
typedef void *fftw_plan;]]


-- defines structures
ffi.cdef[[
extern fftw_plan fftw_plan_guru_dft_r2c(int rank, const fftw_iodim *dims,
                                          int batch_rank, const fftw_iodim *batch_dims,
                                          double *in, fftw_complex *out, 
                                          unsigned flags);


extern fftw_plan fftw_plan_guru_dft(int rank, const fftw_iodim *dims,
                                      int batch_rank, const fftw_iodim *batch_dims,
                                      fftw_complex *in, fftw_complex *out,
                                      int sign, unsigned flags);


extern fftwf_plan fftwf_plan_guru_dft(int rank, const fftwf_iodim *dims,
                                        int batch_rank, const fftwf_iodim *batch_dims,
                                        fftwf_complex *in, fftwf_complex *out,
                                        int sign, unsigned flags);
                                        
extern fftwf_plan fftwf_plan_guru_dft_r2c(int rank, const fftwf_iodim *dims,
                                            int batch_rank, const fftwf_iodim *batch_dims,
                                            float *in, fftwf_complex *out, 
                                            unsigned flags);
extern void fftw_execute(const fftw_plan plan);
extern void fftwf_execute(const fftwf_plan plan);

extern fftwf_plan fftwf_plan_dft_2d(int n0,
                                    int n1, 
                                    fftwf_complex *in,
                                    fftwf_complex *out, 
                                    int sign, 
                                    unsigned flags);

extern void fftwf_destroy_plan(fftwf_plan plan);  

extern fftwf_plan fftwf_plan_many_dft(int rank,
                                        const int *n,
                                        int batch,
                                        fftwf_complex *in,
                                        const int *inembed, int istride, int idist,
                                        fftwf_complex *out,
                                        const int *onembed, int ostride, int odist,
                                        int sign, unsigned flags);
extern fftwf_plan fftwf_plan_many_dft_r2c(int rank,
                                            const int *n,
                                            int batch,
                                            float *in,
                                            const int *inembed, int istride, int idist,
                                            fftwf_complex *out,
                                            const int *onembed, int ostride, int odist,
                                            unsigned flags);




]]

-- defines constant 
cuFFT.FORWARD  = -1
cuFFT.BACKWARD =  1
cuFFT.ESTIMATE = 3

-- registers function in a "soumith" style. It checks that function exists before adding it to the list!
local function register(luafuncname, funcname)
   local symexists, msg = pcall(function()
                              local sym = cuFFT_lib[funcname]
                           end)
   if symexists then
      cuFFT[luafuncname] = cuFFT_lib[funcname]
   else 

      error(string.format('%s of the library cuFFT not found!',funcname))
   end
end

register('execute','fftwf_execute')
--register('plan_guru_dft_r2c_f','fftwf_plan_guru_dft_r2c')
register('plan_guru_dft_r2c','fftwf_plan_guru_dft_r2c')
register('plan_guru_dft','fftwf_plan_guru_dft')
register('plan_dft_2d','fftwf_plan_dft_2d')
register('plan_many_dft_r2c','fftwf_plan_many_dft_r2c')
register('destroy_plan','fftwf_destroy_plan')
register('plan_many_dft','fftwf_plan_many_dft')

return cuFFT
