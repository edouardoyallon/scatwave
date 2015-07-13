fftw3-ffi
========

A LuaJIT interface to FFTW3

# Installation #

First, make sure FFTW3 is installed on your system. This package only requires the binary shared libraries (.so, .dylib, .dll).
Please see your package management system to install FFTW3. 
You can also download and compile fftw3 from [FFTW3 web page](www.fftw.org)

```sh
luarocks install https://raw.github.com/soumith/fftw3-ffi/master/rocks/fftw3-scm-1.rockspec
```

# Usage #

```lua
local fftw = require 'fftw3'
...
```
Float version
```lua
local fftw = require 'fftw3'
fftw = fftw.float
...
```


All FFTW C functions are available in the `fftw` namespace returned by require. The only difference is the naming, which is not prefixed
by `fftw_` anymore. 
