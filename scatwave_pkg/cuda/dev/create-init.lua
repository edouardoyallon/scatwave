--  Adapted from https://github.com/torch/sdl2-ffi/blob/master/dev/create-init.lua
print[[
-- Do not change this file manually
-- Generated with dev/create-init.lua

local ffi = require 'ffi'
local C = ffi.load('fftw3')
local ok, Cf = pcall(function () return ffi.load('fftw3f') end)
if not ok then
     print('Warning: float version of libfftw3: libfftw3f.(so/dylib) not found')
end
local fftw = {C=C}
local fftwf = {Cf = Cf}
fftw.float = fftwf

require 'fftw3.cdefs'

local defines = require 'fftw3.defines'
defines.register_hashdefs(fftw, C)
defines.register_hashdefsf(fftwf, Cf)

local function register(luafuncname, funcname)
   local symexists, msg = pcall(function()
                              local sym = C[funcname]
                           end)
   if symexists then
      fftw[luafuncname] = C[funcname]
   end
end

local function registerf(luafuncname, funcname)
   local symexists, msg = pcall(function()
                              local sym = Cf[funcname]
                           end)
   if symexists then
      fftwf[luafuncname] = Cf[funcname]
   end
end
]]

local defined = {}

local txt = io.open('cdefs.lua'):read('*all')
for funcname in txt:gmatch('fftw_([^%=,%.%;<%s%(%)]+)%s*%(') do
   if funcname and not defined[funcname] then
      local luafuncname = funcname:gsub('^..', function(str)
                                                  if str == 'RW' then
                                                     return str
                                                  else
                                                     return string.lower(str:sub(1,1)) .. str:sub(2,2)
                                                  end
                                               end)
      print(string.format("register('%s', 'fftw_%s')", luafuncname, funcname))
      defined[funcname] = true
   end
end

print()

for defname in txt:gmatch('fftw_([^%=,%.%;<%s%(%)|%[%]]+)') do
   if not defined[defname] then
      print(string.format("register('%s', 'fftw_%s')", defname, defname))
   end
end

print()
print()
print()

local definedf = {}

for funcname in txt:gmatch('fftwf_([^%=,%.%;<%s%(%)]+)%s*%(') do
   if funcname and not definedf[funcname] then
      local luafuncname = funcname:gsub('^..', function(str)
                                                  if str == 'RW' then
                                                     return str
                                                  else
                                                     return string.lower(str:sub(1,1)) .. str:sub(2,2)
                                                  end
                                               end)
      print(string.format("registerf('%s', 'fftwf_%s')", luafuncname, funcname))
      definedf[funcname] = true
   end
end

print()

for defname in txt:gmatch('fftwf_([^%=,%.%;<%s%(%)|%[%]]+)') do
   if not definedf[defname] then
      print(string.format("registerf('%s', 'fftwf_%s')", defname, defname))
   end
end

print[[

return fftw
]]
