CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
FIND_PACKAGE(Torch REQUIRED)


IF(LUAROCKS_PREFIX)
    MESSAGE(STATUS "Installing ScatWave through Luarocks")
    STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
    MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()


FILE(GLOB luasrc *.lua)
FILE(GLOB luasrc_cuda cuda/*.lua)



INSTALL(
  FILES
  ${luasrc_cuda}
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/scatwave/cuda")

ADD_TORCH_PACKAGE(scatwave ""  "${luasrc}" "ScatWave toolbox")
