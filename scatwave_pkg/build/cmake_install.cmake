# Install script for directory: /home/eugene/Dropbox/PhD/scatnet/scatwave/scatwave_pkg

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/home/eugene/torch/install")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "Release")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/scatwave/scm-1/lua/scatwave/cuda" TYPE FILE FILES
    "/home/eugene/Dropbox/PhD/scatnet/scatwave/scatwave_pkg/cuda/engine_CUDA_nvidia.lua"
    "/home/eugene/Dropbox/PhD/scatnet/scatwave/scatwave_pkg/cuda/wrapper_CUDA_fft_nvidia.lua"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/luarocks/rocks/scatwave/scm-1/lua/scatwave" TYPE FILE FILES
    "/home/eugene/Dropbox/PhD/scatnet/scatwave/scatwave_pkg/network_2d_translation.lua"
    "/home/eugene/Dropbox/PhD/scatnet/scatwave/scatwave_pkg/complex.lua"
    "/home/eugene/Dropbox/PhD/scatnet/scatwave/scatwave_pkg/init.lua"
    "/home/eugene/Dropbox/PhD/scatnet/scatwave/scatwave_pkg/unit_test.lua"
    "/home/eugene/Dropbox/PhD/scatnet/scatwave/scatwave_pkg/filters_bank.lua"
    "/home/eugene/Dropbox/PhD/scatnet/scatwave/scatwave_pkg/engine.lua"
    "/home/eugene/Dropbox/PhD/scatnet/scatwave/scatwave_pkg/conv_lib.lua"
    "/home/eugene/Dropbox/PhD/scatnet/scatwave/scatwave_pkg/wrapper_fft.lua"
    "/home/eugene/Dropbox/PhD/scatnet/scatwave/scatwave_pkg/tools.lua"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
ELSE(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
ENDIF(CMAKE_INSTALL_COMPONENT)

FILE(WRITE "/home/eugene/Dropbox/PhD/scatnet/scatwave/scatwave_pkg/build/${CMAKE_INSTALL_MANIFEST}" "")
FOREACH(file ${CMAKE_INSTALL_MANIFEST_FILES})
  FILE(APPEND "/home/eugene/Dropbox/PhD/scatnet/scatwave/scatwave_pkg/build/${CMAKE_INSTALL_MANIFEST}" "${file}\n")
ENDFOREACH(file)
