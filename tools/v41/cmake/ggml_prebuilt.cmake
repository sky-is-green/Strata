# tools/v41/cmake/ggml_prebuilt.cmake - link the ggml that STRATA_GGML_DIR has ALREADY built.
#
#   cmake -DSTRATA_GGML_DIR=/path/to/llama.cpp -DSTRATA_GGML_PREBUILT=/path/to/llama.cpp/build ...
#
# Why: on gfx1100 the HIP backend of ggml takes hours to compile (the flash-attention template instances), while the
# engine's own sources are minutes.  This creates the same target names the rest of the build links - ggml, ggml-base,
# ggml-cpu and, when that build has one, ggml-hip or ggml-cuda - from the archives in that build directory.  The
# headers come from STRATA_GGML_DIR, i.e. the source tree the archives were built from, so nothing can drift; a
# normal configure never includes this file (STRATA_GGML_PREBUILT is empty by default).
if(NOT STRATA_GGML_DIR)
  message(FATAL_ERROR "STRATA_GGML_PREBUILT needs STRATA_GGML_DIR (the source tree the build directory belongs to)")
endif()
set(_gp_src "${STRATA_GGML_DIR}/ggml")
set(_gp "${STRATA_GGML_PREBUILT}/ggml/src")
set(_gp_bin "${STRATA_GGML_PREBUILT}/bin")
set(_gp_archives "${_gp}/libggml.a" "${_gp}/libggml-cpu.a" "${_gp}/libggml-base.a")
set(_gp_hip "${_gp}/ggml-hip/libggml-hip.a")
if(NOT EXISTS "${_gp_hip}")
  set(_gp_hip "")
endif()
set(_gp_cuda "${_gp}/ggml-cuda/libggml-cuda.a")
if(NOT EXISTS "${_gp_cuda}")
  set(_gp_cuda "")
endif()
# A build configured with -DBUILD_SHARED_LIBS=ON has no archives: its libraries are <build>/bin/libggml*.so.
set(_gp_shared "")
if(NOT EXISTS "${_gp}/libggml.a" AND EXISTS "${_gp_bin}/libggml.so")
  set(_gp_shared "${_gp_bin}/libggml.so" "${_gp_bin}/libggml-cpu.so" "${_gp_bin}/libggml-base.so")
  if(EXISTS "${_gp_bin}/libggml-hip.so")
    set(_gp_hip "${_gp_bin}/libggml-hip.so")
  elseif(EXISTS "${_gp_bin}/libggml-cuda.so")
    set(_gp_cuda "${_gp_bin}/libggml-cuda.so")
  endif()
  foreach(_f IN LISTS _gp_shared)
    if(NOT EXISTS "${_f}")
      message(FATAL_ERROR "STRATA_GGML_PREBUILT: ${_f} is missing - that build did not produce the ggml libraries")
    endif()
  endforeach()
endif()
foreach(_f IN LISTS _gp_archives)
  if(NOT _gp_shared AND NOT EXISTS "${_f}")
    message(FATAL_ERROR "STRATA_GGML_PREBUILT: ${_f} is missing - build ggml in ${STRATA_GGML_PREBUILT} first")
  endif()
endforeach()
if(STRATA_ENABLE_HIP AND NOT _gp_hip)
  message(FATAL_ERROR "STRATA_GGML_PREBUILT: no ggml-hip library in ${STRATA_GGML_PREBUILT}: configure that build "
                      "with -DGGML_HIP=ON for this card's architecture")
endif()

# One group: ggml.a's backend registry references ggml_backend_cpu_reg / _cuda_reg and the backends reference
# ggml-base, so a one-pass static link needs the cycle broken (llama.cpp links the same archives this way).  The
# registry also references backends that build had enabled but this engine does not link (Vulkan here): the stub
# library defines those as empty registrations, which the registry skips.  Shared libraries need none of that: the
# link is just the .so list plus their dependencies.
add_library(_strata_ggml_prebuilt_stubs STATIC "${CMAKE_CURRENT_SOURCE_DIR}/tools/v41/cmake/ggml_prebuilt_stubs.cpp")
target_include_directories(_strata_ggml_prebuilt_stubs PRIVATE "${_gp_src}/include")
if(_gp_shared)
  set(_gp_link ${_gp_shared})
  # the SONAME links (libggml.so.0 -> ...) are what the loader wants; the stub still answers a registration the
  # shared libggml.so may reference (Vulkan) without pulling that backend in
  list(APPEND _gp_link _strata_ggml_prebuilt_stubs)
  if(_gp_hip)
    list(APPEND _gp_link "${_gp_hip}")
  endif()
  if(_gp_cuda)
    list(APPEND _gp_link "${_gp_cuda}")
  endif()
else()
  set(_gp_link "-Wl,--start-group" ${_gp_archives})
  if(_gp_hip)
    list(APPEND _gp_link "${_gp_hip}")
  endif()
  if(_gp_cuda)
    list(APPEND _gp_link "${_gp_cuda}")
  endif()
  list(APPEND _gp_link "-Wl,--end-group")
  # after the group, so the member that defines the missing registration is actually pulled in
  list(APPEND _gp_link _strata_ggml_prebuilt_stubs)
endif()
if(_gp_hip)
  # hipblas brings rocblas with it; a distro hipBLAS that does not is handled by cmake/hip_backend.cmake, whose
  # roc::rocblas target is only there when that package was found
  list(APPEND _gp_link hip::host roc::hipblas)
  if(TARGET roc::rocblas)
    list(APPEND _gp_link roc::rocblas)
  endif()
endif()
if(_gp_cuda)
  list(APPEND _gp_link CUDA::cudart)
endif()

add_library(_strata_ggml_prebuilt INTERFACE)
target_link_libraries(_strata_ggml_prebuilt INTERFACE ${_gp_link})
target_include_directories(_strata_ggml_prebuilt INTERFACE
  "${_gp_src}/include" "${_gp_src}/src" "${_gp_src}/src/ggml-cpu")
# that build may have had OpenMP on (ggml-cpu then calls libgomp)
find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_strata_ggml_prebuilt INTERFACE OpenMP::OpenMP_CXX)
endif()

foreach(_t IN ITEMS ggml ggml-base ggml-cpu)
  add_library(${_t} INTERFACE IMPORTED GLOBAL)
  target_link_libraries(${_t} INTERFACE _strata_ggml_prebuilt)
endforeach()
if(_gp_hip)
  add_library(ggml-hip INTERFACE IMPORTED GLOBAL)
  target_link_libraries(ggml-hip INTERFACE _strata_ggml_prebuilt)
endif()
if(_gp_cuda)
  add_library(ggml-cuda INTERFACE IMPORTED GLOBAL)
  target_link_libraries(ggml-cuda INTERFACE _strata_ggml_prebuilt)
endif()
message(STATUS "Strata: ggml linked prebuilt from ${STRATA_GGML_PREBUILT} (${_gp_hip}${_gp_cuda})")
