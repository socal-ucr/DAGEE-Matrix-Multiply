# Compilation

`make -j`

Compile flags supported:

DEBUG : Enable debug and verification

TIME : Time the execution runtime

To run it with any other compile flags, use:

`make $COMPILE_FLAG=1 -j`

# Notes

## Make sure ATMI is compiled and installed following the [ATMI install instructions](https://github.com/socal-ucr/atmi/blob/15ab2af651a6a394d37e080bfee3735fcaeb6d7b/INSTALL.md).

### Default paths for ATMI, DAGEE, ROCM, ROCBLAS are set as follows. If necessary, export appropriate paths as environment variables.

`ATMI_PATH = $(ROOT)/atmi`

`DAGEE_PATH = $(ROOT)/DAGEE`

`ROCM_PATH = /opt/rocm`

`ROCBLAS_PATH = $(ROCM_PATH)/rocblas`
