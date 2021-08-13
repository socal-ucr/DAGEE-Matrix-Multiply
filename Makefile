ROCM_PATH ?= /opt/rocm
ROCBLAS_PATH ?= $(ROCM_PATH)/rocblas

DEBUG ?= 0
TIME ?= 0

CXX = $(ROCM_PATH)/bin/hipcc
INC = -Iinclude -I$(ROCM_PATH)/include -I$(ROCBLAS_PATH)/include
CXXFLAGS =  $(INC)
LDFLAGS = -L$(ROCBLAS_PATH)/lib/ -lrocblas

ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG
endif

ifeq ($(TIME), 1)
	CXXFLAGS += -DTIME
endif

all: GPU_Strassen CPU_Strassen test_rocblas_wrappers

GPU_Strassen:
	$(CXX) $(CXXFLAGS) src/GPU_Strassen.cpp $(LDFLAGS) -o GPU_Strassen

CPU_Strassen:
	$(CXX) $(CXXFLAGS) src/CPU_Strassen.cpp $(LDFLAGS) -o CPU_Strassen

test_rocblas_wrappers:
	$(CXX) $(CXXFLAGS) src/test_rocblas_wrappers.cpp $(LDFLAGS) -o test_rocblas_wrappers

clean:
	rm -rf GPU_Strassen CPU_Strassen test_rocblas_wrappers
