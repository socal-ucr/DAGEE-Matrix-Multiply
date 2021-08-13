ROCM_PATH = $(HOME)/.opt/rocm
ROCBLAS_PATH = $(ROCM_PATH)/rocblas

CXX = $(ROCM_PATH)/bin/hipcc
INC = -I$(ROCM_PATH)/include -I$(HOME)/.opt/rocblas/include -I$(ROCBLAS_PATH)/include
CXXFLAGS =  $(INC)
LDFLAGS = -L$(ROCBLAS_PATH)/lib/ -lrocblas

all: GPU_Strassen CPU_Strassen test_rocblas_wrappers

GPU_Strassen:
	$(CXX) $(INC) src/GPU_Strassen.cpp $(LDFLAGS) -o GPU_Strassen

CPU_Strassen:
	$(CXX) $(INC) src/CPU_Strassen.cpp $(LDFLAGS) -o CPU_Strassen

test_rocblas_wrappers:
	$(CXX) $(INC) src/test_rocblas_wrappers.cpp $(LDFLAGS) -o test_rocblas_wrappers

clean:
	rm -rf GPU_Strassen CPU_Strassen test_rocblas_wrappers
