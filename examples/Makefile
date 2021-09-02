ROCM_PATH ?= /opt/rocm
ROCBLAS_PATH ?= $(ROCM_PATH)/rocblas
DAGEE_PATH ?= $(CURDIR)/../DAGEE
ATMI_PATH ?= $(ROCM_PATH)/atmi

DEBUG ?= 0
TIME ?= 0

CXX = $(ROCM_PATH)/bin/hipcc
INC = -Iinclude -I$(ROCM_PATH)/include -I$(ROCBLAS_PATH)/include
DAGEE_INC = -I$(DAGEE_PATH)/DAGEE-lib/include -I$(DAGEE_PATH)/cppUtils/include
ATMI_INC = -I$(ATMI_PATH)/include

CXXFLAGS =  $(INC)
LDFLAGS = -L$(ROCBLAS_PATH)/lib/ -lrocblas -Wl,-rpath=$(ROCBLAS_PATH)/lib


ATMI_LDFLAGS = -L$(ATMI_PATH)/lib -latmi_runtime -Wl,-rpath=$(ATMI_PATH)/lib


ifeq ($(DEBUG), 1)
	CXXFLAGS += -DDEBUG
endif

ifeq ($(TIME), 1)
	CXXFLAGS += -DTIME
endif

all: CPU_Strassen GPU_Strassen test_rocblas_wrappers test_ops_dagee test_cu_mask GPU_Strassen_dagee CPU_Strassen_dagee CPU_naive GPU_naive

CPU_Strassen:
	$(CXX) $(CXXFLAGS) src/CPU_Strassen.cpp $(LDFLAGS) -o CPU_Strassen

GPU_Strassen:
	$(CXX) $(CXXFLAGS) src/GPU_Strassen.cpp $(LDFLAGS) -o GPU_Strassen

CPU_naive:
	$(CXX) $(CXXFLAGS) src/CPU_naive.cpp $(LDFLAGS) -o CPU_naive

GPU_naive:
	$(CXX) $(CXXFLAGS) src/GPU_naive.cpp $(LDFLAGS) -o GPU_naive

test_rocblas_wrappers:
	$(CXX) $(CXXFLAGS) src/test_rocblas_wrappers.cpp $(LDFLAGS) -o test_rocblas_wrappers

test_ops_dagee:
	$(CXX) $(DAGEE_INC) $(ATMI_INC) $(CXXFLAGS) src/test_ops_dagee.cpp $(LDFLAGS) $(ATMI_LDFLAGS) -o test_ops_dagee

test_cu_mask:
	$(CXX) $(DAGEE_INC) $(ATMI_INC) $(CXXFLAGS) src/test_cu_mask.cpp $(LDFLAGS) $(ATMI_LDFLAGS) -o test_cu_mask

GPU_Strassen_dagee:
	$(CXX) $(DAGEE_INC) $(ATMI_INC) $(CXXFLAGS) src/GPU_Strassen_dagee.cpp $(LDFLAGS) $(ATMI_LDFLAGS) -o GPU_Strassen_dagee

CPU_Strassen_dagee:
	$(CXX) $(DAGEE_INC) $(ATMI_INC) $(CXXFLAGS) src/CPU_Strassen_dagee.cpp $(LDFLAGS) $(ATMI_LDFLAGS) -o CPU_Strassen_dagee

clean:
	rm -rf GPU_Strassen CPU_Strassen test_rocblas_wrappers test_ops_dagee test_cu_mask GPU_Strassen_dagee CPU_Strassen_dagee CPU_naive GPU_naive
