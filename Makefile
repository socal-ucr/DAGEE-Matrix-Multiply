CXX = /home/rleri001/.opt/rocm/bin/hipcc
INC = -I/home/rleri001/.opt/rocm/include -I/home/rleri001/.opt/rocblas/include -I/home/rleri001/rocBLAS/library/src/include
CXXFLAGS =  $(INC)
LDFLAGS = -L/home/rleri001/.opt/rocblas/lib/ -lrocblas

all: GPU_Strassen CPU_Strassen test_rocblas_wrappers

GPU_Strassen:
	$(CXX) $(INC) src/GPU_Strassen.cpp $(LBLIBS) $(LDFLAGS) -o GPU_Strassen

CPU_Strassen:
	$(CXX) $(INC) src/CPU_Strassen.cpp $(LBLIBS) $(LDFLAGS) -o CPU_Strassen

test_rocblas_wrappers:
	$(CXX) $(INC) src/test_rocblas_wrappers.cpp $(LBLIBS) $(LDFLAGS) -o test_rocblas_wrappers

clean:
	rm -rf GPU_Strassen CPU_Strassen test_rocblas_wrappers
