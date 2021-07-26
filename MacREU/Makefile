CXX = /home/rleri001/.opt/rocm/bin/hipcc
INC = -I/home/rleri001/.opt/rocm/include -I/home/rleri001/.opt/rocblas/include -I/home/rleri001/rocBLAS/library/src/include
CXXFLAGS =  $(INC)
LDFLAGS = -L/home/rleri001/.opt/rocblas/lib/ -lrocblas

SRC = sequential-strassen.cpp
OBJ = $(SRC:.cpp=.o)
EXEC = sequential-strassen

all: $(EXEC)

$(EXEC):
	$(CXX) $(INC) $(SRC) $(LBLIBS) $(LDFLAGS) -o $(EXEC)

clean:
	rm -rf $(EXEC)