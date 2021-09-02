#include <iostream>

#include <GPU_Utils.hpp>
#include <verify.hpp>

#include <dagee/ATMIdagExecutor.h>
#include <dagee/ATMIalloc.h>

__global__ void getSMIDs(unsigned *SMIDs) {
  if(hipThreadIdx_x == 0) {
    SMIDs[hipBlockIdx_x] = __smid();
  }
}

void dagee_test(int num_cu) {

  dim3 threadsPerBlock;
  dim3 blocks;

  threadsPerBlock.x = 1024;
  blocks.x = 64;

  std::vector<unsigned int> empty_vec(blocks.x);

  using GpuExec = dagee::GpuExecutorAtmi;
  using DagExec = dagee::ATMIdagExecutor<GpuExec>;
  dagee::AllocManagerAtmi bufMgr;
  GpuExec gpuEx;
  DagExec dagEx(gpuEx);
  auto *dag = dagEx.makeDAG();


  unsigned *SMIDs = bufMgr.makeSharedCopy(empty_vec);

  auto Func = gpuEx.registerKernel<unsigned *, dim3>(&getSMIDs);

  auto SMIDTask = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, Func, SMIDs));

  dagEx.execute(dag);


  for(int i=0; i<blocks.x; i++) {
    printf("%d:%u, ", i, SMIDs[i]);
  }
  
  printf("\n");
}


int main(int argc, char **argv)
{
    int num_cus = (argc > 1) ? atoi(argv[1]) : 8;
    dagee_test(num_cus);

    std::cout << "Success" << std::endl;

    return 0;
}







