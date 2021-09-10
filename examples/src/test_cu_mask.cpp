#include <iostream>

#include <GPU_Utils.hpp>
#include <verify.hpp>

#include <dagee/ATMIdagExecutor.h>
#include <dagee/ATMIalloc.h>

__global__ void getSMIDs(uint32_t *SMIDs,uint32_t stream) {
  if(hipThreadIdx_x == 0) {
    SMIDs[hipBlockIdx_x+(stream*hipGridDim_x)] = __smid();
  }
}

void dagee_test(int num_cu) {

  dim3 threadsPerBlock;
  dim3 blocks;
  uint32_t streams = 3;

  threadsPerBlock.x = 1024;
  blocks.x = 64;

  std::vector<uint32_t> empty_vec(blocks.x*streams);

  using GpuExec = dagee::GpuExecutorAtmi;
  using DagExec = dagee::ATMIdagExecutor<GpuExec>;
  dagee::AllocManagerAtmi bufMgr;
  GpuExec gpuEx;
  DagExec dagEx(gpuEx);
  auto *dag = dagEx.makeDAG();


  uint32_t *SMIDs = bufMgr.makeSharedCopy(empty_vec);

  auto Func = gpuEx.registerKernel<unsigned *, dim3>(&getSMIDs);

  auto SMID_0Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, Func, SMIDs,0));
  auto SMID_1Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, Func, SMIDs,1));
  auto SMID_2Task = dag->addNode(gpuEx.makeTask(blocks, threadsPerBlock, Func, SMIDs,2));


  dag->addEdge(SMID_0Task, SMID_1Task);
  dag->addEdge(SMID_1Task, SMID_2Task);

  dagEx.execute(dag);


  for(int i=0; i<blocks.x*streams; i++) {
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
