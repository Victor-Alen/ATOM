// Read frozen hipBLASLt heuristics for Mellum's strided BF16 QK batched GEMM.
// This enumerates sequence lengths, never prompt contents or output tokens.
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <cstdlib>
#include <iostream>
#include <cstdint>
#define OK(x) do { auto st=(x); if(st!=HIPBLAS_STATUS_SUCCESS) { std::cerr << #x << " status " << st << '\n'; return 1; } } while(0)
int main(int argc,char** argv) {
  int first=argc>1?std::atoi(argv[1]):1;
  int last=argc>2?std::atoi(argv[2]):16384;
  size_t workspace=argc>3?std::strtoull(argv[3],nullptr,10):33554432;
  if(first<1 || last<first || last>32768)return 2;
  hipblasLtHandle_t handle;
  OK(hipblasLtCreate(&handle));
  hipblasLtMatmulDesc_t desc;
  OK(hipblasLtMatmulDescCreate(&desc,HIPBLAS_COMPUTE_32F,HIP_R_32F));
  hipblasOperation_t ta=HIPBLAS_OP_T, tb=HIPBLAS_OP_N;
  OK(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));
  OK(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
  hipblasLtMatmulPreference_t pref;
  OK(hipblasLtMatmulPreferenceCreate(&pref));
  OK(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace)));
  for(int n=first;n<=last;++n) {
    hipblasLtMatrixLayout_t a,b,c;
    OK(hipblasLtMatrixLayoutCreate(&a,HIP_R_16BF,128,n,128));
    OK(hipblasLtMatrixLayoutCreate(&b,HIP_R_16BF,128,n,4096));
    OK(hipblasLtMatrixLayoutCreate(&c,HIP_R_16BF,n,n,n));
    int32_t batch=32;
    int64_t sa=int64_t(n)*128,sb=128,sc=int64_t(n)*n;
    OK(hipblasLtMatrixLayoutSetAttribute(a,HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch)));
    OK(hipblasLtMatrixLayoutSetAttribute(b,HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch)));
    OK(hipblasLtMatrixLayoutSetAttribute(c,HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,&batch,sizeof(batch)));
    OK(hipblasLtMatrixLayoutSetAttribute(a,HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&sa,sizeof(sa)));
    OK(hipblasLtMatrixLayoutSetAttribute(b,HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&sb,sizeof(sb)));
    OK(hipblasLtMatrixLayoutSetAttribute(c,HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,&sc,sizeof(sc)));
    hipblasLtMatmulHeuristicResult_t out;
    int count=0;
    OK(hipblasLtMatmulAlgoGetHeuristic(handle,desc,a,b,c,c,pref,1,&out,&count));
    if(count)std::cout<<n<<'\t'<<hipblaslt_ext::getKernelNameFromAlgo(handle,out.algo)<<'\t'<<out.workspaceSize<<'\n';
    else std::cout<<n<<"\tUNSUPPORTED\t0\n";
    hipblasLtMatrixLayoutDestroy(a);hipblasLtMatrixLayoutDestroy(b);hipblasLtMatrixLayoutDestroy(c);
  }
  hipblasLtMatmulPreferenceDestroy(pref);hipblasLtMatmulDescDestroy(desc);hipblasLtDestroy(handle);
}
