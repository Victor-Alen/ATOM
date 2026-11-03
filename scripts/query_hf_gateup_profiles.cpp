#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include <iostream>
#include <cstdlib>
#define OK(x) do{auto st=(x);if(st!=HIPBLAS_STATUS_SUCCESS){std::cerr<<#x<<" status "<<st<<"\n";return 1;}}while(0)
int main(int argc,char**argv){
 int maxm=argc>1?atoi(argv[1]):32768;size_t workspace=argc>2?strtoull(argv[2],nullptr,10):33554432;
 hipblasLtHandle_t h;OK(hipblasLtCreate(&h));hipblasLtMatmulDesc_t desc;OK(hipblasLtMatmulDescCreate(&desc,HIPBLAS_COMPUTE_32F,HIP_R_32F));
 hipblasOperation_t ta=HIPBLAS_OP_T,tb=HIPBLAS_OP_N;OK(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)));OK(hipblasLtMatmulDescSetAttribute(desc,HIPBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)));
 hipblasLtMatrixLayout_t a;OK(hipblasLtMatrixLayoutCreate(&a,HIP_R_16BF,2304,1792,2304));hipblasLtMatmulPreference_t pref;OK(hipblasLtMatmulPreferenceCreate(&pref));OK(hipblasLtMatmulPreferenceSetAttribute(pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace,sizeof(workspace)));
 for(int m=1;m<=maxm;m++){
  hipblasLtMatrixLayout_t b,c;OK(hipblasLtMatrixLayoutCreate(&b,HIP_R_16BF,2304,m,2304));OK(hipblasLtMatrixLayoutCreate(&c,HIP_R_16BF,1792,m,1792));hipblasLtMatmulHeuristicResult_t out;int n=0;OK(hipblasLtMatmulAlgoGetHeuristic(h,desc,a,b,c,c,pref,1,&out,&n));if(!n){std::cerr<<"No heuristic "<<m<<"\n";return 2;}
  std::cout<<m<<"\t"<<hipblaslt_ext::getKernelNameFromAlgo(h,out.algo)<<"\t"<<out.workspaceSize<<"\n";hipblasLtMatrixLayoutDestroy(b);hipblasLtMatrixLayoutDestroy(c);
 }
 hipblasLtMatmulPreferenceDestroy(pref);hipblasLtMatrixLayoutDestroy(a);hipblasLtMatmulDescDestroy(desc);hipblasLtDestroy(h);
}
