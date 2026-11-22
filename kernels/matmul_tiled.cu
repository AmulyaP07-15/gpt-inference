#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#define TILE 16

__global__ void matmul_tiled_kernel(const float*  A, const float* B, float* C ,int M, int K,int N )
{
//A[M*K]:input activations,one row per token
//B[K*N]:weights one column per one output feature
//C[M*N]:output  matrix,one row per token

	//

//statically allocated shared memory array
// M = number of token rows (1 or 500 in our benchmark).
// K = number of input features per token (768 for GPT-2 c_fc).
// N = number of output features per token (3072 for GPT-2 c_fc).

// TILE: 16 slots per side of each shared tile.

// tileA: 16 token rows x 16 input features from A.
// tileB: 16 input features x 16 output columns from B.
// The whole block shares these two tiles and refills them each round.

// row: this thread's row in the full output C, also its token row in A.
// col: this thread's column in the full output C, also its weight column in B.
// sum: this thread's private running total for C[row][col].
// Start sum at zero once, before the outer loop, and keep it across rounds.

__shared__ float  tileA[TILE][TILE];
__shared__ float tileB[TILE][TILE];

int  row =blockIdx.y*blockDim.y+threadIdx.y;
int col=blockIdx.x*blockDim.x+threadIdx.x;

//give each thread its own rumnning total
float sum =0.0f;
// kStart: first input feature of this chunk in the full matrices (0, 16, 32, ...).
// aCol: column to read from A. Features go across A, so use kStart + threadIdx.x.
// bRow: row to read from B. Features go down B, so use kStart + threadIdx.y.
// k: feature position inside the shared tiles (0 to 15). It restarts each chunk.

for (int kStart=0;kStart<K;kStart+=TILE){//  outer:48 rounds 
	
	int aCol=kStart+threadIdx.x;
if(row<M && aCol<K){
	tileA[threadIdx.y][threadIdx.x]=A[row*K+aCol];
}
else{
tileA[threadIdx.y][threadIdx.x]=0.0f;
}//which element is being copied from the A to the shared memory 

	int bRow=threadIdx.y+kStart;
	if (bRow<K && col<N)
	{
	tileB[threadIdx.y][threadIdx.x]=B[bRow*N+col];
	}
	else{
		tileB[threadIdx.y][threadIdx.x]=0.0f;}

	__syncthreads();

	for(int k=0;k<TILE;k++){
	sum+=tileA[threadIdx.y][k]*tileB[k][threadIdx.x];
	}
	__syncthreads();

}
if(row<M && col< N)
{
C[row*N+col]=sum;
}
}
torch::Tensor matmul_tiled_cuda(torch::Tensor A,torch:: Tensor B){
	A=A.contiguous();
	B=B.contiguous();

	int M=A.size(0);
	int K=A.size(1);
	int N=B.size(1);

	auto C=torch::empty({M,N},A.options());

	dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matmul_tiled_kernel<<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(),
        M, K, N);
	return C;}
// Set up the Python module using the name PyTorch gives the extension.
// m is the module we add functions to.
 // When Python calls matmul_tiled(A, B), run the C++ launcher.
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
	m.def("matmul_tiled",&matmul_tiled_cuda);

}

