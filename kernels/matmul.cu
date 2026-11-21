#include <torch/extension.h>

// A: [M, K]  B: [K, N]  C: [M, N]   (M = 4, K = 768, N = 3072)
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// Runs on the CPU: takes PyTorch tensors, launches the kernel, returns C

torch::Tensor matmul_cuda(torch::Tensor A,torch::Tensor B){
A=A.contiguous();//checks if the rows are stored one after another
B=B.contiguous();

int M=A.size(0);//num of tokens
int K=A.size(1);//768 the input features
int N=B.size(1);//3072 output features

//Empty output on the same GPU,same dtype as A
torch::Tensor C=torch::empty({M,N},A.options());

//Launch Config
dim3 threads(16,16);//x=16 cols ,y=16 rows per block
dim3 blocks((N+15)/16,(M+15)/16);//round up 192 blocks across ,1 down

matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

return C;

}

// Lets Python call matmul_cuda as ext.matmul(A, B)
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
m.def("matmul", &matmul_cuda, "Naive matmul (CUDA)");
}
