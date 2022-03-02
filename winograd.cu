#include<stdio.h>
// implementation for F(4 \times 4,3 \times 3)
const int m = 4;
const int r = 3;
const int alpha = m + r -1;

const int N = 1;
const int C = 3;
const int H = 128;
const int W = 128;
const int K = 16;
const int kH = r;
const int kW = r;
const int newH = H - kH + 1;
const int newW = W - kW + 1;

const int BLOCK_PER_ROW = W / m + (W%m!=0) ;
const int BLOCK_PER_COLUMN = H / W + (H%W!=0);
const int P = BLOCK_PER_COLUMN * BLOCK_PER_ROW;

//g[K,C,kH,kW]
//U[K,C,6,6]
//grid [K,ceil(C/32)]
//block [32]
__constant__ float G[6][3] = {1.0/4,0,0,-1.0/6,-1.0/6,-1.0/6,-1.0/6,1.0/6,-1.0/6,1.0/24,1.0/12,1.0/6,1.0/24,-1.0/12,1.0/6,0,0,1};
__global__ void generateU(const float* g,float* U)
{
    int k = blockIdx.x;
    int c = blockIdx.y * 32 + threadIdx.x;

    if( c < C)
    {
        g+=k*(C*kH*kW) + c*(kH*kW);
        U+=k*(C*36)+c*(36);
        float buf[6][3]={0};

        #pragma unroll(6)
        for(int i=0;i<6;i++)
            for(int j=0;j<3;j++)
                buf[i][j] = G[i][0] * g[0*3+j] + G[i][1] * g[1*3+j] + G[i][2] * g[2*3+j];
        
        #pragma unroll(6)
        for(int i=0;i<6;i++)
            for(int j=0;j<6;j++)    
                U[i*6+j] = buf[i][0] * G[j][0] + buf[i][1] * G[j][1] + buf[i][2] * G[j][2];
    }
}

//d[N,C,H,W] 
//V[N,C,BLOCK_PER_COLUMN,BLOCK_PER_ROW,alpha,alpha] -> V[N,C,P,alpha,alpha]
//grid [N,C,ceil(P/32)]
//block [32]
__constant__ float Bt[6][6] ={  {4,0,-5,0,1,0},{0,-4,-4,1,1,0},
                                {0,4,-4,-1,1,0},{0,-2,-1,2,1,0},
                                {0,2,-1,-2,1,0},{0,4,0,-5,0,1} 
                            };
__global__ void generateV(const float* d,float* V)
{
    int n = blockIdx.x;
    int c = blockIdx.y;
    int p = blockIdx.z*blockDim.x + threadIdx.x;
    int h = (p/BLOCK_PER_ROW)*m;
    int w = (p%BLOCK_PER_ROW)*m;
    if(p<P){
        float buf[6][6]={0};
        d+=n*(C*H*W) + c*(H*W) + h*W + w;
        V+=n*(C*P*alpha*alpha) + c *(P*alpha*alpha) + p*(alpha*alpha);

        for(int i=0;i<6;i++)
            for(int j=0;j<6;j++)
                for(int k=0;k<alpha;k++)
                    if(h+k<H&&w+j<W)
                        buf[i][j] += Bt[i][k] * d[k*W+j];
        
        for(int i=0;i<alpha;i++)
            for(int j=0;j<alpha;j++)
            {
                V[i*alpha+j] = 0;
                for(int k=0;k<6;k++)
                    V[i*alpha+j]+=buf[i][k] * Bt[j][k];
            }
    }
}

//U[K,C,alpha,alpha]
//V[N,C,P,alpha,alpha]
//M[N,K,P,alpha,alpha]
//grid[N,K,P]
//block[alpha*alpha]
__global__ void generateM(const float*U,const float*V,float*M)
{
    int n = blockIdx.x;
    int k = blockIdx.y;
    int p = blockIdx.z;
    int x = threadIdx.x;

    U+=k*C*alpha*alpha;
    V+=n*C*P*alpha*alpha+p*alpha*alpha;
    M+=n*(K*P*alpha*alpha)+k*(P*alpha*alpha)+p*(alpha*alpha);
    float sum = 0.0;
    for(int c = 0;c < C;c++){
        sum+=(U[c*(alpha*alpha)+x]*V[c*(P*alpha*alpha)+x]);
    }
    M[x] = sum;
}

__constant__ float At[m][alpha] = {
    {1,1,1,1,1,0},
    {0,1,-1,2,-2,0},
    {0,1,1,4,4,0},
    {0,1,-1,8,-8,1}
};

//M[N,K,P,alpha,alpha]
//Y[N,K,P,m,m]
//grid[N,K,ceil(P/32)]
//block[32]
__global__ void generateY(const float*M,float* Y)
{
    int n = blockIdx.x;
    int k = blockIdx.y;
    int p = blockIdx.z * blockDim.x + threadIdx.x;

    M+=n*(K*P*alpha*alpha) + k * (P*alpha*alpha) + p * (alpha*alpha);
    Y+=n*(K*P*m*m) + k*(P*m*m) + p*(m*m);

    float buf[m][alpha]={0};
    for(int i=0;i<m;i++)
        for(int j=0;j<alpha;j++)
            for(int k=0;k<6;k++)
                buf[i][j]+=At[i][k]*M[k*alpha+j];
    for(int i=0;i<m;i++)
        for(int j=0;j<m;j++)
        {
            Y[i*m+j]=0;
            for(int k=0;k<alpha;k++)
                Y[i*m+j]+=buf[i][k]*At[j][k];
        }
}

//tmpY[N,K,P,m,m] -> tmpY[N,K,BLOCK_PERCOLUMN,BLOCK_PERROW,m,m]
//Y[N,K,newH,newW]
//Y[n,k,h,w] = 
//grid[N,K,P]
//block[32]
__global__ void transpose(float* tmpY,float* Y)
{
    int n = blockIdx.x;
    int k = blockIdx.y;
    int p = blockIdx.z;
    int x = threadIdx.x;
    
    int h_base = p/BLOCK_PER_ROW*m;
    int w_base = p%BLOCK_PER_ROW*m; 
    
    int h_offset = x/m;
    int w_offset = x%m;
    Y+=n*(K*newH*newW);
    tmpY+=n*(K*P*m*m) + k*(P*m*m) + p*(m*m);
    if(x<m*m)
    {
        Y[(h_base+h_offset)*W+(w_base+w_offset)] = tmpY[m];
    }
}


void winograd(const float* d,const float* g,float* Y)
{
    float* U;
    cudaMalloc(&U,K*C*6*6*sizeof(float));
    generateU<<<dim3(K,ceil( (double)C/32 )),dim3(32)>>>(g,U);

    float* V;
    cudaMalloc(&V,N*C*P*alpha*alpha*sizeof(float));
    generateV<<<dim3(N,C,ceil((double)P/32 ) ),dim3(32)>>>(d,V);

    float* M;
    cudaMalloc(&M,N*K*P*alpha*alpha*sizeof(float));
    generateM<<<dim3(N,K,P),dim3(alpha*alpha)>>>(U,V,M);

    float* tmpY;
    cudaMalloc(&tmpY,N*K*P*m*m*sizeof(float));
    generateY<<<dim3(N,K,ceil((double)P/32)),dim3(32)>>>(M,tmpY);
   
    transpose<<<dim3(N,K,P),dim3(32)>>>(tmpY,Y);
    
    cudaFree(tmpY);
    cudaFree(M);
    cudaFree(V);
    cudaFree(U);
}

int main()
{
    float* g = new float[K*C*kH*kW];
    float* d = new float[N*C*H*W];
    float* Y = new float[N*K*newH*newW];

    for(int i=0;i<K*C*kH*kW;i++)
        g[i] = 1.0;
    for(int i=0;i<N*C*H*W;i++)
        d[i] = 1.0;


    float* gCuda;
    float* dCuda;
    float* YCuda;
    cudaMalloc(&gCuda,K*C*kH*kW*sizeof(float) );
    cudaMalloc(&dCuda,N*C*H*W*sizeof(float) );
    cudaMalloc(&YCuda,N*K*newH*newW*sizeof(float) );

    cudaMemcpy(gCuda,g,K*C*kH*kW*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(dCuda,d,N*C*H*W*sizeof(float),cudaMemcpyHostToDevice);

    winograd(dCuda,gCuda,YCuda);

    cudaMemcpy(Y,YCuda,sizeof(float)*N*K*newH*newW,cudaMemcpyDeviceToHost);

    for(int i=0;i<newW;i++)
        printf("%f ",Y[i]);

    cudaFree(YCuda);
    cudaFree(dCuda);
    cudaFree(gCuda);

    delete [] Y;
    delete [] d;
    delete [] g;

    return 0;
}