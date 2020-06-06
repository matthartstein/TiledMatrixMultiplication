/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/

    // INSERT KERNEL CODE HERE
    
    // Create two new tiles for Matrix A and Matrix B at shared memory
    __shared__ float aTile[TILE_SIZE][TILE_SIZE];
    __shared__ float bTile[TILE_SIZE][TILE_SIZE];
    
    // Initialize variables
    float temp = 0.0;
    int xBlock = blockIdx.x;
    int yBlock = blockIdx.y;
    int xThread = threadIdx.x;
    int yThread = threadIdx.y;
    int row = yBlock * TILE_SIZE + yThread;
    int col = xBlock * TILE_SIZE + xThread;
    
    // Perform calculations
    for(int i = 0; i < ((k + TILE_SIZE - 1) / TILE_SIZE); i++) {
    
        // A Tile
        if(xThread + (i * TILE_SIZE) < k && row < m) {
            aTile[yThread][xThread] = A[(row * k) + xThread + (i * TILE_SIZE)];
        }
        else {
            aTile[yThread][xThread] = 0.0f; // Set to zero
        }
    
        // B Tile
        if(yThread + (i * TILE_SIZE) < k && col < n) {
            bTile[yThread][xThread] = B[(i * TILE_SIZE + yThread) * n + col];
        }
        else {
            bTile[yThread][xThread] = 0.0f; // Set to zero
        }
        
        __syncthreads();    // Synchronization point; force threads to wait
        
        // Update temp value
        for(int z = 0; z < TILE_SIZE; z++) {
            temp += aTile[yThread][z] * bTile[z][xThread];
        }
        
        __syncthreads();    // Sychronization point; force threads to wait
    }
    
    // Copy temp value to C Matrix
    if(row < m && col < n) {
        C[row * n + col] = temp;
    }
}

void basicSgemm(char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc)
{
    if ((transa != 'N') && (transa != 'n')) {
	printf("unsupported value of 'transa'\n");
    	return;
    }

    if ((transb != 'N') && (transb != 'n')) {
	printf("unsupported value of 'transb'\n");
	return;
    }

    if ((alpha - 1.0f > 1e-10) || (alpha - 1.0f < -1e-10)) {
	printf("unsupported value of alpha\n");
	return;
    }

    if ((beta - 0.0f > 1e-10) || (beta - 0.0f < -1e-10)) {
	printf("unsupported value of beta\n");
	return;
    }

    // Initialize thread block and kernel grid dimensions ---------------------

    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE
    dim3 dim_grid((n / BLOCK_SIZE) + 1, (m / BLOCK_SIZE) + 1);
    dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
    mysgemm <<< dim_grid, dim_block >>> (m, n, k, A, B, C);
}