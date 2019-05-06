#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <sys/stat.h>

#include "../common/GL/glew.h"
//#include <GL/glew.h>
//#include <GL/glaux.h>

#include "cuda.h"
#include "cutil.h"
#include "cuda_gl_interop.h"

#include "..\GLKLib\GLK.h"

#include "PMBody.h"
#include "LDNIcpuSolid.h"
#include "LDNIcudaSolid.h"
#include "LDNIcudaOperation.h"

#include <thrust/version.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <iostream>

//--------------------------------------------------------------------------------------------
extern __global__ void LDNIDistanceField_CountBitInInteger(unsigned int *d_index, int nodeNum, int res);
extern __global__ void LDNIDistanceField_CountBitInArray(unsigned int *d_index, unsigned int *m_3dArray, int nodeNum, int res);
extern __global__ void LDNIDistanceField__writeTexToVBO(float3 *d_output, int res, int* table_index, float width, float3 origin, int nodeNum);
extern __global__ void LDNIDistanceField__writeTexToArray(unsigned short *d_output, int res, unsigned int *table_index, unsigned int* temp_index,  int nodeNum);
extern __global__ void LDNIDistanceField__writeArrayToVBO(float3 *d_output, int res, unsigned int* table_index, unsigned int *m_3dArray, float width, float3 origin, int nodeNum);
extern __global__ void LDNIDistanceField__Sort2DArray(unsigned short *d_output, unsigned int *d_index, int res, int nodeNum);
extern __global__ void LDNIDistanceField__GenerateProbablySiteInYByGivenDistance(unsigned int *bitSites, unsigned short *sites, unsigned int *site_index, int res, int offsetPixel, int nodeNum);
extern __global__ void LDNIDistanceField__FilterProbablySiteInYByGivenDistance(unsigned int *bitSites, unsigned short *sites, unsigned int *sites_index, int res, int offsetPixel, int nodeNum);
extern __global__ void LDNIDistanceField__GenerateProbablySiteInXByGivenDistance(unsigned int *bitSites, unsigned short *sites, unsigned int *site_index, int res, int offsetPixel, int nodeNum);
extern __global__ void LDNIDistanceField__FilterProbablySiteInXByGivenDistance(unsigned int *bitSites, unsigned short *sites, unsigned int *sites_index, int res, int offsetPixel, int nodeNum);


extern __global__ void LDNIDistanceField__GenerateProbablySiteInX(unsigned int *bitDeleted, unsigned int *bitForNextLoop, unsigned int *counter, ushort2 *sites, unsigned int *sites_index, int res, int offsetPixel, int nodeNum);
extern __global__ void LDNIDistanceField__GenerateProbablySiteInY(unsigned int *bitDeleted, unsigned int *bitForNextLoop, unsigned int *counter, unsigned short *sites, unsigned int *sites_index, int res, int offsetPixel, int nodeNum);
extern __global__ void LDNIDistanceField__GenerateProbablySiteInXLoop(unsigned int *bitDeleted, unsigned int *bitForNextLoop, unsigned int *counter, ushort2 *sites, unsigned int *sites_index, int res, int offsetPixel, int nodeNum, short loopID);
extern __global__ void LDNIDistanceField__GenerateProbablySiteInYLoop(unsigned int *bitDeleted, unsigned int *bitForNextLoop, unsigned int *counter, unsigned short *sites, unsigned int *sites_index, int res, int offsetPixel, int nodeNum, short loopID);
extern __global__ void LDNIDistanceField__CountProbablySiteInY(unsigned int *bitDeleted, unsigned int *counter, int res, int nodeNum);

extern __global__ void LDNIDistanceField__SortProbablySite(unsigned int *sites, unsigned int *sites_index, int res, int nodeNum);
extern __global__ void LDNIDistanceField__SortProbablySite2(unsigned int *sites, unsigned int *sites_index, int res, int nodeNum);
extern __global__ void LDNIDistanceField__GetSiteByDist(ushort3 *sites, unsigned int *counter, unsigned int *sites_index, unsigned int *sites_off, int offdist, int res, int nodeNum);

extern __global__ void LDNIDistanceField__writeSitesToVBO(float3 *d_output, int res, unsigned int *counter, unsigned int* d_input, float width, float3 origin, int nodeNum);
extern __global__ void LDNIDistanceField__Test(float3 *d_output, int res, unsigned int *counter, ushort2 *site, unsigned int* site_index, float width, float3 origin, int nodeNum);

extern __global__ void LDNIDistanceField__GetProbablySiteInY(unsigned int *bitDeleted, unsigned int* counter, unsigned int *sites, unsigned int *sites_index, unsigned short *sites_x, unsigned int *sites_index_x, int3 res, int nodeNum);
extern __global__ void LDNIDistanceField__GetProbablySiteInX(unsigned int *bitDeleted, unsigned int* counter, unsigned int *sites, unsigned int *sites_index, unsigned int *sites_in, unsigned int *sites_index_in, int3 res, int nodeNum);

extern __global__ void LDNIDistanceField__MaurerAxisInY(unsigned int *bitDeleted, unsigned short *sites, unsigned int *sites_index, int3 res, int offsetPixel, int nodeNum);
extern __global__ void LDNIDistanceField__MaurerAxisInX(unsigned int *bitDeleted, unsigned int *sites, unsigned int *sites_index, int3 res, int offsetPixel, int nodeNum);
extern __global__ void LDNIDistanceField__kernelMergeBandsY_2(unsigned int *bitDeleted, unsigned short *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum);
extern __global__ void LDNIDistanceField__kernelMergeBandsY_4(unsigned int *bitDeleted, unsigned short *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum);
extern __global__ void LDNIDistanceField__kernelMergeBandsY_8(unsigned int *bitDeleted, unsigned short *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum);
extern __global__ void LDNIDistanceField__kernelMergeBandsY_16(unsigned int *bitDeleted, unsigned short *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum);
extern __global__ void LDNIDistanceField__kernelMergeBandsY_32(unsigned int *bitDeleted, unsigned short *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum);

extern __global__ void LDNIDistanceField__kernelMergeBandsX_2(unsigned int *bitDeleted, unsigned int *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum);
extern __global__ void LDNIDistanceField__kernelMergeBandsX_4(unsigned int *bitDeleted, unsigned int *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum);
extern __global__ void LDNIDistanceField__kernelMergeBandsX_8(unsigned int *bitDeleted, unsigned int *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum);
extern __global__ void LDNIDistanceField__kernelMergeBandsX_16(unsigned int *bitDeleted, unsigned int *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum);
extern __global__ void LDNIDistanceField__kernelMergeBandsX_32(unsigned int *bitDeleted, unsigned int *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum);

extern __global__ void LDNIDistanceField__countArrayToVBO(int3 res, unsigned int* counter, unsigned int *sites, unsigned int *sites_index, int offdist, int nodeNum);
extern __global__ void LDNIDistanceField__writeResultToVBO(float3 *d_output, int3 res, unsigned int* counter, unsigned int *sites, unsigned int *sites_index, int offdist, float width, float3 origin, int nodeNum);
//-----------------------------PBA Distance Field---------------------------------------------------------------

extern __global__ void PBADistanceField__writeTexToArray(int *d_output, int res, int nodeNum, unsigned int* counter);
extern __global__ void PBADistanceField_kernelFloodZ(int *output, int size, int mod, int bandSize);
extern __global__ void PBADistanceField_kernelPropagateInterband(int *output, int size, int mod, int bandSize);
extern __global__ void PBADistanceField_kernelUpdateVertical(int *output, int size, int mod, int bandSize);
extern __global__ void PBADistanceField_kernelMaurerAxis(int *stack, int size, int mod, int bandSize, int test);
extern __global__ void PBADistanceField_kernelMergeBands(int *stack, int *forward, int size, int mod, int bandSize);
extern __global__ void PBADistanceField_kernelCreateForwardPointers(int *output, int size, int mod, int bandSize); 
extern __global__ void PBADistanceField_kernelColorAxis(int *output, int size); 
extern __global__ void PBADistanceField_kernelTransposeXY(int *data, int log2Width, int mask);
extern __global__ void PBADistanceField__writeArrayToVBO(float3 *d_output, int res, unsigned int* counter, int *outputDF, int offdist, float width, float3 origin, int nodeNum);
extern __global__ void PBADistanceField__countArrayToVBO(int res, unsigned int* counter, int *outputDF, int offdist, int nodeNum);
extern __global__ void PBADistanceField__writeCompactArray(int *d_output, int *d_input, unsigned int *counter, int nodeNum);
//--------------------------------------------------------------------------------------------

extern __device__ unsigned int bitCount(unsigned int i);
extern __device__ unsigned int GetFirstBitPos(unsigned int source);
extern __device__ unsigned int GetLastBitPos(unsigned int source);
extern __device__ unsigned int SetBitPos(unsigned int pos);
extern __device__ float interpointY(int x1, int y1, int z1, int x2, int y2, int z2, int x0, int z0);
extern __device__ bool GetBitPos(unsigned int pos, unsigned int source);
extern __device__ unsigned int Reversebit(unsigned int v);
extern __device__ int middlepointY(unsigned int site1, unsigned int site2, int z0);
extern __device__ int middlepointX(unsigned int site1, unsigned int site2, int y0, int z0);

//texture<unsigned int> site_tex;
texture<uint4,3> site_tex;

#define BANDWIDTH 32
#define MAX_INT 	201326592
#define PBAMARKER -1
#define INFINITY    0x3ff
#define TOID(x, y, z, w)    (__mul24(__mul24(z, w) + (y), w) + (x))
#define TOID_CPU(x, y, z, w)    ((z) * (w) * (w) + (y) * (w) + (x))
#define ENCODE(x, y, z)  (((x) << 20) | ((y) << 10) | (z))
#define DECODE(value, x, y, z) \
	x = (value) >> 20; \
	y = ((value) >> 10) & 0x3ff; \
	z = (value) & 0x3ff

#define GET_X(value)	((value) >> 20)
#define GET_Y(value)	(((value) >> 10) & 0x3ff)
#define GET_Z(value)	(((value) == PBAMARKER) ? MAX_INT : ((value) & 0x3ff))

#define ROTATEXY(x)   ((((x) & 0xffc00) << 10) | \
	(((x) & 0x3ff00000) >> 10) | \
	((x) & 0x3ff))

#define BLOCKX      32
#define BLOCKY      4
#define BLOCKXY     16

#define GET_STACK(value)	((value >> 16) & 0xffff)
#define GET_PTR(value)	((value) & 0xffff)
#define ENCODE_STACK(a, b)   (((a) << 16) | (b & 0xffff))
#define ENCODE_STACK_3(a, b, c)  (((a) << 20) | ((b) << 10) | (c & 0x3ff))
#define ENCODE_PTR(value, b)   ((value & 0xffff0000) | (b & 0xffff))
#define ENCODE_Z(value, z)  ((value & 0xfffffC00) | (z & 0x3ff))

texture<int> pbaTexColor; 
texture<int> pbaTexLinks; 
//texture<short> pbaTexPointer; 
texture<int> pbaTexPointer; 

void LDNIcudaOperation::PBADistanceFieldGeneration(QuadTrglMesh *mesh, GLuint *vbo, unsigned int &vbosize, int res, int offdist, float boundingBox[])
{
	if (res > 512) return;

	int fboSize = res;
	int nVertices;

	int phase1Band  = 16; 
	int phase2Band	= 16; 
	int phase3Band	= 2; 

	
	int **pbaTextures;   
	int pbaMemSize; 
	int pbaCurrentBuffer; 
	int pbaTexSize;  


	pbaTextures = (int **) malloc(2 * sizeof(int *)); 
	pbaTexSize = fboSize;
	pbaMemSize = pbaTexSize * pbaTexSize * pbaTexSize * sizeof(int); 

	CUDA_SAFE_CALL(cudaMalloc((void **) &pbaTextures[0], pbaMemSize)); 
	CUDA_SAFE_CALL(cudaMalloc((void **) &pbaTextures[1], pbaMemSize)); 



	
	// PBA initialization
	if (!PBADistanceField_SitesGeneration(mesh, vbo, nVertices, res, boundingBox, pbaTextures[0]))
		return;
	pbaCurrentBuffer = 0;

	// Read sites to CPU
	int *sites;
	printf("Start %d \n", nVertices);
	unsigned int* counter;
	CUDA_SAFE_CALL(cudaMalloc((void**) &counter, sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset( counter, 0, sizeof(unsigned int)) );
	CUDA_SAFE_CALL(cudaMalloc((void**) &sites, nVertices*sizeof(int))); 
	CUDA_SAFE_CALL(cudaMemset( sites, 0, nVertices*sizeof(int)) );
	PBADistanceField__writeCompactArray<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(sites, pbaTextures[0], counter, res*res*res);

	int* cpu_sites = (int*)malloc(nVertices*sizeof(int));
	CUDA_SAFE_CALL(cudaMemcpy( cpu_sites, sites, nVertices*sizeof(int),cudaMemcpyDeviceToHost));

	printf("End\n");
	
	

	// Compute the 3D distance field

	/************* Compute Z axis *************/
	// --> (X, Y, Z)   
	pbaCurrentBuffer = PBADistanceField_pba3DColorZAxis(pbaTextures, res, phase1Band, pbaCurrentBuffer);

	
	/************* Compute Y axis *************/
	// --> (X, Y, Z)
	pbaCurrentBuffer = PBADistanceField_pba3DComputeProximatePointsYAxis(pbaTextures, res, phase2Band, pbaCurrentBuffer, 0);
	pbaCurrentBuffer = PBADistanceField_pba3DColorYAxis(pbaTextures, res, phase3Band, pbaCurrentBuffer);
	// --> (Y, X, Z)
	PBADistanceField_pba3DTransposeXY(pbaTextures[pbaCurrentBuffer], res);
	
	cudaThreadSynchronize();
	printf("starting X ==================================\n");

	/************** Compute X axis *************/
	// Compute X axis
	pbaCurrentBuffer = PBADistanceField_pba3DComputeProximatePointsYAxis(pbaTextures, res, phase2Band, pbaCurrentBuffer, 1);
	pbaCurrentBuffer = PBADistanceField_pba3DColorYAxis(pbaTextures, res, phase3Band, pbaCurrentBuffer);
	
	// --> (Y, X, Z)
	PBADistanceField_pba3DTransposeXY(pbaTextures[pbaCurrentBuffer], res);

	
	cudaFree(sites);
	cudaFree(pbaTextures[1-pbaCurrentBuffer]);

	char inputStr[10];
	printf("\Check Error (very slow)? (y/n): ");
	scanf("%s",inputStr);
	if (inputStr[0]=='y' || inputStr[0]=='Y')
	{
		PBADistanceField_CompareResult(pbaTextures[pbaCurrentBuffer], res, nVertices, cpu_sites);
	}
	free(cpu_sites);

	// Generate Offset & display
	cudaGraphicsResource *resource;
	float gWidth=(boundingBox[1]-boundingBox[0])/(float)res;
	float width = gWidth*(float)res;
	float origin[3];
	origin[0]=boundingBox[0]+gWidth*0.5f;
	origin[1]=boundingBox[2]+gWidth*0.5f;
	origin[2]=boundingBox[4]+gWidth*0.5f;

	
	//unsigned int* counter;
	//CUDA_SAFE_CALL(cudaMalloc((void**) &counter, sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset( counter, 0, sizeof(unsigned int)) );

	PBADistanceField__countArrayToVBO<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(res, counter, pbaTextures[pbaCurrentBuffer], offdist, res*res*res);
	CUDA_SAFE_CALL(cudaMemcpy( &vbosize, counter, sizeof(unsigned int),cudaMemcpyDeviceToHost));
	
	printf("size ---------- %ld \n", vbosize);
	if (vbosize <= 0) 
	{
		printf("Error in PBA Distance Computation !!! \n");
		cudaFree(pbaTextures[0]);
		cudaFree(pbaTextures[1]);
		cudaFree(counter);
		free(pbaTextures);
		return;
	}

	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, vbosize*3*sizeof(float), 0, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&resource, *vbo, cudaGraphicsRegisterFlagsWriteDiscard));


	CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &resource, 0));
	size_t num_bytes; 
	float3 *dptr;
	CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, resource));
	CUDA_SAFE_CALL(cudaMemset(  counter, 0, sizeof(unsigned int)) );


	PBADistanceField__writeArrayToVBO<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(dptr, res, counter, pbaTextures[pbaCurrentBuffer], offdist, width, make_float3(origin[0],origin[1],origin[2]), res*res*res);
	
	CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &resource, 0));
	printf("CUDA mapped VBO: VBO Size %ld bytes\n", vbosize);
	


	
	cudaFree(pbaTextures[pbaCurrentBuffer]);
	free(pbaTextures); 
	cudaFree(counter);


}

int LDNIcudaOperation::PBADistanceField_pba3DColorZAxis(int **pbaTextures, int res, int m1, int cbuffer)
{
	int pbaCurrentBuffer = cbuffer;

	dim3 block = dim3(BLOCKX, BLOCKY); 
	dim3 grid = dim3((res / block.x) * m1, res / block.y); 

	cudaBindTexture(0, pbaTexColor, pbaTextures[pbaCurrentBuffer]); 
	PBADistanceField_kernelFloodZ<<< grid, block >>>(pbaTextures[1 - pbaCurrentBuffer], res, res / block.x, res / m1); 
	pbaCurrentBuffer = 1 - pbaCurrentBuffer; 

	if (m1 > 1) {
		// Passing information between bands
		cudaBindTexture(0, pbaTexColor, pbaTextures[pbaCurrentBuffer]); 
		PBADistanceField_kernelPropagateInterband<<< grid, block >>>(pbaTextures[1 - pbaCurrentBuffer], res, res / block.x, res / m1); 

		cudaBindTexture(0, pbaTexLinks, pbaTextures[1 - pbaCurrentBuffer]); 
		PBADistanceField_kernelUpdateVertical<<< grid, block >>>(pbaTextures[pbaCurrentBuffer], res, res / block.x, res / m1); 
	}
	return pbaCurrentBuffer;
}

int LDNIcudaOperation::PBADistanceField_pba3DComputeProximatePointsYAxis(int **pbaTextures, int res, int m2, int cbuffer, int test)
{
	int pbaCurrentBuffer = cbuffer;

	int iStack = 1 - pbaCurrentBuffer;     
	int iForward = pbaCurrentBuffer;   

	dim3 block = dim3(BLOCKX, BLOCKY); 
	dim3 grid = dim3((res / block.x) * m2, res / block.y); 

	 //printf("forward %d %d \n",iStack, iForward);

	// Compute proximate points locally in each band
	
	cudaBindTexture(0, pbaTexColor, pbaTextures[pbaCurrentBuffer]); 
	PBADistanceField_kernelMaurerAxis<<< grid, block >>>(pbaTextures[iStack], res, res / block.x, res / m2, test); 
	//cudaThreadSynchronize();


	// Construct forward pointers
	cudaBindTexture(0, pbaTexLinks, pbaTextures[iStack]); 
	PBADistanceField_kernelCreateForwardPointers<<< grid, block >>>(pbaTextures[iForward], res, res / block.x, res / m2); 

	//
	cudaBindTexture(0, pbaTexPointer, pbaTextures[iForward]); 

	
	// Repeatly merging two bands into one
	for (int noBand = m2; noBand > 1; noBand /= 2) {
		grid = dim3((res / block.x) * (noBand / 2), res / block.y); 
		
		PBADistanceField_kernelMergeBands<<< grid, block >>>(pbaTextures[iStack], 
			 pbaTextures[iForward], res, res / block.x, res / noBand); 
		
		//printf("test %d %d %d %d\n", iForward, iStack, m2);
	   //break;
	}    
	

	cudaUnbindTexture(pbaTexLinks); 
	cudaUnbindTexture(pbaTexColor); 
	cudaUnbindTexture(pbaTexPointer); 

	return pbaCurrentBuffer;
}

int LDNIcudaOperation::PBADistanceField_pba3DColorYAxis(int **pbaTextures, int res, int m3, int cbuffer)
{
	int pbaCurrentBuffer = cbuffer;
	dim3 block = dim3(BLOCKX, m3);    
	dim3 grid = dim3(res / block.x, res);  

	cudaBindTexture(0, pbaTexColor, pbaTextures[1 - pbaCurrentBuffer]); 
	PBADistanceField_kernelColorAxis<<< grid, block >>>(pbaTextures[pbaCurrentBuffer], res); 
	cudaUnbindTexture(pbaTexColor); 

	return pbaCurrentBuffer;
}

void LDNIcudaOperation::PBADistanceField_pba3DTransposeXY(int *&inputDF, int res)
{
	dim3 block(BLOCKXY, BLOCKXY); 
	dim3 grid((res / BLOCKXY) * res, res / BLOCKXY); 

	int log2Width; 
	int tmp = res; 
	log2Width = 0; 
	while (tmp > 1) { tmp /= 2; log2Width++; }


	PBADistanceField_kernelTransposeXY<<< grid, block >>>(inputDF, log2Width, res - 1); 
}



void LDNIcudaOperation::DistanceFieldGeneration(QuadTrglMesh *mesh, GLuint *vbo, unsigned int &vbosize, int res, int offdist, float boundingBox[])
{

	int arrsize = res*res;
	unsigned int* sites_index;
	unsigned short *sites;
	int siteNum;
	LDNIDistanceField_SitesGeneration(mesh, vbo, siteNum, res, boundingBox, sites_index, sites);
	if (siteNum <= 0) 	
	{
		cudaFree(sites);
		cudaFree(sites_index);
		return ;
	}

	//check whether the sites on each ray are sorted (Just for in case, should be sorted during the writing kernel)
	//LDNIDistanceField__Sort2DArray<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(sites, sites_index, res, res*res);
	//LDNIDistanceField__GenerateProbablySiteInYByGivenDistance<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitSites, sites, sites_index, res, offdist,  res*res*res);
	//LDNIDistanceField__GenerateProbablySiteInXByGivenDistance<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitSites, sites, sites_index, res, offdist,  res*res*res);
	//LDNIDistanceField__FilterProbablySiteInYByGivenDistance<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitSites, sites, sites_index, res, offdist-1,  res*res*res);
	//LDNIDistanceField__FilterProbablySiteInXByGivenDistance<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitSites, sites, sites_index, res, offdist-1,  res*res*res);

	
	long time = clock();
	unsigned int* bitDeleted;
	int bitsize = res*res*(res/32);
	CUDA_SAFE_CALL(cudaMalloc((void**) & bitDeleted, bitsize*sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset( bitDeleted, 0, bitsize*sizeof(unsigned int)) );

	


	
	LDNIDistanceField__MaurerAxisInY<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, sites, sites_index, make_int3(res, res/BANDWIDTH, res), offdist, res*(res/BANDWIDTH)*res);
	
	cudaThreadSynchronize();

	if (res > 32)
	{
		LDNIDistanceField__kernelMergeBandsY_2<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, sites, sites_index, make_int3(res, res/2, res),  offdist, 2, 16, res*(res/2)*res);
		//printf("Y-32\n");
	}
	

	if (res > 64)
	{
		LDNIDistanceField__kernelMergeBandsY_4<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, sites, sites_index, make_int3(res, res/4, res),  offdist, 4, 8, res*(res/4)*res);
		//printf("Y-64\n");
	}
	

	if (res > 128){
		LDNIDistanceField__kernelMergeBandsY_8<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, sites, sites_index, make_int3(res, res/8, res),  offdist, 8, 4, res*(res/8)*res);
		//printf("Y-128\n");
	}

	if (res > 256)
	{
		LDNIDistanceField__kernelMergeBandsY_16<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, sites, sites_index, make_int3(res, res/16, res),  offdist, 16, 2, res*(res/16)*res);
		//printf("Y-256\n");
	}
	

	if (res > 512)
	{
		LDNIDistanceField__kernelMergeBandsY_32<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, sites, sites_index, make_int3(res, res/32, res),  offdist, 32, 1, res*(res/32)*res);
		//printf("Y-512\n");
	}
	

	
	cudaThreadSynchronize();
	printf("time 1 : %ld(ms) \n", clock()-time); time = clock();
	
	unsigned int* sites_index_y;
	unsigned int* numofBit = (unsigned int*)malloc(sizeof(unsigned int));
	CUDA_SAFE_CALL(cudaMalloc((void**) &sites_index_y, (arrsize+1)*sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset( sites_index_y, 0, (arrsize+1)*sizeof(unsigned int)));
	
	LDNIDistanceField__CountProbablySiteInY<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, sites_index_y, res, res*res);
		
	thrust::device_ptr<unsigned int> dev_ptr(sites_index_y); //	Wrap raw pointers with dev_ptr
	thrust::exclusive_scan(dev_ptr, dev_ptr+(arrsize+1), dev_ptr); //	in-place scan
	numofBit[0]=dev_ptr[arrsize];
	
	cudaThreadSynchronize();
	printf("time 2 : %ld(ms) \n", clock()-time); time = clock();
	printf("Get Sites in Y : %d \n", numofBit[0]);


	unsigned int* sites_y;
	unsigned int* temp2D;
	CUDA_SAFE_CALL(cudaMalloc((void**) &sites_y, numofBit[0]*sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset( sites_y, 0, numofBit[0]*sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMalloc((void**) &temp2D, (arrsize+1)*sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset( temp2D, 0, (arrsize+1)*sizeof(unsigned int)));

	LDNIDistanceField__GetProbablySiteInY<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, temp2D, sites_y, sites_index_y, sites, sites_index, make_int3(res, res/BANDWIDTH, res), res*(res/BANDWIDTH)*res);

	cudaFree(temp2D);
	cudaFree(sites_index);
	cudaFree(sites);

	CUDA_SAFE_CALL(cudaMemset( bitDeleted, 0, bitsize*sizeof(unsigned int)) );
	//LDNIDistanceField__SortProbablySite<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(sites_y, sites_index_y, res, res*res);

	printf("time 3 : %ld(ms) \n", clock()-time); time = clock();

	/* //for debugging
	thrust::device_ptr<unsigned int> dev_ptr2(temp2D); //	Wrap raw pointers with dev_ptr
	thrust::exclusive_scan(dev_ptr2, dev_ptr2+(arrsize+1), dev_ptr2); //	in-place scan
	numofBit[0]=dev_ptr2[arrsize];

	printf("Proved Sites in Y : %d \n", numofBit[0]);*/

	//-------------------------------X direction---------------------------------------//


	LDNIDistanceField__MaurerAxisInX<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, sites_y, sites_index_y, make_int3(res, res/BANDWIDTH, res), offdist, res*(res/BANDWIDTH)*res);
	cudaThreadSynchronize();

	
	printf("time 4 : %ld(ms) \n", clock()-time); time = clock();

	if (res > 32)
	LDNIDistanceField__kernelMergeBandsX_2<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, sites_y, sites_index_y, make_int3(res, res/2, res),  offdist, 2, 16, res*(res/2)*res);
	
	if (res > 64)
	LDNIDistanceField__kernelMergeBandsX_4<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, sites_y, sites_index_y, make_int3(res, res/4, res),  offdist, 4, 8, res*(res/4)*res);

	if (res > 128)
	LDNIDistanceField__kernelMergeBandsX_8<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, sites_y, sites_index_y, make_int3(res, res/8, res),  offdist, 8, 4, res*(res/8)*res);

	if (res > 256)
	LDNIDistanceField__kernelMergeBandsX_16<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, sites_y, sites_index_y, make_int3(res, res/16, res),  offdist, 16, 2, res*(res/16)*res);

	if (res > 512)
	LDNIDistanceField__kernelMergeBandsX_32<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, sites_y, sites_index_y, make_int3(res, res/32, res),  offdist, 32, 1, res*(res/32)*res);

	cudaThreadSynchronize();

	printf("time 5 : %ld(ms) \n", clock()-time); time = clock();

	unsigned int* sites_index_x;
	CUDA_SAFE_CALL(cudaMalloc((void**) &sites_index_x, (arrsize+1)*sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset( sites_index_x, 0, (arrsize+1)*sizeof(unsigned int)));

	LDNIDistanceField__CountProbablySiteInY<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, sites_index_x, res, res*res);

	thrust::device_ptr<unsigned int> dev_ptr2(sites_index_x); //	Wrap raw pointers with dev_ptr
	thrust::exclusive_scan(dev_ptr2, dev_ptr2+(arrsize+1), dev_ptr2); //	in-place scan
	numofBit[0]=dev_ptr2[arrsize];

	cudaThreadSynchronize();
	printf("time 6 : %ld(ms) \n", clock()-time); time = clock();
	printf("Get Sites in X : %d \n", numofBit[0]);

	unsigned int* sites_x;
	CUDA_SAFE_CALL(cudaMalloc((void**) &sites_x, numofBit[0]*sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset( sites_x, 0, numofBit[0]*sizeof(unsigned int)));

	CUDA_SAFE_CALL(cudaMalloc((void**) &temp2D, (arrsize+1)*sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset( temp2D, 0, (arrsize+1)*sizeof(unsigned int)));

	LDNIDistanceField__GetProbablySiteInX<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bitDeleted, temp2D, sites_x, sites_index_x, sites_y, sites_index_y, make_int3(res, res/BANDWIDTH, res), res*(res/BANDWIDTH)*res);


	 //for debugging
	/*thrust::device_ptr<unsigned int> dev_ptr3(temp2D); //	Wrap raw pointers with dev_ptr
	thrust::exclusive_scan(dev_ptr3, dev_ptr3+(arrsize+1), dev_ptr3); //	in-place scan
	numofBit[0]=dev_ptr3[arrsize];

	printf("Proved Sites in Y : %d \n", numofBit[0]);*/

	cudaThreadSynchronize();
	printf("time 7 : %ld(ms) \n", clock()-time); time = clock();
	cudaFree(temp2D);
	cudaFree(sites_index_y);
	cudaFree(sites_y);
	cudaFree(bitDeleted);

	//LDNIDistanceField__SortProbablySite2<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(sites_x, sites_index_x, res, res*res);
	//-------------------------------Get Sites for Rendering---------------------------------------//

		
	//Display
	cudaGraphicsResource *resource;
	float gWidth=(boundingBox[1]-boundingBox[0])/(float)res;
	float width = gWidth*(float)res;
	float origin[3];
	origin[0]=boundingBox[0]+gWidth*0.5f;
	origin[1]=boundingBox[2]+gWidth*0.5f;
	origin[2]=boundingBox[4]+gWidth*0.5f;
	
	unsigned int* counter;
	CUDA_SAFE_CALL(cudaMalloc((void**) &counter, sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset( counter, 0, sizeof(unsigned int)));
	LDNIDistanceField__countArrayToVBO<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(make_int3(res, res, res), counter, sites_x, sites_index_x, offdist, res*res*res);
	CUDA_SAFE_CALL(cudaMemcpy( numofBit, counter, sizeof(unsigned int),cudaMemcpyDeviceToHost));
	//vbosize = LDNIDistanceField_ReadArrayToVBO(rr, vbo, bitDeleted, res, width, origin);
	//-----------------------------------------------------------------------------------//
	printf("Final Site %d \n", numofBit[0]);
	
	if (numofBit[0] <= 0) 
	{
		cudaFree(bitDeleted);
		cudaFree(sites_index_x);
		cudaFree(sites_x);
		cudaFree(counter);

		return;
	}
	vbosize = numofBit[0];

	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, numofBit[0]*3*sizeof(float), 0, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&resource, *vbo, cudaGraphicsRegisterFlagsWriteDiscard));
 

	CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &resource, 0));
	size_t num_bytes; 
	float3 *dptr;
	CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, resource));
	CUDA_SAFE_CALL(cudaMemset(  counter, 0, sizeof(unsigned int)) );

	LDNIDistanceField__writeResultToVBO<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(dptr, make_int3(res, res, res), counter, sites_x, sites_index_x, offdist, width, make_float3(origin[0],origin[1],origin[2]), res*res*res);
	 
	CUDA_SAFE_CALL(cudaMemcpy( &vbosize, counter, sizeof(unsigned int),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &resource, 0));
	printf("CUDA mapped VBO: VBO Size %ld %ld bytes\n", vbosize, numofBit[0]);

	//-----------------------------------------------------------------------------------//

	
	
	cudaFree(counter);
	cudaFree(bitDeleted);
	cudaFree(sites_index_x);
	cudaFree(sites_x);
	

}



int LDNIcudaOperation::LDNIDistanceField_ReadArrayToVBO(cudaGraphicsResource *resource, GLuint *vbo, unsigned int *m_3dArray, int res, float width, float origin[3])
{
	unsigned int* countVertex;
	CUDA_SAFE_CALL(cudaMalloc((void**) & countVertex,sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset(  countVertex, 0, sizeof(unsigned int)) );
	// Declare Host Variable
	int* vbo_size = (int*)malloc(sizeof(int));

	//Step 1 : Find out the size of VBO
	LDNIDistanceField_CountBitInArray<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(countVertex, m_3dArray, res*res*(res/32), res);
	CUDA_SAFE_CALL(cudaMemcpy( vbo_size, countVertex, sizeof(unsigned int),cudaMemcpyDeviceToHost));
	printf("Distance Offset: VBO Size %ld bytes\n", vbo_size[0]);

	if (vbo_size[0] <= 0) return 0;

	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, vbo_size[0]*3*sizeof(float), 0, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&resource, *vbo, cudaGraphicsRegisterFlagsWriteDiscard));


	CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &resource, 0));
	size_t num_bytes; 
	float3 *dptr;
	CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, resource));
	CUDA_SAFE_CALL(cudaMemset(  countVertex, 0, sizeof(int)) );

	LDNIDistanceField__writeArrayToVBO<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(dptr, res, countVertex, m_3dArray, width, make_float3(origin[0],origin[1],origin[2]),  res*res*(res/32));

	CUDA_SAFE_CALL(cudaMemcpy( vbo_size, countVertex, sizeof(int),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &resource, 0));
	printf("CUDA mapped VBO: VBO Size %ld bytes\n", vbo_size[0]);


	cudaFree(countVertex);

	return vbo_size[0];
}

int LDNIcudaOperation::LDNIDistanceField_Read3DTextureToVBO(cudaGraphicsResource *resource, GLuint* vbo, int res, float width, float origin[3])
{
	/*int* countVertex;
	CUDA_SAFE_CALL(cudaMalloc((void**) & countVertex,sizeof(int)));
	CUDA_SAFE_CALL(cudaMemset(  countVertex, 0, sizeof(int)) );
	// Declare Host Variable
	int* vbo_size = (int*)malloc(sizeof(int));

	//Step 1 : Find out the size of VBO
	LDNIDistanceField_CountBitInInteger<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(countVertex, res*res*(res/128), res);
	CUDA_SAFE_CALL(cudaMemcpy( vbo_size, countVertex, sizeof(int),cudaMemcpyDeviceToHost));
	printf("CUDA mapped VBO: VBO Size %ld bytes\n", vbo_size[0]);

	if (vbo_size[0] <= 0) return 0;

	//Step 2 : Create the VBO
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, vbo_size[0]*3*sizeof(float), 0, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	CUDA_SAFE_CALL(cudaGraphicsGLRegisterBuffer(&resource, *vbo, cudaGraphicsRegisterFlagsWriteDiscard));

	//Step 3 :  Write VBO
	CUDA_SAFE_CALL(cudaGraphicsMapResources(1, &resource, 0));
	size_t num_bytes; 
	float3 *dptr;
	CUDA_SAFE_CALL(cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, resource));
	CUDA_SAFE_CALL(cudaMemset(  countVertex, 0, sizeof(int)) );

	LDNIDistanceField__writeTexToVBO<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(dptr, res, countVertex, width, make_float3(origin[0],origin[1],origin[2]),  res*res*(res/128));

	CUDA_SAFE_CALL(cudaMemcpy( vbo_size, countVertex, sizeof(int),cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaGraphicsUnmapResources(1, &resource, 0));
	printf("CUDA mapped VBO: VBO Size %ld bytes\n", vbo_size[0]);

	cudaFree(countVertex);
	return vbo_size[0];*/
	return 0;
}

bool LDNIcudaOperation::PBADistanceField_SitesGeneration(QuadTrglMesh *mesh, GLuint *vbo, int &vbosize, int res, float boundingBox[], int *&inputDF)
{
	const bool bCube=true;
	float origin[3],gWidth, width;		long time=clock(),totalTime=clock();
	int i,nodeNum;
	char fileadd[256];

	if ((boundingBox[0]==boundingBox[1]) && (boundingBox[2]==boundingBox[3]) && (boundingBox[4]==boundingBox[5])) {
		mesh->CompBoundingBox(boundingBox);

		if (bCube) {
			float xx=(boundingBox[0]+boundingBox[1])*0.5f;
			float yy=(boundingBox[2]+boundingBox[3])*0.5f;
			float zz=(boundingBox[4]+boundingBox[5])*0.5f;
			float ww=boundingBox[1]-boundingBox[0];
			if ((boundingBox[3]-boundingBox[2])>ww) ww=boundingBox[3]-boundingBox[2];
			if ((boundingBox[5]-boundingBox[4])>ww) ww=boundingBox[5]-boundingBox[4];

			ww=ww*0.55+ww/(float)(res-1)*2.0;

			boundingBox[0]=xx-ww;	boundingBox[1]=xx+ww;
			boundingBox[2]=yy-ww;	boundingBox[3]=yy+ww;
			boundingBox[4]=zz-ww;	boundingBox[5]=zz+ww;
		}
	}

	gWidth=(boundingBox[1]-boundingBox[0])/(float)res;
	width = gWidth*(float)res;
	origin[0]=boundingBox[0]+gWidth*0.5f;
	origin[1]=boundingBox[2]+gWidth*0.5f;
	origin[2]=boundingBox[4]+gWidth*0.5f;

	
	if (glewInit() != GLEW_OK) {printf("glewInit failed. Exiting...\n");	return false;}
	if (glewIsSupported("GL_VERSION_2_0")) {printf("\nReady for OpenGL 2.0\n");} else {printf("OpenGL 2.0 not supported\n"); return false;}

	//-----------------------------------------------------------------------------------------
	int dispListIndex;		GLhandleARB g_programObj, g_vertexShader, g_GeometryShader, g_FragShader;
	GLuint vertexTexture;	
	const char *VshaderString[1],*GshaderString[1],*FshaderString[1];
	GLint bCompiled = 0, bLinked = 0;
	char str[4096] = "";		int xF,yF;
	short nAxis;
	GLenum	buffers[16] = {GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT
						 , GL_COLOR_ATTACHMENT4_EXT, GL_COLOR_ATTACHMENT5_EXT, GL_COLOR_ATTACHMENT6_EXT, GL_COLOR_ATTACHMENT7_EXT
						 , GL_COLOR_ATTACHMENT8_EXT, GL_COLOR_ATTACHMENT9_EXT, GL_COLOR_ATTACHMENT10_EXT, GL_COLOR_ATTACHMENT11_EXT
						 , GL_COLOR_ATTACHMENT12_EXT, GL_COLOR_ATTACHMENT13_EXT, GL_COLOR_ATTACHMENT14_EXT, GL_COLOR_ATTACHMENT15_EXT};
	
		
	//-----------------------------------------------------------------------------------------
	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"..\\Shader\\sampleLDNIVertexShader.vert");
	g_vertexShader = glCreateShaderObjectARB( GL_VERTEX_SHADER_ARB );
	unsigned char *ShaderAssembly = _readShaderFile( fileadd );
	VshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_vertexShader, 1, VshaderString, NULL );
	glCompileShaderARB( g_vertexShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_vertexShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {		
		glGetInfoLogARB(g_vertexShader, sizeof(str), NULL, str);
		printf("Warning: Vertex Shader Compile Error\n\n");	return false;
	}

	//-----------------------------------------------------------------------------
	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"..\\Shader\\newSampleLDNIGShader.geo");
	g_GeometryShader = glCreateShaderObjectARB( GL_GEOMETRY_SHADER_EXT );
	ShaderAssembly = _readShaderFile( fileadd );
	GshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_GeometryShader, 1, GshaderString, NULL );
	glCompileShaderARB( g_GeometryShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_GeometryShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {
		glGetInfoLogARB(g_GeometryShader, sizeof(str), NULL, str);
		printf("Warning: Geo Shader Compile Error\n\n");		return false;
	}

	//-----------------------------------------------------------------------------
	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"..\\Shader\\voxelLDNIFragmentShader.frag");
	g_FragShader = glCreateShaderObjectARB( GL_FRAGMENT_SHADER_ARB );
	ShaderAssembly = _readShaderFile( fileadd );
	FshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_FragShader, 1, FshaderString, NULL );
	glCompileShaderARB( g_FragShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_FragShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {
		glGetInfoLogARB(g_FragShader, sizeof(str), NULL, str);
		printf("Warning: Fragment Shader Compile Error\n\n %s", str);	return false;
	}

	g_programObj = glCreateProgramObjectARB();
	if (glGetError()!=GL_NO_ERROR) printf("Error: OpenGL! 1 \n\n");
	glAttachObjectARB( g_programObj, g_vertexShader );		if (glGetError()!=GL_NO_ERROR) printf("Error: attach Vertex Shader!\n\n");
	glAttachObjectARB( g_programObj, g_GeometryShader );	if (glGetError()!=GL_NO_ERROR) printf("Error: attach Geometry Shader!\n\n");
	glAttachObjectARB( g_programObj, g_FragShader );		if (glGetError()!=GL_NO_ERROR) printf("Error: attach Fragment Shader!\n\n");
	//-----------------------------------------------------------------------------
	//	texture setting for fragment shader
	//memset(fileadd,0,256*sizeof(char));
	//strcat(fileadd, "Outdata");
	int maxColorBuffers, maxTextureSize;
	int layer = res/128;
	
	
	glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS_EXT, &maxColorBuffers );
	glGetIntegerv( GL_MAX_3D_TEXTURE_SIZE_EXT, &maxTextureSize );

	int z_tile = ceil(layer/(float)maxColorBuffers);
	printf("max texture size %d %d\n", maxTextureSize, layer);
	
	char value[10];
	for(i=0; i < min(layer, maxColorBuffers); i++){
		memset(fileadd,0,256*sizeof(char));
		strcat(fileadd, "Outdata");
		value[0] = '\0';
		sprintf(value, "%d", i+1 );
		strcat(fileadd, value);
		glBindFragDataLocationEXT(g_programObj,i,fileadd);
	}
	
	int tilesize = min(layer, maxColorBuffers)*128;
	//-------------------------------------------------------------------------------

	glLinkProgramARB( g_programObj);
	glGetObjectParameterivARB( g_programObj, GL_OBJECT_LINK_STATUS_ARB, &bLinked );
	if( bLinked == false ) {
		glGetInfoLogARB( g_programObj, sizeof(str), NULL, str );
		printf("Linking Fail: %s\n",str);	return false;
	}

	//-----------------------------------------------------------------------------------------
	//	Step 2:  creating texture for vertex array and binding 

	long texBindingTime=clock();
	glGetError();	// for clean-up the error generated before
	nodeNum=mesh->GetNodeNumber();	_texCalProduct(nodeNum,xF,yF);
	int temp;
	for(temp=1;temp<xF;temp *= 2) {}
	xF = temp;	//if (xF<64) xF=64;
	yF = (int)(nodeNum/xF)+1; if (yF<64) yF=64;
	printf("Texture Size: xF=%d yF=%d\n",xF,yF);
	float* verTex=(float*)malloc(xF*yF*3*sizeof(float));
	memset(verTex,0,xF*yF*3*sizeof(float));
	memcpy(verTex,mesh->GetNodeArrayPtr(),nodeNum*3*sizeof(float));
	glEnable(GL_TEXTURE_RECTANGLE_ARB);
	glGenTextures(1, &vertexTexture);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, vertexTexture);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB,GL_TEXTURE_WRAP_S,GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB,GL_TEXTURE_WRAP_T,GL_CLAMP);
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGB32F_ARB, xF, yF, 0, GL_RGB, GL_FLOAT, verTex);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
	free(verTex);
	if (glGetError()!=GL_NO_ERROR) printf("Error: GL_TEXTURE_RECTANGLE_ARB texture binding!\n\n");
	texBindingTime=clock()-texBindingTime;
	printf("\nTime for binding texture onto the graphics memory - %ld (ms)\n\n",texBindingTime);

	//-----------------------------------------------------------------------------------------
	//	Step 3:  building GL-list for activating the geometry shader
	unsigned int ver[4];
	int faceNum=mesh->GetFaceNumber();
	dispListIndex = glGenLists(1);
	glNewList(dispListIndex, GL_COMPILE);
	glBegin(GL_POINTS);
	for(i=0;i<faceNum;i++) {
		mesh->GetFaceNodes(i+1,ver[0],ver[1],ver[2],ver[3]);
		glVertex3i(ver[0]-1,ver[1]-1,ver[2]-1);
		if (mesh->IsQuadFace(i+1)) {glVertex3i(ver[0]-1,ver[2]-1,ver[3]-1);}	// one more triangle
	}
	glEnd();
	glEndList();

	//-----------------------------------------------------------------------------------------
	//	Step 4:  using program objects and the texture
	GLint id0,id1,id2,id3,id4;	float centerPos[3];
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB,vertexTexture);
	glUseProgramObjectARB(g_programObj);
	id0 = glGetUniformLocationARB(g_programObj,"sizeNx");
	glUniform1iARB(id0,xF);
	centerPos[0]=(boundingBox[0]+boundingBox[1])*0.5f;
	centerPos[1]=(boundingBox[2]+boundingBox[3])*0.5f;
	centerPos[2]=(boundingBox[4]+boundingBox[5])*0.5f;
	id1 = glGetUniformLocationARB(g_programObj,"Cent");
	glUniform3fARB(id1,centerPos[0],centerPos[1],centerPos[2]);
	id2 = glGetUniformLocationARB(g_programObj,"res");
	glUniform1iARB(id2,res);
	id3 = glGetUniformLocationARB(g_programObj,"tilesize");
	glUniform1iARB(id3,tilesize);
	if (glGetError()!=GL_NO_ERROR) printf("Error: vertex texture binding!\n\n");
	printf("Create shader texture\n");
	//-----------------------------------------------------------------------------------------
	//	Step 5:  Prepare 3D texture for voxelization
	GLuint PrimitiveVoxel[3];
	glEnable(GL_TEXTURE_3D_EXT);
	glGenTextures(1, &PrimitiveVoxel[0]);   // x-axis
	glBindTexture(GL_TEXTURE_3D_EXT, PrimitiveVoxel[0]);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_S,GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_T,GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_R,GL_CLAMP);
	//glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_RGBA32UI_EXT, res, res, min(layer, maxColorBuffers), 0, GL_RGBA_INTEGER_EXT,  GL_UNSIGNED_INT, 0 );
	//if res <= 2048 , create texture directly. Otherwise, need to subdivide the texture
	glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_RGBA32UI_EXT, res, res, layer, 0, GL_RGBA_INTEGER_EXT,  GL_UNSIGNED_INT, 0 );
	glBindTexture(GL_TEXTURE_3D_EXT, 0);

	glGenTextures(1, &PrimitiveVoxel[1]);   // y-axis
	glBindTexture(GL_TEXTURE_3D_EXT, PrimitiveVoxel[1]);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_S,GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_T,GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_R,GL_CLAMP);
	//glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_RGBA32UI_EXT, res, res, min(layer, maxColorBuffers), 0, GL_RGBA_INTEGER_EXT,  GL_UNSIGNED_INT, 0 );
	glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_RGBA32UI_EXT, res, res, layer, 0, GL_RGBA_INTEGER_EXT,  GL_UNSIGNED_INT, 0 );
	glBindTexture(GL_TEXTURE_3D_EXT, 0);

	glGenTextures(1, &PrimitiveVoxel[2]);   // z-axis
	glBindTexture(GL_TEXTURE_3D_EXT, PrimitiveVoxel[2]);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_S,GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_T,GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_R,GL_CLAMP);
	//glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_RGBA32UI_EXT, res, res, min(layer, maxColorBuffers), 0, GL_RGBA_INTEGER_EXT,  GL_UNSIGNED_INT, 0 );
	glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_RGBA32UI_EXT, res, res, layer, 0, GL_RGBA_INTEGER_EXT,  GL_UNSIGNED_INT, 0 );
	glBindTexture(GL_TEXTURE_3D_EXT, 0);

	
	//-----------------------------------------------------------------------------------------
	//	Step 6:  Voxelization
	GLuint fbo;
	int buffersize = min(layer, maxColorBuffers);
	int tile;
	for(tile=0; tile < z_tile; tile++)
	{
		for(nAxis=0; nAxis < 3; nAxis++)
		{
			glGenFramebuffersEXT(1, &fbo);
			glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,fbo); 
			glBindTexture(GL_TEXTURE_3D_EXT, PrimitiveVoxel[nAxis]);
			for(int a=tile*maxColorBuffers; a < min(maxColorBuffers,layer-(tile*maxColorBuffers)); a++)	glFramebufferTexture3DEXT(GL_FRAMEBUFFER_EXT, buffers[a] ,GL_TEXTURE_3D_EXT, PrimitiveVoxel[nAxis], 0, a); 

			
			id4 = glGetUniformLocationARB(g_programObj,"tile");
			glUniform1iARB(id4,tile);

			glDrawBuffers(buffersize,buffers);
			glEnable(GL_DEPTH_TEST);
			glDisable(GL_STENCIL_TEST);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glDisable(GL_POLYGON_OFFSET_FILL);
			glDisable(GL_POLYGON_OFFSET_LINE);
			glDisable(GL_BLEND);	
			glDisable(GL_POLYGON_SMOOTH);	// turn off anti-aliasing
			glDisable(GL_POINT_SMOOTH);
			glDisable(GL_LINE_SMOOTH);
			glDisable(GL_MAP_COLOR);
			glDisable(GL_DITHER);
			glShadeModel(GL_FLAT);
			glDisable(GL_LIGHTING);   glDisable(GL_LIGHT0);
			glDisable(GL_LOGIC_OP);
			glDisable(GL_COLOR_MATERIAL);
			glDisable(GL_ALPHA_TEST);
			glEnable(GL_COLOR_LOGIC_OP);
			glLogicOp(GL_OR);


			glViewport(0,0,res,res);
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(-width*0.5,width*0.5,-width*0.5,width*0.5,width*0.5,-width*0.5);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			glClearColorIuiEXT(0,0,0,0);
			glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
			glDepthFunc(GL_ALWAYS);
			glPushMatrix();
			switch(nAxis) {
				case 0:{glRotatef(-90,0,1,0);	glRotatef(-90,1,0,0); }break;
				case 1:{glRotatef(90,0,1,0);	glRotatef(90,0,0,1);  }break;
			}
			glCallList(dispListIndex); 	
			glFlush();

			glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0); 
			glBindTexture(GL_TEXTURE_3D_EXT, 0);



			glDisable(GL_COLOR_LOGIC_OP);
			glEnable(GL_POLYGON_OFFSET_FILL);
			glEnable(GL_POLYGON_OFFSET_LINE);
			glEnable(GL_BLEND);	
			glEnable(GL_DITHER);
			glDisable(GL_STENCIL_TEST);
			glDepthFunc(GL_LESS);
			glEnable(GL_MAP_COLOR);				
			glShadeModel(GL_SMOOTH);   
			glEnable(GL_LIGHTING);   glEnable(GL_LIGHT0);
			glEnable(GL_POINT_SMOOTH);
			glClearColorIuiEXT(0,0,0,0);

			glDeleteFramebuffersEXT (1,&fbo);
		}
	}
	glUseProgramObjectARB(0);

	glDeleteLists(dispListIndex, 1);
	glBindTexture( GL_TEXTURE_RECTANGLE_ARB, 0);
	glDeleteTextures(1, &vertexTexture);
	

	glDeleteObjectARB( g_vertexShader);
	glDeleteObjectARB( g_GeometryShader);
	glDeleteObjectARB( g_FragShader);
	glDeleteObjectARB( g_programObj);

	//-----------------------------------------------------------------------------------------
	//	Step 7:  Build Composite Shader
	
	
	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"..\\Shader\\CompositeVertexShader.vert");
	g_vertexShader = glCreateShaderObjectARB( GL_VERTEX_SHADER_ARB );
	ShaderAssembly = _readShaderFile( fileadd );
	VshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_vertexShader, 1, VshaderString, NULL );
	glCompileShaderARB( g_vertexShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_vertexShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {		
		glGetInfoLogARB(g_vertexShader, sizeof(str), NULL, str);
		printf("Warning: Composite Vertex Shader Compile Error\n\n %s ", str);	return false;
	}


	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"..\\Shader\\CompositeFragmentShader.frag");
	g_FragShader = glCreateShaderObjectARB( GL_FRAGMENT_SHADER_ARB );
	ShaderAssembly = _readShaderFile( fileadd );
	FshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_FragShader, 1, FshaderString, NULL );
	glCompileShaderARB( g_FragShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_FragShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {
		glGetInfoLogARB(g_FragShader, sizeof(str), NULL, str);
		printf("Warning: Composite Fragment Shader Compile Error\n\n %s", str);	return false;
	}

	
	g_programObj = glCreateProgramObjectARB();
	if (glGetError()!=GL_NO_ERROR) 	printf("Error: OpenGL! \n\n");
	glAttachObjectARB( g_programObj, g_vertexShader );		if (glGetError()!=GL_NO_ERROR) printf("Error: attach Vertex Shader!\n\n");
	glAttachObjectARB( g_programObj, g_FragShader );		if (glGetError()!=GL_NO_ERROR) printf("Error: attach Fragment Shader!\n\n");

	
	for(i=0; i < min(layer, maxColorBuffers); i++){
		memset(fileadd,0,256*sizeof(char));
		strcat(fileadd, "Outdata");
		value[0] = '\0';
		sprintf(value, "%d", i+1 );
		strcat(fileadd, value);
		glBindFragDataLocationEXT(g_programObj,i,fileadd);
	}

	//-------------------------------------------------------------------------------

	glLinkProgramARB( g_programObj);
	glGetObjectParameterivARB( g_programObj, GL_OBJECT_LINK_STATUS_ARB, &bLinked );
	if( bLinked == false ) {
		glGetInfoLogARB( g_programObj, sizeof(str), NULL, str );
		printf("Linking Fail: %s\n",str);	return false;
	}

	//-----------------------------------------------------------------------------------------
	//	Step 8:  Composite the voxelization result
	cudaGraphicsResource *resource;
	int t_index = glGetAttribLocation( g_programObj, "in_coord");
	CUDA_SAFE_CALL( cudaGraphicsGLRegisterImage(&resource, PrimitiveVoxel[2], GL_TEXTURE_3D, cudaGraphicsMapFlagsReadOnly) );
	for(tile=0; tile < z_tile; tile++)
	{
		
		glGenFramebuffersEXT(1, &fbo);
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,fbo); 
		glBindTexture(GL_TEXTURE_3D_EXT,PrimitiveVoxel[2]);
		for(int a=tile*maxColorBuffers; a < min(maxColorBuffers,layer-(tile*maxColorBuffers)); a++)	
			glFramebufferTexture3DEXT(GL_FRAMEBUFFER_EXT, buffers[a] ,GL_TEXTURE_3D_EXT, PrimitiveVoxel[2], 0, a); 
		
		//CUDA_SAFE_CALL( cudaGraphicsGLRegisterImage(&resource, PrimitiveVoxel[2], GL_TEXTURE_3D, cudaGraphicsMapFlagsReadOnly) );
		glUseProgramObjectARB(g_programObj);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_3D_EXT,PrimitiveVoxel[0]);
		glDisable(GL_TEXTURE_3D_EXT);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_3D_EXT,PrimitiveVoxel[1]);
		glDisable(GL_TEXTURE_3D_EXT);

		GLuint fetchXIndex = glGetSubroutineIndex(g_programObj, GL_FRAGMENT_SHADER, "FetchTextureX");
		GLuint fetchYIndex = glGetSubroutineIndex(g_programObj, GL_FRAGMENT_SHADER, "FetchTextureY");
	
	
		

		GLint tex0;
		tex0 = glGetUniformLocationARB(g_programObj,"Xtex");
		glUniform1iARB(tex0,0);
		tex0 = glGetUniformLocationARB(g_programObj,"Ytex");
		glUniform1iARB(tex0,1);

		id0 = glGetUniformLocationARB(g_programObj,"res");
		glUniform1iARB(id0,res);

		glDrawBuffers(min(maxColorBuffers,layer-(tile*maxColorBuffers)),buffers);
	
		

		glDisable(GL_STENCIL_TEST);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDisable(GL_POLYGON_OFFSET_FILL);
		glDisable(GL_POLYGON_OFFSET_LINE);
		glDisable(GL_BLEND);	
		glDisable(GL_POLYGON_SMOOTH);	// turn off anti-aliasing
		glDisable(GL_POINT_SMOOTH);
		glDisable(GL_LINE_SMOOTH);
		glDisable(GL_MAP_COLOR);
		glDisable(GL_DITHER);
		glShadeModel(GL_FLAT);
		glDisable(GL_LIGHTING);   glDisable(GL_LIGHT0);
		glDisable(GL_COLOR_MATERIAL);
		glDisable(GL_ALPHA_TEST);
		glDisable(GL_LOGIC_OP);

		glEnable(GL_COLOR_LOGIC_OP);
		glLogicOp(GL_OR);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

		glViewport(0, 0, res, res);

		glClearColorIuiEXT(0,0,0,0);
		glClear( GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		glUniformSubroutinesuiv( GL_FRAGMENT_SHADER, 1, &fetchXIndex);


		float l = -1.0-(1.0/(tilesize/128));
		glBegin(GL_QUADS);
		for(int i=tile*maxColorBuffers+1; i<=min(maxColorBuffers,layer-(tile*maxColorBuffers)) ; i++)
		{
			glVertexAttrib3f(t_index, 0, res, i-1); glVertex3f(-1.0f, 1.0f, l + i*(2.0/(tilesize/128)));
			glVertexAttrib3f(t_index, res, res	, i-1);	glVertex3f( 1.0f, 1.0f, l + i*(2.0/(tilesize/128)));
			glVertexAttrib3f(t_index, res, 0, i-1);	glVertex3f( 1.0f,-1.0f, l + i*(2.0/(tilesize/128)));
			glVertexAttrib3f(t_index, 0, 0	, i-1);	glVertex3f(-1.0f,-1.0f, l + i*(2.0/(tilesize/128)));

		}
		glEnd();
		glFlush();
		//glCallList(dispListIndex); 	
		//

		/*float layer = -1.0-(1.0/(res/128));
		glBegin(GL_QUADS);
		for(int i=1;i<=(res/128);i++)
		{
			glTexCoord3i(0	  ,  res	, i-1);	glVertex3f(-1.0f, 1.0f, layer + i*(2.0/(res/128)));
			glTexCoord3i(res,  res	, i-1);	glVertex3f( 1.0f, 1.0f, layer + i*(2.0/(res/128)));
			glTexCoord3i(res,		0	, i-1);	glVertex3f( 1.0f,-1.0f, layer + i*(2.0/(res/128)));
			glTexCoord3i(0	  ,		0	, i-1);	glVertex3f(-1.0f,-1.0f, layer + i*(2.0/(res/128)));
		}
		glEnd();
		glFlush();*/


		

		glUniformSubroutinesuiv( GL_FRAGMENT_SHADER, 1, &fetchYIndex);

		l = -1.0-(1.0/(tilesize/128));
		glBegin(GL_QUADS);
		for(int i=tile*maxColorBuffers+1; i<=min(maxColorBuffers,layer-(tile*maxColorBuffers)) ; i++)
		{
			glVertexAttrib3f(t_index, 0, res, i-1); glVertex3f(-1.0f, 1.0f, l + i*(2.0/(tilesize/128)));
			glVertexAttrib3f(t_index, res, res	, i-1);	glVertex3f( 1.0f, 1.0f, l + i*(2.0/(tilesize/128)));
			glVertexAttrib3f(t_index, res, 0, i-1);	glVertex3f( 1.0f,-1.0f, l + i*(2.0/(tilesize/128)));
			glVertexAttrib3f(t_index, 0, 0	, i-1);	glVertex3f(-1.0f,-1.0f, l + i*(2.0/(tilesize/128)));
		}
		glEnd();
		glFlush();

		



		glDisable(GL_COLOR_LOGIC_OP);
		glEnable(GL_POLYGON_OFFSET_FILL);
		glEnable(GL_POLYGON_OFFSET_LINE);
		glEnable(GL_BLEND);	
		glEnable(GL_DITHER);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glEnable(GL_MAP_COLOR);				
		glShadeModel(GL_SMOOTH);   
		glEnable(GL_LIGHTING);   glEnable(GL_LIGHT0);
		glEnable(GL_POINT_SMOOTH);
		glDisable(GL_COLOR_LOGIC_OP);
		glClearColorIuiEXT(0,0,0,0);
	}
	glBindTexture(GL_TEXTURE_3D_EXT,0);

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0); 
	glDeleteFramebuffersEXT (1,&fbo);
	glUseProgramObjectARB(0);

	


	CUDA_SAFE_CALL( cudaGraphicsMapResources( 1, &resource, NULL ) );
	cudaArray *in_array;
	CUDA_SAFE_CALL( cudaGraphicsSubResourceGetMappedArray( &in_array, resource, 0, 0));
	CUDA_SAFE_CALL( cudaBindTextureToArray(site_tex, in_array) );
	CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 1, &resource, NULL ) );
	

	

	//vbosize = LDNIDistanceField_Read3DTextureToVBO(resource, vbo, res, width, origin);

	/*int arrsize = res*res;
	CUDA_SAFE_CALL(cudaMalloc((void**) &sites_index, (arrsize+1)*sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset( sites_index, 0, (arrsize+1)*sizeof(unsigned int)) );


	LDNIDistanceField_CountBitInInteger<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(sites_index, res*res*(res/128), res);

	thrust::device_ptr<unsigned int> dev_ptr(sites_index); //	Wrap raw pointers with dev_ptr
	thrust::exclusive_scan(dev_ptr, dev_ptr+(arrsize+1), dev_ptr); //	in-place scan
	unsigned int siteNum=dev_ptr[arrsize];
	printf("Number of Sites: ----- %d\n",siteNum);
	vbosize = siteNum;



	CUDA_SAFE_CALL(cudaMalloc((void**) &sites, siteNum*sizeof(unsigned short)));
	CUDA_SAFE_CALL(cudaMemset( sites, 0, siteNum*sizeof(unsigned short)) );
	

	unsigned int *temp2D;
	CUDA_SAFE_CALL(cudaMalloc((void**) &temp2D, arrsize*sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset( temp2D, 0, arrsize*sizeof(unsigned int)) );

	LDNIDistanceField__writeTexToArray<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(sites, res, sites_index, temp2D, res*res*(res/128));


	cudaFree(temp2D);
	cudaFree(counter);*/

	unsigned int *counter;
	CUDA_SAFE_CALL(cudaMalloc((void**) &counter,sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset(  counter, 0, sizeof(unsigned int)) );

	PBADistanceField__writeTexToArray<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(inputDF, res, res*res*(res/128), counter);
	CUDA_SAFE_CALL( cudaMemcpy( &vbosize, counter, sizeof(unsigned int), cudaMemcpyDeviceToHost ) );


	cudaGraphicsUnregisterResource(resource);
	
/**/
	glDeleteObjectARB( g_vertexShader);
	glDeleteObjectARB( g_GeometryShader);
	glDeleteObjectARB( g_FragShader);
	glDeleteObjectARB( g_programObj);

	
	glDeleteTextures(3, PrimitiveVoxel);
	glDisable(GL_TEXTURE_3D_EXT);
	glDisable(GL_TEXTURE_RECTANGLE_ARB);
		

	return true;
}


bool LDNIcudaOperation::LDNIDistanceField_SitesGeneration(QuadTrglMesh *mesh, GLuint *vbo, int &vbosize, int res, float boundingBox[], unsigned int *&sites_index, unsigned short *&sites)
{
	const bool bCube=true;
	float origin[3],gWidth, width;		long time=clock(),totalTime=clock();
	int i,nodeNum;
	char fileadd[256];

	if ((boundingBox[0]==boundingBox[1]) && (boundingBox[2]==boundingBox[3]) && (boundingBox[4]==boundingBox[5])) {
		mesh->CompBoundingBox(boundingBox);

		if (bCube) {
			float xx=(boundingBox[0]+boundingBox[1])*0.5f;
			float yy=(boundingBox[2]+boundingBox[3])*0.5f;
			float zz=(boundingBox[4]+boundingBox[5])*0.5f;
			float ww=boundingBox[1]-boundingBox[0];
			if ((boundingBox[3]-boundingBox[2])>ww) ww=boundingBox[3]-boundingBox[2];
			if ((boundingBox[5]-boundingBox[4])>ww) ww=boundingBox[5]-boundingBox[4];

			ww=ww*0.55+ww/(float)(res-1)*2.0;

			boundingBox[0]=xx-ww;	boundingBox[1]=xx+ww;
			boundingBox[2]=yy-ww;	boundingBox[3]=yy+ww;
			boundingBox[4]=zz-ww;	boundingBox[5]=zz+ww;
		}
	}

	gWidth=(boundingBox[1]-boundingBox[0])/(float)res;
	width = gWidth*(float)res;
	origin[0]=boundingBox[0]+gWidth*0.5f;
	origin[1]=boundingBox[2]+gWidth*0.5f;
	origin[2]=boundingBox[4]+gWidth*0.5f;

	
	if (glewInit() != GLEW_OK) {printf("glewInit failed. Exiting...\n");	return false;}
	if (glewIsSupported("GL_VERSION_2_0")) {printf("\nReady for OpenGL 2.0\n");} else {printf("OpenGL 2.0 not supported\n"); return false;}

	//-----------------------------------------------------------------------------------------
	int dispListIndex;		GLhandleARB g_programObj, g_vertexShader, g_GeometryShader, g_FragShader;
	GLuint vertexTexture;	
	const char *VshaderString[1],*GshaderString[1],*FshaderString[1];
	GLint bCompiled = 0, bLinked = 0;
	char str[4096] = "";		int xF,yF;
	short nAxis;
	GLenum	buffers[16] = {GL_COLOR_ATTACHMENT0_EXT, GL_COLOR_ATTACHMENT1_EXT, GL_COLOR_ATTACHMENT2_EXT, GL_COLOR_ATTACHMENT3_EXT
						 , GL_COLOR_ATTACHMENT4_EXT, GL_COLOR_ATTACHMENT5_EXT, GL_COLOR_ATTACHMENT6_EXT, GL_COLOR_ATTACHMENT7_EXT
						 , GL_COLOR_ATTACHMENT8_EXT, GL_COLOR_ATTACHMENT9_EXT, GL_COLOR_ATTACHMENT10_EXT, GL_COLOR_ATTACHMENT11_EXT
						 , GL_COLOR_ATTACHMENT12_EXT, GL_COLOR_ATTACHMENT13_EXT, GL_COLOR_ATTACHMENT14_EXT, GL_COLOR_ATTACHMENT15_EXT};
	
		
	//-----------------------------------------------------------------------------------------
	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"..\\Shader\\sampleLDNIVertexShader.vert");
	g_vertexShader = glCreateShaderObjectARB( GL_VERTEX_SHADER_ARB );
	unsigned char *ShaderAssembly = _readShaderFile( fileadd );
	VshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_vertexShader, 1, VshaderString, NULL );
	glCompileShaderARB( g_vertexShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_vertexShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {		
		glGetInfoLogARB(g_vertexShader, sizeof(str), NULL, str);
		printf("Warning: Vertex Shader Compile Error\n\n");	return false;
	}

	//-----------------------------------------------------------------------------
	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"..\\Shader\\newSampleLDNIGShader.geo");
	g_GeometryShader = glCreateShaderObjectARB( GL_GEOMETRY_SHADER_EXT );
	ShaderAssembly = _readShaderFile( fileadd );
	GshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_GeometryShader, 1, GshaderString, NULL );
	glCompileShaderARB( g_GeometryShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_GeometryShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {
		glGetInfoLogARB(g_GeometryShader, sizeof(str), NULL, str);
		printf("Warning: Geo Shader Compile Error\n\n");		return false;
	}

	//-----------------------------------------------------------------------------
	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"..\\Shader\\voxelLDNIFragmentShader.frag");
	g_FragShader = glCreateShaderObjectARB( GL_FRAGMENT_SHADER_ARB );
	ShaderAssembly = _readShaderFile( fileadd );
	FshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_FragShader, 1, FshaderString, NULL );
	glCompileShaderARB( g_FragShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_FragShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {
		glGetInfoLogARB(g_FragShader, sizeof(str), NULL, str);
		printf("Warning: Fragment Shader Compile Error\n\n %s", str);	return false;
	}

	g_programObj = glCreateProgramObjectARB();
	if (glGetError()!=GL_NO_ERROR) printf("Error: OpenGL! 1 \n\n");
	glAttachObjectARB( g_programObj, g_vertexShader );		if (glGetError()!=GL_NO_ERROR) printf("Error: attach Vertex Shader!\n\n");
	glAttachObjectARB( g_programObj, g_GeometryShader );	if (glGetError()!=GL_NO_ERROR) printf("Error: attach Geometry Shader!\n\n");
	glAttachObjectARB( g_programObj, g_FragShader );		if (glGetError()!=GL_NO_ERROR) printf("Error: attach Fragment Shader!\n\n");
	//-----------------------------------------------------------------------------
	//	texture setting for fragment shader
	//memset(fileadd,0,256*sizeof(char));
	//strcat(fileadd, "Outdata");
	int maxColorBuffers, maxTextureSize;
	int layer = res/128;
	
	
	glGetIntegerv( GL_MAX_COLOR_ATTACHMENTS_EXT, &maxColorBuffers );
	glGetIntegerv( GL_MAX_3D_TEXTURE_SIZE_EXT, &maxTextureSize );

	int z_tile = ceil(layer/(float)maxColorBuffers);
	printf("max texture size %d %d\n", maxTextureSize, layer);
	
	char value[10];
	for(i=0; i < min(layer, maxColorBuffers); i++){
		memset(fileadd,0,256*sizeof(char));
		strcat(fileadd, "Outdata");
		value[0] = '\0';
		sprintf(value, "%d", i+1 );
		strcat(fileadd, value);
		glBindFragDataLocationEXT(g_programObj,i,fileadd);
	}
	
	int tilesize = min(layer, maxColorBuffers)*128;
	//-------------------------------------------------------------------------------

	glLinkProgramARB( g_programObj);
	glGetObjectParameterivARB( g_programObj, GL_OBJECT_LINK_STATUS_ARB, &bLinked );
	if( bLinked == false ) {
		glGetInfoLogARB( g_programObj, sizeof(str), NULL, str );
		printf("Linking Fail: %s\n",str);	return false;
	}

	//-----------------------------------------------------------------------------------------
	//	Step 2:  creating texture for vertex array and binding 

	long texBindingTime=clock();
	glGetError();	// for clean-up the error generated before
	nodeNum=mesh->GetNodeNumber();	_texCalProduct(nodeNum,xF,yF);
	int temp;
	for(temp=1;temp<xF;temp *= 2) {}
	xF = temp;	//if (xF<64) xF=64;
	yF = (int)(nodeNum/xF)+1; if (yF<64) yF=64;
	printf("Texture Size: xF=%d yF=%d\n",xF,yF);
	float* verTex=(float*)malloc(xF*yF*3*sizeof(float));
	memset(verTex,0,xF*yF*3*sizeof(float));
	memcpy(verTex,mesh->GetNodeArrayPtr(),nodeNum*3*sizeof(float));
	glEnable(GL_TEXTURE_RECTANGLE_ARB);
	glGenTextures(1, &vertexTexture);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, vertexTexture);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB,GL_TEXTURE_WRAP_S,GL_CLAMP);
	glTexParameteri(GL_TEXTURE_RECTANGLE_ARB,GL_TEXTURE_WRAP_T,GL_CLAMP);
	glTexImage2D(GL_TEXTURE_RECTANGLE_ARB, 0, GL_RGB32F_ARB, xF, yF, 0, GL_RGB, GL_FLOAT, verTex);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB, 0);
	free(verTex);
	if (glGetError()!=GL_NO_ERROR) printf("Error: GL_TEXTURE_RECTANGLE_ARB texture binding!\n\n");
	texBindingTime=clock()-texBindingTime;
	printf("\nTime for binding texture onto the graphics memory - %ld (ms)\n\n",texBindingTime);

	//-----------------------------------------------------------------------------------------
	//	Step 3:  building GL-list for activating the geometry shader
	unsigned int ver[4];
	int faceNum=mesh->GetFaceNumber();
	dispListIndex = glGenLists(1);
	glNewList(dispListIndex, GL_COMPILE);
	glBegin(GL_POINTS);
	for(i=0;i<faceNum;i++) {
		mesh->GetFaceNodes(i+1,ver[0],ver[1],ver[2],ver[3]);
		glVertex3i(ver[0]-1,ver[1]-1,ver[2]-1);
		if (mesh->IsQuadFace(i+1)) {glVertex3i(ver[0]-1,ver[2]-1,ver[3]-1);}	// one more triangle
	}
	glEnd();
	glEndList();

	//-----------------------------------------------------------------------------------------
	//	Step 4:  using program objects and the texture
	GLint id0,id1,id2,id3,id4;	float centerPos[3];
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE_ARB,vertexTexture);
	glUseProgramObjectARB(g_programObj);
	id0 = glGetUniformLocationARB(g_programObj,"sizeNx");
	glUniform1iARB(id0,xF);
	centerPos[0]=(boundingBox[0]+boundingBox[1])*0.5f;
	centerPos[1]=(boundingBox[2]+boundingBox[3])*0.5f;
	centerPos[2]=(boundingBox[4]+boundingBox[5])*0.5f;
	id1 = glGetUniformLocationARB(g_programObj,"Cent");
	glUniform3fARB(id1,centerPos[0],centerPos[1],centerPos[2]);
	id2 = glGetUniformLocationARB(g_programObj,"res");
	glUniform1iARB(id2,res);
	id3 = glGetUniformLocationARB(g_programObj,"tilesize");
	glUniform1iARB(id3,tilesize);
	if (glGetError()!=GL_NO_ERROR) printf("Error: vertex texture binding!\n\n");
	printf("Create shader texture\n");
	//-----------------------------------------------------------------------------------------
	//	Step 5:  Prepare 3D texture for voxelization
	GLuint PrimitiveVoxel[3];
	glEnable(GL_TEXTURE_3D_EXT);
	glGenTextures(1, &PrimitiveVoxel[0]);   // x-axis
	glBindTexture(GL_TEXTURE_3D_EXT, PrimitiveVoxel[0]);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_S,GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_T,GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_R,GL_CLAMP);
	//glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_RGBA32UI_EXT, res, res, min(layer, maxColorBuffers), 0, GL_RGBA_INTEGER_EXT,  GL_UNSIGNED_INT, 0 );
	//if res <= 2048 , create texture directly. Otherwise, need to subdivide the texture
	glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_RGBA32UI_EXT, res, res, layer, 0, GL_RGBA_INTEGER_EXT,  GL_UNSIGNED_INT, 0 );
	glBindTexture(GL_TEXTURE_3D_EXT, 0);

	glGenTextures(1, &PrimitiveVoxel[1]);   // y-axis
	glBindTexture(GL_TEXTURE_3D_EXT, PrimitiveVoxel[1]);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_S,GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_T,GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_R,GL_CLAMP);
	//glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_RGBA32UI_EXT, res, res, min(layer, maxColorBuffers), 0, GL_RGBA_INTEGER_EXT,  GL_UNSIGNED_INT, 0 );
	glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_RGBA32UI_EXT, res, res, layer, 0, GL_RGBA_INTEGER_EXT,  GL_UNSIGNED_INT, 0 );
	glBindTexture(GL_TEXTURE_3D_EXT, 0);

	glGenTextures(1, &PrimitiveVoxel[2]);   // z-axis
	glBindTexture(GL_TEXTURE_3D_EXT, PrimitiveVoxel[2]);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_S,GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_T,GL_CLAMP);
	glTexParameteri(GL_TEXTURE_3D_EXT,GL_TEXTURE_WRAP_R,GL_CLAMP);
	//glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_RGBA32UI_EXT, res, res, min(layer, maxColorBuffers), 0, GL_RGBA_INTEGER_EXT,  GL_UNSIGNED_INT, 0 );
	glTexImage3DEXT(GL_TEXTURE_3D_EXT, 0, GL_RGBA32UI_EXT, res, res, layer, 0, GL_RGBA_INTEGER_EXT,  GL_UNSIGNED_INT, 0 );
	glBindTexture(GL_TEXTURE_3D_EXT, 0);

	
	//-----------------------------------------------------------------------------------------
	//	Step 6:  Voxelization
	GLuint fbo;
	int buffersize = min(layer, maxColorBuffers);
	int tile;
	for(tile=0; tile < z_tile; tile++)
	{
		for(nAxis=0; nAxis < 3; nAxis++)
		{
			glGenFramebuffersEXT(1, &fbo);
			glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,fbo); 
			glBindTexture(GL_TEXTURE_3D_EXT, PrimitiveVoxel[nAxis]);
			for(int a=tile*maxColorBuffers; a < min(maxColorBuffers,layer-(tile*maxColorBuffers)); a++)	glFramebufferTexture3DEXT(GL_FRAMEBUFFER_EXT, buffers[a] ,GL_TEXTURE_3D_EXT, PrimitiveVoxel[nAxis], 0, a); 

			printf("tile - %d %d %d \n", z_tile, tile, buffersize);
			id4 = glGetUniformLocationARB(g_programObj,"tile");
			glUniform1iARB(id4,tile);

			glDrawBuffers(buffersize,buffers);
			glEnable(GL_DEPTH_TEST);
			glDisable(GL_STENCIL_TEST);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			glDisable(GL_POLYGON_OFFSET_FILL);
			glDisable(GL_POLYGON_OFFSET_LINE);
			glDisable(GL_BLEND);	
			glDisable(GL_POLYGON_SMOOTH);	// turn off anti-aliasing
			glDisable(GL_POINT_SMOOTH);
			glDisable(GL_LINE_SMOOTH);
			glDisable(GL_MAP_COLOR);
			glDisable(GL_DITHER);
			glShadeModel(GL_FLAT);
			glDisable(GL_LIGHTING);   glDisable(GL_LIGHT0);
			glDisable(GL_LOGIC_OP);
			glDisable(GL_COLOR_MATERIAL);
			glDisable(GL_ALPHA_TEST);
			glEnable(GL_COLOR_LOGIC_OP);
			glLogicOp(GL_OR);


			glViewport(0,0,res,res);
			glMatrixMode(GL_PROJECTION);
			glLoadIdentity();
			glOrtho(-width*0.5,width*0.5,-width*0.5,width*0.5,width*0.5,-width*0.5);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			glClearColorIuiEXT(0,0,0,0);
			glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
			glDepthFunc(GL_ALWAYS);
			glPushMatrix();
			switch(nAxis) {
				case 0:{glRotatef(-90,0,1,0);	glRotatef(-90,1,0,0); }break;
				case 1:{glRotatef(90,0,1,0);	glRotatef(90,0,0,1);  }break;
			}
			glCallList(dispListIndex); 	
			glFlush();

			glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0); 
			glBindTexture(GL_TEXTURE_3D_EXT, 0);



			glDisable(GL_COLOR_LOGIC_OP);
			glEnable(GL_POLYGON_OFFSET_FILL);
			glEnable(GL_POLYGON_OFFSET_LINE);
			glEnable(GL_BLEND);	
			glEnable(GL_DITHER);
			glDisable(GL_STENCIL_TEST);
			glDepthFunc(GL_LESS);
			glEnable(GL_MAP_COLOR);				
			glShadeModel(GL_SMOOTH);   
			glEnable(GL_LIGHTING);   glEnable(GL_LIGHT0);
			glEnable(GL_POINT_SMOOTH);
			glClearColorIuiEXT(0,0,0,0);

			glDeleteFramebuffersEXT (1,&fbo);
		}
	}
	glUseProgramObjectARB(0);

	glDeleteLists(dispListIndex, 1);
	glBindTexture( GL_TEXTURE_RECTANGLE_ARB, 0);
	glDeleteTextures(1, &vertexTexture);
	

	glDeleteObjectARB( g_vertexShader);
	glDeleteObjectARB( g_GeometryShader);
	glDeleteObjectARB( g_FragShader);
	glDeleteObjectARB( g_programObj);

	//-----------------------------------------------------------------------------------------
	//	Step 7:  Build Composite Shader
	
	
	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"..\\Shader\\CompositeVertexShader.vert");
	g_vertexShader = glCreateShaderObjectARB( GL_VERTEX_SHADER_ARB );
	ShaderAssembly = _readShaderFile( fileadd );
	VshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_vertexShader, 1, VshaderString, NULL );
	glCompileShaderARB( g_vertexShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_vertexShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {		
		glGetInfoLogARB(g_vertexShader, sizeof(str), NULL, str);
		printf("Warning: Composite Vertex Shader Compile Error\n\n %s ", str);	return false;
	}


	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"..\\Shader\\CompositeFragmentShader.frag");
	g_FragShader = glCreateShaderObjectARB( GL_FRAGMENT_SHADER_ARB );
	ShaderAssembly = _readShaderFile( fileadd );
	FshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_FragShader, 1, FshaderString, NULL );
	glCompileShaderARB( g_FragShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_FragShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {
		glGetInfoLogARB(g_FragShader, sizeof(str), NULL, str);
		printf("Warning: Composite Fragment Shader Compile Error\n\n %s", str);	return false;
	}

	
	g_programObj = glCreateProgramObjectARB();
	if (glGetError()!=GL_NO_ERROR) 	printf("Error: OpenGL! \n\n");
	glAttachObjectARB( g_programObj, g_vertexShader );		if (glGetError()!=GL_NO_ERROR) printf("Error: attach Vertex Shader!\n\n");
	glAttachObjectARB( g_programObj, g_FragShader );		if (glGetError()!=GL_NO_ERROR) printf("Error: attach Fragment Shader!\n\n");

	
	for(i=0; i < min(layer, maxColorBuffers); i++){
		memset(fileadd,0,256*sizeof(char));
		strcat(fileadd, "Outdata");
		value[0] = '\0';
		sprintf(value, "%d", i+1 );
		strcat(fileadd, value);
		glBindFragDataLocationEXT(g_programObj,i,fileadd);
	}

	//-------------------------------------------------------------------------------

	glLinkProgramARB( g_programObj);
	glGetObjectParameterivARB( g_programObj, GL_OBJECT_LINK_STATUS_ARB, &bLinked );
	if( bLinked == false ) {
		glGetInfoLogARB( g_programObj, sizeof(str), NULL, str );
		printf("Linking Fail: %s\n",str);	return false;
	}

	//-----------------------------------------------------------------------------------------
	//	Step 8:  Composite the voxelization result
	cudaGraphicsResource *resource;
	int t_index = glGetAttribLocation( g_programObj, "in_coord");
	CUDA_SAFE_CALL( cudaGraphicsGLRegisterImage(&resource, PrimitiveVoxel[2], GL_TEXTURE_3D, cudaGraphicsMapFlagsReadOnly) );
	for(tile=0; tile < z_tile; tile++)
	{
		
		glGenFramebuffersEXT(1, &fbo);
		glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,fbo); 
		glBindTexture(GL_TEXTURE_3D_EXT,PrimitiveVoxel[2]);
		for(int a=tile*maxColorBuffers; a < min(maxColorBuffers,layer-(tile*maxColorBuffers)); a++)	
			glFramebufferTexture3DEXT(GL_FRAMEBUFFER_EXT, buffers[a] ,GL_TEXTURE_3D_EXT, PrimitiveVoxel[2], 0, a); 
		
		//CUDA_SAFE_CALL( cudaGraphicsGLRegisterImage(&resource, PrimitiveVoxel[2], GL_TEXTURE_3D, cudaGraphicsMapFlagsReadOnly) );
		glUseProgramObjectARB(g_programObj);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_3D_EXT,PrimitiveVoxel[0]);
		glDisable(GL_TEXTURE_3D_EXT);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_3D_EXT,PrimitiveVoxel[1]);
		glDisable(GL_TEXTURE_3D_EXT);

		GLuint fetchXIndex = glGetSubroutineIndex(g_programObj, GL_FRAGMENT_SHADER, "FetchTextureX");
		GLuint fetchYIndex = glGetSubroutineIndex(g_programObj, GL_FRAGMENT_SHADER, "FetchTextureY");
	
	
		

		GLint tex0;
		tex0 = glGetUniformLocationARB(g_programObj,"Xtex");
		glUniform1iARB(tex0,0);
		tex0 = glGetUniformLocationARB(g_programObj,"Ytex");
		glUniform1iARB(tex0,1);

		id0 = glGetUniformLocationARB(g_programObj,"res");
		glUniform1iARB(id0,res);

		glDrawBuffers(min(maxColorBuffers,layer-(tile*maxColorBuffers)),buffers);
	
		

		glDisable(GL_STENCIL_TEST);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glDisable(GL_POLYGON_OFFSET_FILL);
		glDisable(GL_POLYGON_OFFSET_LINE);
		glDisable(GL_BLEND);	
		glDisable(GL_POLYGON_SMOOTH);	// turn off anti-aliasing
		glDisable(GL_POINT_SMOOTH);
		glDisable(GL_LINE_SMOOTH);
		glDisable(GL_MAP_COLOR);
		glDisable(GL_DITHER);
		glShadeModel(GL_FLAT);
		glDisable(GL_LIGHTING);   glDisable(GL_LIGHT0);
		glDisable(GL_COLOR_MATERIAL);
		glDisable(GL_ALPHA_TEST);
		glDisable(GL_LOGIC_OP);

		glEnable(GL_COLOR_LOGIC_OP);
		glLogicOp(GL_OR);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

		glViewport(0, 0, res, res);

		glClearColorIuiEXT(0,0,0,0);
		glClear( GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

		glUniformSubroutinesuiv( GL_FRAGMENT_SHADER, 1, &fetchXIndex);


		float l = -1.0-(1.0/(tilesize/128));
		glBegin(GL_QUADS);
		for(int i=tile*maxColorBuffers+1; i<=min(maxColorBuffers,layer-(tile*maxColorBuffers)) ; i++)
		{
			glVertexAttrib3f(t_index, 0, res, i-1); glVertex3f(-1.0f, 1.0f, l + i*(2.0/(tilesize/128)));
			glVertexAttrib3f(t_index, res, res	, i-1);	glVertex3f( 1.0f, 1.0f, l + i*(2.0/(tilesize/128)));
			glVertexAttrib3f(t_index, res, 0, i-1);	glVertex3f( 1.0f,-1.0f, l + i*(2.0/(tilesize/128)));
			glVertexAttrib3f(t_index, 0, 0	, i-1);	glVertex3f(-1.0f,-1.0f, l + i*(2.0/(tilesize/128)));

		}
		glEnd();
		glFlush();
		//glCallList(dispListIndex); 	
		//

		/*float layer = -1.0-(1.0/(res/128));
		glBegin(GL_QUADS);
		for(int i=1;i<=(res/128);i++)
		{
			glTexCoord3i(0	  ,  res	, i-1);	glVertex3f(-1.0f, 1.0f, layer + i*(2.0/(res/128)));
			glTexCoord3i(res,  res	, i-1);	glVertex3f( 1.0f, 1.0f, layer + i*(2.0/(res/128)));
			glTexCoord3i(res,		0	, i-1);	glVertex3f( 1.0f,-1.0f, layer + i*(2.0/(res/128)));
			glTexCoord3i(0	  ,		0	, i-1);	glVertex3f(-1.0f,-1.0f, layer + i*(2.0/(res/128)));
		}
		glEnd();
		glFlush();*/


		

		glUniformSubroutinesuiv( GL_FRAGMENT_SHADER, 1, &fetchYIndex);

		l = -1.0-(1.0/(tilesize/128));
		glBegin(GL_QUADS);
		for(int i=tile*maxColorBuffers+1; i<=min(maxColorBuffers,layer-(tile*maxColorBuffers)) ; i++)
		{
			glVertexAttrib3f(t_index, 0, res, i-1); glVertex3f(-1.0f, 1.0f, l + i*(2.0/(tilesize/128)));
			glVertexAttrib3f(t_index, res, res	, i-1);	glVertex3f( 1.0f, 1.0f, l + i*(2.0/(tilesize/128)));
			glVertexAttrib3f(t_index, res, 0, i-1);	glVertex3f( 1.0f,-1.0f, l + i*(2.0/(tilesize/128)));
			glVertexAttrib3f(t_index, 0, 0	, i-1);	glVertex3f(-1.0f,-1.0f, l + i*(2.0/(tilesize/128)));
		}
		glEnd();
		glFlush();

		



		glDisable(GL_COLOR_LOGIC_OP);
		glEnable(GL_POLYGON_OFFSET_FILL);
		glEnable(GL_POLYGON_OFFSET_LINE);
		glEnable(GL_BLEND);	
		glEnable(GL_DITHER);
		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);
		glEnable(GL_MAP_COLOR);				
		glShadeModel(GL_SMOOTH);   
		glEnable(GL_LIGHTING);   glEnable(GL_LIGHT0);
		glEnable(GL_POINT_SMOOTH);
		glDisable(GL_COLOR_LOGIC_OP);
		glClearColorIuiEXT(0,0,0,0);
	}
	glBindTexture(GL_TEXTURE_3D_EXT,0);

	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT,0); 
	glDeleteFramebuffersEXT (1,&fbo);
	glUseProgramObjectARB(0);

	


	CUDA_SAFE_CALL( cudaGraphicsMapResources( 1, &resource, NULL ) );
	cudaArray *in_array;
	CUDA_SAFE_CALL( cudaGraphicsSubResourceGetMappedArray( &in_array, resource, 0, 0));
	CUDA_SAFE_CALL( cudaBindTextureToArray(site_tex, in_array) );
	CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 1, &resource, NULL ) );
	printf("Memory Spent %.2f(MB)\n",(res*res*res/8)*1e-6);/**/

	
	//vbosize = LDNIDistanceField_Read3DTextureToVBO(resource, vbo, res, width, origin);

	int arrsize = res*res;
	CUDA_SAFE_CALL(cudaMalloc((void**) &sites_index, (arrsize+1)*sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset( sites_index, 0, (arrsize+1)*sizeof(unsigned int)) );


	LDNIDistanceField_CountBitInInteger<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(sites_index, res*res*(res/128), res);

	thrust::device_ptr<unsigned int> dev_ptr(sites_index); //	Wrap raw pointers with dev_ptr
	thrust::exclusive_scan(dev_ptr, dev_ptr+(arrsize+1), dev_ptr); //	in-place scan
	unsigned int siteNum=dev_ptr[arrsize];
	printf("Number of Sites: ----- %d\n",siteNum);
	vbosize = siteNum;



	CUDA_SAFE_CALL(cudaMalloc((void**) &sites, siteNum*sizeof(unsigned short)));
	CUDA_SAFE_CALL(cudaMemset( sites, 0, siteNum*sizeof(unsigned short)) );
	unsigned int *counter;
	CUDA_SAFE_CALL(cudaMalloc((void**) &counter,sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset(  counter, 0, sizeof(unsigned int)) );

	unsigned int *temp2D;
	CUDA_SAFE_CALL(cudaMalloc((void**) &temp2D, arrsize*sizeof(unsigned int)));
	CUDA_SAFE_CALL(cudaMemset( temp2D, 0, arrsize*sizeof(unsigned int)) );

	LDNIDistanceField__writeTexToArray<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(sites, res, sites_index, temp2D, res*res*(res/128));


	cudaFree(temp2D);
	cudaFree(counter);
	

	cudaGraphicsUnregisterResource(resource);
	
/**/
	glDeleteObjectARB( g_vertexShader);
	glDeleteObjectARB( g_GeometryShader);
	glDeleteObjectARB( g_FragShader);
	glDeleteObjectARB( g_programObj);

	
	glDeleteTextures(3, PrimitiveVoxel);
	glDisable(GL_TEXTURE_3D_EXT);
	glDisable(GL_TEXTURE_RECTANGLE_ARB);
		

	return true;

}

void LDNIcudaOperation::PBADistanceField_CompareResult(int *inputDF, int res, int numOfSite, int *sites)
{
	float totalDistError = 0.0; 
	float maxDistError = 0.0; 
	int errorCount = 0; 

	int dx, dy, dz, nx, ny, nz; 
	double dist, myDist, correctDist, error;

	int* output = (int*)malloc(res*res*res*sizeof(int));
	CUDA_SAFE_CALL(cudaMemcpy(output, inputDF, res*res*res*sizeof(int), cudaMemcpyDeviceToHost)); 

	for (int i = 0; i < res; i++)
		for (int j = 0; j < res; j++) 
			for (int k = 0; k < res; k++) {
				int id = TOID_CPU(i, j, k, res); 
				DECODE(output[id], nx, ny, nz); 

				dx = nx - i; dy = ny - j; dz = nz - k; 
				correctDist = myDist = dx * dx + dy * dy + dz * dz; 

//if (output[k*res*res+j*res+i] == 0)
				//if (i == 0 && j == 245 && k == 231)
				//{
				//	printf("error~~~~~~~~~ %d %d %d \n", i, j, k);
				//	printf(" Error!!!!!!!!!!!! %d %d %d %d %d %f \n", res, output[id], nx,ny,nz, myDist);
				//}


				for (int t = 0; t < numOfSite; t++) {
					DECODE(sites[t], nx, ny, nz); 
					dx = nx - i; dy = ny - j; dz = nz - k; 
					dist = dx * dx + dy * dy + dz * dz; 

					


					if (dist < correctDist)
					{
						/*if (i == 0 && j == 245 && k == 231)
						{
							printf("%d %d %f %f %d %d %d \n", t, sites[t], correctDist, dist, nx,ny,nz);
						}*/
						correctDist = dist; 
						
					}
				}

				if (correctDist != myDist) {
					error = fabs(sqrt(myDist) - sqrt(correctDist)); 
					if (i == 0 && j == 245 && k == 231)
					{
						//printf(" Error!!!!!!!!!!!! %d %d %d \n", i, j, k);
					printf(" Error!!!!!!!!!!!! %f %f %f %d %d %d \n", myDist, dist, correctDist, i,j,k);
					}
					errorCount++; 
					totalDistError += error; 

					if (error > maxDistError)
						maxDistError = error; 
				}
			}

	free(output);
}

//--------------------------------------------------------------------------------------------
//	Kernel functions 
//--------------------------------------------------------------------------------------------
__global__ void LDNIDistanceField_CountBitInArray(unsigned int *d_index, unsigned int *m_3dArray, int nodeNum, int res)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int temp;
	int ix,iy,iz;
	unsigned int count = 0;
	while(tid<nodeNum) {
		temp = m_3dArray[tid];
		count = bitCount(temp);
		atomicAdd(d_index,count);
		/*count = 0;
		ix=tid%res;	iy=(tid/res)%res;  iz=(tid/(res*res));
		temp = tex3D(site_tex,ix,iy,iz);
		count= bitCount(temp.x);
		count+= bitCount(temp.y);
		count+= bitCount(temp.z);
		count+= bitCount(temp.w);

		//atomicAdd(d_output,count);
		atomicAdd(&d_index[iy*res+ix],count);
*/
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void LDNIDistanceField_CountBitInInteger(unsigned int *d_index, int nodeNum, int res)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	uint4 temp;
	int ix,iy,iz;
	unsigned int count = 0;
	while(tid<nodeNum) {
		count = 0;
		ix=tid%res;	iy=(tid/res)%res;  iz=(tid/(res*res));
		temp = tex3D(site_tex,ix,iy,iz);
		count= bitCount(temp.x);
		count+= bitCount(temp.y);
		count+= bitCount(temp.z);
		count+= bitCount(temp.w);

		//atomicAdd(d_output,count);
		atomicAdd(&d_index[iy*res+ix],count);

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void LDNIDistanceField__FilterProbablySiteInXByGivenDistance(unsigned int *bitSites, unsigned short *sites, unsigned int *sites_index, int res, int offsetPixel, int nodeNum)
{	
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	unsigned int st = 0, num = 0;
	unsigned int chunksize = blockDim.x * gridDim.x;
	unsigned int temp;
	int currentSite, prevSite, dist1, dist2;
	short currentIndex, ind;
	float d;
	short i,j;
	unsigned int buffer[THREADS_PER_BLOCK] = {0};

	while(tid<nodeNum) {
		iy = tid%res;
		iz = (tid%(chunksize*res)/res)/(chunksize/res);
		ix = (tid/(chunksize*res))*(chunksize/res)+(tid%(chunksize*res)%(chunksize)/res);

		if (iz*res*res+iy*res+ix > nodeNum)
		{
			//printf("error %d %d %d %d %d %d\n", tid, ix, iy, iz,(tid/(chunksize*res)),(tid%(chunksize*res)/(res*res)) );
			return;
		}

		if (iz == 0) 
		{
			st = sites_index[iy*res+ix];
			num = sites_index[iy*res+ix+1]-st;

			if (num > 0)
				currentSite = sites[st];

			prevSite = 0;
			currentIndex = 0;
		}



		if (num > 0)
		{
			//if (ix ==512 && iy == 512)
			//	printf("tid %d %d %d %d %d %d %d %d \n", iz, num, st, prevSite, currentSite, currentIndex, sites[st], sites[st+1]);
			if (iz == currentSite) 
			{
				prevSite = currentSite;
				currentIndex++;
				if (currentIndex >= num)	
				{prevSite = 0;}
				else	
				{currentSite = sites[st+currentIndex];}
			}

			if (prevSite <=0)
			{
				dist1 = abs((int)iz-currentSite);
				if(dist1 <= offsetPixel)
				{
					d = sqrt((float)(offsetPixel*offsetPixel - dist1*dist1));
					ind = (int)d;
					if (d >= 1)
					{
						//buffer[offsetPixel-ind] = buffer[offsetPixel-ind] | SetBitPos(iz%32);
						//buffer[offsetPixel+ind] = buffer[offsetPixel+ind] | SetBitPos(iz%32);
						temp = SetBitPos(iz%32);
						for(i=offsetPixel-ind; i <= offsetPixel+ind; i++)
						{
							buffer[i] = buffer[i] | temp;
						}
					}
					else
					{
						buffer[offsetPixel] = buffer[offsetPixel] | SetBitPos(iz%32);
					}

					//if (ix ==512 && iy == 512)
					//printf("test %d %d %d %d %d\n", iz, dist1, ind, prevSite, currentSite);
				}

			}
			else
			{
				dist2 = abs((int)iz-currentSite);
				dist1 = abs((int)iz-prevSite);
				if (dist1 <= offsetPixel || dist2 <=offsetPixel)
				{
					if (dist1 <= dist2)
					{
						d = sqrt((float)(offsetPixel*offsetPixel - dist1*dist1));

					}
					else
					{
						d = sqrt((float)(offsetPixel*offsetPixel - dist2*dist2));
					}

					ind = (int)d;
					if (d >= 1)
					{
						//buffer[offsetPixel-ind] = buffer[offsetPixel-ind] | SetBitPos(iz%32);
						//buffer[offsetPixel+ind] = buffer[offsetPixel+ind] | SetBitPos(iz%32);
						temp = SetBitPos(iz%32);
						for(i=offsetPixel-ind; i <= offsetPixel+ind; i++)
						{
							buffer[i] = buffer[i] | temp;
						}
					}
					else
					{
						buffer[offsetPixel] = buffer[offsetPixel] | SetBitPos(iz%32);
					}
					//if (ix ==512 && iy == 512)
					//	printf("test %d %d %d %d %d %d\n", iz, dist1, dist2, ind, prevSite, currentSite);
				}
			}
		}

		if ((iz+1)%32 == 0 && num>0)
		{
			j=0;
			//for(i=max(0,iy-offsetPixel); i<=min(res,iy+offsetPixel); j++,i++)
			for(i=ix-offsetPixel; i<=ix+offsetPixel; j++,i++)
			{
				if (i<0 || i >= res) continue;
				if (buffer[j]!=0) 
				{
					atomicXor(&bitSites[(iz/32)*res*res+iy*res+i], buffer[j] );
				}

			}

			for(j=0;j<offsetPixel*2+1;j++)
				buffer[j]=0;
		}
		tid += blockDim.x * gridDim.x;


	}
}


__global__ void LDNIDistanceField__FilterProbablySiteInYByGivenDistance(unsigned int *bitSites, unsigned short *sites, unsigned int *sites_index, int res, int offsetPixel, int nodeNum)
{	
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	unsigned int st = 0, num = 0;
	unsigned int chunksize = blockDim.x * gridDim.x;
	unsigned int temp;
	int currentSite, prevSite, dist1, dist2;
	short currentIndex, ind;
	float d;
	short i,j;
	unsigned int buffer[THREADS_PER_BLOCK] = {0};

	while(tid<nodeNum) {
		iy = tid%res;
		iz = (tid%(chunksize*res)/res)/(chunksize/res);
		ix = (tid/(chunksize*res))*(chunksize/res)+(tid%(chunksize*res)%(chunksize)/res);

		if (iz*res*res+iy*res+ix > nodeNum)
		{
			//printf("error %d %d %d %d %d %d\n", tid, ix, iy, iz,(tid/(chunksize*res)),(tid%(chunksize*res)/(res*res)) );
			return;
		}

		if (iz == 0) 
		{
			st = sites_index[iy*res+ix];
			num = sites_index[iy*res+ix+1]-st;

			if (num > 0)
				currentSite = sites[st];

			prevSite = 0;
			currentIndex = 0;
		}



		if (num > 0)
		{
			//if (ix ==512 && iy == 512)
			//	printf("tid %d %d %d %d %d %d %d %d \n", iz, num, st, prevSite, currentSite, currentIndex, sites[st], sites[st+1]);
			if (iz == currentSite) 
			{
				prevSite = currentSite;
				currentIndex++;
				if (currentIndex >= num)	
				{currentSite = 0;}
				else	
				{currentSite = sites[st+currentIndex];}
			}

			if (prevSite <=0 && currentSite > 0)
			{
				dist1 = abs((int)iz-currentSite);
				if(dist1 <= offsetPixel && iz <= currentSite)
				//if(dist1 <= offsetPixel)
				{
					d = sqrt((float)(offsetPixel*offsetPixel - dist1*dist1));
					ind = (int)d;
					if (d >= 1)
					{
						//buffer[offsetPixel-ind] = buffer[offsetPixel-ind] | SetBitPos(iz%32);
						//buffer[offsetPixel+ind] = buffer[offsetPixel+ind] | SetBitPos(iz%32);
						temp = SetBitPos(iz%32);
						for(i=offsetPixel-ind; i <= offsetPixel+ind; i++)
						{
							buffer[i] = buffer[i] | temp;
						}
					}
					else
					{
						buffer[offsetPixel] = buffer[offsetPixel] | SetBitPos(iz%32);
					}

					//if (ix ==512 && iy == 512)
					//printf("test %d %d %d %d %d\n", iz, dist1, ind, prevSite, currentSite);
				}

			}
			else if (prevSite > 0 && currentSite <= 0)
			{
				dist2 = abs((int)iz-prevSite);
				if(dist2 <= offsetPixel && iz >= prevSite)
					//if(dist1 <= offsetPixel)
				{
					d = sqrt((float)(offsetPixel*offsetPixel - dist2*dist2));
					ind = (int)d;
					if (d >= 1)
					{
						//buffer[offsetPixel-ind] = buffer[offsetPixel-ind] | SetBitPos(iz%32);
						//buffer[offsetPixel+ind] = buffer[offsetPixel+ind] | SetBitPos(iz%32);
						temp = SetBitPos(iz%32);
						for(i=offsetPixel-ind; i <= offsetPixel+ind; i++)
						{
							buffer[i] = buffer[i] | temp;
						}
					}
					else
					{
						buffer[offsetPixel] = buffer[offsetPixel] | SetBitPos(iz%32);
					}

					//if (ix ==512 && iy == 512)
					//printf("test %d %d %d %d %d\n", iz, dist1, ind, prevSite, currentSite);
				}
			}
			else if (prevSite > 0 && currentSite > 0)
			{
				dist2 = abs((int)iz-currentSite);
				dist1 = abs((int)iz-prevSite);
				if (dist1 <= offsetPixel || dist2 <=offsetPixel)
				{
					if (dist1 <= dist2 && iz <= prevSite)
					//if (dist1 <= dist2 )
					{
						d = sqrt((float)(offsetPixel*offsetPixel - dist1*dist1));
						ind = (int)d;
						if (d >= 1)
						{
							//buffer[offsetPixel-ind] = buffer[offsetPixel-ind] | SetBitPos(iz%32);
							//buffer[offsetPixel+ind] = buffer[offsetPixel+ind] | SetBitPos(iz%32);
							temp = SetBitPos(iz%32);
							for(i=offsetPixel-ind; i <= offsetPixel+ind; i++)
							{
								buffer[i] = buffer[i] | temp;
							}
						}
						else
						{
							buffer[offsetPixel] = buffer[offsetPixel] | SetBitPos(iz%32);
						}

					}
					else if (dist1 > dist2 && iz <= currentSite)
					//else
					{
						d = sqrt((float)(offsetPixel*offsetPixel - dist2*dist2));
						ind = (int)d;
						if (d >= 1)
						{
							//buffer[offsetPixel-ind] = buffer[offsetPixel-ind] | SetBitPos(iz%32);
							//buffer[offsetPixel+ind] = buffer[offsetPixel+ind] | SetBitPos(iz%32);
							temp = SetBitPos(iz%32);
							for(i=offsetPixel-ind; i <= offsetPixel+ind; i++)
							{
								buffer[i] = buffer[i] | temp;
							}
						}
						else
						{
							buffer[offsetPixel] = buffer[offsetPixel] | SetBitPos(iz%32);
						}
					}

					
					//if (ix ==512 && iy == 512)
					//	printf("test %d %d %d %d %d %d\n", iz, dist1, dist2, ind, prevSite, currentSite);
				}
			}
		}

		if ((iz+1)%32 == 0 && num>0)
		{
			j=0;
			//for(i=max(0,iy-offsetPixel); i<=min(res,iy+offsetPixel); j++,i++)
			for(i=iy-offsetPixel; i<=iy+offsetPixel; j++,i++)
			{
				if (i<0 || i >= res) continue;
				if (buffer[j]!=0) 
				{
					atomicXor(&bitSites[(iz/32)*res*res+i*res+ix], buffer[j] );
				}

			}

			for(j=0;j<offsetPixel*2+1;j++)
				buffer[j]=0;
		}
		tid += blockDim.x * gridDim.x;


	}
}
#define LDNIMARKER	1024
__global__ void LDNIDistanceField__GetSiteByDist(ushort3 *sites, unsigned int *counter, unsigned int *sites_index, unsigned int *sites_off, int offdist, int res, int nodeNum)
{
	unsigned int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int chunksize = blockDim.x * gridDim.x;
	unsigned int ix,iy,iz, st, num, ind,i;
	ushort3 current_id, prev_id, temp;
	unsigned int dist, bitResult, count;
	float2 value;
	float off;
	
	
	while(index<nodeNum) {
		iy = index%res;
		iz = (index%(chunksize*res)/res)/(chunksize/res);
		ix = (index/(chunksize*res))*(chunksize/res)+(index%(chunksize*res)%(chunksize)/res);

		if (iz == 0)		
		{
			st = sites_index[iy*res+ix];
			num = sites_index[iy*res+ix+1]-st;

			if (num>0) current_id = sites[st];
			prev_id = make_ushort3(LDNIMARKER,LDNIMARKER,LDNIMARKER);

			ind = 0;
			bitResult = 0;
			count = 0;
			off = 0.0;
		}

		if (num > 0)
		{
			if (iz == current_id.x)
			{
				prev_id = current_id;
				ind++;
				if (ind >= num)	
					current_id = make_ushort3(LDNIMARKER,LDNIMARKER,LDNIMARKER);
				else
					current_id = sites[st+ind];

				bitResult = bitResult | SetBitPos(iz%32);
				count++;

				//if (ix == 334 && iy == 299 )
					//printf("id: %d %d %d  %d \n", prev_id.x, prev_id.y, prev_id.z, ind);
			}

			value.x = sqrt((float)((prev_id.x-iz)*(prev_id.x-iz)+(prev_id.y-ix)*(prev_id.y-ix)+(prev_id.z-iy)*(prev_id.z-iy)));
			value.y = sqrt((float)((current_id.x-iz)*(current_id.x-iz)+(current_id.y-ix)*(current_id.y-ix)+(current_id.z-iy)*(current_id.z-iy)));

			
			//if (ix == 334 && iy == 299)
			//{
			//	printf("id: %d %d %d %d %d \n", iz, current_id.x, current_id.y, current_id.z, ind);
				//for(i=0; i <num; i++)
				//{
				//	temp = sites[st+i];
				//	printf("id: %d %d %d  %d \n", temp.x, temp.y, temp.z, i);
				//}
			//}
			//dist =  (value.x < value.y)? value.x:value.y;
			off = (value.x < value.y)? value.x:value.y;

			
			//if (ix == 334 && iy == 299 && iz == 301)
			//{
			//	printf("prev: %d %d %d  %d %d\n", prev_id.x, prev_id.y, prev_id.z, st, num);
			//	printf("curr: %d %d %d \n", current_id.x, current_id.y, current_id.z);
			//	printf("%f %f %f %d %d %d %d \n", off,  value.x, value.y, offdist, ix, iy, iz);
			//}

			//if (off > offdist && off < offdist+1.0)
			//{
				/*bitResult = bitResult | SetBitPos(iz%32);
				count++;*/
				
			//}


			if ((iz+1)%32 == 0)
			{
				sites_off[(iz/32)*res*res+iy*res+ix]= bitResult;
				bitResult = 0;
			}
			if (iz == res-1)
			{
				atomicAdd(counter, count);
			}

			
		}
	


		index += blockDim.x * gridDim.x;
	}
}

__global__ void LDNIDistanceField__SortProbablySite2(unsigned int *sites, unsigned int *sites_index, int res, int nodeNum)
{
	unsigned int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iy, st, num, temp;
	short i,j;
	unsigned int tempdepth;
	unsigned int depth[256];

	while(index<nodeNum) {
		st = sites_index[index];
		num = sites_index[index+1]-st;

		if (num > 0)
		{
			if (num > 256) { printf("too many num on one thread!!! %d\n", num); return;};

			for(i=0;i<num;i++) 
			{
				depth[i]=sites[st+i];
				

			}
			for(i=0;i<num;i++) {
				for(j=i+1;j<num;j++) {
					if (GET_X(depth[i]) > GET_X(depth[j]) ){
						tempdepth=depth[i];	depth[i]=depth[j];	depth[j]=tempdepth;
						
					}
				}
			}
			for(i=0;i<num;i++) 
			{
				sites[st+i]=depth[i];
				//if (index == 143640)
				//	printf("depth %d %d %d \n", GET_X(depth[i]), GET_Y(depth[i]), GET_Z(depth[i]));
				
			}
		}
		index += blockDim.x * gridDim.x;
	}
}

__global__ void LDNIDistanceField__SortProbablySite(unsigned int *sites, unsigned int *sites_index, int res, int nodeNum)
{
	unsigned int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iy, st, num, temp;
	short i,j;
	unsigned int tempdepth;
	unsigned int depth[256];

	while(index<nodeNum) {
		st = sites_index[index];
		num = sites_index[index+1]-st;

		if (num > 0)
		{
			if (num > 256) { printf("too many num on one thread!!! %d\n", num); return;};

			/*if (506*res + 256 == index)
				printf("num %d \n", num);*/

			for(i=0;i<num;i++) 
			{
				depth[i]=sites[st+i];
			/*	if (506*res + 256 == index)
					printf("nnnn %d \n", depth[i]);*/

			}
			for(i=0;i<num;i++) {
				for(j=i+1;j<num;j++) {
					//f (depth[i].x>depth[j].x) {
					if (GET_STACK(depth[i]) > GET_STACK(depth[j]) ){
						tempdepth=depth[i];	depth[i]=depth[j];	depth[j]=tempdepth;
					}
				}
			}
			for(i=0;i<num;i++) 
			{
				//if (tz == 250 && tx == 431 )
				//if (index == 220922)	
				//	printf("%d %d %d \n", i, GET_STACK(depth[i]), GET_PTR(depth[i]));
				sites[st+i]=depth[i];
			}
		}
		/*else
		{
			printf("no site %d \n", index);
		}*/




		index += blockDim.x * gridDim.x;
	}

}



__global__ void LDNIDistanceField__GetProbablySiteInY(unsigned int *bitDeleted, unsigned int* counter, unsigned int *sites, unsigned int *sites_index, unsigned short *sites_x, unsigned int *sites_index_x, int3 res, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int chunksize = blockDim.x * gridDim.x;
	unsigned int ix,iy,iz;
	unsigned int bitresult, st_y, num_y;
	//short current_id, prev_id, dist;
	short middle_id[BANDWIDTH], ind[BANDWIDTH], current_id[BANDWIDTH], next_id[BANDWIDTH];
	short num[BANDWIDTH],  i, j, k, count;
	unsigned int st[BANDWIDTH], stack[BANDWIDTH];
	

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);

		bitresult = 0;
		if (iz == 0)
		{
			j = iy*BANDWIDTH;
			for(i=0; i < BANDWIDTH; i++)
			{
				st[i] = sites_index_x[(j+i)*res.x+ix];
				num[i] = sites_index_x[(j+i)*res.x+ix+1]-st[i];


				if (num[i] > 1) 
				{
					current_id[i] = sites_x[st[i]];
					next_id[i] = sites_x[st[i]+1];
					ind[i] = 2;
					middle_id[i] = (short)ceil((current_id[i]+next_id[i])/2.0);

				}
				else if (num[i] == 1)
				{
					current_id[i] = sites_x[st[i]];
					ind[i] = 1;
					middle_id[i] = LDNIMARKER;

				}
				else	
				{
					middle_id[i] = -1;
					current_id[i] = LDNIMARKER;
					next_id[i]= LDNIMARKER;
					ind[i] = 0;
				}

			}
		}

		bitresult = ~bitDeleted[iy*res.x*res.z+ix*res.x+iz];

		//if (__popc(bitresult)>0)
		//{
			count = 0;
			/*if (ix == 32 && iz == 1)
				printf("test test %d %d %d %d \n", ix, iy, iz, __popc(bitresult));*/
			for(i=0; i < BANDWIDTH ; i++)
			{
				
				if (num[i]>0 && GetBitPos(i, bitresult))
				{
					if (iz < middle_id[i])
					{
						stack[count] = ENCODE_STACK(iy*BANDWIDTH+i, current_id[i]);

						//if (ix == 256 && iy == 5 && iz == 508)
						//if (ix == 65 && iy == 3 && i == 8 )
						//	printf("test test %d %d %d \n", stack[count], current_id[i], iy*BANDWIDTH+i );
					}
					else
					{
						if (ind[i] < num[i])
						{
							k = sites_x[st[i]+ind[i]];
							ind[i]++;
							middle_id[i] = ceil((next_id[i] + k)/2.0);
							current_id[i] = next_id[i];
							next_id[i] = k;
						}
						else
						{
							middle_id[i] = LDNIMARKER;
							current_id[i] = next_id[i];
						}
						stack[count] = ENCODE_STACK(iy*BANDWIDTH+i, current_id[i]);
					}
					count++;
				}				
			}

			//if (ix == 32 && iz == 1)
			//	printf("@@@ %d %d %d %d \n", ix, iy, iz, count);

			//if (ix == 256 && iy == 5 && iz == 508)
			//		printf("test test %d %d \n", count, st_y);
			st_y = sites_index[ix*res.x+iz];
			i = atomicAdd(&counter[ix*res.x+iz],count);

			for(j=0; j < count ; j++)
			{
				sites[st_y+i+j] = stack[j];	

				//if (ix == 256 && iy == 5 && iz == 508)
				//if (ix == 25 && iz == 250)
				//	printf("@@ %d %d %d %d \n", j, i, GET_STACK(stack[j]), GET_PTR(stack[j]));

			}


		//}

		/*if (iz == 0)
		{	
			st = sites_index_x[iy*res+ix];
			num = sites_index_x[iy*res+ix+1]-st;

			if (num>0) current_id = sites_x[st];
			prev_id = LDNIMARKER;

			ind = 0;
		}

		if (num > 0)
		{
			if (iz%32 == 0)
			{
				value = bitDeleted[(iz/32)*res*res+iy*res+ix];
			}

			if (iz == current_id)
			{
				prev_id = current_id;
				ind++;
				if (ind >= num)	
					current_id = LDNIMARKER;
				else
					current_id = sites_x[st+ind];
			}
			
		
			if (!GetBitPos(iz%32, value))
			{
				dist =  (abs((int)(prev_id-iz)) < abs((int)(current_id-iz)))? prev_id:current_id;
				st_y = sites_index[ix*res+iz];
				i = atomicAdd(&counter[ix*res+iz],1);
				sites[st_y+i] = make_ushort2(iy, dist);				
			}
		}*/


		

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void LDNIDistanceField__GetProbablySiteInX(unsigned int *bitDeleted, unsigned int* counter, unsigned int *sites, unsigned int *sites_index, unsigned int *sites_in, unsigned int *sites_index_in, int3 res, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int chunksize = blockDim.x * gridDim.x;
	unsigned int ix,iy,iz;
	unsigned int bitresult, st_y, num_y;
	unsigned int current_id[BANDWIDTH], next_id[BANDWIDTH];
	short num[BANDWIDTH], ind[BANDWIDTH],  i;
	int j, k, count, temp;
	unsigned int st[BANDWIDTH], stack[BANDWIDTH];
	int middle_id[BANDWIDTH];
	

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);

		bitresult = 0;
		if (iz == 0)
		{
			j = iy*BANDWIDTH;
			for(i=0; i < BANDWIDTH; i++)
			{
				st[i] = sites_index_in[(j+i)*res.x+ix];
				num[i] = sites_index_in[(j+i)*res.x+ix+1]-st[i];


				if (num[i] > 1) 
				{
					current_id[i] = sites_in[st[i]];
					next_id[i] = sites_in[st[i]+1];
					ind[i] = 2;
					middle_id[i] = middlepointY(current_id[i], next_id[i], ix);

				}
				else if (num[i] == 1)
				{
					current_id[i] = sites_in[st[i]];
					ind[i] = 1;
					middle_id[i] = LDNIMARKER;

				}
				else	
				{
					middle_id[i] = -1;
					current_id[i] = LDNIMARKER;
					next_id[i]= LDNIMARKER;
					ind[i] = 0;
				}

			}
		}

		bitresult = ~bitDeleted[iy*res.x*res.z+ix*res.x+iz];

		//if (__popc(bitresult)>0)
		//{
			count = 0;

			for(i=0; i < BANDWIDTH ; i++)
			{
				if (num[i]>0)
				{
					
					if ((int)iz < middle_id[i])
					{
						if (GetBitPos(i, bitresult))
						{
							stack[count] = ENCODE_STACK_3(iy*BANDWIDTH+i, GET_STACK(current_id[i]), GET_PTR(current_id[i]));
							count++;
						}
					}
					else    
					{
						if (ind[i] < num[i])
						{
							j = sites_in[st[i]+ind[i]];
							ind[i]++;
							temp = middlepointY(next_id[i], j, ix);
							while (temp <= middle_id[i])
							{
								next_id[i] = j;
								j = sites_in[st[i]+ind[i]];
								ind[i]++;
								temp = middlepointY(next_id[i], j, ix);

							}

							middle_id[i] = temp;
							current_id[i] = next_id[i];
							next_id[i] = j;
						}
						else
						{
							middle_id[i] = LDNIMARKER;
							current_id[i] = next_id[i];
						}

						if (GetBitPos(i, bitresult))
						{
							stack[count] = ENCODE_STACK_3(iy*BANDWIDTH+i, GET_STACK(current_id[i]), GET_PTR(current_id[i]));
							count++; 
							/*if (ix == 311 && iz == 256 && iy == 3 )
							{
								printf("middle %d %d %d %d %d %d %d %d %d %d\n", count, i,iy*BANDWIDTH+i, bitresult,  middle_id[i], GET_STACK(current_id[i]), GET_PTR(current_id[i]), GET_X(stack[]) );
							}*/
						}

					}
					//if (ix == 311 && iy == 9 && i == 0)
					//{
						//for(int test = 0; test < num[i]; test++)
						//for(int test = 0; test < num[i] ; test++)
						//	printf("%d %d %d %d %d %d\n", num[i],  middlepointY(sites_in[st[i]]+test, sites_in[st[i]+test+1], ix), test, iy*BANDWIDTH+i, GET_STACK(sites_in[st[i]+test]), GET_PTR(sites_in[st[i]+test]));
						//printf("%d %d %d \n", num[i], GET_STACK(sites_in[st[i]+test]), GET_PTR(sites_in[st[i]+test]));
						//printf("%d %d %d %d \n", iz, middle_id[i], GET_STACK(current_id[i]), GET_PTR(current_id[i]));
					//}
					
				}
				
				
			}

			
			//if (ix == 32 && iz == 1)
			//	printf("@@@ %d %d %d %d \n", ix, iy, iz, count);

			//if (ix == 256 && iy == 5 && iz == 508)
			//		printf("test test %d %d \n", count, st_y);
			st_y = sites_index[ix*res.x+iz];
			i = atomicAdd(&counter[ix*res.x+iz],count);

			for(j=0; j < count ; j++)
			{
				sites[st_y+i+j] = stack[j];	

				//if (ix == 256 && iy == 5 && iz == 508)
				//if (ix == 280 && iz == 280 && iy < 6)
				//	printf("@@ %d %d %d %d %d %d %d\n", bitresult, iy,  j, i, GET_X(stack[j]), GET_Y(stack[j]), GET_Z(stack[j]));

				//if (GET_X(stack[j]) == 25 && GET_Y(stack[j]) == 329 && GET_Z(stack[j]) == 293)
				//	printf("?? %d %d %d \n", ix, iy, iz);

			}


		
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void LDNIDistanceField__CountProbablySiteInY(unsigned int *bitDeleted, unsigned int *counter, int res, int nodeNum)
{
	unsigned int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz,count=0,value, st, num;
	short i;
	
	
	while(index<nodeNum) {
		ix = index%res;
		iy = index/res;

		count = 0;
		for (i = 0; i < res/32; i++)
		{
			value = ~bitDeleted[i*res*res+iy*res+ix];
			count += __popc(value);

		}

		///if (ix == 0 && iy < 32)
		//	printf("no site !!! %d %d %d\n", ix, iy, count);

		

		atomicAdd(&counter[iy*res+ix],count);
		index += blockDim.x * gridDim.x;
	}
}


__global__ void LDNIDistanceField__GenerateProbablySiteInXLoop(unsigned int *bitDeleted, unsigned int *bitForNextLoop, unsigned int *counter, ushort2 *sites, unsigned int *sites_index, int res, int offsetPixel, int nodeNum, short loopID)
{
	unsigned int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	ushort2 current_id[3];
	short ind[3];
	ushort2 prev_id[3];
	unsigned int st[3], num[3], bitResult, bitCheck;
	float x1, x2;
	ushort2 p[3];
	unsigned int count;


	while(index<nodeNum) {
		iy = index%res;
		iz = (index%(chunksize*res)/res)/(chunksize/res);
		ix = (index/(chunksize*res))*(chunksize/res)+(index%(chunksize*res)%(chunksize)/res);

		if (iz*res*res+iy*res+ix > nodeNum)
		{			
			return;
		}

		if (iy > 0 && iy < res-1)
		{
			if (iz == 0)
			{
				st[1] = sites_index[iy*res+ix];
				num[1] = sites_index[iy*res+ix+1]-st[1];

				st[0] = sites_index[(iy-1)*res+ix];
				num[0] = sites_index[(iy-1)*res+ix+1]-st[0];

				if ((iy+loopID) < res)
				{
					st[2] = sites_index[(iy+loopID)*res+ix];
					num[2] = sites_index[(iy+loopID)*res+ix+1]-st[2];
				}
				else
				{
					st[2] = 0;
					num[2] = 0;
				}
				

				if (num[0]>0) current_id[0] = sites[st[0]];
				if (num[1]>0) current_id[1] = sites[st[1]];
				if (num[2]>0) current_id[2] = sites[st[2]];

				//if (ix == 26 && iy == 25)
				//	printf("%d %d %d %d %d %d \n", num[0], num[1], num[2], current_id[0], current_id[1], current_id[2]);

				prev_id[0] = make_ushort2(LDNIMARKER, LDNIMARKER);  //iy-1
				prev_id[1] = make_ushort2(LDNIMARKER, LDNIMARKER);   //iy
				prev_id[2] = make_ushort2(LDNIMARKER, LDNIMARKER);   //iy+loopID

				ind[0] = 0;
				ind[1] = 0;
				ind[2] = 0;

				bitResult = 0;
				bitCheck = 0;
				count = 0;


			}

			if (num[0] > 0 && num[1] > 0 && num[2] > 0)
			{
				if (iz%32 == 0)
				{
					bitCheck = bitForNextLoop[(iz/32)*res*res+iy*res+ix];
				}

				//if (ix == 125 && iy == 256)
				//	printf("%d %d %d %d\n", iz, bitCheck,GetBitPos(iz%32, bitCheck) );

				if (iz != current_id[1].x)
				{
					if ( GetBitPos(iz%32, bitCheck))
					{

					
						p[0] =  ((prev_id[0].x-iz)*(prev_id[0].x-iz)+(prev_id[0].y-ix)*(prev_id[0].y-ix) < (current_id[0].x-iz)*(current_id[0].x-iz)+(current_id[0].y-ix)*(current_id[0].y-ix))? prev_id[0]:current_id[0];
						p[1] =  ((prev_id[1].x-iz)*(prev_id[1].x-iz)+(prev_id[1].y-ix)*(prev_id[1].y-ix) < (current_id[1].x-iz)*(current_id[1].x-iz)+(current_id[1].y-ix)*(current_id[1].y-ix))? prev_id[1]:current_id[1];
						p[2] =  ((prev_id[2].x-iz)*(prev_id[2].x-iz)+(prev_id[2].y-ix)*(prev_id[2].y-ix) < (current_id[2].x-iz)*(current_id[2].x-iz)+(current_id[2].y-ix)*(current_id[2].y-ix))? prev_id[2]:current_id[2];

						x1 =  interpointY(iy-1, p[0].x, p[0].y, iy, p[1].x, p[1].y, ix, iz) ;
						x2 =  interpointY(iy, p[1].x, p[1].y, iy+loopID, p[2].x, p[2].y, ix, iz) ;


						if (x1 >= x2)
						{
							bitResult = bitResult | SetBitPos(iz%32);
							count++;
						}

					}

				}
				else
				{
					prev_id[1] = current_id[1];
					ind[1]++;
					if (ind[1] >= num[1])	
						current_id[1] = make_ushort2(LDNIMARKER, LDNIMARKER);
					else
						current_id[1] = sites[st[1]+ind[1]];
				}



				if (iz == current_id[0].x)   
				{
					prev_id[0] = current_id[0];
					ind[0]++;
					if (ind[0] >= num[0])	
						current_id[0] = make_ushort2(LDNIMARKER, LDNIMARKER);
					else
						current_id[0] = sites[st[0]+ind[0]];
				}
				if (iz == current_id[2].x)
				{
					prev_id[2] = current_id[2];
					ind[2]++;
					if (ind[2] >= num[2])	
						current_id[2] = make_ushort2(LDNIMARKER, LDNIMARKER);
					else
						current_id[2] = sites[st[2]+ind[2]];
				}

				if ((iz+1)%32 == 0)
				{
					bitForNextLoop[(iz/32)*res*res+iy*res+ix]= bitResult;
					atomicOr(&bitDeleted[(iz/32)*res*res+(iy+loopID-1)*res+ix], bitResult);
					bitResult = 0;
				}
				if (iz == res-1)
				{
					atomicAdd(counter, count);
				}
			}
		}

		index += blockDim.x * gridDim.x;

	
	}

}

__global__ void LDNIDistanceField__GenerateProbablySiteInYLoop(unsigned int *bitDeleted, unsigned int *bitForNextLoop, unsigned int *counter, unsigned short *sites, unsigned int *sites_index, int res, int offsetPixel, int nodeNum, short loopID)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	short current_id[3], ind[3];
	short prev_id[3];
	unsigned int st[3], num[3], bitResult, bitCheck;
	float y1, y2;
	short z[3];
	unsigned int count;


	while(tid<nodeNum) {
		iy = tid%res;
		iz = (tid%(chunksize*res)/res)/(chunksize/res);
		ix = (tid/(chunksize*res))*(chunksize/res)+(tid%(chunksize*res)%(chunksize)/res);

		if (iz*res*res+iy*res+ix > nodeNum)
		{			
			return;
		}

		if (iy > 0 && iy < res-1)
		{

		
			if (iz == 0)
			{
				st[1] = sites_index[iy*res+ix];
				num[1] = sites_index[iy*res+ix+1]-st[1];

				st[0] = sites_index[(iy-1)*res+ix];
				num[0] = sites_index[(iy-1)*res+ix+1]-st[0];

				if ((iy+loopID) < res)
				{
					st[2] = sites_index[(iy+loopID)*res+ix];
					num[2] = sites_index[(iy+loopID)*res+ix+1]-st[2];
				}
				else
				{
					st[2] = 0;
					num[2] = 0;
				}
				

				if (num[0]>0) current_id[0] = sites[st[0]];
				if (num[1]>0) current_id[1] = sites[st[1]];
				if (num[2]>0) current_id[2] = sites[st[2]];

				//if (ix == 26 && iy == 25)
				//	printf("%d %d %d %d %d %d \n", num[0], num[1], num[2], current_id[0], current_id[1], current_id[2]);

				prev_id[0] = LDNIMARKER;  //iy-1
				prev_id[1] = LDNIMARKER;  //iy
				prev_id[2] = LDNIMARKER;  //iy+loopID

				ind[0] = 0;
				ind[1] = 0;
				ind[2] = 0;

				bitResult = 0;
				bitCheck = 0;
				count = 0;


			}

			if (num[0] > 0 && num[1] > 0 && num[2] > 0 )//&& ix == 125 && ((iy <= 252 && iy>=200)))
			{
				if (iz%32 == 0)
				{
					bitCheck = bitForNextLoop[(iz/32)*res*res+iy*res+ix];
					//if (ix == 26 && iy == 25)
					//	printf("%d %d \n", iz, bitCheck);
				}

				//if (ix == 125 && iy == 256)
				//	printf("%d %d %d %d\n", iz, bitCheck,GetBitPos(iz%32, bitCheck) );

				if (iz != current_id[1])
				{
					if ( GetBitPos(iz%32, bitCheck))
					{

					
					z[0] =  (abs((int)(prev_id[0]-iz)) < abs((int)(current_id[0]-iz)))? prev_id[0]:current_id[0];
					z[1] =  (abs((int)(prev_id[1]-iz)) < abs((int)(current_id[1]-iz)))? prev_id[1]:current_id[1];
					z[2] =  (abs((int)(prev_id[2]-iz)) < abs((int)(current_id[2]-iz)))? prev_id[2]:current_id[2];

				

					y1 =  interpointY(ix, iy-1, z[0], ix, iy, z[1], ix, iz) ;
					y2 =  interpointY(ix, iy, z[1], ix, iy+loopID, z[2], ix, iz) ;

					if (y1 >= y2)
					{
						bitResult = bitResult | SetBitPos(iz%32);
						count++;
					}

					/*if (ix == 26 && iy == 25)
					{
						printf("%d %d %d %d %f %f %d\n", iz, z[0], z[1], z[2], y1, y2, count);
						printf(" %d %d %d %d %d %d \n", prev_id[0], prev_id[1], prev_id[2], current_id[0], current_id[1], current_id[2]);
					}*/

					if (ix == 125 && iy == 251 && iz == 211)
					{
						printf("%d %d %d %d %d %f %f %d\n", iy, iz, z[0], z[1], z[2], y1, y2, count);
						//printf("a) %d %d %d %d %d %d %d %d \n",ix, iy-1, z[0], ix, iy, z[1], ix, iz);
						//printf("b) %d %d %d %d %d %d %d %d \n",ix, iy, z[1], ix, iy+loopID, z[2], ix, iz);
						//printf(" %d %d %d %d %d %d \n", prev_id[0], prev_id[1], prev_id[2], current_id[0], current_id[1], current_id[2]);
					}

					}

				}
				else
				{
					prev_id[1] = current_id[1];
					ind[1]++;
					if (ind[1] >= num[1])	
						current_id[1] = LDNIMARKER;
					else
						current_id[1] = sites[st[1]+ind[1]];
				}



				if (iz == current_id[0])   
				{
					prev_id[0] = current_id[0];
					ind[0]++;
					if (ind[0] >= num[0])	
						current_id[0] = LDNIMARKER;
					else
						current_id[0] = sites[st[0]+ind[0]];
				}
				if (iz == current_id[2])
				{
					prev_id[2] = current_id[2];
					ind[2]++;
					if (ind[2] >= num[2])	
						current_id[2] = LDNIMARKER;
					else
						current_id[2] = sites[st[2]+ind[2]];
				}

				if ((iz+1)%32 == 0)
				{
					bitForNextLoop[(iz/32)*res*res+iy*res+ix]= bitResult;
					atomicOr(&bitDeleted[(iz/32)*res*res+(iy+loopID-1)*res+ix], bitResult);
					bitResult = 0;
				}
				if (iz == res-1)
				{
					//if (count > 0)
					//	printf("%d %d %d\n", ix, iy, count);
					atomicAdd(counter, count);
				}
			}
		}

		tid += blockDim.x * gridDim.x;

	}


}

__global__ void LDNIDistanceField__GenerateProbablySiteInX(unsigned int *bitDeleted, unsigned int *bitForNextLoop, unsigned int *counter, ushort2 *sites, unsigned int *sites_index, int res, int offsetPixel, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	ushort2 current_id[3];
	short ind[3];
	ushort2 prev_id[3];
	unsigned int st[3], num[3], bitResult;
	float x1, x2;
	ushort2 p[3];
	int count=0;
	
	while(tid<nodeNum) {
		iy = tid%res; // x axis
		iz = (tid%(chunksize*res)/res)/(chunksize/res); // y axis
		ix = (tid/(chunksize*res))*(chunksize/res)+(tid%(chunksize*res)%(chunksize)/res); // z axis

		if (iz*res*res+iy*res+ix > nodeNum)
		{		
			return;
		}

		if (iy > 0 && iy < res-1)
		{
			if (iz == 0)
			{
				st[1] = sites_index[iy*res+ix];
				num[1] = sites_index[iy*res+ix+1]-st[1];

				st[0] = sites_index[(iy-1)*res+ix];
				num[0] = sites_index[(iy-1)*res+ix+1]-st[0];

				st[2] = sites_index[(iy+1)*res+ix];
				num[2] = sites_index[(iy+1)*res+ix+1]-st[2];

				if (num[0]>0) current_id[0] = sites[st[0]];
				if (num[1]>0) current_id[1] = sites[st[1]];
				if (num[2]>0) current_id[2] = sites[st[2]];

				prev_id[0] = make_ushort2(LDNIMARKER,LDNIMARKER);  //iy-1
				prev_id[1] = make_ushort2(LDNIMARKER,LDNIMARKER);  //iy
				prev_id[2] = make_ushort2(LDNIMARKER,LDNIMARKER);  //iy+1

				ind[0] = 0;
				ind[1] = 0;
				ind[2] = 0;

				bitResult = 0;
				count = 0;

			}

			if (num[0] > 0 && num[1] > 0 && num[2] > 0)
			{
				if (iz != current_id[1].x)
				{
					
					p[0] =  ((prev_id[0].x-iz)*(prev_id[0].x-iz)+(prev_id[0].y-ix)*(prev_id[0].y-ix) < (current_id[0].x-iz)*(current_id[0].x-iz)+(current_id[0].y-ix)*(current_id[0].y-ix))? prev_id[0]:current_id[0];
					p[1] =  ((prev_id[1].x-iz)*(prev_id[1].x-iz)+(prev_id[1].y-ix)*(prev_id[1].y-ix) < (current_id[1].x-iz)*(current_id[1].x-iz)+(current_id[1].y-ix)*(current_id[1].y-ix))? prev_id[1]:current_id[1];
					p[2] =  ((prev_id[2].x-iz)*(prev_id[2].x-iz)+(prev_id[2].y-ix)*(prev_id[2].y-ix) < (current_id[2].x-iz)*(current_id[2].x-iz)+(current_id[2].y-ix)*(current_id[2].y-ix))? prev_id[2]:current_id[2];

					x1 =  interpointY(iy-1, p[0].x, p[0].y, iy, p[1].x, p[1].y, ix, iz) ;
					x2 =  interpointY(iy, p[1].x, p[1].y, iy+1, p[2].x, p[2].y, ix, iz) ;

	
					if (x1 >= x2)
					{
						bitResult = bitResult | SetBitPos(iz%32);
						count++;
					}
				}
				else
				{
					prev_id[1] = current_id[1];
					ind[1]++;
					if (ind[1] >= num[1])	
						current_id[1] = make_ushort2(LDNIMARKER, LDNIMARKER);
					else
						current_id[1] = sites[st[1]+ind[1]];
				}

				if (iz == current_id[0].x)
				{
					//if (ix == 125 && iy == 256)
					//	printf("--------------\n");
					prev_id[0] = current_id[0];
					ind[0]++;
					if (ind[0] >= num[0])	
						current_id[0] = make_ushort2(LDNIMARKER, LDNIMARKER);
					else
						current_id[0] = sites[st[0]+ind[0]];
				}
				if (iz == current_id[2].x)
				{
					prev_id[2] = current_id[2];
					ind[2]++;
					if (ind[2] >= num[2])	
						current_id[2] = make_ushort2(LDNIMARKER, LDNIMARKER);
					else
						current_id[2] = sites[st[2]+ind[2]];
				}

				if ((iz+1)%32 == 0)
				{
					bitForNextLoop[(iz/32)*res*res+iy*res+ix]= bitResult;
					bitDeleted[(iz/32)*res*res+iy*res+ix]= bitResult;
					bitResult = 0;
				}

				if (iz == res-1)
				{
					//if (iy==256)
					//	printf("count %d %d \n", ix, count);
					atomicAdd(counter, count);
				}
			}
		}


		tid += blockDim.x * gridDim.x;
	}
	
}

//#define ENCODE_16BIT(a, b)  (((a) << 8) | (b))



__global__ void LDNIDistanceField__kernelMergeBandsY_32(unsigned int *bitDeleted, unsigned short *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	short middle_id[32], ind[32], current_id[32], next_id[32];
	float y_inter[32];
	short num[32], i, j, k, count;
	unsigned int st[32];
	unsigned int stack[32]; // stack + ptr
	unsigned int bitresult = 0;
	int lasty;
	float y1, y2;
	unsigned int mask = 0; 
	bool loopflag = false;
	

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);
		
		
		if (iz == 0)
		{
			j = iy*bandNum;
			for(i=0; i < bandNum; i++)
			{
				st[i] = sites_index[(j+i)*res.x+ix];
				num[i] = sites_index[(j+i)*res.x+ix+1]-st[i];


				if (num[i] > 1) 
				{
					current_id[i] = sites[st[i]];
					next_id[i] = sites[st[i]+1];
					ind[i] = 2;
					middle_id[i] = (short)ceil((current_id[i]+next_id[i])/2.0);

				}
				else if (num[i] == 1)
				{
					current_id[i] = sites[st[i]];
					ind[i] = 1;
					middle_id[i] = LDNIMARKER;

				}
				else	
				{
					middle_id[i] = -1;
					current_id[i] = LDNIMARKER;
					next_id[i]= LDNIMARKER;
					ind[i] = 0;
				}

			}
		}

		if (iy%warpWidth == 0)
		{
			bitresult = ~bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz];
			
		}



		// broadcast the bitresult to other thread
		bitresult = __shfl((int)bitresult, ((iy/warpWidth)%bandNum)*warpWidth);

	

		count = 0;
		lasty = -1;
		

		//define last bit for each band
		//------------------------------------------------------
		//mask = 65535 << bandNum*(iy%warpWidth);  // 65535 = 0x0000ffff
		mask = bitresult;// & mask;

		if (__popc(mask) > 0) 
		{
			lasty = (iy/warpWidth)*warpSize 
				+ (32 - __ffs(__brev(mask))); // Get the most significant bit (__ffs return 1 - 32)
		}
		else
		{
			lasty = -1;
		}
		
		
		

		//link the last bit for each band ** can be optimize!!!!!!
		//------------------------------------------------------
		k = 0;

	
		//lasty = __shfl(lasty, max(0,iy%32-1));
		k = __shfl(lasty, (int)(iy%32-1));

		if ((int)(iy%32-1) >= 0) lasty = k;
	
	
		mask = __all(lasty >= 0);

	
		k = 0;

		while (mask == 0)
		{
			j = __shfl(lasty, (int)(iy%32-1));

			if (lasty < 0 && (int)(iy%32-1) >= 0) 
			{
				lasty = j;
			}
			k++;
			mask = __all(lasty >= 0); // make sure all the thread obtain the 
			if (k >= warpSize) break; // in case, but should not happen
		}
		


		if (iy%32 == 0)
			lasty = -1;

		k = -1;	
		//------------------------------------------------------
		// define stack (closest site on z axis and the pointer to previous site)
		for(i=0; i < bandNum ; i++)
		{
			if (GetBitPos((iy*bandNum+i)%32, bitresult))
			{


				if (iz < middle_id[i])
					stack[i] = ENCODE_STACK(current_id[i], lasty);
				else    
				{
					if (ind[i] < num[i])
					{
						j = sites[st[i]+ind[i]];
						ind[i]++;
						middle_id[i] = ceil((next_id[i] + j)/2.0);
						current_id[i] = next_id[i];
						next_id[i] = j;
					}
					else
					{
						middle_id[i] = LDNIMARKER;
						current_id[i] = next_id[i];
					}
					stack[i] = ENCODE_STACK(current_id[i], lasty);
					
					
				}
				count++;   
				   

				lasty = iy*bandNum + i;
				k = current_id[i]; // keep record the last current_id
	
			
				
			}
			else
			{
				
				//stack[i] = ENCODE_STACK(-1, lasty);
				stack[i] = ENCODE_STACK(k, lasty); // the last element in array = the last site
			}  
		
		}  
		//------------------------------------------------------
		// Calculate intersection point for each site

		y1 = -1 * res.x* res.x;
		for(i=0; i < bandNum ; i++)
		{
			
			lasty = GET_PTR(stack[i]);
			
			mask =  __shfl((int)stack[bandNum-1], (lasty%(warpSize*bandNum))/(bandNum));  // always get the last element in array for the last site of other thread

			if (GetBitPos((iy*bandNum+i)%32, bitresult))
			{
				if (lasty < res.x && GET_STACK(stack[i]) < res.x) // lasty < res.x  --> make sure current site is linking to previous site   
				{
					if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum)) // when previous site is not in another thread
					{
						y_inter[i] = interpointY(ix, lasty, GET_STACK(stack[lasty%bandNum]), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						
					}
					else
					{
						y_inter[i] = interpointY(ix, lasty, GET_STACK(mask), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						
						
					}

					
					
				}
				y1 = y_inter[i];
			}
			else
			{
				y_inter[i] = y1;
			}
		}

		//------------------------------------------------------
		// Start warp diverge
		//bitresult = 0; // clear in every thread for storing result later
		loopflag = true;
		int test = 0;
		while (loopflag)
		{
			loopflag = false;
			mask = 0;
			count = 0;
			y2 = 0.0;
			test++;
			for(i=0; i < bandNum ; i++)
			{
				if (count > 0) break;
				lasty = GET_PTR(stack[i]);
				y1 = __shfl(y_inter[bandNum-1],  (lasty%(warpSize*bandNum))/(bandNum));
				
				if (GetBitPos((iy*bandNum+i)%32, bitresult))
				{
					if (lasty < res.x )
					{
						if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum))
						{
							y1 = y_inter[lasty%bandNum];
							y2 = y_inter[i];
							
							if (y1 >= y2)
							{
								count++;
								if (count == 1) 	mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;

								/*if (ix == 206 && iz == 300 )
								{
									printf("1=! %d %d %d %d %d %d %f %f\n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								}*/
							}
							else 
							{
								if (count == 0) mask = 0;
							}

						}
						else
						{						
							y2 = y_inter[i];
							if (y1 >= y2)
							{
								count++;
								if (count == 1)		mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;

								/*if (ix == 206 && iz == 300 )
								{
									printf("2=! %d %d %d %d %d %d %f %f\n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								}*/
							}
							else
							{
								if (count == 0) mask = 0;
							}
						}
					}
					else
					{
						if (count == 0) mask = 0;
					}
				}
				else
				{
					if (count == 0) mask = 0;
				}

			}

			lasty = mask;
			
			mask = __ballot(count > 0);

			// update the stack

			if (mask > 0)
			{
				loopflag = true;
				k = __ffs(mask);

				lasty = __shfl(lasty,  k-1);

				i = GET_STACK(lasty);
				j = GET_PTR(lasty);

				lasty = __shfl((int)stack[i%bandNum],  (i%(warpSize*bandNum))/(bandNum));
				k = GET_PTR(lasty);
				lasty = __shfl((int)stack[k%bandNum],  (k%(warpSize*bandNum))/(bandNum));


				if (iy == j/bandNum)
				{
					stack[j%bandNum] = ENCODE_PTR(stack[j%bandNum], k);
					y_inter[j%bandNum] =  interpointY(ix, k, GET_STACK(lasty), ix, j, GET_STACK(stack[j%bandNum]), ix, iz) ;


					for(count=j%bandNum+1; count < bandNum ; count++)
					{
						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[j%bandNum];
						else break;
					}

				}
				

				if (iy == i/bandNum)
				{
					bitresult = bitresult & ~(SetBitPos(i%32));
					if ((i-1) >= 0)
						y_inter[i%bandNum] = y_inter[(max(0,i-1))%bandNum];

					for(count=i%bandNum+1; count < bandNum ; count++)
					{
						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[i%bandNum];
						else break;
					}
				}


				mask = __shfl((int)bitresult,  (i%(warpSize*bandNum))/(bandNum));

				if (i%(warpSize*bandNum)/warpSize == iy%(warpSize)/(warpWidth))
				{
					bitresult = mask;	
				}

			}
			else break;
		}
		
		if (iy%warpWidth == 0)
		{
			bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz] = ~bitresult;
		}
		tid += chunksize;
	}
}


__global__ void LDNIDistanceField__kernelMergeBandsY_16(unsigned int *bitDeleted, unsigned short *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	short middle_id[16], ind[16], current_id[16], next_id[16];
	float y_inter[16];
	short num[16], i, j, k, count;
	unsigned int st[16];
	unsigned int stack[16]; // stack + ptr
	unsigned int bitresult = 0;
	int lasty;
	float y1, y2;
	unsigned int mask = 0; 
	bool loopflag = false;
	

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);
		
		
		if (iz == 0)
		{
			j = iy*bandNum;
			for(i=0; i < bandNum; i++)
			{
				st[i] = sites_index[(j+i)*res.x+ix];
				num[i] = sites_index[(j+i)*res.x+ix+1]-st[i];


				if (num[i] > 1) 
				{
					current_id[i] = sites[st[i]];
					next_id[i] = sites[st[i]+1];
					ind[i] = 2;
					middle_id[i] = (short)ceil((current_id[i]+next_id[i])/2.0);

				}
				else if (num[i] == 1)
				{
					current_id[i] = sites[st[i]];
					ind[i] = 1;
					middle_id[i] = LDNIMARKER;

				}
				else	
				{
					middle_id[i] = -1;
					current_id[i] = LDNIMARKER;
					next_id[i]= LDNIMARKER;
					ind[i] = 0;
				}

			}
		}

		if (iy%warpWidth == 0)
		{
			bitresult = ~bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz];
			
		}



		// broadcast the bitresult to other thread
		bitresult = __shfl((int)bitresult, ((iy/warpWidth)%bandNum)*warpWidth);

	

		count = 0;
		lasty = -1;
		

		//define last bit for each band
		//------------------------------------------------------
		mask = 65535 << bandNum*(iy%warpWidth);  // 65535 = 0x0000ffff
		mask = bitresult & mask;

		if (__popc(mask) > 0) 
		{
			lasty = (iy/warpWidth)*warpSize 
				+ (32 - __ffs(__brev(mask))); // Get the most significant bit (__ffs return 1 - 32)
		}
		else
		{
			lasty = -1;
		}
		
		
		

		//link the last bit for each band ** can be optimize!!!!!!
		//------------------------------------------------------
		k = 0;

	
		//lasty = __shfl(lasty, max(0,iy%32-1));
		k = __shfl(lasty, (int)(iy%32-1));

		if ((int)(iy%32-1) >= 0) lasty = k;
	
	
		mask = __all(lasty >= 0);

	
		k = 0;

		while (mask == 0)
		{
			j = __shfl(lasty, (int)(iy%32-1));

			if (lasty < 0 && (int)(iy%32-1) >= 0) 
			{
				lasty = j;
			}
			k++;
			mask = __all(lasty >= 0); // make sure all the thread obtain the 
			if (k >= warpSize) break; // in case, but should not happen
		}
		


		if (iy%32 == 0)
			lasty = -1;

		k = -1;	
		//------------------------------------------------------
		// define stack (closest site on z axis and the pointer to previous site)
		for(i=0; i < bandNum ; i++)
		{
			if (GetBitPos((iy*bandNum+i)%32, bitresult))
			{


				if (iz < middle_id[i])
					stack[i] = ENCODE_STACK(current_id[i], lasty);
				else    
				{
					if (ind[i] < num[i])
					{
						j = sites[st[i]+ind[i]];
						ind[i]++;
						middle_id[i] = ceil((next_id[i] + j)/2.0);
						current_id[i] = next_id[i];
						next_id[i] = j;
					}
					else
					{
						middle_id[i] = LDNIMARKER;
						current_id[i] = next_id[i];
					}
					stack[i] = ENCODE_STACK(current_id[i], lasty);
					
					
				}
				count++;   
				   

				lasty = iy*bandNum + i;
				k = current_id[i]; // keep record the last current_id
	
			
				
			}
			else
			{
				
				//stack[i] = ENCODE_STACK(-1, lasty);
				stack[i] = ENCODE_STACK(k, lasty); // the last element in array = the last site
			}  
		
		}  
		//------------------------------------------------------
		// Calculate intersection point for each site

		y1 = -1 * res.x* res.x;
		for(i=0; i < bandNum ; i++)
		{
			
			lasty = GET_PTR(stack[i]);
			
			mask =  __shfl((int)stack[bandNum-1], (lasty%(warpSize*bandNum))/(bandNum));  // always get the last element in array for the last site of other thread

			if (GetBitPos((iy*bandNum+i)%32, bitresult))
			{
				if (lasty < res.x && GET_STACK(stack[i]) < res.x) // lasty < res.x  --> make sure current site is linking to previous site   
				{
					if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum)) // when previous site is not in another thread
					{
						y_inter[i] = interpointY(ix, lasty, GET_STACK(stack[lasty%bandNum]), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						
					}
					else
					{
						y_inter[i] = interpointY(ix, lasty, GET_STACK(mask), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						
						
					}

					
					
				}
				y1 = y_inter[i];
			}
			else
			{
				y_inter[i] = y1;
			}
		}

		//------------------------------------------------------
		// Start warp diverge
		//bitresult = 0; // clear in every thread for storing result later
		loopflag = true;
		int test = 0;
		while (loopflag)
		{
			loopflag = false;
			mask = 0;
			count = 0;
			y2 = 0.0;
			test++;
			for(i=0; i < bandNum ; i++)
			{
				if (count > 0) break;
				lasty = GET_PTR(stack[i]);
				y1 = __shfl(y_inter[bandNum-1],  (lasty%(warpSize*bandNum))/(bandNum));
				
				if (GetBitPos((iy*bandNum+i)%32, bitresult))
				{
					if (lasty < res.x )
					{
						if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum))
						{
							y1 = y_inter[lasty%bandNum];
							y2 = y_inter[i];
							
							if (y1 >= y2)
							{
								count++;
								if (count == 1) 	mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;

								//if (iz == 250 && ix == 431)
								//{
								//	printf("16-1=! %d %d %d %d %d %d %f %f\n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								//}
							}
							else 
							{
								if (count == 0) mask = 0;
							}

						}
						else
						{						
							y2 = y_inter[i];
							if (y1 >= y2)
							{
								count++;
								if (count == 1)		mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;

								//if (iz == 250 && ix == 431)
								//{
								//	printf("16-2=! %d %d %d %d %d %d %f %f\n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								//}
							}
							else
							{
								if (count == 0) mask = 0;
							}
						}
					}
					else
					{
						if (count == 0) mask = 0;
					}
				}
				else
				{
					if (count == 0) mask = 0;
				}

			}

			lasty = mask;
			
			mask = __ballot(count > 0);

			// update the stack

			if (mask > 0)
			{
				loopflag = true;
				k = __ffs(mask);

				lasty = __shfl(lasty,  k-1);

				i = GET_STACK(lasty);
				j = GET_PTR(lasty);

				lasty = __shfl((int)stack[i%bandNum],  (i%(warpSize*bandNum))/(bandNum));
				k = GET_PTR(lasty);
				lasty = __shfl((int)stack[k%bandNum],  (k%(warpSize*bandNum))/(bandNum));


				if (iy == j/bandNum)
				{
					stack[j%bandNum] = ENCODE_PTR(stack[j%bandNum], k);
					y_inter[j%bandNum] =  interpointY(ix, k, GET_STACK(lasty), ix, j, GET_STACK(stack[j%bandNum]), ix, iz) ;


					for(count=j%bandNum+1; count < bandNum ; count++)
					{
						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[j%bandNum];
						else break;
					}

				}
				

				if (iy == i/bandNum)
				{
					bitresult = bitresult & ~(SetBitPos(i%32));
					if ((i-1) >= 0)
						y_inter[i%bandNum] = y_inter[(max(0,i-1))%bandNum];

					for(count=i%bandNum+1; count < bandNum ; count++)
					{
						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[i%bandNum];
						else break;
					}
				}


				mask = __shfl((int)bitresult,  (i%(warpSize*bandNum))/(bandNum));

				if (i%(warpSize*bandNum)/warpSize == iy%(warpSize)/(warpWidth))
				{
					bitresult = mask;	
				}

			}
			else break;
		}
		
		if (iy%warpWidth == 0)
		{

			bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz] = ~bitresult;

			//if (ix == 503 && iz == 64)
			//	printf("-- %d %d \n", iy/warpWidth, ~bitresult);
		}
		tid += chunksize;
	}
}

__global__ void LDNIDistanceField__kernelMergeBandsY_8(unsigned int *bitDeleted, unsigned short *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	short middle_id[8], ind[8], current_id[8], next_id[8];
	float y_inter[8];
	short num[8], i, j, k, count;
	unsigned int st[8];
	unsigned int stack[8]; // stack + ptr
	unsigned int bitresult = 0;
	int lasty;
	float y1, y2;
	unsigned int mask = 0; 
	bool loopflag = false;
	

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);
		
		
		if (iz == 0)
		{
			j = iy*bandNum;
			for(i=0; i < bandNum; i++)
			{
				st[i] = sites_index[(j+i)*res.x+ix];
				num[i] = sites_index[(j+i)*res.x+ix+1]-st[i];


				if (num[i] > 1) 
				{
					current_id[i] = sites[st[i]];
					next_id[i] = sites[st[i]+1];
					ind[i] = 2;
					middle_id[i] = (short)ceil((current_id[i]+next_id[i])/2.0);

				}
				else if (num[i] == 1)
				{
					current_id[i] = sites[st[i]];
					ind[i] = 1;
					middle_id[i] = LDNIMARKER;

				}
				else	
				{
					middle_id[i] = -1;
					current_id[i] = LDNIMARKER;
					next_id[i]= LDNIMARKER;
					ind[i] = 0;
				}

			}
		}

		if (iy%warpWidth == 0)
		{
			bitresult = ~bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz];
			
		}



		// broadcast the bitresult to other thread
		bitresult = __shfl((int)bitresult, ((iy/warpWidth)%bandNum)*warpWidth);

	

		count = 0;
		lasty = -1;
		

		//define last bit for each band
		//------------------------------------------------------
		mask = 255 << bandNum*(iy%warpWidth);  // 255 = 0x000000ff
		mask = bitresult & mask;

		if (__popc(mask) > 0) 
		{
			lasty = (iy/warpWidth)*warpSize 
				+ (32 - __ffs(__brev(mask))); // Get the most significant bit (__ffs return 1 - 32)
		}
		else
		{
			lasty = -1;
		}
		
		
		

		//link the last bit for each band ** can be optimize!!!!!!
		//------------------------------------------------------
		k = 0;

	
		//lasty = __shfl(lasty, max(0,iy%32-1));
		k = __shfl(lasty, (int)(iy%32-1));

		if ((int)(iy%32-1) >= 0) lasty = k;
	
	
		mask = __all(lasty >= 0);

	
		k = 0;

		while (mask == 0)
		{
			j = __shfl(lasty, (int)(iy%32-1));

			if (lasty < 0 && (int)(iy%32-1) >= 0) 
			{
				lasty = j;
			}
			k++;
			mask = __all(lasty >= 0); // make sure all the thread obtain the 
			if (k >= warpSize) break; // in case, but should not happen
		}
		


		if (iy%32 == 0)
			lasty = -1;

		
		/*if (ix == 256 && iz == 0 )
		{
			printf("=! %d %d %d %d %d %d %d \n", i, iy , iy%32, lasty, bitresult, num[0], num[1]);
		}*/
	

		k = -1;	
		//------------------------------------------------------
		// define stack (closest site on z axis and the pointer to previous site)
		for(i=0; i < bandNum ; i++)
		{
			if (GetBitPos((iy*bandNum+i)%32, bitresult))
			{


				if (iz < middle_id[i])
					stack[i] = ENCODE_STACK(current_id[i], lasty);
				else    
				{
					if (ind[i] < num[i])
					{
						j = sites[st[i]+ind[i]];
						ind[i]++;
						middle_id[i] = ceil((next_id[i] + j)/2.0);
						current_id[i] = next_id[i];
						next_id[i] = j;
					}
					else
					{
						middle_id[i] = LDNIMARKER;
						current_id[i] = next_id[i];
					}
					stack[i] = ENCODE_STACK(current_id[i], lasty);
					
					
				}
				count++;   
				   

				lasty = iy*bandNum + i;
				k = current_id[i]; // keep record the last current_id
	
			
				
			}
			else
			{
				
				//stack[i] = ENCODE_STACK(-1, lasty);
				stack[i] = ENCODE_STACK(k, lasty); // the last element in array = the last site
			}  
		
		}  
		//------------------------------------------------------
		// Calculate intersection point for each site

		/*if (ix == 256 && iz == 0 )
						{
							printf("=! %d %d %d %d %d %d %d %d %f \n", i, iy , iy%32, lasty, GET_STACK(mask),iy*bandNum+i,GET_STACK(stack[i]), mask, y_inter[i]);
						}*/

		/*if (ix == 256 && iz == 0 && iy == 65)
						{
							printf("=! %d %d %d %d %d %d %d %d %f \n", i, iy , iy%32, lasty, GET_STACK(stack[lasty%bandNum]),iy*bandNum+i,GET_STACK(stack[i]), mask, y_inter[i]);
						}*/


		y1 = -1 * res.x* res.x;
		for(i=0; i < bandNum ; i++)
		{
			
			lasty = GET_PTR(stack[i]);
			
			mask =  __shfl((int)stack[bandNum-1], (lasty%(warpSize*bandNum))/(bandNum));  // always get the last element in array for the last site of other thread

			if (GetBitPos((iy*bandNum+i)%32, bitresult))
			{
				if (lasty < res.x && GET_STACK(stack[i]) < res.x) // lasty < res.x  --> make sure current site is linking to previous site   
				{
					if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum)) // when previous site is not in another thread
					{
						y_inter[i] = interpointY(ix, lasty, GET_STACK(stack[lasty%bandNum]), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						
					}
					else
					{
						y_inter[i] = interpointY(ix, lasty, GET_STACK(mask), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						
						
					}

					
					
				}
				y1 = y_inter[i];
			}
			else
			{
				y_inter[i] = y1;
			}
		}

		//------------------------------------------------------
		// Start warp diverge
		//bitresult = 0; // clear in every thread for storing result later
		loopflag = true;
		int test = 0;
		while (loopflag)
		{
			loopflag = false;
			mask = 0;
			count = 0;
			y2 = 0.0;
			test++;
			for(i=0; i < bandNum ; i++)
			{
				if (count > 0) break;
				lasty = GET_PTR(stack[i]);
				y1 = __shfl(y_inter[bandNum-1],  (lasty%(warpSize*bandNum))/(bandNum));
				
				
				/*if (ix == 256 && iz == 500 && iy==81 )
				{
					printf("4=! %d %d %d %d %d %f\n", i, iy , iy%32, lasty, bitresult, y1);
				}*/


				if (GetBitPos((iy*bandNum+i)%32, bitresult))
				{
					if (lasty < res.x )
					{
						if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum))
						{
							y1 = y_inter[lasty%bandNum];
							y2 = y_inter[i];
							
							if (y1 >= y2)
							{
								count++;
								if (count == 1) 	mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;

								//if (iz == 250 && ix == 431)
								//{
								//	printf("8-1=! %d %d %d %d %d %d %f %f\n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								//}
							}
							else 
							{
								if (count == 0) mask = 0;
							}

						}
						else
						{						
							y2 = y_inter[i];
							if (y1 >= y2)
							{
								count++;
								if (count == 1)		mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;

								//if (iz == 250 && ix == 431)
								//{
								//	printf("8-2=! %d %d %d %d %d %d %f %f\n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								//}
							}
							else
							{
								if (count == 0) mask = 0;
							}
						}
					}
					else
					{
						if (count == 0) mask = 0;
					}
				}
				else
				{
					if (count == 0) mask = 0;
				}

			}

			lasty = mask;
			
			mask = __ballot(count > 0);

			//if (ix == 256 && iz == 500 && iy == 80)
			//{
			//	printf("3=! %d %d \n", mask, count);
			//}
			

			// update the stack

			if (mask > 0)
			{
				loopflag = true;
				k = __ffs(mask);

				lasty = __shfl(lasty,  k-1);

				i = GET_STACK(lasty);
				j = GET_PTR(lasty);

				/*if (ix == 254 && iz == 500 )
				{
					printf("4=! %d %d %d %d %d %d %d\n", k, iy , iy%32, lasty, bitresult, i, j);
				}*/

				
				lasty = __shfl((int)stack[i%bandNum],  (i%(warpSize*bandNum))/(bandNum));
				k = GET_PTR(lasty);
				lasty = __shfl((int)stack[k%bandNum],  (k%(warpSize*bandNum))/(bandNum));


				if (iy == j/bandNum)
				{
					stack[j%bandNum] = ENCODE_PTR(stack[j%bandNum], k);
					y_inter[j%bandNum] =  interpointY(ix, k, GET_STACK(lasty), ix, j, GET_STACK(stack[j%bandNum]), ix, iz) ;


					for(count=j%bandNum+1; count < bandNum ; count++)
					{
						

						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[j%bandNum];
						else break;

						/*if (ix == 254 && iz == 500 )
						{
							printf("4=! %d %d %d %d %d\n", j, iy , iy%32, count, GET_PTR(stack[count]));
						}*/
					}

				}
				

				if (iy == i/bandNum)
				{
					bitresult = bitresult & ~(SetBitPos(i%32));
					if ((i-1) >= 0)
						y_inter[i%bandNum] = y_inter[(max(0,i-1))%bandNum];

					for(count=i%bandNum+1; count < bandNum ; count++)
					{
						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[i%bandNum];
						else break;
					}
				}


				mask = __shfl((int)bitresult,  (i%(warpSize*bandNum))/(bandNum));

				if (i%(warpSize*bandNum)/warpSize == iy%(warpSize)/(warpWidth))
				{
					bitresult = mask;	
				}

			}
			else break;

			//if (test > 40) break;

		}
		
		if (iy%warpWidth == 0)
		{
			bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz] = ~bitresult;
		}
		tid += chunksize;
	}
}


__global__ void LDNIDistanceField__kernelMergeBandsY_4(unsigned int *bitDeleted, unsigned short *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	short middle_id[4], ind[4], current_id[4], next_id[4];
	float y_inter[4];
	short num[4], i, j, k, count;
	unsigned int st[4];
	unsigned int stack[4]; // stack + ptr
	unsigned int bitresult = 0;
	int lasty;
	float y1, y2;
	unsigned int mask = 0; 
	bool loopflag = false;
	

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);
		
		
		if (iz == 0)
		{
			j = iy*bandNum;
			for(i=0; i < bandNum; i++)
			{
				st[i] = sites_index[(j+i)*res.x+ix];
				num[i] = sites_index[(j+i)*res.x+ix+1]-st[i];


				if (num[i] > 1) 
				{
					current_id[i] = sites[st[i]];
					next_id[i] = sites[st[i]+1];
					ind[i] = 2;
					middle_id[i] = (short)ceil((current_id[i]+next_id[i])/2.0);

				}
				else if (num[i] == 1)
				{
					current_id[i] = sites[st[i]];
					ind[i] = 1;
					middle_id[i] = LDNIMARKER;

				}
				else	
				{
					middle_id[i] = -1;
					current_id[i] = LDNIMARKER;
					next_id[i]= LDNIMARKER;
					ind[i] = 0;
				}

			}
		}

		if (iy%warpWidth == 0)
		{
			bitresult = ~bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz];
			
		}



		// broadcast the bitresult to other thread
		bitresult = __shfl((int)bitresult, ((iy/warpWidth)%bandNum)*warpWidth);

	

		count = 0;
		lasty = -1;
		

		//define last bit for each band
		//------------------------------------------------------
		mask = 15 << bandNum*(iy%warpWidth);  // 15 = 0x0000000f
		mask = bitresult & mask;

		if (__popc(mask) > 0) 
		{
			lasty = (iy/warpWidth)*warpSize 
				+ (32 - __ffs(__brev(mask))); // Get the most significant bit (__ffs return 1 - 32)
		}
		else
		{
			lasty = -1;
		}
		
		
		

		//link the last bit for each band ** can be optimize!!!!!!
		//------------------------------------------------------
		k = 0;

	
		//lasty = __shfl(lasty, max(0,iy%32-1));
		k = __shfl(lasty, (int)(iy%32-1));

		if ((int)(iy%32-1) >= 0) lasty = k;
	
	
		mask = __all(lasty >= 0);

	
		k = 0;

		while (mask == 0)
		{
			j = __shfl(lasty, (int)(iy%32-1));

			if (lasty < 0 && (int)(iy%32-1) >= 0) 
			{
				lasty = j;
			}
			k++;
			mask = __all(lasty >= 0); // make sure all the thread obtain the 
			if (k >= warpSize) break; // in case, but should not happen
		}
		


		if (iy%32 == 0)
			lasty = -1;

		
		/*if (ix == 256 && iz == 0 )
		{
			printf("=! %d %d %d %d %d %d %d \n", i, iy , iy%32, lasty, bitresult, num[0], num[1]);
		}*/
	

		k = -1;	
		//------------------------------------------------------
		// define stack (closest site on z axis and the pointer to previous site)
		for(i=0; i < bandNum ; i++)
		{
			if (GetBitPos((iy*bandNum+i)%32, bitresult))
			{


				if (iz < middle_id[i])
					stack[i] = ENCODE_STACK(current_id[i], lasty);
				else    
				{
					if (ind[i] < num[i])
					{
						j = sites[st[i]+ind[i]];
						ind[i]++;
						middle_id[i] = ceil((next_id[i] + j)/2.0);
						current_id[i] = next_id[i];
						next_id[i] = j;
					}
					else
					{
						middle_id[i] = LDNIMARKER;
						current_id[i] = next_id[i];
					}
					stack[i] = ENCODE_STACK(current_id[i], lasty);
					
					
				}
				count++;   
				   

				lasty = iy*bandNum + i;
				k = current_id[i]; // keep record the last current_id
	
			
				
			}
			else
			{
				
				//stack[i] = ENCODE_STACK(-1, lasty);
				stack[i] = ENCODE_STACK(k, lasty); // the last element in array = the last site
			}  
		
		}  
		//------------------------------------------------------
		// Calculate intersection point for each site

		/*if (ix == 256 && iz == 0 )
						{
							printf("=! %d %d %d %d %d %d %d %d %f \n", i, iy , iy%32, lasty, GET_STACK(mask),iy*bandNum+i,GET_STACK(stack[i]), mask, y_inter[i]);
						}*/

		/*if (ix == 256 && iz == 0 && iy == 65)
						{
							printf("=! %d %d %d %d %d %d %d %d %f \n", i, iy , iy%32, lasty, GET_STACK(stack[lasty%bandNum]),iy*bandNum+i,GET_STACK(stack[i]), mask, y_inter[i]);
						}*/


		y1 = -1 * res.x* res.x;
		for(i=0; i < bandNum ; i++)
		{
			
			lasty = GET_PTR(stack[i]);
			
			mask =  __shfl((int)stack[bandNum-1], (lasty%(warpSize*bandNum))/(bandNum));  // always get the last element in array for the last site of other thread

			if (GetBitPos((iy*bandNum+i)%32, bitresult))
			{
				if (lasty < res.x && GET_STACK(stack[i]) < res.x) // lasty < res.x  --> make sure current site is linking to previous site   
				{
					if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum)) // when previous site is not in another thread
					{
						y_inter[i] = interpointY(ix, lasty, GET_STACK(stack[lasty%bandNum]), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						
					}
					else
					{
						y_inter[i] = interpointY(ix, lasty, GET_STACK(mask), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						
						
					}

					
					
				}
				y1 = y_inter[i];
			}
			else
			{
				y_inter[i] = y1;
			}
		}

		//------------------------------------------------------
		// Start warp diverge
		//bitresult = 0; // clear in every thread for storing result later
		loopflag = true;
		int test = 0;
		while (loopflag)
		{
			loopflag = false;
			mask = 0;
			count = 0;
			y2 = 0.0;
			test++;
			for(i=0; i < bandNum ; i++)
			{
				if (count > 0) break;
				lasty = GET_PTR(stack[i]);
				y1 = __shfl(y_inter[bandNum-1],  (lasty%(warpSize*bandNum))/(bandNum));
				
				
				/*if (iz == 1 && ix == 32 && (iy*bandNum+i)==326)
				{
					printf("4=! %d %d %d %d %d %f %d \n", i, iy , iy%32, lasty, bitresult, y1, (lasty%(warpSize*bandNum))/(bandNum));
				}*/
				


				if (GetBitPos((iy*bandNum+i)%32, bitresult))
				{
					if (lasty < res.x )
					{
						if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum))
						{
							y1 = y_inter[lasty%bandNum];
							y2 = y_inter[i];
							
							if (y1 >= y2)
							{
								count++;
								if (count == 1) 	mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;

								//if (iz == 250 && ix == 431)
								//{
								//	printf("4-1=! %d %d %d %d %d %d %f %f\n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								//}
							}
							else 
							{
								if (count == 0) mask = 0;
							}

						}
						else
						{						
							y2 = y_inter[i];
							if (y1 >= y2)
							{
								count++;
								if (count == 1)		mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;

								//if (iz == 250 && ix == 431)
								//{
								//	printf("4-2=! %d %d %d %d %d %d %f %f\n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								//}
							}
							else
							{
								if (count == 0) mask = 0;
							}
						}
					}
					else
					{
						if (count == 0) mask = 0;
					}
				}
				else
				{
					if (count == 0) mask = 0;
				}

			}

			lasty = mask;
			
			mask = __ballot(count > 0);

			//if (ix == 256 && iz == 500 && iy == 80)
			//{
			//	printf("3=! %d %d \n", mask, count);
			//}
			

			// update the stack

			if (mask > 0)
			{
				loopflag = true;
				k = __ffs(mask);

				lasty = __shfl(lasty,  k-1);

				i = GET_STACK(lasty);
				j = GET_PTR(lasty);

				//if (iz == 1 && ix == 32)
				//{
				//	printf("4=! %d %d %d %d %d %d %d\n", k, iy , iy%32, lasty, bitresult, i, j);
				//}

				
				lasty = __shfl((int)stack[i%bandNum],  (i%(warpSize*bandNum))/(bandNum));
				k = GET_PTR(lasty);
				lasty = __shfl((int)stack[k%bandNum],  (k%(warpSize*bandNum))/(bandNum));


				if (iy == j/bandNum)
				{
					stack[j%bandNum] = ENCODE_PTR(stack[j%bandNum], k);
					y_inter[j%bandNum] =  interpointY(ix, k, GET_STACK(lasty), ix, j, GET_STACK(stack[j%bandNum]), ix, iz) ;


					//if (iz == 1 && ix == 32)
					//{
					//	printf("5=! %d %d %d %d %f\n", j, iy , iy%32, count, y_inter[j%bandNum]);
					//}


					for(count=j%bandNum+1; count < bandNum ; count++)
					{
						
						

						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[j%bandNum];
						else break;

						
					}

				}
				

				if (iy == i/bandNum)
				{
					bitresult = bitresult & ~(SetBitPos(i%32));

					if ((i-1) >= 0)
						y_inter[i%bandNum] = y_inter[(max(0,i-1))%bandNum];

					for(count=i%bandNum+1; count < bandNum ; count++)
					{
						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[i%bandNum];
						else break;
					}

					//if (iz == 1 && ix == 32 && (i >= 300 && i <= 304))
					//	printf("7=! %d %d %d %d %f\n", i, iy , iy%32, count, y_inter[i%bandNum]);
				}


				mask = __shfl((int)bitresult,  (i%(warpSize*bandNum))/(bandNum));

				if (i%(warpSize*bandNum)/warpSize == iy%(warpSize)/(warpWidth))
				{
					bitresult = mask;	
				}

			}
			else break;

			//if (test > 40) break;

		}
		
		if (iy%warpWidth == 0)
		{
			bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz] = ~bitresult;
		}
		tid += chunksize;
	}
}

	    
__global__ void LDNIDistanceField__kernelMergeBandsY_2(unsigned int *bitDeleted, unsigned short *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	short middle_id[2], ind[2], current_id[2], next_id[2];
	float y_inter[2];
	short num[2], i, j, k, count;
	unsigned int st[2];
	unsigned int stack[2]; // stack + ptr
	unsigned int bitresult = 0;
	int lasty;
	float y1, y2;
	unsigned int mask = 0; 
	bool loopflag = false;
	

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);
		
		
		if (iz == 0)
		{
			j = iy*bandNum;
			for(i=0; i < bandNum; i++)
			{
				st[i] = sites_index[(j+i)*res.x+ix];
				num[i] = sites_index[(j+i)*res.x+ix+1]-st[i];


				if (num[i] > 1) 
				{
					current_id[i] = sites[st[i]];
					next_id[i] = sites[st[i]+1];
					ind[i] = 2;
					middle_id[i] = (short)ceil((current_id[i]+next_id[i])/2.0);

				}
				else if (num[i] == 1)
				{
					current_id[i] = sites[st[i]];
					ind[i] = 1;
					middle_id[i] = LDNIMARKER;

				}
				else	
				{
					middle_id[i] = -1;
					current_id[i] = LDNIMARKER;
					next_id[i]= LDNIMARKER;
					ind[i] = 0;
				}

			}
			//if (ix == 65 && j+i == 104 )
			//	printf(" %d %d %d %d %d %d\n", iz, num[i], stack[i], current_id[i], middle_id[i], sites[st[i]]);
		}

		if (iy%warpWidth == 0)
		{
			//bitresult[(iy/(BANDWIDTH/bandNum))%2] = bitDeleted[(iy/(BANDWIDTH/bandNum))*res.x*res.z+ix*res.x+iz];
			bitresult = ~bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz];
			
		}



		// broadcast the bitresult to other thread
		bitresult = __shfl((int)bitresult, ((iy/warpWidth)%2)*warpWidth);

	

		count = 0;
		lasty = -1;
		

		//define last bit for each band
		//------------------------------------------------------
		mask = 3 << bandNum*(iy%warpWidth);
		mask = bitresult & mask;

		if (bitCount(mask) > 0) 
		{
			
			lasty = (iy/warpWidth)*warpSize 
				+ (32 - __ffs(__brev(mask))); // Get the most significant bit (__ffs return 1 - 32)
		}
		else
		{
			lasty = -1;
		}
		
		
		

		//link the last bit for each band ** can be optimize!!!!!!
		//------------------------------------------------------
		k = 0;

	
		//lasty = __shfl(lasty, max(0,iy%32-1));
		k = __shfl(lasty, (int)(iy%32-1));

		if ((int)(iy%32-1) >= 0) lasty = k;
	
	
		mask = __all(lasty >= 0);

	
		k = 0;

		while (mask == 0)
		{
			j = __shfl(lasty, (int)(iy%32-1));

			if (lasty < 0 && (int)(iy%32-1) >= 0) 
			{
				lasty = j;
			}
			k++;
			mask = __all(lasty >= 0); // make sure all the thread obtain the 
			if (k >= warpSize) break; // in case, but should not happen
		}
		


		if (iy%32 == 0)
			lasty = -1;

		
		/*if (ix == 256 && iz == 0 )
		{
			printf("=! %d %d %d %d %d %d %d \n", i, iy , iy%32, lasty, bitresult, num[0], num[1]);
		}*/
	

		k = -1;	
		//------------------------------------------------------
		// define stack (closest site on z axis and the pointer to previous site)
		for(i=0; i < bandNum ; i++)
		{
			if (GetBitPos((iy*bandNum+i)%32, bitresult))
			{


				if (iz < middle_id[i])
					stack[i] = ENCODE_STACK(current_id[i], lasty);
				else    
				{
					if (ind[i] < num[i])
					{
						j = sites[st[i]+ind[i]];
						ind[i]++;
						middle_id[i] = ceil((next_id[i] + j)/2.0);
						current_id[i] = next_id[i];
						next_id[i] = j;
					}
					else
					{
						middle_id[i] = LDNIMARKER;
						current_id[i] = next_id[i];
					}
					stack[i] = ENCODE_STACK(current_id[i], lasty);
					
					
				}
				count++;   
				   

				lasty = iy*bandNum + i;
				k = current_id[i]; // keep record the last current_id
	
			
				
			}
			else
			{
				
				//stack[i] = ENCODE_STACK(-1, lasty);
				stack[i] = ENCODE_STACK(k, lasty); // the last element in array = the last site
			}  
		
		}  
		//------------------------------------------------------
		// Calculate intersection point for each site

		/*if (ix == 256 && iz == 0 )
						{
							printf("=! %d %d %d %d %d %d %d %d %f \n", i, iy , iy%32, lasty, GET_STACK(mask),iy*bandNum+i,GET_STACK(stack[i]), mask, y_inter[i]);
						}*/

		/*if (ix == 256 && iz == 0 && iy == 65)
						{
							printf("=! %d %d %d %d %d %d %d %d %f \n", i, iy , iy%32, lasty, GET_STACK(stack[lasty%bandNum]),iy*bandNum+i,GET_STACK(stack[i]), mask, y_inter[i]);
						}*/


		y1 = -1 * res.x* res.x;
		for(i=0; i < bandNum ; i++)
		{
			
			lasty = GET_PTR(stack[i]);
			
			mask =  __shfl((int)stack[bandNum-1], (lasty%(warpSize*bandNum))/(bandNum));  // always get the last element in array for the last site of other thread

			if (GetBitPos((iy*bandNum+i)%32, bitresult))
			{
				if (lasty < res.x && GET_STACK(stack[i]) < res.x) // lasty < res.x  --> make sure current site is linking to previous site   
				{
					if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum)) // when previous site is not in another thread
					{
						y_inter[i] = interpointY(ix, lasty, GET_STACK(stack[lasty%bandNum]), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						
					}
					else
					{
						y_inter[i] = interpointY(ix, lasty, GET_STACK(mask), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						
						
					}

					
					
				}
				y1 = y_inter[i];
			}
			else
			{
				y_inter[i] = y1;
			}
		}

		/*if (ix == 256 && iz == 0 )
		{
			printf("=! %d %d %d %d %d %d %d %d %f \n", i, iy , iy%32, lasty, GET_STACK(stack[lasty%bandNum]),iy*bandNum+i,GET_STACK(stack[i]), mask, y_inter[i]);
		}*/
		//------------------------------------------------------
		// Start warp diverge
		//bitresult = 0; // clear in every thread for storing result later
		loopflag = true;
		int test = 0;
		while (loopflag)
		{
			loopflag = false;
			mask = 0;
			count = 0;
			y2 = 0.0;
			test++;
			for(i=0; i < bandNum ; i++)
			{
				if (count > 0) break;
				lasty = GET_PTR(stack[i]);
				y1 = __shfl(y_inter[bandNum-1],  (lasty%(warpSize*bandNum))/(bandNum));
				
				
				/*if (ix == 256 && iz == 500 && iy==81 )
				{
					printf("4=! %d %d %d %d %d %f\n", i, iy , iy%32, lasty, bitresult, y1);
				}*/

				//if (iz == 250 && ix == 431 && (iy*bandNum+i) == 96)
				//{
				//	printf("1=! %d %d %d %d %d %d %f %f\n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
				//}
				

				if (GetBitPos((iy*bandNum+i)%32, bitresult))
				{
					if (lasty < res.x )
					{
						if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum))
						{
							y1 = y_inter[lasty%bandNum];
							y2 = y_inter[i];
							
							if (y1 >= y2)
							{
								count++;
								if (count == 1) 	mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;

								//if (iz == 250 && ix == 431 )
								//{
								//	printf("2-1=! %d %d %d %d %d %d %f %f\n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								//}
							}
							else 
							{
								if (count == 0) mask = 0;

								
							}

						}
						else
						{						
							y2 = y_inter[i];
							if (y1 >= y2)
							{
								count++;
								if (count == 1)		mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;
								
								//if (iz == 250 && ix == 431)
								//{
								//	printf("2-2=! %d %d %d %d %d %d %f %f\n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								//}
								
							}
							else
							{
								if (count == 0) mask = 0;
								
							}
						}
					}
					else
					{
						if (count == 0) mask = 0;
					}
				}
				else
				{
					if (count == 0) mask = 0;
				}

			}

			lasty = mask;
			
			mask = __ballot(count > 0);

			//if (ix == 256 && iz == 500 && iy == 80)
			//{
			//	printf("3=! %d %d \n", mask, count);
			//}
			

			// update the stack

			if (mask > 0)
			{
				loopflag = true;
				k = __ffs(mask);

				lasty = __shfl(lasty,  k-1);

				i = GET_STACK(lasty);
				j = GET_PTR(lasty);

				/*if (ix == 254 && iz == 500 )
				{
					printf("4=! %d %d %d %d %d %d %d\n", k, iy , iy%32, lasty, bitresult, i, j);
				}*/

				
				lasty = __shfl((int)stack[i%bandNum],  (i%(warpSize*bandNum))/(bandNum));
				k = GET_PTR(lasty);
				lasty = __shfl((int)stack[k%bandNum],  (k%(warpSize*bandNum))/(bandNum));


				if (iy == j/bandNum)
				{
					stack[j%bandNum] = ENCODE_PTR(stack[j%bandNum], k);
					y_inter[j%bandNum] =  interpointY(ix, k, GET_STACK(lasty), ix, j, GET_STACK(stack[j%bandNum]), ix, iz) ;


					for(count=j%bandNum+1; count < bandNum ; count++)
					{
						

						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[j%bandNum];
						else break;

						/*if (ix == 254 && iz == 500 )
						{
							printf("4=! %d %d %d %d %d\n", j, iy , iy%32, count, GET_PTR(stack[count]));
						}*/
					}

				}
				

				if (iy == i/bandNum)
				{
					bitresult = bitresult & ~(SetBitPos(i%32));
					y_inter[i%bandNum] = y_inter[(max(0,i-1))%bandNum];
				}


				mask = __shfl((int)bitresult,  (i%(warpSize*bandNum))/(bandNum));

				if (i%(warpSize*bandNum)/warpSize == iy%(warpSize)/(warpWidth))
				{
					bitresult = mask;	
				}

			}
			else break;

		}
		
		if (iy%warpWidth == 0)
		{
			//bitresult[(iy/(BANDWIDTH/bandNum))%2] = bitDeleted[(iy/(BANDWIDTH/bandNum))*res.x*res.z+ix*res.x+iz];
			bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz] = ~bitresult;

		}

		
/*
		while (mask != 0)
		{
			lasty = __ffs(mask);
			count = 0;
			
			for(i=0; i < bandNum ; i++)
			{
				j = GET_PTR(stack[i]);
				if ( j > 0)
				{
					lasty = __shfl(y_inter[j%2],  (j%(warpSize*bandNum))/(bandNum));

					if (y_inter[i] < lasty)
					{	
						j = GET_PTR(lasty);
						lasty = __shfl((int)stack[j%2],  (j%(warpSize*bandNum))/(bandNum));
						y_inter[i] = interpointY(ix, j, GET_STACK(lasty), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						lasty = GET_STACK(stack[i]);
						stack[i] = ENCODE_STACK(lasty, k);
						count++;
						break;
					}
				}
			}
			
			mask = __ballot(count > 0);
		}


		//------------------------------------------------------
		// Store the result 
		bitresult = 0;
		lasty = -1;
		if (iy%(warpSize/bandNum) == 0)
		{
			k = (iy/(warpSize/bandNum))%(bandNum);
						
			
			
			lasty = -1;
			for(j=warpSize/bandNum-1; j > 0 ; j--)
			{
				for(i=bandNum; i > 0 ; i--)
				{
					mask = __shfl((int)stack[i], k*(warpSize/bandNum)+j);
					lasty = GET_PTR(mask);
					if (lasty >= 0)
					{
						bitresult = bitresult | SetBitPos(j*bandNum+i);
						break;
					}
				}
				if (lasty >= 0) break;
			}


			while (lasty >= 0)
			{
				j = lasty%(warpSize*bandNum)/bandNum;

				if (j/(warpSize/bandNum) != k) break;
				
				mask = __shfl((int)stack[lasty%2], j);
				lasty = GET_PTR(mask);
				if (lasty > 0)
				{
					bitresult = bitresult | SetBitPos(j*bandNum+(lasty%bandNum));
				}
			}
		
			if (k+1 < bandNum)
			{
				lasty = __shfl(lasty, (k+1)*(warpSize/bandNum));
				if (lasty > 0)
				{
					bitresult = bitresult &  (!(SetBitPos(lasty%32)-1));
				}
			}


			bitDeleted[(iy/(warpSize/bandNum))*res.x*res.z+ix*res.x+iz] = bitresult;


		}*/




		tid += chunksize;
	}
}



__global__ void LDNIDistanceField__MaurerAxisInY(unsigned int *bitDeleted, unsigned short *sites, unsigned int *sites_index, int3 res, int offsetPixel, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	short middle_id[BANDWIDTH], ind[BANDWIDTH], current_id[BANDWIDTH], next_id[BANDWIDTH];
	short num[BANDWIDTH], i, j, k;
	unsigned int st[BANDWIDTH];
	short stack[BANDWIDTH], count;
	//unsigned int bitresult[BANDWIDTH];
	unsigned int bitresult;
	float y1, y2;
	short ptr[BANDWIDTH];

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);

		bitresult = 0;
		if (iz == 0)
		{
			j = iy*BANDWIDTH;
			for(i=0; i < BANDWIDTH; i++)
			{
				st[i] = sites_index[(j+i)*res.x+ix];
				num[i] = sites_index[(j+i)*res.x+ix+1]-st[i];

				
				if (num[i] > 1) 
				{
					current_id[i] = sites[st[i]];
					next_id[i] = sites[st[i]+1];
					ind[i] = 2;
					middle_id[i] = (short)ceil((current_id[i]+next_id[i])/2.0);
					
				}
				else if (num[i] == 1)
				{
					current_id[i] = sites[st[i]];
					ind[i] = 1;
					middle_id[i] = LDNIMARKER;

				}
				else	
				{
					middle_id[i] = -1;
					current_id[i] = LDNIMARKER;
					next_id[i]= LDNIMARKER;
					ind[i] = 0;
				}
				
			}
		}

		count = 0;
		k = -1;
		for(i=0; i < BANDWIDTH ; i++)
		{
			
			if (num[i]>0)
			{
				ptr[i] = k;
				if (iz < middle_id[i])
					stack[i] = current_id[i];
				else
				{
					if (ind[i] < num[i])
					{
						k = sites[st[i]+ind[i]];
						ind[i]++;
						middle_id[i] = ceil((next_id[i] + k)/2.0);
						current_id[i] = next_id[i];
						next_id[i] = k;
					}
					else
					{
						middle_id[i] = LDNIMARKER;
						current_id[i] = next_id[i];
					}
					stack[i] = current_id[i];
				}
				count++;
				k = i;

				
			}
			else
			{
				stack[i] = -1;
				ptr[i] = k;
				bitresult = bitresult | SetBitPos(i);
			}

			//if (iz == 250 && ix == 431 && iy == 3)
			/*if (ix == 65 && iy == 3 && i == 8 )
					printf(" %d %d %d %d %d %d\n", iz, num[i], stack[i], current_id[i], middle_id[i], sites[st[i]]);*/
		}


		if (count > 2)
		{
			k=0;
			for(i=0; i < BANDWIDTH ; i++)
			{
				if (stack[i] > 0)
				{
					if (k < 2)
					{
						k++;
						continue;
					}
					
					while (k>=2)
					{
						y1 =  interpointY(ix, j+ptr[ptr[i]], stack[ptr[ptr[i]]], ix, j+ptr[i], stack[ptr[i]], ix, iz) ;
						y2 =  interpointY(ix, j+ptr[i], stack[ptr[i]], ix, j+i, stack[i], ix, iz) ;
				

						


						if (y1 < y2)
							break;

						//if (iz == 250 && ix == 431 && iy < 4)
						//{
						//	printf("ptr %d %f %f %d %d %d\n", j+i, y1, y2, k, j+ptr[i], stack[ptr[i]]);
							//printf("y1 : %d %d %d %d %d %d %d %d \n", ix, j+ptr[ptr[i]], stack[ptr[ptr[i]]], ix, j+ptr[i], stack[ptr[i]], ix, iz);
							//printf("y2 : %d %d %d %d %d %d %d %d \n", ix, j+ptr[i], stack[ptr[i]], ix, j+i, stack[i], ix, iz);
						//}
						//if (ix == 256 && (j+i) == 178 && iz == 0)
						

						k--;
						stack[ptr[i]] = -1;
						bitresult = bitresult | SetBitPos(ptr[i]);
						ptr[i] = ptr[ptr[i]];

					}

					k++;

				}

			}


			bitDeleted[iy*res.x*res.z+ix*res.x+iz] = bitresult;

			//if (iz == 250 && ix == 431 && iy < 4)
			//if (ix == 256 && iz ==0)
			//	printf("--------------%d %d \n", iy, bitresult, count );
			//for(i=0; i < BANDWIDTH ; i++)
			//{
			//	bitDeleted[iy*res*res+ix*res+iz]
			//}
		}
		else
		{
			bitDeleted[iy*res.x*res.z+ix*res.x+iz] = bitresult;
			//if (ix == 256 && iz ==0)
			//	printf("--------------%d %d %d\n", iy, bitresult, count );
		}

		
		
		

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void LDNIDistanceField__kernelMergeBandsX_32(unsigned int *bitDeleted, unsigned int *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	//unsigned int ix,iy,iz;
	int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	int  current_id[32], next_id[16];
	float y_inter[32];
	short ind[32], num[32], i;
	int j, k, count;
	unsigned int st[32];
	int stack[32]; // stack + ptr
	unsigned int bitresult = 0;
	int lasty;
	float y1, y2;
	unsigned int mask = 0; 
	bool loopflag = false;
	int middle_id[32], temp;
	

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);
		
		
		if (iz == 0)
		{
			j = iy*bandNum;
			for(i=0; i < bandNum; i++)
			{
				st[i] = sites_index[(j+i)*res.x+ix];
				num[i] = sites_index[(j+i)*res.x+ix+1]-st[i];


				if (num[i] > 1) 
				{
					current_id[i] = sites[st[i]];
					next_id[i] = sites[st[i]+1];
					ind[i] = 2;
					middle_id[i] = middlepointY(current_id[i], next_id[i], ix);

				}
				else if (num[i] == 1)
				{
					current_id[i] = sites[st[i]];
					ind[i] = 1;
					middle_id[i] = LDNIMARKER;

				}
				else	
				{
					middle_id[i] = -1;
					current_id[i] = LDNIMARKER;
					next_id[i]= LDNIMARKER;
					ind[i] = 0;
				}

				//if (ix == 250 && j+i == 78 && iz == 0)
				//{
				//printf("stack %d %d %d %d %d %d\n", i, iz, num[i], GET_STACK(stack[i]), GET_PTR(stack[i]));
				//	for(int test = 0; test < num[i] ; test++)
				//		printf("%d %d %d %d %d %d\n", num[i],  middlepointY(sites[st[i]]+test, sites[st[i]+test+1], ix), test, iy*BANDWIDTH+i, GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]));

				//}

			}
		}

		if (iy%warpWidth == 0)
		{
			//bitresult[(iy/(BANDWIDTH/bandNum))%2] = bitDeleted[(iy/(BANDWIDTH/bandNum))*res.x*res.z+ix*res.x+iz];
			bitresult = ~bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz];
			
		}



		// broadcast the bitresult to other thread
		bitresult = __shfl((int)bitresult, ((iy/warpWidth)%bandNum)*warpWidth);

	

		count = 0;
		lasty = -1;
		

		//define last bit for each band
		//------------------------------------------------------
		//mask = 65535 << bandNum*(iy%warpWidth);
		mask = bitresult;// & mask;

		if (__popc(mask) > 0) 
		{
			
			lasty = (iy/warpWidth)*warpSize 
				+ (32 - __ffs(__brev(mask))); // Get the most significant bit (__ffs return 1 - 32)
		}
		else
		{
			lasty = -1;
		}
		
		
		

		//link the last bit for each band ** can be optimize!!!!!!
		//------------------------------------------------------
		k = 0;

	
		//lasty = __shfl(lasty, max(0,iy%32-1));
		k = __shfl(lasty, (int)(iy%32-1));

		if ((int)(iy%32-1) >= 0) lasty = k;
	
	
		mask = __all(lasty >= 0);

	
		k = 0;

		while (mask == 0)
		{
			j = __shfl(lasty, (int)(iy%32-1));

			if (lasty < 0 && (int)(iy%32-1) >= 0) 
			{
				lasty = j;
			}
			k++;
			mask = __all(lasty >= 0); // make sure all the thread obtain the 
			if (k >= warpSize) break; // in case, but should not happen
		}
		


		if (iy%32 == 0)
			lasty = -1;
	

		k = -1;	temp = 0;
		//------------------------------------------------------
		// define stack (closest site on z axis and the pointer to previous site)
		for(i=0; i < bandNum ; i++)
		{
			//if (GetBitPos((iy*bandNum+i)%32, bitresult))
			//{
				if ((int)iz < middle_id[i])
				{
					if (GetBitPos((iy*bandNum+i)%32, bitresult))
					{
					stack[i] = ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty);
					count++; 
					lasty = iy*bandNum + i;
					k = current_id[i]; // keep record the last current_id
					}
					else
					stack[i] = ENCODE_STACK_3(GET_STACK(k), GET_PTR(k), lasty); // the last element in array = the last site
				}
				else    
				{
					if (ind[i] < num[i])
					{
						j = sites[st[i]+ind[i]];
						ind[i]++;
						temp = middlepointY(next_id[i], j, ix);
						while (temp <= middle_id[i])
						{
							next_id[i] = j;
							j = sites[st[i]+ind[i]];
							ind[i]++;
							temp = middlepointY(next_id[i], j, ix);

						}

						middle_id[i] = temp;
						current_id[i] = next_id[i];
						next_id[i] = j;
					}
					else
					{
						middle_id[i] = LDNIMARKER;
						current_id[i] = next_id[i];
					}

					if (GetBitPos((iy*bandNum+i)%32, bitresult))
					{
						stack[i] = ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty);
						count++; 
						lasty = iy*bandNum + i;
						k = current_id[i]; // keep record the last current_id
					}
					else
					stack[i] = ENCODE_STACK_3(GET_STACK(k), GET_PTR(k), lasty); // the last element in array = the last site

				}
				  

				//if (ix == 250 && iy*bandNum+i == 78)
				//{
				//	printf("^^^^ %d %d %d %d %d %d %d \n", iz,  middle_id[i], GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty, ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty), stack[i] );

				//for(int test = 0; test < num[i]; test++)
				//	printf("%d %d %d %d %d %d\n", num[i],  middlepointY(sites[st[i]]+test, sites[st[i]+test+1], ix), test, iy*bandNum+i, GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]));
				//printf("%d %d \n",GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]) );
				//}


			//	lasty = iy*bandNum + i;
				//k = current_id[i]; // keep record the last current_id

		}  

		//------------------------------------------------------
		// Calculate intersection point for each site

		y1 = -1 * res.x* res.x;
		for(i=0; i < bandNum ; i++)
		{
			
			//lasty = GET_PTR(stack[i]);
			lasty = GET_Z(stack[i]);
			
			mask =  __shfl((int)stack[bandNum-1], (lasty%(warpSize*bandNum))/(bandNum));  // always get the last element in array for the last site of other thread

			if (GetBitPos((iy*bandNum+i)%32, bitresult))
			{
				//if (lasty < res.x && GET_STACK(stack[i]) < res.x) // lasty < res.x  --> make sure current site is linking to previous site   
				if (lasty < res.x && GET_X(stack[i]) < res.x && GET_Y(stack[i]) < res.x) // lasty < res.x  --> make sure current site is linking to previous site   
				{
					if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum)) // when previous site is not in another thread
					{
						//y_inter[i] = interpointY(ix, lasty, GET_STACK(stack[lasty%bandNum]), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;

						y_inter[i] = interpointY(GET_X(stack[lasty%bandNum]), lasty, GET_Y(stack[lasty%bandNum]), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix) ;
						/*if (iz == 111 && ix == 250 && (iy == 48 || iy == 49))
						{
							printf("2-a=! %d %d %d %d %d %d %d %d %d %d %f\n", i, iy , GET_X(stack[lasty%bandNum]), lasty, GET_Y(stack[lasty%bandNum]), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix, y_inter[i]);
						}*/
						
					}
					else
					{
						//y_inter[i] = interpointY(ix, lasty, GET_STACK(mask), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						y_inter[i] = interpointY(GET_X(mask), lasty, GET_Y(mask), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix) ;
						/*if (iz == 111 && ix == 250 && (iy == 48 || iy == 49))
						{
							printf("2-b=! %d %d %d %d %d %d %d %d %d %d %f\n", i, iy , GET_X(mask), lasty, GET_Y(mask), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix, y_inter[i]);
						}*/
						
						
					}

					
					
				}
				else
				{
					y_inter[i] = y1;
				}
				y1 = y_inter[i];
			}
			else
			{
				y_inter[i] = y1;
			}
		}
		
		//------------------------------------------------------
		// Start warp diverge
		//bitresult = 0; // clear in every thread for storing result later
		loopflag = true;
		
		while (loopflag)
		{
			loopflag = false;
			mask = 0;
			count = 0;
			y2 = 0.0;
			
			for(i=0; i < bandNum ; i++)
			{
				if (count > 0) break;
				lasty = GET_Z(stack[i]);
				y1 = __shfl(y_inter[bandNum-1],  (lasty%(warpSize*bandNum))/(bandNum));
				

				if (GetBitPos((iy*bandNum+i)%32, bitresult))
				{
					if (lasty < res.x )
					{
						if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum))
						{
							y1 = y_inter[lasty%bandNum];
							y2 = y_inter[i];
							
							if (y1 >= y2)
							{
								count++;
								if (count == 1) 	mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;

								if (iz == 311 && ix == 256 )
								{
									printf("2-1=! %d %d %d %d %d %d %f %f \n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								}

								/*if (iz == 111 && ix == 250 && (j+i) < 64 && (j+i)>=32)
								{
									printf("test test?  %d %d %f %f %d %d %d \n y1 : %d %d %d %d %d %d %d %d \n y2 : %d %d %d %d %d %d %d %d \n", ptr[i], i, y1, y2, GET_PTR(stack[ptr[i]]), j+ptr[i], GET_STACK(stack[ptr[i]]),  GET_STACK(stack[ptr[ptr[i]]]), j+ptr[ptr[i]], GET_PTR(stack[ptr[ptr[i]]]), GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), ix, iz,  GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), GET_STACK(stack[i]), j+i, GET_PTR(stack[i]), ix, iz);

								}*/
							}
							else 
							{
								if (count == 0) mask = 0;

								
							}

						}
						else
						{						
							y2 = y_inter[i];
							if (y1 >= y2)
							{
								count++;
								if (count == 1)		mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;
								
								if (iz == 311 && ix == 256 )
								{
									printf("2-2=! %d %d %d %d %d %d %f %f \n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								}
								
							}
							else
							{
								if (count == 0) mask = 0;
								
							}
						}
					}
					else
					{
						if (count == 0) mask = 0;
					}
				}
				else
				{
					if (count == 0) mask = 0;
				}

			}

			lasty = mask;
			
			mask = __ballot(count > 0);

			// update the stack

			if (mask > 0)
			{
				loopflag = true;
				k = __ffs(mask);

				lasty = __shfl(lasty,  k-1);

				i = GET_STACK(lasty);
				j = GET_PTR(lasty);

			
				lasty = __shfl((int)stack[i%bandNum],  (i%(warpSize*bandNum))/(bandNum));
				k = GET_Z(lasty);
				lasty = __shfl((int)stack[k%bandNum],  (k%(warpSize*bandNum))/(bandNum));


				if (iy == j/bandNum)
				{
					
					stack[j%bandNum] = ENCODE_Z(stack[j%bandNum], k);
					y_inter[j%bandNum] =  interpointY(GET_X(lasty), k, GET_Y(lasty), GET_X(stack[j%bandNum]), j, GET_Y(stack[j%bandNum]), iz, ix) ;


					for(count=j%bandNum+1; count < bandNum ; count++)
					{
						

						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[j%bandNum];
						else break;
					}

				}
				

				if (iy == i/bandNum)
				{
					bitresult = bitresult & ~(SetBitPos(i%32));

					if ((i-1) >= 0)
						y_inter[i%bandNum] = y_inter[(max(0,i-1))%bandNum];

					for(count=i%bandNum+1; count < bandNum ; count++)
					{
						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[i%bandNum];
						else break;
					}
				}


				mask = __shfl((int)bitresult,  (i%(warpSize*bandNum))/(bandNum));

				if (i%(warpSize*bandNum)/warpSize == iy%(warpSize)/(warpWidth))
				{
					bitresult = mask;	
				}

			}
			else break;

		}
		
		if (iy%warpWidth == 0)
		{
			bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz] = ~bitresult;
		}

		tid += chunksize;
	}
}

__global__ void LDNIDistanceField__kernelMergeBandsX_16(unsigned int *bitDeleted, unsigned int *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	//unsigned int ix,iy,iz;
	int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	int  current_id[16], next_id[16];
	float y_inter[16];
	short ind[16], num[16], i;
	int j, k, count;
	unsigned int st[16];
	int stack[16]; // stack + ptr
	unsigned int bitresult = 0;
	int lasty;
	float y1, y2;
	unsigned int mask = 0; 
	bool loopflag = false;
	int middle_id[16], temp;
	

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);
		
		
		if (iz == 0)
		{
			j = iy*bandNum;
			for(i=0; i < bandNum; i++)
			{
				st[i] = sites_index[(j+i)*res.x+ix];
				num[i] = sites_index[(j+i)*res.x+ix+1]-st[i];


				if (num[i] > 1) 
				{
					current_id[i] = sites[st[i]];
					next_id[i] = sites[st[i]+1];
					ind[i] = 2;
					middle_id[i] = middlepointY(current_id[i], next_id[i], ix);

				}
				else if (num[i] == 1)
				{
					current_id[i] = sites[st[i]];
					ind[i] = 1;
					middle_id[i] = LDNIMARKER;

				}
				else	
				{
					middle_id[i] = -1;
					current_id[i] = LDNIMARKER;
					next_id[i]= LDNIMARKER;
					ind[i] = 0;
				}

				//if (ix == 250 && j+i == 78 && iz == 0)
				//{
				//printf("stack %d %d %d %d %d %d\n", i, iz, num[i], GET_STACK(stack[i]), GET_PTR(stack[i]));
				//	for(int test = 0; test < num[i] ; test++)
				//		printf("%d %d %d %d %d %d\n", num[i],  middlepointY(sites[st[i]]+test, sites[st[i]+test+1], ix), test, iy*BANDWIDTH+i, GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]));

				//}

			}
		}

		if (iy%warpWidth == 0)
		{
			//bitresult[(iy/(BANDWIDTH/bandNum))%2] = bitDeleted[(iy/(BANDWIDTH/bandNum))*res.x*res.z+ix*res.x+iz];
			bitresult = ~bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz];
			
		}



		// broadcast the bitresult to other thread
		bitresult = __shfl((int)bitresult, ((iy/warpWidth)%bandNum)*warpWidth);

	

		count = 0;
		lasty = -1;
		

		//define last bit for each band
		//------------------------------------------------------
		mask = 65535 << bandNum*(iy%warpWidth);
		mask = bitresult & mask;

		if (__popc(mask) > 0) 
		{
			
			lasty = (iy/warpWidth)*warpSize 
				+ (32 - __ffs(__brev(mask))); // Get the most significant bit (__ffs return 1 - 32)
		}
		else
		{
			lasty = -1;
		}
		
		
		

		//link the last bit for each band ** can be optimize!!!!!!
		//------------------------------------------------------
		k = 0;

	
		//lasty = __shfl(lasty, max(0,iy%32-1));
		k = __shfl(lasty, (int)(iy%32-1));

		if ((int)(iy%32-1) >= 0) lasty = k;
	
	
		mask = __all(lasty >= 0);

	
		k = 0;

		while (mask == 0)
		{
			j = __shfl(lasty, (int)(iy%32-1));

			if (lasty < 0 && (int)(iy%32-1) >= 0) 
			{
				lasty = j;
			}
			k++;
			mask = __all(lasty >= 0); // make sure all the thread obtain the 
			if (k >= warpSize) break; // in case, but should not happen
		}
		


		if (iy%32 == 0)
			lasty = -1;
	

		k = -1;	temp = 0;
		//------------------------------------------------------
		// define stack (closest site on z axis and the pointer to previous site)
		for(i=0; i < bandNum ; i++)
		{
			//if (GetBitPos((iy*bandNum+i)%32, bitresult))
			//{
				if ((int)iz < middle_id[i])
				{
					if (GetBitPos((iy*bandNum+i)%32, bitresult))
					{
					stack[i] = ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty);
					count++; 
					lasty = iy*bandNum + i;
					k = current_id[i]; // keep record the last current_id
					}
					else
					stack[i] = ENCODE_STACK_3(GET_STACK(k), GET_PTR(k), lasty); // the last element in array = the last site
				}
				else    
				{
					if (ind[i] < num[i])
					{
						j = sites[st[i]+ind[i]];
						ind[i]++;
						temp = middlepointY(next_id[i], j, ix);
						while (temp <= middle_id[i])
						{
							next_id[i] = j;
							j = sites[st[i]+ind[i]];
							ind[i]++;
							temp = middlepointY(next_id[i], j, ix);

						}

						middle_id[i] = temp;
						current_id[i] = next_id[i];
						next_id[i] = j;
					}
					else
					{
						middle_id[i] = LDNIMARKER;
						current_id[i] = next_id[i];
					}

					if (GetBitPos((iy*bandNum+i)%32, bitresult))
					{
						stack[i] = ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty);
						count++; 
						lasty = iy*bandNum + i;
						k = current_id[i]; // keep record the last current_id
					}
					else
					stack[i] = ENCODE_STACK_3(GET_STACK(k), GET_PTR(k), lasty); // the last element in array = the last site

				}
				  

				//if (ix == 250 && iy*bandNum+i == 78)
				//{
				//	printf("^^^^ %d %d %d %d %d %d %d \n", iz,  middle_id[i], GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty, ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty), stack[i] );

				//for(int test = 0; test < num[i]; test++)
				//	printf("%d %d %d %d %d %d\n", num[i],  middlepointY(sites[st[i]]+test, sites[st[i]+test+1], ix), test, iy*bandNum+i, GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]));
				//printf("%d %d \n",GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]) );
				//}


			//	lasty = iy*bandNum + i;
				//k = current_id[i]; // keep record the last current_id

		}  

		//------------------------------------------------------
		// Calculate intersection point for each site

		y1 = -1 * res.x* res.x;
		for(i=0; i < bandNum ; i++)
		{
			
			//lasty = GET_PTR(stack[i]);
			lasty = GET_Z(stack[i]);
			
			mask =  __shfl((int)stack[bandNum-1], (lasty%(warpSize*bandNum))/(bandNum));  // always get the last element in array for the last site of other thread

			if (GetBitPos((iy*bandNum+i)%32, bitresult))
			{
				//if (lasty < res.x && GET_STACK(stack[i]) < res.x) // lasty < res.x  --> make sure current site is linking to previous site   
				if (lasty < res.x && GET_X(stack[i]) < res.x && GET_Y(stack[i]) < res.x) // lasty < res.x  --> make sure current site is linking to previous site   
				{
					if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum)) // when previous site is not in another thread
					{
						//y_inter[i] = interpointY(ix, lasty, GET_STACK(stack[lasty%bandNum]), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;

						y_inter[i] = interpointY(GET_X(stack[lasty%bandNum]), lasty, GET_Y(stack[lasty%bandNum]), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix) ;
						/*if (iz == 111 && ix == 250 && (iy == 48 || iy == 49))
						{
							printf("2-a=! %d %d %d %d %d %d %d %d %d %d %f\n", i, iy , GET_X(stack[lasty%bandNum]), lasty, GET_Y(stack[lasty%bandNum]), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix, y_inter[i]);
						}*/
						
					}
					else
					{
						//y_inter[i] = interpointY(ix, lasty, GET_STACK(mask), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						y_inter[i] = interpointY(GET_X(mask), lasty, GET_Y(mask), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix) ;
						/*if (iz == 111 && ix == 250 && (iy == 48 || iy == 49))
						{
							printf("2-b=! %d %d %d %d %d %d %d %d %d %d %f\n", i, iy , GET_X(mask), lasty, GET_Y(mask), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix, y_inter[i]);
						}*/
						
						
					}

					
					
				}
				else
				{
					y_inter[i] = y1;
				}
				y1 = y_inter[i];
			}
			else
			{
				y_inter[i] = y1;
			}
		}
		
		//------------------------------------------------------
		// Start warp diverge
		//bitresult = 0; // clear in every thread for storing result later
		loopflag = true;
		
		while (loopflag)
		{
			loopflag = false;
			mask = 0;
			count = 0;
			y2 = 0.0;
			
			for(i=0; i < bandNum ; i++)
			{
				if (count > 0) break;
				lasty = GET_Z(stack[i]);
				y1 = __shfl(y_inter[bandNum-1],  (lasty%(warpSize*bandNum))/(bandNum));
				

				if (GetBitPos((iy*bandNum+i)%32, bitresult))
				{
					if (lasty < res.x )
					{
						if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum))
						{
							y1 = y_inter[lasty%bandNum];
							y2 = y_inter[i];
							
							if (y1 >= y2)
							{
								count++;
								if (count == 1) 	mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;

								/*if (iz == 311 && ix == 256 )
								{
									printf("2-1=! %d %d %d %d %d %d %f %f \n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								}*/

								/*if (iz == 111 && ix == 250 && (j+i) < 64 && (j+i)>=32)
								{
									printf("test test?  %d %d %f %f %d %d %d \n y1 : %d %d %d %d %d %d %d %d \n y2 : %d %d %d %d %d %d %d %d \n", ptr[i], i, y1, y2, GET_PTR(stack[ptr[i]]), j+ptr[i], GET_STACK(stack[ptr[i]]),  GET_STACK(stack[ptr[ptr[i]]]), j+ptr[ptr[i]], GET_PTR(stack[ptr[ptr[i]]]), GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), ix, iz,  GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), GET_STACK(stack[i]), j+i, GET_PTR(stack[i]), ix, iz);

								}*/
							}
							else 
							{
								if (count == 0) mask = 0;

								
							}

						}
						else
						{						
							y2 = y_inter[i];
							if (y1 >= y2)
							{
								count++;
								if (count == 1)		mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;
								
								/*if (iz == 311 && ix == 256 )
								{
									printf("2-2=! %d %d %d %d %d %d %f %f \n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								}*/
								
							}
							else
							{
								if (count == 0) mask = 0;
								
							}
						}
					}
					else
					{
						if (count == 0) mask = 0;
					}
				}
				else
				{
					if (count == 0) mask = 0;
				}

			}

			lasty = mask;
			
			mask = __ballot(count > 0);

			// update the stack

			if (mask > 0)
			{
				loopflag = true;
				k = __ffs(mask);

				lasty = __shfl(lasty,  k-1);

				i = GET_STACK(lasty);
				j = GET_PTR(lasty);

			
				lasty = __shfl((int)stack[i%bandNum],  (i%(warpSize*bandNum))/(bandNum));
				k = GET_Z(lasty);
				lasty = __shfl((int)stack[k%bandNum],  (k%(warpSize*bandNum))/(bandNum));


				if (iy == j/bandNum)
				{
					
					stack[j%bandNum] = ENCODE_Z(stack[j%bandNum], k);
					y_inter[j%bandNum] =  interpointY(GET_X(lasty), k, GET_Y(lasty), GET_X(stack[j%bandNum]), j, GET_Y(stack[j%bandNum]), iz, ix) ;


					for(count=j%bandNum+1; count < bandNum ; count++)
					{
						

						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[j%bandNum];
						else break;
					}

				}
				

				if (iy == i/bandNum)
				{
					bitresult = bitresult & ~(SetBitPos(i%32));

					if ((i-1) >= 0)
						y_inter[i%bandNum] = y_inter[(max(0,i-1))%bandNum];

					for(count=i%bandNum+1; count < bandNum ; count++)
					{
						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[i%bandNum];
						else break;
					}
				}


				mask = __shfl((int)bitresult,  (i%(warpSize*bandNum))/(bandNum));

				if (i%(warpSize*bandNum)/warpSize == iy%(warpSize)/(warpWidth))
				{
					bitresult = mask;	
				}

			}
			else break;

		}
		
		if (iy%warpWidth == 0)
		{
			bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz] = ~bitresult;
		}

		tid += chunksize;
	}
}

__global__ void LDNIDistanceField__kernelMergeBandsX_8(unsigned int *bitDeleted, unsigned int *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	//unsigned int ix,iy,iz;
	int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	int  current_id[8], next_id[8];
	float y_inter[8];
	short ind[8], num[8], i;
	int j, k, count;
	unsigned int st[8];
	int stack[8]; // stack + ptr
	unsigned int bitresult = 0;
	int lasty;
	float y1, y2;
	unsigned int mask = 0; 
	bool loopflag = false;
	int middle_id[8], temp;
	

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);
		
		
		if (iz == 0)
		{
			j = iy*bandNum;
			for(i=0; i < bandNum; i++)
			{
				st[i] = sites_index[(j+i)*res.x+ix];
				num[i] = sites_index[(j+i)*res.x+ix+1]-st[i];


				if (num[i] > 1) 
				{
					current_id[i] = sites[st[i]];
					next_id[i] = sites[st[i]+1];
					ind[i] = 2;
					middle_id[i] = middlepointY(current_id[i], next_id[i], ix);

				}
				else if (num[i] == 1)
				{
					current_id[i] = sites[st[i]];
					ind[i] = 1;
					middle_id[i] = LDNIMARKER;

				}
				else	
				{
					middle_id[i] = -1;
					current_id[i] = LDNIMARKER;
					next_id[i]= LDNIMARKER;
					ind[i] = 0;
				}

				//if (ix == 250 && j+i == 78 && iz == 0)
				//{
				//printf("stack %d %d %d %d %d %d\n", i, iz, num[i], GET_STACK(stack[i]), GET_PTR(stack[i]));
				//	for(int test = 0; test < num[i] ; test++)
				//		printf("%d %d %d %d %d %d\n", num[i],  middlepointY(sites[st[i]]+test, sites[st[i]+test+1], ix), test, iy*BANDWIDTH+i, GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]));

				//}

			}
		}

		if (iy%warpWidth == 0)
		{
			//bitresult[(iy/(BANDWIDTH/bandNum))%2] = bitDeleted[(iy/(BANDWIDTH/bandNum))*res.x*res.z+ix*res.x+iz];
			bitresult = ~bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz];
			
		}



		// broadcast the bitresult to other thread
		bitresult = __shfl((int)bitresult, ((iy/warpWidth)%bandNum)*warpWidth);

	

		count = 0;
		lasty = -1;
		

		//define last bit for each band
		//------------------------------------------------------
		mask = 255 << bandNum*(iy%warpWidth);
		mask = bitresult & mask;

		if (__popc(mask) > 0) 
		{
			
			lasty = (iy/warpWidth)*warpSize 
				+ (32 - __ffs(__brev(mask))); // Get the most significant bit (__ffs return 1 - 32)
		}
		else
		{
			lasty = -1;
		}
		
		
		

		//link the last bit for each band ** can be optimize!!!!!!
		//------------------------------------------------------
		k = 0;

	
		//lasty = __shfl(lasty, max(0,iy%32-1));
		k = __shfl(lasty, (int)(iy%32-1));

		if ((int)(iy%32-1) >= 0) lasty = k;
	
	
		mask = __all(lasty >= 0);

	
		k = 0;

		while (mask == 0)
		{
			j = __shfl(lasty, (int)(iy%32-1));

			if (lasty < 0 && (int)(iy%32-1) >= 0) 
			{
				lasty = j;
			}
			k++;
			mask = __all(lasty >= 0); // make sure all the thread obtain the 
			if (k >= warpSize) break; // in case, but should not happen
		}
		


		if (iy%32 == 0)
			lasty = -1;
	

		k = -1;	temp = 0;
		//------------------------------------------------------
		// define stack (closest site on z axis and the pointer to previous site)
		for(i=0; i < bandNum ; i++)
		{
			//if (GetBitPos((iy*bandNum+i)%32, bitresult))
			//{
				if ((int)iz < middle_id[i])
				{
					if (GetBitPos((iy*bandNum+i)%32, bitresult))
					{
					stack[i] = ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty);
					count++; 
					lasty = iy*bandNum + i;
					k = current_id[i]; // keep record the last current_id
					}
					else
					stack[i] = ENCODE_STACK_3(GET_STACK(k), GET_PTR(k), lasty); // the last element in array = the last site
				}
				else    
				{
					if (ind[i] < num[i])
					{
						j = sites[st[i]+ind[i]];
						ind[i]++;
						temp = middlepointY(next_id[i], j, ix);
						while (temp <= middle_id[i])
						{
							next_id[i] = j;
							j = sites[st[i]+ind[i]];
							ind[i]++;
							temp = middlepointY(next_id[i], j, ix);

						}

						middle_id[i] = temp;
						current_id[i] = next_id[i];
						next_id[i] = j;
					}
					else
					{
						middle_id[i] = LDNIMARKER;
						current_id[i] = next_id[i];
					}

					if (GetBitPos((iy*bandNum+i)%32, bitresult))
					{
						stack[i] = ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty);
						count++; 
						lasty = iy*bandNum + i;
						k = current_id[i]; // keep record the last current_id
					}
					else
					stack[i] = ENCODE_STACK_3(GET_STACK(k), GET_PTR(k), lasty); // the last element in array = the last site

				}
				  

				//if (ix == 250 && iy*bandNum+i == 78)
				//{
				//	printf("^^^^ %d %d %d %d %d %d %d \n", iz,  middle_id[i], GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty, ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty), stack[i] );

				//for(int test = 0; test < num[i]; test++)
				//	printf("%d %d %d %d %d %d\n", num[i],  middlepointY(sites[st[i]]+test, sites[st[i]+test+1], ix), test, iy*bandNum+i, GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]));
				//printf("%d %d \n",GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]) );
				//}


			//	lasty = iy*bandNum + i;
				//k = current_id[i]; // keep record the last current_id

		}  

		//------------------------------------------------------
		// Calculate intersection point for each site

		y1 = -1 * res.x* res.x;
		for(i=0; i < bandNum ; i++)
		{
			
			//lasty = GET_PTR(stack[i]);
			lasty = GET_Z(stack[i]);
			
			mask =  __shfl((int)stack[bandNum-1], (lasty%(warpSize*bandNum))/(bandNum));  // always get the last element in array for the last site of other thread

			if (GetBitPos((iy*bandNum+i)%32, bitresult))
			{
				//if (lasty < res.x && GET_STACK(stack[i]) < res.x) // lasty < res.x  --> make sure current site is linking to previous site   
				if (lasty < res.x && GET_X(stack[i]) < res.x && GET_Y(stack[i]) < res.x) // lasty < res.x  --> make sure current site is linking to previous site   
				{
					if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum)) // when previous site is not in another thread
					{
						//y_inter[i] = interpointY(ix, lasty, GET_STACK(stack[lasty%bandNum]), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;

						y_inter[i] = interpointY(GET_X(stack[lasty%bandNum]), lasty, GET_Y(stack[lasty%bandNum]), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix) ;
						/*if (iz == 111 && ix == 250 && (iy == 48 || iy == 49))
						{
							printf("2-a=! %d %d %d %d %d %d %d %d %d %d %f\n", i, iy , GET_X(stack[lasty%bandNum]), lasty, GET_Y(stack[lasty%bandNum]), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix, y_inter[i]);
						}*/
						
					}
					else
					{
						//y_inter[i] = interpointY(ix, lasty, GET_STACK(mask), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						y_inter[i] = interpointY(GET_X(mask), lasty, GET_Y(mask), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix) ;
						/*if (iz == 111 && ix == 250 && (iy == 48 || iy == 49))
						{
							printf("2-b=! %d %d %d %d %d %d %d %d %d %d %f\n", i, iy , GET_X(mask), lasty, GET_Y(mask), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix, y_inter[i]);
						}*/
						
						
					}

					
					
				}
				else
				{
					y_inter[i] = y1;
				}
				y1 = y_inter[i];
			}
			else
			{
				y_inter[i] = y1;
			}
		}
		
		//------------------------------------------------------
		// Start warp diverge
		//bitresult = 0; // clear in every thread for storing result later
		loopflag = true;
		
		while (loopflag)
		{
			loopflag = false;
			mask = 0;
			count = 0;
			y2 = 0.0;
			
			for(i=0; i < bandNum ; i++)
			{
				if (count > 0) break;
				lasty = GET_Z(stack[i]);
				y1 = __shfl(y_inter[bandNum-1],  (lasty%(warpSize*bandNum))/(bandNum));
				

				if (GetBitPos((iy*bandNum+i)%32, bitresult))
				{
					if (lasty < res.x )
					{
						if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum))
						{
							y1 = y_inter[lasty%bandNum];
							y2 = y_inter[i];
							
							if (y1 >= y2)
							{
								count++;
								if (count == 1) 	mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;

								/*if (iz == 311 && ix == 500 )
								{
									printf("2-1=! %d %d %d %d %d %d %f %f \n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								}*/

								/*if (iz == 111 && ix == 250 && (j+i) < 64 && (j+i)>=32)
								{
									printf("test test?  %d %d %f %f %d %d %d \n y1 : %d %d %d %d %d %d %d %d \n y2 : %d %d %d %d %d %d %d %d \n", ptr[i], i, y1, y2, GET_PTR(stack[ptr[i]]), j+ptr[i], GET_STACK(stack[ptr[i]]),  GET_STACK(stack[ptr[ptr[i]]]), j+ptr[ptr[i]], GET_PTR(stack[ptr[ptr[i]]]), GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), ix, iz,  GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), GET_STACK(stack[i]), j+i, GET_PTR(stack[i]), ix, iz);

								}*/
							}
							else 
							{
								if (count == 0) mask = 0;

								
							}

						}
						else
						{						
							y2 = y_inter[i];
							if (y1 >= y2)
							{
								count++;
								if (count == 1)		mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;
								
								/*if (iz == 311 && ix == 500 )
								{
									printf("2-2=! %d %d %d %d %d %d %f %f \n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								}*/
								
							}
							else
							{
								if (count == 0) mask = 0;
								
							}
						}
					}
					else
					{
						if (count == 0) mask = 0;
					}
				}
				else
				{
					if (count == 0) mask = 0;
				}

			}

			lasty = mask;
			
			mask = __ballot(count > 0);

			// update the stack

			if (mask > 0)
			{
				loopflag = true;
				k = __ffs(mask);

				lasty = __shfl(lasty,  k-1);

				i = GET_STACK(lasty);
				j = GET_PTR(lasty);

			
				lasty = __shfl((int)stack[i%bandNum],  (i%(warpSize*bandNum))/(bandNum));
				k = GET_Z(lasty);
				lasty = __shfl((int)stack[k%bandNum],  (k%(warpSize*bandNum))/(bandNum));


				if (iy == j/bandNum)
				{
					
					stack[j%bandNum] = ENCODE_Z(stack[j%bandNum], k);
					y_inter[j%bandNum] =  interpointY(GET_X(lasty), k, GET_Y(lasty), GET_X(stack[j%bandNum]), j, GET_Y(stack[j%bandNum]), iz, ix) ;


					for(count=j%bandNum+1; count < bandNum ; count++)
					{
						

						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[j%bandNum];
						else break;
					}

				}
				

				if (iy == i/bandNum)
				{
					bitresult = bitresult & ~(SetBitPos(i%32));

					if ((i-1) >= 0)
						y_inter[i%bandNum] = y_inter[(max(0,i-1))%bandNum];

					for(count=i%bandNum+1; count < bandNum ; count++)
					{
						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[i%bandNum];
						else break;
					}
				}


				mask = __shfl((int)bitresult,  (i%(warpSize*bandNum))/(bandNum));

				if (i%(warpSize*bandNum)/warpSize == iy%(warpSize)/(warpWidth))
				{
					bitresult = mask;	
				}

			}
			else break;

		}
		
		if (iy%warpWidth == 0)
		{
			bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz] = ~bitresult;
		}

		tid += chunksize;
	}
}

__global__ void LDNIDistanceField__kernelMergeBandsX_4(unsigned int *bitDeleted, unsigned int *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	//unsigned int ix,iy,iz;
	int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	int  current_id[4], next_id[4];
	float y_inter[4];
	short ind[4], num[4], i;
	int j, k, count;
	unsigned int st[4];
	int stack[4]; // stack + ptr
	unsigned int bitresult = 0;
	int lasty;
	float y1, y2;
	unsigned int mask = 0; 
	bool loopflag = false;
	int middle_id[4], temp;
	

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);
		
		
		if (iz == 0)
		{
			j = iy*bandNum;
			for(i=0; i < bandNum; i++)
			{
				st[i] = sites_index[(j+i)*res.x+ix];
				num[i] = sites_index[(j+i)*res.x+ix+1]-st[i];


				if (num[i] > 1) 
				{
					current_id[i] = sites[st[i]];
					next_id[i] = sites[st[i]+1];
					ind[i] = 2;
					middle_id[i] = middlepointY(current_id[i], next_id[i], ix);

				}
				else if (num[i] == 1)
				{
					current_id[i] = sites[st[i]];
					ind[i] = 1;
					middle_id[i] = LDNIMARKER;

				}
				else	
				{
					middle_id[i] = -1;
					current_id[i] = LDNIMARKER;
					next_id[i]= LDNIMARKER;
					ind[i] = 0;
				}

				//if (ix == 250 && j+i == 78 && iz == 0)
				//{
				//printf("stack %d %d %d %d %d %d\n", i, iz, num[i], GET_STACK(stack[i]), GET_PTR(stack[i]));
				//	for(int test = 0; test < num[i] ; test++)
				//		printf("%d %d %d %d %d %d\n", num[i],  middlepointY(sites[st[i]]+test, sites[st[i]+test+1], ix), test, iy*BANDWIDTH+i, GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]));

				//}

			}
		}

		if (iy%warpWidth == 0)
		{
			//bitresult[(iy/(BANDWIDTH/bandNum))%2] = bitDeleted[(iy/(BANDWIDTH/bandNum))*res.x*res.z+ix*res.x+iz];
			bitresult = ~bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz];
			
		}



		// broadcast the bitresult to other thread
		bitresult = __shfl((int)bitresult, ((iy/warpWidth)%bandNum)*warpWidth);

	

		count = 0;
		lasty = -1;
		

		//define last bit for each band
		//------------------------------------------------------
		mask = 15 << bandNum*(iy%warpWidth);
		mask = bitresult & mask;

		if (__popc(mask) > 0) 
		{
			
			lasty = (iy/warpWidth)*warpSize 
				+ (32 - __ffs(__brev(mask))); // Get the most significant bit (__ffs return 1 - 32)
		}
		else
		{
			lasty = -1;
		}
		
		
		

		//link the last bit for each band ** can be optimize!!!!!!
		//------------------------------------------------------
		k = 0;

	
		//lasty = __shfl(lasty, max(0,iy%32-1));
		k = __shfl(lasty, (int)(iy%32-1));

		if ((int)(iy%32-1) >= 0) lasty = k;
	
	
		mask = __all(lasty >= 0);

	
		k = 0;

		while (mask == 0)
		{
			j = __shfl(lasty, (int)(iy%32-1));

			if (lasty < 0 && (int)(iy%32-1) >= 0) 
			{
				lasty = j;
			}
			k++;
			mask = __all(lasty >= 0); // make sure all the thread obtain the 
			if (k >= warpSize) break; // in case, but should not happen
		}
		


		if (iy%32 == 0)
			lasty = -1;
	

		k = -1;	temp = 0;
		//------------------------------------------------------
		// define stack (closest site on z axis and the pointer to previous site)
		for(i=0; i < bandNum ; i++)
		{
			//if (GetBitPos((iy*bandNum+i)%32, bitresult))
			//{
				if ((int)iz < middle_id[i])
				{
					if (GetBitPos((iy*bandNum+i)%32, bitresult))
					{
					stack[i] = ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty);
					count++; 
					lasty = iy*bandNum + i;
					k = current_id[i]; // keep record the last current_id
					}
					else
					stack[i] = ENCODE_STACK_3(GET_STACK(k), GET_PTR(k), lasty); // the last element in array = the last site
				}
				else    
				{
					if (ind[i] < num[i])
					{
						j = sites[st[i]+ind[i]];
						ind[i]++;
						temp = middlepointY(next_id[i], j, ix);
						while (temp <= middle_id[i])
						{
							next_id[i] = j;
							j = sites[st[i]+ind[i]];
							ind[i]++;
							temp = middlepointY(next_id[i], j, ix);

						}

						middle_id[i] = temp;
						current_id[i] = next_id[i];
						next_id[i] = j;
					}
					else
					{
						middle_id[i] = LDNIMARKER;
						current_id[i] = next_id[i];
					}

					if (GetBitPos((iy*bandNum+i)%32, bitresult))
					{
						stack[i] = ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty);
						count++; 
						lasty = iy*bandNum + i;
						k = current_id[i]; // keep record the last current_id
					}
					else
					stack[i] = ENCODE_STACK_3(GET_STACK(k), GET_PTR(k), lasty); // the last element in array = the last site

				}
				  

				//if (ix == 250 && iy*bandNum+i == 78)
				//{
				//	printf("^^^^ %d %d %d %d %d %d %d \n", iz,  middle_id[i], GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty, ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty), stack[i] );

				//for(int test = 0; test < num[i]; test++)
				//	printf("%d %d %d %d %d %d\n", num[i],  middlepointY(sites[st[i]]+test, sites[st[i]+test+1], ix), test, iy*bandNum+i, GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]));
				//printf("%d %d \n",GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]) );
				//}


			//	lasty = iy*bandNum + i;
				//k = current_id[i]; // keep record the last current_id

		}  

		//------------------------------------------------------
		// Calculate intersection point for each site

		y1 = -1 * res.x* res.x;
		for(i=0; i < bandNum ; i++)
		{
			
			//lasty = GET_PTR(stack[i]);
			lasty = GET_Z(stack[i]);
			
			mask =  __shfl((int)stack[bandNum-1], (lasty%(warpSize*bandNum))/(bandNum));  // always get the last element in array for the last site of other thread

			if (GetBitPos((iy*bandNum+i)%32, bitresult))
			{
				//if (lasty < res.x && GET_STACK(stack[i]) < res.x) // lasty < res.x  --> make sure current site is linking to previous site   
				if (lasty < res.x && GET_X(stack[i]) < res.x && GET_Y(stack[i]) < res.x) // lasty < res.x  --> make sure current site is linking to previous site   
				{
					if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum)) // when previous site is not in another thread
					{
						//y_inter[i] = interpointY(ix, lasty, GET_STACK(stack[lasty%bandNum]), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;

						y_inter[i] = interpointY(GET_X(stack[lasty%bandNum]), lasty, GET_Y(stack[lasty%bandNum]), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix) ;
						/*if (iz == 111 && ix == 250 && (iy == 48 || iy == 49))
						{
							printf("2-a=! %d %d %d %d %d %d %d %d %d %d %f\n", i, iy , GET_X(stack[lasty%bandNum]), lasty, GET_Y(stack[lasty%bandNum]), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix, y_inter[i]);
						}*/
						/*if (iz == 280 && ix == 280 && iy == 30 )
						{
							printf("2-a=! %d %d %d %d %d %d %d %d %d %d %f\n", i, iy , GET_X(stack[lasty%bandNum]), lasty, GET_Y(stack[lasty%bandNum]), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix, y_inter[i]);
						}*/
						
					}
					else
					{
						//y_inter[i] = interpointY(ix, lasty, GET_STACK(mask), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						y_inter[i] = interpointY(GET_X(mask), lasty, GET_Y(mask), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix) ;
						/*if (iz == 111 && ix == 250 && (iy == 48 || iy == 49))
						{
							printf("2-b=! %d %d %d %d %d %d %d %d %d %d %f\n", i, iy , GET_X(mask), lasty, GET_Y(mask), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix, y_inter[i]);
						}*/
						/*if (iz == 280 && ix == 280 && iy == 30 )
						{
							printf("2-b=! %d %d %d %d %d %d %d %d %d %d %f\n", i, iy , GET_X(stack[lasty%bandNum]), lasty, GET_Y(stack[lasty%bandNum]), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix, y_inter[i]);
						}*/
						
					}

					
					
				}
				else
				{
					y_inter[i] = y1;
				}
				y1 = y_inter[i];
			}
			else
			{
				y_inter[i] = y1;
			}
		}
		
		//------------------------------------------------------
		// Start warp diverge
		//bitresult = 0; // clear in every thread for storing result later
		loopflag = true;
		
		while (loopflag)
		{
			loopflag = false;
			mask = 0;
			count = 0;
			y2 = 0.0;
			
			for(i=0; i < bandNum ; i++)
			{
				if (count > 0) break;
				lasty = GET_Z(stack[i]);
				y1 = __shfl(y_inter[bandNum-1],  (lasty%(warpSize*bandNum))/(bandNum));
				

				if (GetBitPos((iy*bandNum+i)%32, bitresult))
				{
					if (lasty < res.x )
					{
						if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum))
						{
							y1 = y_inter[lasty%bandNum];
							y2 = y_inter[i];
							
							if (y1 >= y2)
							{
								count++;
								if (count == 1) 	mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;

								/*if (iz == 311 && ix == 500 )
								{
									printf("2-1=! %d %d %d %d %d %d %f %f \n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								}*/

								/*if (iz == 111 && ix == 250 && (j+i) < 64 && (j+i)>=32)
								{
									printf("test test?  %d %d %f %f %d %d %d \n y1 : %d %d %d %d %d %d %d %d \n y2 : %d %d %d %d %d %d %d %d \n", ptr[i], i, y1, y2, GET_PTR(stack[ptr[i]]), j+ptr[i], GET_STACK(stack[ptr[i]]),  GET_STACK(stack[ptr[ptr[i]]]), j+ptr[ptr[i]], GET_PTR(stack[ptr[ptr[i]]]), GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), ix, iz,  GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), GET_STACK(stack[i]), j+i, GET_PTR(stack[i]), ix, iz);

								}*/
							}
							else 
							{
								if (count == 0) mask = 0;

								
							}

						}
						else
						{						
							y2 = y_inter[i];
							if (y1 >= y2)
							{
								count++;
								if (count == 1)		mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;
								
								/*if (iz == 311 && ix == 500 )
								{
									printf("2-2=! %d %d %d %d %d %d %f %f \n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								}*/
								
							}
							else
							{
								if (count == 0) mask = 0;
								
							}
						}
					}
					else
					{
						if (count == 0) mask = 0;
					}
				}
				else
				{
					if (count == 0) mask = 0;
				}

			}

			lasty = mask;
			
			mask = __ballot(count > 0);

			// update the stack

			if (mask > 0)
			{
				loopflag = true;
				k = __ffs(mask);

				lasty = __shfl(lasty,  k-1);

				i = GET_STACK(lasty);
				j = GET_PTR(lasty);

			
				lasty = __shfl((int)stack[i%bandNum],  (i%(warpSize*bandNum))/(bandNum));
				k = GET_Z(lasty);
				lasty = __shfl((int)stack[k%bandNum],  (k%(warpSize*bandNum))/(bandNum));


				if (iy == j/bandNum)
				{
					
					stack[j%bandNum] = ENCODE_Z(stack[j%bandNum], k);
					y_inter[j%bandNum] =  interpointY(GET_X(lasty), k, GET_Y(lasty), GET_X(stack[j%bandNum]), j, GET_Y(stack[j%bandNum]), iz, ix) ;


					for(count=j%bandNum+1; count < bandNum ; count++)
					{
						

						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[j%bandNum];
						else break;
					}

				}
				

				if (iy == i/bandNum)
				{
					bitresult = bitresult & ~(SetBitPos(i%32));

					if ((i-1) >= 0)
						y_inter[i%bandNum] = y_inter[(max(0,i-1))%bandNum];

					for(count=i%bandNum+1; count < bandNum ; count++)
					{
						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[i%bandNum];
						else break;
					}
				}


				mask = __shfl((int)bitresult,  (i%(warpSize*bandNum))/(bandNum));

				if (i%(warpSize*bandNum)/warpSize == iy%(warpSize)/(warpWidth))
				{
					bitresult = mask;	
				}

			}
			else break;

		}
		
		if (iy%warpWidth == 0)
		{
			bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz] = ~bitresult;
		}

		tid += chunksize;
	}
}



__global__ void LDNIDistanceField__kernelMergeBandsX_2(unsigned int *bitDeleted, unsigned int *sites, unsigned int *sites_index, int3 res, int offsetPixel, const int bandNum, const int warpWidth, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	//unsigned int ix,iy,iz;
	int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	int  current_id[2], next_id[2];
	float y_inter[2];
	short ind[2], num[2], i;
	int j, k, count;
	unsigned int st[2];
	int stack[2]; // stack + ptr
	unsigned int bitresult = 0;
	int lasty;
	float y1, y2;
	unsigned int mask = 0; 
	bool loopflag = false;
	int middle_id[2], temp;
	

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);
		
		/*if (iz > 280) 
		{
			tid += chunksize;
			continue;
		}*/
		if (iz == 0)
		{
			j = iy*bandNum;
			for(i=0; i < bandNum; i++)
			{
				st[i] = sites_index[(j+i)*res.x+ix];
				num[i] = sites_index[(j+i)*res.x+ix+1]-st[i];


				if (num[i] > 1) 
				{
					current_id[i] = sites[st[i]];
					next_id[i] = sites[st[i]+1];
					ind[i] = 2;
					middle_id[i] = middlepointY(current_id[i], next_id[i], ix);

				}
				else if (num[i] == 1)
				{
					current_id[i] = sites[st[i]];
					ind[i] = 1;
					middle_id[i] = LDNIMARKER;

				}
				else	
				{
					middle_id[i] = -1;
					current_id[i] = LDNIMARKER;
					next_id[i]= LDNIMARKER;
					ind[i] = 0;
				}

				//if (ix == 250 && j+i == 78 && iz == 0)
				//{
				//printf("stack %d %d %d %d %d %d\n", i, iz, num[i], GET_STACK(stack[i]), GET_PTR(stack[i]));
				//	for(int test = 0; test < num[i] ; test++)
				//		printf("%d %d %d %d %d %d\n", num[i],  middlepointY(sites[st[i]]+test, sites[st[i]+test+1], ix), test, iy*BANDWIDTH+i, GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]));

				//}

			}
		}

		if (iy%warpWidth == 0)
		{
			//bitresult[(iy/(BANDWIDTH/bandNum))%2] = bitDeleted[(iy/(BANDWIDTH/bandNum))*res.x*res.z+ix*res.x+iz];
			bitresult = ~bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz];
			
		}



		// broadcast the bitresult to other thread
		bitresult = __shfl((int)bitresult, ((iy/warpWidth)%2)*warpWidth);

	

		count = 0;
		lasty = -1;
		

		//define last bit for each band
		//------------------------------------------------------
		mask = 3 << bandNum*(iy%warpWidth);
		mask = bitresult & mask;

		if (bitCount(mask) > 0) 
		{
			
			lasty = (iy/warpWidth)*warpSize 
				+ (32 - __ffs(__brev(mask))); // Get the most significant bit (__ffs return 1 - 32)
		}
		else
		{
			lasty = -1;
		}
		
		
		

		//link the last bit for each band ** can be optimize!!!!!!
		//------------------------------------------------------
		k = 0;

	
		//lasty = __shfl(lasty, max(0,iy%32-1));
		k = __shfl(lasty, (int)(iy%32-1));

		if ((int)(iy%32-1) >= 0) lasty = k;
	
	
		mask = __all(lasty >= 0);

	
		k = 0;

		while (mask == 0)
		{
			j = __shfl(lasty, (int)(iy%32-1));

			if (lasty < 0 && (int)(iy%32-1) >= 0) 
			{
				lasty = j;
			}
			k++;
			mask = __all(lasty >= 0); // make sure all the thread obtain the 
			if (k >= warpSize) break; // in case, but should not happen
		}
		


		if (iy%32 == 0)
			lasty = -1;

		
		/*if (ix == 256 && iz == 0 )
		{
			printf("=! %d %d %d %d %d %d %d \n", i, iy , iy%32, lasty, bitresult, num[0], num[1]);
		}*/
	

		k = -1;	temp = 0;
		//------------------------------------------------------
		// define stack (closest site on z axis and the pointer to previous site)
		for(i=0; i < bandNum ; i++)
		{
			//if (GetBitPos((iy*bandNum+i)%32, bitresult))
			//{
				if ((int)iz < middle_id[i])
				{
					if (GetBitPos((iy*bandNum+i)%32, bitresult))
					{
					stack[i] = ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty);
					count++; 
					lasty = iy*bandNum + i;
					k = current_id[i]; // keep record the last current_id
					}
					else
					stack[i] = ENCODE_STACK_3(GET_STACK(k), GET_PTR(k), lasty); // the last element in array = the last site
				}
				else    
				{
					if (ind[i] < num[i])
					{
						j = sites[st[i]+ind[i]];
						ind[i]++;
						temp = middlepointY(next_id[i], j, ix);
						while (temp <= middle_id[i])
						{
							next_id[i] = j;
							j = sites[st[i]+ind[i]];
							ind[i]++;
							temp = middlepointY(next_id[i], j, ix);

						}

						middle_id[i] = temp;
						current_id[i] = next_id[i];
						next_id[i] = j;
					}
					else
					{
						middle_id[i] = LDNIMARKER;
						current_id[i] = next_id[i];
					}

					if (GetBitPos((iy*bandNum+i)%32, bitresult))
					{
						stack[i] = ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty);
						count++; 
						lasty = iy*bandNum + i;
						k = current_id[i]; // keep record the last current_id
					}
					else
					stack[i] = ENCODE_STACK_3(GET_STACK(k), GET_PTR(k), lasty); // the last element in array = the last site

				}
				  

				//if (iz == 280 && ix == 280 && iy*bandNum+i == 64)
				//{
				//	printf("^^^^ %d %d %d %d %d %d %d %d %d %d %d\n",bitresult,  iz,  middle_id[i], GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty, ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty), stack[i], GET_X(stack[i]), GET_Y(stack[i]), GET_Z(stack[i]) );

				//for(int test = 0; test < num[i]; test++)
				//	printf("%d %d %d %d %d %d\n", num[i],  middlepointY(sites[st[i]]+test, sites[st[i]+test+1], ix), test, iy*bandNum+i, GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]));
				//printf("%d %d \n",GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]) );
				//}


			//	lasty = iy*bandNum + i;
				//k = current_id[i]; // keep record the last current_id





			//}
			//else
			//{
			//	stack[i] = ENCODE_STACK_3(GET_STACK(k), GET_PTR(k), lasty); // the last element in array = the last site
			//}  

		}  

		//for(i=0; i < bandNum ; i++)
		//{
		//	if (GetBitPos((iy*bandNum+i)%32, bitresult))
		//	{
		//		if ((int)iz < middle_id[i])
		//			stack[i] = ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty);
		//			
		//		else    
		//		{
		//			if (ind[i] < num[i])
		//			{
		//				j = sites[st[i]+ind[i]];
		//				ind[i]++;
		//				temp = middlepointY(next_id[i], j, ix);
		//				while (temp <= middle_id[i])
		//				{
		//					next_id[i] = j;
		//					j = sites[st[i]+ind[i]];
		//					ind[i]++;
		//					temp = middlepointY(next_id[i], j, ix);
		//					
		//				}
		//				
		//				middle_id[i] = temp;
		//				current_id[i] = next_id[i];
		//				next_id[i] = j;
		//			}
		//			else
		//			{
		//				middle_id[i] = LDNIMARKER;
		//				current_id[i] = next_id[i];
		//			}
		//			stack[i] = ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty);
		//			
		//			
		//		}
		//		count++;   
		//		   
		//		//if (ix == 250 && iy*bandNum+i == 78)
		//		//{
		//		//	printf("^^^^ %d %d %d %d %d %d %d \n", iz,  middle_id[i], GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty, ENCODE_STACK_3(GET_STACK(current_id[i]), GET_PTR(current_id[i]), lasty), stack[i] );

		//			//for(int test = 0; test < num[i]; test++)
		//			//	printf("%d %d %d %d %d %d\n", num[i],  middlepointY(sites[st[i]]+test, sites[st[i]+test+1], ix), test, iy*bandNum+i, GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]));
		//				//printf("%d %d \n",GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]) );
		//		//}
		//		

		//		lasty = iy*bandNum + i;
		//		k = current_id[i]; // keep record the last current_id

		//		
	
		//	
		//		
		//	}
		//	else
		//	{
		//		stack[i] = ENCODE_STACK_3(GET_STACK(k), GET_PTR(k), lasty); // the last element in array = the last site
		//	}  
		//
		//}  
		//------------------------------------------------------
		// Calculate intersection point for each site

		/*if (ix == 256 && iz == 0 )
						{
							printf("=! %d %d %d %d %d %d %d %d %f \n", i, iy , iy%32, lasty, GET_STACK(mask),iy*bandNum+i,GET_STACK(stack[i]), mask, y_inter[i]);
						}*/

		/*if (ix == 256 && iz == 0 && iy == 65)
						{
							printf("=! %d %d %d %d %d %d %d %d %f \n", i, iy , iy%32, lasty, GET_STACK(stack[lasty%bandNum]),iy*bandNum+i,GET_STACK(stack[i]), mask, y_inter[i]);
						}*/


		y1 = -1 * res.x* res.x;
		for(i=0; i < bandNum ; i++)
		{
			
			//lasty = GET_PTR(stack[i]);
			lasty = GET_Z(stack[i]);
			
			mask =  __shfl((int)stack[bandNum-1], (lasty%(warpSize*bandNum))/(bandNum));  // always get the last element in array for the last site of other thread

			/*if (ix == 280 && iz == 280 && iy*bandNum+i==116)
			{
				//printf("2-a=! %d %d %d %d %d %d %d %d %d %d %f\n", i, iy , GET_X(stack[lasty%bandNum]), lasty, GET_Y(stack[lasty%bandNum]), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix, y_inter[i]);
				printf("%d %d %d \n", lasty, mask,GetBitPos((iy*bandNum+i)%32, bitresult) );
			}*/
			if (GetBitPos((iy*bandNum+i)%32, bitresult))
			{
				//if (lasty < res.x && GET_STACK(stack[i]) < res.x) // lasty < res.x  --> make sure current site is linking to previous site   
				if (lasty < res.x && GET_X(stack[i]) < res.x && GET_Y(stack[i]) < res.x) // lasty < res.x  --> make sure current site is linking to previous site   
				{
					if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum)) // when previous site is not in another thread
					{
						//y_inter[i] = interpointY(ix, lasty, GET_STACK(stack[lasty%bandNum]), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;

						y_inter[i] = interpointY(GET_X(stack[lasty%bandNum]), lasty, GET_Y(stack[lasty%bandNum]), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix) ;
						/*if (ix == 280 && iz == 280 && iy*bandNum+i==116)
						{
							printf("2-a=! %d %d %d %d %d %d %d %d %d %d %f\n", i, iy , GET_X(stack[lasty%bandNum]), lasty, GET_Y(stack[lasty%bandNum]), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix, y_inter[i]);
						}*/
						
					}
					else
					{
						//y_inter[i] = interpointY(ix, lasty, GET_STACK(mask), ix, iy*bandNum+i, GET_STACK(stack[i]), ix, iz) ;
						y_inter[i] = interpointY(GET_X(mask), lasty, GET_Y(mask), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix) ;
						/*if (iz == 280 && ix == 280 && iy*bandNum+i==116)
						{
							printf("2-b=! %d %d %d %d %d %d %d %d %d %d %f\n", i, iy , GET_X(mask), lasty, GET_Y(mask), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix, y_inter[i]);
						}*/
						
						
					}

					
					
				}
				else
				{
					y_inter[i] = y1;
				}
				y1 = y_inter[i];

				/*if (iz == 280 && ix == 280 && iy*bandNum+i==116)
				{
					printf("2-d=! %d %d %d %d %d %d %d %d %d %d %f\n", i, iy , GET_X(mask), lasty, GET_Y(mask), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix, y_inter[i]);
				}*/
			}
			else
			{
				y_inter[i] = y1;
				/*if (iz == 280 && ix == 280 && iy*bandNum+i==116)
				{
					printf("2-c=! %d %d %d %d %d %d %d %d %d %d %f\n", i, iy , GET_X(mask), lasty, GET_Y(mask), GET_X(stack[i]), iy*bandNum+i, GET_Y(stack[i]), iz, ix, y_inter[i]);
				}*/
			}
		}

		/*if (ix == 256 && iz == 0 )
		{
			printf("=! %d %d %d %d %d %d %d %d %f \n", i, iy , iy%32, lasty, GET_STACK(stack[lasty%bandNum]),iy*bandNum+i,GET_STACK(stack[i]), mask, y_inter[i]);
		}*/

		/*if (iz == 111 && ix == 250 )
		{
			printf("bitresult ! %d %d %d \n", iy , iy%32, bitresult);
		}*/
		//------------------------------------------------------
		// Start warp diverge
		//bitresult = 0; // clear in every thread for storing result later
		loopflag = true;
		int test = 0;
		while (loopflag)
		{
			loopflag = false;
			mask = 0;
			count = 0;
			y2 = 0.0;
			test++;
			for(i=0; i < bandNum ; i++)
			{
				if (count > 0) break;
				//lasty = GET_PTR(stack[i]);
				lasty = GET_Z(stack[i]);
				y1 = __shfl(y_inter[bandNum-1],  (lasty%(warpSize*bandNum))/(bandNum));
				
				
				/*if (ix == 256 && iz == 500 && iy==81 )
				{
					printf("4=! %d %d %d %d %d %f\n", i, iy , iy%32, lasty, bitresult, y1);
				}*/

				//if (iz == 250 && ix == 431 && (iy*bandNum+i) == 96)
				//{
				//	printf("1=! %d %d %d %d %d %d %f %f\n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
				//}
				

				if (GetBitPos((iy*bandNum+i)%32, bitresult))
				{
					if (lasty < res.x )
					{
						if ((lasty%(warpSize*bandNum))/(bandNum) == ((iy*bandNum+i)%(warpSize*bandNum))/(bandNum))
						{
							y1 = y_inter[lasty%bandNum];
							y2 = y_inter[i];
							
							if (y1 >= y2)
							{
								count++;
								if (count == 1) 	mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;

								/*if (iz == 280 && ix == 280 && (iy*bandNum+i)>=100 && (iy*bandNum+i)< 125)
								{
									printf("2-1=! %d %d %d %d %d %d %f %f \n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								}*/

								/*if (iz == 111 && ix == 250 && (j+i) < 64 && (j+i)>=32)
								{
									printf("test test?  %d %d %f %f %d %d %d \n y1 : %d %d %d %d %d %d %d %d \n y2 : %d %d %d %d %d %d %d %d \n", ptr[i], i, y1, y2, GET_PTR(stack[ptr[i]]), j+ptr[i], GET_STACK(stack[ptr[i]]),  GET_STACK(stack[ptr[ptr[i]]]), j+ptr[ptr[i]], GET_PTR(stack[ptr[ptr[i]]]), GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), ix, iz,  GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), GET_STACK(stack[i]), j+i, GET_PTR(stack[i]), ix, iz);

								}*/
							}
							else 
							{
								if (count == 0) mask = 0;

								
							}

						}
						else
						{						
							y2 = y_inter[i];
							if (y1 >= y2)
							{
								count++;
								if (count == 1)		mask = ENCODE_STACK(lasty, iy*bandNum+i);
								else mask = 0;
								
								/*if (iz == 280 && ix == 280 && (iy*bandNum+i)>=100 && (iy*bandNum+i)< 125)
								{
									printf("2-2=! %d %d %d %d %d %d %f %f \n", i, iy , iy%32, lasty, bitresult, count, y1, y2);
								}*/
								
							}
							else
							{
								if (count == 0) mask = 0;
								
							}
						}
					}
					else
					{
						if (count == 0) mask = 0;
					}
				}
				else
				{
					if (count == 0) mask = 0;
				}

			}

			lasty = mask;
			
			mask = __ballot(count > 0);

			//if (ix == 256 && iz == 500 && iy == 80)
			//{
			//	printf("3=! %d %d \n", mask, count);
			//}
			

			// update the stack

			if (mask > 0)
			{
				loopflag = true;
				k = __ffs(mask);

				lasty = __shfl(lasty,  k-1);

				i = GET_STACK(lasty);
				j = GET_PTR(lasty);

				/*if (ix == 254 && iz == 500 )
				{
					printf("4=! %d %d %d %d %d %d %d\n", k, iy , iy%32, lasty, bitresult, i, j);
				}*/

				
				lasty = __shfl((int)stack[i%bandNum],  (i%(warpSize*bandNum))/(bandNum));
				k = GET_Z(lasty);
				lasty = __shfl((int)stack[k%bandNum],  (k%(warpSize*bandNum))/(bandNum));


				if (iy == j/bandNum)
				{
					//stack[j%bandNum] = ENCODE_PTR(stack[j%bandNum], k);
					stack[j%bandNum] = ENCODE_Z(stack[j%bandNum], k);
					//y_inter[j%bandNum] =  interpointY(ix, k, GET_STACK(lasty), ix, j, GET_STACK(stack[j%bandNum]), ix, iz) ;
					y_inter[j%bandNum] =  interpointY(GET_X(lasty), k, GET_Y(lasty), GET_X(stack[j%bandNum]), j, GET_Y(stack[j%bandNum]), iz, ix) ;

					/*if (iz == 111 && ix == 250 && (iy == 48))
					{
						printf("tset %f %d \n %d %d %d %d %d %d %d %d\n", y_inter[j%bandNum], j, GET_X(lasty), k, GET_Y(lasty), GET_X(stack[j%bandNum]), j, GET_Y(stack[j%bandNum]), iz, ix);
					}*/

					for(count=j%bandNum+1; count < bandNum ; count++)
					{
						

						if (!GetBitPos((iy*bandNum+count)%32, bitresult))
							y_inter[count] =  y_inter[j%bandNum];
						else break;

						/*if (ix == 254 && iz == 500 )
						{
							printf("4=! %d %d %d %d %d\n", j, iy , iy%32, count, GET_PTR(stack[count]));
						}*/
					}

				}
				

				if (iy == i/bandNum)
				{
					bitresult = bitresult & ~(SetBitPos(i%32));
					y_inter[i%bandNum] = y_inter[(max(0,i-1))%bandNum];
				}


				mask = __shfl((int)bitresult,  (i%(warpSize*bandNum))/(bandNum));

				if (i%(warpSize*bandNum)/warpSize == iy%(warpSize)/(warpWidth))
				{
					bitresult = mask;	
				}

			}
			else break;

		}
		
		if (iy%warpWidth == 0)
		{
			bitDeleted[(iy/warpWidth)*res.x*res.z+ix*res.x+iz] = ~bitresult;
		}

		tid += chunksize;
	}
}

__global__ void LDNIDistanceField__MaurerAxisInX(unsigned int *bitDeleted, unsigned int *sites, unsigned int *sites_index, int3 res, int offsetPixel, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	unsigned int  ind[BANDWIDTH], current_id[BANDWIDTH], next_id[BANDWIDTH];
	short num[BANDWIDTH], i, j, count;
	unsigned int st[BANDWIDTH];
	unsigned int stack[BANDWIDTH];
	unsigned int bitresult;
	float y1, y2;
	short ptr[BANDWIDTH];
	int middle_id[BANDWIDTH], k, temp;

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);

		bitresult = 0;
		if (iz == 0)
		{
			j = iy*BANDWIDTH;
			for(i=0; i < BANDWIDTH; i++)
			{
				st[i] = sites_index[(j+i)*res.x+ix];
				num[i] = sites_index[(j+i)*res.x+ix+1]-st[i];


				

				/*if (GET_STACK(sites[st[i]]) == 330 &&  GET_PTR(sites[st[i]]) == 291)
				{
					printf(" ^^ %d %d %d %d %d \n", ix, i, iz, GET_STACK(sites[st[i]]), GET_PTR(sites[st[i]]));

				}*/
				//if (iy == 1 && iz == 0 && ix == 1)
				//{
				//	printf("how ? %d %d %d %d %d %d %d \n",num[i], GET_STACK(sites[st[i]]), GET_PTR(sites[st[i]]), GET_STACK(sites[st[i]+1]), GET_PTR(sites[st[i]+1]), GET_STACK(sites[st[i]+2]), GET_PTR(sites[st[i]+2]) );
				//}
				
				if (num[i] > 1) 
				{
					current_id[i] = sites[st[i]];
					next_id[i] = sites[st[i]+1];
					ind[i] = 2;
					middle_id[i] = middlepointY(current_id[i], next_id[i], ix);
					//middle_id[i] = (short)ceil((current_id[i]+next_id[i])/2.0);	
				}
				else if (num[i] == 1)
				{
					current_id[i] = sites[st[i]];
					ind[i] = 1;
					middle_id[i] = LDNIMARKER;

				}
				else	
				{
					middle_id[i] = -1;
					current_id[i] = LDNIMARKER;
					next_id[i]= LDNIMARKER;
					ind[i] = 0;
				}

				//if (iz == 25 && ix == 250)
				//	printf("num %d %d %d %d %d %d\n", num[i], current_id[i], next_id[i], j+i, middle_id[i]);
				
			}
		}

		count = 0;
		k = -1;
		temp = 0;
		for(i=0; i < BANDWIDTH ; i++)
		{
			
			if (num[i]>0)
			{
				ptr[i] = k;


				if ((int)iz < middle_id[i])
					stack[i] = current_id[i];
				else
				{
					if (ind[i] < num[i])
					{
						k = sites[st[i]+ind[i]];
						ind[i]++;
						temp = middlepointY(next_id[i], k, ix);
						
						while (temp <= middle_id[i])
						{
							next_id[i] = k;
							k = sites[st[i]+ind[i]];
							ind[i]++;
							temp = middlepointY(next_id[i], k, ix);
						}
						middle_id[i] = temp;
						current_id[i] = next_id[i];
						next_id[i] = k;

						
					}
					else
					{
						middle_id[i] = LDNIMARKER;
						current_id[i] = next_id[i];
					}
					stack[i] = current_id[i];

					//if ( ix == 250 && iy == 2 && i == 14)
					//{
					// printf("stack %d %d %d %d %d %d\n", k, ind[i], iz, num[i], middle_id[i], temp);
					////	printf("test~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
					//}
					
				}
				count++;

				/*if ( ix == 250 && iy == 2 && i == 14)
				{
					printf("stack %d %d %d %d %d \n", iz, middle_id[i], GET_STACK(current_id[i]), GET_PTR(current_id[i]), ind[i]);
				}*/

				//if (iy == 13 && i == 15 && ix == 250 )
				//if ( ix == 250 && iy == 2 && i == 1 && iz == 0)
				//if (iy == 0 && ix == 250 && i == 25)
				/*if (ix == 250 && iy == 2 && i == 14 && iz == 111)
				{
					printf("stack %d %d %d %d %d %d\n", i, iz, num[i], GET_STACK(stack[i]), GET_PTR(stack[i]));
					for(int test = 0; test < num[i] ; test++)
						printf("%d %d %d %d %d %d\n", num[i],  middlepointY(sites[st[i]]+test, sites[st[i]+test+1], ix), test, iy*BANDWIDTH+i, GET_STACK(sites[st[i]+test]), GET_PTR(sites[st[i]+test]));

				}*/

				k = i;

				/*if ( ix == 250 && iy == 3)
				{
					printf("stack %d %d %d %d %d %d\n", k, ind[i], iz, num[i], middle_id[i], temp);
				}*/

				
			}
			else
			{
				stack[i] = -1;
				ptr[i] = k;
				bitresult = bitresult | SetBitPos(i);
			}

			
				//printf("test test?  %d %d %d %d %d %d %d \n", ptr[i], i,  GET_STACK(current_id[i]),GET_PTR(current_id[i]), middle_id[i] , GET_STACK(next_id[i]),GET_PTR(next_id[i]));
		}


		if (count > 2)
		{
			k=0;
			
			for(i=0; i < BANDWIDTH ; i++)
			{
				//if (iy == 0 && iz ==0 && ix == 0)
				//	printf("test test  %d %d \n", count, stack[i] );

				if (GET_PTR(stack[i]) <  res.x || GET_STACK(stack[i]) < res.x)
				{
					if (k < 2)
					{
						k++;
						continue;
					}
					
					while (k>=2)
					{
						//y1 =  interpointY(ix, j+ptr[ptr[i]], stack[ptr[ptr[i]]], ix, j+ptr[i], stack[ptr[i]], ix, iz) ;
						//y2 =  interpointY(ix, j+ptr[i], stack[ptr[i]], ix, j+i, stack[i], ix, iz) ;
				
						//y1 =  interpointY(GET_PTR(stack[ptr[ptr[i]]]), j+ptr[ptr[i]], GET_STACK(stack[ptr[ptr[i]]]), GET_PTR(stack[ptr[i]]), j+ptr[i], GET_STACK(stack[ptr[i]]), ix, iz) ;
						//y2 =  interpointY(GET_PTR(stack[ptr[i]]), j+ptr[i], GET_STACK(stack[ptr[i]]), GET_PTR(stack[i]), j+i, GET_STACK(stack[i]), ix, iz) ;

						y1 =  interpointY(GET_STACK(stack[ptr[ptr[i]]]), j+ptr[ptr[i]], GET_PTR(stack[ptr[ptr[i]]]), GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), iz, ix) ;
						y2 =  interpointY(GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), GET_STACK(stack[i]), j+i, GET_PTR(stack[i]), iz, ix) ;



						//if (ix == 256 && (j+i) == 178 && iz == 0)
						//{
						//	printf("ptr %d %f %f %d %d %d\n", j+i, y1, y2, k, j, i);
						//	printf("y1 : %d %d %d %d %d %d %d %d \n", ix, j+ptr[ptr[i]], stack[ptr[ptr[i]]], ix, j+ptr[i], stack[ptr[i]], ix, iz);
						//	printf("y2 : %d %d %d %d %d %d %d %d \n", ix, j+ptr[i], stack[ptr[i]], ix, j+i, stack[i], ix, iz);
						//}

						/*if (iy == 1 && iz == 0 && ix == 1)
						{
							printf("test test?  %d %d %f %f %d %d %d \n", ptr[i], i, y1, y2, GET_PTR(stack[ptr[i]]), j+ptr[i], GET_STACK(stack[ptr[i]]) );
							printf("y1 : %d %d %d %d %d %d %d %d \n", GET_STACK(stack[ptr[ptr[i]]]), j+ptr[ptr[i]], GET_PTR(stack[ptr[ptr[i]]]), GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), ix, iz) ;
							printf("y2 : %d %d %d %d %d %d %d %d \n", GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), GET_STACK(stack[i]), j+i, GET_PTR(stack[i]), ix, iz) ;

						}*/

						/*if (ix == 250 && j+i == 78 && iz == 111)
						{
							printf("test test?  %d %d %f %f %d %d %d \n", ptr[i], i, y1, y2, GET_PTR(stack[ptr[i]]), j+ptr[i], GET_STACK(stack[ptr[i]]) );
							printf("y1 : %d %d %d %d %d %d %d %d \n", GET_STACK(stack[ptr[ptr[i]]]), j+ptr[ptr[i]], GET_PTR(stack[ptr[ptr[i]]]), GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), ix, iz) ;
							printf("y2 : %d %d %d %d %d %d %d %d \n", GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), GET_STACK(stack[i]), j+i, GET_PTR(stack[i]), ix, iz) ;
						}*/


						//if ((j+i) >= 420 && (j+i) <= 440 && iz == 111 && ix == 250)
						
						
						

						if (y1 < y2)
							break;

						//if ((j+i) == 430 && iz == 111 && ix == 250)
						

						/*if (iz == 280 && ix == 280 && (j+i) < 128 && (j+i)>=96)
						{
							printf("test test? %d  %d %d %f %f %d %d %d \n y1 : %d %d %d %d %d %d %d %d \n y2 : %d %d %d %d %d %d %d %d \n", bitresult, ptr[i], i, y1, y2, GET_PTR(stack[ptr[i]]), j+ptr[i], GET_STACK(stack[ptr[i]]),  GET_STACK(stack[ptr[ptr[i]]]), j+ptr[ptr[i]], GET_PTR(stack[ptr[ptr[i]]]), GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), ix, iz,  GET_STACK(stack[ptr[i]]), j+ptr[i], GET_PTR(stack[ptr[i]]), GET_STACK(stack[i]), j+i, GET_PTR(stack[i]), ix, iz);

						}*/

						k--;
						stack[ptr[i]] = -1;
						bitresult = bitresult | SetBitPos(ptr[i]);
						ptr[i] = ptr[ptr[i]];

					}

					k++;

				}

			}


			bitDeleted[iy*res.x*res.z+ix*res.x+iz] = bitresult;

			//if (ix == 256 && iz ==0)
			//	printf("--------------%d %d \n", iy, bitresult, count );
			//for(i=0; i < BANDWIDTH ; i++)
			//{
			//	bitDeleted[iy*res*res+ix*res+iz]
			//}
		}
		else
		{
			bitDeleted[iy*res.x*res.z+ix*res.x+iz] = bitresult;
			//if (ix == 256 && iz ==0)
			//	printf("--------------%d %d %d\n", iy, bitresult, count );
		}

		
		
		

		tid += blockDim.x * gridDim.x;
	}
}


__global__ void LDNIDistanceField__GenerateProbablySiteInY(unsigned int *bitDeleted, unsigned int *bitForNextLoop, unsigned int *counter, unsigned short *sites, unsigned int *sites_index, int res, int offsetPixel, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	unsigned int chunksize = blockDim.x * gridDim.x;
	short current_id[3], ind[3];
	short prev_id[3];
	unsigned int st[3], num[3], bitResult;
	float y1, y2;
	short z[3];
	int count=0;
	
	
	while(tid<nodeNum) {
		iy = tid%res;
		iz = (tid%(chunksize*res)/res)/(chunksize/res);
		ix = (tid/(chunksize*res))*(chunksize/res)+(tid%(chunksize*res)%(chunksize)/res);

		if (iz*res*res+iy*res+ix > nodeNum)
		{		
			return;
		}

		if (iy > 0 && iy < res-1)
		{
			if (iz == 0)
			{
				st[1] = sites_index[iy*res+ix];
				num[1] = sites_index[iy*res+ix+1]-st[1];

				st[0] = sites_index[(iy-1)*res+ix];
				num[0] = sites_index[(iy-1)*res+ix+1]-st[0];

				st[2] = sites_index[(iy+1)*res+ix];
				num[2] = sites_index[(iy+1)*res+ix+1]-st[2];

				if (num[0]>0) current_id[0] = sites[st[0]];
				if (num[1]>0) current_id[1] = sites[st[1]];
				if (num[2]>0) current_id[2] = sites[st[2]];

				prev_id[0] = LDNIMARKER;  //iy-1
				prev_id[1] = LDNIMARKER;  //iy
				prev_id[2] = LDNIMARKER;  //iy+1

				ind[0] = 0;
				ind[1] = 0;
				ind[2] = 0;

				bitResult = 0;
				count = 0;

				//if (ix == 125 && iy == 251)
				//if (ix == 125 && (iy <=252 && iy>=200))
				//{
					//printf("%d %d %d \n",num[0], num[1], num[2] );

					/*for(int i=0; i<num[0]; i++)
					{
						printf("sites 0 : %d \n",sites[st[0]+i]);

					}*/

					//for(int i=0; i<num[1]; i++)
					//{
					//	printf("sites 1 : %d %d %d \n",ix, iy, sites[st[1]+i]);

					//}
					//printf("------------- \n");

					/*for(int i=0; i<num[2]; i++)
					{
						printf("sites 2 : %d \n",sites[st[2]+i]);

					}*/
				//}

			
			}

			if (num[0] > 0 && num[1] > 0 && num[2] > 0)
			{
				if (iz != current_id[1])
				{
					z[0] =  (abs((int)(prev_id[0]-iz)) < abs((int)(current_id[0]-iz)))? prev_id[0]:current_id[0];
					z[1] =  (abs((int)(prev_id[1]-iz)) < abs((int)(current_id[1]-iz)))? prev_id[1]:current_id[1];
					z[2] =  (abs((int)(prev_id[2]-iz)) < abs((int)(current_id[2]-iz)))? prev_id[2]:current_id[2];

					y1 =  interpointY(ix, iy-1, z[0], ix, iy, z[1], ix, iz) ;
					y2 =  interpointY(ix, iy, z[1], ix, iy+1, z[2], ix, iz) ;

					if (ix == 125 && iy == 251 && iz == 211)
					{
						printf("%d %d %d %d %f %f %d\n", iz, z[0], z[1], z[2], y1, y2, count);
						printf("a) %d %d %d %d %d %d %d %d \n",ix, iy-1, z[0], ix, iy, z[1], ix, iz);
						printf("b) %d %d %d %d %d %d %d %d \n",ix, iy, z[1], ix, iy+1, z[2], ix, iz);
						printf(" %d %d %d %d %d %d \n", prev_id[0], prev_id[1], prev_id[2], current_id[0], current_id[1], current_id[2]);
					}

					if (y1 >= y2)
					{
						bitResult = bitResult | SetBitPos(iz%32);
						count++;
					}


				}
				else
				{
					prev_id[1] = current_id[1];
					ind[1]++;
					if (ind[1] >= num[1])	
						current_id[1] = LDNIMARKER;
					else
						current_id[1] = sites[st[1]+ind[1]];
				}

				if (iz == current_id[0])
				{
					//if (ix == 125 && iy == 256)
					//	printf("--------------\n");
					prev_id[0] = current_id[0];
					ind[0]++;
					if (ind[0] >= num[0])	
						current_id[0] = LDNIMARKER;
					else
						current_id[0] = sites[st[0]+ind[0]];
				}
				if (iz == current_id[2])
				{
					prev_id[2] = current_id[2];
					ind[2]++;
					if (ind[2] >= num[2])	
						current_id[2] = LDNIMARKER;
					else
						current_id[2] = sites[st[2]+ind[2]];
				}

				if ((iz+1)%32 == 0)
				{
					bitForNextLoop[(iz/32)*res*res+iy*res+ix]= bitResult;
					bitDeleted[(iz/32)*res*res+iy*res+ix]= bitResult;
					bitResult = 0;
				}

				if (iz == res-1)
				{
					//if (iy==256)
					//	printf("count %d %d \n", ix, count);
					atomicAdd(counter, count);
				}


			}
		}

		tid += blockDim.x * gridDim.x;

	}

	
}

__global__ void LDNIDistanceField__GenerateProbablySiteInYByGivenDistance(unsigned int *bitSites, unsigned short *sites, unsigned int *sites_index, int res, int offsetPixel, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	unsigned int st = 0, num = 0;
	unsigned int chunksize = blockDim.x * gridDim.x;
	unsigned int temp;
	int currentSite, prevSite, dist1, dist2;
	short currentIndex, ind;
	float d;
	short i,j;
	unsigned int buffer[THREADS_PER_BLOCK] = {0};

	while(tid<nodeNum) {
		iy = tid%res;
		iz = (tid%(chunksize*res)/res)/(chunksize/res);
		ix = (tid/(chunksize*res))*(chunksize/res)+(tid%(chunksize*res)%(chunksize)/res);

		if (iz*res*res+iy*res+ix > nodeNum)
		{
			//printf("error %d %d %d %d %d %d\n", tid, ix, iy, iz,(tid/(chunksize*res)),(tid%(chunksize*res)/(res*res)) );
			return;
		}

		if (iz == 0) 
		{
			st = sites_index[iy*res+ix];
			num = sites_index[iy*res+ix+1]-st;

			if (num > 0)
				currentSite = sites[st];

			prevSite = 0;
			currentIndex = 0;
		}

		

		if (num > 0)
		{
			//if (ix ==512 && iy == 512)
			//	printf("tid %d %d %d %d %d %d %d %d \n", iz, num, st, prevSite, currentSite, currentIndex, sites[st], sites[st+1]);
			if (iz == currentSite) 
			{
				prevSite = currentSite;
				currentIndex++;
				if (currentIndex >= num)	
				{
					prevSite = 0;
				}
				//{currentSite = 0;}
				else	
				{currentSite = sites[st+currentIndex];}
			}

			//if (prevSite <=0 && currentSite > 0)
			if (prevSite <=0)
			{
				dist1 = abs((int)iz-currentSite);
				//if(dist1 <= offsetPixel && iz <= currentSite)
				if(dist1 <= offsetPixel)
				{
					d = sqrt((float)(offsetPixel*offsetPixel - dist1*dist1));
					ind = (int)d;
					if (d >= 1)
					{
						//buffer[offsetPixel-ind] = buffer[offsetPixel-ind] | SetBitPos(iz%32);
						//buffer[offsetPixel+ind] = buffer[offsetPixel+ind] | SetBitPos(iz%32);
						temp = SetBitPos(iz%32);
						for(i=offsetPixel-ind; i <= offsetPixel+ind; i++)
						{
							buffer[i] = buffer[i] | temp;
						}
					}
					else
					{
						buffer[offsetPixel] = buffer[offsetPixel] | SetBitPos(iz%32);
					}

					//if (ix ==512 && iy == 512)
					//	printf("test %d %d %d %d %d\n", iz, dist1, ind, prevSite, currentSite);
				}
				
			}
			/*else if (prevSite > 0 && currentSite <= 0)
			{	
				dist2 = abs((int)iz-prevSite);
				if(dist2 <= offsetPixel && iz >= prevSite)
					//if(dist1 <= offsetPixel)
				{
					d = sqrt((float)(offsetPixel*offsetPixel - dist2*dist2));
					ind = (int)d;
					if (d >= 1)
					{
						//buffer[offsetPixel-ind] = buffer[offsetPixel-ind] | SetBitPos(iz%32);
						//buffer[offsetPixel+ind] = buffer[offsetPixel+ind] | SetBitPos(iz%32);
						temp = SetBitPos(iz%32);
						for(i=offsetPixel-ind; i <= offsetPixel+ind; i++)
						{
							buffer[i] = buffer[i] | temp;
						}
					}
					else
					{
						buffer[offsetPixel] = buffer[offsetPixel] | SetBitPos(iz%32);
					}

					//if (ix ==512 && iy == 512)
					//	printf("test %d %d %d %d %d\n", iz, dist1, ind, prevSite, currentSite);
				}
			}
			else if (prevSite > 0 && currentSite > 0)*/
			else
			{
				dist2 = abs((int)iz-currentSite);
				dist1 = abs((int)iz-prevSite);
				if (dist1 <= offsetPixel || dist2 <=offsetPixel)
				{
					//if (dist1 <= dist2 && iz <= prevSite)
					if (dist1 <= dist2)
					{
						d = sqrt((float)(offsetPixel*offsetPixel - dist1*dist1));

						ind = (int)d;
						if (d >= 1)
						{
							//buffer[offsetPixel-ind] = buffer[offsetPixel-ind] | SetBitPos(iz%32);
							//buffer[offsetPixel+ind] = buffer[offsetPixel+ind] | SetBitPos(iz%32);
							temp = SetBitPos(iz%32);
							for(i=offsetPixel-ind; i <= offsetPixel+ind; i++)
							{
								buffer[i] = buffer[i] | temp;
							}
						}
						else 
						{
							buffer[offsetPixel] = buffer[offsetPixel] | SetBitPos(iz%32);
						}
						
					}
					//else if (dist1 > dist2 && iz <= currentSite)
					else
					{
						d = sqrt((float)(offsetPixel*offsetPixel - dist2*dist2));

						ind = (int)d;
						if (d >= 1)
						{
							//buffer[offsetPixel-ind] = buffer[offsetPixel-ind] | SetBitPos(iz%32);
							//buffer[offsetPixel+ind] = buffer[offsetPixel+ind] | SetBitPos(iz%32);
							temp = SetBitPos(iz%32);
							for(i=offsetPixel-ind; i <= offsetPixel+ind; i++)
							{
								buffer[i] = buffer[i] | temp;
							}
						}
						else 
						{
							buffer[offsetPixel] = buffer[offsetPixel] | SetBitPos(iz%32);
						}
					}

					
					//if (ix ==512 && iy == 512)
					//	printf("test %d %d %d %d %d %d\n", iz, dist1, dist2, ind, prevSite, currentSite);
				}
			}
		}

		if ((iz+1)%32 == 0 && num>0)
		{
			j=0;
			//for(i=max(0,iy-offsetPixel); i<=min(res,iy+offsetPixel); j++,i++)
			/*for(i=iy-offsetPixel; i<=iy+offsetPixel; j++,i++)
			{
				if (i<0 || i >= res) continue;
				//if (buffer[j]!=0 && bitSites[(iz/32)*res*res+i*res+ix]!= buffer[j]) 
				if (buffer[j]!=0) 
				{
					atomicOr(&bitSites[(iz/32)*res*res+i*res+ix], buffer[j] );
				}

			}*/

			for(j=0;j<offsetPixel*2+1;j++)
				buffer[j]=0;
		}
		tid += blockDim.x * gridDim.x;


	}
}

__global__ void LDNIDistanceField__GenerateProbablySiteInXByGivenDistance(unsigned int *bitSites, unsigned short *sites, unsigned int *sites_index, int res, int offsetPixel, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	unsigned int st = 0, num = 0;
	unsigned int chunksize = blockDim.x * gridDim.x;
	unsigned int temp;
	int currentSite, prevSite, dist1, dist2;
	short currentIndex, ind;
	float d;
	short i,j;
	unsigned int buffer[THREADS_PER_BLOCK] = {0};

	while(tid<nodeNum) {
		iy = tid%res;
		iz = (tid%(chunksize*res)/res)/(chunksize/res);
		ix = (tid/(chunksize*res))*(chunksize/res)+(tid%(chunksize*res)%(chunksize)/res);

		if (iz*res*res+iy*res+ix > nodeNum)
		{
			//printf("error %d %d %d %d %d %d\n", tid, ix, iy, iz,(tid/(chunksize*res)),(tid%(chunksize*res)/(res*res)) );
			return;
		}

		if (iz == 0) 
		{
			st = sites_index[iy*res+ix];
			num = sites_index[iy*res+ix+1]-st;

			if (num > 0)
				currentSite = sites[st];

			prevSite = 0;
			currentIndex = 0;
		}



		if (num > 0)
		{
			//if (ix ==512 && iy == 512)
			//	printf("tid %d %d %d %d %d %d %d %d \n", iz, num, st, prevSite, currentSite, currentIndex, sites[st], sites[st+1]);
			if (iz == currentSite) 
			{
				prevSite = currentSite;
				currentIndex++;
				if (currentIndex >= num)	
				{prevSite = 0;}
				else	
				{currentSite = sites[st+currentIndex];}
			}

			if (prevSite <=0)
			{
				dist1 = abs((int)iz-currentSite);
				if(dist1 <= offsetPixel)
				{
					d = sqrt((float)(offsetPixel*offsetPixel - dist1*dist1));
					ind = (int)d;
					if (d >= 1)
					{
						//buffer[offsetPixel-ind] = buffer[offsetPixel-ind] | SetBitPos(iz%32);
						//buffer[offsetPixel+ind] = buffer[offsetPixel+ind] | SetBitPos(iz%32);
						temp = SetBitPos(iz%32);
						for(i=offsetPixel-ind; i <= offsetPixel+ind; i++)
						{
							buffer[i] = buffer[i] | temp;
						}
					}
					else
					{
						buffer[offsetPixel] = buffer[offsetPixel] | SetBitPos(iz%32);
					}

					//if (ix ==512 && iy == 512)
					//printf("test %d %d %d %d %d\n", iz, dist1, ind, prevSite, currentSite);
				}

			}
			else
			{
				dist2 = abs((int)iz-currentSite);
				dist1 = abs((int)iz-prevSite);
				if (dist1 <= offsetPixel || dist2 <=offsetPixel)
				{
					if (dist1 <= dist2)
					{
						d = sqrt((float)(offsetPixel*offsetPixel - dist1*dist1));

					}
					else
					{
						d = sqrt((float)(offsetPixel*offsetPixel - dist2*dist2));
					}

					ind = (int)d;
					if (d >= 1)
					{
						//buffer[offsetPixel-ind] = buffer[offsetPixel-ind] | SetBitPos(iz%32);
						//buffer[offsetPixel+ind] = buffer[offsetPixel+ind] | SetBitPos(iz%32);
						temp = SetBitPos(iz%32);
						for(i=offsetPixel-ind; i <= offsetPixel+ind; i++)
						{
							buffer[i] = buffer[i] | temp;
						}
					}
					else
					{
						buffer[offsetPixel] = buffer[offsetPixel] | SetBitPos(iz%32);
					}
					//if (ix ==512 && iy == 512)
					//	printf("test %d %d %d %d %d %d\n", iz, dist1, dist2, ind, prevSite, currentSite);
				}
			}
		}

		if ((iz+1)%32 == 0 && num>0)
		{
			j=0;
			//for(i=max(0,iy-offsetPixel); i<=min(res,iy+offsetPixel); j++,i++)
			for(i=ix-offsetPixel; i<=ix+offsetPixel; j++,i++)
			{
				if (i<0 || i >= res) continue;
				if (buffer[j]!=0) 
				{
					atomicOr(&bitSites[(iz/32)*res*res+iy*res+i], buffer[j] );
				}

			}

			for(j=0;j<offsetPixel*2+1;j++)
				buffer[j]=0;
		}
		tid += blockDim.x * gridDim.x;


	}
}


__global__ void LDNIDistanceField__writeTexToArray(unsigned short *d_output, int res, unsigned int *table_index, unsigned int* temp_index,  int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	uint4 temp;
	unsigned int num=0,i,st,v,a=0,ind, count = 0;
	unsigned int chunksize = blockDim.x * gridDim.x;

	while(tid<nodeNum) {
		ix=tid%res;	iy=(tid/res)%res;  iz=(tid/(res*res));
		temp = tex3D(site_tex,ix,iy,iz);
		st = table_index[iy*res+ix];
		num = table_index[iy*res+ix+1]-st;

		if (num>0) {
			count = bitCount(temp.x);
			ind=0;
			if (count>0) 
			{
				ind = atomicAdd(&temp_index[iy*res+ix],count);
				for(i=0; i < count ; i++){
					v = GetFirstBitPos(temp.x);
					a = 1 << v;
					d_output[st+ind+i] = iz*128 + v;	
					temp.x = temp.x & (~a);
				}
			}
			a=0; v=0; count=0;
			count = bitCount(temp.y);
			if (count>0) 
			{
				ind = atomicAdd(&temp_index[iy*res+ix],count);
				for(i=0; i < count ; i++){
					v = GetFirstBitPos(temp.y);
					a = 1 << v;
					d_output[st+ind+i] = iz*128 + 32 + v;
					temp.y = temp.y & (~a);
				}
			}
			a=0; v=0; count=0;
			count = bitCount(temp.z);
			if (count>0) 
			{
				ind = atomicAdd(&temp_index[iy*res+ix],count);
				for(i=0; i < count ; i++){
					v = GetFirstBitPos(temp.z);
					a = 1 << v;
					d_output[st+ind+i] = iz*128 + 64 + v;
					temp.z = temp.z & (~a);
				}
			}

			a=0; v=0; count=0;
			count = bitCount(temp.w);
			if (count>0) 
			{
				ind = atomicAdd(&temp_index[iy*res+ix],count);
				for(i=0; i < count ; i++){
					v = GetFirstBitPos(temp.w);
					a = 1 << v;
					d_output[st+ind+i] = iz*128 + 96 + v;
					temp.w = temp.w & (~a);
				}
			}

			
			//if (ix == 512 && iy == 512)
			//	printf("what %d %d \n", d_output[st], d_output[st+1]);
		}
		


		tid += blockDim.x * gridDim.x;


	}


}

__global__ void PBADistanceField__writeCompactArray(int *d_output, int *d_input, unsigned int *counter, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int i;

	while(tid<nodeNum) {
		

		if (d_input[tid]> -1)
		{
			i = atomicAdd(counter, 1);
			d_output[i] = d_input[tid];
			//if (i == 307076)
			//	printf("$$$$$ %d %d %d %d %d \n", i, d_input[90000000], GET_X(d_input[90000000]), GET_Y(d_input[90000000]), GET_Z(d_input[90000000]) );
		}
		
		tid += blockDim.x * gridDim.x;
	}
}

__global__ void PBADistanceField__writeTexToArray(int *d_output, int res, int nodeNum, unsigned int* counter)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iy,iz;
	uint4 temp;
	unsigned int i,id, count;
	unsigned int chunksize = blockDim.x * gridDim.x;
	int marker = -1;

	while(tid<nodeNum) {
		ix=tid%res;	iy=(tid/res)%res;  iz=(tid/(res*res));
		temp = tex3D(site_tex,ix,iy,iz);

		
		id=0;
		count = 0;
		for(i=0; i < 32; i++)
		{
			id =  TOID(ix, iy, iz*128+i, res);

			
			if (GetBitPos(i,temp.x)) 
			{
				d_output[id] = ENCODE(ix, iy, iz*128+i);		
				//if (ix == 125 && iy == 250)
				count++;
			}
			else
			{
				d_output[id] =  -1;
			}
		}
		
		
		id=0;
		for(i=0; i < 32; i++)
		{
			id =  TOID(ix, iy, iz*128+32+i, res);
			if (GetBitPos(i,temp.y)) 
			{
				d_output[id] = ENCODE(ix, iy, iz*128+32+i);	

				count++;
			}
			else
			{
				d_output[id] =  -1;
			}
		}


		id=0;
		for(i=0; i < 32; i++)
		{
			id =  TOID(ix, iy, iz*128+64+i, res);
			if (GetBitPos(i,temp.z)) 
			{
				d_output[id] = ENCODE(ix, iy, iz*128+64+i);	

				count++;
			}
			else
			{
				d_output[id] =  -1;
			}
		}

		
		id=0;
		for(i=0; i < 32; i++)
		{
			id =  TOID(ix, iy, iz*128+96+i, res);
			if (GetBitPos(i,temp.w)) 
			{
				d_output[id] = ENCODE(ix, iy, iz*128+96+i);	

				count++;
			}
			else
			{
				d_output[id] =  -1;
			}
		}


		atomicAdd(counter, count);
		tid += chunksize;
	}


}


__global__ void LDNIDistanceField__Sort2DArray(unsigned short *d_output, unsigned int *d_index, int res, int nodeNum)
{
	unsigned int tid=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int st,num,i,j;
	unsigned short tempdepth;
	unsigned short depth[512];

	while(tid<nodeNum) {
		st = d_index[tid];
		num = d_index[tid+1]-st;

		if (num > 0)
		{
			if (num > 512) { printf("too many num on one thread!!! %d\n", num); return;};
			for(i=0;i<num;i++) depth[i]=d_output[st+i];
			for(i=0;i<num;i++) {
				for(j=i+1;j<num;j++) {
					if (depth[i]>depth[j]) {
						printf("sort need ? %d %d \n", depth[i], depth[j]);
						tempdepth=depth[i];	depth[i]=depth[j];	depth[j]=tempdepth;
					}
				}
			}
			for(i=0;i<num;i++) d_output[st+i]=depth[i];
		}



		tid += blockDim.x * gridDim.x;
	}
}

__global__ void LDNIDistanceField__Test(float3 *d_output, int res, unsigned int *counter, ushort2 *site, unsigned int* site_index, float width, float3 origin, int nodeNum)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int ix,iy,iz;
	unsigned int st, num;
	unsigned int count=0,i,ind,v,a=0;
	float ww = 1.0/float(res);
	float gw = width/float(res);
	//ushort3 temp;
	ushort2 temp;

	while(tid<nodeNum) {
		ix=tid%res;	iy=(tid/res);
		st = site_index[iy*res+ix];
		num = site_index[iy*res+ix+1]-st;

		if (num > 0)
		{
			ind = atomicAdd(counter,num);
			for(i=0; i < num; i++)
			{
				temp = site[st+i];
				//d_output[ind+i] = make_float3(origin.x+(ww*(temp.x))*width, origin.x+(gw*temp.y), origin.y+(gw*temp.z));	
				d_output[ind+i] = make_float3(origin.x+(gw*iy), origin.y+(gw*temp.x), origin.z+(gw*temp.y));	
			}
		}
		//if (count>0) {
		//	ind = atomicAdd(counter,count);
		//	for(i=0; i < count ; i++){
		//		v = GetFirstBitPos(temp);
		//		a = 1 << v;
		//		//d_output[ind+i] = make_float3(origin.x+(gw*ix), origin.y+(gw*iy), origin.z+(ww*(iz*32+v))*width);	
		//		d_output[ind+i] = make_float3(origin.x+(ww*(iz*32+v))*width, origin.x+(gw*ix), origin.y+(gw*iy));	

		//		temp = temp & (~a);
		//	}
		//}


		tid += blockDim.x * gridDim.x;
	}
}

__global__ void LDNIDistanceField__writeSitesToVBO(float3 *d_output, int res, unsigned int *counter, unsigned int* d_input, float width, float3 origin, int nodeNum)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int ix,iy,iz;
	unsigned int temp;
	unsigned int count=0,i,ind,v,a=0;
	float ww = 1.0/float(res);
	float gw = width/float(res);

	while(tid<nodeNum) {
		ix=tid%res;	iy=(tid/res)%res;  iz=(tid/(res*res));
		temp = d_input[iz*res*res+iy*res+ix];
		count = bitCount(temp);
		if (count>0) {
			ind = atomicAdd(counter,count);
			for(i=0; i < count ; i++){
				v = GetFirstBitPos(temp);
				a = 1 << v;
				//d_output[ind+i] = make_float3(origin.x+(gw*ix), origin.y+(gw*iy), origin.z+(ww*(iz*32+v))*width);	
				d_output[ind+i] = make_float3(origin.x+(ww*(iz*32+v))*width, origin.x+(gw*ix), origin.y+(gw*iy));	

				temp = temp & (~a);
			}
		}
		

		tid += blockDim.x * gridDim.x;
	}


}

__global__ void LDNIDistanceField__writeResultToVBO(float3 *d_output, int3 res, unsigned int* counter, unsigned int *sites, unsigned int *sites_index, int offdist, float width, float3 origin, int nodeNum)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int ix,iy,iz;
	unsigned int st, num, ind;
	unsigned int current_id, next_id;
	unsigned int chunksize = blockDim.x * gridDim.x;
	int middle_id, k, temp;
	double dist = 0.0;
	int dx, dy, dz, id;
	float ww = 1.0/float(res.x);
	float gw = width/float(res.x);

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);

		if (iz == 0)
		{

			st = sites_index[iy*res.x+ix];
			num = sites_index[iy*res.x+ix+1]-st;


			if (num > 1) 
			{
				current_id = sites[st];
				next_id = sites[st+1];
				ind = 2;
				middle_id =  middlepointX(current_id, next_id, ix , iy);
			}
			else if (num == 1)
			{
				current_id = sites[st];
				ind = 1;
				middle_id = LDNIMARKER;

			}
			else	
			{
				middle_id = -1;
				current_id = LDNIMARKER;
				next_id= LDNIMARKER;
				ind = 0;
			}
		}

		if (num > 0)
		{
			if (iz < middle_id)
			{
				dx = GET_X(current_id)-iz;	dy = GET_Y(current_id)-ix;	dz = GET_Z(current_id)-iy;
				dist = dx * dx + dy * dy + dz * dz; 
				dist = sqrt(dist);
				if ((int)dist == offdist )
				{
					id = atomicAdd(counter, 1);
					d_output[id] = make_float3(origin.x+(gw*iz), origin.y+(gw*ix), origin.z+(gw*iy));
				}
			}
			else
			{
				if (ind < num)
				{
					k = sites[st+ind];
					ind++;
					temp = 	middlepointX(next_id, k, ix , iy);

					

					while (temp <= middle_id || iz >= temp)
					{
						next_id = k;
						k = sites[st+ind];
						ind++;
						temp = middlepointX(next_id, k, ix , iy);
					}
					middle_id = temp;
					current_id = next_id;
					next_id = k;

				


				}
				else
				{
					middle_id = LDNIMARKER;
					current_id = next_id;
				}

				dx = GET_X(current_id)-iz;	dy = GET_Y(current_id)-ix;	dz = GET_Z(current_id)-iy;
				dist = dx * dx + dy * dy + dz * dz; 
				dist = sqrt(dist);
				if ((int)dist == offdist )
				{
					id = atomicAdd(counter, 1);
					d_output[id] = make_float3(origin.x+(gw*iz), origin.y+(gw*ix), origin.z+(gw*iy));
				}
			}
			//if (ix == 256 && iy == 311)
			//	printf("current %d %d %d %d %d \n", iz, middle_id,  GET_X(current_id),	GET_Y(current_id),	 GET_Z(current_id));
			
		}
		tid += chunksize;
	}
}

__global__ void LDNIDistanceField__countArrayToVBO(int3 res, unsigned int* counter, unsigned int *sites, unsigned int *sites_index, int offdist, int nodeNum)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int ix,iy,iz;
	unsigned int st, num, ind;
	unsigned int current_id, next_id;
	unsigned int chunksize = blockDim.x * gridDim.x;
	int middle_id, k, temp;
	double dist = 0.0;
	int dx, dy, dz;

	while(tid<nodeNum) {
		iy = (tid%res.z)%res.y;
		iz = (tid%(chunksize*res.z))/chunksize;
		ix = (tid/(chunksize*res.z))*(chunksize/res.y)+(tid%(chunksize*res.z)%(chunksize)/res.y);

		if (iz == 0)
		{

			st = sites_index[iy*res.x+ix];
			num = sites_index[iy*res.x+ix+1]-st;


			if (num > 1) 
			{
				current_id = sites[st];
				next_id = sites[st+1];
				ind = 2;
				middle_id =  middlepointX(current_id, next_id, ix , iy);
			}
			else if (num == 1)
			{
				current_id = sites[st];
				ind = 1;
				middle_id = LDNIMARKER;

			}
			else	
			{
				middle_id = -1;
				current_id = LDNIMARKER;
				next_id= LDNIMARKER;
				ind = 0;
			}
		}

		if (num > 0)
		{
			if (iz < middle_id)
			{
				dx = GET_X(current_id)-iz;	dy = GET_Y(current_id)-ix;	dz = GET_Z(current_id)-iy;
				dist = dx * dx + dy * dy + dz * dz; 
				dist = sqrt(dist);
				if ((int)dist == offdist )
				{
					atomicAdd(counter, 1);
				}
			}
			else
			{
				if (ind < num)
				{
					k = sites[st+ind];
					ind++;
					temp = 	middlepointX(next_id, k, ix , iy);

					while (temp <= middle_id || iz >= temp)
					{
						next_id = k;
						k = sites[st+ind];
						ind++;
						temp = middlepointX(next_id, k, ix , iy);
					}
					middle_id = temp;
					current_id = next_id;
					next_id = k;


				}
				else
				{
					middle_id = LDNIMARKER;
					current_id = next_id;
				}

				dx = GET_X(current_id)-iz;	dy = GET_Y(current_id)-ix;	dz = GET_Z(current_id)-iy;
				dist = dx * dx + dy * dy + dz * dz; 
				dist = sqrt(dist);
				if ((int)dist == offdist )
				{
					atomicAdd(counter, 1);
				}
			}
		}
		tid += chunksize;
	}

}

__global__ void PBADistanceField__countArrayToVBO(int res, unsigned int* counter, int *outputDF, int offdist, int nodeNum)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int ix,iy,iz;
	int dx, dy, dz;
	int nx, ny, nz;
	int id;
	double dist = 0.0;


	while(tid<nodeNum) {
		ix=tid%res;	iy=(tid/res)%res;  iz=(tid/(res*res));


	//	if (ix == 125 && iy == 250)
		//	printf("dist 0--------------%f %d %d %d \n", dist, ix, iy, iz);


		DECODE(outputDF[tid], nx, ny, nz); 

		//if (ix == 0 && iy == 245 && iz == 231)
		//	printf("dist 0--------------%d %d %d %d  %d %d %d \n", outputDF[tid], nx, ny , nz , ix, iy, iz);


		dx = nx - ix; dy = ny - iy; dz = nz - iz; 
		dist = dx * dx + dy * dy + dz * dz; 
		dist = sqrt(dist);

		

		if ((int)dist == offdist )
		{
			
			atomicAdd(counter, 1);
		}

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void PBADistanceField__writeArrayToVBO(float3 *d_output, int res, unsigned int* counter, int *outputDF, int offdist, float width, float3 origin, int nodeNum)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int ix,iy,iz;
	int dx, dy, dz;
	int nx, ny, nz;
	int id;
	double dist, dist2;
	float ww = 1.0/float(res);
	float gw = width/float(res);

	while(tid<nodeNum) {
		ix=tid%res;	iy=(tid/res)%res;  iz=(tid/(res*res));

		DECODE(outputDF[tid], nx, ny, nz); 

		dx = nx - ix; dy = ny - iy; dz = nz - iz; 
		dist = dx * dx + dy * dy + dz * dz; 
		dist2 = sqrt(dist);

		if (floor(dist2) == offdist )
		//if (dist >= offdist && dist <= offdist+1)
		{
			id = atomicAdd(counter, 1);
			d_output[id] = make_float3(origin.x+(gw*ix), origin.y+(gw*iy), origin.z+(gw*iz));
		}

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void LDNIDistanceField__writeArrayToVBO(float3 *d_output, int res, unsigned int* table_index, unsigned int *m_3dArray, float width, float3 origin, int nodeNum)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int ix,iy,iz;
	unsigned int temp;
	unsigned int count=0,i,ind,v,a=0;
	float ww = 1.0/float(res);
	float gw = width/float(res);

	while(tid<nodeNum) {
		ix=tid%res;	iy=(tid/res)%res;  iz=(tid/(res*res));
		temp = m_3dArray[tid];
		count = bitCount(temp);
		if (count>0) {
			ind = atomicAdd(table_index,count);
			for(i=0; i < count ; i++){
				v = GetFirstBitPos(temp);
				a = 1 << v;
				d_output[ind+i] = make_float3(origin.x+(gw*ix), origin.y+(gw*iy), origin.z+(ww*(iz*32+v))*width);	
				temp = temp & (~a);
			}
		}		

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void LDNIDistanceField__writeTexToVBO(float3 *d_output, int res, int* table_index, float width, float3 origin, int nodeNum)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int ix,iy,iz;
	uint4 temp;
	unsigned int count=0,i,ind,v,a=0;
	float ww = 1.0/float(res);
	float gw = width/float(res);

	while(tid<nodeNum) {
		ix=tid%res;	iy=(tid/res)%res;  iz=(tid/(res*res));
		temp = tex3D(site_tex,ix,iy,iz);
		count = bitCount(temp.x);
		if (count>0) {
			ind = atomicAdd(table_index,count);
			for(i=0; i < count ; i++){
				v = GetFirstBitPos(temp.x);
				a = 1 << v;
				d_output[ind+i] = make_float3(origin.x+(gw*ix), origin.y+(gw*iy), origin.z+(ww*(iz*128+v))*width);	

				temp.x = temp.x & (~a);
			}
		}
		a=0; v=0; count=0;
		count = bitCount(temp.y);
		if (count>0){
			ind = atomicAdd(table_index,count);
			for(i=0; i < count ; i++){
				v = GetFirstBitPos(temp.y);
				a = 1 << v;
				d_output[ind+i] = make_float3(origin.x+(gw*ix), origin.y+(gw*iy), origin.z+(ww*(iz*128+32+v))*width);	
				temp.y = temp.y & (~a);
			}
		}
		a=0; v=0; count=0;
		count = bitCount(temp.z);
		if (count>0){
			ind = atomicAdd(table_index,count);
			for(i=0; i < count ; i++){
				v = GetFirstBitPos(temp.z);
				a = 1 << v;
				d_output[ind+i] = make_float3(origin.x+(gw*ix), origin.y+(gw*iy), origin.z+(ww*(iz*128+64+v))*width);		
				temp.z = temp.z & (~a);
			}
		}
		a=0; v=0; count=0;
		count = bitCount(temp.w);
		if (count>0){
			ind = atomicAdd(table_index,count);
			for(i=0; i < count ; i++){
				v = GetFirstBitPos(temp.w);
				a = 1 << v;
				d_output[ind+i] = make_float3(origin.x+(gw*ix), origin.y+(gw*iy), origin.z+(ww*(iz*128+96+v))*width);				
				temp.w = temp.w & (~a);
			}
		}

		tid += blockDim.x * gridDim.x;
	}
	/*int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int ix,iy,iz;
	//uint4 temp;
	unsigned int value;
	unsigned int count=0,i,ind,v,a=0;
	float ww = 1.0/float(res);
	float gw = width/float(res);

	while(tid<nodeNum) {
		ix=tid%res;	iy=(tid/res)%res;  iz=(tid/(res*res));
		//temp = tex3D(uint_tex3D,ix,iy,iz);
		value = tex1D(site_tex, tid);
		count = bitCount(value);
		if (count>0) {
			ind = atomicAdd(table_index,count);
			for(i=0; i < count ; i++){
				v = GetFirstBitPos(value);
				a = 1 << v;
				d_output[ind+i] = make_float3(origin.x+(gw*ix), origin.y+(gw*iy), origin.z+(ww*(iz*32+v))*width);	

				value = value & (~a);
			}
		}
		tid += blockDim.x * gridDim.x;
	}*/

}

__global__ void PBADistanceField_kernelPropagateInterband(int *output, int size, int mod, int bandSize) 
{
	int blkX = blockIdx.x % mod; 
	int band = blockIdx.x / mod; 

	int tx = blkX * blockDim.x + threadIdx.x; 
	int ty = blockIdx.y * blockDim.y + threadIdx.y; 

	int inc = bandSize * size * size; 
	int nz, nid, nDist, myDist; 
	int pixel; 

	// Top row, look backward
	int tz = __mul24(band, bandSize); 
	int topId = TOID(tx, ty, tz, size); 
	int bottomId = TOID(tx, ty, tz + bandSize - 1, size); 

	pixel = tex1Dfetch(pbaTexColor, topId); 
	nz = GET_Z(pixel); 
	myDist = abs(nz - tz); 

	for (nid = bottomId - inc; nid >= 0; nid -= inc) {
		pixel = tex1Dfetch(pbaTexColor, nid); 

		if (pixel != PBAMARKER) { 
			nz = pixel & 0x3ff; 
			nDist = abs(nz - tz); 

			if (nDist < myDist) 
				output[topId] = pixel; 

			break;	
		}
	}

	// Last row, look downward
	tz = tz + bandSize - 1; 
	pixel = tex1Dfetch(pbaTexColor, bottomId);
	nz = GET_Z(pixel); 
	myDist = abs(nz - tz); 

	for (int ii = tz + 1, nid = topId + inc; ii < size; ii += bandSize, nid += inc) {
		pixel = tex1Dfetch(pbaTexColor, nid); 

		if (pixel != PBAMARKER) { 
			nz = pixel & 0x3ff; 
			nDist = abs(nz - tz); 

			if (nDist < myDist) 
				output[bottomId] = pixel; 

			break; 
		}
	}
}

__global__ void PBADistanceField_kernelFloodZ(int *output, int size, int mod, int bandSize) 
{
	int blkX = blockIdx.x % mod; 
	int band = blockIdx.x / mod; 

	int tx = blkX * blockDim.x + threadIdx.x; 
	int ty = blockIdx.y * blockDim.y + threadIdx.y; 
	int tz = band * bandSize; 

	int plane = size * size; 
	int id = TOID(tx, ty, tz, size); 
	int pixel1, pixel2; 

	pixel1 = PBAMARKER; 

	// Sweep down
	for (int i = 0; i < bandSize; i++, id += plane) {
		pixel2 = tex1Dfetch(pbaTexColor, id); 

		//if (tx == 256 && ty == 132 && tz == 0) printf("1 %d %d %d %d\n", pixel2, tx, ty, tz);

		if (pixel2 != PBAMARKER) 
			pixel1 = pixel2; 

		output[id] = pixel1; 

		//if (id == 67840) printf("1 %d %d %d %d\n", pixel1, tx, ty, tz);
		
	}

	int dist1, dist2, nz; 

	id -= plane + plane; 

	// Sweep up
	for (int i = bandSize - 2; i >= 0; i--, id -= plane) {
		//if (id == 67840) printf("2 %d \n", pixel1);
		nz = GET_Z(pixel1); 
		//if (id == 67840) printf("3 %d \n", nz);
		dist1 = abs(nz - (tz + i)); 

		//if (id == 67840) printf("4 %d \n", dist1);
		pixel2 = output[id];
		//if (id == 67840) printf("5 %d \n", pixel2);
		nz = GET_Z(pixel2); 
		//if (id == 67840) printf("6 %d \n", nz);
		dist2 = abs(nz - (tz + i)); 

		//if (id == 67840) printf("7 %d  %d %d\n", dist2, dist1, pixel1);

		if (dist2 < dist1) 
			pixel1 = pixel2; 

		output[id] = pixel1; 

		//if (id == 67840) printf("8 %d \n", pixel1);
	}
}

__global__ void PBADistanceField_kernelUpdateVertical(int *output, int size, int mod, int bandSize) 
{
	int blkX = blockIdx.x % mod; 
	int band = blockIdx.x / mod; 

	int tx = blkX * blockDim.x + threadIdx.x; 
	int ty = blockIdx.y * blockDim.y + threadIdx.y; 
	int tz = band * bandSize; 
	int id = TOID(tx, ty, tz, size); 
	int plane = size * size; 

	int top = tex1Dfetch(pbaTexLinks, id); 
	int bottom = tex1Dfetch(pbaTexLinks, TOID(tx, ty, tz + bandSize - 1, size)); 
	int topZ = GET_Z(top); 
	int bottomZ = GET_Z(bottom); 
	int pixel; 

	int dist, myDist, nz; 

	for (int i = 0; i < bandSize; i++, id += plane) {
		pixel = tex1Dfetch(pbaTexColor, id); 
		nz = GET_Z(pixel); 
		myDist = abs(nz - (tz + i)); 

		dist = abs(topZ - (tz + i)); 
		if (dist < myDist) { myDist = dist; pixel = top; }

		dist = abs(bottomZ - (tz + i)); 
		if (dist < myDist) pixel = bottom; 

		output[id] = pixel; 
	}
}

__global__ void PBADistanceField_kernelMaurerAxis(int *stack, int size, int mod, int bandSize, int test) 
{
	int blkX = blockIdx.x % mod; 
	int band = blockIdx.x / mod; 

	int tx = blkX * blockDim.x + threadIdx.x; 
	int ty = band * bandSize; 
	int tz = blockIdx.y * blockDim.y + threadIdx.y; 

	int lastY = INFINITY; 
	int stackX_1, stackY_1 = INFINITY, stackZ_1, stackX_2, stackY_2 = INFINITY, stackZ_2; 
	int p = PBAMARKER, nx, ny, nz, s1, s2; 
	float i1, i2;     

	
	for (int i = 0; i < bandSize; i++, ty++) {
		p = tex1Dfetch(pbaTexColor, TOID(tx, ty, tz, size));

		//if (tx == 1 && (ty < 64 && ty >= 32) && tz == 200 && test == 1)
		//if (tz == 250 && ty == 33 && tx <= 512 && test == 1)
		//if (tx == 431 && tz == 250 && test == 0 && ty > 80 && ty < 101)



		//if (tz == 250 && test == 1 && ty == 25)
		//if (tz == 250 && ty == 65 && test == 1)
		//if (tz == 250 && tx == 62 && test == 0)
		//{
		//	DECODE(p, nx, ny, nz); 
			//if (ny == 330 && nz == 291 && test == 1)
		//		printf("ptr %d %d %d %d %d %d %d\n", tx, ty, tz , nx, ny, nz , i);
		//}




		//if (tx == 256 && tz == 0 && ty == 132)
		//{
		//	printf("ptr %d %d %d\n", ty, p, TOID(tx, ty, tz, size ));
			//printf("y1 : %d %d %d %d %d %d %d %d %d \n",stackX_1, stackY_2, stackZ_1, stackX_2, lastY, stackZ_2, tx, tz, s1);
			//printf("y2 : %d %d %d %d %d %d %d %d %d \n", stackX_2, lastY, stackZ_2, nx, ty, nz, tx, tz, s2);
		//}
		
		if (p != PBAMARKER) {
			while (stackY_2 != INFINITY) {
				DECODE(s1, stackX_1, stackY_1, stackZ_1); 
				DECODE(s2, stackX_2, stackY_2, stackZ_2); 
				i1 = interpointY(stackX_1, stackY_2, stackZ_1, stackX_2, lastY, stackZ_2, tx, tz); 
				DECODE(p, nx, ny, nz); 
				i2 = interpointY(stackX_2, lastY, stackZ_2, nx, ty, nz, tx, tz); 

				

				/*if (tx == 256 && tz == 0 && ty == 132)
				{
					printf("ptr %d %f %f %d %d\n", ty, i1, i2, i, lastY);
					printf("y1 : %d %d %d %d %d %d %d %d %d \n",stackX_1, stackY_2, stackZ_1, stackX_2, lastY, stackZ_2, tx, tz, s1);
					printf("y2 : %d %d %d %d %d %d %d %d %d \n", stackX_2, lastY, stackZ_2, nx, ty, nz, tx, tz, s2);
				}*/

				
				//if (tx == 1 && (ty < 64 && ty >= 32) && tz == 0 && test == 1)
				//{
					//printf("ptr %d %d %d %f %f %d %d\n", tx, ty, tz , i1, i2, i, lastY);
					//printf("y1 : %d %d %d %d %d %d %d %d %d \n",stackX_1, stackY_2, stackZ_1, stackX_2, lastY, stackZ_2, tx, tz, s1);
					//printf("y2 : %d %d %d %d %d %d %d %d %d \n", stackX_2, lastY, stackZ_2, nx, ty, nz, tx, tz, s2);
				//}

				/*if (tz == 250 && (ty >= 416 && ty < 448) && tx == 0 && test == 1)
				{
					printf("ptr %d %d %d %f %f %d %d\n", tx, ty, tz , i1, i2, i, lastY);
					printf("y1 : %d %d %d %d %d %d %d %d %d \n",stackX_1, stackY_2, stackZ_1, stackX_2, lastY, stackZ_2, tx, tz, s1);
					printf("y2 : %d %d %d %d %d %d %d %d %d \n", stackX_2, lastY, stackZ_2, nx, ty, nz, tx, tz, s2);
				}*/


				/*if (tx == 431 && tz == 250 && test == 0 && ty > 80 && ty < 101)
				{
					printf("ptr %d %d %d %f %f %d %d\n", tx, ty, tz , i1, i2, i, lastY);
				}*/

				
				//if (tz == 250 && tx == 111 && (ty <= 440 && ty >= 420))
				


				if (i1 < i2) 
					break;

				//if (tz == 250 && ty == 33 && tx <= 512 && test == 1)
				
				/*if (tz == 280 && tx == 280 && ty < 128 && ty >= 96 && test == 1)
				{
					printf("ptr %d %d %d %f %f %d %d \n y1 : %d %d %d %d %d %d %d %d \n y2 : %d %d %d %d %d %d %d %d \n", tx, ty, tz , i1, i2, i, lastY, stackX_1, stackY_2, stackZ_1, stackX_2, lastY, stackZ_2, tx, tz, stackX_2, lastY, stackZ_2, nx, ty, nz, tx, tz);
				}*/
				

				lastY = stackY_2; s2 = s1; stackY_2 = stackY_1;

				if (stackY_2 != INFINITY)
					s1 = stack[TOID(tx, stackY_2, tz, size)]; 
			}
			DECODE(p, nx, ny, nz); 

			


			s1 = s2; s2 = ENCODE(nx, lastY, nz); 
			stackY_2 = lastY; lastY = ty; 
			stack[TOID(tx, ty, tz, size)] = s2; 

			/*if (tx == 431 && tz == 250 && test == 0 && ty > 80 && ty < 101)
			{
					DECODE(s2, nx, ny, nz); 

				//if (ny == 330 && nz == 291 && test == 1)
				printf("ptr2 %d %d %d %d %d %d %d\n", tx, ty, tz , nx, ny, nz , s2);
			}*/
		}
	}

	if (p == PBAMARKER) 
		stack[TOID(tx, ty-1, tz, size)] = ENCODE(INFINITY, lastY, INFINITY); 
}

__global__ void PBADistanceField_kernelMergeBands(int *stack, int *forward, int size, int mod, int bandSize) 
{
	int blkX = blockIdx.x % mod; 
	int band1 = (blockIdx.x / mod) * 2; 
	int band2 = band1 + 1; 

	int tx = blkX * blockDim.x + threadIdx.x; 
	int tz = blockIdx.y * blockDim.y + threadIdx.y; 

	int firstY, lastY, next, p, id; 
	int3 stack_1, stack_2, current; 
	float i1, i2;

	firstY = band2 * bandSize; 
	lastY = firstY - 1; 

	/*if ( tx == 431 && tz == 250)
	{
		int nx, ny, nz;
		p = tex1Dfetch(pbaTexLinks, TOID(431, 97, 250, size)); 
		DECODE(p, nx, ny, nz); 

	//	//if (ny == 330 && nz == 291 && test == 1)
		printf("ptr %d %d %d %d %d %d \n", tx, bandSize, tz , nx, ny, nz );
	}*/

	// Band 1, get the two last items
	p = tex1Dfetch(pbaTexLinks, TOID(tx, lastY, tz, size)); 
	DECODE(p, stack_2.x, stack_2.y, stack_2.z); 

	/*if ( tx == 431 && tz == 250 && bandSize == 64 && band2 == 1)
	{
		printf("ptr111 %d %d %d %d \n", lastY, stack_2.x, stack_2.y, stack_2.z );
	}*/
	

	if (stack_2.x == INFINITY) {     // Not a site
		lastY = stack_2.y; 

		if (lastY != INFINITY) {
			p = tex1Dfetch(pbaTexLinks, TOID(tx, lastY, tz, size)); 
			DECODE(p, stack_2.x, stack_2.y, stack_2.z); 
		}
	}	

	if (stack_2.y != INFINITY) {
		p = tex1Dfetch(pbaTexLinks, TOID(tx, stack_2.y, tz, size)); 
		DECODE(p, stack_1.x, stack_1.y, stack_1.z); 
	}

	// Band 2, get the first item
	next = tex1Dfetch(pbaTexPointer, TOID(tx, firstY, tz, size)); 



	if (next < 0) 		// Not a site
		firstY = -next; 

	if (firstY != INFINITY) {
		id = TOID(tx, firstY, tz, size); 
		p = tex1Dfetch(pbaTexLinks, id); 
		DECODE(p, current.x, current.y, current.z); 
	}

	/*if ( tx == 431 && tz == 250 && bandSize == 64 && band2 == 1)
	{
		printf("ptr222 %d %d %d %d %d %d %d %d %d %d %d\n", firstY, band2, stack_1.x, stack_1.y, stack_1.z, stack_2.x, stack_2.y, stack_2.z, current.x, current.y, current.z );
	}*/

	int top = 0; 

	int count = 0; //Deb
	while (top < 2 && firstY != INFINITY) {
		while (stack_2.y != INFINITY) {
			i1 = interpointY(stack_1.x, stack_2.y, stack_1.z, stack_2.x, lastY, stack_2.z, tx, tz); 
			i2 = interpointY(stack_2.x, lastY, stack_2.z, current.x, firstY, current.z, tx, tz); 

		
			//if (tx == 503 && tz == 70)
			//	printf("-- %d %d \n", lastY, stack_2.y);

			
			if (tz == 280 && tx == 280 && bandSize == 32 && lastY == 116 )// &&  firstY < 70 )
			{
				//printf("!----------- %d %d %d %f %f %d %d %d \n", stack_2.y, lastY, firstY, i1, i2, stack_1.z, stack_2.z, current.z);
				printf("y1 : %d %d %d %d %d %d \n y2 : %d %d %d %d %d %d %d %d %f %f\n", stack_1.x, stack_2.y, stack_1.z, stack_2.x, lastY, stack_2.z, stack_2.x, lastY, stack_2.z, current.x, firstY, current.z, tx, tz, i1, i2);
			}

			if (i1 < i2) 
					break; 
			//if (bandSize == 128 && tz == 311 && tx == 256 )// &&  firstY < 70 )
		
			
			//if (bandSize == 128 && tz == 250 && tx == 431)
			
			
			
			
			count++;

			lastY = stack_2.y; stack_2 = stack_1; 
			top--; 

		

			

			if (stack_2.y != INFINITY) {
				p = stack[TOID(tx, stack_2.y, tz, size)]; 
				DECODE(p, stack_1.x, stack_1.y, stack_1.z); 
			}
		}

		// Update pointers to link the current node to the stack
		stack[id] = ENCODE(current.x, lastY, current.z); 

		//if (tz == 250 && tx == 431 && bandSize == 64 && band2 == 1)
		//{
			//int3 test;
			//DECODE(stack[TOID(tx, 97, tz, size)], test.x, test.y, test.z); 
			//printf("stack %d %d %d %d %d %d %d \n", bandSize, test.x, test.y, test.z, current.x, lastY, current.z );
		//	printf("stack %d %d %d %d %d %d %d \n", bandSize, id%size, (id/(size))%size, id/(size*size), current.x, lastY, current.z  );
		//}

		if (lastY != INFINITY) 
		{
			forward[TOID(tx, lastY, tz, size)] = firstY; 
			//if (tz == 250 && tx == 431 && bandSize == 64 && band2 == 1)
			//	printf("forward %d %d %d \n", bandSize, lastY, firstY );

		}

		top = max(1, top + 1); 

		// Advance the current pointer forward
		stack_1 = stack_2; stack_2 = make_int3(current.x, lastY, current.z); lastY = firstY; 
		firstY = tex1Dfetch(pbaTexPointer, id); 

		if (firstY != INFINITY) {
			id = TOID(tx, firstY, tz, size); 
			p = tex1Dfetch(pbaTexLinks, id); 
			DECODE(p, current.x, current.y, current.z); 
		}
	}

	//if (count >= 39)
	//printf("test %d %d %d %d\n", tx, tz, count, bandSize);
	// Update the head pointer
	firstY = band1 * bandSize; 
	lastY = band2 * bandSize; 

	if (tex1Dfetch(pbaTexPointer, TOID(tx, firstY, tz, size)) == -INFINITY) 
		forward[TOID(tx, firstY, tz, size)] = -abs(tex1Dfetch(pbaTexPointer, TOID(tx, lastY, tz, size))); 

	// Update the tail pointer
	firstY = band1 * bandSize + bandSize - 1; 
	lastY = band2 * bandSize + bandSize - 1; 

	p = tex1Dfetch(pbaTexLinks, TOID(tx, lastY, tz, size)); 
	DECODE(p, current.x, current.y, current.z); 

	if (current.x == INFINITY && current.y == INFINITY) {
		p = tex1Dfetch(pbaTexLinks, TOID(tx, firstY, tz, size)); 
		DECODE(p, stack_1.x, stack_1.y, stack_1.z); 

		if (stack_1.x == INFINITY) 
			current.y = stack_1.y; 
		else
			current.y = firstY; 


		stack[TOID(tx, lastY, tz, size)] = ENCODE(current.x, current.y, current.z); 

		//if (tz == 250 && tx == 431 && bandSize == 64 && band2 == 1)
		//{
		//	printf("-stack %d %d %d %d %d \n", bandSize, lastY, current.x, current.y, current.z );
		//}
	}

/*	if (tz == 250 && tx == 431 && bandSize == 256)
	{
		int nx, ny, nz;

		for(int a = 0; a < 512 ; a++)
		{
			DECODE(stack[TOID(tx, a, tz, size)], nx, ny, nz); 
			printf("%d %d %d %d \n", a, nx, ny, nz);
		}
	}*/

}
__global__ void PBADistanceField_kernelCreateForwardPointers(int *output, int size, int mod, int bandSize) 
{
	int blkX = blockIdx.x % mod; 
	int band = blockIdx.x / mod; 

	int tx = blkX * blockDim.x + threadIdx.x; 
	int ty = (band+1) * bandSize - 1; 
	int tz = blockIdx.y * blockDim.y + threadIdx.y; 

	int lasty = INFINITY, nexty; 
	int current, id; 

	// Get the tail pointer
	current = tex1Dfetch(pbaTexLinks, TOID(tx, ty, tz, size)); 

	if (GET_X(current) == INFINITY) 
		nexty = GET_Y(current); 
	else
		nexty = ty; 

	id = TOID(tx, ty, tz, size); 

	for (int i = 0; i < bandSize; i++, ty--, id -= size) 
		if (ty == nexty) {
			output[id] = lasty; 
			nexty = GET_Y(tex1Dfetch(pbaTexLinks, id)); 
			lasty = ty; 
		}

		// Store the pointer to the head at the first pixel of this band
		if (lasty != ty + 1) 
			output[id + size] = -lasty; 
}


__global__ void PBADistanceField_kernelColorAxis(int *output, int size) 
{
	__shared__ int3 s_Stack1[BLOCKX], s_Stack2[BLOCKX];
	__shared__ int s_lastY[BLOCKX]; 
	__shared__ float s_ii[BLOCKX]; 

	int col = threadIdx.x; 
	int tid = threadIdx.y; 
	int tx = blockIdx.x * blockDim.x + col; 
	int tz = blockIdx.y; 

	int3 stack_1, stack_2; 
	int p, lastY; 
	float ii; 

	if (tid == blockDim.y - 1) { 
		lastY = size - 1; 

		p = tex1Dfetch(pbaTexColor, TOID(tx, lastY, tz, size)); 
		DECODE(p, stack_2.x, stack_2.y, stack_2.z); 

		if (stack_2.x == INFINITY) {     // Not a site
			lastY = stack_2.y; 

			if (lastY != INFINITY) {
				p = tex1Dfetch(pbaTexColor, TOID(tx, lastY, tz, size)); 
				DECODE(p, stack_2.x, stack_2.y, stack_2.z); 
			}
		}

		if (stack_2.y != INFINITY) { 
			p = tex1Dfetch(pbaTexColor, TOID(tx, stack_2.y, tz, size)); 
			DECODE(p, stack_1.x, stack_1.y, stack_1.z); 
			ii = interpointY(stack_1.x, stack_2.y, stack_1.z, stack_2.x, lastY, stack_2.z, tx, tz); 
		}

		//if (tz == 250 && tx == 431)
		//{
			//printf("~~~~%f %d %f \n", s_ii[col], col, ii);
		//	printf("~~~ %d %d %d %d %d %d %d %d %d \n", blockDim.y - 1, stack_1.x, stack_2.y, stack_1.z, stack_2.x, lastY, stack_2.z, tx, tz);
		//}

		s_Stack1[col] = stack_1; s_Stack2[col] = stack_2; s_lastY[col] = lastY; s_ii[col] = ii; 

		
	}

	__syncthreads(); 

	if (tz == 311 && tx == 256)
	{
		/*int nx, ny, nz;
		for(int a = 0; a < 512 ; a++)
		{
			p = tex1Dfetch(pbaTexColor, TOID(tx, a, tz, size)); 
			DECODE(p, nx,ny, nz); 
			printf("%d %d %d %d \n",a,  nx, ny,nz);
		}*/
		
		/*p = tex1Dfetch(pbaTexColor, TOID(431, 97, 250, size)); 
		DECODE(p, nx,ny, nz); 
		printf("%d %d %d %d \n", 97, nx, ny,nz);
		p = tex1Dfetch(pbaTexColor, TOID(431, 98, 250, size)); 
		DECODE(p, nx,ny, nz); 
		printf("%d %d %d %d \n", 98, nx, ny,nz);
		p = tex1Dfetch(pbaTexColor, TOID(431, 99, 250, size)); 
		DECODE(p, nx,ny, nz); 
		printf("%d %d %d %d \n",99,  nx, ny,nz);
		p = tex1Dfetch(pbaTexColor, TOID(431, 100, 250, size)); 
		DECODE(p, nx,ny, nz); 
		printf("%d %d %d %d \n", 100, nx, ny,nz);
		p = tex1Dfetch(pbaTexColor, TOID(431, 101, 250, size)); 
		DECODE(p, nx,ny, nz); 
		printf("%d %d %d %d \n", 101, nx, ny,nz);
		p = tex1Dfetch(pbaTexColor, TOID(431, 102, 250, size)); 
		DECODE(p, nx,ny, nz); 
		printf("%d %d %d %d \n", 102, nx, ny,nz);
		p = tex1Dfetch(pbaTexColor, TOID(431, 103, 250, size)); 
		DECODE(p, nx,ny, nz); 
		printf("%d %d %d %d \n", 103, nx, ny,nz);
		p = tex1Dfetch(pbaTexColor, TOID(431, 104, 250, size)); 
		DECODE(p, nx,ny, nz); 
		printf("%d %d %d %d \n", 104, nx, ny,nz);
		p = tex1Dfetch(pbaTexColor, TOID(431, 105, 250, size)); 
		DECODE(p, nx,ny, nz); 
		printf("%d %d %d %d \n", 105, nx, ny,nz);*/
	}
	

	for (int ty = size - 1 - tid; ty >= 0; ty -= blockDim.y) {
		stack_1 = s_Stack1[col]; stack_2 = s_Stack2[col]; lastY = s_lastY[col]; ii = s_ii[col]; 

		/**/
		//if (tz == 250 && tx == 431)
		//	printf("@@@ %d %d %d %d %d %d \n",tx, ty, tz, stack_2.x, lastY, stack_2.z);

		

		while (stack_2.y != INFINITY) {
			if (ty > ii) 
				break; 
			/*if (tz == 250 && tx == 431 )
			{
				printf("------ %d %f %d\n", ty, ii, stack_2.y);
			}*/
			
			lastY = stack_2.y; stack_2 = stack_1;

			

			if (stack_2.y != INFINITY) {
				p = tex1Dfetch(pbaTexColor, TOID(tx, stack_2.y, tz, size)); 
				DECODE(p, stack_1.x, stack_1.y, stack_1.z); 

				ii = interpointY(stack_1.x, stack_2.y, stack_1.z, stack_2.x, lastY, stack_2.z, tx, tz); 
			}
		}

		__syncthreads(); 

		
		/*if (tz == 250 && tx == 431 )
		{
			printf("Encode %d %d %d %d \n", ty, stack_2.x, lastY, stack_2.z);
		}*/

		output[TOID(tx, ty, tz, size)] = ENCODE(stack_2.x, lastY, stack_2.z); 

		if (tid == blockDim.y - 1) {
			s_Stack1[col] = stack_1; s_Stack2[col] = stack_2; s_lastY[col] = lastY; s_ii[col] = ii; 
		}

		__syncthreads(); 
	}

	
	//if (tz == 280 && tx == 280)
	//{
	//	int nx, ny, nz;
	//	for(int a = 0; a < 512 ; a++)
	//	{
	//		p = output[TOID(tx, a, tz, size)];//tex1Dfetch(pbaTexColor, TOID(431, a, 250, size)); 
	//		DECODE(p, nx,ny, nz); 
	//		printf("%d %d %d %d \n",a, nx,ny, nz);
	//	}
	//}
}

__global__ void PBADistanceField_kernelTransposeXY(int *data, int log2Width, int mask)
{
	__shared__ int block1[BLOCKXY][BLOCKXY + 1];
	__shared__ int block2[BLOCKXY][BLOCKXY + 1];

	int blkX = blockIdx.y; 
	int blkY = blockIdx.x >> log2Width; 
	int blkZ = blockIdx.x & mask; 

	if (blkX > blkY) 
		return ; 

	int x, y, z, id1, id2; 
	int pixel; 

	

	blkX = __mul24(blkX, BLOCKXY); 
	blkY = __mul24(blkY, BLOCKXY); 
	z = blkZ << log2Width; 

	// read the cube into shared memory
	x = blkX + threadIdx.x;
	y = blkY + threadIdx.y;
	id1 = ((z + y) << log2Width) + x;
	block1[threadIdx.y][threadIdx.x] = data[id1];

	x = blkY + threadIdx.x;
	y = blkX + threadIdx.y;
	id2 = ((z + y) << log2Width) + x;
	block2[threadIdx.y][threadIdx.x] = data[id2];

	__syncthreads();


	if (id2 == 0) printf("------------------------------------- hahahaha\n");
	// write the rotated cube to global memory
	pixel = block1[threadIdx.x][threadIdx.y];
	data[id2] = ROTATEXY(pixel); 
	pixel = block2[threadIdx.x][threadIdx.y];
	data[id1] = ROTATEXY(pixel); 
}

//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
__constant__ unsigned int MultiplyDeBruijnBitPosition[] = {  0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8, 
31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9 };

__constant__ unsigned char BitReverseTable256[] = 
{
	0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0, 
	0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8, 
	0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4, 
	0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC, 
	0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2, 
	0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
	0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6, 
	0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
	0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,
	0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9, 
	0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,
	0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
	0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3, 
	0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
	0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7, 
	0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF
};


__device__ inline unsigned int bitCount(unsigned int i)
{
	i = i - ((i >> 1) & 0x55555555);
	i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
	return ((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
}

__device__ inline unsigned int Reversebit(unsigned int v)
{
	unsigned int r;

	r = (BitReverseTable256[ v & 0xff] << 24) | 
		(BitReverseTable256[(v >> 8) & 0xff] << 16) | 
		(BitReverseTable256[(v >> 16) & 0xff] << 8) |
		(BitReverseTable256[(v >> 24) & 0xff]);

	return r;
}

__device__ inline unsigned int GetFirstBitPos(unsigned int source)
{
	return (MultiplyDeBruijnBitPosition[((unsigned int)((source & -source) * 0x077CB531U)) >> 27]);
}

__device__ inline unsigned int GetLastBitPos(unsigned int source)
{
	unsigned int r = Reversebit(source);
	return (31-(MultiplyDeBruijnBitPosition[((unsigned int)((r & -r) * 0x077CB531U)) >> 27]));
}

__device__ inline unsigned int SetBitPos(unsigned int pos)
{
	return (1 << pos);
}
__device__ inline bool GetBitPos(unsigned int pos, unsigned int source)
{
	return (source & (1 << pos));
}

__device__ inline float interpointY(int x1, int y1, int z1, int x2, int y2, int z2, int x0, int z0) 
{
	float xM = (x1 + x2) / 2.0f; 
	float yM = (y1 + y2) / 2.0f; 
	float zM = (z1 + z2) / 2.0f;    
	float nx = x2 - x1; 
	float ny = y2 - y1; 
	float nz = z2 - z1; 

	return yM + (nx * (xM - x0) + nz * (zM - z0)) / ny; 
}

__device__ inline int middlepointY(unsigned int site1, unsigned int site2, int z0) 
{
	int dy22 = (GET_PTR(site2)-z0)*(GET_PTR(site2)-z0);
	int dy12 = (GET_PTR(site1)-z0)*(GET_PTR(site1)-z0);
	int d1 = GET_STACK(site1);
	int d2 = GET_STACK(site2);

	
	return int(0.5 * ((dy22-dy12)/(float)(d2-d1)  + d1+d2))+1;
}

__device__ inline int middlepointX(unsigned int site1, unsigned int site2, int y0, int z0) 
{
	int xPlusx = GET_X(site1) + GET_X(site2);
	int xMinusx = GET_X(site1) - GET_X(site2);
	int yPlusy = GET_Y(site1) + GET_Y(site2);
	int yMinusy = GET_Y(site1) - GET_Y(site2);
	int zPlusz = GET_Z(site1) + GET_Z(site2);
	int zMinusz = GET_Z(site1) - GET_Z(site2);

	return int(0.5 * ((zMinusz*(zPlusz-2.0*z0)+yMinusy*(yPlusy-2.0*y0))/(float)xMinusx  + xPlusx))+1;
}