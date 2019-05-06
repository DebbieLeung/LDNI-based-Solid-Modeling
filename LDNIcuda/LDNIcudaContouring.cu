/*
 *  Copyright (C) 2014, Geometric Design and Manufacturing Lab in THE CHINESE UNIVERSITY OF HONG KONG
 *  All rights reserved.
 *   
 *		 http://ldnibasedsolidmodeling.sourceforge.net/
 *  
 *   
 *  Redistribution and use in source and binary forms, with or without modification, 
 *  are permitted provided that the following conditions are met:
 *  
 *  1. Redistributions of source code must retain the above copyright notice, 
 *     this list of conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright notice, 
 *     this list of conditions and the following disclaimer in the documentation 
 *	   and/or other materials provided with the distribution.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 *   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
 *   WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. 
 *   IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
 *   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 *   BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
 *   OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
 *   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 *   ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
 *   OF SUCH DAMAGE.
 */



#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <sys/stat.h>

#include "cuda.h"
#include "cutil.h"

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/fill.h>
#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/unique.h>
#include <cstdlib>

#include "../GLKLib/GLKGeometry.h"
#include "PMBody.h"
#include "LDNIcudaSolid.h"
#include "LDNIcudaOperation.h"


//--------------------------------------------------------------------------------------------
extern __global__ void krLDNIContouring_initCompactIndexRays(unsigned int *compactIndexArray, int nMeshRes, 
															 unsigned int *indexArray, int res, int nStepSize, unsigned int emptyValue);
extern __global__ void krLDNIContouring_votingGridNodes(unsigned int *compactIndexArray, int nMeshRes, int rayNum,
														unsigned int *indexArray, int res, int nStepSize, float *depthArray, 
														bool *gridNodes, float ww, int nAxis);
extern __global__ void krLDNIContouring_constructBndCells(unsigned int *bndCell, bool *gridNodes, int nMeshRes, int zStartIndex, int zStep, unsigned int invalidValue);
extern __global__ void krLDNIContouring_constructVertexArray(unsigned int *bndCell, int bndCellNum, int nMeshRes, int nStepSize,
									unsigned int *xIndexArray, float *xNxArray, float *xNyArray, float *xDepthArray,
									unsigned int *yIndexArray, float *yNxArray, float *yNyArray, float *yDepthArray,
									unsigned int *zIndexArray, float *zNxArray, float *zNyArray, float *zDepthArray,
									float ox, float oy, float oz, float ww, int res, float *vPosArray, bool bWithIntersectionPrevention);
extern __global__ void krLDNIContouring_constructFaceArray(unsigned int *bndCell, int bndCellNum, bool *gridNodes, int nMeshRes, int nStepSize,
									int nAxis, unsigned int *faceArray, int invalidValue, float *vPosArray,
									float ox, float oy, float oz, float ww, float *vAdditionalVerArray, int *additionalVerNum,
									bool bWithIntersectionPrevention);

//--------------------------------------------------------------------------------------------

extern __global__ void krFDMContouring_RotationBoundingBox(double *bndBoxX, double *bndBoxY, double *bndBoxZ, unsigned int* count, 
														   short nAxis, double3 rotdir, double angle, int res, unsigned int *IndexArray,
														   float *DepthArray, float3 origin, float gwidth);

extern __global__ void krFDMContouring_BinarySampling(bool *gridNode, double3 rotdir, float angle, int res, 
											   unsigned int *xIndexArray, float *xDepthArray,
											   unsigned int *yIndexArray, float *yDepthArray,
											   unsigned int *zIndexArray, float *zDepthArray,
											   float3 origin, double2 imgOrigin, float gwidth, int3 imageRes, 
											   int nodeNum, float thickness, float imgWidth);
extern __global__ void krFDMContouring_CountAllStick(bool *gridNodes, int3 imageRes, int cellNum, unsigned int* count);



extern __global__ void krFDMContouring_FindAllStickInAxisX(bool *gridNodes, unsigned int* stickIndex, unsigned int* counter,
														float2 *stickStart, float2 *stickEnd, int3 imageRes, int cellNum, 
														double2 imgOri, float imgWidth, int* stickID, int* prevStickId, int2 *stickDir);


extern __global__ void krFDMContouring_FindAllStickInAxisZ(bool *gridNodes, unsigned int* stickIndex, unsigned int* counter,
														  float2 *stickStart, float2 *stickEnd, int3 imageRes, int cellNum, 
														  double2 imgOri, float imgWidth, int* stickID, int* prevStickId, int2 *stickDir);

extern __global__ void krFDMContouring_ConstrainedSmoothing(float2 *newStart, float2 *newEnd, float2 *stickStart, float2 *stickEnd, 
															int *stickID, int stickNum, int2 *stickDir,
															float *accMovement, double moveCoef, float imgWidth, bool *bIterFlag, 
															double2 imgOri);

extern __global__ void krFDMContouring_ConstrainedSmoothing(float2* newStart, float2* newEnd, float2 *stickStart, float2 *stickEnd,
															int* stickID, int stickNum, float2 *oldStick, float *accMovement, 
															double moveCoef, float imgWidth, double2 imgOri);

extern __global__ void krFDMContouring_BuildHashTable(int* prevStickIndex, int stickNum, 
													  unsigned int* sortedStickId, int* sortedPrevStickId);

extern __global__ void krFDMContouring_Count3DBit(unsigned int *d_output, unsigned int* bitresult, int cellNum, int3 texSize);

extern __global__ void krFDMContouring_BuildHashTableForZ(bool *gridNodes, int3 imageRes, int cellNum, unsigned int* bitHashTable);


extern __global__ void krFDMContouring_SubtractSolidRegion(bool *gridNodes, bool *suptNodes, bool *seedNodes, int nodeNum, int3 imageRes, int iy);

extern __global__ void krFDMContouring_Dilation(bool *gridNodes, bool* output, int nodeNum, int3 imageRes, double realThreshold, int gridRadius, int iy);

extern __global__ void krFDMContouring_Erosion(bool *gridNodes, bool* output, int nodeNum, int3 imageRes, double realThreshold, int gridRadius);

extern __global__ void krFDMContouring_Filter1(bool *gridNodes, bool *outNodes, int nodeNum, int3 imageRes, int iy);

extern __global__ void krFDMContouring_Filter2(short2 *disMap, bool *outNodes, bool *suptNodes, int nodeNum, int3 imageRes, bool *RegionFlag, double t, int disTexSize, int iy);

extern __global__ void krFDMContouring_Filter3(bool* outNodes, bool *seedNodes, int nodeNum);

extern __global__ void krFDMContouring_Filter4(bool* outNodes, bool *inNodesA, bool *inNodesB, int nodeNum, int3 imageRes, int iy);

extern __global__ void krFDMContouring_Filter5(bool* outNodes, bool *inNodesA, bool *gridNodes, int nodeNum, int iy, int3 imageRes);

extern __global__ void krFDMContouring_integrateImageintoGrossImage(bool* outNodes, bool *inNodes, int nodeNum, int3 imageRes, int iy);

extern __global__ void krFDMContouring_InitializedValue(short2 *dMap, int nodeNum, int value);

extern __global__ void krFDMContouring_InitializedDistanceMap(bool *gridNodes, short2 *dMap, int nodeNum, int realNum, int3 imageRes, int iy, int texwidth);

extern __global__ void krFDMContouring_kernelFloodDown(short2 *output, int size, int bandSize);

extern __global__ void krFDMContouring_kernelFloodUp(short2 *output, int size, int bandSize);

extern __global__ void krFDMContouring_kernelPropagateInterband(short2 *output, int size, int bandSize);

extern __global__ void krFDMContouring_kernelUpdateVertical(short2 *output, int size, int band, int bandSize);

extern __global__ void krFDMContouring_kernelTranspose(short2 *data, int size);

extern __global__ void krFDMContouring_kernelProximatePoints(short2 *stack, int size, int bandSize);

extern __global__ void krFDMContouring_kernelCreateForwardPointers(short2 *output, int size, int bandSize);

extern __global__ void krFDMContouring_kernelMergeBands(short2 *output, int size, int bandSize);

extern __global__ void krFDMContouring_kernelDoubleToSingleList(short2 *output, int size);

extern __global__ void krFDMContouring_kernelColor(short2 *output, int size);

extern __global__ void krFDMContouring_CopyNodesrom3Dto2D(bool *m_2DNodes, bool *m_3DNodes, int nodeNum, int3 imageRes, int iy);

extern __global__ void krFDMContouring_CopyNodesrom2Dto3D(bool *m_2DNodes, bool *m_3DNodes, int nodeNum, int3 imageRes, int iy);

extern __global__ void krFDMContouring_VerticalSpptPxlProp(bool *gridNodes, bool *suptNodes, int nodeNum, int3 imageRes, int iy);


extern __global__ void krSLAContouring_Initialization(bool *tempImg, bool *targetImg, bool *gridNodes, int nodeNum, int3 imageRes, int iy);

extern __global__ void krSLAContouring_Filter1(bool *assistImg, bool *tempImg, bool *targetImg, int nodeNum, int2 suptRes, int suptRadius, int iy);

extern __global__ void krSLAContouring_CopyNodesToIntX(bool *nodeImg, unsigned int *intex, int nodeNum, int2 indRes, int2 imgRes, int iy);
extern __global__ void krSLAContouring_CopyNodesToIntZ(bool *nodeImg, unsigned int *intex, int nodeNum, int2 indRes, int2 imgRes, int iy);
extern __global__ void krSLAContouring_OrthoSearchRemainAnchorZ(bool *assistImg, bool *tempImg, bool *targetImg, int nodeNum, int2 indRes, int2 imgRes, int iy);
extern __global__ void krSLAContouring_OrthoSearchRemainAnchorX(bool *assistImg, bool *tempImg, bool *targetImg, int nodeNum, int2 indRes, int2 imgRes, int iy);
extern __global__ void krSLAContouring_Filter2(bool *assistImg, bool *tempImg, bool *targetImg, int *count, int nodeNum, int iy);
extern __global__ void krSLAContouring_Filter3(bool *outNodes, bool *inNodes, int nodeNum, int iy);
extern __global__ void krSLAContouring_Filter4(bool *outNodes, bool *inNodes, bool *imgNodes, bool *regionFinish, int nodeNum, int iy);
extern __global__ void krSLAContouring_Filter5(bool *gridNodes, bool *tempImg, bool *suptNodes, unsigned int *linkIndex, int nodeNum, int3 imageRes, int iy);
extern __global__ void krSLAContouring_FindAllLinks(unsigned int *linkIndex, unsigned int *linkLayerC, short *linkLayerD, short2 *linkID, bool *tempImg, unsigned int *count, int nodeNum, int3 imageRes);
extern __global__ void krSLAContouring_RelateAllLinksBetweenLayers(unsigned int *linkIndex, unsigned int *linkLayerC, bool *gridNodes, bool *suptNodes, int nodeNum, int3 imageRes);
extern __global__ void krSLAContouring_ReverseNodes(bool *outNodes, bool *outNodes2, bool *inNodes, int nodeNum, int iy);
extern __global__ void krSLAContouring_InitializedDistanceMap(bool *seedNodes, short2 *dMap, int nodeNum, int2 imageRes, int iy, int texwidth);


extern __global__ void krSLAContouring_CounterLink1(short2 *linkID, bool *linkcount, int* count, int nodeNum, int linkNum, int linkThreshold);
extern __global__ void krSLAContouring_FilterLink1(short2 *linkID, unsigned int *linklayerC, short *linklayerD, int* count, bool *linkcount, 
												   int nodeNum, int linkNum, int linkThreshold, int lengthofLayer, int furtherStepLength);
extern __global__ void krSLAContouring_CalConnectionMap(short2 *linkID, int i, int j, bool *bflag, int *pixelNum, int nodeNum, int2 imgRes, double nSampleWidth);
extern __global__ void krSLAContouring_CheckConnectionMap(short2 *linkID, int i, int j, bool *bflag, int *pixelNum, int nodeNum, int2 imgRes);
extern __global__ void krSLAContouring_ConnectionMapOnLayers(bool* gridNodes, bool* tempImg, bool* linkMap, bool* bflagDelete, short2* linkID, int i, int j, int pixelNum, int lengthofLayer, int furtherStepLength, int layerNumforOneCircle, int endlayer, int startlayer, int3 imgRes, int nodeNum);
extern __global__ void krSLAContouring_DeleteMapOnLayers(bool* tempImg, bool* bflagDelete, int layerNumforOneCircle, int2 imgRes, int nodeNum);
extern __global__ void krSLAContouring_FilterLink2(bool* suptNodes, bool* tempImgs, int startlayer, int endlayer, int3 imgRes, int nodeNum);
extern __global__ void krSLAContouring_Filter2DOr(bool* outNodes, bool* inNodes, int nodeNum, int iy);
extern __global__ void krSLAContouring_FillAnchorValue(bool* seedNodes, bool* inNodes, unsigned int* _value, unsigned int marker, int nodeNum, int iy);
extern __global__ void krSLAContouring_FillImageValue(bool* tempImg, bool* suptImg, unsigned int* _value, int2 imgRes, int nodeNum, unsigned int init, int iy);
extern __global__ void krSLAContouring_GetAnchorPoint(bool* targetImg, unsigned int* _value, unsigned int* anchorPt, int2 imgRes, int nodeNum, int iy);

//--------------------------------------------------------------------------------------------
extern __device__ unsigned int _searchCompactBndCellIndex(unsigned int *bndCell, int bndCellNum, unsigned int queryKey);
extern __device__ bool _searchToFillQEFMatrix(float pp[], float A[][3], float B[], int ix, int iy, int iz, int nStepSize, float ww, int res,
									   unsigned int *xIndexArray, float *xNxArray, float *xNyArray, float *xDepthArray,
									   unsigned int *yIndexArray, float *yNxArray, float *yNyArray, float *yDepthArray,
									   unsigned int *zIndexArray, float *zNxArray, float *zNyArray, float *zDepthArray);
extern __device__ bool _singularValueDecomposition(float a[][3], float u[][3], float v[][3]);
extern __device__ void _transpose(float A[][3], float B[][3]);
extern __device__ void _mul(float A[][3], float B[][3], float C[][3]);
extern __device__ void _mul(float A[][3], float x[], float b[]);
extern __device__ float _determination(float a[][3]);

//--------------------------------------------------------------------------------------------
extern __device__ double3 _rotatePointAlongVector(double px, double py, double pz, 
										  double x1, double y1, double z1, 
										  double x2, double y2, double z2,
										  double angle);

extern __device__ bool _detectInOutPoint(float px, float py, float pz, 
								  unsigned int* xIndex, float* xDepth, 
								  unsigned int* yIndex, float* yDepth, 
								  unsigned int* zIndex, float* zDepth,
								  float3 ori_p, float3 origin, float gwidth, int res);

extern __device__ bool _calTwoLineSegmentsIntersection(float2 vMinus1, float2 v_a,
													   float2 v_b, float2 vPlus1, double pt[]);


extern __device__ double3 _calEquation(float2 v1, float2 v2);

extern __device__ bool _calTwoLinesIntersection(double3 l1, double3 l2, double pt[]);

extern __device__ inline unsigned int SetBitPos(unsigned int pos);

extern __device__ inline unsigned int bitCount(unsigned int i);

extern __device__ float interpoint(int x1, int y1, int x2, int y2, int x0);


extern __global__ void krFDMContouring_Test3D(bool* outNodes, int nodeNum, int3 imageRes);
extern __global__ void krFDMContouring_Test2D(bool* outNodes, int nodeNum, int2 imageRes);

#define MARKER      -32768
#define BLOCKSIZE	64
#define TILE_DIM	32
#define BLOCK_ROWS	8
texture<short2> disTexColor; 
texture<short2> disTexLinks; 



//--------------------------------------------------------------------------------------------
//	The class member functions
//--------------------------------------------------------------------------------------------
void LDNIcudaOperation::LDNISLAContouring_GenerationwithSupporting(LDNIcudaSolid* solid, ContourMesh *c_mesh, ContourMesh *supt_mesh, float nSampleWidth, bool bImgDisplay, float thickness, float anchorR, float threshold, float cylinderRadius, float patternThickness)
{
	//float thickness = 0.01;
	float distortRatio = 1.0;
	double distortError = distortRatio*nSampleWidth*nSampleWidth;
	double smoothErrorCoef = 0.001;
	int VSAiter = 20;
	int simpRatio = 10;
	float angle = 0.0;
	double movementCoef = 0.65;
	double rotBoundingBox[6];
	float supportingSize = 0.012;
	double clipPlaneNm[3] = {0.0,1.0,0.0};
	int neighborsNum = ceil(0.15*sqrt(supportingSize)/solid->GetSampleWidth()); //this is only needed for constant supporting size 

	bool *gridNodes;
	float2 *stickStart, *stickEnd;
	unsigned int *stickIndex;
	int *stickID;
	int *prevStickID;
	int2 *stickDir;

	//----------------------------------------------------------------------------------------------
	//	Step 1: initialization
	long time = clock();
	angle = LDNIFDMContouring_CompRotationBoundingBox(solid, rotBoundingBox, clipPlaneNm);
	int BinaryImageSize[3];
	BinaryImageSize[1] = (int)floor((rotBoundingBox[3]-rotBoundingBox[2])/thickness);
	BinaryImageSize[0] = (int)ceil((rotBoundingBox[1]-rotBoundingBox[0])/nSampleWidth);
	BinaryImageSize[2] = (int)ceil((rotBoundingBox[5]-rotBoundingBox[4])/nSampleWidth);

	

	//printf("Bounding box %f %f %f %f %f %f \n", rotBoundingBox[0], rotBoundingBox[1], rotBoundingBox[2], rotBoundingBox[3], rotBoundingBox[4], rotBoundingBox[5]);
	printf("-------------------------------------------\n");
	printf("Image Size : %d X %d X %d \n", BinaryImageSize[0],BinaryImageSize[1],BinaryImageSize[2]);
	float range;

	//range = MAX(rotBoundingBox[1]-rotBoundingBox[0], rotBoundingBox[3]-rotBoundingBox[2]);
	//range = MAX(range, rotBoundingBox[5]-rotBoundingBox[4]);

	int baseLayerNum = 10;

	//----------------------------------------------------------------------------------------------
	//	Step 2: Binary Sampling
	if (bImgDisplay)
	{
		bool *suptNodes;
		c_mesh->SetThickness(thickness);
		
		c_mesh->SetImageResolution(BinaryImageSize);

		
		LDNIFDMContouring_BinarySamlping(solid, c_mesh, rotBoundingBox, BinaryImageSize, angle, thickness, clipPlaneNm, nSampleWidth, gridNodes);

		
		bool *tempNodes = (bool*)malloc(BinaryImageSize[0]*BinaryImageSize[1]*BinaryImageSize[2]*sizeof(bool));
		CUDA_SAFE_CALL( cudaMemcpy( tempNodes, gridNodes, (BinaryImageSize[0]*BinaryImageSize[1]*BinaryImageSize[2])*sizeof(bool), cudaMemcpyDeviceToHost ) );
		//for(int i =0 ; i < BinaryImageSize[2]; i++)
		//{
			
		//	CUDA_SAFE_CALL( cudaMemcpy( tempNodes+i*(BinaryImageSize[0]*BinaryImageSize[1]), gridNodes+i*(BinaryImageSize[0]*BinaryImageSize[1]), (BinaryImageSize[0]*BinaryImageSize[1])*sizeof(bool), cudaMemcpyDeviceToHost ) );
		//}
		
		
		
		
		LDNISLAContouring_SupportImageGeneration(solid, supt_mesh, suptNodes, gridNodes, rotBoundingBox, 
			clipPlaneNm, thickness, nSampleWidth, distortRatio, BinaryImageSize, anchorR, threshold, cylinderRadius, patternThickness);

		char inputStr[10];
		bool bSave = false;
		bool bDisplay = false;
		printf("\Write Image to BMP? (y/n): ");
		scanf("%s",inputStr);
		if (inputStr[0]=='y' || inputStr[0]=='Y') bSave=true;

		printf("\Display Image on Screen? (y/n): ");
		scanf("%s",inputStr);
		if (inputStr[0]=='y' || inputStr[0]=='Y') bDisplay=true;

		//cudaThreadSynchronize();
		printf("Support Image Size : %d x %d x %d \n",BinaryImageSize[0],(BinaryImageSize[1]-1),BinaryImageSize[2] );
		bool *tempSuptNodes = (bool*)malloc(BinaryImageSize[0]*(BinaryImageSize[1]-1)*BinaryImageSize[2]*sizeof(bool));
		//for(int i =0 ; i < BinaryImageSize[2]; i++)
		//{
		CUDA_SAFE_CALL( cudaMemcpy( tempSuptNodes, suptNodes, (BinaryImageSize[0]*(BinaryImageSize[1]-1)*BinaryImageSize[2])*sizeof(bool), cudaMemcpyDeviceToHost ) );
			//CUDA_SAFE_CALL( cudaMemcpy( tempSuptNodes, suptNodes, sizeof(bool), cudaMemcpyDeviceToHost ) );
			//CUDA_SAFE_CALL( cudaMemcpy( tempSuptNodes+i*(BinaryImageSize[0]*(BinaryImageSize[1]-1)), suptNodes+i*(BinaryImageSize[0]*(BinaryImageSize[1]-1)), (BinaryImageSize[0]*(BinaryImageSize[1]-1))*sizeof(bool), cudaMemcpyDeviceToHost ) );
		//}
		

		c_mesh->ArrayToImage( tempNodes, tempSuptNodes, BinaryImageSize, baseLayerNum, bSave, bDisplay);
		//c_mesh->ArrayToImage(tempNodes, BinaryImageSize);
		
		cudaFree(suptNodes);
		free(tempSuptNodes);
		free(tempNodes);
	}
	else
	{


		c_mesh->setImageOrigin(rotBoundingBox[0], rotBoundingBox[4]);
		c_mesh->setSampleWidth(nSampleWidth);

		LDNIFDMContouring_BinarySamlping(solid, c_mesh, rotBoundingBox, BinaryImageSize, angle, thickness, clipPlaneNm,
			nSampleWidth, gridNodes, stickStart, stickEnd, stickIndex, stickID, prevStickID, stickDir);


		cudaFree(stickIndex);
		//----------------------------------------------------------------------------------------------
		//	Step 3: Constrained Smoothing
		LDNIFDMContouring_BuildSearchStickIndex(stickID, prevStickID, c_mesh->GetTotalStickNum());

		LDNIFDMContouring_ConstrainedSmoothing(solid, c_mesh, rotBoundingBox, BinaryImageSize,
			nSampleWidth, stickStart, stickEnd, stickID, stickDir);


		cudaFree(stickStart);
		cudaFree(stickEnd);
		cudaFree(stickDir);
		cudaFree(stickID);

		//----------------------------------------------------------------------------------------------
		//	Step 4: Supporting Structure



		LDNISLAContouring_SupportImageGeneration(solid, supt_mesh, gridNodes, rotBoundingBox, 
			clipPlaneNm, thickness, nSampleWidth, distortRatio, BinaryImageSize,anchorR, threshold, cylinderRadius, patternThickness);


	}


	//cudaFree(gridNodes);
	//----------------------------------------------------------------------------------------------
	//	Step 6: Clean up memory
	printf("-------------------------------------------\n");
	printf("Total time for SLA contour generation and Readback %ld(ms)\n", clock()-time);
}

void LDNIcudaOperation::LDNIFDMContouring_GenerationwithSupporting(LDNIcudaSolid* solid, ContourMesh *c_mesh, ContourMesh *supt_mesh, float nSampleWidth)
{
	float thickness = 0.01;
	float distortRatio = 1.0;
	double distortError = distortRatio*nSampleWidth*nSampleWidth;
	double smoothErrorCoef = 0.001;
	int VSAiter = 20;
	int simpRatio = 10;
	float angle = 0.0;
	double movementCoef = 0.65;
	double rotBoundingBox[6];
	float supportingSize = 0.012;
	double clipPlaneNm[3] = {0.0,1.0,0.0};
	int neighborsNum = ceil(0.15*sqrt(0.012)/solid->GetSampleWidth()); //this is only needed for constant supporting size 

	bool *gridNodes;
	float2 *stickStart, *stickEnd;
	unsigned int *stickIndex;
	int *stickID;
	int *prevStickID;
	int2 *stickDir;
	
	//----------------------------------------------------------------------------------------------
	//	Step 1: initialization
	long time = clock();
	angle = LDNIFDMContouring_CompRotationBoundingBox(solid, rotBoundingBox, clipPlaneNm);
	int BinaryImageSize[3];
	BinaryImageSize[1] = (int)floor((rotBoundingBox[3]-rotBoundingBox[2])/thickness);
	BinaryImageSize[0] = (int)ceil((rotBoundingBox[1]-rotBoundingBox[0])/nSampleWidth);
	BinaryImageSize[2] = (int)ceil((rotBoundingBox[5]-rotBoundingBox[4])/nSampleWidth);


	//printf("Bounding box %f %f %f %f %f %f \n", rotBoundingBox[0], rotBoundingBox[1], rotBoundingBox[2], rotBoundingBox[3], rotBoundingBox[4], rotBoundingBox[5]);
	printf("-------------------------------------------\n");
	printf("Image Size : %d X %d X %d \n", BinaryImageSize[0],BinaryImageSize[1],BinaryImageSize[2]);
	float range;

	//range = MAX(rotBoundingBox[1]-rotBoundingBox[0], rotBoundingBox[3]-rotBoundingBox[2]);
	//range = MAX(range, rotBoundingBox[5]-rotBoundingBox[4]);

	c_mesh->setImageOrigin(rotBoundingBox[0], rotBoundingBox[4]);
	c_mesh->setSampleWidth(nSampleWidth);
	
	//----------------------------------------------------------------------------------------------
	//	Step 2: Binary Sampling
	LDNIFDMContouring_BinarySamlping(solid, c_mesh, rotBoundingBox, BinaryImageSize, angle, thickness, clipPlaneNm,
									nSampleWidth, gridNodes, stickStart, stickEnd, stickIndex, stickID, prevStickID, stickDir);


	cudaFree(stickIndex);
		//----------------------------------------------------------------------------------------------
	//	Step 3: Constrained Smoothing
	LDNIFDMContouring_BuildSearchStickIndex(stickID, prevStickID, c_mesh->GetTotalStickNum());
		
	bool bOutPutSGM = false;
	char inputStr[10];
	printf("\Export SGM File? (y/n): ");
	scanf("%s",inputStr);
	if (inputStr[0]=='y' || inputStr[0]=='Y') 
	{
		bOutPutSGM = true;
	}

	LDNIFDMContouring_ConstrainedSmoothing(solid, c_mesh, rotBoundingBox, BinaryImageSize,
										 nSampleWidth, stickStart, stickEnd, stickID, stickDir, bOutPutSGM);


	cudaFree(stickStart);
	cudaFree(stickEnd);
	cudaFree(stickDir);
	cudaFree(stickID);

	//----------------------------------------------------------------------------------------------
	//	Step 4: Supporting Structure

	
	LDNIFDMContouring_SupportContourGeneration(solid, supt_mesh, gridNodes, rotBoundingBox, 
						clipPlaneNm, thickness, nSampleWidth, distortRatio, BinaryImageSize, bOutPutSGM);
	
	
	//----SGM code by KaChun----//
	
		
	if (bOutPutSGM)
	{
		long time=clock();	printf("Starting to write the SMG file ...\n");
		c_mesh->Output_SGM_FILE(supt_mesh);		
		printf("Completed and take %ld (ms)\n",clock()-time);
		
	}
	
	//----SGM code by KaChun----//

	//cudaFree(gridNodes);
	//----------------------------------------------------------------------------------------------
	//	Step 6: Clean up memory
	printf("-------------------------------------------\n");
	printf("Total time for FDM contour generation and Readback %ld(ms)\n", clock()-time);
}


void LDNIcudaOperation::LDNISLAContouring_Generation(LDNIcudaSolid* solid, ContourMesh *c_mesh, int dx, int dz, float thickness)
{
	float distortRatio = 1.0;
	double clipPlaneNm[3] = {0.0,1.0,0.0};
	double rotBoundingBox[6];
	float angle = 0.0;
	long time = clock();
	angle = LDNIFDMContouring_CompRotationBoundingBox(solid, rotBoundingBox, clipPlaneNm);
	int BinaryImageSize[3];
	double imageRange[2];

	double scalefactor = (260/25.4)/(rotBoundingBox[1]-rotBoundingBox[0]);

	BinaryImageSize[0] = dx;
	BinaryImageSize[1] = (int)floor((rotBoundingBox[3]-rotBoundingBox[2])/thickness);
	BinaryImageSize[2] = dz;

	double gridwidthx = (260/25.4)/dx;
	double gridwidthz = (165/25.4)/dz;
	double pixelWidth = (gridwidthx > gridwidthz) ? gridwidthx : gridwidthz;
	imageRange[0] = dx*pixelWidth;
	imageRange[1] = dz*pixelWidth;

	c_mesh->SetImageResolution(BinaryImageSize);


	printf("Image Size : %d X %d X %d \n", dx,BinaryImageSize[1],dz);

	LDNISLAContouring_BinarySampling(solid, c_mesh, rotBoundingBox, BinaryImageSize, thickness, clipPlaneNm, pixelWidth, imageRange, angle);

	printf("-------------------------------------------\n");
	printf("Total time for SLA image generation and Readback %ld(ms)\n", clock()-time);

}

void LDNIcudaOperation::LDNIFDMContouring_Generation(LDNIcudaSolid* solid, ContourMesh *c_mesh, float nSampleWidth)
{
	float thickness = 0.01;
	float distortRatio = 1.0;
	double distortError = distortRatio*nSampleWidth*nSampleWidth;
	double smoothErrorCoef = 0.001;
	int VSAiter = 20;
	int simpRatio = 10;
	float angle = 0.0;
	double movementCoef = 0.65;
	double rotBoundingBox[6];
	float supportingSize = 0.012;
	double clipPlaneNm[3] = {0.0,1.0,0.0};
	int neighborsNum = ceil(0.15*sqrt(0.012)/solid->GetSampleWidth()); //this is only needed for constant supporting size 

	bool *gridNodes;
	float2 *stickStart, *stickEnd;
	unsigned int *stickIndex;
	int *stickID;
	int *prevStickID;
	int2 *stickDir;
	
	//----------------------------------------------------------------------------------------------
	//	Step 1: initialization
	long time = clock();
	angle = LDNIFDMContouring_CompRotationBoundingBox(solid, rotBoundingBox, clipPlaneNm);
	int BinaryImageSize[3];
	BinaryImageSize[1] = (int)floor((rotBoundingBox[3]-rotBoundingBox[2])/thickness);
	BinaryImageSize[0] = (int)ceil((rotBoundingBox[1]-rotBoundingBox[0])/nSampleWidth);
	BinaryImageSize[2] = (int)ceil((rotBoundingBox[5]-rotBoundingBox[4])/nSampleWidth);


	//printf("Bounding box %f %f %f %f %f %f \n", rotBoundingBox[0], rotBoundingBox[1], rotBoundingBox[2], rotBoundingBox[3], rotBoundingBox[4], rotBoundingBox[5]);
	printf("-------------------------------------------\n");
	printf("Image Size : %d X %d X %d \n", BinaryImageSize[0],BinaryImageSize[1],BinaryImageSize[2]);
	float range;

	
	c_mesh->setImageOrigin(rotBoundingBox[0], rotBoundingBox[4]);
	c_mesh->setSampleWidth(nSampleWidth);

	//range = MAX(rotBoundingBox[1]-rotBoundingBox[0], rotBoundingBox[3]-rotBoundingBox[2]);
	//range = MAX(range, rotBoundingBox[5]-rotBoundingBox[4]);

	
	
	//----------------------------------------------------------------------------------------------
	//	Step 2: Binary Sampling
	LDNIFDMContouring_BinarySamlping(solid, c_mesh, rotBoundingBox, BinaryImageSize, angle, thickness, clipPlaneNm,
									nSampleWidth, gridNodes, stickStart, stickEnd, stickIndex, stickID, prevStickID, stickDir);


	cudaFree(stickIndex);
	cudaFree(gridNodes);
	//----------------------------------------------------------------------------------------------
	//	Step 3: Constrained Smoothing
	LDNIFDMContouring_BuildSearchStickIndex(stickID, prevStickID, c_mesh->GetTotalStickNum());
		
	/*LDNIFDMContouring_ConstrainedSmoothing(solid, c_mesh, rotBoundingBox,  BinaryImageSize,	angle, thickness, 
										clipPlaneNm, nSampleWidth, stickStart, stickEnd, stickID);*/
	LDNIFDMContouring_ConstrainedSmoothing(solid, c_mesh, rotBoundingBox, BinaryImageSize,
										 nSampleWidth, stickStart, stickEnd, stickID, stickDir);


	cudaFree(stickStart);
	cudaFree(stickEnd);
	cudaFree(stickDir);
	cudaFree(stickID);

	//----------------------------------------------------------------------------------------------
	//	Step 4: Supporting Structure
	


	//----------------------------------------------------------------------------------------------
	//	Step 6: Clean up memory
	printf("-------------------------------------------\n");
	printf("Total time for FDM contour generation and Readback %ld(ms)\n", clock()-time);

	
	


}


void LDNIcudaOperation::LDNIFDMContouring_ConstrainedSmoothing(LDNIcudaSolid* solid, ContourMesh *c_mesh, double rotBoundingBox[], int imageSize[],
															   float nSampleWidth, float2 *stickStart, float2 *stickEnd, int *stickID, int2 *stickDir, bool bOutPutSGM)
{
	float *accMovement;
	short iteratorNum = 30, iterCounter;
	unsigned int i, j;
	double moveCoef = 0.65;
	double smoothCoef = 0.001;
	float2 *newStartSticks, *newEndSticks;//, *oldSticks;
	int stickNum = c_mesh->GetTotalStickNum();
	double imgOrigin[2];
	imgOrigin[0] = rotBoundingBox[0];
	imgOrigin[1] = rotBoundingBox[4];
	bool *bIterflag;
	bool isContinue = false;
	//----------------------------------------------------------------------------------------------
	//	Step 1: Build Table for Smoothed Stick
	

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(newStartSticks)		, stickNum*sizeof(float2) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)newStartSticks, 0		, stickNum*sizeof(float2) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(newEndSticks)			, stickNum*sizeof(float2) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)newEndSticks, 0			, stickNum*sizeof(float2) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(accMovement)			, stickNum*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)accMovement, 0.0			, stickNum*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(bIterflag)			, stickNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)bIterflag, true			, stickNum*sizeof(bool) ) );


	
	float *accum = (float*)malloc(stickNum*sizeof(float));
	bool *bflag = (bool*)malloc(imageSize[1]*sizeof(float));
	memset(bflag, 1, imageSize[1]*sizeof(bool));

	
	int start = 0;
	double acc = 0.0;
	for(iterCounter=1; iterCounter <= iteratorNum; iterCounter++)
	{
		//store the movement for each stick
		krFDMContouring_ConstrainedSmoothing<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(newStartSticks, newEndSticks, stickStart, stickEnd, 
			stickID, stickNum, stickDir, accMovement, moveCoef, nSampleWidth, bIterflag, make_double2(imgOrigin[0],imgOrigin[1]));

		CUDA_SAFE_CALL( cudaMemcpy( stickStart, newStartSticks, stickNum*2*sizeof(float), cudaMemcpyDeviceToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( stickEnd, newEndSticks, stickNum*2*sizeof(float), cudaMemcpyDeviceToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( accum, accMovement, stickNum*sizeof(float), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemset( (void*)accMovement, 0.0	, stickNum*sizeof(float) ) );


		isContinue = false;
		start = 0;
		for(i=0; i < imageSize[1] ; i++)
		{
			acc = 0.0;
			for(j=start; j < start + c_mesh->GetLayerStickNum(i); j++)
			{
				acc +=accum[j]; //accummulate the movement for each slice
				
			}
			if(((acc/c_mesh->GetLayerStickNum(i))) < smoothCoef*nSampleWidth) // stop iteration for particular layer
			{
				bflag[i] = false;
				thrust::device_ptr<bool> v(bIterflag);	
				thrust::fill(v+start, v+start+c_mesh->GetLayerStickNum(i), false);
			}
			
			start += c_mesh->GetLayerStickNum(i);

			isContinue = (isContinue || bflag[i]); //break until all iterations stop
		}
		
		if (!isContinue) break;

	}

	float *cpuStickStart = (float*)malloc(stickNum*2*sizeof(float));
	float *cpuStickEnd = (float*)malloc(stickNum*2*sizeof(float));
	int *cpuStickID = (int*)malloc(stickNum*sizeof(int));

	CUDA_SAFE_CALL( cudaMemcpy( cpuStickStart, newStartSticks, stickNum*2*sizeof(float), cudaMemcpyDeviceToHost ) );
	CUDA_SAFE_CALL( cudaMemcpy( cpuStickEnd, newEndSticks, stickNum*2*sizeof(float), cudaMemcpyDeviceToHost ) );
	CUDA_SAFE_CALL( cudaMemcpy( cpuStickID, stickID, stickNum*sizeof(int), cudaMemcpyDeviceToHost ) );

	//  StickID store the array position for previous stick

	c_mesh->ArrayToContour(cpuStickStart, cpuStickEnd, imgOrigin, nSampleWidth ); // for display only
	
	if (bOutPutSGM)
	{
		printf("bOutPutSGM %d \n", bOutPutSGM);
		int *cpuStickDir = (int*)malloc(2*stickNum*sizeof(int));
		CUDA_SAFE_CALL( cudaMemcpy( cpuStickDir, stickDir, 2*stickNum*sizeof(int), cudaMemcpyDeviceToHost ) );
		c_mesh->BuildContourTopology(cpuStickStart, cpuStickEnd, cpuStickID, stickNum, cpuStickDir);
	
		free(cpuStickDir);
	}

	free(cpuStickStart);
	free(cpuStickEnd);
	free(cpuStickID);




	cudaFree(newStartSticks);
	cudaFree(newEndSticks);
	cudaFree(accMovement);
	cudaFree(bIterflag);
	cudaFree(stickDir);
	free(accum);
	free(bflag);


}

//void LDNIcudaOperation::LDNIFDMContouring_ConstrainedSmoothing(LDNIcudaSolid* solid, ContourMesh *c_mesh, double rotBoundingBox[], int imageSize[],
//															   float angle, float thickness, double clipPlanNm[], float nSampleWidth,
//															   float2 *stickStart, float2 *stickEnd, int *stickID)
//{
//	float *accMovement;
//	short iteratorNum = 30, iterCounter;
//	double moveCoef = 0.65;
//	double smoothCoef = 0.001;
//	float2 *newStartSticks, *newEndSticks, *oldSticks;
//	int stickNum = c_mesh->GetTotalStickNum();
//	double imgOrigin[2];
//	imgOrigin[0] = rotBoundingBox[0];
//	imgOrigin[1] = rotBoundingBox[4];
//
//	//----------------------------------------------------------------------------------------------
//	//	Step 1: Build Table for Smoothed Stick
//
//	CUDA_SAFE_CALL( cudaMalloc( (void**)&(newStartSticks)		, stickNum*sizeof(float2) ) );
//	CUDA_SAFE_CALL( cudaMemset( (void*)newStartSticks, 0		, stickNum*sizeof(float2) ) );
//	CUDA_SAFE_CALL( cudaMalloc( (void**)&(newEndSticks)			, stickNum*sizeof(float2) ) );
//	CUDA_SAFE_CALL( cudaMemset( (void*)newEndSticks, 0			, stickNum*sizeof(float2) ) );
//	CUDA_SAFE_CALL( cudaMalloc( (void**)&(oldSticks)			, stickNum*sizeof(float2) ) );
//	CUDA_SAFE_CALL( cudaMemset( (void*)oldSticks, 0				, stickNum*sizeof(float2) ) );
//	CUDA_SAFE_CALL( cudaMalloc( (void**)&(accMovement)			, sizeof(float) ) );
//	CUDA_SAFE_CALL( cudaMemset( (void*)accMovement, 0.0			, sizeof(float) ) );
//
//
//	CUDA_SAFE_CALL( cudaMemcpy( oldSticks, stickStart, stickNum*2*sizeof(float), cudaMemcpyDeviceToDevice ) );
//
//	float *accum = (float*)malloc(sizeof(float));
//	for(iterCounter=1; iterCounter <= iteratorNum; iterCounter++)
//	{
//
//			
//			krFDMContouring_ConstrainedSmoothing<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(newStartSticks, newEndSticks, stickStart, stickEnd,
//																		stickID, stickNum, oldSticks, accMovement, moveCoef, nSampleWidth, make_double2(imgOrigin[0],imgOrigin[1]));
//
//			CUDA_SAFE_CALL( cudaMemcpy( stickStart, newStartSticks, stickNum*2*sizeof(float), cudaMemcpyDeviceToDevice ) );
//			CUDA_SAFE_CALL( cudaMemcpy( stickEnd, newEndSticks, stickNum*2*sizeof(float), cudaMemcpyDeviceToDevice ) );
//			CUDA_SAFE_CALL( cudaMemcpy( accum, accMovement, sizeof(float), cudaMemcpyDeviceToHost ) );
//
//			CUDA_SAFE_CALL( cudaMemset( (void*)accMovement, 0.0	, sizeof(float) ) );
//			printf("iteration %d  %f\n", iterCounter, accum[0]);
//			if(((accum[0]/stickNum)/nSampleWidth) < smoothCoef)
//				break;
//
//	}
//
//
//
//	float *cpuStickStart = (float*)malloc(stickNum*2*sizeof(float));
//	float *cpuStickEnd = (float*)malloc(stickNum*2*sizeof(float));
//	int *cpuStickID = (int*)malloc(stickNum*sizeof(int));
//
//	CUDA_SAFE_CALL( cudaMemcpy( cpuStickStart, newStartSticks, stickNum*2*sizeof(float), cudaMemcpyDeviceToHost ) );
//	CUDA_SAFE_CALL( cudaMemcpy( cpuStickEnd, newEndSticks, stickNum*2*sizeof(float), cudaMemcpyDeviceToHost ) );
//	CUDA_SAFE_CALL( cudaMemcpy( cpuStickID, stickID, stickNum*sizeof(int), cudaMemcpyDeviceToHost ) );
//
//	//CUDA_SAFE_CALL( cudaMemcpy( cpuStickStart, stickStart, stickNum*2*sizeof(float), cudaMemcpyDeviceToHost ) );
//	//CUDA_SAFE_CALL( cudaMemcpy( cpuStickEnd, stickEnd, stickNum*2*sizeof(float), cudaMemcpyDeviceToHost ) );
//	
//
//
//	c_mesh->ConvertContourToVSAMesh(cpuStickStart, cpuStickEnd, cpuStickID, stickNum);
//	//c_mesh->ArrayToContour(cpuStickStart, cpuStickEnd, imgOrigin, nSampleWidth );
//
//
//	free(cpuStickStart);
//	free(cpuStickEnd);
//	free(cpuStickID);
//
//
//	
//	cudaFree(newStartSticks);
//	cudaFree(newEndSticks);
//	cudaFree(oldSticks);
//}





void LDNIcudaOperation::LDNISLAContouring_GrowthAndSwallow(double t, bool *&suptNodes, bool *&seedNodes, int i, int imageSize[],  double nSampleWidth, short2 *disTextureA, short2 *disTextureB, int disTexSize)
{
	bool* m_RegionFinish;
	bool *tempNodes;
	bool *tempSeeds;

	double realThreshold = (2.5*nSampleWidth-nSampleWidth)/nSampleWidth;
	int gridRadius = (int)floor(realThreshold);
	int3 imgRes = make_int3(imageSize[0], imageSize[1], imageSize[2]);
	double realT = (t-nSampleWidth)/nSampleWidth;
	bool* bRegionFinish = (bool*)malloc(sizeof(bool));

	LDNISLAContouring_BuildDistanceMap(seedNodes, i, imageSize, disTextureA, disTextureB, disTexSize);

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(tempNodes), imageSize[0]*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)tempNodes, false, imageSize[0]*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(tempSeeds), imageSize[0]*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)tempSeeds, false, imageSize[0]*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(m_RegionFinish), sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)m_RegionFinish, false, sizeof(bool) ) );


	CUDA_SAFE_CALL( cudaMemcpy( tempNodes, seedNodes, imageSize[0]*imageSize[2]*sizeof(bool), cudaMemcpyDeviceToDevice ) );

	bRegionFinish[0] = false;



	while(!bRegionFinish[0])
	{
		bRegionFinish[0] = true;
		CUDA_SAFE_CALL( cudaMemset( (void*)m_RegionFinish, true, sizeof(bool) ) );

		//dilation
		CUDA_SAFE_CALL( cudaMemcpy( tempSeeds, tempNodes, imageSize[0]*imageSize[2]*sizeof(bool), cudaMemcpyDeviceToDevice ) );
		krFDMContouring_Dilation<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempSeeds, tempNodes, imageSize[0]*imageSize[2], imgRes, realThreshold, gridRadius, i);
		krFDMContouring_Filter1<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(seedNodes, tempNodes, imageSize[0]*imageSize[2], imgRes, i);


		//intersection with original input image
		krFDMContouring_Filter2<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(disTextureB, tempNodes, suptNodes, imageSize[0]*imageSize[2], imgRes, m_RegionFinish, realT*realT,  disTexSize, i);
		CUDA_SAFE_CALL( cudaMemcpy(bRegionFinish, m_RegionFinish, sizeof(bool), cudaMemcpyDeviceToHost) );
		krFDMContouring_Filter3<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempNodes, seedNodes, imageSize[0]*imageSize[2]);


	}

	//printf("test \n");
	krFDMContouring_Filter1<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(seedNodes, suptNodes, imageSize[0]*imageSize[2], imgRes, i);
	//printf("test 2\n");



	cudaFree(tempNodes);
	cudaFree(tempSeeds);
	cudaFree(m_RegionFinish);
}

void LDNIcudaOperation::LDNIFDMContouring_GrowthAndSwallow(bool *gridNodes, int i, int imageSize[], double t, double nSampleWidth, bool *&suptRegion, bool *&solidRegion, short2 *distanceMapA, short2 *distanceMapB , int disTexSize, bool *tempNodes, bool *tempSeeds)
{
	bool* m_RegionFinish;
	
//	bool *tempNodes;
//	bool *tempSeeds;

	double realThreshold = (2.5*nSampleWidth-nSampleWidth)/nSampleWidth;
	int gridRadius = (int)floor(realThreshold);
	int3 imgRes = make_int3(imageSize[0], imageSize[1], imageSize[2]);
	double realT = (t-nSampleWidth)/nSampleWidth;
	bool* bRegionFinish = (bool*)malloc(sizeof(bool));
	//long time = clock();

	LDNIFDMContouring_BuildDistanceMap(gridNodes, i, imageSize, distanceMapA, distanceMapB, disTexSize);
	//cudaThreadSynchronize();
//	printf("1 %ld(ms)\n", clock()-time);

	//CUDA_SAFE_CALL( cudaMalloc( (void**)&(tempNodes), imageSize[0]*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)tempNodes, false, imageSize[0]*imageSize[2]*sizeof(bool) ) );
	//CUDA_SAFE_CALL( cudaMalloc( (void**)&(tempSeeds), imageSize[0]*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)tempSeeds, false, imageSize[0]*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(m_RegionFinish), sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)m_RegionFinish, false, sizeof(bool) ) );


	CUDA_SAFE_CALL( cudaMemcpy( tempNodes, solidRegion, imageSize[0]*imageSize[2]*sizeof(bool), cudaMemcpyDeviceToDevice ) );

	bRegionFinish[0] = false;

	//time = clock();

	while(!bRegionFinish[0])
	{
		bRegionFinish[0] = true;
		CUDA_SAFE_CALL( cudaMemset( (void*)m_RegionFinish, true, sizeof(bool) ) );

		//dilation
		CUDA_SAFE_CALL( cudaMemcpy( tempSeeds, tempNodes, imageSize[0]*imageSize[2]*sizeof(bool), cudaMemcpyDeviceToDevice ) );
		krFDMContouring_Dilation<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempSeeds, tempNodes, imageSize[0]*imageSize[2], imgRes, realThreshold, gridRadius, i);
		krFDMContouring_Filter1<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(solidRegion, tempNodes, imageSize[0]*imageSize[2], imgRes, i);


		//intersection with original input image
		krFDMContouring_Filter2<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(distanceMapB, tempNodes, suptRegion, imageSize[0]*imageSize[2], imgRes, m_RegionFinish, realT*realT,  disTexSize, i);
		CUDA_SAFE_CALL( cudaMemcpy(bRegionFinish, m_RegionFinish, sizeof(bool), cudaMemcpyDeviceToHost) );
		krFDMContouring_Filter3<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempNodes, solidRegion, imageSize[0]*imageSize[2]);


	}

	krFDMContouring_Filter1<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(solidRegion, suptRegion, imageSize[0]*imageSize[2], imgRes, i);
	//cudaThreadSynchronize();
	//printf("2 %ld(ms)\n", clock()-time);


	
	//cudaFree(tempNodes);
	//cudaFree(tempSeeds);
	cudaFree(m_RegionFinish);


}

void LDNIcudaOperation::LDNIFDMContouring_Closing(int imageSize[], double t, double nSampleWidth, bool *&inputNodes, int i, bool *tempNodes, bool *tempSeeds)
{
	double realThreshold = (t-nSampleWidth)/nSampleWidth;
	int gridRadius = (int)floor(realThreshold);
	int3 imgRes = make_int3(imageSize[0], imageSize[1], imageSize[2]);
	int3 suptimgRes = make_int3(imageSize[0], imageSize[1]-1, imageSize[2]);

	//long time = clock();
	//bool *tempNodes;
	//bool *tempSeeds;
	//CUDA_SAFE_CALL( cudaMalloc( (void**)&(tempNodes), imageSize[0]*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)tempNodes, false, imageSize[0]*imageSize[2]*sizeof(bool) ) );
	//CUDA_SAFE_CALL( cudaMalloc( (void**)&(tempSeeds), imageSize[0]*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)tempSeeds, false, imageSize[0]*imageSize[2]*sizeof(bool) ) );
	cudaThreadSynchronize();
	//printf("1 %ld(ms) \n", clock()-time);	time = clock();
	//CUDA_SAFE_CALL( cudaMemcpy( tempNodes, inputNodes, imageSize[0]*imageSize[2]*sizeof(bool), cudaMemcpyDeviceToDevice ) );
	krFDMContouring_CopyNodesrom3Dto2D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempNodes, inputNodes, imageSize[0]*imageSize[2], suptimgRes, i);
	CUDA_SAFE_CALL( cudaMemcpy( tempSeeds, tempNodes, imageSize[0]*imageSize[2]*sizeof(bool), cudaMemcpyDeviceToDevice ) );


	krFDMContouring_Dilation<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempNodes, tempSeeds, imageSize[0]*imageSize[2], imgRes, realThreshold, gridRadius, i);
	cudaThreadSynchronize();
	//printf("2 %ld(ms) \n", clock()-time);	time = clock();
	
	CUDA_SAFE_CALL( cudaMemcpy( tempNodes, tempSeeds, imageSize[0]*imageSize[2]*sizeof(bool), cudaMemcpyDeviceToDevice ) );
	krFDMContouring_Erosion<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempNodes, tempSeeds, imageSize[0]*imageSize[2], imgRes, realThreshold, gridRadius);

	krFDMContouring_CopyNodesrom2Dto3D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempSeeds, inputNodes, imageSize[0]*imageSize[2], suptimgRes, i);
	cudaThreadSynchronize();
	//printf("3 %ld(ms) \n", clock()-time);	time = clock();


	//cudaFree(tempNodes);
}

void LDNIcudaOperation::LDNISLAContouring_AssistForImageGrouping(bool *&targetImg, bool *&tempImg, double increaseDist, int2 imgRes, int i, double nSampleWidth)
{
	bool *tempNodes;
	bool *lastemp;
	bool* m_RegionFinish;
	

	bool* bRegionFinish = (bool*)malloc(sizeof(bool));
	double realThreshold = (increaseDist-nSampleWidth)/nSampleWidth;
	int gridRadius = (int)floor(realThreshold);
	int3 imgR = make_int3(imgRes.x, 0, imgRes.y);

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(tempNodes) , imgRes.x*imgRes.y*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(lastemp) , imgRes.x*imgRes.y*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(m_RegionFinish), sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)m_RegionFinish, false, sizeof(bool) ) );




	CUDA_SAFE_CALL( cudaMemcpy( tempNodes, targetImg, imgRes.x*imgRes.y*sizeof(bool), cudaMemcpyDeviceToDevice ) );
	CUDA_SAFE_CALL( cudaMemcpy( lastemp, tempNodes, imgRes.x*imgRes.y*sizeof(bool), cudaMemcpyDeviceToDevice ) );

	bRegionFinish[0] = false;

	while(!bRegionFinish[0])
	{
		bRegionFinish[0] = true;
		CUDA_SAFE_CALL( cudaMemset( (void*)m_RegionFinish, true, sizeof(bool) ) );
		
		CUDA_SAFE_CALL( cudaMemcpy( lastemp, tempNodes, imgRes.x*imgRes.y*sizeof(bool), cudaMemcpyDeviceToDevice ) );
		krFDMContouring_Dilation<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(lastemp, tempNodes, imgRes.x*imgRes.y, imgR, realThreshold, gridRadius, i);
		krSLAContouring_Filter3<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(lastemp, tempNodes, imgRes.x*imgRes.y, i);

		krSLAContouring_Filter4<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempNodes, lastemp, targetImg, m_RegionFinish, imgRes.x*imgRes.y, i);

		CUDA_SAFE_CALL( cudaMemcpy(bRegionFinish, m_RegionFinish, sizeof(bool), cudaMemcpyDeviceToHost) );

	}

	krSLAContouring_ReverseNodes<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(targetImg, tempImg, tempNodes, imgRes.x*imgRes.y, i);


	
	cudaFree(tempNodes);
	cudaFree(lastemp);
	cudaFree(m_RegionFinish);

}





struct getAnchorDiff : public thrust::binary_function<short2, short2, bool>
{
	short2 radius;

	__host__ __device__
		getAnchorDiff(short2 _radius) : radius(_radius) {}

	__host__ __device__
		bool operator()(short2 x, short2 y)
	{
		if ((y.x-x.x)*(y.x-x.x)+(y.y-x.y)*(y.y-x.y) > radius.x*radius.y)
			return true;
		else
			return false;
	}
};



struct transformOp : public thrust::unary_function<unsigned int, unsigned int>
{
	unsigned int w, h;

	__host__ __device__
		transformOp(unsigned int _w, unsigned int _h) : w(_w), h(_h) {}


	__host__ __device__
		unsigned int operator()(unsigned int index)
	{
		unsigned int m,n;

		n = index%w;
		m = index/w;


		return (n*h+m);
	}
};

/*template <typename T>
struct transformPt : public thrust::unary_function<T,short2>
{
	T lo, hi;

	__host__ __device__
		transformPt(T _lo, T _hi) : lo(_lo), hi(_hi) {}

	__host__ __device__
		short2 operator()(T x)
	{
		
		return make_short2(0,1);
	}
};*/

unsigned int LDNIcudaOperation::LDNIFlooding_Color2DFlooding(unsigned int *&index, bool *&key, unsigned int *&value, unsigned int arrsize, unsigned int init, int2 imgRes)
{
	bool bflag[4] = {true};
	bool result = true;
	unsigned int* verify;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(verify)		, arrsize*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)verify, 0		, arrsize*sizeof(unsigned int) ) );

	thrust::device_ptr<bool> key_ptr(key);
	thrust::device_ptr<unsigned int> index_ptr(index);
	thrust::device_ptr<unsigned int> value_ptr(value);
	thrust::device_ptr<unsigned int> verify_ptr(verify);

	thrust::equal_to<bool> binary_pred;
	thrust::minimum<unsigned int>  binary_op;

		

	
	while(true)
	{
		//Step 1: forward flooding
		{
			//Sorting value and key base on index
			thrust::copy(index_ptr, index_ptr + arrsize, verify_ptr);
			thrust::sort_by_key(index_ptr, index_ptr + arrsize, value_ptr);
			thrust::sort_by_key(verify_ptr, verify_ptr + arrsize, key_ptr);


			//Copying the value before scan and compare with the result after scan, to see any changes
			thrust::copy(value_ptr, value_ptr + arrsize, verify_ptr);
			thrust::inclusive_scan_by_key(key_ptr, key_ptr + arrsize, value_ptr, value_ptr, binary_pred, binary_op); // in-place scan
			bflag[0] = thrust::equal(value_ptr, value_ptr + arrsize, verify_ptr);
		}

		//Step 2: downward flooding
		{
			//transform the index
			thrust::transform(index_ptr, index_ptr + arrsize, index_ptr, transformOp(imgRes.x, imgRes.y)); 

			//Sorting value and key base on index
			thrust::copy(index_ptr, index_ptr + arrsize, verify_ptr);
			thrust::sort_by_key(index_ptr, index_ptr + arrsize, value_ptr);
			thrust::sort_by_key(verify_ptr, verify_ptr + arrsize, key_ptr);


			//Copying the value before scan and compare with the result after scan, to see any changes
			thrust::copy(value_ptr, value_ptr + arrsize, verify_ptr);
			thrust::inclusive_scan_by_key(key_ptr, key_ptr + arrsize, value_ptr, value_ptr, binary_pred, binary_op); // in-place scan
			bflag[1] = thrust::equal(value_ptr, value_ptr + arrsize, verify_ptr);

			thrust::transform(index_ptr, index_ptr + arrsize, index_ptr, transformOp(imgRes.y, imgRes.x));
		}


		//Step 3: backware flooding
		{

			thrust::reverse(value_ptr, value_ptr + arrsize);
			thrust::reverse(key_ptr, key_ptr + arrsize);

			thrust::copy(index_ptr, index_ptr + arrsize, verify_ptr);
			thrust::sort_by_key(index_ptr, index_ptr + arrsize, value_ptr);
			thrust::sort_by_key(verify_ptr, verify_ptr + arrsize, key_ptr);

			//Copying the value before scan and compare with the result after scan, to see any changes
			thrust::copy(value_ptr, value_ptr + arrsize, verify_ptr);
			thrust::inclusive_scan_by_key(key_ptr, key_ptr + arrsize, value_ptr, value_ptr, binary_pred, binary_op); // in-place scan
			bflag[2] = thrust::equal(value_ptr, value_ptr + arrsize, verify_ptr);

			//reverse back, same as index
			thrust::reverse(value_ptr, value_ptr + arrsize);
			thrust::reverse(key_ptr, key_ptr + arrsize);
		}

		

		//Step 3: upward flooding
		{
			thrust::transform(index_ptr, index_ptr + arrsize, index_ptr, transformOp(imgRes.x, imgRes.y)); 

			thrust::reverse(value_ptr, value_ptr + arrsize);
			thrust::reverse(key_ptr, key_ptr + arrsize);

			thrust::copy(index_ptr, index_ptr + arrsize, verify_ptr);
			thrust::sort_by_key(index_ptr, index_ptr + arrsize, value_ptr);
			thrust::sort_by_key(verify_ptr, verify_ptr + arrsize, key_ptr);

			//Copying the value before scan and compare with the result after scan, to see any changes
			thrust::copy(value_ptr, value_ptr + arrsize, verify_ptr);
			thrust::inclusive_scan_by_key(key_ptr, key_ptr + arrsize, value_ptr, value_ptr, binary_pred, binary_op); // in-place scan
			bflag[3] = thrust::equal(value_ptr, value_ptr + arrsize, verify_ptr);

			//reverse back, same as index
			thrust::reverse(value_ptr, value_ptr + arrsize);
			thrust::reverse(key_ptr, key_ptr + arrsize);
			//thrust::reverse(index_ptr, index_ptr + arrsize);
			
			//transform the index
			thrust::transform(index_ptr, index_ptr + arrsize, index_ptr, transformOp(imgRes.y, imgRes.x));
		}

		

		//printf("flood 1 %d %d %d %d\n", bflag[0], bflag[1], bflag[2], bflag[3]);
		//break;
		//result = bflag[0] || bflag[1] || bflag[2] || bflag[3]; // false = no more flooding area
		if (bflag[0] && bflag[1] && bflag[2] && bflag[3]) break;

	}

		thrust::copy(index_ptr, index_ptr + arrsize, verify_ptr);
		thrust::sort_by_key(index_ptr, index_ptr + arrsize, value_ptr);
		thrust::sort_by_key(verify_ptr, verify_ptr + arrsize, key_ptr);

		
	return 0;

	

	

}



// Purpose : scan the whole image for remaining region, add anchors
// and exclude anchor-support region until there is no remainder
// Algorithm : 
// (1) for each discovered nodes, perform the dilation to find the corresponding region
// (2) using flooding to find the number of region (overlapped seen as one region) and identify them
// (3) find the anchor point for each region
// (4) exclude the anchor support region

void LDNIcudaOperation::LDNISLAContouring_ThirdClassCylinder(double threshold, bool *&targetImg, bool *&assistImg, bool *&tempImg, int2 imgRes, double nSampleWidth, int i, short2 *disTextureA, short2 *disTextureB, int disTexSize)
{

	
	int imageSize[3] = {imgRes.x, 0, imgRes.y};
	int nodeNum = imgRes.x*imgRes.y;
	int regionNum;
	unsigned int MAX_MARKER = max(imgRes.x, imgRes.y)*max(imgRes.x, imgRes.y)+10;//nodeNum+10;//((1 << 31)-1)*2 +1; // 4294967295

	thrust::device_ptr<bool> target_ptr(targetImg);
	int c_count = thrust::count(target_ptr, target_ptr + nodeNum, 1);



	if (c_count > 0)
	{
		//printf("start...\n");
		//printf("count %d %d \n", c_count, i);
		double realThreshold = (2.5*nSampleWidth-nSampleWidth)/nSampleWidth;
		int gridRadius = (int)floor(realThreshold);
		krFDMContouring_Dilation<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(targetImg, assistImg, nodeNum, make_int3(imgRes.x, i, imgRes.y), realThreshold, gridRadius, i);



		unsigned int* anchorPtIndex;
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(anchorPtIndex)		, nodeNum*sizeof(unsigned int) ) );
		CUDA_SAFE_CALL( cudaMemset( (void*)anchorPtIndex, 0			, nodeNum*sizeof(unsigned int) ) );

		unsigned int* anchorPtValue;
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(anchorPtValue)		, nodeNum*sizeof(unsigned int) ) );
		CUDA_SAFE_CALL( cudaMemset( (void*)anchorPtValue, 0			, nodeNum*sizeof(unsigned int) ) );


		// Initialize flooding index
		thrust::device_ptr<unsigned int> index_ptr(anchorPtIndex);
		thrust::sequence(index_ptr, index_ptr + nodeNum, 0, 1);

		// Initialize flooding value
		krSLAContouring_FillAnchorValue<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(targetImg, assistImg, anchorPtValue, MAX_MARKER, nodeNum, i);

		// region flooding
		//if (i==87)
		regionNum = LDNIFlooding_Color2DFlooding(anchorPtIndex, assistImg, anchorPtValue, nodeNum, MAX_MARKER, imgRes);

		// get the anchor point

		thrust::fill(index_ptr, index_ptr + nodeNum, MAX_MARKER);
		krSLAContouring_GetAnchorPoint<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(targetImg, anchorPtValue, anchorPtIndex, imgRes, nodeNum, i);
		// fill image
		CUDA_SAFE_CALL( cudaMemset( (void*)assistImg, false	, nodeNum*sizeof(bool) ) );
		//if (i==68)
		krSLAContouring_FillImageValue<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(assistImg, tempImg, anchorPtIndex, imgRes, nodeNum, MAX_MARKER, i);
		CUDA_SAFE_CALL( cudaMemset( (void*)targetImg, false	, nodeNum*sizeof(bool) ) );

		//if (i==104)
		//	krFDMContouring_Test2D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempImg, nodeNum, imgRes);

		//
		LDNISLAContouring_GrowthAndSwallow(threshold, targetImg, assistImg, i, imageSize,  nSampleWidth, disTextureA, disTextureB, disTexSize);


	


		cudaFree(anchorPtIndex);
		cudaFree(anchorPtValue);
		//printf("End...\n");

	}




	
	


}



void LDNIcudaOperation::LDNISLAContouring_GenerateConnectionforCylinders(unsigned int *linkLayerC, short *linkLayerD, 
																		 short2 *linkID, bool *gridNodes, bool *&suptNodes, int imageSize[], int linkThreshold, 
																		 int lengthofLayer, int furtherStepLength, int linkNum, double nSampleWidth)
{
	int *linkCount;
	bool *linkfilter;

	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(linkCount), sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)linkCount, 0, sizeof(unsigned int) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(linkfilter), linkNum*linkNum*sizeof(bool) ));
	CUDA_SAFE_CALL( cudaMemset( (void*)linkfilter, false, linkNum*linkNum*sizeof(bool)));

	krSLAContouring_CounterLink1<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(linkID, linkfilter, linkCount, linkNum*linkNum, linkNum, linkThreshold);

	/*int *numofLink = (int*)malloc(sizeof(int));
	CUDA_SAFE_CALL( cudaMemcpy( numofLink, linkCount, sizeof(int), cudaMemcpyDeviceToHost ) );

	printf("num of link %d \n", numofLink[0]);*/

	CUDA_SAFE_CALL( cudaMemset( (void*)linkCount, 0, sizeof(unsigned int) ) );
	krSLAContouring_FilterLink1<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(linkID, linkLayerC, linkLayerD, linkCount, linkfilter, linkNum*linkNum, linkNum, linkThreshold, lengthofLayer, furtherStepLength);

	
	int *numofLink = (int*)malloc(sizeof(int));
	CUDA_SAFE_CALL( cudaMemcpy( numofLink, linkCount, sizeof(int), cudaMemcpyDeviceToHost ) );

	printf("num of link %d \n", numofLink[0]);


	bool *blink = (bool*)malloc(linkNum*linkNum*sizeof(bool));
	CUDA_SAFE_CALL( cudaMemcpy( blink, linkfilter, linkNum*linkNum*sizeof(bool), cudaMemcpyDeviceToHost ) );
	unsigned int *cpu_layerC = (unsigned int*)malloc(linkNum*sizeof(unsigned int));
	CUDA_SAFE_CALL( cudaMemcpy( cpu_layerC, linkLayerC, linkNum*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
	short *cpu_layerD = (short*)malloc(linkNum*sizeof(short));
	CUDA_SAFE_CALL( cudaMemcpy( cpu_layerD, linkLayerD, linkNum*sizeof(short), cudaMemcpyDeviceToHost ) );


	bool *linkMap;
	bool *tempLayer;
	bool *bflagLayer;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(linkMap), imageSize[0]*imageSize[2]*sizeof(bool) ));
	CUDA_SAFE_CALL( cudaMemset( (void*)linkMap, false, imageSize[0]*imageSize[2]*sizeof(bool)));
	CUDA_SAFE_CALL( cudaMemset( (void*)linkCount, 0, sizeof(int)));

	int m, n;
	int *numofPixel = (int*)malloc(sizeof(int));
	int st_layer, ed_layer, numofLayer;
	int layerNumforOneCircle;

	for(int i=0; i < linkNum*linkNum; i++)
	{
		if (blink[i])
		{
			m = i%linkNum;	n = i/linkNum;
			numofPixel[0] = 0;


			krSLAContouring_CalConnectionMap<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(linkID, m, n, linkMap, linkCount, imageSize[0]*imageSize[2], make_int2(imageSize[0], imageSize[2]), nSampleWidth);

			
			krSLAContouring_CheckConnectionMap<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(linkID, m, n, linkMap, linkCount, imageSize[0]*imageSize[2], make_int2(imageSize[0], imageSize[2]));

			thrust::device_ptr<bool> target_ptr(linkMap);
			numofPixel[0] = thrust::count(target_ptr, target_ptr + (imageSize[0]*imageSize[2]), 1);

		
			

			numofPixel[0] = numofPixel[0] - 2 - 1;
			


			if (numofPixel[0]>lengthofLayer )
			{
				
				st_layer = cpu_layerC[m] >= cpu_layerC[n] ? cpu_layerC[m] : cpu_layerC[n];
				ed_layer = cpu_layerD[m] <= cpu_layerD[n] ? cpu_layerD[m] : cpu_layerD[n];
				numofLayer = abs(st_layer-ed_layer)+1-2-1; // if(this->layerind<startinglayer+2 || this->layerind>= endinglayer)
				
				

				CUDA_SAFE_CALL( cudaMalloc( (void**)&(tempLayer), numofLayer*imageSize[0]*imageSize[2]*sizeof(bool) ));
				CUDA_SAFE_CALL( cudaMemset( (void*)tempLayer, false, numofLayer*imageSize[0]*imageSize[2]*sizeof(bool)));
				CUDA_SAFE_CALL( cudaMalloc( (void**)&(bflagLayer), numofLayer*sizeof(bool) ));
				CUDA_SAFE_CALL( cudaMemset( (void*)bflagLayer, false, numofLayer*sizeof(bool)));

				
				layerNumforOneCircle = (numofPixel[0]-(lengthofLayer-1)+1)/furtherStepLength+1;

				int ccc = thrust::count(target_ptr, target_ptr + (imageSize[0]*imageSize[2]), 1);
				//printf("num of pixel %d %d %d %d \n", numofPixel[0],numofLayer, layerNumforOneCircle, ccc);
				
				krSLAContouring_ConnectionMapOnLayers<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(gridNodes, tempLayer, linkMap, bflagLayer, linkID, m, n, numofPixel[0], lengthofLayer, furtherStepLength, layerNumforOneCircle, ed_layer, st_layer, make_int3(imageSize[0], imageSize[2], imageSize[1]), imageSize[0]*imageSize[2]*numofLayer);
				krSLAContouring_DeleteMapOnLayers<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempLayer, bflagLayer, layerNumforOneCircle, make_int2(imageSize[0], imageSize[2]), imageSize[0]*imageSize[2]*numofLayer);
				krSLAContouring_FilterLink2<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(suptNodes, tempLayer, st_layer, ed_layer, make_int3(imageSize[0], imageSize[2], imageSize[1]-1), imageSize[0]*imageSize[2]*numofLayer);

/**/
				cudaFree(bflagLayer);
				cudaFree(tempLayer);

				//break;
			}

			CUDA_SAFE_CALL( cudaMemset( (void*)linkMap, false, imageSize[0]*imageSize[2]*sizeof(bool)));
			CUDA_SAFE_CALL( cudaMemset( (void*)linkCount, 0, sizeof(int)));
		}
		
	}
	

	

	cudaFree(linkMap);
	cudaFree(linkfilter);
	cudaFree(linkCount);

	free(numofLink);
	free(blink);
	free(cpu_layerC);
	free(cpu_layerD);
	free(numofPixel);


}

void LDNIcudaOperation::LDNISLAContouring_SupportImageGeneration(LDNIcudaSolid* solid, ContourMesh *c_mesh, bool *&suptNodes, bool *gridNodes, double rotBoundingBox[], 
																 double clipPlaneNm[], double thickness, double nSampleWidth, double distortRatio, int imageSize[], float anchorR, float thres, float cylinderRadius, float patternThickness)
{
	
	bool *tempImage;
	bool *targetImage;
	bool *assistImage;
	bool *temp3D;


	double anchorRadius = anchorR;
	double threshold = thres;

	int suptRadius = (int)floor(anchorRadius*1.414/nSampleWidth);
	int suptGridResX = (imageSize[0]-1)/suptRadius+1;
	int suptGridResZ = (imageSize[2]-1)/suptRadius+1;


	int i,j;
	int nodeNum = imageSize[0]*imageSize[2];
	int3 imgRes = make_int3(imageSize[0], imageSize[1], imageSize[2]);
	int3 suptimgRes = make_int3(imageSize[0], imageSize[1]-1, imageSize[2]);

	long time[10] = {0};

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(temp3D), imageSize[0]*(imageSize[1]-1)*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)temp3D, 0, imageSize[0]*(imageSize[1]-1)*imageSize[2]*sizeof(bool) ) );


	CUDA_SAFE_CALL( cudaMalloc( (void**)&(suptNodes), imageSize[0]*(imageSize[1]-1)*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)suptNodes, false, imageSize[0]*(imageSize[1]-1)*imageSize[2]*sizeof(bool) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(tempImage), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)tempImage, false, nodeNum*sizeof(bool) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(targetImage), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)targetImage, false, nodeNum*sizeof(bool) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(assistImage), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)assistImage, false, nodeNum*sizeof(bool) ) );


	short2 **disTextures = (short2 **) malloc(2 * sizeof(short2 *)); 

	int disTexSize = max(imageSize[0],imageSize[2]);
	int factor = ceil((float)disTexSize / BLOCKSIZE);
	disTexSize = BLOCKSIZE*factor;

	int disMemSize = disTexSize * disTexSize * sizeof(short2); 

	// Allocate 2 textures

	cudaMalloc((void **) &disTextures[0], disMemSize); 
	cudaMalloc((void **) &disTextures[1], disMemSize); 


	unsigned int *LinkIndex;
	CUDA_SAFE_CALL(cudaMalloc((void **) &LinkIndex, (nodeNum+1)*sizeof(unsigned int))); 
	CUDA_SAFE_CALL(cudaMemset((void*)LinkIndex, 0, (nodeNum+1)*sizeof(unsigned int)));



	long t;


	for(i=imageSize[1]-2; i>-1; i--)
	{
		t = clock();
		krSLAContouring_Initialization<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempImage, targetImage, gridNodes, nodeNum, imgRes, i);
		CUDA_SAFE_CALL( cudaMemcpy( assistImage, targetImage, nodeNum*sizeof(bool), cudaMemcpyDeviceToDevice ) );
		time[0]+= clock()-t;

		t = clock();
		LDNISLAContouring_GrowthAndSwallow(threshold, targetImage, tempImage, i, imageSize,  nSampleWidth, disTextures[0], disTextures[1], disTexSize);


		time[1]+= clock()-t;
		//add new support cylinder if necessary
		CUDA_SAFE_CALL( cudaMemset( (void*)tempImage, false, nodeNum*sizeof(bool) ) );
		CUDA_SAFE_CALL( cudaMemset( (void*)assistImage, false, nodeNum*sizeof(bool) ) );
		krSLAContouring_Filter1<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(assistImage, tempImage, targetImage, suptGridResX*suptGridResZ, make_int2(imageSize[0], imageSize[2]), suptRadius, i);


		//first step: support region growth in first class cylinders
		LDNISLAContouring_GrowthAndSwallow(anchorRadius, targetImage, assistImage, i, imageSize,  nSampleWidth, disTextures[0], disTextures[1], disTexSize);

		//if (i==103)



		//second step: prepare second class cylinders and perform support region growth in second class cylinders
		CUDA_SAFE_CALL( cudaMemset( (void*)assistImage, false, nodeNum*sizeof(bool) ) );
		krSLAContouring_OrthoSearchRemainAnchorZ<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(assistImage, tempImage, targetImage,  suptGridResX,
			make_int2(suptRadius, imageSize[2]), make_int2(imageSize[0], imageSize[2]), i);



		krSLAContouring_OrthoSearchRemainAnchorX<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(assistImage, tempImage, targetImage,  suptGridResZ,
			make_int2(suptRadius, imageSize[0]), make_int2(imageSize[0], imageSize[2]), i);



		LDNISLAContouring_GrowthAndSwallow(anchorRadius, targetImage, assistImage, i, imageSize,  nSampleWidth, disTextures[0], disTextures[1], disTexSize);

		//third step: prepare third class cylinders and support region growth in all third class cylinders
		CUDA_SAFE_CALL( cudaMemset( (void*)assistImage, false, nodeNum*sizeof(bool) ) );
		LDNISLAContouring_ThirdClassCylinder(anchorRadius, targetImage, assistImage, tempImage, make_int2(imageSize[0], imageSize[2]), nSampleWidth, i, disTextures[0], disTextures[1], disTexSize);

		//generate support structure map for this layer and update the cylinder position information arrays

		krSLAContouring_Filter5<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(gridNodes, tempImage, suptNodes, LinkIndex, nodeNum, imgRes, i);

		krFDMContouring_CopyNodesrom2Dto3D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempImage, temp3D, nodeNum, suptimgRes, i);
				
	}

	cudaFree(disTextures[0]);
	cudaFree(disTextures[1]);
	free(disTextures);
	cudaFree(assistImage);
	cudaFree(targetImage);
	cudaFree(tempImage);



	thrust::device_ptr<unsigned int> dev_ptr(LinkIndex); //	Wrap raw pointers with dev_ptr
	thrust::exclusive_scan(dev_ptr, dev_ptr+(nodeNum+1), dev_ptr); //	in-place scan
	unsigned int LinkNum=dev_ptr[nodeNum];
	printf("max links ----- %d\n",LinkNum);



	short *linkLayerD;
	short2 *linkID;
	unsigned int *linkLayerC;
	unsigned int *temp2D;

	CUDA_SAFE_CALL(cudaMalloc((void **) &temp2D, nodeNum*sizeof(unsigned int))); 
	CUDA_SAFE_CALL(cudaMemset((void*)temp2D, 0, nodeNum*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL(cudaMalloc((void **) &linkLayerC, LinkNum*sizeof(unsigned int))); 
	CUDA_SAFE_CALL(cudaMalloc((void **) &linkLayerD, LinkNum*sizeof(short))); 
	CUDA_SAFE_CALL(cudaMemset((void*)linkLayerD, 0, LinkNum*sizeof(short) ) );
	CUDA_SAFE_CALL(cudaMalloc((void **) &linkID, LinkNum*sizeof(short2))); 
	CUDA_SAFE_CALL(cudaMemset((void*)linkID, 0, LinkNum*sizeof(short2) ) );


	thrust::device_ptr<unsigned int> c_ptr(linkLayerC);
	thrust::fill(c_ptr, c_ptr + LinkNum, imgRes.y+1);

	krSLAContouring_FindAllLinks<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(LinkIndex, linkLayerC, linkLayerD, linkID, temp3D, temp2D, suptimgRes.x*suptimgRes.y*suptimgRes.z, suptimgRes);

	cudaFree(temp3D);
	cudaFree(temp2D);



	krSLAContouring_RelateAllLinksBetweenLayers<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(LinkIndex, linkLayerC, gridNodes, suptNodes, suptimgRes.x*suptimgRes.y*suptimgRes.z, imgRes);

	cudaFree(LinkIndex);




	CUDA_SAFE_CALL( cudaMalloc( (void**)&(tempImage), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)tempImage, false, nodeNum*sizeof(bool) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(targetImage), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)targetImage, false, nodeNum*sizeof(bool) ) );


	double realThreshold = (cylinderRadius*nSampleWidth-nSampleWidth)/nSampleWidth;
	int gridRadius = (int)floor(realThreshold);

	for(i=imageSize[1]-2; i>-1; i--)
	{
		
		krFDMContouring_CopyNodesrom3Dto2D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(targetImage, suptNodes, nodeNum, suptimgRes, i);
		CUDA_SAFE_CALL( cudaMemcpy( tempImage, targetImage, nodeNum*sizeof(bool), cudaMemcpyDeviceToDevice ) );
		krFDMContouring_Dilation<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(targetImage, tempImage, nodeNum, imgRes, realThreshold, gridRadius, i);
		krFDMContouring_CopyNodesrom2Dto3D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempImage, suptNodes, nodeNum, suptimgRes, i);
		

	}
	cudaFree(targetImage);
	cudaFree(tempImage);

	for(i=imageSize[1]-3; i>-1; i--)
		krFDMContouring_VerticalSpptPxlProp<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(gridNodes, suptNodes, nodeNum, imgRes, i);

	int linkThreshold = (int)(0.4/nSampleWidth);
	int lengthofLayer = linkThreshold/8;
	int furtherStepLength = lengthofLayer/2;



	LDNISLAContouring_GenerateConnectionforCylinders(linkLayerC, linkLayerD, linkID, gridNodes, suptNodes, imageSize, linkThreshold, 
		lengthofLayer, furtherStepLength, LinkNum, nSampleWidth);


	cudaFree(linkLayerC);
	cudaFree(linkLayerD);
	cudaFree(linkID);
	cudaFree(gridNodes);


	realThreshold = (patternThickness*nSampleWidth-nSampleWidth)/nSampleWidth;
	gridRadius = (int)floor(realThreshold);

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(tempImage), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)tempImage, false, nodeNum*sizeof(bool) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(targetImage), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)targetImage, false, nodeNum*sizeof(bool) ) );

	for(i=imageSize[1]-2; i>-1; i--)
	{
		
		krFDMContouring_CopyNodesrom3Dto2D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(targetImage, suptNodes, nodeNum, suptimgRes, i);
		CUDA_SAFE_CALL( cudaMemcpy( tempImage, targetImage, nodeNum*sizeof(bool), cudaMemcpyDeviceToDevice ) );
		krFDMContouring_Dilation<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(targetImage, tempImage, nodeNum, imgRes, realThreshold, gridRadius, i);
		krFDMContouring_CopyNodesrom2Dto3D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempImage, suptNodes, nodeNum, suptimgRes, i);
		
	}

	cudaFree(targetImage);
	cudaFree(tempImage);


}

//Debbie
void LDNIcudaOperation::LDNISLAContouring_SupportImageGeneration(LDNIcudaSolid* solid, ContourMesh *c_mesh, bool *gridNodes, double rotBoundingBox[], 
																   double clipPlaneNm[], double thickness, double nSampleWidth, double distortRatio, 
																   int imageSize[],float anchorR, float thres, float cylinderRadius, float patternThickness)
{
	bool *suptNodes;
	bool *tempImage;
	bool *targetImage;
	bool *assistImage;
	bool *temp3D;


	double anchorRadius = anchorR;
	double threshold = thres;

	int suptRadius = (int)floor(anchorRadius*1.414/nSampleWidth);
	int suptGridResX = (imageSize[0]-1)/suptRadius+1;
	int suptGridResZ = (imageSize[2]-1)/suptRadius+1;

	
	int i,j;
	int nodeNum = imageSize[0]*imageSize[2];
	int3 imgRes = make_int3(imageSize[0], imageSize[1], imageSize[2]);
	int3 suptimgRes = make_int3(imageSize[0], imageSize[1]-1, imageSize[2]);

	long time[10] = {0};

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(temp3D), imageSize[0]*(imageSize[1]-1)*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)temp3D, 0, imageSize[0]*(imageSize[1]-1)*imageSize[2]*sizeof(bool) ) );


	CUDA_SAFE_CALL( cudaMalloc( (void**)&(suptNodes), imageSize[0]*(imageSize[1]-1)*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)suptNodes, false, imageSize[0]*(imageSize[1]-1)*imageSize[2]*sizeof(bool) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(tempImage), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)tempImage, false, nodeNum*sizeof(bool) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(targetImage), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)targetImage, false, nodeNum*sizeof(bool) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(assistImage), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)assistImage, false, nodeNum*sizeof(bool) ) );


	short2 **disTextures = (short2 **) malloc(2 * sizeof(short2 *)); 

	int disTexSize = max(imageSize[0],imageSize[2]);
	int factor = ceil((float)disTexSize / BLOCKSIZE);
	disTexSize = BLOCKSIZE*factor;

	int disMemSize = disTexSize * disTexSize * sizeof(short2); 

	// Allocate 2 textures

	cudaMalloc((void **) &disTextures[0], disMemSize); 
	cudaMalloc((void **) &disTextures[1], disMemSize); 


	unsigned int *LinkIndex;
	CUDA_SAFE_CALL(cudaMalloc((void **) &LinkIndex, (nodeNum+1)*sizeof(unsigned int))); 
	CUDA_SAFE_CALL(cudaMemset((void*)LinkIndex, 0, (nodeNum+1)*sizeof(unsigned int)));



	long t;
	

	for(i=imageSize[1]-2; i>-1; i--)
	{
		t = clock();
		krSLAContouring_Initialization<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempImage, targetImage, gridNodes, nodeNum, imgRes, i);
		CUDA_SAFE_CALL( cudaMemcpy( assistImage, targetImage, nodeNum*sizeof(bool), cudaMemcpyDeviceToDevice ) );
		time[0]+= clock()-t;

		t = clock();
		LDNISLAContouring_GrowthAndSwallow(threshold, targetImage, tempImage, i, imageSize,  nSampleWidth, disTextures[0], disTextures[1], disTexSize);

		
		time[1]+= clock()-t;
		//add new support cylinder if necessary
		CUDA_SAFE_CALL( cudaMemset( (void*)tempImage, false, nodeNum*sizeof(bool) ) );
		CUDA_SAFE_CALL( cudaMemset( (void*)assistImage, false, nodeNum*sizeof(bool) ) );
		krSLAContouring_Filter1<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(assistImage, tempImage, targetImage, suptGridResX*suptGridResZ, make_int2(imageSize[0], imageSize[2]), suptRadius, i);

		
		//first step: support region growth in first class cylinders
		LDNISLAContouring_GrowthAndSwallow(anchorRadius, targetImage, assistImage, i, imageSize,  nSampleWidth, disTextures[0], disTextures[1], disTexSize);


		//second step: prepare second class cylinders and perform support region growth in second class cylinders
		CUDA_SAFE_CALL( cudaMemset( (void*)assistImage, false, nodeNum*sizeof(bool) ) );
		krSLAContouring_OrthoSearchRemainAnchorZ<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(assistImage, tempImage, targetImage,  suptGridResX,
																						make_int2(suptRadius, imageSize[2]), make_int2(imageSize[0], imageSize[2]), i);

		

		krSLAContouring_OrthoSearchRemainAnchorX<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(assistImage, tempImage, targetImage,  suptGridResZ,
																						make_int2(suptRadius, imageSize[0]), make_int2(imageSize[0], imageSize[2]), i);

		

		LDNISLAContouring_GrowthAndSwallow(anchorRadius, targetImage, assistImage, i, imageSize,  nSampleWidth, disTextures[0], disTextures[1], disTexSize);

		
		//third step: prepare third class cylinders and support region growth in all third class cylinders
		CUDA_SAFE_CALL( cudaMemset( (void*)assistImage, false, nodeNum*sizeof(bool) ) );
		LDNISLAContouring_ThirdClassCylinder(anchorRadius, targetImage, assistImage, tempImage, make_int2(imageSize[0], imageSize[2]), nSampleWidth, i, disTextures[0], disTextures[1], disTexSize);
		
		
		//generate support structure map for this layer and update the cylinder position information arrays

		krSLAContouring_Filter5<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(gridNodes, tempImage, suptNodes, LinkIndex, nodeNum, imgRes, i);

		


		krFDMContouring_CopyNodesrom2Dto3D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempImage, temp3D, nodeNum, suptimgRes, i);
		
		
	}

	cudaFree(disTextures[0]);
	cudaFree(disTextures[1]);
	free(disTextures);
	cudaFree(assistImage);
	cudaFree(targetImage);
	cudaFree(tempImage);

	
	
	thrust::device_ptr<unsigned int> dev_ptr(LinkIndex); //	Wrap raw pointers with dev_ptr
	thrust::exclusive_scan(dev_ptr, dev_ptr+(nodeNum+1), dev_ptr); //	in-place scan
	unsigned int LinkNum=dev_ptr[nodeNum];
	printf("max links ----- %d\n",LinkNum);

	

	short *linkLayerD;
	short2 *linkID;
	unsigned int *linkLayerC;
	unsigned int *temp2D;

	CUDA_SAFE_CALL(cudaMalloc((void **) &temp2D, nodeNum*sizeof(unsigned int))); 
	CUDA_SAFE_CALL(cudaMemset((void*)temp2D, 0, nodeNum*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL(cudaMalloc((void **) &linkLayerC, LinkNum*sizeof(unsigned int))); 
	//CUDA_SAFE_CALL(cudaMemset((void*)linkLayerC, imgRes.y+1, LinkNum*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL(cudaMalloc((void **) &linkLayerD, LinkNum*sizeof(short))); 
	CUDA_SAFE_CALL(cudaMemset((void*)linkLayerD, 0, LinkNum*sizeof(short) ) );
	CUDA_SAFE_CALL(cudaMalloc((void **) &linkID, LinkNum*sizeof(short2))); 
	CUDA_SAFE_CALL(cudaMemset((void*)linkID, 0, LinkNum*sizeof(short2) ) );


	thrust::device_ptr<unsigned int> c_ptr(linkLayerC);
	thrust::fill(c_ptr, c_ptr + LinkNum, imgRes.y+1);

	krSLAContouring_FindAllLinks<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(LinkIndex, linkLayerC, linkLayerD, linkID, temp3D, temp2D, suptimgRes.x*suptimgRes.y*suptimgRes.z, suptimgRes);

	cudaFree(temp3D);
	cudaFree(temp2D);

	

	krSLAContouring_RelateAllLinksBetweenLayers<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(LinkIndex, linkLayerC, gridNodes, suptNodes, suptimgRes.x*suptimgRes.y*suptimgRes.z, imgRes);

	cudaFree(LinkIndex);
	
	
	

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(tempImage), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)tempImage, false, nodeNum*sizeof(bool) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(targetImage), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)targetImage, false, nodeNum*sizeof(bool) ) );


	double realThreshold = (cylinderRadius*nSampleWidth-nSampleWidth)/nSampleWidth;
	int gridRadius = (int)floor(realThreshold);

	for(i=imageSize[1]-2; i>-1; i--)
	{
		krFDMContouring_CopyNodesrom3Dto2D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(targetImage, suptNodes, nodeNum, suptimgRes, i);
		CUDA_SAFE_CALL( cudaMemcpy( tempImage, targetImage, nodeNum*sizeof(bool), cudaMemcpyDeviceToDevice ) );
		krFDMContouring_Dilation<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(targetImage, tempImage, nodeNum, imgRes, realThreshold, gridRadius, i);
				

		krFDMContouring_CopyNodesrom2Dto3D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempImage, suptNodes, nodeNum, suptimgRes, i);
	}
	cudaFree(targetImage);
	cudaFree(tempImage);

	for(i=imageSize[1]-3; i>-1; i--)
		krFDMContouring_VerticalSpptPxlProp<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(gridNodes, suptNodes, nodeNum, imgRes, i);
	
	int linkThreshold = (int)(0.4/nSampleWidth);
	int lengthofLayer = linkThreshold/8;
	int furtherStepLength = lengthofLayer/2;


	

	LDNISLAContouring_GenerateConnectionforCylinders(linkLayerC, linkLayerD, linkID, gridNodes, suptNodes, imageSize, linkThreshold, 
		lengthofLayer, furtherStepLength, LinkNum, nSampleWidth);


	cudaFree(linkLayerC);
	cudaFree(linkLayerD);
	cudaFree(linkID);
	cudaFree(gridNodes);

	

	realThreshold = (patternThickness*nSampleWidth-nSampleWidth)/nSampleWidth;
	gridRadius = (int)floor(realThreshold);

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(tempImage), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)tempImage, false, nodeNum*sizeof(bool) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(targetImage), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)targetImage, false, nodeNum*sizeof(bool) ) );

	for(i=imageSize[1]-2; i>-1; i--)
	{
		krFDMContouring_CopyNodesrom3Dto2D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(targetImage, suptNodes, nodeNum, suptimgRes, i);
		CUDA_SAFE_CALL( cudaMemcpy( tempImage, targetImage, nodeNum*sizeof(bool), cudaMemcpyDeviceToDevice ) );
		krFDMContouring_Dilation<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(targetImage, tempImage, nodeNum, imgRes, realThreshold, gridRadius, i);
		krFDMContouring_CopyNodesrom2Dto3D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempImage, suptNodes, nodeNum, suptimgRes, i);
	}

	cudaFree(targetImage);
	cudaFree(tempImage);

	//contouring
	float2 *stickStart;
	float2 *stickEnd;
	unsigned int *stickIndex;
	int *stickID;
	int *prevStickID;
	int2 *stickDir;

	imageSize[1] = imageSize[1]-1;
	printf("Support Image Size : %d x %d x %d \n", imageSize[0], imageSize[1], imageSize[2]);

	LDNIFDMContouring_SupportFindAllStick(solid, c_mesh, rotBoundingBox, imageSize, thickness, nSampleWidth, suptNodes,	stickStart, stickEnd, 
		stickIndex, stickID, prevStickID, stickDir);


	cudaFree(stickIndex);
	cudaFree(suptNodes);
	//----------------------------------------------------------------------------------------------
	//	Step 3: Constrained Smoothing
	LDNIFDMContouring_BuildSearchStickIndex(stickID, prevStickID, c_mesh->GetTotalStickNum());


	LDNIFDMContouring_ConstrainedSmoothing(solid, c_mesh, rotBoundingBox, imageSize,
		nSampleWidth, stickStart, stickEnd, stickID, stickDir);


	cudaFree(stickStart);
	cudaFree(stickEnd);
	cudaFree(stickDir);
	cudaFree(stickID);


}

void LDNIcudaOperation::LDNIFDMContouring_SupportContourGeneration(LDNIcudaSolid* solid, ContourMesh *c_mesh, bool *gridNodes, double rotBoundingBox[], 
																   double clipPlaneNm[], double thickness, double nSampleWidth, double distortRatio, 
																   int imageSize[], bool bOutPutSGM)
{
	bool *suptNodes;
	bool *solidNodes;
	bool *suptTemp;
	bool *grossImage;
	bool *test1;
	bool *test2;
	
	double t = 0.025;
	double realT = (t-nSampleWidth)/nSampleWidth;
	int3 imgRes = make_int3(imageSize[0], imageSize[1], imageSize[2]);
	int3 suptimgRes = make_int3(imageSize[0], imageSize[1]-1, imageSize[2]);
	int i;
	int nodeNum = imageSize[0]*imageSize[2];
	
	long time[10] = {0,0,0,0,0,0,0,0,0,0};
	long te;

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(suptNodes), imageSize[0]*(imageSize[1]-1)*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)suptNodes, false, imageSize[0]*(imageSize[1]-1)*imageSize[2]*sizeof(bool) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(solidNodes), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)solidNodes, false, nodeNum*sizeof(bool) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(suptTemp), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)suptTemp, false, nodeNum*sizeof(bool) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(grossImage), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)grossImage, false, nodeNum*sizeof(bool) ) );


	CUDA_SAFE_CALL( cudaMalloc( (void**)&(test1), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)test1, false, nodeNum*sizeof(bool) ) );

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(test2), nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)test2, false, nodeNum*sizeof(bool) ) );

	int st;
	

	short2 **disTextures = (short2 **) malloc(2 * sizeof(short2 *)); 

	int disTexSize = max(imageSize[0],imageSize[2]);
	int factor = ceil((float)disTexSize / BLOCKSIZE);
	disTexSize = BLOCKSIZE*factor;

	int disMemSize = disTexSize * disTexSize * sizeof(short2); 

	// Allocate 2 textures
	cudaMalloc((void **) &disTextures[0], disMemSize); 
	cudaMalloc((void **) &disTextures[1], disMemSize); 


	
	for(i=imageSize[1]-2; i>-1; i--)
	{
		te = clock();
		CUDA_SAFE_CALL( cudaMemset( (void*)solidNodes, false, nodeNum*sizeof(bool) ) );
		krFDMContouring_SubtractSolidRegion<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(gridNodes, suptNodes, solidNodes, nodeNum, imgRes, i);
		cudaThreadSynchronize();
		time[0] += clock()-te;

		te = clock();
		krFDMContouring_CopyNodesrom3Dto2D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(suptTemp, suptNodes, nodeNum, suptimgRes, i);
		cudaThreadSynchronize();
		time[1] += clock()-te;

		te = clock();
		LDNIFDMContouring_GrowthAndSwallow(gridNodes, i, imageSize, t, nSampleWidth, suptTemp, solidNodes, disTextures[0], disTextures[1], disTexSize, test1, test2);
		cudaThreadSynchronize();
		time[2] += clock()-te;

		te = clock();
		krFDMContouring_Filter4<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(solidNodes, suptNodes, suptTemp, nodeNum, suptimgRes, i);
		cudaThreadSynchronize();
		time[3] += clock()-te;

		te = clock();
		krFDMContouring_integrateImageintoGrossImage<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(grossImage, suptNodes, nodeNum, suptimgRes, i);
		cudaThreadSynchronize();
		time[4] += clock()-te;

		te = clock();
		krFDMContouring_CopyNodesrom2Dto3D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(grossImage, suptNodes, nodeNum, suptimgRes, i);
		cudaThreadSynchronize();
		time[5] += clock() -te;

		te = clock();
		LDNIFDMContouring_Closing(imageSize, 2.0*t, nSampleWidth, suptNodes, i, test1, test2);
		cudaThreadSynchronize();
		time[6] += clock()-te;

		te = clock();
		krFDMContouring_Filter5<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(suptNodes, solidNodes, gridNodes, nodeNum, i, imgRes);
		cudaThreadSynchronize();
		time[7] += clock()-te;

		//krFDMContouring_Test<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(suptNodes, imageSize[0]*(imageSize[1]-1)*imageSize[2], suptimgRes);
	}

	cudaFree(test1);
	cudaFree(test2);
	cudaFree(disTextures[0]);
	cudaFree(disTextures[1]);
	free(disTextures);
	cudaFree(grossImage);
		
	//VerticalSpptPxlProp()
	te = clock();
	for(i=imageSize[1]-3; i>-1; i--)
		krFDMContouring_VerticalSpptPxlProp<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(gridNodes, suptNodes, nodeNum, imgRes, i);
	time[8] += clock()-te;
		
	//SeparatePartAndSpptFDM
	CUDA_SAFE_CALL( cudaMemset( (void*)solidNodes, false, nodeNum*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)suptTemp, false, nodeNum*sizeof(bool) ) );
	double realThreshold = (2.5*nSampleWidth-nSampleWidth)/nSampleWidth;
	int gridRadius = (int)floor(realThreshold);

	te = clock();
	for(i=imageSize[1]-2; i>-1; i--)
	{
		krFDMContouring_CopyNodesrom3Dto2D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(solidNodes, gridNodes, nodeNum, imgRes, i);
		CUDA_SAFE_CALL( cudaMemcpy( suptTemp, solidNodes, imageSize[0]*imageSize[2]*sizeof(bool), cudaMemcpyDeviceToDevice ) );
		krFDMContouring_Dilation<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(solidNodes, suptTemp, imageSize[0]*imageSize[2], imgRes, realThreshold, gridRadius, i);
		krFDMContouring_CopyNodesrom3Dto2D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(solidNodes, suptNodes, nodeNum, suptimgRes, i);
		krFDMContouring_Filter1<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(suptTemp ,solidNodes, imageSize[0]*imageSize[2], imgRes, i);
		krFDMContouring_CopyNodesrom2Dto3D<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(solidNodes, suptNodes, nodeNum, suptimgRes, i);

	}
	time[9] += clock()-te;

	printf("time (0): %ld(ms) \n", time[0]);
	printf("time (1): %ld(ms) \n", time[1]);
	printf("time (2): %ld(ms) \n", time[2]);
	printf("time (3): %ld(ms) \n", time[3]);
	printf("time (4): %ld(ms) \n", time[4]);
	printf("time (5): %ld(ms) \n", time[5]);
	printf("time (6): %ld(ms) \n", time[6]);
	printf("time (7): %ld(ms) \n", time[7]);
	printf("time (8): %ld(ms) \n", time[8]);
	printf("time (9): %ld(ms) \n", time[9]);

	

	printf("Finish image binary sampling for supporting. \n");

	cudaFree(suptTemp);
	cudaFree(solidNodes);
	cudaFree(gridNodes);


	//Generate Contour

	float2 *stickStart;
	float2 *stickEnd;
	unsigned int *stickIndex;
	int *stickID;
	int *prevStickID;
	int2 *stickDir;

	imageSize[1] = imageSize[1]-1;
	printf("Support Image Size : %d x %d x %d \n", imageSize[0], imageSize[1], imageSize[2]);

	LDNIFDMContouring_SupportFindAllStick(solid, c_mesh, rotBoundingBox, imageSize, thickness, nSampleWidth, suptNodes,	stickStart, stickEnd, 
										stickIndex, stickID, prevStickID, stickDir);
	
	
	cudaFree(stickIndex);
	cudaFree(suptNodes);
	//----------------------------------------------------------------------------------------------
	//	Step 3: Constrained Smoothing
	LDNIFDMContouring_BuildSearchStickIndex(stickID, prevStickID, c_mesh->GetTotalStickNum());
		

	LDNIFDMContouring_ConstrainedSmoothing(solid, c_mesh, rotBoundingBox, imageSize,
										 nSampleWidth, stickStart, stickEnd, stickID, stickDir, bOutPutSGM);


	cudaFree(stickStart);
	cudaFree(stickEnd);
	cudaFree(stickDir);
	cudaFree(stickID);


	


}


void LDNIcudaOperation::LDNIFDMContouring_BuildDistanceMap(bool *gridNodes, int layerID, int imageSize[], short2 *&disTexturesA, short2 *&disTexturesB,  int disTexSize)
{
	
	
	int phase1Band  = 16; 
	int phase2Band  = 16; 
	int phase3Band  = 16; 
	long time = clock();

	krFDMContouring_InitializedValue<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>( disTexturesA, disTexSize*disTexSize, MARKER);

	//Initialize the Distance Map
	krFDMContouring_InitializedDistanceMap<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(gridNodes, disTexturesA, imageSize[0]*imageSize[2], imageSize[0]*imageSize[2], 
																	make_int3(imageSize[0],imageSize[1],imageSize[2]), layerID, disTexSize);
	
	//printf("Start computing distance field of layer (%d) ......\n",layerID);
	//Phase 1:  Flood vertically in their own bands
	dim3 block = dim3(BLOCKSIZE);   
	dim3 grid = dim3(disTexSize / block.x, phase1Band); 
	
	
	cudaBindTexture(0, disTexColor, disTexturesA); 
	krFDMContouring_kernelFloodDown<<< grid, block >>>(disTexturesB, disTexSize, disTexSize / phase1Band); 
	cudaUnbindTexture(disTexColor); 
	

	cudaBindTexture(0, disTexColor, disTexturesB); 
	krFDMContouring_kernelFloodUp<<< grid, block >>>(disTexturesB, disTexSize, disTexSize / phase1Band); 
	

	//Phase 1:  Passing information between bands
	grid = dim3(disTexSize / block.x, phase1Band); 
	krFDMContouring_kernelPropagateInterband<<< grid, block >>>(disTexturesA, disTexSize, disTexSize / phase1Band); 
	

	cudaBindTexture(0, disTexLinks, disTexturesA); 
	krFDMContouring_kernelUpdateVertical<<< grid, block >>>(disTexturesB, disTexSize, phase1Band, disTexSize / phase1Band); 
	cudaUnbindTexture(disTexLinks); 
	cudaUnbindTexture(disTexColor); 
	

	//Phase 1: Transpose
	block = dim3(TILE_DIM, BLOCK_ROWS); 
	grid = dim3(disTexSize / TILE_DIM, disTexSize / TILE_DIM); 

	cudaBindTexture(0, disTexColor, disTexturesB); 
	krFDMContouring_kernelTranspose<<< grid, block >>>(disTexturesB, disTexSize); 
	cudaUnbindTexture(disTexColor); 
	


	//Phase 2: Compute proximate points locally in each band
	block = dim3(BLOCKSIZE);   
	grid = dim3(disTexSize / block.x, phase2Band); 
	cudaBindTexture(0, disTexColor, disTexturesB); 
	krFDMContouring_kernelProximatePoints<<< grid, block >>>(disTexturesA, disTexSize, disTexSize / phase2Band); 
	cudaBindTexture(0, disTexLinks, disTexturesA); 
	krFDMContouring_kernelCreateForwardPointers<<< grid, block >>>(disTexturesA, disTexSize, disTexSize / phase2Band);

	//Phase 2:  Repeatly merging two bands into one
	for (int noBand = phase2Band; noBand > 1; noBand /= 2) {
		grid = dim3(disTexSize / block.x, noBand / 2); 
		krFDMContouring_kernelMergeBands<<< grid, block >>>(disTexturesA, disTexSize, disTexSize / noBand); 
	}

	//Phase 2:  Replace the forward link with the X coordinate of the seed to remove the need of looking at the other texture. We need it for coloring.
	grid = dim3(disTexSize / block.x, disTexSize); 
	krFDMContouring_kernelDoubleToSingleList<<< grid, block >>>(disTexturesA, disTexSize);
	cudaUnbindTexture(disTexLinks); 
	cudaUnbindTexture(disTexColor); 


	//Phase 3: 
	block = dim3(BLOCKSIZE / phase2Band, phase2Band); 
	grid = dim3(disTexSize / block.x); 
	cudaBindTexture(0, disTexColor, disTexturesA); 
	krFDMContouring_kernelColor<<< grid, block >>>(disTexturesB, disTexSize); 
	cudaUnbindTexture(disTexColor); 


	//Phase 3: Transpose
	block = dim3(TILE_DIM, BLOCK_ROWS); 
	grid = dim3(disTexSize / TILE_DIM, disTexSize / TILE_DIM); 

	cudaBindTexture(0, disTexColor, disTexturesB); 
	krFDMContouring_kernelTranspose<<< grid, block >>>(disTexturesB, disTexSize); 
	cudaUnbindTexture(disTexColor); 

	//--------------------------------------------------------------------------------------------------------------------------------------------------
	//printf("Finish computing distance field of layer. %ld(ms)\n", clock()-time);
	


}


void LDNIcudaOperation::LDNISLAContouring_BuildDistanceMap(bool *seedNodes, int i, int imageSize[], short2 *&disTexturesA, short2 *&disTexturesB, int disTexSize)
{
	int phase1Band  = 16; 
	int phase2Band  = 16; 
	int phase3Band  = 16; 

	krFDMContouring_InitializedValue<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>( disTexturesA, disTexSize*disTexSize, MARKER);

	krSLAContouring_InitializedDistanceMap<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(seedNodes, disTexturesA, imageSize[0]*imageSize[2], make_int2(imageSize[0],imageSize[2]), i, disTexSize);

	dim3 block = dim3(BLOCKSIZE);   
	dim3 grid = dim3(disTexSize / block.x, phase1Band); 


	cudaBindTexture(0, disTexColor, disTexturesA); 
	krFDMContouring_kernelFloodDown<<< grid, block >>>(disTexturesB, disTexSize, disTexSize / phase1Band); 
	cudaUnbindTexture(disTexColor); 


	cudaBindTexture(0, disTexColor, disTexturesB); 
	krFDMContouring_kernelFloodUp<<< grid, block >>>(disTexturesB, disTexSize, disTexSize / phase1Band); 


	//Phase 1:  Passing information between bands
	grid = dim3(disTexSize / block.x, phase1Band); 
	krFDMContouring_kernelPropagateInterband<<< grid, block >>>(disTexturesA, disTexSize, disTexSize / phase1Band); 


	cudaBindTexture(0, disTexLinks, disTexturesA); 
	krFDMContouring_kernelUpdateVertical<<< grid, block >>>(disTexturesB, disTexSize, phase1Band, disTexSize / phase1Band); 
	cudaUnbindTexture(disTexLinks); 
	cudaUnbindTexture(disTexColor); 


	//Phase 1: Transpose
	block = dim3(TILE_DIM, BLOCK_ROWS); 
	grid = dim3(disTexSize / TILE_DIM, disTexSize / TILE_DIM); 

	cudaBindTexture(0, disTexColor, disTexturesB); 
	krFDMContouring_kernelTranspose<<< grid, block >>>(disTexturesB, disTexSize); 
	cudaUnbindTexture(disTexColor); 



	//Phase 2: Compute proximate points locally in each band
	block = dim3(BLOCKSIZE);   
	grid = dim3(disTexSize / block.x, phase2Band); 
	cudaBindTexture(0, disTexColor, disTexturesB); 
	krFDMContouring_kernelProximatePoints<<< grid, block >>>(disTexturesA, disTexSize, disTexSize / phase2Band); 
	cudaBindTexture(0, disTexLinks, disTexturesA); 
	krFDMContouring_kernelCreateForwardPointers<<< grid, block >>>(disTexturesA, disTexSize, disTexSize / phase2Band);

	//Phase 2:  Repeatly merging two bands into one
	for (int noBand = phase2Band; noBand > 1; noBand /= 2) {
		grid = dim3(disTexSize / block.x, noBand / 2); 
		krFDMContouring_kernelMergeBands<<< grid, block >>>(disTexturesA, disTexSize, disTexSize / noBand); 
	}

	//Phase 2:  Replace the forward link with the X coordinate of the seed to remove the need of looking at the other texture. We need it for coloring.
	grid = dim3(disTexSize / block.x, disTexSize); 
	krFDMContouring_kernelDoubleToSingleList<<< grid, block >>>(disTexturesA, disTexSize);
	cudaUnbindTexture(disTexLinks); 
	cudaUnbindTexture(disTexColor); 


	//Phase 3: 
	block = dim3(BLOCKSIZE / phase2Band, phase2Band); 
	grid = dim3(disTexSize / block.x); 
	cudaBindTexture(0, disTexColor, disTexturesA); 
	krFDMContouring_kernelColor<<< grid, block >>>(disTexturesB, disTexSize); 
	cudaUnbindTexture(disTexColor); 


	//Phase 3: Transpose
	block = dim3(TILE_DIM, BLOCK_ROWS); 
	grid = dim3(disTexSize / TILE_DIM, disTexSize / TILE_DIM); 

	cudaBindTexture(0, disTexColor, disTexturesB); 
	krFDMContouring_kernelTranspose<<< grid, block >>>(disTexturesB, disTexSize); 
	cudaUnbindTexture(disTexColor); 

}

// Given two array with unique index, we can build the connection between these two array 
//(these two array have the same size and same range of value)
void LDNIcudaOperation::LDNIFDMContouring_BuildSearchStickIndex(int *&stickID, int *prevStickID, int stickNum)
{
	unsigned int* key_index[2];  //0, 1, 2, 3, ....stickNum
		

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(key_index[0]), stickNum*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)key_index[0], 0, stickNum*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(key_index[1]), stickNum*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)key_index[1], 0, stickNum*sizeof(unsigned int) ) );

	//fill two array with index 0,1,2,3,...to stickNum
	thrust::device_ptr<unsigned int> index_ptr(key_index[0]);
	thrust::sequence(index_ptr, index_ptr + stickNum, 0, 1);

	thrust::device_ptr<unsigned int> prev_index_ptr(key_index[1]);
	thrust::sequence(prev_index_ptr, prev_index_ptr + stickNum, 0, 1);

	
	//---------------------------------------------------------------------------------------------------

	//reorder key_index[0] according to stickID.
	//Given e.g. stickID[] = { 512, 213, 768, 222, 111, 333, 555}, 
	//after sorting, 
	//stickID[] = {111, 213, 222, 333, 512, 555, 768}
	//key_index[0] = {4, 1, 3, 5, 0, 6, 2}
	thrust::device_ptr<int> _ptrKey(stickID);	
	thrust::sort_by_key(_ptrKey, _ptrKey + stickNum, index_ptr);
	

	//reorder key_index[1] according to prevstickID. 
	//Given e.g. prevstickID[] = { 555, 512, 213, 768, 222, 111, 333}, 
	//after sorting, 
	//prevstickID[] = {111, 213, 222, 333, 512, 555, 768}
	//key_index[1] = {5, 2, 4, 6, 1, 0, 3}
	thrust::device_ptr<int> _ptrPrevKey(prevStickID);	
	thrust::sort_by_key(_ptrPrevKey, _ptrPrevKey + stickNum, prev_index_ptr);

	//---------------------------------------------------------------------------------------------------

	//reorder key_index[1]
	//fill prevStickID[] = {0, 1, 2, 3, 4, 5, 6}
	//after sorting
	//key_index[1] = {0, 1, 2, 3, 4, 5, 6}
	//prevStickID[] = {5, 4, 1, 6, 2, 0, 3}
	thrust::fill(_ptrPrevKey, _ptrPrevKey + stickNum, 0);
	thrust::sequence(_ptrPrevKey, _ptrPrevKey + stickNum, 0, 1);
	thrust::sort_by_key(prev_index_ptr, prev_index_ptr + stickNum, _ptrPrevKey);


	//---------------------------------------------------------------------------------------------------
	
	//Get the prevID for each stickID 
	// prevStickID[] = {5, 4, 1, 6, 2, 0, 3}
    // key_index[0] = {4, 1, 3, 5, 0, 6, 2}
	CUDA_SAFE_CALL( cudaMemset( (void*)stickID, 0, stickNum*sizeof(unsigned int) ) );
	krFDMContouring_BuildHashTable<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(stickID, stickNum, key_index[0],  prevStickID);

	// stickID[] = {6, 0, 1, 2, 3, 4, 5}
	// so that we know 
	// stickID[0] previous node is in stickID[6]
	// stickID[1] previous node is in stickID[0]..etc

	cudaFree(key_index[0]);
	cudaFree(key_index[1]);
	cudaFree(prevStickID);

	
}


void LDNIcudaOperation::LDNISLAContouring_BinarySampling(LDNIcudaSolid* solid, ContourMesh *c_mesh, double rotBoundingBox[], int imageSize[], float thickness
														 , double clipPlanNm[], double pixelWidth, double imageRange[], float angle)
{

	int nRes = solid->GetResolution();
	float gwidth = solid->GetSampleWidth();
	float origin[3];
	solid->GetOrigin(origin[0],origin[1],origin[2]);
	int nodenum = imageSize[0]*imageSize[1]*imageSize[2];

	double imageOri[2];
	imageOri[0] = (rotBoundingBox[1]+rotBoundingBox[0])*0.5 - imageRange[0]*0.5;
	imageOri[1] = (rotBoundingBox[5]+rotBoundingBox[4])*0.5 - imageRange[1]*0.5;


	bool *gridNodes;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(gridNodes), imageSize[0]*imageSize[1]*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)gridNodes, 0, imageSize[0]*imageSize[1]*imageSize[2]*sizeof(bool) ) );


	krFDMContouring_BinarySampling<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(gridNodes, make_double3(clipPlanNm[0],clipPlanNm[1],clipPlanNm[2])
																	, angle, nRes, solid->GetIndexArrayPtr(0), solid->GetSampleDepthArrayPtr(0),
																	solid->GetIndexArrayPtr(1), solid->GetSampleDepthArrayPtr(1),
																	solid->GetIndexArrayPtr(2), solid->GetSampleDepthArrayPtr(2),
																	make_float3(origin[0], origin[1], origin[2]), 
																	make_double2(imageOri[0],imageOri[1]), gwidth, 
																	make_int3(imageSize[0],imageSize[1],imageSize[2]), nodenum, thickness, pixelWidth);

	
	bool *pt = (bool*)malloc(imageSize[0]*imageSize[1]*imageSize[2]*sizeof(bool));
	CUDA_SAFE_CALL( cudaMemcpy( pt, gridNodes, imageSize[0]*imageSize[1]*imageSize[2]*sizeof(bool), cudaMemcpyDeviceToHost ) );

	c_mesh->ArrayToImage(pt, imageSize);

	
	

	cudaFree(gridNodes);
	free(pt);


}




void LDNIcudaOperation::LDNIFDMContouring_SupportFindAllStick(LDNIcudaSolid* solid, ContourMesh *c_mesh, double rotBoundingBox[], int imageSize[], float thickness,  
										  float nSampleWidth, bool *suptNodes,  float2 *&stickStart, float2 *&stickEnd, unsigned int *&stickIndex, int *&stickID, int *&prevStickID,  int2 *&stickDir)
{
	int nRes = solid->GetResolution();
	float gwidth = solid->GetSampleWidth();
	float origin[3];
	solid->GetOrigin(origin[0],origin[1],origin[2]);
	int nodenum = imageSize[0]*imageSize[1]*imageSize[2];


	CUDA_SAFE_CALL( cudaMalloc( (void**)&(stickIndex), (imageSize[1]+1)*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)stickIndex, 0, (imageSize[1]+1)*sizeof(unsigned int) ) );

	
	krFDMContouring_CountAllStick<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(suptNodes, make_int3(imageSize[0]-1,imageSize[1],imageSize[2]-1), 
		(imageSize[0]-1)*imageSize[1]*(imageSize[2]-1), stickIndex);
	

	thrust::device_ptr<unsigned int> dev_ptr(stickIndex); //	Wrap raw pointers with dev_ptr
	thrust::exclusive_scan(dev_ptr, dev_ptr+(imageSize[1]+1), dev_ptr); //	in-place scan
	unsigned int s_Num = dev_ptr[imageSize[1]];
	printf("total number of supporting sticks ----- %d\n",s_Num);




	unsigned int* stickCounter;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(stickStart)	, s_Num*sizeof(float2) ) ); //start point of stick
	CUDA_SAFE_CALL( cudaMemset( (void*)stickStart, 0	, s_Num*sizeof(float2) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(stickEnd)		, s_Num*sizeof(float2) ) ); // end point of stick
	CUDA_SAFE_CALL( cudaMemset( (void*)stickEnd, 0		, s_Num*sizeof(float2) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(stickID)		, s_Num*sizeof(int) ) ); // stick id == grid id
	CUDA_SAFE_CALL( cudaMemset( (void*)stickID, 0		, s_Num*sizeof(int) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(prevStickID)	, s_Num*sizeof(int) ) ); // id of previous stick
	CUDA_SAFE_CALL( cudaMemset( (void*)prevStickID, 0	, s_Num*sizeof(int) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(stickCounter)	, (imageSize[1])*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)stickCounter, 0	, (imageSize[1])*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(stickDir)	, s_Num*sizeof(int2) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)stickDir, 0	, s_Num*sizeof(int2) ) );


	krFDMContouring_FindAllStickInAxisX<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(suptNodes, stickIndex, stickCounter,
											stickStart, stickEnd, make_int3(imageSize[0],imageSize[1],imageSize[2]),
						(imageSize[0]-1)*imageSize[1]*(imageSize[2]-1), make_double2(rotBoundingBox[0],rotBoundingBox[4]),
											nSampleWidth, stickID, prevStickID, stickDir);


	krFDMContouring_FindAllStickInAxisZ<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(suptNodes, stickIndex, stickCounter,
											stickStart, stickEnd, make_int3(imageSize[0],imageSize[1],imageSize[2]),
						(imageSize[0]-1)*imageSize[1]*(imageSize[2]-1), make_double2(rotBoundingBox[0],rotBoundingBox[4]),
											nSampleWidth, stickID, prevStickID, stickDir);

	

	//float *cpuStickStart = (float*)malloc(s_Num*2*sizeof(float));
	//float *cpuStickEnd = (float*)malloc(s_Num*2*sizeof(float));
	unsigned int *cpuStickIndex = (unsigned int*)malloc((imageSize[1]+1)*sizeof(unsigned int));
	

	//CUDA_SAFE_CALL( cudaMemcpy( cpuStickStart, stickStart, s_Num*2*sizeof(float), cudaMemcpyDeviceToHost ) );
	//CUDA_SAFE_CALL( cudaMemcpy( cpuStickEnd, stickEnd, s_Num*2*sizeof(float), cudaMemcpyDeviceToHost ) );
	CUDA_SAFE_CALL( cudaMemcpy( cpuStickIndex, stickIndex, (imageSize[1]+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
	

	c_mesh->SetThickness(thickness);
	c_mesh->SetOrigin(origin[0],origin[1],origin[2]);	
	c_mesh->MallocMemory(cpuStickIndex, imageSize, s_Num);
	//c_mesh->ArrayToContour(cpuStickStart, cpuStickEnd, cpuStickIndex);


	//free(cpuStickStart);
	//free(cpuStickEnd);
	free(cpuStickIndex);
	

	cudaFree(stickCounter);


	/*cudaFree(stickStart);
	cudaFree(stickEnd);
	cudaFree(stickIndex);
	cudaFree(prevStickID);
	cudaFree(stickDir);*/



}

// Binary Sampling Function for SLA
void LDNIcudaOperation::LDNIFDMContouring_BinarySamlping(LDNIcudaSolid* solid, ContourMesh *c_mesh, double rotBoundingBox[], 
														 int imageSize[], float angle, float thickness, double clipPlanNm[],
														 float nSampleWidth, bool *&gridNodes)
{
	int nRes = solid->GetResolution();
	float gwidth = solid->GetSampleWidth();
	float origin[3];
	solid->GetOrigin(origin[0],origin[1],origin[2]);
	int nodenum = imageSize[0]*imageSize[1]*imageSize[2];



	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(gridNodes), imageSize[0]*imageSize[1]*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)gridNodes, 0, imageSize[0]*imageSize[1]*imageSize[2]*sizeof(bool) ) );


	krFDMContouring_BinarySampling<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(gridNodes, make_double3(clipPlanNm[0],clipPlanNm[1],clipPlanNm[2])
								, angle, nRes, solid->GetIndexArrayPtr(0), solid->GetSampleDepthArrayPtr(0),
								solid->GetIndexArrayPtr(1), solid->GetSampleDepthArrayPtr(1),
								solid->GetIndexArrayPtr(2), solid->GetSampleDepthArrayPtr(2),
								make_float3(origin[0], origin[1], origin[2]), make_double2(rotBoundingBox[0],rotBoundingBox[4]), gwidth, make_int3(imageSize[0],imageSize[1],imageSize[2]), 
								nodenum, thickness, nSampleWidth);
	

	
}

// Binary Sampling Function for FDM
void LDNIcudaOperation::LDNIFDMContouring_BinarySamlping(LDNIcudaSolid* solid, ContourMesh *c_mesh, double rotBoundingBox[], 
														 int imageSize[], float angle, float thickness, double clipPlanNm[],
														 float nSampleWidth, bool *&gridNodes, float2 *&stickStart, 
														 float2 *&stickEnd, unsigned int *&stickIndex, 
														 int *&stickID, int *&prevStickID, int2 *&stickDir)
{
	//bool *gridNodes;
	int nRes = solid->GetResolution();
	float gwidth = solid->GetSampleWidth();
	float origin[3];
	solid->GetOrigin(origin[0],origin[1],origin[2]);
	int nodenum = imageSize[0]*imageSize[1]*imageSize[2];



	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(gridNodes), imageSize[0]*imageSize[1]*imageSize[2]*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)gridNodes, 0.0, imageSize[0]*imageSize[1]*imageSize[2]*sizeof(bool) ) );


	krFDMContouring_BinarySampling<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(gridNodes, make_double3(clipPlanNm[0],clipPlanNm[1],clipPlanNm[2])
								, angle, nRes, solid->GetIndexArrayPtr(0), solid->GetSampleDepthArrayPtr(0),
								solid->GetIndexArrayPtr(1), solid->GetSampleDepthArrayPtr(1),
								solid->GetIndexArrayPtr(2), solid->GetSampleDepthArrayPtr(2),
								make_float3(origin[0], origin[1], origin[2]), make_double2(rotBoundingBox[0],rotBoundingBox[4]), gwidth, make_int3(imageSize[0],imageSize[1],imageSize[2]), 
								nodenum, thickness, nSampleWidth);
	

	// Count the number of stick for each slice
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(stickIndex), (imageSize[1]+1)*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)stickIndex, 0, (imageSize[1]+1)*sizeof(unsigned int) ) );

	
	krFDMContouring_CountAllStick<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(gridNodes, make_int3(imageSize[0]-1,imageSize[1],imageSize[2]-1), 
																			(imageSize[0]-1)*imageSize[1]*(imageSize[2]-1), stickIndex);
	

	// Sum up and get the total number of stick
	thrust::device_ptr<unsigned int> dev_ptr(stickIndex); //	Wrap raw pointers with dev_ptr
	thrust::exclusive_scan(dev_ptr, dev_ptr+(imageSize[1]+1), dev_ptr); //	in-place scan
	unsigned int s_Num = dev_ptr[imageSize[1]];
	printf("total number of sticks ----- %d\n",s_Num);


	//float2 *stickStart;
	//float2 *stickEnd;
	unsigned int* stickCounter;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(stickStart)	, s_Num*sizeof(float2) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)stickStart, 0	, s_Num*sizeof(float2) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(stickEnd)		, s_Num*sizeof(float2) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)stickEnd, 0		, s_Num*sizeof(float2) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(stickID)		, s_Num*sizeof(int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)stickID, 0		, s_Num*sizeof(int) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(prevStickID)	, s_Num*sizeof(int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)prevStickID, 0	, s_Num*sizeof(int) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(stickCounter)	, (imageSize[1])*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)stickCounter, 0	, (imageSize[1])*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(stickDir)	, s_Num*sizeof(int2) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)stickDir, 0	, s_Num*sizeof(int2) ) );

	krFDMContouring_FindAllStickInAxisX<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(gridNodes, stickIndex, stickCounter,
										stickStart, stickEnd, make_int3(imageSize[0],imageSize[1],imageSize[2]),
										(imageSize[0]-1)*imageSize[1]*(imageSize[2]-1), make_double2(rotBoundingBox[0],rotBoundingBox[4]),
										nSampleWidth, stickID, prevStickID, stickDir);


	krFDMContouring_FindAllStickInAxisZ<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(gridNodes, stickIndex, stickCounter,
										stickStart, stickEnd, make_int3(imageSize[0],imageSize[1],imageSize[2]),
										(imageSize[0]-1)*imageSize[1]*(imageSize[2]-1), make_double2(rotBoundingBox[0],rotBoundingBox[4]),
										nSampleWidth, stickID, prevStickID, stickDir);


	//----------------------------------------------------------------------------------------------
	//	Display contour reconstructed by the topology preserving marching square method

	//float *cpuStickStart = (float*)malloc(s_Num*2*sizeof(float));
	//float *cpuStickEnd = (float*)malloc(s_Num*2*sizeof(float));
	unsigned int *cpuStickIndex = (unsigned int*)malloc((imageSize[1]+1)*sizeof(unsigned int));
	//unsigned int *test = (unsigned int*)malloc(s_Num*sizeof(unsigned int));
	
	//CUDA_SAFE_CALL( cudaMemcpy( cpuStickStart, stickStart, s_Num*2*sizeof(float), cudaMemcpyDeviceToHost ) );
	//CUDA_SAFE_CALL( cudaMemcpy( cpuStickEnd, stickEnd, s_Num*2*sizeof(float), cudaMemcpyDeviceToHost ) );
	CUDA_SAFE_CALL( cudaMemcpy( cpuStickIndex, stickIndex, (imageSize[1]+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
	//CUDA_SAFE_CALL( cudaMemcpy( test, stickID, s_Num*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );



	//unsigned int *cpuCount = (unsigned int*)malloc((imageSize[1])*sizeof(unsigned int));
	//CUDA_SAFE_CALL( cudaMemcpy( cpuCount, stickCounter, (imageSize[1])*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );

	//printf("tttttttttttttt %d \n", test[0]);

	//free(test);
	
	c_mesh->SetThickness(thickness);
	c_mesh->SetOrigin(origin[0],origin[1],origin[2]);	
	c_mesh->MallocMemory(cpuStickIndex, imageSize, s_Num);
	//c_mesh->ArrayToContour(cpuStickStart, cpuStickEnd, cpuStickIndex);
	

	//free(cpuStickStart);
	//free(cpuStickEnd);
	free(cpuStickIndex);
	//free(cpuCount);
	
	cudaFree(stickCounter);



	
}

#define C_EPS 1.0e-8
float LDNIcudaOperation::LDNIFDMContouring_CompRotationBoundingBox(LDNIcudaSolid* solid, double rotBoundingBox[], double clipPlaneNm[])
{
	
	//double C_EPS =	1.0e-8;
	double yAxis[3] = {0.0, 1.0, 0.0};
	int nRes = solid->GetResolution();
	double gWidth = solid->GetSampleWidth();
	float3 origin;
	
	rotBoundingBox[0] = rotBoundingBox[2] = rotBoundingBox[4] = 1.0e20;
	rotBoundingBox[1] = rotBoundingBox[3] = rotBoundingBox[5] = -1.0e20;
	solid->GetOrigin(origin.x, origin.y, origin.z);
	int arrsize = nRes*nRes, i ,j;
	short nAxis;
	
	float rotAngle = 0.0;
	GLKGeometry geo;
	{
		double rotateAxis[3];
		double theta = acos(geo.VectorProject(clipPlaneNm, yAxis)/sqrt(clipPlaneNm[0]*clipPlaneNm[0]+clipPlaneNm[1]*clipPlaneNm[1]+clipPlaneNm[2]*clipPlaneNm[2]));
		geo.VectorProduct(clipPlaneNm, yAxis, rotateAxis);
		

		if(theta<C_EPS*10000)
		{
			theta = 0.0;
			rotateAxis[0] = 1.0;
			rotateAxis[1] = 0.0;
			rotateAxis[2] = 0.0;
		}
		else if(PI-theta<C_EPS*10000)
		{
			theta = PI;
			rotateAxis[0] = 1.0;
			rotateAxis[1] = 0.0;
			rotateAxis[2] = 0.0;
		}
		geo.Normalize(rotateAxis);
		rotAngle = theta*57.29577951;
		
	}
	
	

	double *bdbox[3];
	unsigned int* countRay;
	unsigned int cRay;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(bdbox[0]), arrsize*2*sizeof(double) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(bdbox[1]), arrsize*2*sizeof(double) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(bdbox[2]), arrsize*2*sizeof(double) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(countRay), sizeof(unsigned int) ) );


	for(nAxis=0; nAxis < 3; nAxis++)
	{
		CUDA_SAFE_CALL( cudaMemset( (void*)bdbox[0], 0.0, arrsize*2*sizeof(double) ) );
		CUDA_SAFE_CALL( cudaMemset( (void*)bdbox[1], 0.0, arrsize*2*sizeof(double) ) );
		CUDA_SAFE_CALL( cudaMemset( (void*)bdbox[2], 0.0, arrsize*2*sizeof(double) ) );	
		CUDA_SAFE_CALL( cudaMemset(  countRay, 0, sizeof(unsigned int)) );


		krFDMContouring_RotationBoundingBox<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bdbox[0], bdbox[1], bdbox[2], countRay, 
														nAxis, make_double3(clipPlaneNm[0],clipPlaneNm[1],clipPlaneNm[2]), rotAngle, nRes, solid->GetIndexArrayPtr(nAxis),
														solid->GetSampleDepthArrayPtr(nAxis), origin, gWidth);

		CUDA_SAFE_CALL(cudaMemcpy( &cRay, countRay, sizeof(unsigned int),cudaMemcpyDeviceToHost));
		thrust::device_ptr<double> bdPtrX(bdbox[0]);
		thrust::device_vector<double>::iterator iter = thrust::max_element(bdPtrX, bdPtrX + cRay);
		rotBoundingBox[1] = MAX(*iter, rotBoundingBox[1]);
		iter = thrust::min_element(bdPtrX, bdPtrX + cRay);
		rotBoundingBox[0] = MIN(*iter, rotBoundingBox[0]);

		

		thrust::device_ptr<double> bdPtrY(bdbox[1]);
		iter = thrust::max_element(bdPtrY, bdPtrY + cRay);
		rotBoundingBox[3] = MAX(*iter, rotBoundingBox[3]);
		iter = thrust::min_element(bdPtrY, bdPtrY + cRay);
		rotBoundingBox[2] = MIN(*iter, rotBoundingBox[2]);

		

		thrust::device_ptr<double> bdPtrZ(bdbox[2]);
		iter = thrust::max_element(bdPtrZ, bdPtrZ + cRay);
		rotBoundingBox[5] = MAX(*iter, rotBoundingBox[5]);
		iter = thrust::min_element(bdPtrZ, bdPtrZ + cRay);
		rotBoundingBox[4] = MIN(*iter, rotBoundingBox[4]);

		


	}

	rotBoundingBox[1] += 14.733*gWidth;
	rotBoundingBox[0] -= 14.733*gWidth;
	rotBoundingBox[3] += 1.733 *gWidth;
	rotBoundingBox[2] -= 1.733 *gWidth;
	rotBoundingBox[5] += 14.733*gWidth;
	rotBoundingBox[4] -= 14.733*gWidth;

	


	cudaFree(bdbox[0]);		cudaFree(bdbox[1]);		cudaFree(bdbox[2]);
	return rotAngle;
}





void LDNIcudaOperation::LDNIToBRepReconstruction(LDNIcudaSolid* solid, QuadTrglMesh* &mesh, int nMeshRes, 
												 bool bWithIntersectionPrevention)
{
	int nAxis,nStepSize,res,zIndex;
	cudaEvent_t     startClock, stopClock;		float elapsedTime;
	CUDA_SAFE_CALL( cudaEventCreate( &startClock ) );
	CUDA_SAFE_CALL( cudaEventCreate( &stopClock ) );

	//----------------------------------------------------------------------------------------------
	//	Step 1: initialization
	res=solid->GetResolution();		nStepSize=(int)ceil((float)res/(float)nMeshRes);	if (nStepSize<1) nStepSize=1;
	printf("res=%d  nStepSize=%d  nMeshRes=%d\n",res,nStepSize,nMeshRes);

	//----------------------------------------------------------------------------------------------
	//	Step 2: build the grid nodes
	CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
	//--------------------------------------------------------------------------------------------------------
	unsigned int *compactIndexArray;	unsigned int emptyValue = (nMeshRes+1)*(nMeshRes+1) + 100;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(compactIndexArray), (nMeshRes+1)*(nMeshRes+1)*sizeof(unsigned int) ) );
	bool *gridNodes;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(gridNodes), (nMeshRes+1)*(nMeshRes+1)*(nMeshRes+1)*sizeof(bool) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)gridNodes, false, (nMeshRes+1)*(nMeshRes+1)*(nMeshRes+1)*sizeof(bool) ) );
	for(nAxis=0;nAxis<3;nAxis++) {
		//------------------------------------------------------------------------------------------
		//	Initialize the compact rays
		krLDNIContouring_initCompactIndexRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(compactIndexArray,nMeshRes,
							solid->GetIndexArrayPtr(nAxis),res,nStepSize,emptyValue);
 
		//------------------------------------------------------------------------------------------
		//	Compaction of the rays - by removing the empty rays
		thrust::device_ptr<unsigned int> devRayIndexPtr(compactIndexArray);
		int rayNum = thrust::remove(devRayIndexPtr, devRayIndexPtr + (nMeshRes+1)*(nMeshRes+1), emptyValue) - devRayIndexPtr;
//		printf("The resultant ray num: %d\n",rayNum);

		//------------------------------------------------------------------------------------------
		//	Voting for grid nodes
		krLDNIContouring_votingGridNodes<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(compactIndexArray,nMeshRes,rayNum,
							solid->GetIndexArrayPtr(nAxis),res,nStepSize,solid->GetSampleDepthArrayPtr(nAxis),
							gridNodes,solid->GetSampleWidth(),nAxis);
//		break;
	}
	cudaFree(compactIndexArray);	
	//--------------------------------------------------------------------------------------------------------
	CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
	CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
	printf("The construction of grid nodes takes: %3.1f (ms)\n",elapsedTime);

	//----------------------------------------------------------------------------------------------
	//	Step 3: build the boundary cells
	CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
	//--------------------------------------------------------------------------------------------------------
	const int maxBndCellNum=1024*1024*50;	// this is the space of 200MB since total size is BUFFER_SIZE*4 (i.e., 4 bytes for an "unisgned int")
	int zStep,bndCellNum;
	unsigned int invalidValue=nMeshRes*nMeshRes*nMeshRes+1000;
	zStep=1024*1024*10/(nMeshRes*nMeshRes);		printf("To find boundary cells, the step size is: %d x %d x %d\n",zStep,nMeshRes,nMeshRes);
	unsigned int *compactBndCell;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(compactBndCell), maxBndCellNum*sizeof(unsigned int) ) );
	unsigned int *bndCell;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(bndCell), zStep*nMeshRes*nMeshRes*sizeof(unsigned int) ) );
	bndCellNum=0;
	for(zIndex=0;zIndex<nMeshRes;zIndex+=zStep) {
		int stepSize=zStep;
		if (zIndex+stepSize>nMeshRes) stepSize=nMeshRes-zIndex;
		krLDNIContouring_constructBndCells<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(bndCell,gridNodes,nMeshRes,zIndex,stepSize, invalidValue);
		thrust::device_ptr<unsigned int> devBndCellIndexPtr(bndCell);
		int retainedCellNum = thrust::remove(devBndCellIndexPtr, devBndCellIndexPtr + stepSize*nMeshRes*nMeshRes, invalidValue) - devBndCellIndexPtr;
		if (retainedCellNum==0) continue;
//		printf("retainedCellNum=%d\n",retainedCellNum);
		CUDA_SAFE_CALL( cudaMemcpy( &(compactBndCell[bndCellNum]), bndCell, retainedCellNum*sizeof(unsigned int), cudaMemcpyDeviceToDevice ) );	
		bndCellNum+=retainedCellNum;
	}
	cudaFree(bndCell);
	printf("The boundary-cell num is: %d (maxBndCellNum = %d)\n",bndCellNum,maxBndCellNum);
	//--------------------------------------------------------------------------------------------------------
	CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
	CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
	printf("The construction of boundary cells takes: %3.1f (ms)\n",elapsedTime);

	//----------------------------------------------------------------------------------------------
	//	Step 4: generating vertices in the boundary cells
	CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
	//--------------------------------------------------------------------------------------------------------
	float *vPosArray;	float ox,oy,oz;		solid->GetOrigin(ox,oy,oz);
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(vPosArray), bndCellNum*3*sizeof(float) ) );
	krLDNIContouring_constructVertexArray<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(compactBndCell,bndCellNum,nMeshRes,nStepSize,
				solid->GetIndexArrayPtr(0),solid->GetSampleNxArrayPtr(0),solid->GetSampleNyArrayPtr(0),solid->GetSampleDepthArrayPtr(0),
				solid->GetIndexArrayPtr(1),solid->GetSampleNxArrayPtr(1),solid->GetSampleNyArrayPtr(1),solid->GetSampleDepthArrayPtr(1),
				solid->GetIndexArrayPtr(2),solid->GetSampleNxArrayPtr(2),solid->GetSampleNyArrayPtr(2),solid->GetSampleDepthArrayPtr(2),
				ox,oy,oz,solid->GetSampleWidth(),solid->GetResolution(),vPosArray,bWithIntersectionPrevention);
	//--------------------------------------------------------------------------------------------------------
	CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
	CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
	printf("The construction of vertex table takes: %3.1f (ms)\n",elapsedTime);	

	//----------------------------------------------------------------------------------------------
	//	Step 5: generating faces by the edges of boundary cells
	CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
	//--------------------------------------------------------------------------------------------------------
	unsigned int *faceArray[3];	int faceNum[3];		
	invalidValue=bndCellNum*2+100;		// The invalid face in the faceArray will be assigned with this value so that can be removed
										//	Note that we need to consider about the dynamically added vertices for preventing self-intersection
	float *vAdditionalVerArray;		int *additionalVerNum, addVerNum;	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(vAdditionalVerArray), bndCellNum*3*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(additionalVerNum), sizeof(int) ) );	
	CUDA_SAFE_CALL( cudaMemset( (void*)(additionalVerNum), 0, sizeof(int) ) );
	faceNum[0]=faceNum[1]=faceNum[2]=0;
	for(nAxis=0;nAxis<3;nAxis++) {
		unsigned int *tempFaceArray;	
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(tempFaceArray), bndCellNum*16*sizeof(unsigned int) ) );	

		krLDNIContouring_constructFaceArray<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
					compactBndCell,bndCellNum,gridNodes,nMeshRes,nStepSize,nAxis,tempFaceArray,invalidValue,vPosArray,
					ox,oy,oz,solid->GetSampleWidth(),vAdditionalVerArray,additionalVerNum,bWithIntersectionPrevention);
		thrust::device_ptr<unsigned int> devFaceArrayPtr(tempFaceArray);
		faceNum[nAxis] = (thrust::remove(devFaceArrayPtr, devFaceArrayPtr + bndCellNum*16, invalidValue) - devFaceArrayPtr)/4;
		//printf("faceNum=%d invalidValue=%d\n",faceNum[nAxis],invalidValue);

		if (faceNum[nAxis]>0) {
			CUDA_SAFE_CALL( cudaMalloc( (void**)&(faceArray[nAxis]), faceNum[nAxis]*4*sizeof(unsigned int) ) );	
			CUDA_SAFE_CALL( cudaMemcpy( faceArray[nAxis], tempFaceArray, faceNum[nAxis]*4*sizeof(unsigned int), cudaMemcpyDeviceToDevice ) );
		}

		cudaFree(tempFaceArray);
	}
	CUDA_SAFE_CALL( cudaMemcpy( &addVerNum, additionalVerNum, sizeof(int), cudaMemcpyDeviceToHost ) );
	printf("%d vertices have been constructed for the trivial triangles\n",addVerNum);
	//--------------------------------------------------------------------------------------------------------
	CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
	CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
	printf("The construction of face table takes: %3.1f (ms)\n",elapsedTime);	

	//----------------------------------------------------------------------------------------------
	//	Step 6: uploading the vertex table and the face table into a new QuadTrglMesh
	CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
	//--------------------------------------------------------------------------------------------------------
	mesh=new QuadTrglMesh;
	mesh->MallocMemory(bndCellNum+addVerNum, faceNum[0]+faceNum[1]+faceNum[2]);
	float *meshNodeTable = mesh->GetNodeTablePtr();
	CUDA_SAFE_CALL( cudaMemcpy( meshNodeTable, vPosArray, bndCellNum*3*sizeof(float), cudaMemcpyDeviceToHost ) );
	if (addVerNum>0) CUDA_SAFE_CALL( cudaMemcpy( &(meshNodeTable[bndCellNum*3]), vAdditionalVerArray, addVerNum*3*sizeof(float), cudaMemcpyDeviceToHost ) );
	unsigned int *meshFaceTable = mesh->GetFaceTablePtr();
	if (faceNum[0]>0) CUDA_SAFE_CALL( cudaMemcpy( &(meshFaceTable[0]), faceArray[0], faceNum[0]*4*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
	if (faceNum[1]>0) CUDA_SAFE_CALL( cudaMemcpy( &(meshFaceTable[faceNum[0]*4]), faceArray[1], faceNum[1]*4*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
	if (faceNum[2]>0) CUDA_SAFE_CALL( cudaMemcpy( &(meshFaceTable[(faceNum[0]+faceNum[1])*4]), faceArray[2], faceNum[2]*4*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
	//--------------------------------------------------------------------------------------------------------
	CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
	CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
	printf("Uploading the result into main memory takes: %3.1f (ms)\n",elapsedTime);	

	//----------------------------------------------------------------------------------------------
	//	Step 6: free the memory
	cudaFree(vPosArray);	cudaFree(vAdditionalVerArray);	cudaFree(additionalVerNum);
	for(nAxis=0;nAxis<3;nAxis++) {if (faceNum[nAxis]>0) cudaFree(faceArray[nAxis]);	}
	cudaFree(gridNodes);	cudaFree(compactBndCell);
	//----------------------------------------------------------------------------------------------
	CUDA_SAFE_CALL( cudaEventDestroy( startClock ) );
	CUDA_SAFE_CALL( cudaEventDestroy( stopClock ) );
}


//--------------------------------------------------------------------------------------------
//	Kernel functions 
//--------------------------------------------------------------------------------------------
//Debbie
#define TOID(x, y, size)    (__mul24((y), (size)) + (x))
__global__ void krSLAContouring_GetAnchorPoint(bool* targetImg, unsigned int* _value, unsigned int* anchorPt, int2 imgRes, int nodeNum, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ind, ix, iz;

	while (index<nodeNum) {

		if (targetImg[index])  
		{
			ix = index%imgRes.x; iz = index/imgRes.x;

			//if (iy == 68)
			//	printf("%d %d %d %d %d\n", ix, iz, index,_value[index], ix*imgRes.x+iz);

			//atomicMin( &anchorPt[_value[index]], ix*imgRes.x+iz);
			atomicMin( &anchorPt[_value[index]], ix*max(imgRes.x,imgRes.y)+iz);
		}		

		index += blockDim.x * gridDim.x;
	}

}

__global__ void krSLAContouring_FillImageValue(bool* tempImg, bool* suptImg, unsigned int* _value, int2 imgRes, int nodeNum, unsigned int init,  int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz, value;

	while (index<nodeNum) {

		value = _value[index];

		
		if (value < init)
		{
		ix = value%max(imgRes.x, imgRes.y);	iz = value/max(imgRes.x, imgRes.y);

		tempImg[ix*imgRes.x+iz] = true;
		suptImg[ix*imgRes.x+iz] = true;

		}


		index += blockDim.x * gridDim.x;
	}

}


__global__ void krSLAContouring_FillAnchorValue(bool* seedNodes, bool* inNodes, unsigned int* _value, unsigned int marker, int nodeNum, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ind;

	while (index<nodeNum) {

		if (inNodes[index] || seedNodes[index])
		{
			//if (iy == 68)
			//	printf("fill anchor %d %d %d \n", index, index%213, index/213);
			_value[index] = index;

			//if (iy >= 330)
			//	printf("fill anchor %d %d %d \n", index, index%151, index/151);
		}
		else
		{
			_value[index] = marker;
		}


		index += blockDim.x * gridDim.x;
	}

}


__global__ void krSLAContouring_Filter2DOr(bool* outNodes, bool* inNodes, int nodeNum, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	
	while (index<nodeNum) {
		
		outNodes[index] = outNodes[index] || inNodes[index];

		index += blockDim.x * gridDim.x;
	}

}

__global__ void krSLAContouring_FilterLink2(bool* suptNodes, bool* tempImgs, int startlayer, int endlayer, int3 imgRes, int nodeNum)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int ix, iy, iz;
	int layerID;

	while (index<nodeNum) {
		ix = index%imgRes.x;	iz = (index/imgRes.x)%imgRes.y;
		iy = index/(imgRes.x*imgRes.y);

		layerID = iy + startlayer + 2;

		if (!suptNodes[iz*imgRes.x*imgRes.z+layerID*imgRes.x+ix] && tempImgs[index])
		{
				suptNodes[iz*imgRes.x*imgRes.z+layerID*imgRes.x+ix] = true;
				//if (iz == 191) printf("filter %d %d %d \n", ix, iy, iz);
		}



		index += blockDim.x * gridDim.x;
	}

}

__global__ void krSLAContouring_DeleteMapOnLayers(bool* tempImg, bool* bflagDelete, int layerNumforOneCircle, int2 imgRes, int nodeNum)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int ix, iy, iz;
	int localID;

	while (index<nodeNum) {
		ix = index%imgRes.x;	iz = (index/imgRes.x)%imgRes.y;
		iy = index/(imgRes.x*imgRes.y);

		localID = iy%layerNumforOneCircle;
		if (bflagDelete[iy] && tempImg[index])
		{
			tempImg[index] = false;
		}

		index += blockDim.x * gridDim.x;
	}


}

__global__ void krSLAContouring_ConnectionMapOnLayers(bool* gridNodes, bool* tempImg, bool* linkMap, bool* bflagDelete, short2* linkID, int i, int j, int pixelNum, int lengthofLayer, int furtherStepLength, int layerNumforOneCircle, int endlayer, int startlayer, int3 imgRes, int nodeNum)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int ix, iy, iz, m;
	short2 st, ed;
	int localID;
	int layerID;
	int norm;
	st = linkID[i];
	ed = linkID[j];
	bool bflag = false;

	while (index<nodeNum) {
		ix = index%imgRes.x;	iz = (index/imgRes.x)%imgRes.y;
		iy = index/(imgRes.x*imgRes.y);


		layerID = startlayer+2+iy;
		localID = iy%layerNumforOneCircle;
		if(layerNumforOneCircle-localID > endlayer-layerID)
		{
			index += blockDim.x * gridDim.x;
			//if (ix == 0 && iz == 0) printf("@@ %d %d %d %d %d %d\n", layerNumforOneCircle, localID, startlayer, endlayer, layerID, iy);
			continue;
		}
		
		bflag = false;
		if (linkMap[iz*imgRes.x+ix])//&& iy==4)
		{
			//norm = abs(iz - st.y) + abs(ix - st.x) - 1;
			norm = abs(iz - ed.y) + abs(ix - ed.x) - 1;
			//printf("3: %d %d %d %d %d -- %d %d %d\n", ix, iz, ed.x, ed.y, norm, localID, layerNumforOneCircle-1, pixelNum-1);
			
			if (norm >= 0 && !((localID == layerNumforOneCircle-1) && (norm > pixelNum-1 )) )
			{

				if (norm >= localID*furtherStepLength && norm < localID*furtherStepLength+lengthofLayer)
				{
					tempImg[index] = true;
					//printf("1: %d %d %d %d %d %d %d %d\n", norm, ix, iz, iy, localID, layerID, startlayer, endlayer);
					bflag = true;
				}
				if (norm <= pixelNum-1-(localID*furtherStepLength) && norm > pixelNum-1-(localID*furtherStepLength+lengthofLayer))
				{
					tempImg[index] = true;
					//printf("2: %d %d %d %d %d %d %d %d\n", norm, ix, iz, iy, st.x, st.y, ed.x, ed.y);
					bflag = true;
				}
				

				if (bflag)
				{
					if (gridNodes[iz*imgRes.x*imgRes.z+(layerID)*imgRes.x+ix])
					{
						for(m=1; m <= layerNumforOneCircle-localID-1; m++)
						{
							bflagDelete[iy+m] = true;
							//printf("delete %d %d %d %d %d \n", ix, iz, layerID, iy, iy+m);
							//printf("delete %d \n", layerID);
						}
						
						for(m=iy; m >= iy-localID; m--)
						{
							bflagDelete[m] = true;
							//printf("delete %d \n", layerID);
						}
					}
				}

			}
		}
		



		index += blockDim.x * gridDim.x;
	}

}

__global__ void krSLAContouring_CheckConnectionMap(short2 *linkID, int i, int j, bool *bflag, int *pixelNum, int nodeNum, int2 imgRes)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	short2 st, ed;
	int ix, iy;
	bool btest1, btest2;

	st = linkID[i];
	ed = linkID[j];

	while (index<nodeNum) {
		ix = index%imgRes.x;	iy = index/imgRes.x;
		btest1 = false;
		btest2 = false;

		if (bflag[index])
		{
			if (ix <= max(st.x, ed.x) && ix >= min(st.x, ed.x))
			{
				if (iy <= max(st.y, ed.y) && iy >= min(st.y, ed.y))
				{
					if ((ix == st.x && iy == st.y) || (ix == ed.x && iy == ed.y))
					{
						index += blockDim.x * gridDim.x;
						continue;
					}

					if (ed.x > st.x)
					{
						if (ed.y > st.y)
						{
							if (bflag[(iy+1)*imgRes.x+ix])
								btest1 = true;
							if (bflag[(iy)*imgRes.x+ix+1])
								btest2 = true;

							if (!btest1 && !btest2)
							{
								bflag[(iy)*imgRes.x+ix+1] = true;
								atomicAdd(pixelNum,1);
							}
						}
						else if (ed.y < st.y)
						{
							if (bflag[(iy-1)*imgRes.x+ix])
								btest1 = true;
							if (bflag[(iy)*imgRes.x+ix+1])
								btest2 = true;

							if (!btest1 && !btest2)
							{
								bflag[(iy)*imgRes.x+ix+1] = true;
								atomicAdd(pixelNum,1);
							}
						}
					}
					else if (ed.x < st.x)
					{
						if (ed.y > st.y)
						{
							if (bflag[(iy+1)*imgRes.x+ix])
								btest1 = true;
							if (bflag[(iy)*imgRes.x+ix-1])
								btest2 = true;

							if (!btest1 && !btest2)
							{
								bflag[(iy)*imgRes.x+ix-1] = true;
								atomicAdd(pixelNum,1);
							}
						}
						else if (ed.y < st.y)
						{
							if (bflag[(iy-1)*imgRes.x+ix])
								btest1 = true;
							if (bflag[(iy)*imgRes.x+ix-1])
								btest2 = true;

							if (!btest1 && !btest2)
							{
								bflag[(iy)*imgRes.x+ix-1] = true;
								atomicAdd(pixelNum,1);
							}
						}
					}
				}
			}
		}



		index += blockDim.x * gridDim.x;
	}

}

__global__ void krSLAContouring_CalConnectionMap(short2 *linkID, int i, int j, bool *bflag, int *pixelNum, int nodeNum, int2 imgRes,
												 double nSampleWidth)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	short2 st, ed;
	unsigned int ix, iy;

	st = linkID[i];
	ed = linkID[j];

	float2 edgeSt, edgeEd;
	edgeSt.x = (st.x+0.5)*nSampleWidth;		edgeSt.y = (st.y+0.5)*nSampleWidth;
	edgeEd.x = (ed.x+0.5)*nSampleWidth;		edgeEd.y = (ed.y+0.5)*nSampleWidth;

	float2 edgeT1, edgeT2;
	double pt[2];

	bool intersect;
	//printf("st ed nnn %d %d %d %d \n", st.x, st.y, ed.x, ed.y);

	while (index<nodeNum) {
		ix = index%imgRes.x;	iy = index/imgRes.x;

		if (ix <= max(st.x, ed.x) && ix >= min(st.x, ed.x))
		{
			if (iy <= max(st.y, ed.y) && iy >= min(st.y, ed.y))
			{
				 
				if ((ix == st.x && iy == st.y) || (ix == ed.x && iy == ed.y))
				{
					bflag[index] = true;
					//printf("1 nnn %d %d %d %d %d %d %d \n", st.x, st.y, ed.x, ed.y, ix, iy, index);
					index += blockDim.x * gridDim.x;
					continue;
				}

				edgeT1.x = (ix)*nSampleWidth; edgeT1.y = (iy)*nSampleWidth;
				edgeT2.x = (ix+1)*nSampleWidth;	edgeT2.y = (iy)*nSampleWidth;
				
				intersect = _calTwoLineSegmentsIntersection(edgeT1, edgeSt, edgeEd, edgeT2, pt);

				if (intersect)
				{
					 bflag[index] = true;
					 //printf("2 nnn %d %d %d %d %d %d %d \n", st.x, st.y, ed.x, ed.y, ix, iy, index);
					 atomicAdd(pixelNum,1);
					 index += blockDim.x * gridDim.x;
					 continue;
				}

				edgeT1.x = (ix)*nSampleWidth; edgeT1.y = (iy)*nSampleWidth;
				edgeT2.x = (ix)*nSampleWidth;	edgeT2.y = (iy+1)*nSampleWidth;

				intersect = _calTwoLineSegmentsIntersection(edgeT1, edgeSt, edgeEd, edgeT2, pt);

				if (intersect)
				{
					bflag[index] = true;
					//printf("3 nnn %d %d %d %d %d %d %d \n", st.x, st.y, ed.x, ed.y, ix, iy, index);
					atomicAdd(pixelNum,1);
					index += blockDim.x * gridDim.x;
					continue;
				}

				edgeT1.x = (ix+1)*nSampleWidth; edgeT1.y = (iy)*nSampleWidth;
				edgeT2.x = (ix+1)*nSampleWidth;	edgeT2.y = (iy+1)*nSampleWidth;

				intersect = _calTwoLineSegmentsIntersection(edgeT1, edgeSt, edgeEd, edgeT2, pt);

				if (intersect)
				{
				    bflag[index] = true;
					//printf("4 nnn %d %d %d %d %d %d %d \n", st.x, st.y, ed.x, ed.y, ix, iy, index);
				    atomicAdd(pixelNum,1);
					index += blockDim.x * gridDim.x;
				    continue;
				}

				edgeT1.x = (ix)*nSampleWidth; edgeT1.y = (iy+1)*nSampleWidth;
				edgeT2.x = (ix+1)*nSampleWidth;	edgeT2.y = (iy+1)*nSampleWidth;

				intersect = _calTwoLineSegmentsIntersection(edgeT1, edgeSt, edgeEd, edgeT2, pt);

				if (intersect)
				{
				    bflag[index] = true;
					//printf("5 nnn %d %d %d %d %d %d %d \n", st.x, st.y, ed.x, ed.y, ix, iy, index);
				    atomicAdd(pixelNum,1);
					index += blockDim.x * gridDim.x;
				    continue;
				}

				edgeT1.x = (ix)*nSampleWidth; edgeT1.y = (iy+0.5)*nSampleWidth;
				edgeT2.x = (ix+1)*nSampleWidth;	edgeT2.y = (iy+0.5)*nSampleWidth;

				intersect = _calTwoLineSegmentsIntersection(edgeT1, edgeSt, edgeEd, edgeT2, pt);

				if (intersect)
				{
				   bflag[index] = true;
				  // printf("6 nnn %d %d %d %d %d %d %d \n", st.x, st.y, ed.x, ed.y, ix, iy, index);
				   atomicAdd(pixelNum,1);
				   index += blockDim.x * gridDim.x;
				   continue;

				}
			}
		}



		index += blockDim.x * gridDim.x;
	}
}


__global__ void krSLAContouring_CompressLink1(short2 *clinkID, int* count, bool *linkcount, int nodeNum, int linkNum)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iy, id;

	while (index<nodeNum) {
		ix = index%linkNum;	iy = index/linkNum;

		if (linkcount[index])
		{
			id = atomicAdd(count, 1);
			clinkID[id] = make_short2(ix,iy);
		}

		index += blockDim.x * gridDim.x;
	}
}


__global__ void krSLAContouring_FilterLink1(short2 *linkID, unsigned int *linklayerC, short *linklayerD,  int* count, bool *linkcount, int nodeNum, int linkNum, int linkThreshold, int lengthofLayer, int furtherStepLength)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iy;
	int a, b, c, d, overlapStartinglayerID, overlapEndinglayerID, diff, pixelNum;
	short2 e, f;
	

	while (index<nodeNum) {
		ix = index%linkNum;	iy = index/linkNum;

		if (linkcount[index])
		{
			
			a = linklayerC[ix];
			b = linklayerC[iy];

			

			if (a >= b)
				overlapStartinglayerID = a;
			else
				overlapStartinglayerID = b;

			
			c = linklayerD[ix];
			d = linklayerD[iy];

			

			if (c <= d)
				overlapEndinglayerID = c;
			else
				overlapEndinglayerID = d;

			diff = overlapEndinglayerID - overlapStartinglayerID;

			e = linkID[ix];
			f = linkID[iy];

			
			
			
			if (abs(e.x - f.x) != abs(e.y - f.y))
				pixelNum = abs(e.x - f.x) + abs(e.y - f.y) - 1;
			else
				pixelNum = abs(e.x - f.x) - 1;


			//if (f.x == 59 || e.x == 59)
			//{
				//printf("!! %d %d %d %d %d %d %d %d %d %d %d %d\n",  linkID[ix].x,  linkID[ix].y,  linkID[iy].x,  linkID[iy].y , a, b, c, d, overlapEndinglayerID, overlapStartinglayerID, diff, pixelNum);
				

			//}
			
			if (diff <= (pixelNum-(lengthofLayer-1)+1)/furtherStepLength+1) 
			{
				linkcount[index] = false;
				
			}
			else
			{
				
				 atomicAdd(count, 1);
			}
		}

		index += blockDim.x * gridDim.x;
	}

}

__global__ void krSLAContouring_CounterLink1(short2 *linkID, bool *linkcount, int* count, int nodeNum, int linkNum, int linkThreshold)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iy, iz, id, st;
	short a, b, c, d;
	bool node;

	while (index<nodeNum) {

		ix = index%linkNum;	iy = index/linkNum;

		if (ix <= iy) {
			index += blockDim.x * gridDim.x;
			continue;
		}

		//printf("%d %d %d %d %d %d %d\n", ix, iy, linkNum, linkID[ix].x, linkID[ix].y, linkID[iy].x, linkID[iy].y);
		a = linkID[ix].x;
		b = linkID[ix].y;
		c = linkID[iy].x;
		d = linkID[iy].y;



		if( (a-c)*(a-c)+(b-d)*(b-d) <= linkThreshold*linkThreshold)
		{
			linkcount[index] = true;
			atomicAdd(count, 1);
			//printf("-- %d %d %d %d %d %d\n", a, b, c, d, ix, iy);
		}
		
		index += blockDim.x * gridDim.x;

	}
}

__global__ void krSLAContouring_InitializedDistanceMap(bool *seedNodes, short2 *dMap, int nodeNum, int2 imageRes, int iy, int texwidth)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iz, id;
	while (index<nodeNum) {

			ix = index%imageRes.x;	iz = index/imageRes.x;
			id = iz*texwidth + ix;
			
			if (seedNodes[index])
			{
			
				dMap[id].x = ix;
				dMap[id].y = iz;
			}
			
		index += blockDim.x * gridDim.x;
	}
}

__global__ void krSLAContouring_OrthoSearchRemainAnchorZ(bool *assistImg, bool *tempImg, bool *targetImg, int nodeNum, int2 indRes, int2 imgRes, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, i, accumx, accumz, localcount;
	bool bstart;

	while (index<nodeNum) {
		ix = index*indRes.x;
		bstart = false;
		accumx = 0;
		accumz = 0;
		localcount=0;
		for(i=0; i < indRes.y; i++)
		{
			if (targetImg[i*imgRes.x+ix])
			{
				bstart = true;
				localcount++;
				accumx += ix;
				accumz += i;
			}
			if(!targetImg[i*imgRes.x+ix] && bstart)
			{
				bstart = false;
				accumx /= localcount;
				accumz /= localcount;
				assistImg[accumz*imgRes.x+accumx] = true;
				tempImg[accumz*imgRes.x+accumx]  = true;
				localcount = 0;
				accumx = 0;
				accumz = 0;
			}
		}
		

		
		index += blockDim.x * gridDim.x;
	}
}

__global__ void krSLAContouring_OrthoSearchRemainAnchorX(bool *assistImg, bool *tempImg, bool *targetImg, int nodeNum, int2 indRes, int2 imgRes, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int iz, i, accumx, accumz, localcount;
	bool bstart;

	while (index<nodeNum) {
		iz = index*indRes.x;
		bstart = false;
		accumx = 0;
		accumz = 0;
		localcount=0;
		for(i=0; i < indRes.y; i++)
		{
			if (targetImg[iz*imgRes.x+i])
			{
				bstart = true;
				localcount++;
				accumx += i;
				accumz += iz;
			}
			if(!targetImg[iz*imgRes.x+i] && bstart)
			{
				bstart = false;
				accumx /= localcount;
				accumz /= localcount;
				assistImg[accumz*imgRes.x+accumx] = true;
				tempImg[accumz*imgRes.x+accumx]  = true;
				localcount = 0;
				accumx = 0;
				accumz = 0;
			}
		}



		index += blockDim.x * gridDim.x;
	}
}


__global__ void krSLAContouring_CopyNodesToIntX(bool *nodeImg, unsigned int *intex, int nodeNum, int2 indRes, int2 imgRes, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz, i, value, id;

	while (index<nodeNum) {
		ix = index%indRes.x;	iz = index/indRes.x;
		value = 0;
		for (i=0; i < 32; i++)
		{
			id = (iz*indRes.y)*imgRes.x + ix*32+i;
			if (nodeImg[id])
				value = value | SetBitPos(i);
		}

		intex[index] = value;
		index += blockDim.x * gridDim.x;
	}
}


__global__ void krSLAContouring_CopyNodesToIntZ(bool *nodeImg, unsigned int *intex, int nodeNum, int2 indRes, int2 imgRes, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz, i, value, id;

	while (index<nodeNum) {
		ix = index%indRes.x;	iz = index/indRes.x;
		value = 0;
		for (i=0; i < 32; i++)
		{
			id = (iz*32+i)*imgRes.x + ix*indRes.x;
			if (nodeImg[id])
				value = value | SetBitPos(i);
		}

		intex[index] = value;
		index += blockDim.x * gridDim.x;
	}
}

__global__ void krSLAContouring_ReverseNodes(bool *outNodes, bool *outNodes2, bool *inNodes, int nodeNum, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz;

	while (index<nodeNum) {
		
		ix = index%213;	iz = index/213;
		if (inNodes[index])
		{
			//if (iy == 104)
			//printf("---%d %d %d\n", ix, iy, iz);
			outNodes[index] = false;
			outNodes2[index] = false;
		}


		index += blockDim.x * gridDim.x;
	}
}

__global__ void krSLAContouring_RelateAllLinksBetweenLayers(unsigned int *linkIndex, unsigned int *linkLayerC, bool *gridNodes, bool *suptNodes, int nodeNum, int3 imageRes)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iy, iz, id, s_id, st, num, i;
	bool node, node2;

	while (index<nodeNum) {

		ix = index%imageRes.x;	iy = (index/imageRes.x)%imageRes.y;
		iz = index/(imageRes.x*imageRes.y);

		
		s_id = iz*imageRes.x*(imageRes.y-1)+(iy+1)*imageRes.x+ix; // Mark: could be error when iy+1 >= imageRes.y-1

		if ((iy+1)>=imageRes.y-1) node2 = false;
		else node2 = suptNodes[s_id];

		if(node2 && !gridNodes[index])
		{
			st = linkIndex[iz*imageRes.x+ix];
			num = linkIndex[iz*imageRes.x+ix+1]-st;

			for(i=0; i < num; i++)
			{
				id = atomicMin(&linkLayerC[st+i],iy);
				//if (ix == 27 && iz == 48)
				//	printf("?? %d %d %d %d \n", ix, iz, iy, id);
			}
			
		}

		index += blockDim.x * gridDim.x;

	}
}

__global__ void krSLAContouring_FindAllLinks(unsigned int *linkIndex, unsigned int *linkLayerC, short *linkLayerD, short2 *linkID, bool *tempImg, unsigned int *count, int nodeNum, int3 imageRes)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iy, iz, id, st;
	bool node;

	while (index<nodeNum) {

		ix = index%imageRes.x;	iy = (index/imageRes.x)%imageRes.y;
		iz = index/(imageRes.x*imageRes.y);

	
		if (tempImg[index])
		{
			id = atomicAdd(&count[iz*imageRes.x+ix],1);
			st = linkIndex[iz*imageRes.x+ix];

			linkLayerC[st+id] = iy;
			linkLayerD[st+id] = iy;
			linkID[st+id] = make_short2(ix, iz);

			//printf("find all %d %d %d %d %d\n", index, ix, iz, iy, st+id);
		}


		index += blockDim.x * gridDim.x;

	}
}

__global__ void krSLAContouring_Filter5(bool *gridNodes, bool *tempImg, bool *suptNodes, unsigned int *linkIndex, int nodeNum, int3 imageRes, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz, id, s_id, prevs_id;
	bool node, node2, node3;

	while (index<nodeNum) {

		ix = index%imageRes.x;	iz = index/imageRes.x;

		id = iz*imageRes.x*imageRes.y+iy*imageRes.x+ix;
		s_id = iz*imageRes.x*(imageRes.y-1)+iy*imageRes.x+ix;
		prevs_id = iz*imageRes.x*(imageRes.y-1)+(iy+1)*imageRes.x+ix;

		node = tempImg[index];
		if (node)	
		{
			atomicAdd(&linkIndex[index], 1);
		}

		if (iy+1 >= imageRes.y-1) node3 = false;
		else node3 = suptNodes[prevs_id];

		node2 = (node3 && !gridNodes[id]);

		suptNodes[s_id] = node || node2;

		index += blockDim.x * gridDim.x;

	}
}


__global__ void krSLAContouring_Filter4(bool *outNodes, bool *inNodes, bool *imgNodes, bool *regionFinish, int nodeNum, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	

	while (index<nodeNum) {

		if (inNodes[index])	
		{
			if (imgNodes[index])	
			{
				regionFinish[0] = false;
			}
			else
			{
				outNodes[index] = false;
			}
		}
		

		index += blockDim.x * gridDim.x;

	}
}



__global__ void krSLAContouring_Filter3(bool *outNodes, bool *inNodes, int nodeNum, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	bool node;

	while (index<nodeNum) {
		
		node = outNodes[index];
		outNodes[index] = inNodes[index] && (!node);

		index += blockDim.x * gridDim.x;

	}
}

__global__ void krSLAContouring_Filter2(bool *assistImg, bool *tempImg, bool *targetImg, int *count, int nodeNum, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz;
	bool node;
	int idx, idz;

	while (index<nodeNum) {
		//ix = index%213;	iz = index/213;
		
		if (targetImg[index])
		{
			assistImg[index] = true;
			//if (!tempImg[index]) 
			//if (tempImg[index]) printf("???? %d %d %d \n",ix, iy, iz);
			tempImg[index] = true;
			atomicAdd(count, 1);

			//if (iy == 104) printf("%d \n", index);

		}
		else
		{
			assistImg[index] = false;
		}
		
		index += blockDim.x * gridDim.x;

	}
}

__global__ void krSLAContouring_Filter1(bool *assistImg, bool *tempImg, bool *targetImg, int nodeNum, int2 suptRes, int suptRadius, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz;
	bool node;
	int idx, idz;

	while (index<nodeNum) {

		ix = index%suptRes.x;	iz = index/suptRes.x;

		idx = ix*suptRadius;	idz = iz*suptRadius;
		if (targetImg[idz*suptRes.x+idx])
		{
			assistImg[idz*suptRes.x+idx] = true;
			tempImg[idz*suptRes.x+idx] = true;
		}
		

		index += blockDim.x * gridDim.x;

	}

}


__global__ void krSLAContouring_Initialization(bool *tempImg, bool *targetImg, bool *gridNodes, int nodeNum, int3 imageRes, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz;
	bool node;

	while (index<nodeNum) {
		ix = index%imageRes.x;	//iy = (index/imageRes.x)%imageRes.y+1;
		iz = index/imageRes.x;
		//iz = index/(imageRes.x*imageRes.y);

		node = gridNodes[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix];
		
		tempImg[index] = node;
		targetImg[index] = gridNodes[iz*imageRes.x*imageRes.y+(iy+1)*imageRes.x+ix] && (!node);


		//if ((gridNodes[iz*imageRes.x*imageRes.y+(iy+1)*imageRes.x+ix] && (!node)) && iy == 134)
		//if (iy == 0 && targetImg[index])
		//	printf(" %d %d %d \n", ix, iy, iz);

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krFDMContouring_VerticalSpptPxlProp(bool *gridNodes, bool *suptNodes, int nodeNum, int3 imageRes, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz, id, prev_id, suptid;
	bool node;

	while (index<nodeNum) {	
		ix = index%imageRes.x;	
		iz = index/imageRes.x;

		id = iz*imageRes.x*imageRes.y+iy*imageRes.x+ix;
		suptid = iz*imageRes.x*(imageRes.y-1)+iy*imageRes.x+ix;
		prev_id = iz*imageRes.x*(imageRes.y-1)+(iy+1)*imageRes.x+ix; // Mark: could be error when iy+1 >= imageRes.y-1

		if (suptNodes[prev_id] && !suptNodes[suptid] && !gridNodes[id])
			suptNodes[suptid] = true;

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krFDMContouring_kernelColor(short2 *output, int size) 
{
	__shared__ short2 s_last1[BLOCKSIZE], s_last2[BLOCKSIZE]; 
	__shared__ int s_lasty[BLOCKSIZE]; 

	int col = threadIdx.x; 
	int tid = threadIdx.y; 
	int tx = __mul24(blockIdx.x, blockDim.x) + col; 
	int dx, dy, lasty; 
	unsigned int best, dist; 
	short2 last1, last2; 
	int count = 0;
	if (tid == blockDim.y - 1) {
		lasty = size - 1; 

		
		last2 = tex1Dfetch(disTexColor, __mul24(lasty, size) + tx); 
		
		if (last2.x == MARKER) {
			lasty = last2.y; 
			last2 = tex1Dfetch(disTexColor, __mul24(lasty, size) + tx); 
		}

		if (last2.y >= 0) 
			last1 = tex1Dfetch(disTexColor, __mul24(last2.y, size) + tx); 

		s_last1[col] = last1; s_last2[col] = last2; s_lasty[col] = lasty; 
	}

	__syncthreads(); 
 
	count = 0;
	for (int ty = size - 1 - tid; ty >= 0; ty -= blockDim.y) {
		last1 = s_last1[col]; last2 = s_last2[col]; lasty = s_lasty[col]; 

		
		dx = last2.x - tx; dy = lasty - ty; 
		best = dist = __mul24(dx, dx) + __mul24(dy, dy); 

		while (last2.y >= 0) {
			dx = last1.x - tx; dy = last2.y - ty; 
			dist = __mul24(dx, dx) + __mul24(dy, dy); 

			if (dist > best) 
				break; 
			count++;
			best = dist; lasty = last2.y; last2 = last1;

			if (last2.y >= 0) 
				last1 = tex1Dfetch(disTexColor, __mul24(last2.y, size) + tx); 
		}

		__syncthreads(); 

		
		output[TOID(tx, ty, size)] = make_short2(last2.x, lasty);

		if (tid == blockDim.y - 1) {
			s_last1[col] = last1; s_last2[col] = last2; s_lasty[col] = lasty; 
		}

		__syncthreads(); 
	}
}

__global__ void krFDMContouring_kernelDoubleToSingleList(short2 *output, int size)
{
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; 
	int ty = blockIdx.y; 
	int id = TOID(tx, ty, size); 

	output[id] = make_short2(tex1Dfetch(disTexColor, id).x, tex1Dfetch(disTexLinks, id).y); 

}


__global__ void krFDMContouring_kernelMergeBands(short2 *output, int size, int bandSize)
{
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; 
	int band1 = blockIdx.y * 2; 
	int band2 = band1 + 1; 
	int firsty, lasty; 
	short2 last1, last2, current; 
	// last1 and last2: x component store the x coordinate of the site, 
	// y component store the backward pointer
	// current: y component store the x coordinate of the site, 
	// x component store the forward pointer

	// Get the two last items of the first list
	lasty = __mul24(band2, bandSize) - 1; 


	last2 = make_short2(tex1Dfetch(disTexColor, TOID(tx, lasty, size)).x, 
		tex1Dfetch(disTexLinks, TOID(tx, lasty, size)).y); 

	

	if (last2.x == MARKER) {
		lasty = last2.y; 

		if (lasty >= 0) 
			last2 = make_short2(tex1Dfetch(disTexColor, TOID(tx, lasty, size)).x, 
			tex1Dfetch(disTexLinks, TOID(tx, lasty, size)).y); 
		else
			last2 = make_short2(MARKER, MARKER); 
	}

	if (last2.y >= 0) {
		// Second item at the top of the stack
		last1 = make_short2(tex1Dfetch(disTexColor, TOID(tx, last2.y, size)).x, 
			tex1Dfetch(disTexLinks, TOID(tx, last2.y, size)).y); 
	}

	// Get the first item of the second band
	firsty = __mul24(band2, bandSize); 
	current = make_short2(tex1Dfetch(disTexLinks, TOID(tx, firsty, size)).x, 
		tex1Dfetch(disTexColor, TOID(tx, firsty, size)).x); 

	if (current.y == MARKER) {
		firsty = current.x; 

		if (firsty >= 0) 
			current = make_short2(tex1Dfetch(disTexLinks, TOID(tx, firsty, size)).x, 
			tex1Dfetch(disTexColor, TOID(tx, firsty, size)).x); 
		else
			current = make_short2(MARKER, MARKER); 
	}

	float i1, i2;     

	// Count the number of item in the second band that survive so far. 
	// Once it reaches 2, we can stop. 
	int top = 0; 

	while (top < 2 && current.y >= 0) {
		// While there's still something on the left
		while (last2.y >= 0) {
			i1 = interpoint(last1.x, last2.y, last2.x, lasty, tx); 
			i2 = interpoint(last2.x, lasty, current.y, firsty, tx);

			if (i1 < i2) 
				break; 

			lasty = last2.y; last2 = last1; 
			top--; 

			if (last1.y >= 0) 
				last1 = make_short2(tex1Dfetch(disTexColor, TOID(tx, last1.y, size)).x, 
				output[TOID(tx, last1.y, size)].y); 
		}

		// Update the current pointer 
		
		output[TOID(tx, firsty, size)] = make_short2(current.x, lasty); 

		if (lasty >= 0) 
		{
			
			output[TOID(tx, lasty, size)] = make_short2(firsty, last2.y); 
		}

		last1 = last2; last2 = make_short2(current.y, lasty); lasty = firsty; 
		firsty = current.x; 

		top = max(1, top + 1); 

		// Advance the current pointer to the next one
		if (firsty >= 0) 
			current = make_short2(tex1Dfetch(disTexLinks, TOID(tx, firsty, size)).x, 
			tex1Dfetch(disTexColor, TOID(tx, firsty, size)).x); 
		else
			current = make_short2(MARKER, MARKER); 
	}

	// Update the head and tail pointer. 
	firsty = __mul24(band1, bandSize); 
	lasty = __mul24(band2, bandSize); 
	current = tex1Dfetch(disTexLinks, TOID(tx, firsty, size)); 

	if (current.y == MARKER && current.x < 0) {	// No head?
		last1 = tex1Dfetch(disTexLinks, TOID(tx, lasty, size)); 

		if (last1.y == MARKER)
			current.x = last1.x; 
		else
			current.x = lasty; 

		
		output[TOID(tx, firsty, size)] = current; 
	}

	firsty = __mul24(band1, bandSize) + bandSize - 1; 
	lasty = __mul24(band2, bandSize) + bandSize - 1; 
	current = tex1Dfetch(disTexLinks, TOID(tx, lasty, size)); 

	if (current.x == MARKER && current.y < 0) {	// No tail?
		last1 = tex1Dfetch(disTexLinks, TOID(tx, firsty, size)); 

		if (last1.x == MARKER) 
			current.y = last1.y; 
		else
			current.y = firsty; 

		
		output[TOID(tx, lasty, size)] = current; 
	}
}


__global__ void krFDMContouring_kernelCreateForwardPointers(short2 *output, int size, int bandSize) 
{
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; 
	int ty = __mul24(blockIdx.y+1, bandSize) - 1; 
	int id = TOID(tx, ty, size); 
	int lasty = -1, nexty; 
	short2 current; 

	// Get the tail pointer

	current = tex1Dfetch(disTexLinks, id); 

	if (current.x == MARKER)
		nexty = current.y; 
	else
		nexty = ty; 

	for (int i = 0; i < bandSize; i++, id -= size)
	{
		
		if (ty - i == nexty) {
			current = make_short2(lasty, tex1Dfetch(disTexLinks, id).y);

			
			output[id] = current; 

			lasty = nexty; 
			nexty = current.y; 
		}
	}

		// Store the pointer to the head at the first pixel of this band
	if (lasty != ty - bandSize + 1) 
			output[id + size] = make_short2(lasty, MARKER);  
}


__global__ void krFDMContouring_kernelProximatePoints(short2 *stack, int size, int bandSize) 
{
	int tx = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; 
	int ty = __mul24(blockIdx.y, bandSize); 
	int id = TOID(tx, ty, size); 
	int lasty = -1; 
	short2 last1, last2, current; 
	float i1, i2;     
	

	last1.y = -1; last2.y = -1; 

	for (int i = 0; i < bandSize; i++, id += size) {
		current = tex1Dfetch(disTexColor, id);


		if (current.x != MARKER) {
			
			while (last2.y >= 0) {
			
				i1 = interpoint(last1.x, last2.y, last2.x, lasty, tx); 
				i2 = interpoint(last2.x, lasty, current.x, current.y, tx); 

				

				if (i1 < i2) 
					break;

				lasty = last2.y; last2 = last1; 

				if (last1.y >= 0)
					last1 = stack[TOID(tx, last1.y, size)]; 
			}

			last1 = last2; last2 = make_short2(current.x, lasty); lasty = current.y; 

			stack[id] = last2;
		}
	}

	// Store the pointer to the tail at the last pixel of this band
	if (lasty != ty + bandSize - 1) 
		stack[TOID(tx, ty + bandSize - 1, size)] = make_short2(MARKER, lasty); 
}

__global__ void krFDMContouring_kernelTranspose(short2 *data, int size)
{
	__shared__ short2 block1[TILE_DIM][TILE_DIM + 1];
	__shared__ short2 block2[TILE_DIM][TILE_DIM + 1];

	int blockIdx_y = blockIdx.x;
	int blockIdx_x = blockIdx.x+blockIdx.y;

	if (blockIdx_x >= gridDim.x)
		return ; 

	int blkX, blkY, x, y, id1, id2; 
	short2 pixel; 

	blkX = __mul24(blockIdx_x, TILE_DIM); 
	blkY = __mul24(blockIdx_y, TILE_DIM); 

	x = blkX + threadIdx.x;
	y = blkY + threadIdx.y;
	id1 = __mul24(y, size) + x;

	x = blkY + threadIdx.x;
	y = blkX + threadIdx.y;
	id2 = __mul24(y, size) + x;

	// read the matrix tile into shared memory
	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		block1[threadIdx.y + i][threadIdx.x] = tex1Dfetch(disTexColor, id1 + __mul24(i, size));
		block2[threadIdx.y + i][threadIdx.x] = tex1Dfetch(disTexColor, id2 + __mul24(i, size));

		
	}

	__syncthreads();

	// write the transposed matrix tile to global memory
	for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
		pixel = block1[threadIdx.x][threadIdx.y + i];
		data[id2 + __mul24(i, size)] = make_short2(pixel.y, pixel.x); 

		
		pixel = block2[threadIdx.x][threadIdx.y + i];
		data[id1 + __mul24(i, size)] = make_short2(pixel.y, pixel.x); 

	}
}

__global__ void krFDMContouring_kernelUpdateVertical(short2 *output, int size, int band, int bandSize) 
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 
	int ty = blockIdx.y * bandSize; 

	short2 top = tex1Dfetch(disTexLinks, TOID(tx, ty, size)); 
	short2 bottom = tex1Dfetch(disTexLinks, TOID(tx, ty + bandSize - 1, size)); 
	short2 pixel; 

	int dist, myDist; 

	int id = TOID(tx, ty, size); 

	for (int i = 0; i < bandSize; i++, id += size) {
		pixel = tex1Dfetch(disTexColor, id); 


	
		myDist = abs(pixel.y - (ty + i)); 

		dist = abs(top.y - (ty + i)); 
		if (dist < myDist) { myDist = dist; pixel = top; }


		dist = abs(bottom.y - (ty + i)); 
		if (dist < myDist) pixel = bottom; 

		output[id] = pixel; 
	}
}

__global__ void krFDMContouring_kernelPropagateInterband(short2 *output, int size, int bandSize) 
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 
	int inc = __mul24(bandSize, size); 
	int ny, nid, nDist; 
	short2 pixel; 

	// Top row, look backward
	int ty = __mul24(blockIdx.y, bandSize); 
	int topId = TOID(tx, ty, size); 
	int bottomId = TOID(tx, ty + bandSize - 1, size); 

	//if (topId >= imageSize) 
	//	pixel = make_short2(MARKER, MARKER);
	//else 
	pixel = tex1Dfetch(disTexColor, topId); 

	int myDist = abs(pixel.y - ty); 

	for (nid = bottomId - inc; nid >= 0; nid -= inc) {
		//if (nid >= imageSize) continue;
		pixel = tex1Dfetch(disTexColor, nid); 

		
		
		if (pixel.x != MARKER) { 
			nDist = abs(pixel.y - ty); 

			if (nDist < myDist) 
				output[topId] = pixel; 
			
			//if (topId == 4095) printf("?? %d %d \n", pixel.x, pixel.y);
			break;	
		}
	}

	// Last row, look downward
	ty = ty + bandSize - 1; 

	//if (bottomId >= imageSize) return;

	pixel = tex1Dfetch(disTexColor, bottomId); 
	myDist = abs(pixel.y - ty); 

	for (ny = ty + 1, nid = topId + inc; ny < size; ny += bandSize, nid += inc) {
		//if (nid >= imageSize) break;
		pixel = tex1Dfetch(disTexColor, nid); 

		//if (nid == 4351)
		//	printf("bbb %d %d\n", pixel.x, pixel.y);

	
		if (pixel.x != MARKER) { 
			nDist = abs(pixel.y - ty); 

			if (nDist < myDist) 
				output[bottomId] = pixel; 
			//if (bottomId == 4095) printf("!? %d %d %d\n", pixel.x, pixel.y, nid);
			break; 
		}
	}
}


__global__ void krFDMContouring_kernelFloodUp(short2 *output, int size, int bandSize) 
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 
	int ty = (blockIdx.y+1) * bandSize - 1; 
	int id = TOID(tx, ty, size); 

	short2 pixel1, pixel2; 
	int dist1, dist2; 

	pixel1 = make_short2(MARKER, MARKER); 

	for (int i = 0; i < bandSize; i++, id -= size) {
		//if (id >= imageSize) continue;
		//if (TOID(tx, ty, size) == 8191)
		//	printf("111 %d %d \n", pixel1.x, pixel1.y);

		dist1 = abs(pixel1.y - ty + i); 

		//if (TOID(tx, ty, size) == 8191)
		//	printf("222 %d \n", dist1);

		pixel2 = tex1Dfetch(disTexColor, id); 
		dist2 = abs(pixel2.y - ty + i); 

		if (dist2 < dist1) 
			pixel1 = pixel2; 

		
		//if (TOID(tx, ty, size) == 8191)
		//		printf("aaa %d %d %d %d %d %d %d %d %d\n", pixel2.x, pixel2.y, dist1, dist2, pixel1.x, pixel1.y, ty, i, id);


		output[id] = pixel1; 
	}
}

__global__ void krFDMContouring_kernelFloodDown(short2 *output, int size, int bandSize) 
{
	int tx = blockIdx.x * blockDim.x + threadIdx.x; 
	int ty = blockIdx.y * bandSize; 
	int id = TOID(tx, ty, size); 

	short2 pixel1, pixel2; 

	pixel1 = make_short2(MARKER, MARKER); 

	for (int i = 0; i < bandSize; i++, id += size) {
		
		//if (id >= imageSize) break;
		pixel2 = tex1Dfetch(disTexColor, id); 

	
		

		if (pixel2.x != MARKER) 
			pixel1 = pixel2; 

		//if (TOID(tx, ty, size) == 4351)
		//if (id == 6911)
		//	printf("flood down %d %d %d %d\n", pixel1.x, pixel1.y,  pixel2.x, pixel2.y);

		output[id] = pixel1; 

		
	}
}

__global__ void krFDMContouring_InitializedValue(short2 *dMap, int nodeNum, int value)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;  

	while (index<nodeNum) {

		dMap[index] = make_short2(value, value);

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krFDMContouring_InitializedDistanceMap(bool *gridNodes, short2 *dMap, int nodeNum, int realNum, int3 imageRes, int iy, int texwidth)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix,iz, id;
	while (index<nodeNum) {

		//if (index < realNum)
		//{

			ix = index%imageRes.x;	iz = index/imageRes.x;
			id = iz*texwidth + ix;
			//if (index == 6911)
			//	printf("ix iz %d %d %d\n", ix, iz, id);
			if (gridNodes[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix])
			{
			
				dMap[id].x = ix;
				dMap[id].y = iz;
				//dMap[index].x = MARKER;
				//dMap[index].y = MARKER;
				//if (iy == 137)
				//	printf("ix iy iz : %d %d %d\n", ix, iy, iz);
				
			}
			/*else
			{
				dMap[index].x = MARKER;
				dMap[index].y = MARKER;

			}*/
		/*}
		else
		{
			dMap[index].x = MARKER;
			dMap[index].y = MARKER;
		}*/


		index += blockDim.x * gridDim.x;
	}
}

__global__ void krFDMContouring_Erosion(bool *gridNodes, bool* output, int nodeNum, int3 imageRes, double realThreshold, int gridRadius)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz;
	int m, n;
	bool bflag;

	while (index<nodeNum) {
		ix = index%imageRes.x;		iz = index/imageRes.x;

		if (gridNodes[iz*imageRes.x+ix])
		{
			bflag = true;
			for(m=-gridRadius; m<=gridRadius; m++)
			{
				for(n=-gridRadius; n<=gridRadius; n++)
				{
					if(ix+m<0 || ix+m>=imageRes.x || iz+n<0 || iz+n>=imageRes.z || (m==0 && n==0))
						continue;
					else if(m*m+n*n > realThreshold*realThreshold)
						continue;

					if(!gridNodes[(iz+n)*imageRes.x + (ix+m)])
					{
						output[index] = false;
						bflag = false;
						break;
					}					
				}
				if(!bflag)
					break;
			}
		}
		index += blockDim.x * gridDim.x;

	}
}

__global__ void krFDMContouring_Dilation(bool *seedNodes, bool* output, int nodeNum, int3 imageRes, double realThreshold, int gridRadius, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz;
	int m, n;

	while (index<nodeNum) {
		ix = index%imageRes.x;		iz = index/imageRes.x;

		if (seedNodes[index])
		{
			//if (iy == 87) printf("--%d %d %d %d\n", iy, index, ix, iz);
			for(m=-gridRadius; m<=gridRadius; m++)
			{
				for(n=-gridRadius; n<=gridRadius; n++)
				{
					if(ix+m<0 || ix+m>=imageRes.x || iz+n<0 || iz+n>=imageRes.z || (m==0 && n==0))
						continue;
					else if(m*m+n*n > realThreshold*realThreshold)
						continue;

					output[(iz+n)*imageRes.x+(ix+m)] = true;
					//if (iy == 137 && ix == 74 && iz == 36) printf("%d %d %d\n", index, ix, iz);
				}
			}
		}

		index += blockDim.x * gridDim.x;

	}
}

__global__ void krFDMContouring_Filter1(bool *gridNodes, bool *outNodes, int nodeNum, int3 imageRes, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz;
	bool node;

	while (index<nodeNum) {
		ix = index%imageRes.x;	iz = index/imageRes.x;

		node = outNodes[index];
		//outNodes[index] = node && (!gridNodes[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix]);
		outNodes[index] = node && (!gridNodes[index]);

		//if ((node && (!gridNodes[index])) && iy == 125)
		//	printf("%d %d %d \n", ix, iy, iz);

		index += blockDim.x * gridDim.x;

	}

}

__global__ void krFDMContouring_Test3D(bool* outNodes, int nodeNum, int3 imageRes)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iy, iz;


	while (index<nodeNum) {

		ix = index%imageRes.x;	iy = (index/imageRes.x)%imageRes.y;
		iz = index/(imageRes.x*imageRes.y);

		if (outNodes[index] && iy == 0)
			printf("error ! %d %d %d \n", ix, iy, iz);
		index += blockDim.x * gridDim.x;
	}

}

__global__ void krFDMContouring_Test2D(bool* outNodes, int nodeNum, int2 imageRes)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iy, iz;


	while (index<nodeNum) {

		ix = index%imageRes.x;	//iy = (index/imageRes.x)%imageRes.y;
		iy = index/imageRes.x;
			
		//iz = index/(imageRes.x*imageRes.y);

		if (outNodes[index])
			printf("error ! %d %d %d \n", index, ix, iy);
		index += blockDim.x * gridDim.x;
	}

}

//krFDMContouring_Filter3<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(tempNodes, solidRegion, imageSize[0]*imageSize[2]);
__global__ void krFDMContouring_Filter3(bool* outNodes, bool *seedNodes, int nodeNum)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz;
	bool node;

	while (index<nodeNum) {
		
		
		node = outNodes[index];
		if (node)
		{
			seedNodes[index] = true;
			node = true;
		}
		else
		{
			node = seedNodes[index];
		}

		outNodes[index] = node;

		index += blockDim.x * gridDim.x;

	}

}
//krFDMContouring_Filter4<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(solidNodes, suptNodes, suptTemp, nodeNum, suptimgRes, i);
__global__ void krFDMContouring_Filter4(bool* outNodes, bool *inNodesA, bool *inNodesB, int nodeNum, int3 imageRes, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz;
	bool node;

	while (index<nodeNum) {
		ix = index%imageRes.x;	iz = index/imageRes.x;

		node = inNodesA[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix] && (!inNodesB[index]);
		outNodes[index] = node;
		inNodesA[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix] = inNodesB[index];

		index += blockDim.x * gridDim.x;
	}

}

__global__ void krFDMContouring_Filter5(bool* outNodes, bool *inNodesA, bool *gridNodes, int nodeNum, int iy, int3 imageRes)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz;
	bool node;

	while (index<nodeNum) {

		ix = index%imageRes.x;	iz = index/imageRes.x;

		node = outNodes[iz*imageRes.x*(imageRes.y-1)+iy*imageRes.x+ix];
		outNodes[iz*imageRes.x*(imageRes.y-1)+iy*imageRes.x+ix] = node && (!inNodesA[index]) && (!gridNodes[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix]);
		
		index += blockDim.x * gridDim.x;
	}

}

__global__ void krFDMContouring_integrateImageintoGrossImage(bool* outNodes, bool *inNodes, int nodeNum, int3 imageRes, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz;
	bool node;

	while (index<nodeNum) {

		ix = index%imageRes.x;	iz = index/imageRes.x;
		node = outNodes[index];

		if (inNodes[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix] && !node)
			outNodes[index] = true;

		index += blockDim.x * gridDim.x;
	}

	
}



__global__ void krFDMContouring_Filter2(short2 *disMap, bool *outNodes, bool *suptNodes, int nodeNum, int3 imageRes, bool *RegionFlag, double t, int disTexSize, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz;
	bool node;
	short2 d;
	double dist;

	while (index<nodeNum) {
		ix = index%imageRes.x;	iz = index/imageRes.x;

		node = outNodes[index];
		if (node)
		{
			//d = disMap[index];
			d = disMap[iz*disTexSize+ix];
			d.x = ix - d.x;
			d.y = iz - d.y;
			dist = d.x*d.x + d.y*d.y;

			//if (iy == 137 && ix == 73) printf("%d %d %d %d %f\n", ix, iz, d.x, d.y, dist);

			if(suptNodes[index] && dist <= t)
			{
				RegionFlag[0] = false;
			}
			else
				outNodes[index] = false;

		}

		index += blockDim.x * gridDim.x;

	}

}

__global__ void krFDMContouring_CopyNodesrom3Dto2D(bool *m_2DNodes, bool *m_3DNodes, int nodeNum, int3 imageRes, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz;

	while (index<nodeNum) {
		ix = index%imageRes.x;	
		iz = index/imageRes.x;

		m_2DNodes[index] = m_3DNodes[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix];

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krFDMContouring_CopyNodesrom2Dto3D(bool *m_2DNodes, bool *m_3DNodes, int nodeNum, int3 imageRes, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz;

	while (index<nodeNum) {
		ix = index%imageRes.x;	
		iz = index/imageRes.x;

		m_3DNodes[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix] = m_2DNodes[index];

		//if (iy == 0 && m_2DNodes[index])
		//	printf("copy %d %d %d %d %d %d %d \n", ix, iy, iz, iz*imageRes.x*imageRes.y+iy*imageRes.x+ix, index, imageRes.x, imageRes.y);
		/*if (m_3DNodes[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix])
				printf("warning!!! %d %d %d \n", ix, iy, iz);
		if (m_2DNodes[index])
			printf("2warning!!! %d %d %d \n", ix, iy, iz);*/
		/*if (m_3DNodes[iz*imageRes.x*imageRes.y+27*imageRes.x+ix]) printf("warning!!! %d %d %d \n", ix, iy, iz);
		if (m_3DNodes[iz*imageRes.x*imageRes.y+26*imageRes.x+ix]) printf("warning!!! %d %d %d \n", ix, iy, iz);
		if (m_3DNodes[iz*imageRes.x*imageRes.y+29*imageRes.x+ix]) printf("warning!!! %d %d %d \n", ix, iy, iz);
		if (m_3DNodes[iz*imageRes.x*imageRes.y+30*imageRes.x+ix]) printf("warning!!! %d %d %d \n", ix, iy, iz);*/

		index += blockDim.x * gridDim.x;
	}
}


__global__ void krFDMContouring_SubtractSolidRegion(bool *gridNodes, bool *suptNodes, bool *seedNodes, int nodeNum, int3 imageRes, int iy)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iz;

	while (index<nodeNum) {
		ix = index%imageRes.x;	//iy = (index/imageRes.x)%imageRes.y+1;
		iz = index/imageRes.x;
		//iz = index/(imageRes.x*imageRes.y);
		
		if (gridNodes[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix])
		{
			suptNodes[iz*imageRes.x*(imageRes.y-1)+iy*imageRes.x+ix] = false;
			seedNodes[index] = true;
		}
		else if (gridNodes[iz*imageRes.x*imageRes.y+(iy+1)*imageRes.x+ix])
		{
			suptNodes[iz*imageRes.x*(imageRes.y-1)+iy*imageRes.x+ix] = true;
		}
		else
		{
			suptNodes[iz*imageRes.x*(imageRes.y-1)+iy*imageRes.x+ix] = false;
		}

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krFDMContouring_ConstrainedSmoothing(float2 *newStart, float2 *newEnd, float2 *stickStart, float2 *stickEnd, 
													 int *stickID, int stickNum, int2 *stickDir,float *accMovement, double moveCoef, float imgWidth, bool *bIterFlag, 
													 double2 imgOri)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	float2 st_pt, ed_pt, prev_st, pt1, pt2;
	unsigned int prev_id, i, j;
	double pt[2];
	float acc;
	bool intersect, bVert;
	int2 st_dir;

	while (index<stickNum) {	

		st_pt = stickStart[index];
		ed_pt = stickEnd[index];
		prev_id = stickID[index];
		
		if (bIterFlag[index])
		{
				
				prev_st = stickStart[prev_id];

				st_dir = stickDir[index];
		
				pt[0] = 0.0;
				pt[1] = 0.0;
				acc = 0.0;
				


				if (st_dir.x < 0) // means the starting point is in the middle between two points with same z value
				{
					if (ed_pt.y > st_pt.y)
					{	
						pt1.x = imgOri.x + imgWidth*(-(st_dir.x)+1);
						pt1.y = imgOri.y + imgWidth*st_dir.y;
						pt2.x = imgOri.x + imgWidth*(-(st_dir.x));
						pt2.y = imgOri.y + imgWidth*st_dir.y;
					}
					else
					{
						pt1.x = imgOri.x + imgWidth*(-(st_dir.x));
						pt1.y = imgOri.y + imgWidth*st_dir.y;
						pt2.x = imgOri.x + imgWidth*(-(st_dir.x)+1);
						pt2.y = imgOri.y + imgWidth*st_dir.y;
					}
					bVert = false;
				}
				else
				{
					if (ed_pt.x > st_pt.x)
					{
						pt1.x = imgOri.x + imgWidth*st_dir.x;
						pt1.y = imgOri.y + imgWidth*(st_dir.y);
						pt2.x = imgOri.x + imgWidth*st_dir.x;
						pt2.y = imgOri.y + imgWidth*(st_dir.y+1);
					}
					else
					{
						pt1.x = imgOri.x + imgWidth*st_dir.x;
						pt1.y = imgOri.y + imgWidth*(st_dir.y+1);
						pt2.x = imgOri.x + imgWidth*st_dir.x;
						pt2.y = imgOri.y + imgWidth*(st_dir.y);
					}

					bVert = true;
				}
				i = (int)((st_pt.x - imgOri.x)/imgWidth+0.001);
				/*if (((st_pt.x - imgOri.x)/imgWidth) - i > 0.3) //0.5 is mid point
				{
					if (ed_pt.y > st_pt.y)
					{	
						pt1.x = imgOri.x + imgWidth*(i+1);
						pt1.y = st_pt.y;
						pt2.x = imgOri.x + imgWidth*i;
						pt2.y = st_pt.y;
					}
					else
					{
						pt1.x = imgOri.x + imgWidth*i;
						pt1.y = st_pt.y;
						pt2.x = imgOri.x + imgWidth*(i+1);
						pt2.y = st_pt.y;
					}

					

					bVert = false;
				}
				else
				{
					i = (int)((st_pt.y - imgOri.y)/imgWidth+0.001);

					if (ed_pt.x > st_pt.x)
					{
						pt1.x = st_pt.x;
						pt1.y = imgOri.y + imgWidth*i;
						pt2.x = st_pt.x;
						pt2.y = imgOri.y + imgWidth*(i+1);
					}
					else
					{
						pt1.x = st_pt.x;
						pt1.y = imgOri.y + imgWidth*(i+1);
						pt2.x = st_pt.x;
						pt2.y = imgOri.y + imgWidth*i;
					}
					

					bVert = true;
				}*/


		
				intersect = _calTwoLineSegmentsIntersection(prev_st, pt1, pt2, ed_pt, pt);
				if(!intersect)
				{
					if(bVert)
					{
						pt[0] = st_pt.x;
						pt[1] = ed_pt.y;
					}
					else
					{
						pt[0] = ed_pt.x;
						pt[1] = st_pt.y;
					}
					
				}
		

				ed_pt.x = st_pt.x + moveCoef*moveCoef*(pt[0]-st_pt.x);
				ed_pt.y = st_pt.y + moveCoef*moveCoef*(pt[1]-st_pt.y);


				newStart[index] = ed_pt;
				newEnd[prev_id] = ed_pt;


				//prev_st = oldSticks[index];
	
				prev_st.x =st_pt.x - ed_pt.x;
				prev_st.y = st_pt.y - ed_pt.y;
				//pt[0] = st_pt.x - ed_pt.x;
				//pt[1] = st_pt.y - ed_pt.y;

				acc = sqrt(prev_st.x*prev_st.x + prev_st.y*prev_st.y);
				//acc = sqrt(pt[0]*pt[0] + pt[1]*pt[1]);


				//if (index < 58 && st_pt.x > 0.765 && st_pt.y > 0.135)
				/*if (index == 37)
				{
					printf("%d start %f %f %d %d \n,", index, st_pt.x, st_pt.y, i, st_dir.x );
					printf(" new start %f %f \n,", ed_pt.x, ed_pt.y);
					printf("intersect point %f %f \n,", pt[0], pt[1]);
					printf("start point (grid) %f %f \n",  pt1.x, pt1.y);
					printf("end point (grid) %f %f \n",  pt2.x, pt2.y);
					printf("prev point  %f %f \n",  prev_st.x, prev_st.y);
				}*/
				
				//atomicAdd(&accMovement[index],acc);
				accMovement[index] = acc;
	
		}
		else
		{
			newStart[index] = st_pt;
			newEnd[prev_id] = ed_pt;
		}
		

		index += blockDim.x * gridDim.x;
	}

}

//__global__ void krFDMContouring_ConstrainedSmoothing(float2* newStart, float2* newEnd, float2 *stickStart, float2 *stickEnd,
//													int* stickID, int stickNum, float2 *oldStick, float *accMovement,
//													double moveCoef, float imgWidth, double2 imgOri)
//{
//		
//	int index=threadIdx.x+blockIdx.x*blockDim.x;
//	float2 st_pt, ed_pt, prev_st, pt1, pt2;
//	unsigned int prev_id, i, j;
//	double pt[2];
//	float acc;
//	bool intersect, bVert;
//
//	while (index<stickNum) {	
//
//		st_pt = stickStart[index];
//		ed_pt = stickEnd[index];
//
//		prev_id = stickID[index];
//		prev_st = stickStart[prev_id];
//		
//
//		pt[0] = 0.0;
//		pt[1] = 0.0;
//		acc = 0.0;
//
//		i = (int)((st_pt.x - imgOri.x)/imgWidth+0.001);
//		if (((st_pt.x - imgOri.x)/imgWidth) - i > 0.3) //0.5 is mid point
//		{
//			if (ed_pt.y > st_pt.y)
//			{	
//				pt1.x = imgOri.x + imgWidth*(i+1);
//				pt1.y = st_pt.y;
//				pt2.x = imgOri.x + imgWidth*i;
//				pt2.y = st_pt.y;
//			}
//			else
//			{
//				pt1.x = imgOri.x + imgWidth*i;
//				pt1.y = st_pt.y;
//				pt2.x = imgOri.x + imgWidth*(i+1);
//				pt2.y = st_pt.y;
//			}
//
//			
//
//			bVert = false;
//		}
//		else
//		{
//			i = (int)((st_pt.y - imgOri.y)/imgWidth+0.001);
//
//			if (ed_pt.x > st_pt.x)
//			{
//				pt1.x = st_pt.x;
//				pt1.y = imgOri.y + imgWidth*i;
//				pt2.x = st_pt.x;
//				pt2.y = imgOri.y + imgWidth*(i+1);
//			}
//			else
//			{
//				pt1.x = st_pt.x;
//				pt1.y = imgOri.y + imgWidth*(i+1);
//				pt2.x = st_pt.x;
//				pt2.y = imgOri.y + imgWidth*i;
//			}
//			
//
//			bVert = true;
//		}
//
//
//		
//		intersect = _calTwoLineSegmentsIntersection(prev_st, pt1, pt2, ed_pt, pt);
//		if(!intersect)
//		{
//			if(bVert)
//			{
//				//pt[0] = imgOri.x + imgWidth*st_pt.x;
//				//pt[1] = imgOri.y + imgWidth*ed_pt.y;
//				pt[0] = st_pt.x;
//				pt[1] = ed_pt.y;
//			}
//			else
//			{
//				//pt[0] = imgOri.x + imgWidth*ed_pt.x;
//				//pt[1] = imgOri.y + imgWidth*st_pt.y;
//				pt[0] = ed_pt.x;
//				pt[1] = st_pt.y;
//			}
//			
//		}
//		/*else
//		{
//			pt[0] = imgOri.x + imgWidth*pt[0];
//			pt[1] = imgOri.y + imgWidth*pt[1];
//		}
//	
//		st_pt.x = imgOri.x + imgWidth*st_pt.x;
//		st_pt.y = imgOri.x + imgWidth*st_pt.y;*/
//
//		ed_pt.x = st_pt.x + moveCoef*moveCoef*(pt[0]-st_pt.x);
//		ed_pt.y = st_pt.y + moveCoef*moveCoef*(pt[1]-st_pt.y);
//
//
//		newStart[index] = ed_pt;
//		newEnd[prev_id] = ed_pt;
//
//
//		prev_st = oldStick[index];
//		//st_pt = oldStick[index];
//		prev_st.x = prev_st.x - ed_pt.x;
//		prev_st.y = prev_st.y - ed_pt.y;
//		//st_pt.x = st_pt.x - ed_pt.x;
//		//st_pt.y = st_pt.y - ed_pt.y;
//
//		//acc = sqrt(st_pt.x*st_pt.x + st_pt.y*st_pt.y);
//		
//		acc = sqrt(prev_st.x*prev_st.x + prev_st.y*prev_st.y);
//		atomicAdd(accMovement,acc); 
//		index += blockDim.x * gridDim.x;
//	}
//
//}

__global__ void krFDMContouring_Count3DBit(unsigned int *d_output, unsigned int* bitresult, int cellNum, int3 texSize)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int count;
	unsigned int ix,iy;
	unsigned short i;

	while(index<cellNum) {

		ix = index%(texSize.x);	iy = (index/(texSize.x));
		
		count = 0;
		if (iy==0)
		{


		for(i=0; i < texSize.z; i++)
		{
			count+=bitCount(bitresult[iy*texSize.x*texSize.z+ ix*texSize.z + i]);
			printf("%d %d %d %d %d\n", ix,iy,i,iy*texSize.x*texSize.z+ ix*texSize.z + i, bitresult[iy*texSize.x*texSize.z+ ix*texSize.z + i]);
		}
				
		}
		

		atomicAdd(d_output,count);

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krFDMContouring_BuildHashTable(int* preStickIndex, int stickNum, unsigned int* sortedStickId, int* sortedPrevStickId)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;

	while (index<stickNum) {	
		
		preStickIndex[index] = sortedStickId[sortedPrevStickId[index]];
		
		index += blockDim.x * gridDim.x;
	}
}

__global__ void krFDMContouring_BuildHashTableForZ(bool *gridNodes, int3 imageRes, int cellNum, unsigned int* bitHashTable)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iy, iz, modZ, remainZ, imageResZMod;
	bool a_node, b_node;
	unsigned int bitResult;

	imageResZMod = (int)ceil((imageRes.z-1)/32.0);
	while (index<cellNum) {	
		ix = index%(imageRes.x-1);	iy = (index/(imageRes.x-1))%imageRes.y;
		iz = index/((imageRes.x-1)*imageRes.y);


		bitResult = 0;

		a_node = gridNodes[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix];
		b_node = gridNodes[(iz+1)*imageRes.x*imageRes.y+iy*imageRes.x+ix];

		if (a_node != b_node)
		{
			if (a_node)
			{
				modZ = (iz)/32;
				remainZ = (iz)-modZ*32;
				bitResult = SetBitPos(remainZ);

				atomicOr(&bitHashTable[iy*(imageRes.x-1)*imageResZMod+ ix*imageResZMod + modZ], bitResult);
				//bitHashTable[iy*(imageRes.x-1)*imageResZMod+ ix*imageResZMod + modZ] = bitResult;

			}
			else
			{
				modZ = (iz)/32;
				remainZ = (iz)-modZ*32;
				bitResult = SetBitPos(remainZ);

				atomicOr(&bitHashTable[iy*(imageRes.x-1)*imageResZMod+ (ix-1)*imageResZMod + modZ], bitResult);
			}
		}

		index += blockDim.x * gridDim.x;
	}
}


__global__ void krFDMContouring_CountAllStick(bool *gridNodes, int3 imageRes, int cellNum, unsigned int* count)
{	
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iy, iz, realx = imageRes.x+1;
	bool bnodes;

	while (index<cellNum) {		
		ix = index%imageRes.x;	iy = (index/imageRes.x)%imageRes.y;
		iz = index/(imageRes.x*imageRes.y);

		bnodes = gridNodes[iz*realx*imageRes.y+iy*realx+ix];
		if (bnodes != gridNodes[(iz+1)*realx*imageRes.y+iy*realx+ix])
		{
			atomicAdd(&count[iy],1);
			
		}				
		if (bnodes != gridNodes[iz*realx*imageRes.y+iy*realx+ix+1])
		{
			atomicAdd(&count[iy],1);
			
		}
		
		index += blockDim.x * gridDim.x;
	}
}

__global__ void krFDMContouring_FindAllStickInAxisZ(bool *gridNodes, unsigned int* stickIndex, unsigned int* counter, float2 *stickStart,
													float2 *stickEnd, int3 imageRes, int cellNum, double2 imgOri, float imgWidth, 
													int* stickID, int* prevStickId, int2 *stickDir)
{	
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iy, iz;
	bool a_node, b_node, c_node, d_node, e_node, f_node;
	float2 st_p, ed_p;
	unsigned int st, pos;
	int id, prev_id ;

	while (index<cellNum) {		
		ix = index%(imageRes.x-1);	iy = (index/(imageRes.x-1))%imageRes.y;
		iz = index/((imageRes.x-1)*imageRes.y);

		a_node = gridNodes[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix];
		b_node = gridNodes[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix+1];

	
		if (a_node!=b_node)
		{
			st_p.x = imgOri.x + imgWidth*(ix+0.5);
			st_p.y = imgOri.y + imgWidth*iz;

			c_node = gridNodes[(iz-1)*imageRes.x*imageRes.y+iy*imageRes.x+ix];
			d_node = gridNodes[(iz-1)*imageRes.x*imageRes.y+iy*imageRes.x+ix+1];

			e_node = gridNodes[(iz+1)*imageRes.x*imageRes.y+iy*imageRes.x+ix+1];
			f_node = gridNodes[(iz+1)*imageRes.x*imageRes.y+iy*imageRes.x+ix];
			//st_p.x = ix+0.5;
			//st_p.y = iz;

			if (a_node)
			{
				

				id = -(iz*imageRes.x + ix);
				

				if (d_node)
				{
					ed_p.x = imgOri.x + imgWidth*(ix+1);
					ed_p.y = imgOri.y + imgWidth*(iz-0.5);
					//ed_p.x = ix + 1;
					//ed_p.y = iz - 0.5;
				}
				else if (c_node)
				{
					ed_p.x = imgOri.x + imgWidth*(ix+0.5);
					ed_p.y = imgOri.y + imgWidth*(iz-1);

					//ed_p.x = ix + 0.5;
					//ed_p.y = iz - 1;
				}
				else
				{
					ed_p.x = imgOri.x + imgWidth*ix;
					ed_p.y = imgOri.y + imgWidth*(iz-0.5);
					//ed_p.x = ix;
					//ed_p.y = iz - 0.5;
				}

				if (e_node)
				{
					prev_id = iz*imageRes.x + (ix+1);
				}
				else if (f_node)
				{		
					prev_id = -((iz+1)*imageRes.x + ix);
				}
				else
				{
					prev_id = iz*imageRes.x + ix;
				}
			}
			else
			{
				
	
				id =  -(iz*imageRes.x + ix);
				
				
				if (f_node)
				{
					ed_p.x = imgOri.x + imgWidth*(ix);
					ed_p.y = imgOri.y + imgWidth*(iz+0.5);
					//ed_p.x = ix;
					//ed_p.y = iz+0.5;
				}
				else if (e_node)
				{
					ed_p.x = imgOri.x + imgWidth*(ix+0.5);
					ed_p.y = imgOri.y + imgWidth*(iz+1);
					//ed_p.x = ix+0.5;
					//ed_p.y = iz+1;
				}
				else
				{
					ed_p.x = imgOri.x + imgWidth*(ix+1);
					ed_p.y = imgOri.y + imgWidth*(iz+0.5);
					//ed_p.x = ix+1;
					//ed_p.y = iz+0.5;
				}

				if (c_node)
				{
					prev_id = (iz-1)*imageRes.x + ix;
				}
				else if (d_node)
				{
					prev_id = -((iz-1)*imageRes.x + ix);
				}
				else
				{
					prev_id = (iz-1)*imageRes.x + ix+1;
				}

			}

			/*if (ix == 15 && iy == 0 && iz == 5)
			{
				printf("a b c d e f %d %d %d %d %d %d %d %d", a_node, b_node, c_node, d_node, e_node, f_node,  id, prev_id);
			}*/


			pos = atomicAdd(&counter[iy],1);
			st = stickIndex[iy];
			stickStart[st+pos] = st_p;
			stickEnd[st+pos] = ed_p;
			stickID[st+pos] = id;
			prevStickId[st+pos] = prev_id;
			stickDir[st+pos] = make_int2(-ix,iz);
		}
		index += blockDim.x * gridDim.x;
	}
}

__global__ void krFDMContouring_FindAllStickInAxisX(bool *gridNodes, unsigned int* stickIndex, unsigned int* counter, float2 *stickStart, 
													float2 *stickEnd, int3 imageRes, int cellNum, double2 imgOri, float imgWidth, 
													int* stickID, int* prevStickId, int2 *stickDir)
{	
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	unsigned int ix, iy, iz;
	bool a_node, b_node, c_node, d_node, e_node, f_node;
	float2 st_p, ed_p;
	unsigned int st, pos;
	int id, prev_id;


	while (index<cellNum) {		
		ix = index%(imageRes.x-1);	iy = (index/(imageRes.x-1))%imageRes.y;
		iz = index/((imageRes.x-1)*imageRes.y);

		a_node = gridNodes[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix];
		b_node = gridNodes[(iz+1)*imageRes.x*imageRes.y+iy*imageRes.x+ix];
		

		if (a_node!=b_node)
		{
			st_p.x = imgOri.x + imgWidth*ix;
			st_p.y = imgOri.y + imgWidth*(iz+0.5);

			c_node = gridNodes[(iz+1)*imageRes.x*imageRes.y+iy*imageRes.x+ix+1];
			d_node = gridNodes[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix+1];

			e_node = gridNodes[(iz+1)*imageRes.x*imageRes.y+iy*imageRes.x+ix-1];
			f_node = gridNodes[iz*imageRes.x*imageRes.y+iy*imageRes.x+ix-1];


			if (a_node)
			{
				

				id		= iz*imageRes.x + ix;
				

				if (c_node)
				{
					ed_p.x = imgOri.x + imgWidth*(ix+0.5);
					ed_p.y = imgOri.y + imgWidth*(iz+1);

				}
				else if (d_node)
				{
					ed_p.x = imgOri.x + imgWidth*(ix+1);
					ed_p.y = imgOri.y + imgWidth*(iz+0.5);
					//ed_p.x = ix+1;
					//ed_p.y = iz+0.5;
				}
				else
				{
					ed_p.x = imgOri.x + imgWidth*(ix+0.5);
					ed_p.y = imgOri.y + imgWidth*(iz);
					//ed_p.x = ix+0.5;
					//ed_p.y = iz;
				}

				if (e_node)
				{
					prev_id = -((iz+1)*imageRes.x + ix-1);
				}
				else if (f_node)
				{
					prev_id = iz*imageRes.x + ix - 1;
				}
				else
				{
					prev_id = -(iz*imageRes.x + ix - 1);
				}
			}
			else
			{
				

				//id		= iz*imageRes.x + ix-1;
				id = iz*imageRes.x + ix;
				//prev_id = iz*imageRes.x + ix;

				if (f_node)
				{
					ed_p.x = imgOri.x + imgWidth*(ix-0.5);
					ed_p.y = imgOri.y + imgWidth*(iz);
					//ed_p.x = ix-0.5;
					//ed_p.y = iz;
				}
				else if (e_node)
				{
					ed_p.x = imgOri.x + imgWidth*(ix-1);
					ed_p.y = imgOri.y + imgWidth*(iz+0.5);
					//ed_p.x = (ix-1);
					//ed_p.y = (iz+0.5);
				}
				else
				{
					ed_p.x = imgOri.x + imgWidth*(ix-0.5);
					ed_p.y = imgOri.y + imgWidth*(iz+1);
					//ed_p.x = (ix-0.5);
					//ed_p.y = (iz+1);
				}

				if (d_node)
				{
					prev_id = -(iz*imageRes.x + ix);
				}
				else if (c_node)
				{
					prev_id = iz*imageRes.x + ix + 1;
				}
				else
				{
					prev_id = -((iz+1)*imageRes.x + ix);
				}
			}
			pos = atomicAdd(&counter[iy],1);
			st = stickIndex[iy];
			stickStart[st+pos] = st_p;
			stickEnd[st+pos] = ed_p;
			stickID[st+pos] = id;
			prevStickId[st+pos] = prev_id;
			stickDir[st+pos] = make_int2(ix, iz);
		}
		index += blockDim.x * gridDim.x;
	}
}

__global__ void krFDMContouring_BinarySampling(bool *gridNodes, double3 rotdir, float angle, int res, 
											   unsigned int *xIndexArray, float *xDepthArray,
											   unsigned int *yIndexArray, float *yDepthArray,
											   unsigned int *zIndexArray, float *zDepthArray,
											   float3 origin, double2 imgorigin, float gwidth, int3 imageRes, 
											   int nodeNum, float thickness, float imgWidth)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int num, st;
	unsigned int ix, iy, iz;
	double xx,yy,zz;
	double3 systemCoord, rot_p;
	float3 ori_p;
	unsigned short k;

	systemCoord.x = 0.0;	systemCoord.y = 0.0;	systemCoord.z = 0.0;

	while (index<nodeNum) {
		ix = index%imageRes.x;	iy = (index/imageRes.x)%imageRes.y;
		iz = index/(imageRes.x*imageRes.y);

		xx = imgorigin.x + imgWidth*ix;
		zz = imgorigin.y + imgWidth*iz;
		yy = (iy+1)*thickness;
		
		
		rot_p = _rotatePointAlongVector(xx, yy, zz, systemCoord.x, systemCoord.y, systemCoord.z, rotdir.x, rotdir.y, rotdir.z, angle);
		ori_p.x = origin.x-gwidth*0.5;	ori_p.y = origin.y-gwidth*0.5;	ori_p.z = origin.z-gwidth*0.5;
		

		gridNodes[index] = _detectInOutPoint((float)rot_p.x, (float)rot_p.y, (float)rot_p.z, xIndexArray, xDepthArray, 
												yIndexArray, yDepthArray, zIndexArray, zDepthArray,
												ori_p, origin, gwidth, res);


		index += blockDim.x * gridDim.x;
	}
}


__global__ void krFDMContouring_RotationBoundingBox(double *bndBoxX, double *bndBoxY, double *bndBoxZ, unsigned int* count, short nAxis, double3 rotdir, double angle, int res, unsigned int *IndexArray, float *DepthArray, float3 origin, float gwidth)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int arrsize = res*res;
	int num, st;
	unsigned int i, j, ind;
	double xx,yy,zz;
	double3 rot_st, rot_ed;
	double3 systemCoord;

	systemCoord.x = 0.0;	systemCoord.y = 0.0;	systemCoord.z = 0.0;

	while (index<arrsize) {
		j = index/res;	i = index%res;
		st = IndexArray[index];	num = IndexArray[index+1]-st;
		
		if (num > 0)
		{
			switch (nAxis)
			{
			case 0: xx = fabs(DepthArray[st])+origin.x;
					yy = gwidth*i+origin.y;
					zz = gwidth*j+origin.z;		
					rot_st = _rotatePointAlongVector(xx, yy, zz, systemCoord.x, systemCoord.y, systemCoord.z, rotdir.x, rotdir.y, rotdir.z, angle);
					xx = fabs(DepthArray[st+num-1])+origin.x;
					rot_ed = _rotatePointAlongVector(xx, yy, zz, systemCoord.x, systemCoord.y, systemCoord.z, rotdir.x, rotdir.y, rotdir.z, angle);
					break;
			case 1:	xx = gwidth*j+origin.x;	
					yy = fabs(DepthArray[st])+origin.y;
					zz = gwidth*i+origin.z;		
					rot_st = _rotatePointAlongVector(xx, yy, zz, systemCoord.x, systemCoord.y, systemCoord.z, rotdir.x, rotdir.y, rotdir.z, angle);
					yy = fabs(DepthArray[st+num-1])+origin.y;
					rot_ed = _rotatePointAlongVector(xx, yy, zz, systemCoord.x, systemCoord.y, systemCoord.z, rotdir.x, rotdir.y, rotdir.z, angle);
					break;
			case 2:	xx = gwidth*i+origin.x;	
					yy = gwidth*j+origin.y;	
					zz = fabs(DepthArray[st])+origin.z;	
					rot_st = _rotatePointAlongVector(xx, yy, zz, systemCoord.x, systemCoord.y, systemCoord.z, rotdir.x, rotdir.y, rotdir.z, angle);
					zz = fabs(DepthArray[st+num-1])+origin.z;
					rot_ed = _rotatePointAlongVector(xx, yy, zz, systemCoord.x, systemCoord.y, systemCoord.z, rotdir.x, rotdir.y, rotdir.z, angle);
					break;
			}


				ind = atomicAdd(count,2);

				bndBoxX[ind]	= rot_st.x;
				bndBoxX[ind+1]	= rot_ed.x;					
				bndBoxY[ind]	= rot_st.y;
				bndBoxY[ind+1]	= rot_ed.y;
				bndBoxZ[ind]	= rot_st.z;
				bndBoxZ[ind+1]	= rot_ed.z;
		
			
		}

		index += blockDim.x * gridDim.x;
	}
}


__global__ void krLDNIContouring_constructFaceArray(unsigned int *bndCell, int bndCellNum, bool *gridNodes, int nMeshRes, int nStepSize,
													int nAxis, unsigned int *faceArray, int invalidValue, float *vPosArray,
													float ox, float oy, float oz, float ww,
													float *vAdditionalVerArray, int *additionalVerNum,
													bool bWithIntersectionPrevention)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int cellIndex,delta=nMeshRes*nMeshRes,ix,iy,iz,startIndex,endIndex,ii[4];
	float pp[4][3],dd1,dd2,aa[3][3],sPP[3],ePP[3],avgPos[3],vec[3],projD;
	unsigned int queryKey;
	int k,concaveNum;	float det1,det2;	bool bConvex[4];	
	int newVerIndex;

	while (index<bndCellNum) {
		cellIndex=bndCell[index];
		iz=cellIndex/delta;	iy=cellIndex%delta;	ix=iy%nMeshRes; iy=iy/nMeshRes;

		startIndex=iz*(nMeshRes+1)*(nMeshRes+1)+iy*(nMeshRes+1)+ix;
		switch(nAxis) {
		case 0:{endIndex=iz*(nMeshRes+1)*(nMeshRes+1)+iy*(nMeshRes+1)+ix+1;		}break;
		case 1:{endIndex=iz*(nMeshRes+1)*(nMeshRes+1)+(iy+1)*(nMeshRes+1)+ix;	}break;
		case 2:{endIndex=(iz+1)*(nMeshRes+1)*(nMeshRes+1)+iy*(nMeshRes+1)+ix;	}break;
		}
		faceArray[index*16]=faceArray[index*16+1]=faceArray[index*16+2]=faceArray[index*16+3]=
		faceArray[index*16+4]=faceArray[index*16+5]=faceArray[index*16+6]=faceArray[index*16+7]=
		faceArray[index*16+8]=faceArray[index*16+9]=faceArray[index*16+10]=faceArray[index*16+11]=
		faceArray[index*16+12]=faceArray[index*16+13]=faceArray[index*16+14]=faceArray[index*16+15]=invalidValue;

		if (gridNodes[startIndex]!=gridNodes[endIndex]) {
			//--------------------------------------------------------------------------------------------------
			//	Construct quadrilateral face
			switch(nAxis) {
			case 0:{
				if (gridNodes[startIndex]) {
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+iy*nMeshRes+ix);			ii[0]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+(iy-1)*nMeshRes+ix);		ii[1]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)((iz-1)*nMeshRes*nMeshRes+(iy-1)*nMeshRes+ix);	ii[2]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)((iz-1)*nMeshRes*nMeshRes+iy*nMeshRes+ix);		ii[3]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					sPP[0]=ox+ww*(float)nStepSize*(float)ix;		sPP[1]=oy+ww*(float)nStepSize*(float)iy;		sPP[2]=oz+ww*(float)nStepSize*(float)iz;
					ePP[0]=ox+ww*(float)nStepSize*((float)ix+1.0);	ePP[1]=oy+ww*(float)nStepSize*(float)iy;		ePP[2]=oz+ww*(float)nStepSize*(float)iz;
				}
				else {
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+iy*nMeshRes+ix);			ii[0]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)((iz-1)*nMeshRes*nMeshRes+iy*nMeshRes+ix);		ii[1]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)((iz-1)*nMeshRes*nMeshRes+(iy-1)*nMeshRes+ix);	ii[2]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+(iy-1)*nMeshRes+ix);		ii[3]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					sPP[0]=ox+ww*(float)nStepSize*((float)ix+1.0);	sPP[1]=oy+ww*(float)nStepSize*(float)iy;		sPP[2]=oz+ww*(float)nStepSize*(float)iz;
					ePP[0]=ox+ww*(float)nStepSize*(float)ix;		ePP[1]=oy+ww*(float)nStepSize*(float)iy;		ePP[2]=oz+ww*(float)nStepSize*(float)iz;
				}
				   }break;
			case 1:{
				if (gridNodes[startIndex]) {
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+iy*nMeshRes+ix);			ii[0]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)((iz-1)*nMeshRes*nMeshRes+iy*nMeshRes+ix);		ii[1]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)((iz-1)*nMeshRes*nMeshRes+iy*nMeshRes+ix-1);		ii[2]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+iy*nMeshRes+ix-1);			ii[3]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					sPP[0]=ox+ww*(float)nStepSize*(float)ix;		sPP[1]=oy+ww*(float)nStepSize*(float)iy;		sPP[2]=oz+ww*(float)nStepSize*(float)iz;
					ePP[0]=ox+ww*(float)nStepSize*(float)ix;		ePP[1]=oy+ww*(float)nStepSize*((float)iy+1.0);	ePP[2]=oz+ww*(float)nStepSize*(float)iz;
				}
				else {
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+iy*nMeshRes+ix);			ii[0]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+iy*nMeshRes+ix-1);			ii[1]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)((iz-1)*nMeshRes*nMeshRes+iy*nMeshRes+ix-1);		ii[2]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)((iz-1)*nMeshRes*nMeshRes+iy*nMeshRes+ix);		ii[3]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					sPP[0]=ox+ww*(float)nStepSize*(float)ix;		sPP[1]=oy+ww*(float)nStepSize*((float)iy+1.0);	sPP[2]=oz+ww*(float)nStepSize*(float)iz;
					ePP[0]=ox+ww*(float)nStepSize*(float)ix;		ePP[1]=oy+ww*(float)nStepSize*(float)iy;		ePP[2]=oz+ww*(float)nStepSize*(float)iz;
				}
				   }break;
			case 2:{
				if (gridNodes[startIndex]) {
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+iy*nMeshRes+ix);			ii[0]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+iy*nMeshRes+ix-1);			ii[1]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+(iy-1)*nMeshRes+ix-1);		ii[2]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+(iy-1)*nMeshRes+ix);		ii[3]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					sPP[0]=ox+ww*(float)nStepSize*(float)ix;		sPP[1]=oy+ww*(float)nStepSize*(float)iy;		sPP[2]=oz+ww*(float)nStepSize*(float)iz;
					ePP[0]=ox+ww*(float)nStepSize*(float)ix;		ePP[1]=oy+ww*(float)nStepSize*(float)iy;		ePP[2]=oz+ww*(float)nStepSize*((float)iz+1.0);
				}
				else {
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+iy*nMeshRes+ix);			ii[0]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+(iy-1)*nMeshRes+ix);		ii[1]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+(iy-1)*nMeshRes+ix-1);		ii[2]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					queryKey=(unsigned int)(iz*nMeshRes*nMeshRes+iy*nMeshRes+ix-1);			ii[3]=_searchCompactBndCellIndex(bndCell,bndCellNum,queryKey)+1;
					sPP[0]=ox+ww*(float)nStepSize*(float)ix;		sPP[1]=oy+ww*(float)nStepSize*(float)iy;		sPP[2]=oz+ww*(float)nStepSize*((float)iz+1.0);
					ePP[0]=ox+ww*(float)nStepSize*(float)ix;		ePP[1]=oy+ww*(float)nStepSize*(float)iy;		ePP[2]=oz+ww*(float)nStepSize*(float)iz;
				}
				   }break;
			}

			//--------------------------------------------------------------------------------------------------
			//	Split quadrilateral face into triangular faces
			for(k=0;k<4;k++) {
				ii[k]--;
				pp[k][0]=vPosArray[ii[k]*3];	pp[k][1]=vPosArray[ii[k]*3+1];	pp[k][2]=vPosArray[ii[k]*3+2];
				ii[k]++;
			}
			
			concaveNum=0;
			if (bWithIntersectionPrevention) {
				bConvex[0]=bConvex[1]=bConvex[2]=bConvex[3]=true;		

				for(k=0;k<4;k++) {
					aa[0][0]=pp[k][0]-ePP[0];			aa[0][1]=pp[k][1]-ePP[1];			aa[0][2]=pp[k][2]-ePP[2];
					aa[1][0]=pp[(k+3)%4][0]-ePP[0];		aa[1][1]=pp[(k+3)%4][1]-ePP[1];		aa[1][2]=pp[(k+3)%4][2]-ePP[2];
					aa[2][0]=pp[(k+1)%4][0]-ePP[0];		aa[2][1]=pp[(k+1)%4][1]-ePP[1];		aa[2][2]=pp[(k+1)%4][2]-ePP[2];
					det1=_determination(aa);

					aa[0][0]=pp[k][0]-sPP[0];			aa[0][1]=pp[k][1]-sPP[1];			aa[0][2]=pp[k][2]-sPP[2];
					aa[1][0]=pp[(k+3)%4][0]-sPP[0];		aa[1][1]=pp[(k+3)%4][1]-sPP[1];		aa[1][2]=pp[(k+3)%4][2]-sPP[2];
					aa[2][0]=pp[(k+1)%4][0]-sPP[0];		aa[2][1]=pp[(k+1)%4][1]-sPP[1];		aa[2][2]=pp[(k+1)%4][2]-sPP[2];
					det2=_determination(aa);
			
					if (!((det1>=0.0) && (det2<=0.0))) {bConvex[k]=false; concaveNum++;}
				}

				if (concaveNum==1) {
					for(k=0;k<4;k++) {
						if (!(bConvex[k])) {
							faceArray[index*16]=ii[k];		faceArray[index*16+1]=ii[(k+1)%4];	faceArray[index*16+2]=ii[(k+2)%4];	faceArray[index*16+3]=0;
							faceArray[index*16+4]=ii[k];	faceArray[index*16+5]=ii[(k+2)%4];	faceArray[index*16+6]=ii[(k+3)%4];	faceArray[index*16+7]=0;
							break;
						}
					}
				}
				else if (concaveNum>1) {
	//				faceArray[index*16]=ii[0];		faceArray[index*16+1]=ii[1];	faceArray[index*16+2]=ii[2];	faceArray[index*16+3]=ii[3];

					//--------------------------------------------------------------------------------------------
					//	project the average point onto the edge
					avgPos[0]=avgPos[1]=avgPos[2]=0.0;
					for(k=0;k<4;k++) {avgPos[0]+=pp[k][0];	avgPos[1]+=pp[k][1];	avgPos[2]+=pp[k][2];}
					avgPos[0]=avgPos[0]*0.25;	avgPos[1]=avgPos[1]*0.25;	avgPos[2]=avgPos[2]*0.25;
					//--------------------------------------------------------------------------------------------
					vec[0]=vec[1]=vec[2]=0.0;	
					if (gridNodes[endIndex]) {
						vec[nAxis]=-1.0;
						projD=(avgPos[0]-ePP[0])*vec[0]+(avgPos[1]-ePP[1])*vec[1]+(avgPos[2]-ePP[2])*vec[2];
						if (projD<0.0) projD=0.0;		if (projD>ww*(float)nStepSize) projD=ww*(float)nStepSize;
						avgPos[0]=ePP[0]+projD*vec[0];	avgPos[1]=ePP[1]+projD*vec[1];	avgPos[2]=ePP[2]+projD*vec[2];
					}
					else {
						vec[nAxis]=1.0;
						projD=(avgPos[0]-sPP[0])*vec[0]+(avgPos[1]-sPP[1])*vec[1]+(avgPos[2]-sPP[2])*vec[2];
						if (projD<0.0) projD=0.0;		if (projD>ww*(float)nStepSize) projD=ww*(float)nStepSize;
						avgPos[0]=sPP[0]+projD*vec[0];	avgPos[1]=sPP[1]+projD*vec[1];	avgPos[2]=sPP[2]+projD*vec[2];
					}
					//--------------------------------------------------------------------------------------------
					newVerIndex=atomicAdd(additionalVerNum,1);
					vAdditionalVerArray[newVerIndex*3]=avgPos[0];
					vAdditionalVerArray[newVerIndex*3+1]=avgPos[1];
					vAdditionalVerArray[newVerIndex*3+2]=avgPos[2];

					for(k=0;k<4;k++) {
						faceArray[index*16+k*4]=newVerIndex+bndCellNum+1;	
						// as the bndCellNum is the same as the number of vertices already created
						faceArray[index*16+k*4+1]=ii[k];
						faceArray[index*16+k*4+2]=ii[(k+1)%4];
						faceArray[index*16+k*4+3]=0;
					}
				}
			}

			if (concaveNum==0) {
				dd1=(pp[0][0]-pp[2][0])*(pp[0][0]-pp[2][0])+(pp[0][1]-pp[2][1])*(pp[0][1]-pp[2][1])+(pp[0][2]-pp[2][2])*(pp[0][2]-pp[2][2]);
				dd2=(pp[1][0]-pp[3][0])*(pp[1][0]-pp[3][0])+(pp[1][1]-pp[3][1])*(pp[1][1]-pp[3][1])+(pp[1][2]-pp[3][2])*(pp[1][2]-pp[3][2]);
				if (dd1<dd2) {
					faceArray[index*16]=ii[0];		faceArray[index*16+1]=ii[1];	faceArray[index*16+2]=ii[2];	faceArray[index*16+3]=0;
					faceArray[index*16+4]=ii[0];	faceArray[index*16+5]=ii[2];	faceArray[index*16+6]=ii[3];	faceArray[index*16+7]=0;
				}
				else {
					faceArray[index*16]=ii[0];		faceArray[index*16+1]=ii[1];	faceArray[index*16+2]=ii[3];	faceArray[index*16+3]=0;
					faceArray[index*16+4]=ii[1];	faceArray[index*16+5]=ii[2];		faceArray[index*16+6]=ii[3];	faceArray[index*16+7]=0;
				}
			}
		}

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIContouring_constructVertexArray(unsigned int *bndCell, int bndCellNum, int nMeshRes, int nStepSize,
									unsigned int *xIndexArray, float *xNxArray, float *xNyArray, float *xDepthArray,
									unsigned int *yIndexArray, float *yNxArray, float *yNyArray, float *yDepthArray,
									unsigned int *zIndexArray, float *zNxArray, float *zNyArray, float *zDepthArray,
									float ox, float oy, float oz, float ww, int res, float *vPosArray, bool bWithIntersectionPrevention)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int cellIndex,delta=nMeshRes*nMeshRes,ix,iy,iz,i,j;
	float pp[3],A[3][3],X[3],B[3],UU[3][3],VV[3][3],UUT[3][3],VVT[3][3],maxFactor,scale;
	float minP[3],maxP[3];
	const float criterion=0.1;

	while (index<bndCellNum) {
		cellIndex=(int)(bndCell[index]);
		iz=cellIndex/delta;		iy=cellIndex%delta;		ix=iy%nMeshRes;		iy=iy/nMeshRes;

		pp[0]=((float)ix+0.5)*ww*(float)nStepSize;	
		pp[1]=((float)iy+0.5)*ww*(float)nStepSize;	
		pp[2]=((float)iz+0.5)*ww*(float)nStepSize;
		minP[0]=pp[0]-0.5*ww*(float)nStepSize;	minP[1]=pp[1]-0.5*ww*(float)nStepSize;	minP[2]=pp[2]-0.5*ww*(float)nStepSize;
		maxP[0]=pp[0]+0.5*ww*(float)nStepSize;	maxP[1]=pp[1]+0.5*ww*(float)nStepSize;	maxP[2]=pp[2]+0.5*ww*(float)nStepSize;
		
		if (_searchToFillQEFMatrix(pp,A,B,ix,iy,iz,nStepSize,ww,res,
			xIndexArray,xNxArray,xNyArray,xDepthArray,yIndexArray,yNxArray,yNyArray,yDepthArray,zIndexArray,zNxArray,zNyArray,zDepthArray)) 
		{	// using SVD to determine an optimal position
			if (_singularValueDecomposition(A,UU,VVT)) {
				_transpose(UU,UUT);		_transpose(VVT,VV);
				maxFactor=(fabs(A[0][0])>fabs(A[1][1]))?(A[0][0]):(A[1][1]);
				maxFactor=(fabs(maxFactor)>fabs(A[2][2]))?(maxFactor):(A[2][2]);
				if (fabs(maxFactor)<0.001) {
					for(i=0;i<3;i++) {	for(j=0;j<3;j++) {	A[i][j]=0.0;}}
				}
				else {
					for(i=0;i<3;i++) {
						for(j=0;j<3;j++) {
							if (i!=j) 
								{A[i][j]=0.0;}
							else 
								{if (fabs(A[i][j]/maxFactor)<criterion) A[i][j]=0.0; else A[i][j]=1.0/A[i][j];}
						}
					}
				}
				_mul(UUT,B,X);	_mul(A,X,B);	_mul(VV,B,X);

				//-----------------------------------------------------------------
				//	truncate the update vector and update node position
				scale=1.0; //float temp = 0.0;
				if (bWithIntersectionPrevention) {
					
					if (fabs(X[0])>1.0e-5 && (pp[0]+X[0]*scale)>maxP[0]) scale=(maxP[0]-pp[0])/X[0];
					if (fabs(X[1])>1.0e-5 && (pp[1]+X[1]*scale)>maxP[1]) scale=(maxP[1]-pp[1])/X[1];
					if (fabs(X[2])>1.0e-5 && (pp[2]+X[2]*scale)>maxP[2]) scale=(maxP[2]-pp[2])/X[2];
					if (fabs(X[0])>1.0e-5 && (pp[0]+X[0]*scale)<minP[0]) scale=(minP[0]-pp[0])/X[0];
					if (fabs(X[1])>1.0e-5 && (pp[1]+X[1]*scale)<minP[1]) scale=(minP[1]-pp[1])/X[1];
					if (fabs(X[2])>1.0e-5 && (pp[2]+X[2]*scale)<minP[2]) scale=(minP[2]-pp[2])/X[2];
				}
				pp[0]=pp[0]+X[0]*scale;	pp[1]=pp[1]+X[1]*scale;	pp[2]=pp[2]+X[2]*scale;
			}
		}
		else 
		{
			pp[0]=((float)ix+0.5)*ww*(float)nStepSize;
			pp[1]=((float)iy+0.5)*ww*(float)nStepSize;
			pp[2]=((float)iz+0.5)*ww*(float)nStepSize;
		}
		vPosArray[index*3]=ox+pp[0];	
		vPosArray[index*3+1]=oy+pp[1];	
		vPosArray[index*3+2]=oz+pp[2];

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIContouring_constructBndCells(unsigned int *bndCell, bool *gridNodes, int nMeshRes, int zStartIndex, int zStep, unsigned int invalidValue)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int bndCellArrsize=nMeshRes*nMeshRes*zStep;
	int delta=nMeshRes*nMeshRes,ix,iy,iz,dx,dy,dz,i;

	while(index<bndCellArrsize) {
		bndCell[index]=invalidValue;

		iz=index/delta+zStartIndex;	  iy=index%delta;	ix=iy%nMeshRes; iy=iy/nMeshRes;
		for(i=1;i<8;i++) {
			dz=i/4;	dy=i%4; dx=dy%2; dy=dy/2;
			if (gridNodes[iz*(nMeshRes+1)*(nMeshRes+1)+iy*(nMeshRes+1)+ix]
				!=gridNodes[(iz+dz)*(nMeshRes+1)*(nMeshRes+1)+(iy+dy)*(nMeshRes+1)+(ix+dx)]) {
				bndCell[index]=(unsigned int)(index+delta*zStartIndex);	break;
			}
		}

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIContouring_votingGridNodes(unsigned int *compactIndexArray, int nMeshRes, int rayNum,
												 unsigned int *indexArray, int res, int nStepSize, float *depthArray, 
												 bool *gridNodes, float ww, int nAxis)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int i,j,k,index2;
	int si,se,ss,ee;
	int delta=(nMeshRes+1)*(nMeshRes+1);
	float gridWidth=ww*(float)nStepSize;

	while (index<rayNum) {
		index2 = compactIndexArray[index];
		i = index2 % (nMeshRes+1);		j = index2 / (nMeshRes+1);	// indices in grids

		si=indexArray[(j*nStepSize)*res+(i*nStepSize)];
		se=indexArray[(j*nStepSize)*res+(i*nStepSize)+1];
		for(;si<se;si=si+2) {
			ss=(int)(ceil(fabs(depthArray[si])/gridWidth));	
			ee=(int)(fabs(depthArray[si+1])/gridWidth);
			switch(nAxis) {
			case 0:{for(k=ss;k<=ee;k++) gridNodes[j*delta+i*(nMeshRes+1)+k]=true;	   }break;
			case 1:{for(k=ss;k<=ee;k++) gridNodes[i*delta+k*(nMeshRes+1)+j]=true;	   }break;
			case 2:{for(k=ss;k<=ee;k++) gridNodes[k*delta+j*(nMeshRes+1)+i]=true;	   }break;
			}
		}

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIContouring_initCompactIndexRays(unsigned int *compactIndexArray, int nMeshRes, 
													  unsigned int *indexArray, int res, int nStepSize, unsigned int emptyValue)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int i,j,index2;
	int meshArrSize=(nMeshRes+1)*(nMeshRes+1);

	while (index<meshArrSize) {
		i=index%(nMeshRes+1);	j=index/(nMeshRes+1);

		index2=(j*nStepSize)*res+(i*nStepSize);
		compactIndexArray[index]=emptyValue;
		if ((j*nStepSize<res) && (i*nStepSize<res)) {
			if ((indexArray[index2+1]-indexArray[index2])>0) 
				compactIndexArray[index]=(unsigned int)index;
		}

		index += blockDim.x * gridDim.x;
	}

}

//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------
#define C_EPS 1.0e-8
__device__ bool _calTwoLinesIntersection(double3 l1, double3 l2, double pt[])
{
	double d =l1.x*l2.y - l2.x*l1.y;

	if (fabs(d)<C_EPS) return false;

	pt[0] = -(l1.z*l2.y - l2.z*l1.y)/d;
	pt[1] = (l1.z*l2.x - l2.z*l1.x)/d;

	return true;
}

__device__ bool _calTwoLineSegmentsIntersection(float2 vMinus1, float2 v_a, float2 v_b, float2 vPlus1, double pt[])
{
	double3 L1, L2;
	

	L1 = _calEquation(v_a, v_b);
	L2 = _calEquation(vPlus1, vMinus1);

	
	if (!(_calTwoLinesIntersection(L1,L2,pt))) return false;


	double u1;
	if (fabs(vPlus1.x - vMinus1.x) <= C_EPS)
		u1=(pt[1] - vPlus1.y)/(vMinus1.y - vPlus1.y);
	else
		u1=(pt[0] - vPlus1.x)/(vMinus1.x - vPlus1.x);

	double u2;

	if (fabs(v_a.x - v_b.x) <= C_EPS)
		u2=(pt[1] - v_a.y)/(v_b.y - v_a.y);
	else
		u2=(pt[0] - v_a.x)/(v_b.x - v_a.x);

	if ((u1>=0.0) && (u1<=1.0) && (u2>=0.0) && (u2<=1.0)) return true;

	return false;

}

__device__ double3 _calEquation(float2 v1, float2 v2)
{
	double3 p;
	p.x = v2.y-v1.y;
	p.y = v1.x-v2.x;
	double sqroot = sqrt(p.x*p.x+p.y*p.y);
	p.x = p.x/sqroot;
	p.y = p.y/sqroot;
	
	if (fabs(p.y) < C_EPS)
	{
		p.x = 1.0;
		p.y = 0.0;
		p.z = -v1.x;
		return p;
	}

	p.z = -(p.y*v1.y + p.x*v1.x);
	return p;


}

__device__ bool _detectInOutPoint(float px, float py, float pz, 
								    unsigned int* xIndex, float* xDepth, 
									unsigned int* yIndex, float* yDepth, 
									unsigned int* zIndex, float* zDepth,
									float3 ori_p, float3 origin, float gwidth, int res)
{

	unsigned int i,j,k;
	unsigned int st, num, index, counter;
	float xx,yy,zz;

	i=(int)((px-ori_p.x)/gwidth);
	j=(int)((py-ori_p.y)/gwidth);
	k=(int)((pz-ori_p.z)/gwidth);
	if ((i<0) || (i>=res)) return false;
	if ((j<0) || (j>=res)) return false;
	if ((k<0) || (k>=res)) return false;

	counter = 0;
	st = xIndex[k*res+j];
	num = xIndex[k*res+j+1]-st;


	if (num > 0)
	{
		
		xx=px-origin.x;
		for(index=0;index<num;index+=2) {
			if (xx < fabs(xDepth[st+index])) break;
			if ((xx >= fabs(xDepth[st+index])) 
				&& (xx <= fabs(xDepth[st+index+1]))) {counter++; break;}
		}
	}

	if(counter>0) return true;


	st = yIndex[i*res+k];
	num = yIndex[i*res+k+1]-st;
	if (num > 0)
	{
		
		yy=py-origin.y;
		for(index=0;index<num;index+=2) {
			if (yy < fabs(yDepth[st+index])) break;
			if ((yy >= fabs(yDepth[st+index])) 
				&& (yy <= fabs(yDepth[st+index+1]))) {counter++; break;}
		}
	}
	if(counter>0) return true;


	st = zIndex[j*res+i];
	num = zIndex[j*res+i+1]-st;
	if (num > 0)
	{
		
		zz=pz-origin.z;
		for(index=0;index<num;index+=2) {
			if (zz < fabs(zDepth[st+index])) break;
			if ((zz >= fabs(zDepth[st+index])) 
				&& (zz <= fabs(zDepth[st+index+1]))) {counter++; break;}
		}

	}
	
	
	if(counter>0) return true;
	return false;

}


__device__ double3 _rotatePointAlongVector(double px, double py, double pz, 
										  double x1, double y1, double z1, 
										  double x2, double y2, double z2,
										  double angle)
{
	double rx,ry,rz,rrrr;	double costheta,sintheta;
	double3 p;

	angle=DEGREE_TO_ROTATE(angle);
	costheta=cos(angle);	sintheta=sin(angle);
	p.x=0.0;	p.y=0.0;	p.z=0.0;	px=px-x1;	py=py-y1;	pz=pz-z1;
	rx=x2-x1;	ry=y2-y1;	rz=z2-z1;	rrrr=sqrt(rx*rx+ry*ry+rz*rz);
	rx=rx/rrrr;	ry=ry/rrrr;	rz=rz/rrrr;

	p.x += (costheta + (1 - costheta) * rx * rx) * px;
	p.x += ((1 - costheta) * rx * ry - rz * sintheta) * py;
	p.x += ((1 - costheta) * rx * rz + ry * sintheta) * pz;

	p.y += ((1 - costheta) * rx * ry + rz * sintheta) * px;
	p.y += (costheta + (1 - costheta) * ry * ry) * py;
	p.y += ((1 - costheta) * ry * rz - rx * sintheta) * pz;

	p.z += ((1 - costheta) * rx * rz - ry * sintheta) * px;
	p.z += ((1 - costheta) * ry * rz + rx * sintheta) * py;
	p.z += (costheta + (1 - costheta) * rz * rz) * pz;

	p.x += x1;	p.y += y1;	p.z += z1;

	return p;
}



__device__ float _determination(float a[][3])
{
	float value;

	value=a[0][0]*a[1][1]*a[2][2]+a[0][1]*a[1][2]*a[2][0]+a[0][2]*a[1][0]*a[2][1]
		-a[0][0]*a[1][2]*a[2][1]-a[0][1]*a[1][0]*a[2][2]-a[0][2]*a[1][1]*a[2][0];
	
	return value;
}

__device__ unsigned int _searchCompactBndCellIndex(unsigned int *bndCell, int bndCellNum, unsigned int queryKey)
{
	int index,sIndex,eIndex;	

	sIndex=0;	eIndex=bndCellNum;
	do{
		index=(sIndex+eIndex)/2;
		if (bndCell[index]==queryKey) break;
		if (queryKey>bndCell[index]) sIndex=index; else eIndex=index;
	}while(sIndex!=eIndex);
	
//	for(index=0;index<bndCellNum;index++) {if (bndCell[index]==queryKey) break;	}

	return (unsigned int)index;
}


__device__ bool _searchToFillQEFMatrix(float pp[], float A[][3], float B[], int ix, int iy, int iz, int nStepSize, float ww, int res,
									   unsigned int *xIndexArray, float *xNxArray, float *xNyArray, float *xDepthArray,
									   unsigned int *yIndexArray, float *yNxArray, float *yNyArray, float *yDepthArray,
									   unsigned int *zIndexArray, float *zNxArray, float *zNyArray, float *zDepthArray)
{
	float xmin,ymin,zmin,xmax,ymax,zmax,upper,lower,pos[3],nv[3],dd,proj;
	const float eps=0.001*ww;	
	int nAxis,ii,jj,is,ie,js,je,sampleIndex;
	int k,nNum,index,sampleNum;
	unsigned int *indexArray;
	float *nxArray, *nyArray, *depthArray;
	bool bFoundSamples=false;

	xmin=ww*(float)(ix*nStepSize);		ymin=ww*(float)(iy*nStepSize);		zmin=ww*(float)(iz*nStepSize);			
	xmax=ww*(float)((ix+1)*nStepSize);	ymax=ww*(float)((iy+1)*nStepSize);	zmax=ww*(float)((iz+1)*nStepSize);			
	A[0][0]=A[0][1]=A[0][2]=A[1][0]=A[1][1]=A[1][2]=A[2][0]=A[2][1]=A[2][2]=0.0;
	B[0]=B[1]=B[2]=0.0;

	//-----------------------------------------------------------------------------------------------------------------------
	//	computing the average position
	pp[0]=pp[1]=pp[2]=0.0;	sampleNum=0;
	for(nAxis=0;nAxis<3;nAxis++) {
		switch(nAxis) {
		case 0:{lower=xmin-eps;	upper=xmax+eps;	is=iy*nStepSize; ie=is+nStepSize; js=iz*nStepSize; je=js+nStepSize;
				indexArray=xIndexArray; nxArray=xNxArray; nyArray=xNyArray; depthArray=xDepthArray;
			   }break;
		case 1:{lower=ymin-eps;	upper=ymax+eps;	is=iz*nStepSize; ie=is+nStepSize; js=ix*nStepSize; je=js+nStepSize;
				indexArray=yIndexArray; nxArray=yNxArray; nyArray=yNyArray; depthArray=yDepthArray;
			   }break;
		case 2:{lower=zmin-eps;	upper=zmax+eps;	is=ix*nStepSize; ie=is+nStepSize; js=iy*nStepSize; je=js+nStepSize;
				indexArray=zIndexArray; nxArray=zNxArray; nyArray=zNyArray; depthArray=zDepthArray;
			   }break;
		}

		for(ii=is;ii<=ie;ii++) {
			if ((ii>=res) || (ii<0)) continue;
			for(jj=js;jj<=je;jj++) {
				if ((jj>=res) || (jj<0)) continue;

				index=ii+jj*res;	nNum=indexArray[index+1]-indexArray[index];
				pos[(nAxis+1)%3]=ww*(float)ii;
				pos[(nAxis+2)%3]=ww*(float)jj;
				sampleIndex=indexArray[index];
				for(k=0;k<nNum;k++) {
					if (fabs(depthArray[sampleIndex+k])>upper) break;
					if (fabs(depthArray[sampleIndex+k])<lower) continue;

					pos[nAxis]=fabs(depthArray[sampleIndex+k]);
					pp[0]+=pos[0];	pp[1]+=pos[1];	pp[2]+=pos[2];	
					sampleNum++;
				}
			}
		}
	}
	if (sampleNum==0) return false;
	pp[0]=pp[0]/(float)sampleNum;	pp[1]=pp[1]/(float)sampleNum;	pp[2]=pp[2]/(float)sampleNum;
//	return true;

	//-----------------------------------------------------------------------------------------------------------------------
	//	filling the QEF matrix
	for(nAxis=0;nAxis<3;nAxis++) {
		switch(nAxis) {
		case 0:{lower=xmin-eps;	upper=xmax+eps;	is=iy*nStepSize; ie=(iy+1)*nStepSize; js=iz*nStepSize; je=(iz+1)*nStepSize;
				indexArray=xIndexArray; nxArray=xNxArray; nyArray=xNyArray; depthArray=xDepthArray;
			   }break;
		case 1:{lower=ymin-eps;	upper=ymax+eps;	is=iz*nStepSize; ie=(iz+1)*nStepSize; js=ix*nStepSize; je=(ix+1)*nStepSize;
				indexArray=yIndexArray; nxArray=yNxArray; nyArray=yNyArray; depthArray=yDepthArray;
			   }break;
		case 2:{lower=zmin-eps;	upper=zmax+eps;	is=ix*nStepSize; ie=(ix+1)*nStepSize; js=iy*nStepSize; je=(iy+1)*nStepSize;
				indexArray=zIndexArray; nxArray=zNxArray; nyArray=zNyArray; depthArray=zDepthArray;
			   }break;
		}
		for(ii=is;ii<=ie;ii++) {
			if ((ii>=res) || (ii<0)) continue;
			for(jj=js;jj<=je;jj++) {
				if ((jj>=res) || (jj<0)) continue;

				index=ii+jj*res;	nNum=indexArray[index+1]-indexArray[index];
				pos[(nAxis+1)%3]=ww*(float)ii;
				pos[(nAxis+2)%3]=ww*(float)jj;
				sampleIndex=indexArray[index];
				for(k=0;k<nNum;k++) {
					if (fabs(depthArray[k+sampleIndex])>upper) break;
					if (fabs(depthArray[k+sampleIndex])<lower) continue;

					pos[nAxis]=fabs(depthArray[k+sampleIndex]);
					nv[0]=nxArray[k+sampleIndex];		nv[1]=nyArray[k+sampleIndex];
					dd=1.0-(nv[0]*nv[0]+nv[1]*nv[1]);
					if (dd<0.0) dd=0.0;
					dd=sqrt(dd);
					if (depthArray[k+sampleIndex]>0.0) nv[2]=dd; else nv[2]=-dd;

					proj=(pos[0]-pp[0])*nv[0]+(pos[1]-pp[1])*nv[1]+(pos[2]-pp[2])*nv[2];
					B[0]+=proj*nv[0];	B[1]+=proj*nv[1];	B[2]+=proj*nv[2];

					A[0][0]+=nv[0]*nv[0];	A[0][1]+=nv[0]*nv[1];	A[0][2]+=nv[0]*nv[2];
					A[1][1]+=nv[1]*nv[1];	A[1][2]+=nv[1]*nv[2];
					A[2][2]+=nv[2]*nv[2];

					bFoundSamples=true;
				}
			}
		}
	}
	A[1][0]=A[0][1];	A[2][0]=A[0][2];	A[2][1]=A[1][2];

	return bFoundSamples;
}

__device__ void ppp(float a[][3], float e[], float s[], float v[][3], int m, int n)
{
	int i,j;
    float d;

    if (m>=n) i=n;
    else i=m;
    for (j=1; j<=i-1; j++)
      { a[(j-1)][j-1]=s[j-1];
        a[(j-1)][j]=e[j-1];
      }
    a[(i-1)][i-1]=s[i-1];
    if (m<n) a[(i-1)][i]=e[i-1];
    for (i=1; i<=n-1; i++)
    for (j=i+1; j<=n; j++)
      {
        d=v[i-1][j-1]; v[i-1][j-1]=v[j-1][i-1]; v[j-1][i-1]=d;
      }
}

__device__ void sss(float fg[], float cs[])
{
	float r,d;
    if ((fabs(fg[0])+fabs(fg[1]))==0.0)
      { cs[0]=1.0; cs[1]=0.0; d=0.0;}
    else 
      { d=sqrt(fg[0]*fg[0]+fg[1]*fg[1]);
        if (fabs(fg[0])>fabs(fg[1]))
          { d=fabs(d);
            if (fg[0]<0.0) d=-d;
          }
        if (fabs(fg[1])>=fabs(fg[0]))
          { d=fabs(d);
            if (fg[1]<0.0) d=-d;
          }
        cs[0]=fg[0]/d; cs[1]=fg[1]/d;
      }
    r=1.0;
    if (fabs(fg[0])>fabs(fg[1])) r=cs[1];
    else
      if (cs[0]!=0.0) r=1.0/cs[0];
    fg[0]=d; fg[1]=r;
}

__device__ bool _singularValueDecomposition(float a[][3], float u[][3], float v[][3]) 
{
	float eps=0.00001;
	int i,j,k,l,it,ll,kk,mm,nn,m1,ks;
    float d,dd,t,sm,sm1,em1,sk,ek,b,c,shh,fg[2],cs[2];
    float s[5],e[5],w[5];
	const int m=3,n=3;

//	ka=((m>n)?m:n)+1;

	for(i=0;i<3;i++) {
		for(j=0;j<3;j++) {
			u[i][j]=v[i][j]=0.0;
		}
	}
    
	it=60; k=n;
    if (m-1<n) k=m-1;
    l=m;
    if (n-2<m) l=n-2;
    if (l<0) l=0;
    ll=k;
    if (l>k) ll=l;
    if (ll>=1)
      { for (kk=1; kk<=ll; kk++)
          { if (kk<=k)
              { d=0.0;
                for (i=kk; i<=m; i++)
                  { d=d+a[i-1][kk-1]*a[i-1][kk-1];}
                s[kk-1]=sqrt(d);
                if (s[kk-1]!=0.0)
                  { 
                    if (a[kk-1][kk-1]!=0.0)
                      { s[kk-1]=fabs(s[kk-1]);
                        if (a[kk-1][kk-1]<0.0) s[kk-1]=-s[kk-1];
                      }
                    for (i=kk; i<=m; i++)
                      { 
                        a[i-1][kk-1]=a[i-1][kk-1]/s[kk-1];
                      }
                    a[kk-1][kk-1]=1.0+a[kk-1][kk-1];
                  }
                s[kk-1]=-s[kk-1];
              }
            if (n>=kk+1)
              { for (j=kk+1; j<=n; j++)
                  { if ((kk<=k)&&(s[kk-1]!=0.0))
                      { d=0.0;
                        for (i=kk; i<=m; i++)
                          {
                            d=d+a[i-1][kk-1]*a[i-1][j-1];
                          }
                        d=-d/a[kk-1][kk-1];
                        for (i=kk; i<=m; i++)
                          {
                            a[i-1][j-1]=a[i-1][j-1]+d*a[i-1][kk-1];
                          }
                      }
                    e[j-1]=a[kk-1][j-1];
                  }
              }
            if (kk<=k)
              { for (i=kk; i<=m; i++)
                  {
                    u[i-1][kk-1]=a[i-1][kk-1];
                  }
              }
            if (kk<=l)
              { d=0.0;
                for (i=kk+1; i<=n; i++)
                  d=d+e[i-1]*e[i-1];
                e[kk-1]=sqrt(d);
                if (e[kk-1]!=0.0)
                  { if (e[kk]!=0.0)
                      { e[kk-1]=fabs(e[kk-1]);
                        if (e[kk]<0.0) e[kk-1]=-e[kk-1];
                      }
                    for (i=kk+1; i<=n; i++)
                      e[i-1]=e[i-1]/e[kk-1];
                    e[kk]=1.0+e[kk];
                  }
                e[kk-1]=-e[kk-1];
                if ((kk+1<=m)&&(e[kk-1]!=0.0))
                  { for (i=kk+1; i<=m; i++) w[i-1]=0.0;
                    for (j=kk+1; j<=n; j++)
                      for (i=kk+1; i<=m; i++)
                        w[i-1]=w[i-1]+e[j-1]*a[i-1][j-1];
                    for (j=kk+1; j<=n; j++)
                      for (i=kk+1; i<=m; i++)
                        {
                          a[i-1][j-1]=a[i-1][j-1]-w[i-1]*e[j-1]/e[kk];
                        }
                  }
                for (i=kk+1; i<=n; i++)
                  v[i-1][kk-1]=e[i-1];
              }
          }
      }
    mm=n;
    if (m+1<n) mm=m+1;
    if (k<n) s[k]=a[k][k];
    if (m<mm) s[mm-1]=0.0;
    if (l+1<mm) e[l]=a[l][mm-1];
    e[mm-1]=0.0;
    nn=m;
    if (m>n) nn=n;
    if (nn>=k+1)
      { for (j=k+1; j<=nn; j++)
          { for (i=1; i<=m; i++)
              u[i-1][j-1]=0.0;
            u[j-1][j-1]=1.0;
          }
      }
    if (k>=1)
      { for (ll=1; ll<=k; ll++)
          { kk=k-ll+1;
            if (s[kk-1]!=0.0)
              { if (nn>=kk+1)
                  for (j=kk+1; j<=nn; j++)
                    { d=0.0;
                      for (i=kk; i<=m; i++)
                        {
                          d=d+u[i-1][kk-1]*u[i-1][j-1]/u[kk-1][kk-1];
                        }
                      d=-d;
                      for (i=kk; i<=m; i++)
                        { 
                          u[i-1][j-1]=u[i-1][j-1]+d*u[i-1][kk-1];
                        }
                    }
                  for (i=kk; i<=m; i++)
                    { u[i-1][kk-1]=-u[i-1][kk-1];}
                  u[kk-1][kk-1]=1.0+u[kk-1][kk-1];
                  if (kk-1>=1)
                    for (i=1; i<=kk-1; i++)
                      u[i-1][kk-1]=0.0;
              }
            else
              { for (i=1; i<=m; i++)
                  u[i-1][kk-1]=0.0;
                u[kk-1][kk-1]=1.0;
              }
          }
      }
    for (ll=1; ll<=n; ll++)
      { kk=n-ll+1;
        if ((kk<=l)&&(e[kk-1]!=0.0))
          { for (j=kk+1; j<=n; j++)
              { d=0.0;
                for (i=kk+1; i<=n; i++)
                  { 
                    d=d+v[i-1][kk-1]*v[i-1][j-1]/v[kk][kk-1];
                  }
                d=-d;
                for (i=kk+1; i<=n; i++)
                  { 
                    v[i-1][j-1]=v[i-1][j-1]+d*v[i-1][kk-1];
                  }
              }
          }
        for (i=1; i<=n; i++)
          v[i-1][kk-1]=0.0;
        v[kk-1][kk-1]=1.0;
      }
    for (i=1; i<=m; i++)
    for (j=1; j<=n; j++)
      a[i-1][j-1]=0.0;
    m1=mm; it=60;
    while (1==1)
      { if (mm==0)
          { ppp(a,e,s,v,m,n); return true;
          }
        if (it==0)
          { ppp(a,e,s,v,m,n); return false;
          }
        kk=mm-1;
	while ((kk!=0)&&(fabs(e[kk-1])!=0.0))
          { d=fabs(s[kk-1])+fabs(s[kk]);
            dd=fabs(e[kk-1]);
            if (dd>eps*d) kk=kk-1;
            else e[kk-1]=0.0;
          }
        if (kk==mm-1)
          { kk=kk+1;
            if (s[kk-1]<0.0)
              { s[kk-1]=-s[kk-1];
                for (i=1; i<=n; i++)
                  { v[i-1][kk-1]=-v[i-1][kk-1];}
              }
            while ((kk!=m1)&&(s[kk-1]<s[kk]))
              { d=s[kk-1]; s[kk-1]=s[kk]; s[kk]=d;
                if (kk<n)
                  for (i=1; i<=n; i++)
                    {
                      d=v[i-1][kk-1]; v[i-1][kk-1]=v[i-1][kk]; v[i-1][kk]=d;
                    }
                if (kk<m)
                  for (i=1; i<=m; i++)
                    {
                      d=u[i-1][kk-1]; u[i-1][kk-1]=u[i-1][kk]; u[i-1][kk]=d;
                    }
                kk=kk+1;
              }
            it=60;
            mm=mm-1;
          }
        else
          { ks=mm;
            while ((ks>kk)&&(fabs(s[ks-1])!=0.0))
              { d=0.0;
                if (ks!=mm) d=d+fabs(e[ks-1]);
                if (ks!=kk+1) d=d+fabs(e[ks-2]);
                dd=fabs(s[ks-1]);
                if (dd>eps*d) ks=ks-1;
                else s[ks-1]=0.0;
              }
            if (ks==kk)
              { kk=kk+1;
                d=fabs(s[mm-1]);
                t=fabs(s[mm-2]);
                if (t>d) d=t;
                t=fabs(e[mm-2]);
                if (t>d) d=t;
                t=fabs(s[kk-1]);
                if (t>d) d=t;
                t=fabs(e[kk-1]);
                if (t>d) d=t;
                sm=s[mm-1]/d; sm1=s[mm-2]/d;
                em1=e[mm-2]/d;
                sk=s[kk-1]/d; ek=e[kk-1]/d;
                b=((sm1+sm)*(sm1-sm)+em1*em1)/2.0;
                c=sm*em1; c=c*c; shh=0.0;
                if ((b!=0.0)||(c!=0.0))
                  { shh=sqrt(b*b+c);
                    if (b<0.0) shh=-shh;
                    shh=c/(b+shh);
                  }
                fg[0]=(sk+sm)*(sk-sm)-shh;
                fg[1]=sk*ek;
                for (i=kk; i<=mm-1; i++)
                  { sss(fg,cs);
                    if (i!=kk) e[i-2]=fg[0];
                    fg[0]=cs[0]*s[i-1]+cs[1]*e[i-1];
                    e[i-1]=cs[0]*e[i-1]-cs[1]*s[i-1];
                    fg[1]=cs[1]*s[i];
                    s[i]=cs[0]*s[i];
                    if ((cs[0]!=1.0)||(cs[1]!=0.0))
                      for (j=1; j<=n; j++)
                        {
                          d=cs[0]*v[j-1][i-1]+cs[1]*v[j-1][i];
                          v[j-1][i]=-cs[1]*v[j-1][i-1]+cs[0]*v[j-1][i];
                          v[j-1][i-1]=d;
                        }
                    sss(fg,cs);
                    s[i-1]=fg[0];
                    fg[0]=cs[0]*e[i-1]+cs[1]*s[i];
                    s[i]=-cs[1]*e[i-1]+cs[0]*s[i];
                    fg[1]=cs[1]*e[i];
                    e[i]=cs[0]*e[i];
                    if (i<m)
                      if ((cs[0]!=1.0)||(cs[1]!=0.0))
                        for (j=1; j<=m; j++)
                          {
                            d=cs[0]*u[j-1][i-1]+cs[1]*u[j-1][i];
                            u[j-1][i]=-cs[1]*u[j-1][i-1]+cs[0]*u[j-1][i];
                            u[j-1][i-1]=d;
                          }
                  }
                e[mm-2]=fg[0];
                it=it-1;
              }
            else
              { if (ks==mm)
                  { kk=kk+1;
                    fg[1]=e[mm-2]; e[mm-2]=0.0;
                    for (ll=kk; ll<=mm-1; ll++)
                      { i=mm+kk-ll-1;
                        fg[0]=s[i-1];
                        sss(fg,cs);
                        s[i-1]=fg[0];
                        if (i!=kk)
                          { fg[1]=-cs[1]*e[i-2];
                            e[i-2]=cs[0]*e[i-2];
                          }
                        if ((cs[0]!=1.0)||(cs[1]!=0.0))
                          for (j=1; j<=n; j++)
                            {
                              d=cs[0]*v[j-1][i-1]+cs[1]*v[j-1][mm-1];
                              v[j-1][mm-1]=-cs[1]*v[j-1][i-1]+cs[0]*v[j-1][mm-1];
                              v[j-1][i-1]=d;
                            }
                      }
                  }
                else
                  { kk=ks+1;
                    fg[1]=e[kk-2];
                    e[kk-2]=0.0;
                    for (i=kk; i<=mm; i++)
                      { fg[0]=s[i-1];
                        sss(fg,cs);
                        s[i-1]=fg[0];
                        fg[1]=-cs[1]*e[i-1];
                        e[i-1]=cs[0]*e[i-1];
                        if ((cs[0]!=1.0)||(cs[1]!=0.0))
                          for (j=1; j<=m; j++)
                            {
                              d=cs[0]*u[j-1][i-1]+cs[1]*u[j-1][kk-2];
                              u[j-1][kk-2]=-cs[1]*u[j-1][i-1]+cs[0]*u[j-1][kk-2];
                              u[j-1][i-1]=d;
                            }
                      }
                  }
              }
          }
      }

	return true;
}

__device__ void _transpose(float A[][3], float B[][3])
{
	B[0][0]=A[0][0];	B[0][1]=A[1][0];	B[0][2]=A[2][0];
	B[1][0]=A[0][1];	B[1][1]=A[1][1];	B[1][2]=A[2][1];
	B[2][0]=A[0][2];	B[2][1]=A[1][2];	B[2][2]=A[2][2];
}

__device__ void _mul(float A[][3], float B[][3], float C[][3])
{
	int i,j,l;

    for(i=0;i<3;i++) {
		for(j=0;j<3;j++) {
			C[i][j]=0.0;
			for (l=0;l<3;l++) C[i][j]=C[i][j]+A[i][l]*B[l][j];
		}
	}
}

__device__ void _mul(float A[][3], float x[], float b[])
{
	b[0]=A[0][0]*x[0]+A[0][1]*x[1]+A[0][2]*x[2];
	b[1]=A[1][0]*x[0]+A[1][1]*x[1]+A[1][2]*x[2];
	b[2]=A[2][0]*x[0]+A[2][1]*x[1]+A[2][2]*x[2];
}

__device__ inline unsigned int SetBitPos(unsigned int pos)
{
	return (1 << pos);
}

__device__ inline unsigned int bitCount(unsigned int i)
{
	i = i - ((i >> 1) & 0x55555555);
	i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
	return ((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) >> 24;
}

__device__ float interpoint(int x1, int y1, int x2, int y2, int x0) 
{
	float xM = float(x1 + x2) / 2.0f; 
	float yM = float(y1 + y2) / 2.0f; 
	float nx = x2 - x1; 
	float ny = y2 - y1; 

	return yM + nx * (xM - x0) / ny; 
}