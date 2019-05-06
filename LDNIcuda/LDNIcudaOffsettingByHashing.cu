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
#include <cstdlib>

#include "LDNIcudaSolid.h"
#include "LDNIcudaOperation.h"

//--------------------------------------------------------------------------------------------

extern __global__ void krLDNIOffsetting_fillingHashingTableKeyAndData(float *inputDepthArrayPtr, unsigned int *inputIndexArrayPtr, bool bInputIsNextOrNextNext, 
																	  short nCurrentDir, int res, float gwidth, int stIndex, int arrsize, 
																	  unsigned int *hashElementsKey, unsigned int *hashElementsDataLocation, float *hashElementsData);
extern __global__ void krLDNIOffsetting_fillingHashingIndexTable(int hashTableElementNum, unsigned int *hashElementsKey,
																 unsigned int *hashTableIndexArray, int arrsize);
extern __global__ void krLDNIOffsetting_fillingHashingSortedData(int hashTableElementNum, unsigned int *hashElementsDataLocation, 
																 float *hashElementsUnsortedData, float *hashElementsData);

extern __global__ void krLDNIOffsetting_EstimateWorkLoadingByValidSamplesOnPerpendicularRays(unsigned int *hashTableIndexArray,
								int *devSHBoxToBeSearched, int devNumOfSearchedSHBox, short nAxis, int res, float gwidth, float offset, 
								unsigned int *keyOfRayProcessing, unsigned int *orderOfRayProcessing);
extern __global__ void krLDNIOffsetting_DetermineNonzeroStartIndex(int res, unsigned int *keyOfRayProcessing, unsigned int *stIndex);

extern __global__ void krLDNIOffsetting_InRays(float *inputDepthArrayPtr, unsigned int *inputIndexArrayPtr,
								float offset, int arrsize, short nAxis,
								float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer, int *resBufferIndex,
								unsigned int *resSampleStIndex, unsigned int *resSampleNum);

extern __global__ void krLDNIOffsetting_ByParallelRays(float *inputDepthArrayPtr, unsigned int *inputIndexArrayPtr,
								float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer, int *resBufferIndex,
								unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								int *devRaysToBeSearched, int devNumOfSearchedRays, 
								short nAxis, int res, float gwidth, float offset, int stIndex, int edIndex, 
								float *resNxBuffer2, float *resNyBuffer2, float *resDepthBuffer2, int *resBufferIndex2,
								unsigned int *resSampleStIndex2, unsigned int *resSampleNum2);

extern __global__ void krLDNIOffsetting_BySpatialHashingSamplesOnPerpendicularRays(float *hashElementsData, unsigned int *hashTableIndexArray,
								int *devSHBoxToBeSearched, int devNumOfSearchedSHBox, unsigned int *orderOfRayProcessing, unsigned int nonzeroStIndex, 
								float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer, int *resBufferIndex,
								unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								short nAxis, int res, float gwidth, float offset, int stIndex, int edIndex, 
								float *resNxBuffer2, float *resNyBuffer2, float *resDepthBuffer2, int *resBufferIndex2,
								unsigned int *resSampleStIndex2, unsigned int *resSampleNum2);
extern __global__ void krLDNIOffsetting_OnlyCopyAsNotEffectedBySamplesOnPerpendicularRays(unsigned int *orderOfRayProcessing, unsigned int nonzeroStIndex, 
								float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer, int *resBufferIndex,
								unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								short nAxis, int stIndex, int edIndex, 
								float *resNxBuffer2, float *resNyBuffer2, float *resDepthBuffer2, int *resBufferIndex2,
								unsigned int *resSampleStIndex2, unsigned int *resSampleNum2);
//#else
extern __global__ void krLDNIOffsetting_BySpatialHashingSamplesOnPerpendicularRays(float *hashElementsData, unsigned int *hashTableIndexArray,
								int *devSHBoxToBeSearched, int devNumOfSearchedSHBox,
								float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer, int *resBufferIndex,
								unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								short nAxis, int res, float gwidth, float offset, int stIndex, int edIndex,
								float *resNxBuffer2, float *resNyBuffer2, float *resDepthBuffer2, int *resBufferIndex2,
								unsigned int *resSampleStIndex2, unsigned int *resSampleNum2);
//#endif

extern __global__ void krLDNIOffsetting_CollectResult(float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer,
								unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								float *outputNxArrayPtr, float *outputNzArrayPtr, float *outputDepthArrayPtr, unsigned int *outputIndexArrayPtr, 
								int arrsize);

extern void initResBuffer(float *&resNxBuffer, float *&resNyBuffer, float *&resDepthBuffer, int *&resBufferIndex); 

extern void freeResBuffer(float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer, int *resBufferIndex); 

extern void initStIndexAndNum(unsigned int *&resSampleStIndex, unsigned int *&resSampleNum, int size);

extern void cleanUpStIndexAndNum(unsigned int *resSampleStIndex, unsigned int *resSampleNum, int size, int *resBufferIndex); 

extern void freeStIndexAndNum(unsigned int *resSampleStIndex, unsigned int *resSampleNum);

extern void printProgress(int current, int full);

//--------------------------------------------------------------------------------------------
//	kernels and functions for offsetting without normal vectors

extern __global__ void krLDNIOffsetting_InRays(float *inputDepthArrayPtr, unsigned int *inputIndexArrayPtr,
								float offset, int arrsize, short nAxis,
								float *resDepthBuffer, int *resBufferIndex,
								unsigned int *resSampleStIndex, unsigned int *resSampleNum);

extern __global__ void krLDNIOffsetting_ByParallelRays(float *inputDepthArrayPtr, unsigned int *inputIndexArrayPtr,
								float *resDepthBuffer, int *resBufferIndex,	unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								int *devRaysToBeSearched, int devNumOfSearchedRays, 
								short nAxis, int res, float gwidth, float offset, int stIndex, int edIndex, 
								float *resDepthBuffer2, int *resBufferIndex2, unsigned int *resSampleStIndex2, unsigned int *resSampleNum2);

extern __global__ void krLDNIOffsetting_OnlyCopyAsNotEffectedBySamplesOnPerpendicularRays(unsigned int *orderOfRayProcessing, unsigned int nonzeroStIndex, 
								float *resDepthBuffer, int *resBufferIndex, unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								short nAxis, int stIndex, int edIndex, 
								float *resDepthBuffer2, int *resBufferIndex2, unsigned int *resSampleStIndex2, unsigned int *resSampleNum2);

extern __global__ void krLDNIOffsetting_BySpatialHashingSamplesOnPerpendicularRays(float *hashElementsData, unsigned int *hashTableIndexArray,
								int *devSHBoxToBeSearched, int devNumOfSearchedSHBox, unsigned int *orderOfRayProcessing, unsigned int nonzeroStIndex, 
								float *resDepthBuffer, int *resBufferIndex,	unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								short nAxis, int res, float gwidth, float offset, int stIndex, int edIndex, 
								float *resDepthBuffer2, int *resBufferIndex2, unsigned int *resSampleStIndex2, unsigned int *resSampleNum2);

extern __global__ void krLDNIOffsetting_BySpatialHashingSamplesOnPerpendicularRays(float *hashElementsData, unsigned int *hashTableIndexArray,
								int *devSHBoxToBeSearched, int devNumOfSearchedSHBox,
								float *resDepthBuffer, int *resBufferIndex, unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								short nAxis, int res, float gwidth, float offset, int stIndex, int edIndex,
								float *resDepthBuffer2, int *resBufferIndex2, unsigned int *resSampleStIndex2, unsigned int *resSampleNum2);

extern __global__ void krLDNIOffsetting_CollectResult(float *resDepthBuffer, unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								float *outputNxArrayPtr, float *outputNzArrayPtr, float *outputDepthArrayPtr, unsigned int *outputIndexArrayPtr, 
								int arrsize, int nAxis);

void initResBuffer(float *&resDepthBuffer, int *&resBufferIndex) 
{
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(resBufferIndex), sizeof(int) ) );	
	CUDA_SAFE_CALL( cudaMemset( (void*)(resBufferIndex), 0, sizeof(int) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(resDepthBuffer), BUFFER_SIZE*sizeof(float) ) );	
}

void freeResBuffer(float *resDepthBuffer, int *resBufferIndex) 
{
	cudaFree(resDepthBuffer);	cudaFree(resBufferIndex);
}


//--------------------------------------------------------------------------------------------
//	The class member functions
//--------------------------------------------------------------------------------------------

void LDNIcudaOperation::SolidQuickSuccessiveOffsettingBySpatialHashing(LDNIcudaSolid* inputSolid, LDNIcudaSolid* &newSolid, float offset, bool bWithNormal)
{
	bool bGrownOrShrink;	float boundingBox[6],origin[3],gwidth;		int res;
	int i,nStep;
	float offsetStep=5.0f,segOffsetValue;		

	//--------------------------------------------------------------------------------------------------------
	//	Preparation
	bGrownOrShrink=true;	gwidth=inputSolid->GetSampleWidth();
	if (offset>0.0f) { // growing the working space of inputSolid 
		float dist=_distanceToBoundBoxBoundary(inputSolid);
		//printf("Minimal distance is: %f\n",dist);
		if (dist<1.5*offset) {
			inputSolid->GetOrigin(origin[0],origin[1],origin[2]);	res=inputSolid->GetResolution();
			boundingBox[0]=origin[0]-gwidth*0.5f-offset;		boundingBox[1]=origin[0]+gwidth*(float)res+offset;
			boundingBox[2]=origin[1]-gwidth*0.5f-offset;		boundingBox[3]=origin[1]+gwidth*(float)res+offset;
			boundingBox[4]=origin[2]-gwidth*0.5f-offset;		boundingBox[5]=origin[2]+gwidth*(float)res+offset;
			_expansionLDNIcudaSolidByNewBoundingBox(inputSolid, boundingBox);
		}
	}
	else
		bGrownOrShrink=false;
	offset=fabs(offset);
	offsetStep=offsetStep*gwidth;	
	nStep=(int)ceil(offset/offsetStep);

	LDNIcudaSolid* solidA=inputSolid;
	for(i=1;i<=nStep;i++) {
		segOffsetValue=offsetStep;
		if (offsetStep*(float)i>offset) segOffsetValue=offset-offsetStep*(float)(i-1);

		if (bGrownOrShrink) {
			if (bWithNormal)
				SolidOffsettingBySpatialHashing(solidA,newSolid,segOffsetValue,true);
			else
				SolidOffsettingWithoutNormal(solidA,newSolid,segOffsetValue,true);
		}
		else {
			if (bWithNormal)
				SolidOffsettingBySpatialHashing(solidA,newSolid,-segOffsetValue,true);
			else
				SolidOffsettingWithoutNormal(solidA,newSolid,-segOffsetValue,true);
		}

		if (i!=1) delete solidA;
		solidA=newSolid;	newSolid=NULL;
	}
	newSolid=solidA;
}

void LDNIcudaOperation::SolidOffsettingBySpatialHashing(LDNIcudaSolid* inputSolid, LDNIcudaSolid* &newSolid, float offset, bool bWithRayPacking)
{
	bool bGrownOrShrink;	float boundingBox[6],origin[3],gwidth;		int i,j,k,res,arrsize,ll;		short nAxis;
	cudaEvent_t     startClock, stopClock;		float elapsedTime;
	CUDA_SAFE_CALL( cudaEventCreate( &startClock ) );
	CUDA_SAFE_CALL( cudaEventCreate( &stopClock ) );
	int *raysToBeSearched;		int discreteSupport;
	int *devRaysToBeSearched;	int devNumOfSearchedRays=0;
	int *devSHBoxToBeSearched;	int devNumOfSearchedSHBox=0;

	//--------------------------------------------------------------------------------------------------------
	//	Preparation
	bGrownOrShrink=true;	gwidth=inputSolid->GetSampleWidth();
	if (offset<0.0f) {
		bGrownOrShrink=false; offset=fabs(offset);
	}
	else {	// growing the working space of inputSolid 
		float dist=_distanceToBoundBoxBoundary(inputSolid);
		//printf("Minimal distance is: %f\n",dist);
		if (dist<1.5*offset) {
			inputSolid->GetOrigin(origin[0],origin[1],origin[2]);	res=inputSolid->GetResolution();
			boundingBox[0]=origin[0]-gwidth*0.5f-offset;		boundingBox[1]=origin[0]+gwidth*(float)res+offset;
			boundingBox[2]=origin[1]-gwidth*0.5f-offset;		boundingBox[3]=origin[1]+gwidth*(float)res+offset;
			boundingBox[4]=origin[2]-gwidth*0.5f-offset;		boundingBox[5]=origin[2]+gwidth*(float)res+offset;
			_expansionLDNIcudaSolidByNewBoundingBox(inputSolid, boundingBox);
		}
	}
	//--------------------------------------------------------------------------------------------------------
	//	to determine the rays to be searched
	discreteSupport=(int)(ceil(offset/gwidth))+1;	raysToBeSearched=(int*)malloc((discreteSupport*2+1)*(discreteSupport*2+1)*4*sizeof(int));
	devNumOfSearchedRays=0;		float offsetSQR=offset*offset;
	for(k=1;k<=discreteSupport;k++) {	
		for(j=-k;j<=k;j++) {
			for(i=-k;i<=k;i++) {
				if (i==0 && j==0) continue;
				ll=(int)(MAX(fabs((float)i),fabs((float)j)));	if (ll!=k) continue;
				if ((float)(i*i+j*j)*gwidth*gwidth>offsetSQR) continue;
				raysToBeSearched[devNumOfSearchedRays*2]=i;
				raysToBeSearched[devNumOfSearchedRays*2+1]=j;	devNumOfSearchedRays++;
			}
		}
	}
	if (devNumOfSearchedRays>0) {
		printf("Totally %d rays need to be searched for each ray: [-%d,%d]\n",devNumOfSearchedRays,discreteSupport,discreteSupport);
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(devRaysToBeSearched), devNumOfSearchedRays*2*sizeof(int) ) );	
		CUDA_SAFE_CALL( cudaMemcpy( devRaysToBeSearched, raysToBeSearched, devNumOfSearchedRays*2*sizeof(int),  cudaMemcpyHostToDevice ) );
//		for(i=0;i<devNumOfSearchedRays;i++) printf("%d-th: (%d,%d)\n",i,raysToBeSearched[i*2],raysToBeSearched[i*2+1]);
	}
	free(raysToBeSearched);
	//------------------------------------------------------------ --------------------------------------------
	//	to determine the spatial hashing boxes to be searched
	discreteSupport=(int)(ceil(offset/gwidth+1.415f));	 raysToBeSearched=(int*)malloc((discreteSupport*2+1)*(discreteSupport*2+1)*4*sizeof(int));
	devNumOfSearchedSHBox=0;	float offsetSQR2=(offset+1.415f*gwidth)*(offset+1.415f*gwidth);
	for(k=0;k<=discreteSupport;k++) {	//k=discreteSupport;
		for(j=-k;j<=k;j++) {
			for(i=-k;i<=k;i++) {
				ll=(int)(MAX(fabs((float)i),fabs((float)j)));	if (ll!=k) continue;
				if ((float)(i*i+j*j)*gwidth*gwidth>offsetSQR2) continue;
				raysToBeSearched[devNumOfSearchedSHBox*2]=i;
				raysToBeSearched[devNumOfSearchedSHBox*2+1]=j;	devNumOfSearchedSHBox++;
			}
		}
	}
	if (devNumOfSearchedSHBox>0) {
		printf("Totally %d spatial hashing bins need to be searched for each ray: [-%d,%d]\n",devNumOfSearchedSHBox,discreteSupport,discreteSupport);
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(devSHBoxToBeSearched), devNumOfSearchedSHBox*2*sizeof(int) ) );	
		CUDA_SAFE_CALL( cudaMemcpy( devSHBoxToBeSearched, raysToBeSearched, devNumOfSearchedSHBox*2*sizeof(int),  cudaMemcpyHostToDevice ) );
//		for(i=0;i<devNumOfSearchedSHBox;i++) printf("%d-th: (%d,%d)\n",i,raysToBeSearched[i*2],raysToBeSearched[i*2+1]);
	}
	free(raysToBeSearched);
	//--------------------------------------------------------------------------------------------------------
	newSolid=new LDNIcudaSolid;			newSolid->SetSampleWidth(gwidth);
	res=inputSolid->GetResolution();	newSolid->MallocMemory(res);		arrsize=res*res;
	inputSolid->GetOrigin(origin[0],origin[1],origin[2]);	
	newSolid->SetOrigin(origin[0],origin[1],origin[2]);
	//--------------------------------------------------------------------------------------------------------
	//	Estimate the size of computation
	int sizeOfComputation=(inputSolid->GetSampleNumber())*(discreteSupport+discreteSupport+1)/(3*res);
		//	actually "(inputSolid->GetSampleNumber())/(3*res*res)*((discreteSupport+discreteSupport+1)*res)"
	sizeOfComputation=(int)((float)sizeOfComputation*0.001f+0.5f)*2+1;	sizeOfComputation=sizeOfComputation/2;
	int segmentSize=(int)(ceil(((float)arrsize/(float)sizeOfComputation)/(float)(BLOCKS_PER_GRID*THREADS_PER_BLOCK)))*BLOCKS_PER_GRID*THREADS_PER_BLOCK;
	if (segmentSize==0) {sizeOfComputation=1; segmentSize=arrsize;}
	printf("\nThe computation is split into %d segments with size=%d.\n\n",sizeOfComputation,segmentSize);
	//--------------------------------------------------------------------------------------------------------
	unsigned int *devScanArrayResPtr;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&devScanArrayResPtr, (arrsize+1)*sizeof(unsigned int) ) );
	//--------------------------------------------------------------------------------------------------------
	float *resNxBuffer,*resNyBuffer,*resDepthBuffer;	int *resBufferIndex;
	float *resNxBuffer2,*resNyBuffer2,*resDepthBuffer2;	int *resBufferIndex2;
	unsigned int *resSampleStIndex;		unsigned int *resSampleNum;
	unsigned int *resSampleStIndex2;	unsigned int *resSampleNum2;
	initStIndexAndNum(resSampleStIndex,resSampleNum,arrsize+1);
	initResBuffer(resNxBuffer,resNyBuffer,resDepthBuffer,resBufferIndex);
	initStIndexAndNum(resSampleStIndex2,resSampleNum2,arrsize+1);
	initResBuffer(resNxBuffer2,resNyBuffer2,resDepthBuffer2,resBufferIndex2);

	for(nAxis=0;nAxis<3;nAxis++) {
		int sampleNum=inputSolid->GetSampleNumber(nAxis);	printf("The input sample num: %d\n",sampleNum);

		//--------------------------------------------------------------------------------------------------------
		//	Step 1: Processing the samples on the same ray
		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
		//--------------------------------------------------------------------------------------------------------
		cleanUpStIndexAndNum(resSampleStIndex, resSampleNum, arrsize+1, resBufferIndex);	// clean up buffer-1 to use as output
		krLDNIOffsetting_InRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(inputSolid->GetSampleDepthArrayPtr(nAxis),
											inputSolid->GetIndexArrayPtr(nAxis),offset,arrsize,nAxis,
											resNxBuffer,resNyBuffer,resDepthBuffer,resBufferIndex,resSampleStIndex,resSampleNum);
		//--------------------------------------------------------------------------------------------------------
		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
		printf("The first phase of CUDA-LDNI offsetting (%d-direction) takes: %3.1f (ms)\n",(int)nAxis,elapsedTime);
		
		//--------------------------------------------------------------------------------------------------------
		//	Step 2: Processing the samples on the other rays (but in the same direction)
		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
//		cudaStream_t    streamParaRays;
//		CUDA_SAFE_CALL( cudaStreamCreate( &streamParaRays ) );
		//--------------------------------------------------------------------------------------------------------
		cleanUpStIndexAndNum(resSampleStIndex2, resSampleNum2, arrsize+1, resBufferIndex2);	// clean up buffer-2 to use as output
		for(int stIndex=0;stIndex<arrsize;stIndex+=segmentSize) {
			int sizeToProcess;
			if ((stIndex+segmentSize)<arrsize) sizeToProcess=segmentSize; else sizeToProcess=arrsize-stIndex;
			krLDNIOffsetting_ByParallelRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
						inputSolid->GetSampleDepthArrayPtr(nAxis),inputSolid->GetIndexArrayPtr(nAxis),
						resNxBuffer,resNyBuffer,resDepthBuffer,resBufferIndex,resSampleStIndex,resSampleNum,
						devRaysToBeSearched, devNumOfSearchedRays, nAxis, res, gwidth, offset, stIndex, stIndex+sizeToProcess, 
						resNxBuffer2,resNyBuffer2,resDepthBuffer2,resBufferIndex2,resSampleStIndex2,resSampleNum2);
//			CUDA_SAFE_CALL( cudaStreamSynchronize( streamParaRays ) );
			printProgress(stIndex,arrsize);		//	printf(".");
		}
		printf("\b\b");		//CUDA_SAFE_CALL( cudaStreamDestroy( streamParaRays ) );
		//--------------------------------------------------------------------------------------------------------
		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
//		CUDA_SAFE_CALL( cudaMemcpy( &sampleNum, resBufferIndex2, sizeof(int), cudaMemcpyDeviceToHost ) );	printf("The resultant sample num: %d\n",sampleNum);
		printf("The second phase of CUDA-LDNI offsetting (%d-direction) takes: %3.1f (ms)\n",(int)nAxis,elapsedTime);

		//--------------------------------------------------------------------------------------------------------
		//	Step 3: Building a spatial harshing by the samples on the other rays (in different directions)
		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
		//--------------------------------------------------------------------------------------------------------
		unsigned int *hashTableIndexArray;			int hashTableElementNum;	float *hashElementsData;
		_buildHashingTableForSamplesOnPerpendicularRays(inputSolid,nAxis,hashTableElementNum,hashTableIndexArray,hashElementsData);
		//--------------------------------------------------------------------------------------------------------
		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
		printf("The step of building spatial hashing table takes: %3.1f (ms)\n",elapsedTime);

		//--------------------------------------------------------------------------------------------------------
		//	Step 4: Packing rays by the number of merging times
		unsigned int nonzeroStIndex;
		unsigned int *orderOfRayProcessing;
if (bWithRayPacking) {
		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
		//--------------------------------------------------------------------------------------------------------
		_sortingRaysByPossibleMergingGivenBySpatialHashingSamples(hashElementsData,hashTableIndexArray,
						devSHBoxToBeSearched,devNumOfSearchedSHBox,nAxis,res,gwidth,offset,orderOfRayProcessing,nonzeroStIndex);
		//--------------------------------------------------------------------------------------------------------
		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
		printf("The step of packaging rays by working load estimation takes: %3.1f (ms)\n",elapsedTime);
}

		//--------------------------------------------------------------------------------------------------------
		//	Step 5: Processing the samples in the spatial harshing 
		//		(i.e., the samples on the other rays in different directions)
		cudaStream_t    streamPerpendicularRays;
		CUDA_SAFE_CALL( cudaStreamCreate( &streamPerpendicularRays ) );
		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
		//--------------------------------------------------------------------------------------------------------
		cleanUpStIndexAndNum(resSampleStIndex, resSampleNum, arrsize+1, resBufferIndex);	// clean up buffer-1 to use as output
if (bWithRayPacking) {
		krLDNIOffsetting_OnlyCopyAsNotEffectedBySamplesOnPerpendicularRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
								orderOfRayProcessing, nonzeroStIndex, 
								resNxBuffer2,resNyBuffer2,resDepthBuffer2,resBufferIndex2,resSampleStIndex2,resSampleNum2,
								nAxis, 0, nonzeroStIndex, 
								resNxBuffer,resNyBuffer,resDepthBuffer,resBufferIndex,resSampleStIndex,resSampleNum);
		printf("stIndex=%d   segmentSize=%d   arrsize=%d  devNumOfSearchedSHBox=%d\n",nonzeroStIndex,segmentSize,arrsize,devNumOfSearchedSHBox); 
		for(int stIndex=nonzeroStIndex;stIndex<arrsize;stIndex+=segmentSize) {
			int sizeToProcess;
			if ((stIndex+segmentSize)<arrsize) sizeToProcess=segmentSize; else sizeToProcess=arrsize-stIndex;
			krLDNIOffsetting_BySpatialHashingSamplesOnPerpendicularRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK,0,streamPerpendicularRays>>>(
								hashElementsData,hashTableIndexArray,
								devSHBoxToBeSearched, devNumOfSearchedSHBox, orderOfRayProcessing, nonzeroStIndex,
								resNxBuffer2,resNyBuffer2,resDepthBuffer2,resBufferIndex2,resSampleStIndex2,resSampleNum2,
								nAxis, res, gwidth, offset, stIndex, stIndex+sizeToProcess,
								resNxBuffer,resNyBuffer,resDepthBuffer,resBufferIndex,resSampleStIndex,resSampleNum);
			CUDA_SAFE_CALL( cudaStreamSynchronize( streamPerpendicularRays ) );
			printProgress(stIndex,arrsize);		//	printf(".");
		}
}
else {
		for(int stIndex=0;stIndex<arrsize;stIndex+=segmentSize) {
			int sizeToProcess;
			if ((stIndex+segmentSize)<arrsize) sizeToProcess=segmentSize; else sizeToProcess=arrsize-stIndex;
			krLDNIOffsetting_BySpatialHashingSamplesOnPerpendicularRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK,0,streamPerpendicularRays>>>(
								hashElementsData,hashTableIndexArray,
								devSHBoxToBeSearched, devNumOfSearchedSHBox,
								resNxBuffer2,resNyBuffer2,resDepthBuffer2,resBufferIndex2,resSampleStIndex2,resSampleNum2,
								nAxis, res, gwidth, offset, stIndex, stIndex+sizeToProcess,
								resNxBuffer,resNyBuffer,resDepthBuffer,resBufferIndex,resSampleStIndex,resSampleNum);
			CUDA_SAFE_CALL( cudaStreamSynchronize( streamPerpendicularRays ) );
			printProgress(stIndex,arrsize);		//	printf(".");
		}
}
		printf("\b\b");		CUDA_SAFE_CALL( cudaStreamDestroy( streamPerpendicularRays ) );
		//--------------------------------------------------------------------------------------------------------
		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
//		CUDA_SAFE_CALL( cudaMemcpy( &sampleNum, resBufferIndex2, sizeof(int), cudaMemcpyDeviceToHost ) );	printf("The resultant sample num: %d\n",sampleNum);
		printf("The third phase of CUDA-LDNI offsetting (%d-direction) takes: %3.1f (ms)\n",(int)nAxis,elapsedTime);
if (bWithRayPacking) {
		cudaFree(orderOfRayProcessing);
}

		//--------------------------------------------------------------------------------------------------------
		//	Step 6: Compaction the samples on the newly generated rays
//		CUDA_SAFE_CALL( cudaMemcpy( devScanArrayResPtr, resSampleNum2, (arrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( devScanArrayResPtr, resSampleNum, (arrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToDevice ) );
		//--------------------------------------------------------------------------------------------------------
		thrust::device_ptr<unsigned int> dev_ptr(devScanArrayResPtr);	// Wrap raw pointers with dev_ptr
		thrust::exclusive_scan(dev_ptr, dev_ptr+(arrsize+1), dev_ptr);	// in-place scan
		sampleNum=dev_ptr[arrsize];
		printf("The resultant sample num: %d\n\n",sampleNum);
		//--------------------------------------------------------------------------------------------------------
		newSolid->MallocSampleMemory(nAxis,sampleNum);
		CUDA_SAFE_CALL( cudaMemcpy( newSolid->GetIndexArrayPtr(nAxis), devScanArrayResPtr, (arrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToDevice ) );
		krLDNIOffsetting_CollectResult<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(//resNxBuffer2,resNyBuffer2,resDepthBuffer2,resSampleStIndex2,resSampleNum2,
												resNxBuffer,resNyBuffer,resDepthBuffer,resSampleStIndex,resSampleNum,
												newSolid->GetSampleNxArrayPtr(nAxis),newSolid->GetSampleNyArrayPtr(nAxis),
												newSolid->GetSampleDepthArrayPtr(nAxis),newSolid->GetIndexArrayPtr(nAxis),arrsize);

		//--------------------------------------------------------------------------------------------------------
		//	Step 7: Release the temporary used memory
		cudaFree(hashTableIndexArray);		cudaFree(hashElementsData);		
	}

	//--------------------------------------------------------------------------------------------------------
	//	Free the memory
	freeResBuffer(resNxBuffer,resNyBuffer,resDepthBuffer,resBufferIndex);
	freeStIndexAndNum(resSampleStIndex,resSampleNum);
	freeResBuffer(resNxBuffer2,resNyBuffer2,resDepthBuffer2,resBufferIndex2);
	freeStIndexAndNum(resSampleStIndex2,resSampleNum2);
	if (devNumOfSearchedRays>0) cudaFree(devRaysToBeSearched);
	//--------------------------------------------------------------------------------------------------------
	cudaFree(devScanArrayResPtr);
	//--------------------------------------------------------------------------------------------------------
	CUDA_SAFE_CALL( cudaEventDestroy( startClock ) );
	CUDA_SAFE_CALL( cudaEventDestroy( stopClock ) );

	//--------------------------------------------------------------------------------------------------------
	//	Boolean operation to obtain the final result
//	return;
	if (bGrownOrShrink) {
		_booleanOperation(newSolid, inputSolid, 0);
	}
	else {
		_booleanOperation(inputSolid, newSolid, 2);
		_switchSolid(inputSolid, newSolid);
	}
}

void LDNIcudaOperation::SolidOffsettingWithoutNormal(LDNIcudaSolid* inputSolid, LDNIcudaSolid* &newSolid, float offset, bool bWithRayPacking)
{
	bool bGrownOrShrink;	float boundingBox[6],origin[3],gwidth;		int i,j,k,res,arrsize,ll;		short nAxis;
	cudaEvent_t     startClock, stopClock;		float elapsedTime;
	CUDA_SAFE_CALL( cudaEventCreate( &startClock ) );
	CUDA_SAFE_CALL( cudaEventCreate( &stopClock ) );
	int *raysToBeSearched;		int discreteSupport;
	int *devRaysToBeSearched;	int devNumOfSearchedRays=0;
	int *devSHBoxToBeSearched;	int devNumOfSearchedSHBox=0;

	//--------------------------------------------------------------------------------------------------------
	//	Preparation
	bGrownOrShrink=true;	gwidth=inputSolid->GetSampleWidth();
	if (offset<0.0f) {
		bGrownOrShrink=false; offset=fabs(offset);
	}
	else {	// growing the working space of inputSolid 
		float dist=_distanceToBoundBoxBoundary(inputSolid);
		//printf("Minimal distance is: %f\n",dist);
		if (dist<1.5*offset) {
			inputSolid->GetOrigin(origin[0],origin[1],origin[2]);	res=inputSolid->GetResolution();
			boundingBox[0]=origin[0]-gwidth*0.5f-offset;		boundingBox[1]=origin[0]+gwidth*(float)res+offset;
			boundingBox[2]=origin[1]-gwidth*0.5f-offset;		boundingBox[3]=origin[1]+gwidth*(float)res+offset;
			boundingBox[4]=origin[2]-gwidth*0.5f-offset;		boundingBox[5]=origin[2]+gwidth*(float)res+offset;
			_expansionLDNIcudaSolidByNewBoundingBox(inputSolid, boundingBox);
		}
	}
	//--------------------------------------------------------------------------------------------------------
	//	to determine the rays to be searched
	discreteSupport=(int)(ceil(offset/gwidth))+1;	raysToBeSearched=(int*)malloc((discreteSupport*2+1)*(discreteSupport*2+1)*4*sizeof(int));
	devNumOfSearchedRays=0;		float offsetSQR=offset*offset;
	for(k=1;k<=discreteSupport;k++) {	
		for(j=-k;j<=k;j++) {
			for(i=-k;i<=k;i++) {
				if (i==0 && j==0) continue;
				ll=(int)(MAX(fabs((float)i),fabs((float)j)));	if (ll!=k) continue;
				if ((float)(i*i+j*j)*gwidth*gwidth>offsetSQR) continue;
				raysToBeSearched[devNumOfSearchedRays*2]=i;
				raysToBeSearched[devNumOfSearchedRays*2+1]=j;	devNumOfSearchedRays++;
			}
		}
	}
	if (devNumOfSearchedRays>0) {
		printf("Totally %d rays need to be searched for each ray: [-%d,%d]\n",devNumOfSearchedRays,discreteSupport,discreteSupport);
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(devRaysToBeSearched), devNumOfSearchedRays*2*sizeof(int) ) );	
		CUDA_SAFE_CALL( cudaMemcpy( devRaysToBeSearched, raysToBeSearched, devNumOfSearchedRays*2*sizeof(int),  cudaMemcpyHostToDevice ) );
//		for(i=0;i<devNumOfSearchedRays;i++) printf("%d-th: (%d,%d)\n",i,raysToBeSearched[i*2],raysToBeSearched[i*2+1]);
	}
	free(raysToBeSearched);
	//------------------------------------------------------------ --------------------------------------------
	//	to determine the spatial hashing boxes to be searched
	discreteSupport=(int)(ceil(offset/gwidth+1.415f));	 raysToBeSearched=(int*)malloc((discreteSupport*2+1)*(discreteSupport*2+1)*4*sizeof(int));
	devNumOfSearchedSHBox=0;	float offsetSQR2=(offset+1.415f*gwidth)*(offset+1.415f*gwidth);
	for(k=0;k<=discreteSupport;k++) {	//k=discreteSupport;
		for(j=-k;j<=k;j++) {
			for(i=-k;i<=k;i++) {
				ll=(int)(MAX(fabs((float)i),fabs((float)j)));	if (ll!=k) continue;
				if ((float)(i*i+j*j)*gwidth*gwidth>offsetSQR2) continue;
				raysToBeSearched[devNumOfSearchedSHBox*2]=i;
				raysToBeSearched[devNumOfSearchedSHBox*2+1]=j;	devNumOfSearchedSHBox++;
			}
		}
	}
	if (devNumOfSearchedSHBox>0) {
		printf("Totally %d spatial hashing bins need to be searched for each ray: [-%d,%d]\n",devNumOfSearchedSHBox,discreteSupport,discreteSupport);
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(devSHBoxToBeSearched), devNumOfSearchedSHBox*2*sizeof(int) ) );	
		CUDA_SAFE_CALL( cudaMemcpy( devSHBoxToBeSearched, raysToBeSearched, devNumOfSearchedSHBox*2*sizeof(int),  cudaMemcpyHostToDevice ) );
//		for(i=0;i<devNumOfSearchedSHBox;i++) printf("%d-th: (%d,%d)\n",i,raysToBeSearched[i*2],raysToBeSearched[i*2+1]);
	}
	free(raysToBeSearched);
	//--------------------------------------------------------------------------------------------------------
	newSolid=new LDNIcudaSolid;			newSolid->SetSampleWidth(gwidth);
	res=inputSolid->GetResolution();	newSolid->MallocMemory(res);		arrsize=res*res;
	inputSolid->GetOrigin(origin[0],origin[1],origin[2]);	
	newSolid->SetOrigin(origin[0],origin[1],origin[2]);
	//--------------------------------------------------------------------------------------------------------
	//	Estimate the size of computation
	int sizeOfComputation=(inputSolid->GetSampleNumber())*(discreteSupport+discreteSupport+1)/(3*res);
		//	actually "(inputSolid->GetSampleNumber())/(3*res*res)*((discreteSupport+discreteSupport+1)*res)"
	sizeOfComputation=(int)((float)sizeOfComputation*0.001f+0.5f)*2+1;	sizeOfComputation=sizeOfComputation/2;
	int segmentSize=(int)(ceil(((float)arrsize/(float)sizeOfComputation)/(float)(BLOCKS_PER_GRID*THREADS_PER_BLOCK)))*BLOCKS_PER_GRID*THREADS_PER_BLOCK;
	segmentSize=10*BLOCKS_PER_GRID*THREADS_PER_BLOCK;
	if (segmentSize==0) {sizeOfComputation=1; segmentSize=arrsize;}
	printf("\nThe computation is split into %d segments with size=%d.\n\n",sizeOfComputation,segmentSize);
	//--------------------------------------------------------------------------------------------------------
	unsigned int *devScanArrayResPtr;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&devScanArrayResPtr, (arrsize+1)*sizeof(unsigned int) ) );
	//--------------------------------------------------------------------------------------------------------
	float *resDepthBuffer;	int *resBufferIndex;	float *resDepthBuffer2;	int *resBufferIndex2;
	unsigned int *resSampleStIndex;		unsigned int *resSampleNum;
	unsigned int *resSampleStIndex2;	unsigned int *resSampleNum2;
	initStIndexAndNum(resSampleStIndex,resSampleNum,arrsize+1);
	initResBuffer(resDepthBuffer,resBufferIndex);
	initStIndexAndNum(resSampleStIndex2,resSampleNum2,arrsize+1);
	initResBuffer(resDepthBuffer2,resBufferIndex2);

	for(nAxis=0;nAxis<3;nAxis++) {
		int sampleNum=inputSolid->GetSampleNumber(nAxis);	printf("The input sample num: %d\n",sampleNum);

		//--------------------------------------------------------------------------------------------------------
		//	Step 1: Processing the samples on the same ray
		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
		//--------------------------------------------------------------------------------------------------------
		cleanUpStIndexAndNum(resSampleStIndex, resSampleNum, arrsize+1, resBufferIndex);	// clean up buffer-1 to use as output
		krLDNIOffsetting_InRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(inputSolid->GetSampleDepthArrayPtr(nAxis),
											inputSolid->GetIndexArrayPtr(nAxis),offset,arrsize,nAxis,
											resDepthBuffer,resBufferIndex,resSampleStIndex,resSampleNum);
		//--------------------------------------------------------------------------------------------------------
		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
		printf("The first phase of CUDA-LDNI offsetting (%d-direction) takes: %3.1f (ms)\n",(int)nAxis,elapsedTime);
		
		//--------------------------------------------------------------------------------------------------------
		//	Step 2: Processing the samples on the other rays (but in the same direction)
		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
//		cudaStream_t    streamParaRays;
//		CUDA_SAFE_CALL( cudaStreamCreate( &streamParaRays ) );
		//--------------------------------------------------------------------------------------------------------
		cleanUpStIndexAndNum(resSampleStIndex2, resSampleNum2, arrsize+1, resBufferIndex2);	// clean up buffer-2 to use as output
		for(int stIndex=0;stIndex<arrsize;stIndex+=segmentSize) {
			int sizeToProcess;
			if ((stIndex+segmentSize)<arrsize) sizeToProcess=segmentSize; else sizeToProcess=arrsize-stIndex;
			krLDNIOffsetting_ByParallelRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
						inputSolid->GetSampleDepthArrayPtr(nAxis),inputSolid->GetIndexArrayPtr(nAxis),
						resDepthBuffer,resBufferIndex,resSampleStIndex,resSampleNum,
						devRaysToBeSearched, devNumOfSearchedRays, nAxis, res, gwidth, offset, stIndex, stIndex+sizeToProcess, 
						resDepthBuffer2,resBufferIndex2,resSampleStIndex2,resSampleNum2);
//			CUDA_SAFE_CALL( cudaStreamSynchronize( streamParaRays ) );
			printProgress(stIndex,arrsize);		//	printf(".");
		}
		printf("\b\b");		//CUDA_SAFE_CALL( cudaStreamDestroy( streamParaRays ) );
		//--------------------------------------------------------------------------------------------------------
		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
//		CUDA_SAFE_CALL( cudaMemcpy( &sampleNum, resBufferIndex2, sizeof(int), cudaMemcpyDeviceToHost ) );	printf("The resultant sample num: %d\n",sampleNum);
		printf("The second phase of CUDA-LDNI offsetting (%d-direction) takes: %3.1f (ms)\n",(int)nAxis,elapsedTime);

		//--------------------------------------------------------------------------------------------------------
		//	Step 3: Building a spatial harshing by the samples on the other rays (in different directions)
		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
		//--------------------------------------------------------------------------------------------------------
		unsigned int *hashTableIndexArray;			int hashTableElementNum;	float *hashElementsData;
		_buildHashingTableForSamplesOnPerpendicularRays(inputSolid,nAxis,hashTableElementNum,hashTableIndexArray,hashElementsData);
		//--------------------------------------------------------------------------------------------------------
		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
		printf("The step of building spatial hashing table takes: %3.1f (ms)\n",elapsedTime);

		//--------------------------------------------------------------------------------------------------------
		//	Step 4: Packing rays by the number of merging times
		unsigned int nonzeroStIndex;
		unsigned int *orderOfRayProcessing;
		if (bWithRayPacking) {
			CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
			//--------------------------------------------------------------------------------------------------------
			_sortingRaysByPossibleMergingGivenBySpatialHashingSamples(hashElementsData,hashTableIndexArray,
							devSHBoxToBeSearched,devNumOfSearchedSHBox,nAxis,res,gwidth,offset,orderOfRayProcessing,nonzeroStIndex);
			//--------------------------------------------------------------------------------------------------------
			CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
			CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
			printf("The step of packaging rays by working load estimation takes: %3.1f (ms)\n",elapsedTime);
		}

		//--------------------------------------------------------------------------------------------------------
		//	Step 5: Processing the samples in the spatial harshing 
		//		(i.e., the samples on the other rays in different directions)
		cudaStream_t    streamPerpendicularRays;
		CUDA_SAFE_CALL( cudaStreamCreate( &streamPerpendicularRays ) );
		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
		//--------------------------------------------------------------------------------------------------------
		cleanUpStIndexAndNum(resSampleStIndex, resSampleNum, arrsize+1, resBufferIndex);	// clean up buffer-1 to use as output
		if (bWithRayPacking) {
			krLDNIOffsetting_OnlyCopyAsNotEffectedBySamplesOnPerpendicularRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
									orderOfRayProcessing, nonzeroStIndex, 
									resDepthBuffer2,resBufferIndex2,resSampleStIndex2,resSampleNum2,
									nAxis, 0, nonzeroStIndex, 
									resDepthBuffer,resBufferIndex,resSampleStIndex,resSampleNum);
			printf("stIndex=%d   segmentSize=%d   arrsize=%d  devNumOfSearchedSHBox=%d\n",nonzeroStIndex,segmentSize,arrsize,devNumOfSearchedSHBox); 
			for(int stIndex=nonzeroStIndex;stIndex<arrsize;stIndex+=segmentSize) {
				int sizeToProcess;
				if ((stIndex+segmentSize)<arrsize) sizeToProcess=segmentSize; else sizeToProcess=arrsize-stIndex;
				krLDNIOffsetting_BySpatialHashingSamplesOnPerpendicularRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK,0,streamPerpendicularRays>>>(
									hashElementsData,hashTableIndexArray,
									devSHBoxToBeSearched, devNumOfSearchedSHBox, orderOfRayProcessing, nonzeroStIndex,
									resDepthBuffer2,resBufferIndex2,resSampleStIndex2,resSampleNum2,
									nAxis, res, gwidth, offset, stIndex, stIndex+sizeToProcess,
									resDepthBuffer,resBufferIndex,resSampleStIndex,resSampleNum);
				CUDA_SAFE_CALL( cudaStreamSynchronize( streamPerpendicularRays ) );
				printProgress(stIndex,arrsize);		//	printf(".");
			}
		}
		else {
			for(int stIndex=0;stIndex<arrsize;stIndex+=segmentSize) {
				int sizeToProcess;
				if ((stIndex+segmentSize)<arrsize) sizeToProcess=segmentSize; else sizeToProcess=arrsize-stIndex;
				krLDNIOffsetting_BySpatialHashingSamplesOnPerpendicularRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK,0,streamPerpendicularRays>>>(
									hashElementsData,hashTableIndexArray, devSHBoxToBeSearched, devNumOfSearchedSHBox,
									resDepthBuffer2,resBufferIndex2,resSampleStIndex2,resSampleNum2,
									nAxis, res, gwidth, offset, stIndex, stIndex+sizeToProcess,
									resDepthBuffer,resBufferIndex,resSampleStIndex,resSampleNum);
				CUDA_SAFE_CALL( cudaStreamSynchronize( streamPerpendicularRays ) );
				printProgress(stIndex,arrsize);		//	printf(".");
			}
		}
		printf("\b\b");		CUDA_SAFE_CALL( cudaStreamDestroy( streamPerpendicularRays ) );
		//--------------------------------------------------------------------------------------------------------
		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
//		CUDA_SAFE_CALL( cudaMemcpy( &sampleNum, resBufferIndex2, sizeof(int), cudaMemcpyDeviceToHost ) );	printf("The resultant sample num: %d\n",sampleNum);
		printf("The third phase of CUDA-LDNI offsetting (%d-direction) takes: %3.1f (ms)\n",(int)nAxis,elapsedTime);
		if (bWithRayPacking) {
			cudaFree(orderOfRayProcessing);
		}


		//--------------------------------------------------------------------------------------------------------
		//	Step 6: Compaction the samples on the newly generated rays
//		CUDA_SAFE_CALL( cudaMemcpy( devScanArrayResPtr, resSampleNum2, (arrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( devScanArrayResPtr, resSampleNum, (arrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToDevice ) );
		//--------------------------------------------------------------------------------------------------------
		thrust::device_ptr<unsigned int> dev_ptr(devScanArrayResPtr);	// Wrap raw pointers with dev_ptr
		thrust::exclusive_scan(dev_ptr, dev_ptr+(arrsize+1), dev_ptr);	// in-place scan
		sampleNum=dev_ptr[arrsize];
		printf("The resultant sample num: %d\n\n",sampleNum);
		//--------------------------------------------------------------------------------------------------------
		newSolid->MallocSampleMemory(nAxis,sampleNum);
		CUDA_SAFE_CALL( cudaMemcpy( newSolid->GetIndexArrayPtr(nAxis), devScanArrayResPtr, (arrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToDevice ) );
		krLDNIOffsetting_CollectResult<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(//resDepthBuffer2,resSampleStIndex2,resSampleNum2,
												resDepthBuffer,resSampleStIndex,resSampleNum,
												newSolid->GetSampleNxArrayPtr(nAxis),newSolid->GetSampleNyArrayPtr(nAxis),
												newSolid->GetSampleDepthArrayPtr(nAxis),newSolid->GetIndexArrayPtr(nAxis),arrsize,nAxis);

		//--------------------------------------------------------------------------------------------------------
		//	Step 7: Release the temporary used memory
		cudaFree(hashTableIndexArray);		cudaFree(hashElementsData);		
	}

	//--------------------------------------------------------------------------------------------------------
	//	Free the memory
	freeResBuffer(resDepthBuffer,resBufferIndex);
	freeStIndexAndNum(resSampleStIndex,resSampleNum);
	freeResBuffer(resDepthBuffer2,resBufferIndex2);
	freeStIndexAndNum(resSampleStIndex2,resSampleNum2);
	if (devNumOfSearchedRays>0) cudaFree(devRaysToBeSearched);
	//--------------------------------------------------------------------------------------------------------
	cudaFree(devScanArrayResPtr);
	//--------------------------------------------------------------------------------------------------------
	CUDA_SAFE_CALL( cudaEventDestroy( startClock ) );
	CUDA_SAFE_CALL( cudaEventDestroy( stopClock ) );

	//--------------------------------------------------------------------------------------------------------
	//	Boolean operation to obtain the final result
//	return;
	if (bGrownOrShrink) {
		_booleanOperation(newSolid, inputSolid, 0);
	}
	else {
		_booleanOperation(inputSolid, newSolid, 2);
		_switchSolid(inputSolid, newSolid);
	}
}

void LDNIcudaOperation::_sortingRaysByPossibleMergingGivenBySpatialHashingSamples(float *hashElementsData, unsigned int *hashTableIndexArray,
																				  int *devSHBoxToBeSearched, int devNumOfSearchedSHBox,
																				  short nAxis, int res, float gwidth, float offset, 
																				  unsigned int *&orderOfRayProcessing, unsigned int &nonzeroStIndex)
{
	unsigned int *keyOfRayProcessing;
	int arrsize=res*res;

	CUDA_SAFE_CALL( cudaMalloc( (void**)&keyOfRayProcessing, arrsize*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&orderOfRayProcessing, arrsize*sizeof(unsigned int) ) );

	cudaEvent_t     startClock, stopClock;		float elapsedTime;
	CUDA_SAFE_CALL( cudaEventCreate( &startClock ) );
	CUDA_SAFE_CALL( cudaEventCreate( &stopClock ) );
	CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
	//--------------------------------------------------------------------------------------------------------
	//	filling the keys/index-order by the number of merging times
	krLDNIOffsetting_EstimateWorkLoadingByValidSamplesOnPerpendicularRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
								hashTableIndexArray, devSHBoxToBeSearched, devNumOfSearchedSHBox, 
								nAxis, res, gwidth, offset, keyOfRayProcessing, orderOfRayProcessing);
	//--------------------------------------------------------------------------------------------------------
	CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
	CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
	printf("Work loading estimation takes: %3.1f (ms)\n",elapsedTime);
	CUDA_SAFE_CALL( cudaEventDestroy( startClock ) );
	CUDA_SAFE_CALL( cudaEventDestroy( stopClock ) );

	//--------------------------------------------------------------------------------------------------------
	//	sorting the index-order of rays according to their keys
	thrust::device_ptr<unsigned int> dev_ptrKey(keyOfRayProcessing);	//	Wrap raw pointers with dev_ptr
	thrust::device_ptr<unsigned int> dev_ptrData(orderOfRayProcessing);	//	Wrap raw pointers with dev_ptr
	thrust::sort_by_key(dev_ptrKey, dev_ptrKey + arrsize, dev_ptrData);

	//--------------------------------------------------------------------------------------------------------
	//	determine the nonzero start index
	unsigned int *stIndex;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&stIndex, sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMemset( stIndex, 0, sizeof(unsigned int)) );
	krLDNIOffsetting_DetermineNonzeroStartIndex<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(res,keyOfRayProcessing,stIndex);
	CUDA_SAFE_CALL( cudaMemcpy( &nonzeroStIndex, stIndex, sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
	cudaFree(stIndex);
	unsigned int maxMerging;
	CUDA_SAFE_CALL( cudaMemcpy( &maxMerging, &(keyOfRayProcessing[arrsize-1]), sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
	printf("The first effected ray with the sorted-index: %u (max-merging=%u)\n",nonzeroStIndex,maxMerging);

	//--------------------------------------------------------------------------------------------------------
	//	release the memory of keys
	cudaFree(keyOfRayProcessing);
}

void LDNIcudaOperation::_buildHashingTableForSamplesOnPerpendicularRays(LDNIcudaSolid* inputSolid, short nAxis, 
																		int &hashTableElementNum, 
																		unsigned int *&hashTableIndexArray, 																		
																		float *&hashElementsData)
{
	int res,arrsize,hashTableElementNum1,hashTableElementNum2;	
	unsigned int *hashElementsKey;		unsigned int *hashElementsDataLocation;
	float *hashElementsUnsortedData;

	res=inputSolid->GetResolution();	arrsize=res*res;
	hashTableElementNum1=inputSolid->GetSampleNumber((nAxis+1)%3);
	hashTableElementNum2=inputSolid->GetSampleNumber((nAxis+2)%3);
	hashTableElementNum=hashTableElementNum1+hashTableElementNum2;

	CUDA_SAFE_CALL( cudaMalloc( (void**)&hashTableIndexArray, (arrsize+1)*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)(hashTableIndexArray), 0, (arrsize+1)*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&hashElementsKey, hashTableElementNum*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&hashElementsDataLocation, hashTableElementNum*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&hashElementsData, hashTableElementNum*3*sizeof(float) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&hashElementsUnsortedData, hashTableElementNum*3*sizeof(float) ) );

	//--------------------------------------------------------------------------------------------------------
	//	Step 1: filling the hash-table data and keys
	krLDNIOffsetting_fillingHashingTableKeyAndData<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
								inputSolid->GetSampleDepthArrayPtr((nAxis+1)%3),
								inputSolid->GetIndexArrayPtr((nAxis+1)%3),
								true,
								nAxis,res,
								inputSolid->GetSampleWidth(),
								0, arrsize,
								hashElementsKey,
								hashElementsDataLocation,
								hashElementsUnsortedData);
	krLDNIOffsetting_fillingHashingTableKeyAndData<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
								inputSolid->GetSampleDepthArrayPtr((nAxis+2)%3),
								inputSolid->GetIndexArrayPtr((nAxis+2)%3),
								false,
								nAxis,res,
								inputSolid->GetSampleWidth(),
								hashTableElementNum1, arrsize,
								hashElementsKey,
								hashElementsDataLocation,
								hashElementsUnsortedData);
//	printf("Filling completed!\n");

	//--------------------------------------------------------------------------------------------------------
	//	Step 2: sorting the hash-table data by their keys
	thrust::device_ptr<unsigned int> dev_ptrKey(hashElementsKey);	//	Wrap raw pointers with dev_ptr
	thrust::device_ptr<unsigned int> dev_ptrData(hashElementsDataLocation);	//	Wrap raw pointers with dev_ptr
	thrust::sort_by_key(dev_ptrKey, dev_ptrKey + hashTableElementNum, dev_ptrData);

	//--------------------------------------------------------------------------------------------------------
	//	Step 3: filling the index table of the hash-table data 
	krLDNIOffsetting_fillingHashingIndexTable<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(hashTableElementNum,
																hashElementsKey, hashTableIndexArray, arrsize);
	printf("hash element number=%d \n",hashTableElementNum);

	//--------------------------------------------------------------------------------------------------------
	//	Step 4: relocating the hashElementData according to the sorting results
	krLDNIOffsetting_fillingHashingSortedData<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(hashTableElementNum,
											hashElementsDataLocation,hashElementsUnsortedData,hashElementsData);

	//--------------------------------------------------------------------------------------------------------
	//	Step 5: free the memory
	cudaFree(hashElementsKey);		cudaFree(hashElementsDataLocation);
}

//--------------------------------------------------------------------------------------------
//	The kernel functions
//--------------------------------------------------------------------------------------------


__device__ int krBisectionIntervalSearchOnRaySH(float *depth, int num, float inputDepth)
{
	inputDepth=fabs(inputDepth);
	if (inputDepth<fabs(depth[0])) return 0;
	if (inputDepth>=fabs(depth[num-1])) return num;

	int st=0,ed=num-1,md;
	while((ed-st)>1) {
		md=(st+ed)>>1;	
		if (inputDepth<fabs(depth[md])) 
			{	ed=md;		}
		else if (inputDepth>=fabs(depth[md+1])) 
			{	st=md+1;	}
		else 
			{	return (md+1);}
	}
	return ed;
}

__device__ int krLinearIntervalSearchOnRaySH(float *depth, int num, float inputDepth)
{
	inputDepth=fabs(inputDepth);	int k;
	for(k=0;k<num;k++) {if (inputDepth<fabs(depth[k])) break;}
	return k;
}

__device__ int krMergeSamplesFromSpatialHashing(float *depth, float *nx, float *ny, int num, float depth1, float nx1, float ny1, float depth2, float nx2, float ny2)
{
	int newNum,i,j,k;

	if (num==0) {	
		newNum=2;	depth[0]=depth1;  nx[0]=nx1;  ny[0]=ny1;	depth[1]=depth2;  nx[1]=nx2;  ny[1]=ny2;	
	}
	else {	// the code is based on the assumption that depth1<depth2
#ifdef BISECTION_INTERVAL_SEARCH
		i=krBisectionIntervalSearchOnRaySH(depth,num,depth1);	j=krBisectionIntervalSearchOnRaySH(depth,num,depth2);
#else
		i=krLinearIntervalSearchOnRaySH(depth,num,depth1);		j=krLinearIntervalSearchOnRaySH(depth,num,depth2);
#endif
		if (i==j) {
			if (i%2==0) {	// insert two samples into the place between (i-1) and i
				for(k=num-1;k>=i;k--) {depth[k+2]=depth[k];  nx[k+2]=nx[k];  ny[k+2]=ny[k];}
				depth[i]=depth1;	nx[i]=nx1;		ny[i]=ny1;
				depth[i+1]=depth2;  nx[i+1]=nx2;	ny[i+1]=ny2;
				newNum=num+2;
			}
			else {	// do nothing since new solid falls inside the existing solid
				newNum=num;
			}
		}
		else {
			if (i%2==0 && j%2==0) {			// Action: insert newSample1 at i and newSample1 at i+1, and remove samples between [i,j-1]
				depth[i]=depth1;  nx[i]=nx1;  ny[i]=ny1;
				depth[i+1]=depth2;  nx[i+1]=nx2;  ny[i+1]=ny2;
				newNum=i+2;
				for(k=j;k<num;k++) {depth[newNum]=depth[k]; nx[newNum]=nx[k]; ny[newNum]=ny[k]; newNum++;}
			}
			else if (i%2==0 && j%2!=0) {	// Action: insert newSample1 at i, and remove samples between [i+1,j-1]
				depth[i]=depth1;  nx[i]=nx1;  ny[i]=ny1;
				newNum=i+1;
				for(k=j;k<num;k++) {depth[newNum]=depth[k]; nx[newNum]=nx[k]; ny[newNum]=ny[k]; newNum++;}
			}
			else if (i%2!=0 && j%2==0) {	// Action: insert newSample2 at i, and remove samples between [i+1,j-1]
				depth[i]=depth2;  nx[i]=nx2;  ny[i]=ny2;
				newNum=i+1;
				for(k=j;k<num;k++) {depth[newNum]=depth[k]; nx[newNum]=nx[k]; ny[newNum]=ny[k]; newNum++;}
			}
			else {	//(i%2!=0 && j%2!=0)	// Action: remove samples between [i,j-1]
				newNum=i;
				for(k=j;k<num;k++) {depth[newNum]=depth[k]; nx[newNum]=nx[k]; ny[newNum]=ny[k]; newNum++;}
			}
		}
	}

	return newNum;
}

__device__ int krMergeSamplesFromSpatialHashing(float *depth, int num, float depth1, float depth2)
{
	int newNum,i,j,k;

	if (num==0) {	
		newNum=2;	depth[0]=depth1;  depth[1]=depth2;
	}
	else {	// the code is based on the assumption that depth1<depth2
#ifdef BISECTION_INTERVAL_SEARCH
		i=krBisectionIntervalSearchOnRaySH(depth,num,depth1);	j=krBisectionIntervalSearchOnRaySH(depth,num,depth2);
#else
		i=krLinearIntervalSearchOnRaySH(depth,num,depth1);		j=krLinearIntervalSearchOnRaySH(depth,num,depth2);
#endif
		if (i==j) {
			if (i%2==0) {	// insert two samples into the place between (i-1) and i
				for(k=num-1;k>=i;k--) {depth[k+2]=depth[k];}
				depth[i]=depth1;	depth[i+1]=depth2;  
				newNum=num+2;
			}
			else {	// do nothing since new solid falls inside the existing solid
				newNum=num;
			}
		}
		else {
			if (i%2==0 && j%2==0) {			// Action: insert newSample1 at i and newSample1 at i+1, and remove samples between [i,j-1]
				depth[i]=depth1;  	depth[i+1]=depth2;  
				newNum=i+2;
				for(k=j;k<num;k++) {depth[newNum]=depth[k]; newNum++;}
			}
			else if (i%2==0 && j%2!=0) {	// Action: insert newSample1 at i, and remove samples between [i+1,j-1]
				depth[i]=depth1;  
				newNum=i+1;
				for(k=j;k<num;k++) {depth[newNum]=depth[k]; newNum++;}
			}
			else if (i%2!=0 && j%2==0) {	// Action: insert newSample2 at i, and remove samples between [i+1,j-1]
				depth[i]=depth2;  
				newNum=i+1;
				for(k=j;k<num;k++) {depth[newNum]=depth[k]; newNum++;}
			}
			else {	//(i%2!=0 && j%2!=0)	// Action: remove samples between [i,j-1]
				newNum=i;
				for(k=j;k<num;k++) {depth[newNum]=depth[k]; newNum++;}
			}
		}
	}

	return newNum;
}

__global__ void krLDNIOffsetting_DetermineNonzeroStartIndex(int res, unsigned int *keyOfRayProcessing, unsigned int *stIndex)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x+1;

	while(index<res*res) {
		if (keyOfRayProcessing[index-1]==0 && keyOfRayProcessing[index]!=0) {
			*stIndex = index;
		}

		if (*stIndex!=0) break;

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIOffsetting_EstimateWorkLoadingByValidSamplesOnPerpendicularRays(unsigned int *hashTableIndexArray,
								int *devSHBoxToBeSearched, int devNumOfSearchedSHBox, short nAxis, int res, float gwidth, float offset, 
								unsigned int *keyOfRayProcessing, unsigned int *orderOfRayProcessing)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int i,j,ci,cj,gridIndex,mergedTime;
	unsigned int k,st2,ed2;

	while(index<res*res) {
		ci=index%res;	cj=index/res;

		mergedTime=0;
		for(k=0;k<devNumOfSearchedSHBox;k++) {
			i=ci+devSHBoxToBeSearched[k*2];			if (i<0) continue;	if (i>=res) continue;
			j=cj+devSHBoxToBeSearched[k*2+1];		if (j<0) continue;	if (j>=res) continue;

			gridIndex=i+j*res;
			st2=hashTableIndexArray[gridIndex];		ed2=hashTableIndexArray[gridIndex+1];	
			mergedTime += (ed2-st2);
		}

		keyOfRayProcessing[index]=(unsigned int)mergedTime;
		orderOfRayProcessing[index]=(unsigned int)index;

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIOffsetting_OnlyCopyAsNotEffectedBySamplesOnPerpendicularRays(unsigned int *orderOfRayProcessing, unsigned int nonzeroStIndex, 
													float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer, int *resBufferIndex,
													unsigned int *resSampleStIndex, unsigned int *resSampleNum,
													short nAxis, int stIndex, int edIndex, 
													float *resNxBuffer2, float *resNyBuffer2, float *resDepthBuffer2, int *resBufferIndex2,
													unsigned int *resSampleStIndex2, unsigned int *resSampleNum2)
{
	int id=threadIdx.x+blockIdx.x*blockDim.x+stIndex;
	int rayIndex;
	unsigned int st,num,k,resSt,resNum;	
	float depth[MAX_NUM_OF_SAMPLES_ON_RAY],nx[MAX_NUM_OF_SAMPLES_ON_RAY],ny[MAX_NUM_OF_SAMPLES_ON_RAY];

	while(id<edIndex) {
		rayIndex=orderOfRayProcessing[id];
		st=resSampleStIndex[rayIndex];		num=resSampleNum[rayIndex];

		for(k=0;k<num;k++) {depth[k]=resDepthBuffer[st+k];	nx[k]=resNxBuffer[st+k];	ny[k]=resNyBuffer[st+k];}
		resNum=num;  // num samples in total
		resSt=atomicAdd(resBufferIndex2, resNum);
		resSampleStIndex2[rayIndex]=resSt;		resSampleNum2[rayIndex]=resNum;

		for(k=0;k<resNum;k++) {
			resNxBuffer2[resSt+k]=nx[k];
			resNyBuffer2[resSt+k]=ny[k];
			resDepthBuffer2[resSt+k]=depth[k];
		}

		id += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIOffsetting_BySpatialHashingSamplesOnPerpendicularRays(float *hashElementsData, unsigned int *hashTableIndexArray,
								int *devSHBoxToBeSearched, int devNumOfSearchedSHBox, unsigned int *orderOfRayProcessing, unsigned int nonzeroStIndex, 
								float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer, int *resBufferIndex,
								unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								short nAxis, int res, float gwidth, float offset, int stIndex, int edIndex, 
								float *resNxBuffer2, float *resNyBuffer2, float *resDepthBuffer2, int *resBufferIndex2,
								unsigned int *resSampleStIndex2, unsigned int *resSampleNum2)
{
	int id=threadIdx.x+blockIdx.x*blockDim.x+stIndex;
	float dd,di,dj,depth1,depth2,nv1[3],nv2[3],rr2,rr;
	int i,j,ci,cj,gridIndex,rayIndex;
	unsigned int st,num,k,kk,st2,ed2,resSt,resNum,ii,jj;	
	float depth[MAX_NUM_OF_SAMPLES_ON_RAY],nx[MAX_NUM_OF_SAMPLES_ON_RAY],ny[MAX_NUM_OF_SAMPLES_ON_RAY];

	rr=fabs(offset);	rr2=rr*rr;		ii=(nAxis+1)%3;		jj=(nAxis+2)%3;
	while(id<edIndex) {
		rayIndex=orderOfRayProcessing[id];
		st=resSampleStIndex[rayIndex];		num=resSampleNum[rayIndex];
		ci=rayIndex%res;	cj=rayIndex/res;

		//---------------------------------------------------------------------------------------------------------
		//	collection data
		for(k=0;k<num;k++) {depth[k]=resDepthBuffer[st+k];	nx[k]=resNxBuffer[st+k];	ny[k]=resNyBuffer[st+k];}

		//---------------------------------------------------------------------------------------------------------
		//	search the possible samples in the spatial hashing, and merge the intersected spheres
		for(k=0;k<devNumOfSearchedSHBox;k++) {
			i=ci+devSHBoxToBeSearched[k*2];			if (i<0) continue;	if (i>=res) continue;
			j=cj+devSHBoxToBeSearched[k*2+1];		if (j<0) continue;	if (j>=res) continue;

			gridIndex=i+j*res;
			st2=hashTableIndexArray[gridIndex];		ed2=hashTableIndexArray[gridIndex+1];	
			for(kk=st2;kk<ed2;kk++) {
				di = hashElementsData[kk*3+ii];	dj = hashElementsData[kk*3+jj]; 
				di -= gwidth*(float)ci;			dj -= gwidth*(float)cj;	
				dd=rr2-di*di-dj*dj;
				if (dd<=0.0f) continue;
				dd=sqrt(dd);	depth2=depth1=fabs(hashElementsData[kk*3+nAxis]);

				//------------------------------------------------------------------------------------
				//	sample1: depth-dd;
				depth1=depth1-dd;	
				nv1[nAxis]=-dd/rr; nv1[ii]=-di/rr; nv1[jj]=-dj/rr;	// note that: the center of sphere is located at (i,j) but not (ci,cj)
																	//	therefore '-' is added for nv1[(nAxis+1)%3] and nv1[(nAxis+2)%3]
																	//	and also nv2[(nAxis+1)%3] and nv2[(nAxis+2)%3]
				if (nv1[2]<0.0f) depth1=-depth1;
				//------------------------------------------------------------------------------------
				//	sample2: depth+dd;
				depth2=depth2+dd;
				nv2[nAxis]=-nv1[nAxis]; nv2[ii]=nv1[ii]; nv2[jj]=nv1[jj];
				if (nv2[2]<0.0f) depth2=-depth2;

				num=krMergeSamplesFromSpatialHashing(
							&(depth[0]),&(nx[0]),&(ny[0]),
							num,depth1,nv1[0],nv1[1],depth2,nv2[0],nv2[1]);
			}
			resNum=num;
		}

		//---------------------------------------------------------------------------------------------------------
		//	saving the resultant samples
		if (resNum>0) {  // num samples in total
			resSt=atomicAdd(resBufferIndex2, resNum);
			resSampleStIndex2[rayIndex]=resSt;		resSampleNum2[rayIndex]=resNum;

			for(k=0;k<resNum;k++) {
				resNxBuffer2[resSt+k]=nx[k];
				resNyBuffer2[resSt+k]=ny[k];
				resDepthBuffer2[resSt+k]=depth[k];
			}
		}

		id += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIOffsetting_BySpatialHashingSamplesOnPerpendicularRays(float *hashElementsData, unsigned int *hashTableIndexArray,
								int *devSHBoxToBeSearched, int devNumOfSearchedSHBox,
								float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer, int *resBufferIndex,
								unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								short nAxis, int res, float gwidth, float offset, int stIndex, int edIndex,
								float *resNxBuffer2, float *resNyBuffer2, float *resDepthBuffer2, int *resBufferIndex2,
								unsigned int *resSampleStIndex2, unsigned int *resSampleNum2)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x+stIndex;
	float dd,di,dj,depth1,depth2,nv1[3],nv2[3],rr2,rr;
	int i,j,ci,cj,gridIndex;
	unsigned int st,num,k,kk,st2,ed2,resSt,resNum,ii,jj;	
	float depth[MAX_NUM_OF_SAMPLES_ON_RAY],nx[MAX_NUM_OF_SAMPLES_ON_RAY],ny[MAX_NUM_OF_SAMPLES_ON_RAY];

	rr=fabs(offset);	rr2=rr*rr;		ii=(nAxis+1)%3;		jj=(nAxis+2)%3;
	while(index<edIndex) {
		st=resSampleStIndex[index];		num=resSampleNum[index];
		ci=index%res;	cj=index/res;

		//---------------------------------------------------------------------------------------------------------
		//	collection data
		for(k=0;k<num;k++) {depth[k]=resDepthBuffer[st+k];	nx[k]=resNxBuffer[st+k];	ny[k]=resNyBuffer[st+k];}

		//---------------------------------------------------------------------------------------------------------
		//	search the possible samples in the spatial hashing, and merge the intersected spheres
		for(k=0;k<devNumOfSearchedSHBox;k++) {
			i=ci+devSHBoxToBeSearched[k*2];			if (i<0) continue;	if (i>=res) continue;
			j=cj+devSHBoxToBeSearched[k*2+1];		if (j<0) continue;	if (j>=res) continue;

			gridIndex=i+j*res;
			st2=hashTableIndexArray[gridIndex];		ed2=hashTableIndexArray[gridIndex+1];	
			for(kk=st2;kk<ed2;kk++) {
				di = hashElementsData[kk*3+ii];	dj = hashElementsData[kk*3+jj]; 
				di -= gwidth*(float)ci;			dj -= gwidth*(float)cj;	
				dd=rr2-di*di-dj*dj;
				if (dd<=0.0f) continue;
				dd=sqrt(dd);	depth2=depth1=fabs(hashElementsData[kk*3+nAxis]);

				//------------------------------------------------------------------------------------
				//	sample1: depth-dd;
				depth1=depth1-dd;	
				nv1[nAxis]=-dd/rr; nv1[ii]=-di/rr; nv1[jj]=-dj/rr;	// note that: the center of sphere is located at (i,j) but not (ci,cj)
																	//	therefore '-' is added for nv1[(nAxis+1)%3] and nv1[(nAxis+2)%3]
																	//	and also nv2[(nAxis+1)%3] and nv2[(nAxis+2)%3]
				if (nv1[2]<0.0f) depth1=-depth1;
				//------------------------------------------------------------------------------------
				//	sample2: depth+dd;
				depth2=depth2+dd;
				nv2[nAxis]=-nv1[nAxis]; nv2[ii]=nv1[ii]; nv2[jj]=nv1[jj];
				if (nv2[2]<0.0f) depth2=-depth2;

				num=krMergeSamplesFromSpatialHashing(
							&(depth[0]),&(nx[0]),&(ny[0]),
							num,depth1,nv1[0],nv1[1],depth2,nv2[0],nv2[1]);
			}
			resNum=num;
		}

		//---------------------------------------------------------------------------------------------------------
		//	saving the resultant samples
		if (resNum>0) {  // num samples in total
			resSt=atomicAdd(resBufferIndex2, resNum);
			resSampleStIndex2[index]=resSt;		resSampleNum2[index]=resNum;

			for(k=0;k<resNum;k++) {
				resNxBuffer2[resSt+k]=nx[k];
				resNyBuffer2[resSt+k]=ny[k];
				resDepthBuffer2[resSt+k]=depth[k];
			}
		}

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIOffsetting_fillingHashingSortedData(int hashTableElementNum, unsigned int *hashElementsDataLocation, 
														  float *hashElementsUnsortedData, float *hashElementsData)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int index2;

	while (index<hashTableElementNum) {
		index2=hashElementsDataLocation[index];
		hashElementsData[index*3]=hashElementsUnsortedData[index2*3];
		hashElementsData[index*3+1]=hashElementsUnsortedData[index2*3+1];
		hashElementsData[index*3+2]=hashElementsUnsortedData[index2*3+2];
		
		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIOffsetting_fillingHashingIndexTable(int hashTableElementNum, unsigned int *hashElementsKey,
														  unsigned int *hashTableIndexArray, int arrsize)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	bool bToFill;
	unsigned int currentKey,nextKey;	int i,nextIndex;

	while (index<hashTableElementNum) {
		bToFill=false;
		if (index==0) 
			{	bToFill=true;	}
		else 
			{	if (hashElementsKey[index]!=hashElementsKey[index-1]) bToFill=true;	}

		if (bToFill) {
			currentKey=hashElementsKey[index];		hashTableIndexArray[currentKey]=index;

			//--------------------------------------------------------------------------------------------------
			//	search next key (i.e., the next grid containing points)
			for(nextIndex=index+1;nextIndex<hashTableElementNum;nextIndex++) {
				if (hashElementsKey[nextIndex]!=currentKey) break;
			}
			if (nextIndex!=hashTableElementNum) {
				nextKey=hashElementsKey[nextIndex]; 
			}else {
				nextKey=arrsize;	hashTableIndexArray[nextKey]=hashTableElementNum;
			}

			//--------------------------------------------------------------------------------------------------
			//	fill the index table
			for(i=currentKey+1;i<nextKey;i++) hashTableIndexArray[i]=nextIndex;
		}

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIOffsetting_fillingHashingTableKeyAndData(float *inputDepthArrayPtr, unsigned int *inputIndexArrayPtr, bool bInputIsNextOrNextNext, 
															   short nCurrentDir, int res, float gwidth, int stIndex, int arrsize, 
															   unsigned int *hashElementsKey, unsigned int *hashElementsDataLocation, float *hashElementsData)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int sampleIndex,hashIndex,i,j,st,ed,ii,jj,checkedDir;	
	float gridOrigin,pp[3];

	if (bInputIsNextOrNextNext) checkedDir=(nCurrentDir+1)%3; else checkedDir=(nCurrentDir+2)%3;
	gridOrigin=-0.5f*gwidth;		

	while (index<arrsize) {
		i=index%res;	j=index/res;

		pp[(checkedDir+1)%3]=gwidth*(float)i;
		pp[(checkedDir+2)%3]=gwidth*(float)j;

		st=inputIndexArrayPtr[index];	ed=inputIndexArrayPtr[index+1];
		for(sampleIndex=st;sampleIndex<ed;sampleIndex++) {
			pp[checkedDir]=fabs(inputDepthArrayPtr[sampleIndex]);

			ii=(int)((pp[(nCurrentDir+1)%3]-gridOrigin)/gwidth);	
			jj=(int)((pp[(nCurrentDir+2)%3]-gridOrigin)/gwidth);

			hashIndex=sampleIndex+stIndex;

			hashElementsData[hashIndex*3]=pp[0];
			hashElementsData[hashIndex*3+1]=pp[1];
			hashElementsData[hashIndex*3+2]=pp[2];
			hashElementsDataLocation[hashIndex]=(unsigned int)hashIndex;
			hashElementsKey[hashIndex]=(unsigned int)jj*(unsigned int)res+(unsigned int)ii;
		}

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIOffsetting_BySpatialHashingSamplesOnPerpendicularRays(float *hashElementsData, unsigned int *hashTableIndexArray,
								int *devSHBoxToBeSearched, int devNumOfSearchedSHBox,
								float *resDepthBuffer, int *resBufferIndex, unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								short nAxis, int res, float gwidth, float offset, int stIndex, int edIndex,
								float *resDepthBuffer2, int *resBufferIndex2, unsigned int *resSampleStIndex2, unsigned int *resSampleNum2)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x+stIndex;
	float dd,di,dj,depth1,depth2,rr2,rr;
	int i,j,ci,cj,gridIndex;
	unsigned int st,num,k,kk,st2,ed2,resSt,resNum,ii,jj;	
	float depth[MAX_NUM_OF_SAMPLES_ON_RAY];

	rr=fabs(offset);	rr2=rr*rr;		ii=(nAxis+1)%3;		jj=(nAxis+2)%3;
	while(index<edIndex) {
		st=resSampleStIndex[index];		num=resSampleNum[index];
		ci=index%res;	cj=index/res;

		//---------------------------------------------------------------------------------------------------------
		//	collection data
		for(k=0;k<num;k++) {depth[k]=resDepthBuffer[st+k];}

		//---------------------------------------------------------------------------------------------------------
		//	search the possible samples in the spatial hashing, and merge the intersected spheres
		for(k=0;k<devNumOfSearchedSHBox;k++) {
			i=ci+devSHBoxToBeSearched[k*2];			if (i<0) continue;	if (i>=res) continue;
			j=cj+devSHBoxToBeSearched[k*2+1];		if (j<0) continue;	if (j>=res) continue;

			gridIndex=i+j*res;
			st2=hashTableIndexArray[gridIndex];		ed2=hashTableIndexArray[gridIndex+1];	
			for(kk=st2;kk<ed2;kk++) {
				di = hashElementsData[kk*3+ii];	dj = hashElementsData[kk*3+jj]; 
				di -= gwidth*(float)ci;			dj -= gwidth*(float)cj;	
				dd=rr2-di*di-dj*dj;
				if (dd<=0.0f) continue;
				dd=sqrt(dd);	depth2=depth1=hashElementsData[kk*3+nAxis];
				depth1=depth1-dd;	depth2=depth2+dd;
				num=krMergeSamplesFromSpatialHashing(&(depth[0]),num,depth1,depth2);
			}
			resNum=num;
		}

		//---------------------------------------------------------------------------------------------------------
		//	saving the resultant samples
		if (resNum>0) {  // num samples in total
			resSt=atomicAdd(resBufferIndex2, resNum);
			resSampleStIndex2[index]=resSt;		resSampleNum2[index]=resNum;
			for(k=0;k<resNum;k++) {resDepthBuffer2[resSt+k]=depth[k];}
		}

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIOffsetting_BySpatialHashingSamplesOnPerpendicularRays(float *hashElementsData, unsigned int *hashTableIndexArray,
								int *devSHBoxToBeSearched, int devNumOfSearchedSHBox, unsigned int *orderOfRayProcessing, unsigned int nonzeroStIndex, 
								float *resDepthBuffer, int *resBufferIndex,	unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								short nAxis, int res, float gwidth, float offset, int stIndex, int edIndex, 
								float *resDepthBuffer2, int *resBufferIndex2, unsigned int *resSampleStIndex2, unsigned int *resSampleNum2)
{
	int id=threadIdx.x+blockIdx.x*blockDim.x+stIndex;
	float dd,di,dj,depth1,depth2,rr2,rr;
	int i,j,ci,cj,gridIndex,rayIndex;
	unsigned int st,num,k,kk,st2,ed2,resSt,resNum,ii,jj;	
	float depth[MAX_NUM_OF_SAMPLES_ON_RAY];

	rr=fabs(offset);	rr2=rr*rr;		ii=(nAxis+1)%3;		jj=(nAxis+2)%3;
	while(id<edIndex) {
		rayIndex=orderOfRayProcessing[id];
		st=resSampleStIndex[rayIndex];		num=resSampleNum[rayIndex];
		ci=rayIndex%res;	cj=rayIndex/res;

		//---------------------------------------------------------------------------------------------------------
		//	collection data
		for(k=0;k<num;k++) {depth[k]=resDepthBuffer[st+k];}

		//---------------------------------------------------------------------------------------------------------
		//	search the possible samples in the spatial hashing, and merge the intersected spheres
		for(k=0;k<devNumOfSearchedSHBox;k++) {
			i=ci+devSHBoxToBeSearched[k*2];			if (i<0) continue;	if (i>=res) continue;
			j=cj+devSHBoxToBeSearched[k*2+1];		if (j<0) continue;	if (j>=res) continue;

			gridIndex=i+j*res;
			st2=hashTableIndexArray[gridIndex];		ed2=hashTableIndexArray[gridIndex+1];	
			for(kk=st2;kk<ed2;kk++) {
				di = hashElementsData[kk*3+ii];	dj = hashElementsData[kk*3+jj]; 
				di -= gwidth*(float)ci;			dj -= gwidth*(float)cj;	
				dd=rr2-di*di-dj*dj;
				if (dd<=0.0f) continue;
				dd=sqrt(dd);	depth2=depth1=hashElementsData[kk*3+nAxis];

				depth1=depth1-dd;	depth2=depth2+dd;
				num=krMergeSamplesFromSpatialHashing(&(depth[0]),num,depth1,depth2);
			}
			resNum=num;
		}

		//---------------------------------------------------------------------------------------------------------
		//	saving the resultant samples
		if (resNum>0) {  // num samples in total
			resSt=atomicAdd(resBufferIndex2, resNum);
			resSampleStIndex2[rayIndex]=resSt;		resSampleNum2[rayIndex]=resNum;
			for(k=0;k<resNum;k++) {resDepthBuffer2[resSt+k]=depth[k];}
		}

		id += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIOffsetting_OnlyCopyAsNotEffectedBySamplesOnPerpendicularRays(unsigned int *orderOfRayProcessing, unsigned int nonzeroStIndex, 
													float *resDepthBuffer, int *resBufferIndex, unsigned int *resSampleStIndex, unsigned int *resSampleNum,
													short nAxis, int stIndex, int edIndex, 
													float *resDepthBuffer2, int *resBufferIndex2, unsigned int *resSampleStIndex2, unsigned int *resSampleNum2)
{
	int id=threadIdx.x+blockIdx.x*blockDim.x+stIndex;
	int rayIndex;
	unsigned int st,num,k,resSt,resNum;	
	float depth[MAX_NUM_OF_SAMPLES_ON_RAY];

	while(id<edIndex) {
		rayIndex=orderOfRayProcessing[id];
		st=resSampleStIndex[rayIndex];		num=resSampleNum[rayIndex];

		for(k=0;k<num;k++) {depth[k]=resDepthBuffer[st+k];}
		resNum=num;  // num samples in total
		resSt=atomicAdd(resBufferIndex2, resNum);
		resSampleStIndex2[rayIndex]=resSt;		resSampleNum2[rayIndex]=resNum;

		for(k=0;k<resNum;k++) {resDepthBuffer2[resSt+k]=depth[k];}

		id += blockDim.x * gridDim.x;
	}
}
