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
#include <thrust/tuple.h>
#include <cstdlib>

#include "LDNIcudaSolid.h"
#include "LDNIcudaOperation.h"

//--------------------------------------------------------------------------------------------
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

extern __global__ void krLDNIOffsetting_ByPerpendicularRays(float *inputDepthArrayPtr, unsigned int *inputIndexArrayPtr,
								short nAxisToCheck, /*could be ((nAxis+1)%3) or ((nAxis+2)%3)*/
								float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer, int *resBufferIndex,
								unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								short nAxis, int res, float gwidth, float offset, int stIndex, int edIndex,
								float *resNxBuffer2, float *resNyBuffer2, float *resDepthBuffer2, int *resBufferIndex2,
								unsigned int *resSampleStIndex2, unsigned int *resSampleNum2);

extern __global__ void krLDNIOffsetting_CollectResult(float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer,
								unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								float *outputNxArrayPtr, float *outputNzArrayPtr, float *outputDepthArrayPtr, unsigned int *outputIndexArrayPtr, 
								int arrsize);

void initResBuffer(float *&resNxBuffer, float *&resNyBuffer, float *&resDepthBuffer, int *&resBufferIndex) 
{
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(resBufferIndex), sizeof(int) ) );	
	CUDA_SAFE_CALL( cudaMemset( (void*)(resBufferIndex), 0, sizeof(int) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(resNxBuffer), BUFFER_SIZE*sizeof(float) ) );	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(resNyBuffer), BUFFER_SIZE*sizeof(float) ) );	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(resDepthBuffer), BUFFER_SIZE*sizeof(float) ) );	
}

void freeResBuffer(float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer, int *resBufferIndex) 
{
	cudaFree(resNxBuffer);	cudaFree(resNyBuffer);	cudaFree(resDepthBuffer);	cudaFree(resBufferIndex);
}

void initStIndexAndNum(unsigned int *&resSampleStIndex, unsigned int *&resSampleNum, int size) 
{
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(resSampleStIndex), size*sizeof(unsigned int) ) );	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(resSampleNum), size*sizeof(unsigned int) ) );	
	CUDA_SAFE_CALL( cudaMemset( (void*)(resSampleStIndex), 0, size*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)(resSampleNum), 0, size*sizeof(unsigned int) ) );
}

void cleanUpStIndexAndNum(unsigned int *resSampleStIndex, unsigned int *resSampleNum, int size, int *resBufferIndex) 
{
	CUDA_SAFE_CALL( cudaMemset( (void*)(resSampleStIndex), 0, size*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)(resSampleNum), 0, size*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)(resBufferIndex), 0, sizeof(int) ) );
}

void freeStIndexAndNum(unsigned int *resSampleStIndex, unsigned int *resSampleNum) 
{
	cudaFree(resSampleStIndex);		cudaFree(resSampleNum);
}

void printProgress(int current, int full)
{
	int num=(int)((float)current/(float)full*100.0f);
	if (num%4==0)		printf("\b\b-");
	else if (num%4==1)	printf("\b\b/");
	else if (num%4==2)	printf("\b\b|");
	else if (num%4==3)	printf("\b\b\\");
}

//--------------------------------------------------------------------------------------------
//	The class member functions
//--------------------------------------------------------------------------------------------

void LDNIcudaOperation::SolidOffsetting(LDNIcudaSolid* inputSolid, LDNIcudaSolid* &newSolid, float offset)
{
	bool bGrownOrShrink;	float boundingBox[6],origin[3],gwidth;		int i,j,k,res,arrsize,ll;		short nAxis;
	cudaEvent_t     startClock, stopClock;		float elapsedTime;
	CUDA_SAFE_CALL( cudaEventCreate( &startClock ) );
	CUDA_SAFE_CALL( cudaEventCreate( &stopClock ) );
	int *raysToBeSearched;		int discreteSupport;
	int *devRaysToBeSearched;	int devNumOfSearchedRays=0;

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
		//k=discreteSupport;
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
		printf("Totally %d rays need to be searched! [-%d,%d]\n",devNumOfSearchedRays,discreteSupport,discreteSupport);
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(devRaysToBeSearched), devNumOfSearchedRays*2*sizeof(int) ) );	
		CUDA_SAFE_CALL( cudaMemcpy( devRaysToBeSearched, raysToBeSearched, devNumOfSearchedRays*2*sizeof(int),  cudaMemcpyHostToDevice ) );
//		for(i=0;i<devNumOfSearchedRays;i++) printf("%d-th: (%d,%d)\n",i,raysToBeSearched[i*2],raysToBeSearched[i*2+1]);
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
	sizeOfComputation=(int)((float)sizeOfComputation*0.001f+0.5f)*2+1;	
	int segmentSize=(int)(ceil(((float)arrsize/(float)sizeOfComputation)/(float)(BLOCKS_PER_GRID*THREADS_PER_BLOCK)))*BLOCKS_PER_GRID*THREADS_PER_BLOCK;
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

	for(nAxis=0;nAxis<3;nAxis++) {	//nAxis=0;
		int sampleNum=inputSolid->GetSampleNumber(nAxis);	printf("The input sample num: %d\n",sampleNum);
		cudaStream_t    stream;
		CUDA_SAFE_CALL( cudaStreamCreate( &stream ) );

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
		printf("The first phase of CUDA-LDNI offsetting (%d-direction): %3.1f (ms)\n",(int)nAxis,elapsedTime);
		
		//--------------------------------------------------------------------------------------------------------
		//	Step 2: Processing the samples on the other rays (but in the same direction)
		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
		//--------------------------------------------------------------------------------------------------------
		cleanUpStIndexAndNum(resSampleStIndex2, resSampleNum2, arrsize+1, resBufferIndex2);	// clean up buffer-2 to use as output
		for(int stIndex=0;stIndex<arrsize;stIndex+=segmentSize) {
			int sizeToProcess;
			if ((stIndex+segmentSize)<arrsize) sizeToProcess=segmentSize; else sizeToProcess=arrsize-stIndex;
			krLDNIOffsetting_ByParallelRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK,0,stream>>>(
						inputSolid->GetSampleDepthArrayPtr(nAxis),inputSolid->GetIndexArrayPtr(nAxis),
						resNxBuffer,resNyBuffer,resDepthBuffer,resBufferIndex,resSampleStIndex,resSampleNum,
						devRaysToBeSearched, devNumOfSearchedRays, nAxis, res, gwidth, offset, stIndex, stIndex+sizeToProcess, 
						resNxBuffer2,resNyBuffer2,resDepthBuffer2,resBufferIndex2,resSampleStIndex2,resSampleNum2);
			CUDA_SAFE_CALL( cudaStreamSynchronize( stream ) );
			printProgress(stIndex,arrsize);
		}
		printf("\b\b");
		//--------------------------------------------------------------------------------------------------------
		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
//		CUDA_SAFE_CALL( cudaMemcpy( &sampleNum, resBufferIndex2, sizeof(int), cudaMemcpyDeviceToHost ) );	printf("The resultant sample num: %d\n",sampleNum);
		printf("The second phase of CUDA-LDNI offsetting (%d-direction): %3.1f (ms)\n",(int)nAxis,elapsedTime);

		//--------------------------------------------------------------------------------------------------------
		//	Step 3: Processing the samples on the other rays (in different directions)
		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );
		CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
		//--------------------------------------------------------------------------------------------------------
		cleanUpStIndexAndNum(resSampleStIndex, resSampleNum, arrsize+1, resBufferIndex);	// clean up buffer-1 to use as output
		for(int stIndex=0;stIndex<arrsize;stIndex+=segmentSize) {
			int sizeToProcess;
			if ((stIndex+segmentSize)<arrsize) sizeToProcess=segmentSize; else sizeToProcess=arrsize-stIndex;
			krLDNIOffsetting_ByPerpendicularRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK,0,stream>>>(
						inputSolid->GetSampleDepthArrayPtr((nAxis+1)%3),inputSolid->GetIndexArrayPtr((nAxis+1)%3),1,
						resNxBuffer2,resNyBuffer2,resDepthBuffer2,resBufferIndex2,resSampleStIndex2,resSampleNum2,
						nAxis, res, gwidth, offset, stIndex, stIndex+sizeToProcess, 
						resNxBuffer,resNyBuffer,resDepthBuffer,resBufferIndex,resSampleStIndex,resSampleNum);
			CUDA_SAFE_CALL( cudaStreamSynchronize( stream ) );
			printProgress(stIndex,arrsize);
		}
		//--------------------------------------------------------------------------------------------------------
		cleanUpStIndexAndNum(resSampleStIndex2, resSampleNum2, arrsize+1, resBufferIndex2);	// clean up buffer-2 to use as output
		for(int stIndex=0;stIndex<arrsize;stIndex+=segmentSize) {
			int sizeToProcess;
			if ((stIndex+segmentSize)<arrsize) sizeToProcess=segmentSize; else sizeToProcess=arrsize-stIndex;
			krLDNIOffsetting_ByPerpendicularRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK,0,stream>>>(
						inputSolid->GetSampleDepthArrayPtr((nAxis+2)%3),inputSolid->GetIndexArrayPtr((nAxis+2)%3),2,
						resNxBuffer,resNyBuffer,resDepthBuffer,resBufferIndex,resSampleStIndex,resSampleNum,
						nAxis, res, gwidth, offset, stIndex, stIndex+sizeToProcess, 
						resNxBuffer2,resNyBuffer2,resDepthBuffer2,resBufferIndex2,resSampleStIndex2,resSampleNum2);
			CUDA_SAFE_CALL( cudaStreamSynchronize( stream ) );
			printProgress(stIndex,arrsize);
		}
		printf("\b\b");
/*		CUDA_SAFE_CALL( cudaMemcpy( resSampleNum, resSampleNum2, (arrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( resSampleStIndex, resSampleStIndex2, (arrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( &sampleNum, resBufferIndex2, sizeof(int), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( resNxBuffer, resNxBuffer2, sampleNum*sizeof(float), cudaMemcpyDeviceToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( resNyBuffer, resNyBuffer2, sampleNum*sizeof(float), cudaMemcpyDeviceToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( resDepthBuffer, resDepthBuffer2, sampleNum*sizeof(float), cudaMemcpyDeviceToDevice ) );*/
//		CUDA_SAFE_CALL( cudaMemcpy( &sampleNum, resBufferIndex2, sizeof(int), cudaMemcpyDeviceToHost ) );	printf("The resultant sample num: %d\n",sampleNum);
		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );	
		CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
		printf("The third phase of CUDA-LDNI offsetting (%d-direction): %3.1f (ms)\n",(int)nAxis,elapsedTime);

		//--------------------------------------------------------------------------------------------------------
		//	Step 4: Compaction the samples on the newly generated rays
		CUDA_SAFE_CALL( cudaMemset( (void*)(devScanArrayResPtr), 0, (arrsize+1)*sizeof(unsigned int) ) );
		CUDA_SAFE_CALL( cudaMemcpy( devScanArrayResPtr, resSampleNum2, (arrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToDevice ) );
		//--------------------------------------------------------------------------------------------------------
		thrust::device_ptr<unsigned int> dev_ptr(devScanArrayResPtr);	//	Wrap raw pointers with dev_ptr
		thrust::exclusive_scan(dev_ptr, dev_ptr+(arrsize+1), dev_ptr);	//	in-place scan
		sampleNum = dev_ptr[arrsize];
		printf("The resultant sample num: %d\n\n",sampleNum);
		//--------------------------------------------------------------------------------------------------------
		newSolid->MallocSampleMemory(nAxis,sampleNum);
		CUDA_SAFE_CALL( cudaMemcpy( newSolid->GetIndexArrayPtr(nAxis), devScanArrayResPtr, (arrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToDevice ) );
		krLDNIOffsetting_CollectResult<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(resNxBuffer2,resNyBuffer2,resDepthBuffer2,resSampleStIndex2,resSampleNum2,
												newSolid->GetSampleNxArrayPtr(nAxis),newSolid->GetSampleNyArrayPtr(nAxis),
												newSolid->GetSampleDepthArrayPtr(nAxis),newSolid->GetIndexArrayPtr(nAxis),arrsize);

		CUDA_SAFE_CALL( cudaStreamDestroy( stream ) );
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
	if (bGrownOrShrink) {
		_booleanOperation(newSolid, inputSolid, 0);
	}
	else {
		_booleanOperation(inputSolid, newSolid, 2);
		_switchSolid(inputSolid, newSolid);
	}
}


__global__ void krDistanceToBoundBoxBnd(float* sampleDepth, unsigned int* sampleBasedIndexArray, int sampleNum, int res, float gwidth, float* distance)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	float i,j,boxWidth=gwidth*(float)(res-1);
	int index;

	while (tid<sampleNum) {
		index=sampleBasedIndexArray[tid];
		j = (float)(index / res) * gwidth;	i = (float)(index % res) * gwidth;
		distance[tid]=fabs(sampleDepth[tid]);
		if (boxWidth-fabs(sampleDepth[tid])<distance[tid]) distance[tid]=boxWidth-fabs(sampleDepth[tid]);
		if (i<distance[tid]) distance[tid]=i;
		if (boxWidth-i<distance[tid]) distance[tid]=boxWidth-i;
		if (j<distance[tid]) distance[tid]=j;
		if (boxWidth-j<distance[tid]) distance[tid]=boxWidth-j;

		tid += blockDim.x * gridDim.x;
	}
}

float LDNIcudaOperation::_distanceToBoundBoxBoundary(LDNIcudaSolid* inputSolid)
{
	float gwidth;		int res,nAxis;		float returnvalue;
	unsigned int* sampleBasedIndexArray;
	float* sampleDepthArray;

	gwidth=inputSolid->GetSampleWidth();
	res=inputSolid->GetResolution();	returnvalue=gwidth*(float)res;

	for(nAxis=0;nAxis<3;nAxis++) {
		inputSolid->BuildSampleBasedIndexArray(nAxis, sampleBasedIndexArray);
		sampleDepthArray=inputSolid->GetSampleDepthArrayPtr(nAxis);
		int sampleNum=inputSolid->GetSampleNumber(nAxis);

		thrust::device_vector<float> dev_Distance(sampleNum);
		float* ptr = thrust::raw_pointer_cast(&dev_Distance[0]);
		krDistanceToBoundBoxBnd<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
									sampleDepthArray,sampleBasedIndexArray,sampleNum,res,gwidth,ptr);
		float minDist=thrust::reduce(dev_Distance.begin(), dev_Distance.end(), gwidth*(float)res, thrust::minimum<float>());
		if (minDist<returnvalue) returnvalue=minDist;

		cudaFree(sampleBasedIndexArray);
	}

	return returnvalue;
}


//--------------------------------------------------------------------------------------------
//	The kernel functions	
//--------------------------------------------------------------------------------------------

__device__ int krMergeSamples2(float *depth, float *nx, float *ny, int num, float depth1, float nx1, float ny1, float depth2, float nx2, float ny2)
{	//  Boolean operation based merge, which is slower
	float resNx[MAX_NUM_OF_SAMPLES_ON_RAY],resNy[MAX_NUM_OF_SAMPLES_ON_RAY],resDepth[MAX_NUM_OF_SAMPLES_ON_RAY];
	float depthB[2],nxB[2],nyB[2];	int numB;
	bool last_op,op,insideA,insideB;
	float lastNx,lastNy,lastDepth;
	int newNum,i,aIndex,bIndex;

	if (num==0) {
		newNum=2;	depth[0]=depth1;  nx[0]=nx1;  ny[0]=ny1;	depth[1]=depth2;  nx[1]=nx2;  ny[1]=ny2;
	}
	else{
		last_op=insideA=insideB=false;	newNum=0;	
		depthB[0]=depth1;	depthB[1]=depth2;	nxB[0]=nx1;	nxB[1]=nx2;	nyB[0]=ny1;	nyB[1]=ny2;
		numB=2;

		aIndex=bIndex=0;
		while( (aIndex<num) || (bIndex<numB) ) {	// scaning the samples on solidA and solidB together
			if ((bIndex==numB) || (aIndex<num && fabs(depth[aIndex])<fabs(depthB[bIndex]))) 
			{
				// advancing on ray-A
				lastDepth=depth[aIndex];	lastNx=nx[aIndex];		lastNy=ny[aIndex];
				insideA=!insideA;	aIndex++;
			}
			else {
				// advancing on ray-B
				lastDepth=depthB[bIndex];	lastNx=nxB[bIndex];		lastNy=nyB[bIndex];
				insideB=!insideB;	bIndex++;
			}

			op=LOGIC_UNION(insideA,insideB);

			if (op!=last_op) {
				if (newNum>0 && fabs(fabs(lastDepth)-fabs(resDepth[newNum-1]))<0.00001f) 
						{newNum--;}
				else {
					resDepth[newNum]=lastDepth;	
					resNx[newNum]=lastNx;	resNy[newNum]=lastNy;	
					newNum++;
				}
				last_op=op;
			}
		}

		for(i=0;i<newNum;i++) {depth[i]=resDepth[i];	nx[i]=resNx[i];	ny[i]=resNy[i];}
	}

	return newNum;
}

__device__ int krBisectionIntervalSearchOnRay(float *depth, int num, float inputDepth)
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

__device__ int krLinearIntervalSearchOnRay(float *depth, int num, float inputDepth)
{
	inputDepth=fabs(inputDepth);	int k;
	for(k=0;k<num;k++) {if (inputDepth<fabs(depth[k])) break;}
	return k;
}

__device__ int krMergeSamples(float *depth, float *nx, float *ny, int num, float depth1, float nx1, float ny1, float depth2, float nx2, float ny2)
{
	int newNum,i,j,k;

	if (num==0) {
		newNum=2;	depth[0]=depth1;  nx[0]=nx1;  ny[0]=ny1;	depth[1]=depth2;  nx[1]=nx2;  ny[1]=ny2;
	}
	else {	// the code is based on the assumption that depth1<depth2
#ifdef BISECTION_INTERVAL_SEARCH
		i=krBisectionIntervalSearchOnRay(depth,num,depth1);	j=krBisectionIntervalSearchOnRay(depth,num,depth2);
#else
		i=krLinearIntervalSearchOnRay(depth,num,depth1);	j=krLinearIntervalSearchOnRay(depth,num,depth2);
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

__device__ int krMergeSamples(float *depth, int num, float depth1, float depth2)
{
	int newNum,i,j,k;

	if (num==0) {
		newNum=2;	depth[0]=depth1;  depth[1]=depth2;
	}
	else {	// the code is based on the assumption that depth1<depth2
#ifdef BISECTION_INTERVAL_SEARCH
		i=krBisectionIntervalSearchOnRay(depth,num,depth1);	j=krBisectionIntervalSearchOnRay(depth,num,depth2);
#else
		i=krLinearIntervalSearchOnRay(depth,num,depth1);	j=krLinearIntervalSearchOnRay(depth,num,depth2);
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
				depth[i]=depth1;	depth[i+1]=depth2;
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

__global__ void krLDNIOffsetting_ByPerpendicularRays(float *inputDepthArrayPtr, unsigned int *inputIndexArrayPtr,
								short nAxisToCheck, /*could be ((nAxis+1)%3) or ((nAxis+2)%3)*/
								float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer, int *resBufferIndex,
								unsigned int *resSampleStIndex, unsigned int *resSampleNum,
								short nAxis, int res, float gwidth, float offset, int stIndex, int edIndex, 
								float *resNxBuffer2, float *resNyBuffer2, float *resDepthBuffer2, int *resBufferIndex2,
								unsigned int *resSampleStIndex2, unsigned int *resSampleNum2)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x+stIndex;
	float depth[MAX_NUM_OF_SAMPLES_ON_RAY],nx[MAX_NUM_OF_SAMPLES_ON_RAY],ny[MAX_NUM_OF_SAMPLES_ON_RAY];
	float cDepth,ddd,dd,di,dj,depth1,depth2,nv1[3],nv2[3],rr2,rr;
	int ci,cj,st,num,k,ii,jj,kk,nOff,index2,st2,ed2,num2,resSt,resNum;
	int searchInterval[MAX_NUM_OF_SAMPLES_ON_RAY],intervalNum;

	rr=fabs(offset);	rr2=rr*rr;		nOff=(int)(ceil(rr/gwidth));

	while(index<edIndex) {
		st=resSampleStIndex[index];		num=resSampleNum[index];
		ci=index%res;	cj=index/res;

		//---------------------------------------------------------------------------------------------------------
		//	collection data
		for(k=0;k<num;k++) {depth[k]=resDepthBuffer[st+k];	nx[k]=resNxBuffer[st+k];	ny[k]=resNyBuffer[st+k];}
		searchInterval[0]=0;	intervalNum=1;
		for(k=0;k<num;k+=2) {
			ii=(int)(ceil((fabs(depth[k])+rr)/gwidth))+1;
			jj=(int)(floor((fabs(depth[k+1])-rr)/gwidth))-1;
			if (jj<=ii) continue;
			searchInterval[intervalNum]=ii;		intervalNum++;
			searchInterval[intervalNum]=jj;		intervalNum++;
		}
		searchInterval[intervalNum]=res-1;	intervalNum++;

		if (nAxisToCheck==1) {
			//-----------------------------------------------------------------------------------------------------
			//	search the possible intersection rays, and merge the intersected spheres
			for(k=0;k<intervalNum;k+=2) {
				for(jj=searchInterval[k];jj<=searchInterval[k+1];jj++) {	//	only the valid intervals are processed
//			for(jj=0;jj<res;jj++) {
					cDepth=gwidth*(float)jj;
					for(ii=cj-nOff;ii<=cj+nOff;ii++) {
						if (ii<0) continue;		if (ii>=res) continue;
						index2=jj*res+ii;
						st2=inputIndexArrayPtr[index2];		ed2=inputIndexArrayPtr[index2+1];
						for(kk=st2;kk<ed2;kk++) {
							ddd=fabs(inputDepthArrayPtr[kk]);
							ddd=(gwidth*(float)ci)-ddd;		dj=gwidth*(float)(cj-ii);	dd=rr2-ddd*ddd-dj*dj;
							if (dd<=0.0f) continue;			dd=sqrt(dd);

							//------------------------------------------------------------------------------------
							//	sample1: depth-dd;
							depth1=cDepth-dd;
							nv1[nAxis]=-dd/rr; nv1[(nAxis+1)%3]=ddd/rr;	nv1[(nAxis+2)%3]=dj/rr;
							if (nv1[2]<0.0f) depth1=-depth1;
							//------------------------------------------------------------------------------------
							//	sample2: depth+dd;
							depth2=cDepth+dd;
							nv2[nAxis]=-nv1[nAxis];	nv2[(nAxis+1)%3]=nv1[(nAxis+1)%3];	nv2[(nAxis+2)%3]=nv1[(nAxis+2)%3];
							if (nv2[2]<0.0f) depth2=-depth2;

							num2=krMergeSamples(&(depth[0]),&(nx[0]),&(ny[0]),num,depth1,nv1[0],nv1[1],depth2,nv2[0],nv2[1]);
							num=num2;
						}
					}
				}
			}
			resNum=num;
		}
		else { // "nAxisToCheck==(nAxis+2)%3"
			//-----------------------------------------------------------------------------------------------------
			//	search the possible intersection rays, and merge the intersected spheres
			for(k=0;k<intervalNum;k+=2) {
				for(ii=searchInterval[k];ii<=searchInterval[k+1];ii++) {	//	only the valid intervals are processed
//			for(ii=0;ii<res;ii++) {
					cDepth=gwidth*(float)ii;
					for(jj=ci-nOff;jj<=ci+nOff;jj++) {
						if (jj<0) continue;		if (jj>=res) continue;
						index2=jj*res+ii;
						st2=inputIndexArrayPtr[index2];		ed2=inputIndexArrayPtr[index2+1];
						for(kk=st2;kk<ed2;kk++) {
							ddd=fabs(inputDepthArrayPtr[kk]);
							ddd=(gwidth*(float)cj)-ddd;		di=gwidth*(float)(ci-jj);	dd=rr2-ddd*ddd-di*di;
							if (dd<=0.0) continue;			dd=sqrt(dd);

							//------------------------------------------------------------------------------------
							//	sample1: depth-dd;
							depth1=cDepth-dd;
							nv1[nAxis]=-dd/rr; nv1[(nAxis+1)%3]=di/rr;	nv1[(nAxis+2)%3]=ddd/rr;
							if (nv1[2]<0.0f) depth1=-depth1;
							//------------------------------------------------------------------------------------
							//	sample2: depth+dd;
							depth2=cDepth+dd;
							nv2[nAxis]=-nv1[nAxis];	nv2[(nAxis+1)%3]=nv1[(nAxis+1)%3];	nv2[(nAxis+2)%3]=nv1[(nAxis+2)%3];
							if (nv2[2]<0.0f) depth2=-depth2;

							num2=krMergeSamples(&(depth[0]),&(nx[0]),&(ny[0]),num,depth1,nv1[0],nv1[1],depth2,nv2[0],nv2[1]);
							num=num2;
						}
					}
				}
			}
			resNum=num;
		}

		//------------------------------------------------------------------------
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

__global__ void krLDNIOffsetting_ByParallelRays(float *inputDepthArrayPtr, unsigned int *inputIndexArrayPtr,
													  float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer, int *resBufferIndex,
													  unsigned int *resSampleStIndex, unsigned int *resSampleNum,
													  int *devRaysToBeSearched, int devNumOfSearchedRays, 
													  short nAxis, int res, float gwidth, float offset, int stIndex, int edIndex, 
													  float *resNxBuffer2, float *resNyBuffer2, float *resDepthBuffer2, int *resBufferIndex2,
													  unsigned int *resSampleStIndex2, unsigned int *resSampleNum2)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x+stIndex;
	float depth[MAX_NUM_OF_SAMPLES_ON_RAY],nx[MAX_NUM_OF_SAMPLES_ON_RAY],ny[MAX_NUM_OF_SAMPLES_ON_RAY];
	int ci,cj,st,num,i,j,k,index2,st2,num2,ed2,kk,resSt,resNum;
	float di,dj,dd,rr,rr2,depth1,depth2,nv1[3],nv2[3];

	rr2=offset*offset;	rr=fabs(offset);

	while(index<edIndex) {
		st=resSampleStIndex[index];		num=resSampleNum[index];
		ci=index%res;	cj=index/res;

		//------------------------------------------------------------------------
		//	collection data
		for(k=0;k<num;k++) {depth[k]=resDepthBuffer[st+k];	nx[k]=resNxBuffer[st+k];	ny[k]=resNyBuffer[st+k];}

		//------------------------------------------------------------------------
		//	merge with the spheres generated by the samples on neighboring rays
		for(k=0;k<devNumOfSearchedRays;k++) {
			i=ci+devRaysToBeSearched[k*2];		if (i<0) continue;	if (i>=res) continue;
			j=cj+devRaysToBeSearched[k*2+1];	if (j<0) continue;	if (j>=res) continue;

			di=gwidth*(float)(i-ci);	dj=gwidth*(float)(j-cj);	dd=rr2-di*di-dj*dj;
			if (dd<=0.0f) continue;
			dd=sqrt(dd);
			index2=j*res+i;		st2=inputIndexArrayPtr[index2];		ed2=inputIndexArrayPtr[index2+1];
			for(kk=st2;kk<ed2;kk++) {
				depth1=fabs(inputDepthArrayPtr[kk]);	depth2=depth1;
				//------------------------------------------------------------------------------------
				//	sample1: depth-dd;
				depth1=depth1-dd;	
				nv1[nAxis]=-dd/rr; nv1[(nAxis+1)%3]=-di/rr;	nv1[(nAxis+2)%3]=-dj/rr;	// note that: the center of sphere is located at (i,j) but not (ci,cj)
																						//	therefore '-' is added for nv1[(nAxis+1)%3] and nv1[(nAxis+2)%3]
																						//	and also nv2[(nAxis+1)%3] and nv2[(nAxis+2)%3]
				if (nv1[2]<0.0f) depth1=-depth1;
				//------------------------------------------------------------------------------------
				//	sample2: depth+dd;
				depth2=depth2+dd;
				nv2[nAxis]=-nv1[nAxis];	nv2[(nAxis+1)%3]=nv1[(nAxis+1)%3];	nv2[(nAxis+2)%3]=nv1[(nAxis+2)%3];
				if (nv2[2]<0.0f) depth2=-depth2;

				num2=krMergeSamples(&(depth[0]),&(nx[0]),&(ny[0]),num,depth1,nv1[0],nv1[1],depth2,nv2[0],nv2[1]);
				num=num2;
			}
		}
		resNum=num;

		//------------------------------------------------------------------------
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

__global__ void krLDNIOffsetting_CollectResult(float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer,
											   unsigned int *resSampleStIndex, unsigned int *resSampleNum,
											   float *outputNxArrayPtr, float *outputNzArrayPtr, float *outputDepthArrayPtr, unsigned int *outputIndexArrayPtr, int arrsize)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int i,st,outputSt,num;

	while(index<arrsize) {
		st=resSampleStIndex[index];		num=resSampleNum[index];
		outputSt=outputIndexArrayPtr[index];

		for(i=0;i<num;i++) {
			outputNxArrayPtr[outputSt+i]=resNxBuffer[st+i];
			outputNzArrayPtr[outputSt+i]=resNyBuffer[st+i];
			outputDepthArrayPtr[outputSt+i]=resDepthBuffer[st+i];
		}

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIOffsetting_InRays(float *inputDepthArrayPtr, unsigned int *inputIndexArrayPtr,
											  float offset, int arrsize, short nAxis,
											  float *resNxBuffer, float *resNyBuffer, float *resDepthBuffer, int *resBufferIndex,
											  unsigned int *resSampleStIndex, unsigned int *resSampleNum)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int i,j,inc,st,num,nFlag,nFlagPre,resSt,resNum;
	float temp,depth[MAX_NUM_OF_SAMPLES_ON_RAY*2],nv[3];

	nv[0]=nv[1]=nv[2]=0.0f;	nv[nAxis]=1.0;

	while(index<arrsize) {
		st=inputIndexArrayPtr[index];	num=inputIndexArrayPtr[index+1]-st;

		//------------------------------------------------------------------------
		//	collection data
		for(i=0;i<num;i++) {
			depth[i*2]=-(fabs(inputDepthArrayPtr[st+i])-offset);
			depth[i*2+1]=(fabs(inputDepthArrayPtr[st+i])+offset);
		}
		num=num*2;

		//------------------------------------------------------------------------
		//	sorting the samples (shell-sort or bubble)
		inc=(int)(num/2);
		while(inc>0) {
			for(i=inc;i<num;i++) {
				temp=depth[i];
				j=i;
				while((j>=inc) && (fabs(depth[j-inc])>fabs(temp))) {
					depth[j]=depth[j-inc];
					j=j-inc;
				}
				depth[j]=temp;
			}
			inc=inc/2;
		}

		//------------------------------------------------------------------------
		//	determine the resultant samples
		resNum=0;	nFlagPre=0;
		for(i=0;i<num;i++) {
			if (depth[i]<0.0f) nFlag=nFlagPre+1; else nFlag=nFlagPre-1;
			if ((nFlagPre==0 && nFlag==1) || (nFlagPre==1 && nFlag==0)) resNum++;
			nFlagPre=nFlag;
		}
		if (resNum>0) {
			resSt=atomicAdd(resBufferIndex, resNum);
			resSampleStIndex[index]=resSt;	resSampleNum[index]=resNum;
			
			resNum=0;	nFlagPre=0;
			for(i=0;i<num;i++) {
				if (depth[i]<0.0f) nFlag=nFlagPre+1; else nFlag=nFlagPre-1;
				if ((nFlagPre==0 && nFlag==1) || (nFlagPre==1 && nFlag==0)) {
					if (resNum%2==0) {
						resNxBuffer[resSt+resNum]=-nv[0];
						resNyBuffer[resSt+resNum]=-nv[1];
						resDepthBuffer[resSt+resNum]=-fabs(depth[i]);
					}
					else {
						resNxBuffer[resSt+resNum]=nv[0];
						resNyBuffer[resSt+resNum]=nv[1];
						resDepthBuffer[resSt+resNum]=fabs(depth[i]);
					}
					resNum++;
				}
				nFlagPre=nFlag;
			}
		}
		
		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIOffsetting_InRays(float *inputDepthArrayPtr, unsigned int *inputIndexArrayPtr,
											  float offset, int arrsize, short nAxis,
											  float *resDepthBuffer, int *resBufferIndex,
											  unsigned int *resSampleStIndex, unsigned int *resSampleNum)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int i,j,inc,st,num,nFlag,nFlagPre,resSt,resNum;
	float temp,depth[MAX_NUM_OF_SAMPLES_ON_RAY*2];

	while(index<arrsize) {
		st=inputIndexArrayPtr[index];	num=inputIndexArrayPtr[index+1]-st;

		//------------------------------------------------------------------------
		//	collection data
		for(i=0;i<num;i++) {
			depth[i*2]=-(fabs(inputDepthArrayPtr[st+i])-offset);
			depth[i*2+1]=(fabs(inputDepthArrayPtr[st+i])+offset);
		}
		num=num*2;

		//------------------------------------------------------------------------
		//	sorting the samples (shell-sort or bubble)
		inc=(int)(num/2);
		while(inc>0) {
			for(i=inc;i<num;i++) {
				temp=depth[i];
				j=i;
				while((j>=inc) && (fabs(depth[j-inc])>fabs(temp))) {
					depth[j]=depth[j-inc];
					j=j-inc;
				}
				depth[j]=temp;
			}
			inc=inc/2;
		}

		//------------------------------------------------------------------------
		//	determine the resultant samples
		resNum=0;	nFlagPre=0;
		for(i=0;i<num;i++) {
			if (depth[i]<0.0f) nFlag=nFlagPre+1; else nFlag=nFlagPre-1;
			if ((nFlagPre==0 && nFlag==1) || (nFlagPre==1 && nFlag==0)) resNum++;
			nFlagPre=nFlag;
		}
		if (resNum>0) {
			resSt=atomicAdd(resBufferIndex, resNum);
			resSampleStIndex[index]=resSt;	resSampleNum[index]=resNum;
			
			resNum=0;	nFlagPre=0;
			for(i=0;i<num;i++) {
				if (depth[i]<0.0f) nFlag=nFlagPre+1; else nFlag=nFlagPre-1;
				if ((nFlagPre==0 && nFlag==1) || (nFlagPre==1 && nFlag==0)) {
					resDepthBuffer[resSt+resNum]=fabs(depth[i]);
					resNum++;
				}
				nFlagPre=nFlag;
			}
		}
		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIOffsetting_CollectResult(float *resDepthBuffer, unsigned int *resSampleStIndex, unsigned int *resSampleNum,
											   float *outputNxArrayPtr, float *outputNzArrayPtr, float *outputDepthArrayPtr, unsigned int *outputIndexArrayPtr, 
											   int arrsize, int nAxis)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int i,st,outputSt,num;

	switch(nAxis) {
	case 0:{
		while(index<arrsize) {
			st=resSampleStIndex[index];		num=resSampleNum[index];
			outputSt=outputIndexArrayPtr[index];

			for(i=0;i<num;i+=2) {
				outputNxArrayPtr[outputSt+i]=-1.0;		outputNzArrayPtr[outputSt+i]=0.0;
				outputDepthArrayPtr[outputSt+i]=resDepthBuffer[st+i];
				outputNxArrayPtr[outputSt+i+1]=1.0;		outputNzArrayPtr[outputSt+i+1]=0.0;
				outputDepthArrayPtr[outputSt+i+1]=resDepthBuffer[st+i+1];
			}

			index += blockDim.x * gridDim.x;
		}
		   }break;
	case 1:{
		while(index<arrsize) {
			st=resSampleStIndex[index];		num=resSampleNum[index];
			outputSt=outputIndexArrayPtr[index];

			for(i=0;i<num;i+=2) {
				outputNxArrayPtr[outputSt+i]=0.0;		outputNzArrayPtr[outputSt+i]=-1.0;
				outputDepthArrayPtr[outputSt+i]=resDepthBuffer[st+i];
				outputNxArrayPtr[outputSt+i+1]=0.0;		outputNzArrayPtr[outputSt+i+1]=1.0;
				outputDepthArrayPtr[outputSt+i+1]=resDepthBuffer[st+i+1];
			}

			index += blockDim.x * gridDim.x;
		}
		   }break;
	case 2:{
		while(index<arrsize) {
			st=resSampleStIndex[index];		num=resSampleNum[index];
			outputSt=outputIndexArrayPtr[index];

			for(i=0;i<num;i+=2) {
				outputNxArrayPtr[outputSt+i]=0.0;		outputNzArrayPtr[outputSt+i]=0.0;
				outputDepthArrayPtr[outputSt+i]=-resDepthBuffer[st+i];
				outputNxArrayPtr[outputSt+i+1]=0.0;		outputNzArrayPtr[outputSt+i+1]=0.0;
				outputDepthArrayPtr[outputSt+i+1]=resDepthBuffer[st+i+1];
			}

			index += blockDim.x * gridDim.x;
		}
		   }break;
	}
}

__global__ void krLDNIOffsetting_ByParallelRays(float *inputDepthArrayPtr, unsigned int *inputIndexArrayPtr,
												float *resDepthBuffer, int *resBufferIndex,
												unsigned int *resSampleStIndex, unsigned int *resSampleNum,
												int *devRaysToBeSearched, int devNumOfSearchedRays, 
												short nAxis, int res, float gwidth, float offset, int stIndex, int edIndex, 
												float *resDepthBuffer2, int *resBufferIndex2,
												unsigned int *resSampleStIndex2, unsigned int *resSampleNum2)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x+stIndex;
	float depth[MAX_NUM_OF_SAMPLES_ON_RAY];
	int ci,cj,st,num,i,j,k,index2,st2,num2,ed2,kk,resSt,resNum;
	float di,dj,dd,rr,rr2,depth1,depth2;

	rr2=offset*offset;	rr=fabs(offset);

	while(index<edIndex) {
		st=resSampleStIndex[index];		num=resSampleNum[index];
		ci=index%res;	cj=index/res;

		//------------------------------------------------------------------------
		//	collection data
		for(k=0;k<num;k++) {depth[k]=resDepthBuffer[st+k];	}

		//------------------------------------------------------------------------
		//	merge with the spheres generated by the samples on neighboring rays
		for(k=0;k<devNumOfSearchedRays;k++) {
			i=ci+devRaysToBeSearched[k*2];		if (i<0) continue;	if (i>=res) continue;
			j=cj+devRaysToBeSearched[k*2+1];	if (j<0) continue;	if (j>=res) continue;

			di=gwidth*(float)(i-ci);	dj=gwidth*(float)(j-cj);	dd=rr2-di*di-dj*dj;
			if (dd<=0.0f) continue;
			dd=sqrt(dd);
			index2=j*res+i;		st2=inputIndexArrayPtr[index2];		ed2=inputIndexArrayPtr[index2+1];
			for(kk=st2;kk<ed2;kk++) {
				depth1=fabs(inputDepthArrayPtr[kk]);	depth2=depth1;
				depth1=depth1-dd;	depth2=depth2+dd;

				num2=krMergeSamples(&(depth[0]),num,depth1,depth2);
				num=num2;
			}
		}
		resNum=num;

		//------------------------------------------------------------------------
		//	saving the resultant samples
		if (resNum>0) {  // num samples in total
			resSt=atomicAdd(resBufferIndex2, resNum);
			resSampleStIndex2[index]=resSt;		resSampleNum2[index]=resNum;

			for(k=0;k<resNum;k++) {resDepthBuffer2[resSt+k]=depth[k];}
		}

		index += blockDim.x * gridDim.x;
	}
}
