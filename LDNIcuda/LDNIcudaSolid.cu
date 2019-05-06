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
#include <stdlib.h>
#include <malloc.h>

#include "../common/GL/glew.h"


#include "cuda.h"
#include "cutil.h"
#include "cuda_gl_interop.h"
#include "cutil_gl_error.h"

#include "LDNIcpuSolid.h"
#include "LDNIcudaOperation.h"

#include "LDNIcudaSolid.h"

//---------------------------------------------------------------------------------------------------------------------------------------
extern __global__ void krLDNIcudaSolid_depthSampleAdd(float *depthSamples, float addValue, unsigned int sampleNum);
extern __global__ void krLDNIcudaSolid_fillNewIndexBySampleNumber(unsigned int *newIndexArray, unsigned int *indexArray, int res, int newRes, int sdi, int sdj);
extern __global__ void krLDNIcudaSolid_fillSampleBasedIndexArray(unsigned int *indexArray, int res, unsigned int *sampleBasedIndexArray);
extern __global__ void krLDNIcudaSolid_fillSampleArray(float ox, float oy, float oz, float ww, int res, short nAxis, 
												unsigned int *indexArray, float *dev_sampleNxArray, float *dev_sampleNyArray, float *dev_sampleDepthArray,
												int stIndex, float *devPositionPtr, float *devNormalPtr);

//---------------------------------------------------------------------------------------------------------------------------------------
void LDNIcudaSolid::MallocMemory(int res)
{
	int num;

	FreeMemory();
	m_res=res;	num=res*res;
//	printf("num=%d\n",num);

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(dev_indexArray[0]), (num+1)*sizeof(unsigned int) ) );	
//	printf("cudaMalloc 1 completed\n");
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(dev_indexArray[1]), (num+1)*sizeof(unsigned int) ) );	
//	printf("cudaMalloc 2 completed\n");
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(dev_indexArray[2]), (num+1)*sizeof(unsigned int) ) );	
//	printf("cudaMalloc 3 completed\n");

	CUDA_SAFE_CALL( cudaMemset( (void*)(dev_indexArray[0]), 0, (num+1)*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)(dev_indexArray[1]), 0, (num+1)*sizeof(unsigned int) ) );
	CUDA_SAFE_CALL( cudaMemset( (void*)(dev_indexArray[2]), 0, (num+1)*sizeof(unsigned int) ) );
//	printf("cudaMemset completed\n");
}

void LDNIcudaSolid::FreeMemory()
{
	if (m_res==0) return;

	if (m_xSampleNum!=0) {
		cudaFree( (dev_sampleNxArray[0]) );		cudaFree( (dev_sampleNyArray[0]) );
		cudaFree( (dev_sampleDepthArray[0]) );	m_xSampleNum=0;
	}
	if (m_ySampleNum!=0) {
		cudaFree( (dev_sampleNxArray[1]) );		cudaFree( (dev_sampleNyArray[1]) );
		cudaFree( (dev_sampleDepthArray[1]) );	m_ySampleNum=0;
	}
	if (m_zSampleNum!=0) {
		cudaFree( (dev_sampleNxArray[2]) );		cudaFree( (dev_sampleNyArray[2]) );
		cudaFree( (dev_sampleDepthArray[2]) );	m_zSampleNum=0;
	}

	cudaFree( (dev_indexArray[0]) );
	cudaFree( (dev_indexArray[1]) );
	cudaFree( (dev_indexArray[2]) );

	m_res=0;
//	printf("CUDA memory of LDNIcudaSolid has been released!\n");
}

void LDNIcudaSolid::CleanUpSamples()
{
	int res=m_res;	FreeMemory();	MallocMemory(res);
}

void LDNIcudaSolid::MallocSampleMemory(short nAxis, int sampleNum)
{
	switch(nAxis) {
	case 0:m_xSampleNum=sampleNum;break;
	case 1:m_ySampleNum=sampleNum;break;
	case 2:m_zSampleNum=sampleNum;break;
	}

	CUDA_SAFE_CALL( cudaMalloc( (void**)&(dev_sampleNxArray[nAxis]), sampleNum*sizeof(float) ) );	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(dev_sampleNyArray[nAxis]), sampleNum*sizeof(float) ) );	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(dev_sampleDepthArray[nAxis]), sampleNum*sizeof(float) ) );	
}

void LDNIcudaSolid::CopyIndexArrayToHost(short nAxis, unsigned int* &hostIndexArray)
{
	int num=m_res*m_res;

	hostIndexArray=(unsigned int*)malloc((num+1)*sizeof(unsigned int));
	CUDA_SAFE_CALL( cudaMemcpy( hostIndexArray, dev_indexArray[nAxis], (num+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
}

bool LDNIcudaSolid::FileSave(char *filename)
{
	FILE *fp;

	if (!(fp=fopen(filename,"w+b"))) {printf("LDA file write error! \n"); return false;}

	fwrite(&m_res,sizeof(int),1,fp);
	fwrite(&m_sampleWidth,sizeof(float),1,fp);
	fwrite(m_origin,sizeof(float),3,fp);

	unsigned int *indexArray=(unsigned int*)malloc((m_res*m_res+1)*sizeof(unsigned int));
	for(short nAxis=0;nAxis<3;nAxis++) {
		CUDA_SAFE_CALL( cudaMemcpy( indexArray, dev_indexArray[nAxis], (m_res*m_res+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
		fwrite(indexArray,sizeof(unsigned int),m_res*m_res+1,fp);

		int sampleNum=indexArray[m_res*m_res];
		float* sampleNxArray=(float*)malloc(sampleNum*sizeof(float));
		float* sampleNyArray=(float*)malloc(sampleNum*sizeof(float));
		float* sampleDepthArray=(float*)malloc(sampleNum*sizeof(float));

		CUDA_SAFE_CALL( cudaMemcpy( sampleNxArray, dev_sampleNxArray[nAxis], sampleNum*sizeof(float), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( sampleNyArray, dev_sampleNyArray[nAxis], sampleNum*sizeof(float), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( sampleDepthArray, dev_sampleDepthArray[nAxis], sampleNum*sizeof(float), cudaMemcpyDeviceToHost ) );

		fwrite(sampleNxArray,sizeof(float),sampleNum,fp);
		fwrite(sampleNyArray,sizeof(float),sampleNum,fp);
		fwrite(sampleDepthArray,sizeof(float),sampleNum,fp);

		free(sampleNxArray);	free(sampleNyArray);	free(sampleDepthArray);
	}
	free(indexArray);	fclose(fp);	

	return true;
}

bool LDNIcudaSolid::FileRead(char *filename)
{
	FILE *fp;	int res;

	if (!(fp=fopen(filename,"r+b"))) {printf("LDA file read error! \n"); return false;}

	fread(&res,sizeof(int),1,fp);	
	fread(&m_sampleWidth,sizeof(float),1,fp);
	fread(m_origin,sizeof(float),3,fp);

	MallocMemory(res);
	unsigned int *indexArray=(unsigned int*)malloc((m_res*m_res+1)*sizeof(unsigned int));
	for(short nAxis=0;nAxis<3;nAxis++) {
		fread(indexArray,sizeof(unsigned int),m_res*m_res+1,fp);
		CUDA_SAFE_CALL( cudaMemcpy( dev_indexArray[nAxis], indexArray, (m_res*m_res+1)*sizeof(unsigned int), cudaMemcpyHostToDevice ) );

		int sampleNum=indexArray[m_res*m_res];
		float* sampleNxArray=(float*)malloc(sampleNum*sizeof(float));
		float* sampleNyArray=(float*)malloc(sampleNum*sizeof(float));
		float* sampleDepthArray=(float*)malloc(sampleNum*sizeof(float));
		MallocSampleMemory(nAxis,sampleNum);

		fread(sampleNxArray,sizeof(float),sampleNum,fp);
		fread(sampleNyArray,sizeof(float),sampleNum,fp);
		fread(sampleDepthArray,sizeof(float),sampleNum,fp);

		CUDA_SAFE_CALL( cudaMemcpy( dev_sampleNxArray[nAxis], sampleNxArray, sampleNum*sizeof(float), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dev_sampleNyArray[nAxis], sampleNyArray, sampleNum*sizeof(float), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dev_sampleDepthArray[nAxis], sampleDepthArray, sampleNum*sizeof(float), cudaMemcpyHostToDevice ) );

		free(sampleNxArray);	free(sampleNyArray);	free(sampleDepthArray);
	}
	free(indexArray);	fclose(fp);

	return true;
}

bool LDNIcudaSolid::ExportLDNFile(char *filename)
{
	FILE *fp;	double temp,tempV[3];	int i,j;
	unsigned char nv[3];	double nx,ny,nz;

	printf("export file name %s\n",filename);
	if (!(fp=fopen(filename,"w+b"))) {printf("LDN file write error! \n"); return false;}

	//-------------------------------------------------------------------------------------------------
	//	Write the header information of the file
	fwrite(&m_res,sizeof(int),1,fp);	
	temp=m_sampleWidth;
	fwrite(&temp,sizeof(double),1,fp);
	tempV[0]=m_origin[0];	tempV[1]=m_origin[1];	tempV[2]=m_origin[2];
	fwrite(tempV,sizeof(double),3,fp);

	//-------------------------------------------------------------------------------------------------
	//	Data set preparation on the host
	unsigned int *indexArray[3];	float *sampleNxArray[3],*sampleNyArray[3],*sampleDepthArray[3];
	indexArray[0]=(unsigned int*)malloc((m_res*m_res+1)*sizeof(unsigned int));
	indexArray[1]=(unsigned int*)malloc((m_res*m_res+1)*sizeof(unsigned int));
	indexArray[2]=(unsigned int*)malloc((m_res*m_res+1)*sizeof(unsigned int));
	for(short nAxis=0;nAxis<3;nAxis++) {
		CUDA_SAFE_CALL( cudaMemcpy( indexArray[nAxis], dev_indexArray[nAxis], (m_res*m_res+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );

		int sampleNum=indexArray[nAxis][m_res*m_res];
		sampleNxArray[nAxis]=(float*)malloc(sampleNum*sizeof(float));
		sampleNyArray[nAxis]=(float*)malloc(sampleNum*sizeof(float));
		sampleDepthArray[nAxis]=(float*)malloc(sampleNum*sizeof(float));

		CUDA_SAFE_CALL( cudaMemcpy( sampleNxArray[nAxis], dev_sampleNxArray[nAxis], sampleNum*sizeof(float), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( sampleNyArray[nAxis], dev_sampleNyArray[nAxis], sampleNum*sizeof(float), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( sampleDepthArray[nAxis], dev_sampleDepthArray[nAxis], sampleNum*sizeof(float), cudaMemcpyDeviceToHost ) );
	}

	//-------------------------------------------------------------------------------------------------
	//	Start to write the samples in the way of LDN file
	for(i=0;i<m_res;i++) {
		for(j=0;j<m_res;j++) {
			for(short nCase=0;nCase<3;nCase++) {
				int index=j*m_res+i;
				short num=(short)(indexArray[nCase][index+1]-indexArray[nCase][index]);
				int st=indexArray[nCase][index];
				fwrite(&num,sizeof(short),1,fp);
				for(short k=0;k<num;k++) {
					float depth=fabs(sampleDepthArray[nCase][st+k]);
					fwrite(&depth,sizeof(float),1,fp);
				}
				for(short k=0;k<num;k++) {
					nx=sampleNxArray[nCase][st+k];  if (nx>1.0) nx=1.0;  if (nx<=-1.0) nx=-1.0;
					ny=sampleNyArray[nCase][st+k];  if (ny>1.0) ny=1.0;  if (ny<=-1.0) ny=-1.0;
					nz=1.0-nx*nx-ny*ny;		if (nz<0.0) nz=0.0;		if (nz>1.0) nz=1.0;
					nz=sqrt(nz);	if (sampleDepthArray[nCase][st+k]<0.0) nz=-nz;
					nv[0]=(unsigned char)((nx+1.0)*127.5);
					nv[1]=(unsigned char)((ny+1.0)*127.5);
					nv[2]=(unsigned char)((nz+1.0)*127.5);
					fwrite(nv,sizeof(unsigned char),3,fp);
				}
			}
		}
	}

	//-------------------------------------------------------------------------------------------------
	//	Free the memory
	for(short nAxis=0;nAxis<3;nAxis++) {free(sampleNxArray[nAxis]);	free(sampleNyArray[nAxis]);	free(sampleDepthArray[nAxis]);}
	free(indexArray[0]);	free(indexArray[1]);	free(indexArray[2]);	
	fclose(fp);	

	return true;
}

void LDNIcudaSolid::CopySampleArrayToHost(short nAxis, float* &hostSampleDepthArray, 
										  float* &hostSampleNxArray, float* &hostSampleNyArray)
{
	int sampleNum;

	switch(nAxis) {
	case 0:sampleNum=m_xSampleNum;break;
	case 1:sampleNum=m_ySampleNum;break;
	case 2:sampleNum=m_zSampleNum;break;
	}

	hostSampleDepthArray=(float*)malloc(sampleNum*sizeof(float));
	hostSampleNxArray=(float*)malloc(sampleNum*sizeof(float));
	hostSampleNyArray=(float*)malloc(sampleNum*sizeof(float));

	CUDA_SAFE_CALL( cudaMemcpy( hostSampleDepthArray, (dev_sampleDepthArray[nAxis]), sampleNum*sizeof(float), cudaMemcpyDeviceToHost ) );
	CUDA_SAFE_CALL( cudaMemcpy( hostSampleNxArray, (dev_sampleNxArray[nAxis]), sampleNum*sizeof(float), cudaMemcpyDeviceToHost ) );
	CUDA_SAFE_CALL( cudaMemcpy( hostSampleNyArray, (dev_sampleNyArray[nAxis]), sampleNum*sizeof(float), cudaMemcpyDeviceToHost ) );
}

void LDNIcudaSolid::BuildSampleBasedIndexArray(short nAxis, unsigned int* &sampleBasedIndexArray)
{
	int sampleNum=GetSampleNumber(nAxis);

	CUDA_SAFE_CALL( cudaMalloc( (void**)&sampleBasedIndexArray, sampleNum*sizeof(unsigned int) ) );	
	krLDNIcudaSolid_fillSampleBasedIndexArray<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
									GetIndexArrayPtr(nAxis),m_res,sampleBasedIndexArray);
}

void LDNIcudaSolid::BuildVBOforRendering(GLuint &m_vboPosition, GLuint &m_vboNormal, int &m_vertexNum, bool &m_cudaRegistered)
{
	m_vertexNum=this->GetSampleNumber();	

	glGenBuffersARB( 1, &m_vboPosition );					// Get A Valid Name
	glBindBufferARB( GL_ARRAY_BUFFER_ARB, m_vboPosition );	// Bind The Buffer
	glBufferDataARB( GL_ARRAY_BUFFER_ARB, m_vertexNum*3*sizeof(float), 0, GL_STATIC_DRAW_ARB );	// initial buffer object
	glBindBufferARB( GL_ARRAY_BUFFER_ARB, 0);			
	CUDA_SAFE_CALL( cudaGLRegisterBufferObject(m_vboPosition) );	// register buffer object with CUDA
	CUT_CHECK_ERROR_GL();
	glGenBuffersARB( 1, &m_vboNormal );						// Get A Valid Name
	glBindBufferARB( GL_ARRAY_BUFFER_ARB, m_vboNormal );	// Bind The Buffer
	glBufferDataARB( GL_ARRAY_BUFFER_ARB, m_vertexNum*3*sizeof(float), 0, GL_STATIC_DRAW_ARB );	// initial buffer object
	glBindBufferARB( GL_ARRAY_BUFFER_ARB, 0);
	CUDA_SAFE_CALL( cudaGLRegisterBufferObject(m_vboNormal) );	// register buffer object with CUDA
	CUT_CHECK_ERROR_GL();
	m_cudaRegistered=true;

	float *devPositionPtr,*devNormalPtr;
//	float m_origin[3],m_sampleWidth;		int m_res;
	CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&devPositionPtr, m_vboPosition));
	CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&devNormalPtr, m_vboNormal));
	//--------------------------------------------------------------------------------------------------------------------------------------------------
	krLDNIcudaSolid_fillSampleArray<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(m_origin[0],m_origin[1],m_origin[2],m_sampleWidth,m_res,0,
			dev_indexArray[0],dev_sampleNxArray[0],dev_sampleNyArray[0],dev_sampleDepthArray[0],0,devPositionPtr,devNormalPtr);
	krLDNIcudaSolid_fillSampleArray<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(m_origin[0],m_origin[1],m_origin[2],m_sampleWidth,m_res,1,
			dev_indexArray[1],dev_sampleNxArray[1],dev_sampleNyArray[1],dev_sampleDepthArray[1],m_xSampleNum,devPositionPtr,devNormalPtr);
	krLDNIcudaSolid_fillSampleArray<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(m_origin[0],m_origin[1],m_origin[2],m_sampleWidth,m_res,2,
			dev_indexArray[2],dev_sampleNxArray[2],dev_sampleNyArray[2],dev_sampleDepthArray[2],m_xSampleNum+m_ySampleNum,devPositionPtr,devNormalPtr);
	//--------------------------------------------------------------------------------------------------------------------------------------------------
	CUDA_SAFE_CALL(cudaGLUnmapBufferObject(m_vboNormal));
	CUDA_SAFE_CALL(cudaGLUnmapBufferObject(m_vboPosition));
}

//---------------------------------------------------------------------------------------------------------------
//	kernel functions
//---------------------------------------------------------------------------------------------------------------

__global__ void krLDNIcudaSolid_fillSampleArray(float ox, float oy, float oz, float ww, int res, short nAxis, 
												unsigned int *indexArray, float *dev_sampleNxArray, float *dev_sampleNyArray, float *dev_sampleDepthArray,
												int stIndex, float *devPositionPtr, float *devNormalPtr)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int i,ii,jj,index,arrsize=res*res;
	float depth,origin[3],nx,ny,nz;

	origin[0]=ox;	origin[1]=oy;	origin[2]=oz;

	while (tid<arrsize) {
		for(i=indexArray[tid];i<indexArray[tid+1];i++) {
			index=(stIndex+i)*3;	depth=dev_sampleDepthArray[i];
			ii=tid%res;	jj=tid/res;

			devPositionPtr[index+nAxis]=origin[nAxis]+fabs(depth);				// x-coord of position
			devPositionPtr[index+(nAxis+1)%3]=origin[(nAxis+1)%3]+ww*(float)ii;	// y-coord of position
			devPositionPtr[index+(nAxis+2)%3]=origin[(nAxis+2)%3]+ww*(float)jj;	// z-coord of position
			nx=dev_sampleNxArray[i];	ny=dev_sampleNyArray[i];
			nz=1.0-(nx*nx+ny*ny);
			if (nz<0.0) nz=0.0; else nz=sqrt(nz);
			if (depth<0.0) nz=-nz;
			devNormalPtr[index]=nx;		// x-component of normal
			devNormalPtr[index+1]=ny;	// y-component of normal
			devNormalPtr[index+2]=nz;	// z-component of normal
		}

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIcudaSolid_fillSampleBasedIndexArray(unsigned int *indexArray, int res, unsigned int *sampleBasedIndexArray)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int i,arrsize=res*res;

	while (tid<arrsize) {
		for(i=indexArray[tid];i<indexArray[tid+1];i++) {
			sampleBasedIndexArray[i]=tid;
		}

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIcudaSolid_fillNewIndexBySampleNumber(unsigned int *newIndexArray, unsigned int *indexArray, 
														   int res, int newRes, int sdi, int sdj)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int arrsize=res*res;
	int newIndex,i,j;

	while (tid<arrsize) {
		j = tid / res;	i = tid % res;
		newIndex = (j+sdj)*newRes + (i+sdi);

		newIndexArray[newIndex] = indexArray[tid+1] - indexArray[tid];

		tid += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIcudaSolid_depthSampleAdd(float *depthSamples, float addValue, unsigned int sampleNum)
{
	int tid=threadIdx.x+blockIdx.x*blockDim.x;

	while (tid<sampleNum) {
		if (depthSamples[tid]>0.0f)
			depthSamples[tid] = depthSamples[tid] + addValue;
		else
			depthSamples[tid] = - ((-depthSamples[tid]) + addValue);

		tid += blockDim.x * gridDim.x;
	}
}
