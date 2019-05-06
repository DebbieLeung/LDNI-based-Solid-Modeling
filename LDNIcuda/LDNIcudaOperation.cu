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

#include "../common/GL/glew.h"

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

#define GPU_BASED_SCAN		true

extern GLK _pGLK;
extern bool _bExpandableWorkingSpace;



//--------------------------------------------------------------------------------------------
texture<float4,2> tex2DFloat4In;

extern __global__ void krLDNISuperUnion_CopySamples(float *devNxArrayPtr, float *devNyArrayPtr, float *devDepthArrayPtr, 
											 int n, int arrsize, int res, unsigned int *devIndexArrayPtr);

extern __global__ void krLDNIBoolean_SuperUnionOnRays(float *devNxArrayPtr, float *devNyArrayPtr, float *devDepthArrayPtr, unsigned int *devIndexArrayPtr,
									  unsigned int *devIndexArrayPtrRes, int arrsize);

extern __global__ void krLDNIBoolean_IdentifyEnterLeaveOnRays(float *devNxArrayPtr, float *devNyArrayPtr, float *devDepthArrayPtr, unsigned int *devIndexArrayPtr, int arrsize);

extern __global__ void krLDNIBoolean_BooleanOnRays(float *devNxArrayPtrA, float *devNyArrayPtrA, float *devDepthArrayPtrA, unsigned int *devIndexArrayPtrA,
												   float *devNxArrayPtrB, float *devNyArrayPtrB, float *devDepthArrayPtrB, unsigned int *devIndexArrayPtrB, 
												   unsigned int *devIndexArrayPtrRes, int arrsize, short nOperationType);
extern __global__ void krLDNIBoolean_ResultSampleCollection(float *devNxArrayPtrA, float *devNyArrayPtrA, float *devDepthArrayPtrA, unsigned int *devIndexArrayPtrA,
												   float *devNxArrayPtrB, float *devNyArrayPtrB, float *devDepthArrayPtrB, unsigned int *devIndexArrayPtrB, 
												   float *devNxArrayPtrRes, float *devNyArrayPtrRes, float *devDepthArrayPtrRes, unsigned int *devIndexArrayPtrRes, int arrsize);


extern __global__ void krLDNIBoolean_ResultSampleCollection(float *devNxArrayPtrA, float *devNyArrayPtrA, float *devDepthArrayPtrA, unsigned int *devIndexArrayPtrA,
															float *devNxArrayPtrRes, float *devNyArrayPtrRes, float *devDepthArrayPtrRes, unsigned int *devIndexArrayPtrRes, int arrsize, float width, float gwidth);


extern __global__ void krLDNIBilateralNormalFilter_PerRay(unsigned int* xIndexArray, unsigned int* yIndexArray, unsigned int* zIndexArray,
									float* xNxArray, float* yNxArray, float* zNxArray, float* xNyArray, float* yNyArray, float* zNyArray, 
									float* xDepthArray, float* yDepthArray, float* zDepthArray, float *buffer, 
									int arrsize, short nAxis, int res, float ww, float ox, float oy, float oz, unsigned int nSupportSize, float normalPara);
extern __global__ void krLDNIBilateralNormalFilter_PerSample(unsigned int* xIndexArray, unsigned int* yIndexArray, unsigned int* zIndexArray,
									float* xNxArray, float* yNxArray, float* zNxArray, float* xNyArray, float* yNyArray, float* zNyArray, 
									float* xDepthArray, float* yDepthArray, float* zDepthArray, float *buffer, 
									int sampleNum, short nAxis, int res, float ww, unsigned int nSupportSize, float normalPara);
extern __global__ void krLDNINormalProcessing_PreProc(unsigned int* indexArray, float *buffer, int res, int arrsize);
extern __global__ void krLDNINormalProcessing_Update(int sampleNum, float *nxArray, float *nyArray, float *depthArray, float *buffer);
extern __global__ void krLDNINormalProcessing_OrientationCorrectionByVoting(
									unsigned int* xIndexArray, unsigned int* yIndexArray, unsigned int* zIndexArray,
									float* xNxArray, float* yNxArray, float* zNxArray, float* xNyArray, float* yNyArray, float* zNyArray, 
									float* xDepthArray, float* yDepthArray, float* zDepthArray, float *buffer, 
									int sampleNum, short nAxis, int res, float ww, unsigned int nSupportSize);

extern __global__ void krLDNINormalReconstruction_PerSample(unsigned int* xIndexArray, unsigned int* yIndexArray, unsigned int* zIndexArray, 
									float* xNxArray, float* yNxArray, float* zNxArray, float* xNyArray, float* yNyArray, float* zNyArray, 
									float* xDepthArray, float* yDepthArray, float* zDepthArray, float *buffer, 
									int sampleNum, short nAxis, int res, float ww, unsigned int nSupportSize); 

extern __global__ void krLDNISampling_SortSamples(float *devNxArrayPtr, float *devNyArrayPtr, float *devDepthArrayPtr, 
												  int arrsize, unsigned int *devIndexArrayPtr);
extern __global__ void krLDNISampling_CopySamples(float *devNxArrayPtr, float *devNyArrayPtr, float *devDepthArrayPtr, 
												  int n, int arrsize, float width, float sampleWidth, int res, 
												  unsigned int *devIndexArrayPtr);
extern __global__ void krLDNISampling_CopyIndexAndFindMax(unsigned char *devStencilBufferPtr, unsigned int *devIndexArrayPtr, 
														  unsigned int *devResArrayPtr, int arrsize );

extern __global__ void krLDNIcudaSolid_depthSampleAdd(float *depthSamples, float addValue, unsigned int sampleNum);
extern __global__ void krLDNIcudaSolid_fillNewIndexBySampleNumber(unsigned int *newIndexArray, unsigned int *indexArray, int res, int newRes, int sdi, int sdj);

extern __global__ void krLDNIRegularization_RegularizationOnRays(float *devNxArrayPtr, float *devNyArrayPtr, float *devDepthArrayPtr, 
									unsigned int *devIndexArrayPtr, unsigned int *devIndexArrayPtrRes, int arrsize, float eps);
extern __global__ void krLDNIRegularization_ResultSampleCollection(float *devNxArrayPtr, float *devNyArrayPtr, float *devDepthArrayPtr, 
									unsigned int *devIndexArrayPtr,	float *devNxArrayPtrRes, float *devNyArrayPtrRes, float *devDepthArrayPtrRes, 
									unsigned int *devIndexArrayPtrRes, int arrsize);

extern bool initGLInteroperabilityOnCUDA(int major, int minor);

//--------------------------------------------------------------------------------------------

bool LDNIcudaOperation::MultiObjectSamplingInOneSolid(LDNIcudaSolid* &solid, GLKObList* meshlist, float boundingBox[], int res)
{
	float origin[3],gWidth;
	char fileadd[256];
	long time=clock(),totalTime=clock();
	//---------------------------------------------------------------------------------
	solid=new LDNIcudaSolid;
	solid->MallocMemory(res);
	solid->SetBoundingBox(boundingBox);
	gWidth=(boundingBox[1]-boundingBox[0])/(float)res;
	solid->SetSampleWidth(gWidth);
	origin[0]=boundingBox[0]+gWidth*0.5f;
	origin[1]=boundingBox[2]+gWidth*0.5f;
	origin[2]=boundingBox[4]+gWidth*0.5f;
	solid->SetOrigin(origin[0],origin[1],origin[2]);

	//---------------------------------------------------------------------------------
	//	For using OpenGL Shading Language to implement the sampling procedure
	if (glewInit() != GLEW_OK) {printf("glewInit failed. Exiting...\n");	return false;}
	
	//-----------------------------------------------------------------------------------------
	GLhandleARB g_programObj, g_vertexShader, g_GeometryShader, g_FragShader;
	const char *VshaderString[1],*GshaderString[1], *FshaderString[1];
	GLint bCompiled = 0, bLinked = 0;
	char str[4096] = "";		
	//-----------------------------------------------------------------------------------------
	//	Step 1: Setup the shaders 
	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"SuperUnionLDNIVertexShader.vert");
	g_vertexShader = glCreateShaderObjectARB( GL_VERTEX_SHADER_ARB );
	unsigned char *ShaderAssembly = _readShaderFile( fileadd );
	VshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_vertexShader, 1, VshaderString, NULL );
	glCompileShaderARB( g_vertexShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_vertexShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {
		glGetInfoLogARB(g_vertexShader, sizeof(str), NULL, str);
		printf("Warning: Vertex Shader Compile Error \n%s\n",str);	return false;
	}
	//-----------------------------------------------------------------------------
	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"SuperUnionLDNIGeometryShader.geo");
	g_GeometryShader = glCreateShaderObjectARB( GL_GEOMETRY_SHADER_EXT );
	ShaderAssembly = _readShaderFile( fileadd );
	GshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_GeometryShader, 1, GshaderString, NULL );
	glCompileShaderARB( g_GeometryShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_GeometryShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {
		glGetInfoLogARB(g_GeometryShader, sizeof(str), NULL, str);
		printf("Warning: Geo Shader Compile Error\n%s\n",str);		return false;
	}
	//-----------------------------------------------------------------------------
	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"SuperUnionLDNIFragmentShader.frag");
	g_FragShader = glCreateShaderObjectARB( GL_FRAGMENT_SHADER_ARB );
	ShaderAssembly = _readShaderFile( fileadd );
	FshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_FragShader, 1, FshaderString, NULL );
	glCompileShaderARB( g_FragShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_FragShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {
		glGetInfoLogARB(g_FragShader, sizeof(str), NULL, str);
		printf("Warning: Vertex Shader Compile Error\n\n");	return false;
	}
	g_programObj = glCreateProgramObjectARB();
	if (glGetError()!=GL_NO_ERROR) printf("Error: OpenGL!\n\n");
	glAttachObjectARB( g_programObj, g_vertexShader );		if (glGetError()!=GL_NO_ERROR) printf("Error: attach Vertex Shader!\n\n");
	glAttachObjectARB( g_programObj, g_GeometryShader );	if (glGetError()!=GL_NO_ERROR) printf("Error: attach Geometry Shader!\n\n");
	glAttachObjectARB( g_programObj, g_FragShader );		if (glGetError()!=GL_NO_ERROR) printf("Error: attach Fragment Shader!\n\n");
	//-----------------------------------------------------------------------------
	//	Configuration setting for geometry shader
	glLinkProgramARB( g_programObj);
	glGetObjectParameterivARB( g_programObj, GL_OBJECT_LINK_STATUS_ARB, &bLinked );
	if( bLinked == false ) {
		glGetInfoLogARB( g_programObj, sizeof(str), NULL, str );
		printf("Linking Fail: %s\n",str);	return false;
	}

	//-----------------------------------------------------------------------------------------
	//	Step 2:  creating vertex and index array buffer
	glGetError();	// for clean-up the error generated before
	int meshNum = meshlist->GetCount();
	GLuint* vbo = (GLuint*)malloc(meshNum*sizeof(GLuint));
	GLuint* vboInd = (GLuint*)malloc(meshNum*sizeof(GLuint));
	GLKPOSITION Pos;
	int nodeNum,faceNum,i=0,j=0;
	float* verTex;
	float* tempver;
	int* inDex;
	int* tempinD;
	unsigned int* meshptr;
	int* indexCount;
	indexCount = (int*)malloc(meshNum*sizeof(int));

	printf("Mesh Num : %d \n",meshNum);
	verTex = (float*)malloc(sizeof(float));
	inDex = (int*)malloc(sizeof(int));

	glGenBuffers(meshNum, vbo);
	glGenBuffers(meshNum, vboInd);
	for(Pos=meshlist->GetHeadPosition();Pos!=NULL;j++) {
		QuadTrglMesh *mesh=(QuadTrglMesh *)(meshlist->GetNext(Pos));
		nodeNum = mesh->GetNodeNumber();
		faceNum = mesh->GetFaceNumber();
		printf("node num %d %d\n",nodeNum,faceNum);
		tempver = (float*)realloc(verTex,nodeNum*3*sizeof(float));
		if (tempver!=NULL)
			verTex = tempver;
		else
		{
			free(verTex);
			printf("realloc memeory error!!");
			return false;
		}
		tempinD = (int*)realloc(inDex,faceNum*3*sizeof(int));
		if (tempinD!=NULL)
			inDex = tempinD;
		else
		{
			free(inDex);
			printf("realloc memeory error!!");
			return false;
		}
		memset(verTex,0,nodeNum*3*sizeof(float));
		memcpy(verTex,mesh->GetNodeArrayPtr(),nodeNum*3*sizeof(float));
		memset(inDex,0,faceNum*3*sizeof(int));
		meshptr = mesh->GetFaceTablePtr();
		for(i=0; i < faceNum; i++)
		{	inDex[3*i] = meshptr[4*i]-1;	inDex[3*i+1] = meshptr[4*i+1]-1;	inDex[3*i+2] = meshptr[4*i+2]-1;
		}
		indexCount[j] = faceNum*3;


		glBindBuffer(GL_ARRAY_BUFFER, vbo[j]);
		glBufferData(GL_ARRAY_BUFFER, nodeNum*3*sizeof(GLfloat), 0, GL_STATIC_DRAW);
		glBufferSubData(GL_ARRAY_BUFFER, 0, nodeNum*3*sizeof(GLfloat), verTex);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER_ARB, vboInd[j]);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER_ARB, faceNum*3*sizeof(GL_UNSIGNED_INT), 0, GL_STATIC_DRAW);
		glBufferSubData(GL_ELEMENT_ARRAY_BUFFER_ARB, 0, faceNum*3*sizeof(GL_UNSIGNED_INT), inDex);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);





		if (glGetError()!=GL_NO_ERROR) printf("Error: buffer binding!\n\n");
	}

	free(verTex);
	free(inDex);



	//-----------------------------------------------------------------------------------------
	float centerPos[3];
	centerPos[0]=(boundingBox[0]+boundingBox[1])*0.5f;
	centerPos[1]=(boundingBox[2]+boundingBox[3])*0.5f;
	centerPos[2]=(boundingBox[4]+boundingBox[5])*0.5f;
	glUseProgramObjectARB(g_programObj);
	{

		_decomposeLDNIByFBOPBO(solid, vbo, vboInd, meshNum, centerPos, g_programObj,indexCount);

	}
	glUseProgramObjectARB(0);

	//-----------------------------------------------------------------------------------------
	//	Step 6:  free the memory
	time=clock();
	//-----------------------------------------------------------------------------------------
	glDeleteBuffers(meshNum, vboInd);
	glDeleteBuffers(meshNum, vbo);
	glDeleteObjectARB( g_vertexShader);
	glDeleteObjectARB( g_GeometryShader);
	glDeleteObjectARB( g_FragShader);
	glDeleteObjectARB( g_programObj);
	free(indexCount);
	//------------------------------------------------------------------------
	printf("\nMemory clean-up time is %ld (ms)\n",clock()-time);
	printf("--------------------------------------------------------------\n");
	printf("Total time for sampling is %ld (ms)\n\n",clock()-totalTime);

	return true;
}

bool LDNIcudaOperation::SuperUnionOperation(LDNIcudaSolid* &solid, GLKObList* meshlist, float boundingBox[],int res)
{
	long time=clock(),totalTime=clock();

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

	if (!MultiObjectSamplingInOneSolid(solid, meshlist, boundingBox, res)) return false;
	
	if (!_UnionMultiObjects(solid, res)) return false;



	return true;
}

bool LDNIcudaOperation::_UnionMultiObjects(LDNIcudaSolid* &inputSolid, int res)
{
	unsigned int arrsize=res*res;	
	float width, gwidth;
	float bbox[6];

	if (inputSolid->GetSampleNumber()==0) {
		printf("No Samples!");
		return false;
	}

	inputSolid->GetBoundingBox(bbox);
	width = bbox[1]-bbox[0];
	gwidth = inputSolid->GetSampleWidth();


	//-----------------------------------------------------------------------------------
	//	Step 1: Initialization 
	long time=clock();		
	unsigned int *devIndexArrayResPtr;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&devIndexArrayResPtr, (arrsize+1)*sizeof(unsigned int) ) );
	//-----------------------------------------------------------------------------------
	//	Step 2: computing the Boolean operation results on LDNIs
	for(short nAxis=0;nAxis<3;nAxis++) {
		//---------------------------------------------------------------------------------------------
		//	Sub-step 1: intialization
		CUDA_SAFE_CALL( cudaMemset( (void*)devIndexArrayResPtr, 0, (arrsize+1)*sizeof(unsigned int) ) );
		//---------------------------------------------------------------------------------------------
		float *devNxArrayPtr=inputSolid->GetSampleNxArrayPtr(nAxis);	
		float *devNyArrayPtr=inputSolid->GetSampleNyArrayPtr(nAxis);
		float *devDepthArrayPtr=inputSolid->GetSampleDepthArrayPtr(nAxis);	//if (devDepthArrayPtrA==NULL) printf("Empty ");
		unsigned int *devIndexArrayPtr=inputSolid->GetIndexArrayPtr(nAxis);
	
		//---------------------------------------------------------------------------------------------
		//	Sub-step 2: identify the entering and leaving samples ray by ray
		krLDNIBoolean_IdentifyEnterLeaveOnRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(devNxArrayPtr, devNyArrayPtr, devDepthArrayPtr, devIndexArrayPtr, arrsize);

		//---------------------------------------------------------------------------------------------
		//	Sub-step 3: Sorting the entering and leaving samples ray by ray
		krLDNISampling_SortSamples<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(devNxArrayPtr, devNyArrayPtr, devDepthArrayPtr, arrsize, devIndexArrayPtr);

		//---------------------------------------------------------------------------------------------
		//	Sub-step 4: Super - union samples ray by ray
		krLDNIBoolean_SuperUnionOnRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(devNxArrayPtr, devNyArrayPtr, devDepthArrayPtr, devIndexArrayPtr,
																			devIndexArrayResPtr, arrsize);
		//---------------------------------------------------------------------------------------------
		//	Sub-step 5: compaction of index array
		thrust::device_ptr<unsigned int> dev_ptr(devIndexArrayResPtr); //	Wrap raw pointers with dev_ptr
		thrust::exclusive_scan(dev_ptr, dev_ptr+(arrsize+1), dev_ptr); //	in-place scan
		unsigned int sampleNum=dev_ptr[arrsize];
		//printf("max sample ----- %d\n",sampleNum);

		//---------------------------------------------------------------------------------------------
		//	Sub-step 6: collecting the resultant samples into the sampleArray of solidTileA				
		float *newDevNxArrayPtr, *newDevNyArrayPtr, *newDevDepthArrayPtr;
		inputSolid->MallocSampleMemory(nAxis, sampleNum);
		newDevNxArrayPtr=inputSolid->GetSampleNxArrayPtr(nAxis);
		newDevNyArrayPtr=inputSolid->GetSampleNyArrayPtr(nAxis);
		newDevDepthArrayPtr=inputSolid->GetSampleDepthArrayPtr(nAxis);
		krLDNIBoolean_ResultSampleCollection<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
			devNxArrayPtr, devNyArrayPtr, devDepthArrayPtr, devIndexArrayPtr,
			newDevNxArrayPtr, newDevNyArrayPtr, newDevDepthArrayPtr, devIndexArrayResPtr, arrsize, width, gwidth);
		CUDA_SAFE_CALL( cudaMemcpy( devIndexArrayPtr, devIndexArrayResPtr, (arrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToDevice ) );

		cudaFree(devNxArrayPtr);	cudaFree(devNyArrayPtr);	cudaFree(devDepthArrayPtr);
	}

	//-----------------------------------------------------------------------------------
	//	Step 3: free the memory
	cudaFree(devIndexArrayResPtr);
	printf("Boolean Operation Time (ms): %ld\n",clock()-time);

	return true;
}


void LDNIcudaOperation::_decomposeLDNIByFBOPBO(LDNIcudaSolid *solid, GLuint* vbo, GLuint* vboI, int mesh_count, float Cent[], GLhandleARB g_programObj, int indexCount[])
{
	unsigned int n_max,i,n,mesh_ID;
	float gWidth,origin[3];
	unsigned int overall_n_max=0;
	long readbackTime=0, sortingTime=0, tempTime;
	GLint id0,id1;
	

	cudaEvent_t     startClock, stopClock;
	CUDA_SAFE_CALL( cudaEventCreate( &startClock ) );
	CUDA_SAFE_CALL( cudaEventCreate( &stopClock ) );

	tempTime=clock();
	//------------------------------------------------------------------------
	//	Preparation
	int nRes=solid->GetResolution();		gWidth=solid->GetSampleWidth();
	float width=gWidth*(float)nRes;
	solid->GetOrigin(origin[0],origin[1],origin[2]);
	int arrsize=nRes*nRes;

	//------------------------------------------------------------------------
	//	Step 1: Setup the rendering environment
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_STENCIL_TEST);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_POLYGON_OFFSET_FILL);
	glDisable(GL_POLYGON_OFFSET_LINE);
	glDisable(GL_BLEND);	
	glDisable(GL_POLYGON_SMOOTH);	// turn off anti-aliasing
	glDisable(GL_POINT_SMOOTH);
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_MAP_COLOR);	glDisable(GL_DITHER);
	glShadeModel(GL_FLAT);
	glDisable(GL_LIGHTING);   glDisable(GL_LIGHT0);
	glDisable(GL_LOGIC_OP);
	glDisable(GL_COLOR_MATERIAL);
	glDisable(GL_ALPHA_TEST);
	glGetError();	// for clean-up the error generated before
	//------------------------------------------------------------------------
	//	create the FBO objects and texture for rendering
	if (glewIsSupported("GL_EXT_framebuffer_object") == 0) printf("Warning: FBO is not supported!\n");
	if (glGetError()!=GL_NO_ERROR) printf("Error: before framebuffer generation!\n");
	//------------------------------------------------------------------------
	GLuint fbo;
	glGenFramebuffersEXT(1, &fbo);
	if (glGetError()!=GL_NO_ERROR) printf("Error: framebuffer generation!\n");
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
	if (glGetError()!=GL_NO_ERROR) printf("Error: framebuffer binding!\n");
	//------------------------------------------------------------------------
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, nRes, nRes, 0, GL_RGBA, GL_FLOAT, 0);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, tex, 0);
	if (glGetError()!=GL_NO_ERROR) printf("Error: attaching texture to framebuffer generation!\n");
	cudaGraphicsResource *sampleTex_resource;
	CUDA_SAFE_CALL( cudaGraphicsGLRegisterImage(&sampleTex_resource, tex, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly) );
	//------------------------------------------------------------------------
	GLuint depth_and_stencil_rb;
	glGenRenderbuffersEXT(1, &depth_and_stencil_rb);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depth_and_stencil_rb);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_STENCIL_EXT, nRes, nRes);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depth_and_stencil_rb);
	if (glGetError()!=GL_NO_ERROR) printf("Error: attaching renderbuffer of depth-buffer to framebuffer generation!\n");
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depth_and_stencil_rb);
	if (glGetError()!=GL_NO_ERROR) printf("Error: attaching renderbuffer of stencil-buffer to framebuffer generation!\n");
	//------------------------------------------------------------------------
	GLuint indexPBO;
	glGenBuffers(1,&indexPBO);	//	generation of PBO for index array readback
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, indexPBO);
	glBufferData(GL_PIXEL_PACK_BUFFER_ARB, nRes*nRes*sizeof(unsigned char), NULL, GL_STREAM_READ_ARB);
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	CUDA_SAFE_CALL( cudaGLRegisterBufferObject(indexPBO) );
	//------------------------------------------------------------------------
	if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT)!=GL_FRAMEBUFFER_COMPLETE_EXT) 
		printf("Warning: the setting for rendering on FBO is not correct!\n");
	else
		printf("FBO has been created successfully!\n");
	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0,0,nRes,nRes);
	printf("Preparation time: %ld (ms)\n",clock()-tempTime);

	id0 = glGetUniformLocationARB(g_programObj,"Cent");
	glUniform3fARB(id0,Cent[0],Cent[1],Cent[2]);
	id1 = glGetUniformLocationARB(g_programObj,"mesh_ID");
	//------------------------------------------------------------------------
	//	Step 2: Rendering to get the Hermite samples
	for(short nAxis=0; nAxis<3; nAxis++) { 
		//---------------------------------------------------------------------------------------
		//	Rendering step 1: setting the viewing window
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		//---------------------------------------------------------------------------------------
		//	The eye is located at (0, 0, 0), the near clipping plane is at the z=0 plane
		//		the far clipping plane is at the z=(boundingBox[5]-boundingBox[4]) plane
		glOrtho(-width*0.5f,width*0.5f,-width*0.5f,width*0.5f,width*0.5f,-width*0.5f);
		//	Note that:	in "glOrtho(left,right,bottom,top,near,far);"
		//		(left,right,bottom,top) are located at the boundary of pixel instead of
		//		the center of pixels
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		//---------------------------------------------------------------------------------------
		//	Rendering step 2: determine the number of layers
		glClearColor( 1.0f, 1.0f, 1.0f, 1.0f );	
		glClearDepth(1.0);
		glClearStencil(0);	glColor3f(1,1,1);
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		glDepthFunc(GL_ALWAYS);
		glStencilFunc(GL_GREATER, 1, 0xff);
		glStencilOp(GL_INCR, GL_INCR, GL_INCR);
		glPushMatrix();
		switch(nAxis) {
			case 0:{glRotatef(-90,0,1,0);	glRotatef(-90,1,0,0); }break;
			case 1:{glRotatef(90,0,1,0);	glRotatef(90,0,0,1);  }break;
		}


		glEnableClientState( GL_VERTEX_ARRAY );	
		
		for(mesh_ID = 0; mesh_ID < mesh_count; mesh_ID++)
		{
			glUniform1iARB(id1,mesh_ID);
			glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo[mesh_ID]);	
			glVertexPointer(3, GL_FLOAT, 0, 0);
			glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, vboI[mesh_ID]);
			glDrawElements(GL_TRIANGLES, indexCount[mesh_ID], GL_UNSIGNED_INT, 0);
			
			

		}
		glDisableClientState( GL_VERTEX_ARRAY );

		glFlush();

		
		//--------------------------------------------------------------------------------------------------------
		//	reading stencil buffer into the device memory of CUDA
		tempTime=clock();
		glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, indexPBO);
		GLint OldPackAlignment;
		glGetIntegerv(GL_PACK_ALIGNMENT,&OldPackAlignment); 
		glPixelStorei(GL_PACK_ALIGNMENT,1);	// Important!!! Without this, the read-back could be abnormal.
		glReadPixels(0,0,nRes,nRes,GL_STENCIL_INDEX,GL_UNSIGNED_BYTE,0);
		glPixelStorei(GL_PACK_ALIGNMENT,OldPackAlignment);
		//--------------------------------------------------------------------------------------------------------
		unsigned char *devStencilBufferPtr;
		unsigned int *devResArrayPtr;
		unsigned int *devIndexArrayPtr=solid->GetIndexArrayPtr(nAxis);
		CUDA_SAFE_CALL( cudaGLMapBufferObject( (void **)&devStencilBufferPtr, indexPBO) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&devResArrayPtr, BLOCKS_PER_GRID*sizeof(unsigned int) ) );
		//--------------------------------------------------------------------------------------------------------
		//	building the indexArray on device
		krLDNISampling_CopyIndexAndFindMax<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(devStencilBufferPtr,
			devIndexArrayPtr,devResArrayPtr,arrsize);

		//--------------------------------------------------------------------------------------------------------
		//	read back the max number of layers -- "n_max"
		unsigned int* resArrayPtr;
		resArrayPtr=(unsigned int *)malloc(BLOCKS_PER_GRID*sizeof(unsigned int));
		CUDA_SAFE_CALL( cudaMemcpy( resArrayPtr, devResArrayPtr, BLOCKS_PER_GRID*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
		n_max=0;
		for(i=0;i<BLOCKS_PER_GRID;i++) n_max = MAX(n_max,resArrayPtr[i]);
		cudaFree(devResArrayPtr);		free(resArrayPtr);
		//--------------------------------------------------------------------------------------------------------
		//	read back the number of samples -- "sampleNum"
		unsigned int sampleNum=0;
		tempTime=clock()-tempTime;		//readbackTime+=tempTime;
		printf("Stencil buffer processing time: %ld (ms)\n",tempTime);

		long scanTime=clock();
		//	for debug purpose
		resArrayPtr=(unsigned int *)malloc((arrsize+1)*sizeof(unsigned int));
		CUDA_SAFE_CALL( cudaMemcpy( resArrayPtr, devIndexArrayPtr, (arrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
		sampleNum=0;
		for(int k=0;k<arrsize;k++) {sampleNum+=resArrayPtr[k];	resArrayPtr[k]=sampleNum;}	
		for(int k=arrsize;k>0;k--) {resArrayPtr[k]=resArrayPtr[k-1];}	
		resArrayPtr[0]=0;
		CUDA_SAFE_CALL( cudaMemcpy( devIndexArrayPtr, resArrayPtr, (arrsize+1)*sizeof(unsigned int), cudaMemcpyHostToDevice ) );
		free(resArrayPtr);
		scanTime=clock()-scanTime;	printf("Scanning time: %ld (ms)\n",scanTime);

		//--------------------------------------------------------------------------------------------------------
		CUDA_SAFE_CALL( cudaGLUnmapBufferObject( indexPBO ) );
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
		printf("n_max=%d   sampleNum=%d\n",n_max,sampleNum);
		if (n_max>overall_n_max) overall_n_max=n_max;
		if (sampleNum==0) continue;

		//---------------------------------------------------------------------------------------
		//	Rendering step 3: decomposing the Layered Depth Images (LDIs) and record its corresponding normals
		solid->MallocSampleMemory(nAxis,sampleNum);	
		float* devNxArrayPtr=solid->GetSampleNxArrayPtr(nAxis);
		float* devNyArrayPtr=solid->GetSampleNyArrayPtr(nAxis);
		float* devDepthArrayPtr=solid->GetSampleDepthArrayPtr(nAxis);
		tempTime=clock();
		for(n=1;n<=n_max;n++) {
			CUDA_SAFE_CALL( cudaGraphicsMapResources( 1, &sampleTex_resource, NULL ) );
			cudaArray *in_array;
			CUDA_SAFE_CALL( cudaGraphicsSubResourceGetMappedArray( &in_array, sampleTex_resource, 0, 0));
			CUDA_SAFE_CALL( cudaBindTextureToArray(tex2DFloat4In, in_array) );
			//--------------------------------------------------------------------------------------------------------
			//	fill the sampleArray on device
			krLDNISuperUnion_CopySamples<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(devNxArrayPtr, devNyArrayPtr, 
				devDepthArrayPtr, n, arrsize, nRes, devIndexArrayPtr);
			CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 1, &sampleTex_resource, NULL ) );
			if (n==n_max) break;

			glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
			glStencilFunc(GL_GREATER, n+1, 0xff);
			glStencilOp(GL_KEEP, GL_INCR, GL_INCR);
			{
				
				glEnableClientState( GL_VERTEX_ARRAY );		
				for(mesh_ID = 0; mesh_ID < mesh_count; mesh_ID++)
				{
					glUniform1iARB(id1,mesh_ID);
					glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo[mesh_ID]);	
					glVertexPointer(3, GL_FLOAT, 0, 0);
					glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, vboI[mesh_ID]);
					glDrawElements(GL_TRIANGLES, indexCount[mesh_ID], GL_UNSIGNED_INT, 0);
					

				}
				glDisableClientState( GL_VERTEX_ARRAY );
			}
			glFlush();
		}
		tempTime=clock()-tempTime;		readbackTime+=tempTime;

		//------------------------------------------------------------------------
		//	Rendering step 4: sorting the samples
		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );
		CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
		krLDNISampling_SortSamples<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(devNxArrayPtr, devNyArrayPtr, 
			devDepthArrayPtr, arrsize, devIndexArrayPtr);
		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );
		CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );
		float   elapsedTime;
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,
			startClock, stopClock ) );
		printf( "Sorting time is:  %3.1f (ms)\n", elapsedTime );		
		sortingTime+=(long)elapsedTime;
	}

	//------------------------------------------------------------------------------------
	//	Step 3: Set the rendering parameters back
	//------------------------------------------------------------------------------------
	//	detach FBO
	glPopAttrib();
	//	release memory for PBO and cuda's map	
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	CUDA_SAFE_CALL( cudaGLUnregisterBufferObject( indexPBO ) );
	glDeleteBuffers(1, &indexPBO);
	CUDA_SAFE_CALL( cudaGraphicsUnregisterResource( sampleTex_resource) );
	//	release memory for the 2D texture
	glBindTexture(GL_TEXTURE_2D, 0);
	glDeleteTextures(1, &tex);
	//	release memory for the frame-buffer object
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glDeleteFramebuffersEXT(1, &fbo);
	//	release memory for the render-buffer object
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
	glDeleteRenderbuffersEXT(1, &depth_and_stencil_rb);
	//------------------------------------------------------------------------------------
	glEnable(GL_POLYGON_OFFSET_FILL);
	glEnable(GL_POLYGON_OFFSET_LINE);
	glEnable(GL_BLEND);	
	glEnable(GL_DITHER);
	glDisable(GL_STENCIL_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_MAP_COLOR);				
	glShadeModel(GL_SMOOTH);   
	glEnable(GL_LIGHTING);  glEnable(GL_LIGHT0);
	//	glEnable(GL_POLYGON_SMOOTH);// adding this will make the invalid display on the Thinkpad laptop	
	glEnable(GL_POINT_SMOOTH);
	//	glEnable(GL_LINE_SMOOTH);	// adding this will make the Compaq laptop's running fail
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

	printf("\nn_max=%ld \n",overall_n_max);
	printf("Texture Size: %f (MB)\n",(float)((float)overall_n_max*(float)nRes*(float)nRes*7.0f)/(1024.0f*1024.0f));
	printf("Readback time: %ld (ms)\nSorting time: %ld (ms)\n", 
		readbackTime, sortingTime);

	CUDA_SAFE_CALL( cudaEventDestroy( startClock ) );
	CUDA_SAFE_CALL( cudaEventDestroy( stopClock ) );
}

bool LDNIcudaOperation::BooleanOperation(LDNIcudaSolid* &inputSolid, QuadTrglMesh *meshB, short nOperationType, float boundingBox[])
{
	//float boundingBox[6];	
	LDNIcudaSolid *solidB;

	//-----------------------------------------------------------------------------------
	//	Step 1: converting the mesh surface into a LDNI solid

	
	int res=inputSolid->GetResolution();
	if (nOperationType!=3) {
		LDNIcudaOperation::BRepToLDNISampling( meshB, solidB, boundingBox, res );
	
	}
	else {
		solidB=inputSolid;	inputSolid=0;
		LDNIcudaOperation::BRepToLDNISampling( meshB, inputSolid, boundingBox, res );
		nOperationType=2;
	
	}
	
	//-----------------------------------------------------------------------------------
	//	Step 2: repair and truncate the sampled LDNI solid into the current working space


	//-----------------------------------------------------------------------------------
	//	Step 3: computing the Boolean operation results on LDNIs
	printf("-----------------------------------------------------------------------\n");
	printf("Starting to compute Boolean operation\n");
	printf("-----------------------------------------------------------------------\n");
	_booleanOperation(inputSolid, solidB, nOperationType);
	inputSolid->SetBoundingBox(boundingBox);
	int nres = inputSolid->GetResolution();
	float gWidth=(boundingBox[1]-boundingBox[0])/(float)nres;
	inputSolid->SetSampleWidth(gWidth);

	//-----------------------------------------------------------------------------------
	//	Step 4: free the memory
	delete solidB;
	
	return true;
}

bool LDNIcudaOperation::BooleanOperation(LDNIcudaSolid* &inputSolid, QuadTrglMesh *meshB, short nOperationType)
{
	float boundingBox[6];	LDNIcudaSolid *solidB;

	//-----------------------------------------------------------------------------------
	//	Step 1: converting the mesh surface into a LDNI solid
	if ( _bExpandableWorkingSpace ) {
		meshB->CompBoundingBox(boundingBox);
		_expansionLDNIcudaSolidByNewBoundingBox(inputSolid, boundingBox);
			
	}
	
	int res=inputSolid->GetResolution();
	if (nOperationType!=3) {
		LDNIcudaOperation::BRepToLDNISampling( meshB, solidB, boundingBox, res );
	
	}
	else {
		solidB=inputSolid;	inputSolid=0;
		LDNIcudaOperation::BRepToLDNISampling( meshB, inputSolid, boundingBox, res );
		nOperationType=2;
	
	}
	
	//-----------------------------------------------------------------------------------
	//	Step 2: repair and truncate the sampled LDNI solid into the current working space
	if ( !(_bExpandableWorkingSpace) ) {
		//repair solidB
	
	}

	//-----------------------------------------------------------------------------------
	//	Step 3: computing the Boolean operation results on LDNIs
	printf("-----------------------------------------------------------------------\n");
	printf("Starting to compute Boolean operation\n");
	printf("-----------------------------------------------------------------------\n");
	_booleanOperation(inputSolid, solidB, nOperationType);
	inputSolid->SetBoundingBox(boundingBox);
	int nres = inputSolid->GetResolution();
	float gWidth=(boundingBox[1]-boundingBox[0])/(float)nres;
	inputSolid->SetSampleWidth(gWidth);

	//-----------------------------------------------------------------------------------
	//	Step 4: free the memory
	delete solidB;
	
	return true;
}  

//bool LDNIcudaOperation::BooleanOperation(LDNIcudaSolid* &solidA, LDNIcudaSolid* &solidB, short nOperationType)
//{
//	float boundingBox[6],origin[3];
//	
//	
//
//	//solidA->GetBoundingBox(boundingBox);
//	//_expansionLDNIcudaSolidByNewBoundingBox(solidB, boundingBox);
//
//	//if ( _bExpandableWorkingSpace ) {
//	//	meshB->CompBoundingBox(boundingBox);
//	//	_expansionLDNIcudaSolidByNewBoundingBox(inputSolid, boundingBox);
//
//	//}
//
//	printf("-----------------------------------------------------------------------\n");
//	printf("Starting to compute Boolean operation\n");
//	printf("-----------------------------------------------------------------------\n");
//	_booleanOperation(solidA, solidB, nOperationType);
//	solidA->SetBoundingBox(boundingBox);
//	int nres = solidA->GetResolution();
//	float gWidth=(boundingBox[1]-boundingBox[0])/(float)nres;
//	solidA->SetSampleWidth(gWidth);
//
//	delete solidB;
//	
//	return true;
//}

bool LDNIcudaOperation::BooleanOperation(QuadTrglMesh *meshA, QuadTrglMesh *meshB, int res, short nOperationType, LDNIcudaSolid* &outputSolid, LDNIcudaSolid* &savedSolid)
{
	float boundingBox[6];	LDNIcudaSolid *solidB;		//int stA,numA,stRes,numRes,stB;

	//-----------------------------------------------------------------------------------
	//	Step 1: converting mesh surfaces into LDNIs
	float bndBoxA[6],bndBoxB[6];
	meshA->CompBoundingBox(bndBoxA);	meshB->CompBoundingBox(bndBoxB);
	_compBoundingCube(meshA, meshB, boundingBox, res);	

	if (savedSolid!= NULL)
	{		
		_expansionLDNIcudaSolidByNewBoundingBox(savedSolid, boundingBox);
		res = savedSolid->GetResolution();
	}
	if (nOperationType!=3) {
		BRepToLDNISampling(meshA, outputSolid, boundingBox, res);
		BRepToLDNISampling(meshB, solidB, boundingBox, res);
	}
	else {
		BRepToLDNISampling(meshB, outputSolid, boundingBox, res);
		BRepToLDNISampling(meshA, solidB, boundingBox, res);
		nOperationType=2;
	}

	//-----------------------------------------------------------------------------------
	//	Step 2: boolean operations
	printf("-----------------------------------------------------------------------%d\n");
	printf("Starting to compute Boolean operation\n");
	printf("-----------------------------------------------------------------------%d\n");
	_booleanOperation(outputSolid, solidB, nOperationType);
	/*outputSolid->SetBoundingBox(boundingBox);
	int nres = outputSolid->GetResolution();
	float gWidth=(boundingBox[1]-boundingBox[0])/(float)nres;
	outputSolid->SetSampleWidth(gWidth);*/
	

	delete solidB;

	return true;
}


bool LDNIcudaOperation::BooleanOperation(QuadTrglMesh *meshA, QuadTrglMesh *meshB, int res, short nOperationType, LDNIcudaSolid* &outputSolid)
{
	float boundingBox[6];	LDNIcudaSolid *solidB;		//int stA,numA,stRes,numRes,stB;

	//-----------------------------------------------------------------------------------
	//	Step 1: converting mesh surfaces into LDNIs
	float bndBoxA[6],bndBoxB[6];
	meshA->CompBoundingBox(bndBoxA);	meshB->CompBoundingBox(bndBoxB);
	
	_compBoundingCube(meshA, meshB, boundingBox, res);	
	if (nOperationType!=3) {
		BRepToLDNISampling(meshA, outputSolid, boundingBox, res);
		BRepToLDNISampling(meshB, solidB, boundingBox, res);
	}
	else {
		BRepToLDNISampling(meshB, outputSolid, boundingBox, res);
		BRepToLDNISampling(meshA, solidB, boundingBox, res);
		nOperationType=2;
	}

	//-----------------------------------------------------------------------------------
	//	Step 2: boolean operations
	printf("-----------------------------------------------------------------------\n");
	printf("Starting to compute Boolean operation\n");
	printf("-----------------------------------------------------------------------\n");
	_booleanOperation(outputSolid, solidB, nOperationType);
	//outputSolid->SetBoundingBox(boundingBox);
	printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n");

	delete solidB;

	return true;
}

bool LDNIcudaOperation::_booleanOperation(LDNIcudaSolid* outputSolid, LDNIcudaSolid* solidB, short nOperationType)
{
	int res=outputSolid->GetResolution();
	unsigned int arrsize=res*res;	

	if (outputSolid->GetSampleNumber()==0) {
		if (nOperationType==0) _switchSolid(outputSolid,solidB);	// Union
		if (nOperationType==1) outputSolid->CleanUpSamples();		// Intersection
		// Difference
		if (nOperationType==3) _switchSolid(outputSolid,solidB);	// Inversed Difference
		return true;
	}
	if (solidB->GetSampleNumber()==0) {
		// Union
		if (nOperationType==1) outputSolid->CleanUpSamples();	// Intersection
		// Difference
		if (nOperationType==3) outputSolid->CleanUpSamples();	// Inversed Difference
		return true;
	}

	//-----------------------------------------------------------------------------------
	//	Step 1: Initialization 
	long time=clock();		
	unsigned int *devIndexArrayResPtr;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&devIndexArrayResPtr, (arrsize+1)*sizeof(unsigned int) ) );
	
	//-----------------------------------------------------------------------------------
	//	Step 2: computing the Boolean operation results on LDNIs
	for(short nAxis=0;nAxis<3;nAxis++) {
		//---------------------------------------------------------------------------------------------
		//	Sub-step 1: intialization
		CUDA_SAFE_CALL( cudaMemset( (void*)devIndexArrayResPtr, 0, (arrsize+1)*sizeof(unsigned int) ) );
		//---------------------------------------------------------------------------------------------
		float *devNxArrayPtrA=outputSolid->GetSampleNxArrayPtr(nAxis);	
		float *devNyArrayPtrA=outputSolid->GetSampleNyArrayPtr(nAxis);
		float *devDepthArrayPtrA=outputSolid->GetSampleDepthArrayPtr(nAxis);	//if (devDepthArrayPtrA==NULL) printf("Empty ");
		unsigned int *devIndexArrayPtrA=outputSolid->GetIndexArrayPtr(nAxis);
		float *devNxArrayPtrB=solidB->GetSampleNxArrayPtr(nAxis);
		float *devNyArrayPtrB=solidB->GetSampleNyArrayPtr(nAxis);
		float *devDepthArrayPtrB=solidB->GetSampleDepthArrayPtr(nAxis);
		unsigned int *devIndexArrayPtrB=solidB->GetIndexArrayPtr(nAxis);

		//---------------------------------------------------------------------------------------------
		//	Sub-step 2: computing the result of boolean operation ray by ray
		krLDNIBoolean_BooleanOnRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(devNxArrayPtrA, devNyArrayPtrA, devDepthArrayPtrA, devIndexArrayPtrA,
					devNxArrayPtrB, devNyArrayPtrB, devDepthArrayPtrB, devIndexArrayPtrB, devIndexArrayResPtr, arrsize, nOperationType);

		//---------------------------------------------------------------------------------------------
		//	Sub-step 3: compaction of index array
		thrust::device_ptr<unsigned int> dev_ptr(devIndexArrayResPtr); //	Wrap raw pointers with dev_ptr
		thrust::exclusive_scan(dev_ptr, dev_ptr+(arrsize+1), dev_ptr); //	in-place scan
		unsigned int sampleNum=dev_ptr[arrsize];

		//---------------------------------------------------------------------------------------------
		//	Sub-step 4: collecting the resultant samples into the sampleArray of solidTileA				
		float *newDevNxArrayPtrA, *newDevNyArrayPtrA, *newDevDepthArrayPtrA;
		outputSolid->MallocSampleMemory(nAxis, sampleNum);
		newDevNxArrayPtrA=outputSolid->GetSampleNxArrayPtr(nAxis);
		newDevNyArrayPtrA=outputSolid->GetSampleNyArrayPtr(nAxis);
		newDevDepthArrayPtrA=outputSolid->GetSampleDepthArrayPtr(nAxis);
		krLDNIBoolean_ResultSampleCollection<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
					devNxArrayPtrA, devNyArrayPtrA, devDepthArrayPtrA, devIndexArrayPtrA,
					devNxArrayPtrB, devNyArrayPtrB, devDepthArrayPtrB, devIndexArrayPtrB, 
					newDevNxArrayPtrA, newDevNyArrayPtrA, newDevDepthArrayPtrA, devIndexArrayResPtr, arrsize);
		CUDA_SAFE_CALL( cudaMemcpy( devIndexArrayPtrA, devIndexArrayResPtr, (arrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToDevice ) );

		cudaFree(devNxArrayPtrA);	cudaFree(devNyArrayPtrA);	cudaFree(devDepthArrayPtrA);
	}

	//-----------------------------------------------------------------------------------
	//	Step 3: free the memory
	cudaFree(devIndexArrayResPtr);
	printf("Boolean Operation Time (ms): %ld\n",clock()-time);

	return true;
}

void LDNIcudaOperation::SolidRegularization(LDNIcudaSolid *solid)  // Removing samples that are nearly tangentially contacted
{
	int res=solid->GetResolution();
	unsigned int arrsize=res*res;	

	//-----------------------------------------------------------------------------------
	//	Step 1: Initialization 
	long time=clock();		
	unsigned int *devIndexArrayPtrRes;
	CUDA_SAFE_CALL( cudaMalloc( (void**)&devIndexArrayPtrRes, (arrsize+1)*sizeof(unsigned int) ) );
	float ww=solid->GetSampleWidth();	

	//-----------------------------------------------------------------------------------
	//	Step 2: Remove the tangentially contacted samples
	for(short nAxis=0;nAxis<3;nAxis++) {
		//---------------------------------------------------------------------------------------------
		//	Sub-step 1: intialization
		CUDA_SAFE_CALL( cudaMemset( (void*)devIndexArrayPtrRes, 0, (arrsize+1)*sizeof(unsigned int) ) );
		//---------------------------------------------------------------------------------------------
		float *devNxArrayPtr=solid->GetSampleNxArrayPtr(nAxis);	
		float *devNyArrayPtr=solid->GetSampleNyArrayPtr(nAxis);
		float *devDepthArrayPtr=solid->GetSampleDepthArrayPtr(nAxis);
		unsigned int *devIndexArrayPtr=solid->GetIndexArrayPtr(nAxis);

		//---------------------------------------------------------------------------------------------
		//	Sub-step 2: computing the result of regularization ray by ray
		krLDNIRegularization_RegularizationOnRays<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
					devNxArrayPtr, devNyArrayPtr, devDepthArrayPtr, devIndexArrayPtr, devIndexArrayPtrRes, arrsize, 0.01*ww);
		
		//---------------------------------------------------------------------------------------------
		//	Sub-step 3: compaction of index array
		thrust::device_ptr<unsigned int> dev_ptr(devIndexArrayPtrRes); //	Wrap raw pointers with dev_ptr
		thrust::exclusive_scan(dev_ptr, dev_ptr+(arrsize+1), dev_ptr); //	in-place scan
		unsigned int sampleNum=dev_ptr[arrsize];

		//---------------------------------------------------------------------------------------------
		//	Sub-step 4: collecting the resultant samples into the sampleArray of solidTileA				
		float *devNxArrayPtrRes, *devNyArrayPtrRes, *devDepthArrayPtrRes;
		CUDA_SAFE_CALL( cudaMalloc( (void**)&devNxArrayPtrRes, sampleNum*sizeof(float) ) );	
		CUDA_SAFE_CALL( cudaMalloc( (void**)&devNyArrayPtrRes, sampleNum*sizeof(float) ) );	
		CUDA_SAFE_CALL( cudaMalloc( (void**)&devDepthArrayPtrRes, sampleNum*sizeof(float) ) );	
		krLDNIRegularization_ResultSampleCollection<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
					devNxArrayPtr, devNyArrayPtr, devDepthArrayPtr, devIndexArrayPtr, 
					devNxArrayPtrRes, devNyArrayPtrRes, devDepthArrayPtrRes, devIndexArrayPtrRes, arrsize);
		solid->SetSampleDepthArrayPtr(nAxis,devDepthArrayPtrRes);
		solid->SetSampleNxArrayPtr(nAxis,devNxArrayPtrRes);
		solid->SetSampleNyArrayPtr(nAxis,devNyArrayPtrRes);
		solid->SetIndexArrayPtr(nAxis,devIndexArrayPtrRes);		devIndexArrayPtrRes=devIndexArrayPtr;
		solid->SetSampleNumber(nAxis,sampleNum);

		cudaFree(devNxArrayPtr);	cudaFree(devNyArrayPtr);	cudaFree(devDepthArrayPtr);
	}

	//-----------------------------------------------------------------------------------
	//	Step 3: Free the memory
	cudaFree(devIndexArrayPtrRes);
	printf("Solid Regularization Time (ms): %ld\n",clock()-time);
}

void LDNIcudaOperation::_compBoundingCube(QuadTrglMesh *meshA, QuadTrglMesh *meshB, float boundingBox[], int res)
{
	float bndBoxA[6],bndBoxB[6];
	
	meshA->CompBoundingBox(bndBoxA);	meshB->CompBoundingBox(bndBoxB);
	
	boundingBox[0]=MIN(bndBoxA[0],bndBoxB[0]);
	boundingBox[1]=MAX(bndBoxA[1],bndBoxB[1]);
	boundingBox[2]=MIN(bndBoxA[2],bndBoxB[2]);
	boundingBox[3]=MAX(bndBoxA[3],bndBoxB[3]);
	boundingBox[4]=MIN(bndBoxA[4],bndBoxB[4]);
	boundingBox[5]=MAX(bndBoxA[5],bndBoxB[5]);
	//------------------------------------------------------------------------
	//	making the working space cubic
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

bool LDNIcudaOperation::BRepToLDNISampling(QuadTrglMesh *mesh, LDNIcudaSolid* &solid, float boundingBox[], int res)
{
	const bool bCube=true;
	float origin[3],gWidth;		long time=clock(),totalTime=clock();
	int i,nodeNum;	
	char fileadd[256];
	
	//----------------------------------------------------------------------------------------
	//	Preparation
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
	
	//---------------------------------------------------------------------------------
	solid=new LDNIcudaSolid;
	solid->MallocMemory(res);
	gWidth=(boundingBox[1]-boundingBox[0])/(float)res;
	solid->SetSampleWidth(gWidth);
	origin[0]=boundingBox[0]+gWidth*0.5f;
	origin[1]=boundingBox[2]+gWidth*0.5f;
	origin[2]=boundingBox[4]+gWidth*0.5f;
	solid->SetOrigin(origin[0],origin[1],origin[2]);
	
	//---------------------------------------------------------------------------------
	//	For using OpenGL Shading Language to implement the sampling procedure
	if (glewInit() != GLEW_OK) {printf("glewInit failed. Exiting...\n");	return false;}
	if (glewIsSupported("GL_VERSION_2_0")) {printf("\nReady for OpenGL 2.0\n");} else {printf("OpenGL 2.0 not supported\n"); return false;}
	//-----------------------------------------------------------------------------------------
	int dispListIndex;		GLhandleARB g_programObj, g_vertexShader, g_GeometryShader, g_FragShader;
	GLenum InPrimType=GL_POINTS, OutPrimType=GL_TRIANGLES;		int OutVertexNum=3;
	GLuint vertexTexture;	
	const char *VshaderString[1],*GshaderString[1],*FshaderString[1];
	GLint bCompiled = 0, bLinked = 0;
	char str[4096] = "";		int xF,yF;
	//-----------------------------------------------------------------------------------------
	//	Step 1: Setup the shaders 
	memset(fileadd,0,256*sizeof(char));	
	strcat(fileadd,"sampleLDNIVertexShader.vert");
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
	strcat(fileadd,"sampleLDNIGeometryShader.geo");
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
	strcat(fileadd,"sampleLDNIFragmentShader.frag");
	g_FragShader = glCreateShaderObjectARB( GL_FRAGMENT_SHADER_ARB );
	ShaderAssembly = _readShaderFile( fileadd );
	FshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_FragShader, 1, FshaderString, NULL );
	glCompileShaderARB( g_FragShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_FragShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {
		glGetInfoLogARB(g_FragShader, sizeof(str), NULL, str);
		printf("Warning: Vertex Shader Compile Error\n\n");	return false;
	}
	//-----------------------------------------------------------------------------
	g_programObj = glCreateProgramObjectARB();
	if (glGetError()!=GL_NO_ERROR) printf("Error: OpenGL!\n\n");
	glAttachObjectARB( g_programObj, g_vertexShader );		if (glGetError()!=GL_NO_ERROR) printf("Error: attach Vertex Shader!\n\n");
	glAttachObjectARB( g_programObj, g_GeometryShader );	if (glGetError()!=GL_NO_ERROR) printf("Error: attach Geometry Shader!\n\n");
	glAttachObjectARB( g_programObj, g_FragShader );		if (glGetError()!=GL_NO_ERROR) printf("Error: attach Fragment Shader!\n\n");
	//-----------------------------------------------------------------------------
	//	Configuration setting for geometry shader
	glProgramParameteriEXT(g_programObj, GL_GEOMETRY_INPUT_TYPE_EXT, InPrimType);
	glProgramParameteriEXT(g_programObj, GL_GEOMETRY_OUTPUT_TYPE_EXT, OutPrimType);
	glProgramParameteriEXT(g_programObj, GL_GEOMETRY_VERTICES_OUT_EXT, OutVertexNum); 
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
	GLint id0,id1;	float centerPos[3];
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
	if (glGetError()!=GL_NO_ERROR) printf("Error: vertex texture binding!\n\n");
	printf("Create shader texture\n");
	//-----------------------------------------------------------------------------------------
	//	Step 5:  sampling
	printf("GLList ID: %d\n",dispListIndex);
	time=clock()-time;	printf("GL-List building time (including uploading texture) is %ld (ms)\n",time);
	_decomposeLDNIByFBOPBO(solid,dispListIndex);
	
	//-----------------------------------------------------------------------------------------
	//	Step 6:  free the memory
	time=clock();
	//-----------------------------------------------------------------------------------------
	glDeleteLists(dispListIndex, 1);
	glBindTexture( GL_TEXTURE_RECTANGLE_ARB, 0);
	glDisable(GL_TEXTURE_RECTANGLE_ARB);
	glDeleteTextures(1, &vertexTexture);
	glUseProgramObjectARB(0);
	glDeleteObjectARB( g_vertexShader);
	glDeleteObjectARB( g_GeometryShader);
	glDeleteObjectARB( g_FragShader);
	glDeleteObjectARB( g_programObj);
	//------------------------------------------------------------------------
	printf("\nMemory clean-up time is %ld (ms)\n",clock()-time);
	printf("--------------------------------------------------------------\n");
	printf("Total time for sampling is %ld (ms)\n\n",clock()-totalTime);
	
	return true;
}

void LDNIcudaOperation::_decomposeLDNIByFBOPBO(LDNIcudaSolid *solid, int displayListIndex)
{
	unsigned int n_max,i,n;
	float gWidth,origin[3];
	unsigned int overall_n_max=0;
	long readbackTime=0, sortingTime=0, tempTime;

	cudaEvent_t     startClock, stopClock;
	CUDA_SAFE_CALL( cudaEventCreate( &startClock ) );
	CUDA_SAFE_CALL( cudaEventCreate( &stopClock ) );

	tempTime=clock();
	//------------------------------------------------------------------------
	//	Preparation
	int nRes=solid->GetResolution();		gWidth=solid->GetSampleWidth();
	float width=gWidth*(float)nRes;
	solid->GetOrigin(origin[0],origin[1],origin[2]);
	int arrsize=nRes*nRes;

	//------------------------------------------------------------------------
	//	Step 1: Setup the rendering environment
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_STENCIL_TEST);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_POLYGON_OFFSET_FILL);
	glDisable(GL_POLYGON_OFFSET_LINE);
    glDisable(GL_BLEND);	
	glDisable(GL_POLYGON_SMOOTH);	// turn off anti-aliasing
	glDisable(GL_POINT_SMOOTH);
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_MAP_COLOR);	glDisable(GL_DITHER);
	glShadeModel(GL_FLAT);
	glDisable(GL_LIGHTING);   glDisable(GL_LIGHT0);
	glDisable(GL_LOGIC_OP);
	glDisable(GL_COLOR_MATERIAL);
	glDisable(GL_ALPHA_TEST);
	glGetError();	// for clean-up the error generated before
	//------------------------------------------------------------------------
	//	create the FBO objects and texture for rendering
	if (glewIsSupported("GL_EXT_framebuffer_object") == 0) printf("Warning: FBO is not supported!\n");
	if (glGetError()!=GL_NO_ERROR) printf("Error: before framebuffer generation!\n");
	//------------------------------------------------------------------------
	GLuint fbo;
	glGenFramebuffersEXT(1, &fbo);
	if (glGetError()!=GL_NO_ERROR) printf("Error: framebuffer generation!\n");
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
	if (glGetError()!=GL_NO_ERROR) printf("Error: framebuffer binding!\n");
	//------------------------------------------------------------------------
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, nRes, nRes, 0, GL_RGBA, GL_FLOAT, 0);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, tex, 0);
	if (glGetError()!=GL_NO_ERROR) printf("Error: attaching texture to framebuffer generation!\n");
	cudaGraphicsResource *sampleTex_resource;
	CUDA_SAFE_CALL( cudaGraphicsGLRegisterImage(&sampleTex_resource, tex, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly) );
	//------------------------------------------------------------------------
	GLuint depth_and_stencil_rb;
	glGenRenderbuffersEXT(1, &depth_and_stencil_rb);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depth_and_stencil_rb);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_STENCIL_EXT, nRes, nRes);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depth_and_stencil_rb);
	if (glGetError()!=GL_NO_ERROR) printf("Error: attaching renderbuffer of depth-buffer to framebuffer generation!\n");
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depth_and_stencil_rb);
	if (glGetError()!=GL_NO_ERROR) printf("Error: attaching renderbuffer of stencil-buffer to framebuffer generation!\n");
	//------------------------------------------------------------------------
	GLuint indexPBO;
	glGenBuffers(1,&indexPBO);	//	generation of PBO for index array readback
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, indexPBO);
	glBufferData(GL_PIXEL_PACK_BUFFER_ARB, nRes*nRes*sizeof(unsigned char), NULL, GL_STREAM_READ_ARB);
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	CUDA_SAFE_CALL( cudaGLRegisterBufferObject(indexPBO) );
	//------------------------------------------------------------------------
	if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT)!=GL_FRAMEBUFFER_COMPLETE_EXT) 
		printf("Warning: the setting for rendering on FBO is not correct!\n");
	else
		printf("FBO has been created successfully!\n");
	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0,0,nRes,nRes);
	printf("Preparation time: %ld (ms)\n",clock()-tempTime);

	//------------------------------------------------------------------------
	//	Step 2: Rendering to get the Hermite samples
	for(short nAxis=0; nAxis<3; nAxis++) { 
		//---------------------------------------------------------------------------------------
		//	Rendering step 1: setting the viewing window
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		//---------------------------------------------------------------------------------------
		//	The eye is located at (0, 0, 0), the near clipping plane is at the z=0 plane
		//		the far clipping plane is at the z=(boundingBox[5]-boundingBox[4]) plane
		glOrtho(-width*0.5f,width*0.5f,-width*0.5f,width*0.5f,width*0.5f,-width*0.5f);
		//	Note that:	in "glOrtho(left,right,bottom,top,near,far);"
		//		(left,right,bottom,top) are located at the boundary of pixel instead of
		//		the center of pixels
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		//---------------------------------------------------------------------------------------
		//	Rendering step 2: determine the number of layers
		glClearColor( 1.0f, 1.0f, 1.0f, 1.0f );	
		glClearDepth(1.0);
		glClearStencil(0);	glColor3f(1,1,1);
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		glDepthFunc(GL_ALWAYS);
		glStencilFunc(GL_GREATER, 1, 0xff);
		glStencilOp(GL_INCR, GL_INCR, GL_INCR);
		glPushMatrix();
		switch(nAxis) {
			case 0:{glRotatef(-90,0,1,0);	glRotatef(-90,1,0,0); }break;
			case 1:{glRotatef(90,0,1,0);	glRotatef(90,0,0,1);  }break;
		}
		glCallList(displayListIndex);	glFlush();
		//--------------------------------------------------------------------------------------------------------
		//	reading stencil buffer into the device memory of CUDA
		tempTime=clock();
		glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, indexPBO);
		GLint OldPackAlignment;
		glGetIntegerv(GL_PACK_ALIGNMENT,&OldPackAlignment); 
		glPixelStorei(GL_PACK_ALIGNMENT,1);	// Important!!! Without this, the read-back could be abnormal.
		glReadPixels(0,0,nRes,nRes,GL_STENCIL_INDEX,GL_UNSIGNED_BYTE,0);
		glPixelStorei(GL_PACK_ALIGNMENT,OldPackAlignment);
		//--------------------------------------------------------------------------------------------------------
		unsigned char *devStencilBufferPtr;
		unsigned int *devResArrayPtr;
		unsigned int *devIndexArrayPtr=solid->GetIndexArrayPtr(nAxis);
		CUDA_SAFE_CALL( cudaGLMapBufferObject( (void **)&devStencilBufferPtr, indexPBO) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&devResArrayPtr, BLOCKS_PER_GRID*sizeof(unsigned int) ) );
		//--------------------------------------------------------------------------------------------------------
		//	building the indexArray on device
		krLDNISampling_CopyIndexAndFindMax<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(devStencilBufferPtr,
																		devIndexArrayPtr,devResArrayPtr,arrsize);
		//--------------------------------------------------------------------------------------------------------
		//	read back the max number of layers -- "n_max"
		unsigned int* resArrayPtr;
		resArrayPtr=(unsigned int *)malloc(BLOCKS_PER_GRID*sizeof(unsigned int));
		CUDA_SAFE_CALL( cudaMemcpy( resArrayPtr, devResArrayPtr, BLOCKS_PER_GRID*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
		n_max=0;
		for(i=0;i<BLOCKS_PER_GRID;i++) n_max = MAX(n_max,resArrayPtr[i]);
		cudaFree(devResArrayPtr);		free(resArrayPtr);
		//--------------------------------------------------------------------------------------------------------
		//	read back the number of samples -- "sampleNum"
		unsigned int sampleNum=0;
		tempTime=clock()-tempTime;		//readbackTime+=tempTime;
		printf("Stencil buffer processing time: %ld (ms)\n",tempTime);

		long scanTime=clock();
		//	for debug purpose
		resArrayPtr=(unsigned int *)malloc((arrsize+1)*sizeof(unsigned int));
		CUDA_SAFE_CALL( cudaMemcpy( resArrayPtr, devIndexArrayPtr, (arrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
		sampleNum=0;
		for(int k=0;k<arrsize;k++) {sampleNum+=resArrayPtr[k];	resArrayPtr[k]=sampleNum;}	
		for(int k=arrsize;k>0;k--) {resArrayPtr[k]=resArrayPtr[k-1];}	
		resArrayPtr[0]=0;
		CUDA_SAFE_CALL( cudaMemcpy( devIndexArrayPtr, resArrayPtr, (arrsize+1)*sizeof(unsigned int), cudaMemcpyHostToDevice ) );
		free(resArrayPtr);
		scanTime=clock()-scanTime;	printf("Scanning time: %ld (ms)\n",scanTime);

		//--------------------------------------------------------------------------------------------------------
		CUDA_SAFE_CALL( cudaGLUnmapBufferObject( indexPBO ) );
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
		printf("n_max=%d   sampleNum=%d\n",n_max,sampleNum);
		if (n_max>overall_n_max) overall_n_max=n_max;
		if (sampleNum==0) continue;

		//---------------------------------------------------------------------------------------
		//	Rendering step 3: decomposing the Layered Depth Images (LDIs) and record its corresponding normals
		solid->MallocSampleMemory(nAxis,sampleNum);	
		float* devNxArrayPtr=solid->GetSampleNxArrayPtr(nAxis);
		float* devNyArrayPtr=solid->GetSampleNyArrayPtr(nAxis);
		float* devDepthArrayPtr=solid->GetSampleDepthArrayPtr(nAxis);
		tempTime=clock();
		for(n=1;n<=n_max;n++) {
			CUDA_SAFE_CALL( cudaGraphicsMapResources( 1, &sampleTex_resource, NULL ) );
			cudaArray *in_array;
			CUDA_SAFE_CALL( cudaGraphicsSubResourceGetMappedArray( &in_array, sampleTex_resource, 0, 0));
			CUDA_SAFE_CALL( cudaBindTextureToArray(tex2DFloat4In, in_array) );
			//--------------------------------------------------------------------------------------------------------
			//	fill the sampleArray on device
			krLDNISampling_CopySamples<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(devNxArrayPtr, devNyArrayPtr, 
																	devDepthArrayPtr, n, arrsize, width, gWidth, nRes, devIndexArrayPtr);
			CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 1, &sampleTex_resource, NULL ) );
			if (n==n_max) break;

			glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
			glStencilFunc(GL_GREATER, n+1, 0xff);
			glStencilOp(GL_KEEP, GL_INCR, GL_INCR);
			glCallList(displayListIndex);	glFlush();
		}
		tempTime=clock()-tempTime;		readbackTime+=tempTime;

		//------------------------------------------------------------------------
		//	Rendering step 4: sorting the samples
		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );
		CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
		krLDNISampling_SortSamples<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(devNxArrayPtr, devNyArrayPtr, 
																	devDepthArrayPtr, arrsize, devIndexArrayPtr);
		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );
		CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );
		float   elapsedTime;
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,
											startClock, stopClock ) );
//		printf( "Sorting time is:  %3.1f (ms)\n", elapsedTime );		
		sortingTime+=(long)elapsedTime;
	}

	//------------------------------------------------------------------------------------
	//	Step 3: Set the rendering parameters back
	//------------------------------------------------------------------------------------
	//	detach FBO
	glPopAttrib();
	//	release memory for PBO and cuda's map	
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	CUDA_SAFE_CALL( cudaGLUnregisterBufferObject( indexPBO ) );
	glDeleteBuffers(1, &indexPBO);
	CUDA_SAFE_CALL( cudaGraphicsUnregisterResource( sampleTex_resource) );
	//	release memory for the 2D texture
	glBindTexture(GL_TEXTURE_2D, 0);
	glDeleteTextures(1, &tex);
	//	release memory for the frame-buffer object
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glDeleteFramebuffersEXT(1, &fbo);
	//	release memory for the render-buffer object
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
	glDeleteRenderbuffersEXT(1, &depth_and_stencil_rb);
	//------------------------------------------------------------------------------------
	glEnable(GL_POLYGON_OFFSET_FILL);
	glEnable(GL_POLYGON_OFFSET_LINE);
    glEnable(GL_BLEND);	
	glEnable(GL_DITHER);
	glDisable(GL_STENCIL_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_MAP_COLOR);				
	glShadeModel(GL_SMOOTH);   
	glEnable(GL_LIGHTING);  glEnable(GL_LIGHT0);
//	glEnable(GL_POLYGON_SMOOTH);// adding this will make the invalid display on the Thinkpad laptop	
	glEnable(GL_POINT_SMOOTH);
//	glEnable(GL_LINE_SMOOTH);	// adding this will make the Compaq laptop's running fail
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

	printf("\nn_max=%ld \n",overall_n_max);
	printf("Texture Size: %f (MB)\n",(float)((float)overall_n_max*(float)nRes*(float)nRes*7.0f)/(1024.0f*1024.0f));
	printf("Readback time: %ld (ms)\nSorting time: %ld (ms)\n", 
		readbackTime, sortingTime);

	CUDA_SAFE_CALL( cudaEventDestroy( startClock ) );
	CUDA_SAFE_CALL( cudaEventDestroy( stopClock ) );
}

unsigned char* LDNIcudaOperation::_readShaderFile( const char *fileName )
{
    FILE *file = fopen( fileName, "r" );

    if ( file == NULL ) {
       	printf("Cannot open shader file!");
		return 0;
    }
	struct _stat fileStats;
    if ( _stat( fileName, &fileStats ) != 0 ) {
        printf("Cannot get file stats for shader file!");
        return 0;
    }
    unsigned char *buffer = new unsigned char[fileStats.st_size];
	int bytes = (int)(fread( buffer,1, fileStats.st_size, file ));
    buffer[bytes] = 0;

	fclose( file );

	return buffer;
}

void LDNIcudaOperation::_texCalProduct(int in, int &outx, int &outy)
{
	int left=0,right=0,div3left=0,div3right=0;

	left = int(floor(sqrt((float)in)))-1;
	right = int(ceil(sqrt((float)in)));
	while(left*right < in) {right++;}
	if (left%3 == 0 && left*right>=in) {
		div3left = left;
		div3right = right;
	}
	else if (right%3 == 0 && left*right>=in) {
		div3left = right;
		div3right = left;
	}
	right++;	left--;
	if (left%3 == 0 && left*right>=in) {
		div3left = left;
		div3right = right;
	}
	else if (right%3 == 0 && left*right>=in){
		div3left = right;
		div3right = left;
	}
	while(left*right > in){
		right++;	left--;
		if (left%3 == 0 && left*right>in){
			div3left = left;
			div3right = right;
		}
		else if (right%3 == 0 && left*right>in){
			div3left = right;
			div3right = left;
		}
	}
	if (right*left < in){
		right--;	left++;
		if (left%3 == 0 ){
			div3left = left;
			div3right = right;
		}
		else if (right%3 == 0){
			div3left = right;
			div3right = left;
		} 
	}
	outx=div3left;	outy=div3right;

	if (outx==0 || outy==0) {outx=in; outy=1;}
}

//--------------------------------------------------------------------------------------------
void LDNIcudaOperation::OrientedNormalReconstruction(LDNIcudaSolid *solid, unsigned int nSupportSize, bool bWithOrientationVoting)
{
	unsigned int *indexArray[3];		float *depthArray[3],*nxArray[3],*nyArray[3];
	int res;	short nAxis;
	float ww,origin[3];
	float *buffer;	int sampleNum,xNum,yNum,zNum;

	//---------------------------------------------------------------------------------------------------------
	//	preparation
	res=solid->GetResolution();		ww=solid->GetSampleWidth();
	solid->GetOrigin(origin[0],origin[1],origin[2]);
	for(nAxis=0;nAxis<3;nAxis++) {
		nxArray[nAxis]=solid->GetSampleNxArrayPtr(nAxis);
		nyArray[nAxis]=solid->GetSampleNyArrayPtr(nAxis);
		depthArray[nAxis]=solid->GetSampleDepthArrayPtr(nAxis);
		indexArray[nAxis]=solid->GetIndexArrayPtr(nAxis);	
	}
	xNum=solid->GetSampleNumber(0);	yNum=solid->GetSampleNumber(1);	zNum=solid->GetSampleNumber(2);
	sampleNum=MAX3(xNum,yNum,zNum);
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(buffer), sampleNum*3*sizeof(float) ) );	

	//--------------------------------------------------------------------------------------------------------------------------
	//	Phase 1: estimation of the oriented normal vectors
	for(nAxis=0;nAxis<3;nAxis++) {
		//----------------------------------------------------------------------------------------------------------------------
		//	Preprocessing
		krLDNINormalProcessing_PreProc<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(indexArray[nAxis],buffer,res,res*res);

		//----------------------------------------------------------------------------------------------------------------------
		//	The following kernel is sample-based normal reconstruction
		krLDNINormalReconstruction_PerSample<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(indexArray[0], indexArray[1], indexArray[2], 
								nxArray[0], nxArray[1], nxArray[2], nyArray[0], nyArray[1], nyArray[2], 
								depthArray[0], depthArray[1], depthArray[2], buffer, 
								solid->GetSampleNumber(nAxis), nAxis, res, ww, nSupportSize);

		//----------------------------------------------------------------------------------------------------------------------
		//	Updating the result of computation
		int sNum=solid->GetSampleNumber(nAxis);
		krLDNINormalProcessing_Update<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
							sNum, nxArray[nAxis], nyArray[nAxis], depthArray[nAxis], buffer);
	}

	//--------------------------------------------------------------------------------------------------------------------------
	//	Phase 2: voting based correction of normal vectors' orientation
	if (bWithOrientationVoting)
	for(nAxis=0;nAxis<3;nAxis++) {
		//----------------------------------------------------------------------------------------------------------------------
		//	Preprocessing
		krLDNINormalProcessing_PreProc<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(indexArray[nAxis],buffer,res,res*res);

		//----------------------------------------------------------------------------------------------------------------------
		//	The following kernel is voting-based orientation correction for normal vectors
		krLDNINormalProcessing_OrientationCorrectionByVoting<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(indexArray[0], indexArray[1], indexArray[2], 
								nxArray[0], nxArray[1], nxArray[2], nyArray[0], nyArray[1], nyArray[2], 
								depthArray[0], depthArray[1], depthArray[2], buffer, 
								solid->GetSampleNumber(nAxis), nAxis, res, ww, nSupportSize);
	}

	//-----------------------------------------------------------------------------------------
	//	release the memory
	cudaFree(buffer);
}

void LDNIcudaOperation::ParallelProcessingNormalVector(LDNIcudaSolid *solid, unsigned int nSupportSize, float normalPara)
{
//	cudaEvent_t     startClock, stopClock;		
//	float elapsedTime;
//	CUDA_SAFE_CALL( cudaEventCreate( &startClock ) );
//	CUDA_SAFE_CALL( cudaEventCreate( &stopClock ) );

	unsigned int *indexArray[3];		float *depthArray[3],*nxArray[3],*nyArray[3];
	int res;	short nAxis;
	float ww,origin[3];
	float *buffer;	int sampleNum,xNum,yNum,zNum;

	//---------------------------------------------------------------------------------------------------------
	//	preparation
	res=solid->GetResolution();		ww=solid->GetSampleWidth();
	solid->GetOrigin(origin[0],origin[1],origin[2]);
	for(nAxis=0;nAxis<3;nAxis++) {
		nxArray[nAxis]=solid->GetSampleNxArrayPtr(nAxis);
		nyArray[nAxis]=solid->GetSampleNyArrayPtr(nAxis);
		depthArray[nAxis]=solid->GetSampleDepthArrayPtr(nAxis);
		indexArray[nAxis]=solid->GetIndexArrayPtr(nAxis);	
	}
	xNum=solid->GetSampleNumber(0);	yNum=solid->GetSampleNumber(1);	zNum=solid->GetSampleNumber(2);
	sampleNum=MAX3(xNum,yNum,zNum);
	CUDA_SAFE_CALL( cudaMalloc( (void**)&(buffer), sampleNum*3*sizeof(float) ) );	

	for(nAxis=0;nAxis<3;nAxis++) 
	{	//nAxis=0;
//		CUDA_SAFE_CALL( cudaMemset( (void*)buffer, 0, sampleNum*3*sizeof(float) ) );
//		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );
//		CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
		krLDNINormalProcessing_PreProc<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(indexArray[nAxis],buffer,res,res*res);
//		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );
//		CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
//		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
//		printf("%d-direction pre-processing time: %3.1f (ms)\n",(int)nAxis,elapsedTime);

//		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );
//		CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
		//----------------------------------------------------------------------------------------------------------------------
		//	The following kernel is ray-based filtering, which is too slow to process
/*		krLDNIBilateralNormalFilter_PerRay<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
								indexArray[0], indexArray[1], indexArray[2], 
								nxArray[0], nxArray[1], nxArray[2], 
								nyArray[0], nyArray[1], nyArray[2], 
								depthArray[0], depthArray[1], depthArray[2], buffer, 
								res*res, nAxis, res, ww, origin[0], origin[1], origin[2], nSupportSize, normalPara);*/
		//----------------------------------------------------------------------------------------------------------------------
		//	The following kernel is sample-based filtering
		krLDNIBilateralNormalFilter_PerSample<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
								indexArray[0], indexArray[1], indexArray[2], 
								nxArray[0], nxArray[1], nxArray[2], 
								nyArray[0], nyArray[1], nyArray[2], 
								depthArray[0], depthArray[1], depthArray[2], buffer, 
								solid->GetSampleNumber(nAxis), nAxis, res, ww, nSupportSize, normalPara);

//		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );
//		CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );	// This confirms the kernel's running has completed
//		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
//		printf("%d-direction processing time: %3.1f (ms)\n",(int)nAxis,elapsedTime);

		int sNum=solid->GetSampleNumber(nAxis);
//		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );
//		CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
		krLDNINormalProcessing_Update<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
							sNum, nxArray[nAxis], nyArray[nAxis], depthArray[nAxis], buffer);
//		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );
//		CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );
//		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,	startClock, stopClock ) );
//		printf("Buffer updating time: %3.1f (ms)\n",elapsedTime);
	}

	//-----------------------------------------------------------------------------------------
	//	release the memory
	cudaFree(buffer);
//	CUDA_SAFE_CALL( cudaEventDestroy( startClock ) );
//	CUDA_SAFE_CALL( cudaEventDestroy( stopClock ) );
}

//--------------------------------------------------------------------------------------------
void LDNIcudaOperation::CopyCPUSolidToCUDASolid(LDNIcpuSolid *cpuSolid, LDNIcudaSolid* &cudaSolid)
{
	float ox,oy,oz,gWidth;	int i,num,res;	short nAxis;
	LDNIcpuRay *rays;	LDNIcpuSample *sampleArray;

	cpuSolid->GetOrigin(ox,oy,oz);		
	gWidth=cpuSolid->GetSampleWidth();
	res=cpuSolid->GetResolution();

	cudaSolid=new LDNIcudaSolid;
	cudaSolid->SetOrigin(ox,oy,oz);		cudaSolid->SetSampleWidth(gWidth);
	cudaSolid->MallocMemory(res);

	//-----------------------------------------------------------------------------------------
	//	copy the index arrays
	unsigned int *dev_indexArray,*indexArray;
	num=res*res;
	indexArray=(unsigned int *)malloc((num+1)*sizeof(unsigned int));
	for(nAxis=0;nAxis<3;nAxis++) {
		rays=cpuSolid->GetRayArrayPtr(nAxis);
		indexArray[0]=0;
		for(i=0;i<num;i++) indexArray[i+1]=rays[i].sampleIndex;

		dev_indexArray=cudaSolid->GetIndexArrayPtr(nAxis);

		CUDA_SAFE_CALL( cudaMemcpy( dev_indexArray, indexArray, (num+1)*sizeof(unsigned int), cudaMemcpyHostToDevice ) );
	}
	free(indexArray);

	//-----------------------------------------------------------------------------------------
	//	copy the sample arrays
	for(nAxis=0;nAxis<3;nAxis++) {
		rays=cpuSolid->GetRayArrayPtr(nAxis);
		int sampleNum=rays[res*res-1].sampleIndex;

		float *sampleNxArray,*sampleNyArray,*sampleDepthArray;
		sampleNxArray=(float*)malloc(sampleNum*sizeof(float));
		sampleNyArray=(float*)malloc(sampleNum*sizeof(float));
		sampleDepthArray=(float*)malloc(sampleNum*sizeof(float));

		sampleArray=cpuSolid->GetSampleArrayPtr(nAxis);
		for(i=0;i<sampleNum;i++) {
			sampleNxArray[i]=sampleArray[i].nx;
			sampleNyArray[i]=sampleArray[i].ny;
			if (sampleArray[i].nz<0)
				sampleDepthArray[i]=-sampleArray[i].depth;
			else
				sampleDepthArray[i]=sampleArray[i].depth;
		}

		cudaSolid->MallocSampleMemory(nAxis,sampleNum);
		float *dev_sampleNxArray=cudaSolid->GetSampleNxArrayPtr(nAxis);
		float *dev_sampleNyArray=cudaSolid->GetSampleNyArrayPtr(nAxis);
		float *dev_sampleDepthArray=cudaSolid->GetSampleDepthArrayPtr(nAxis);

		CUDA_SAFE_CALL( cudaMemcpy( dev_sampleNxArray, sampleNxArray, sampleNum*sizeof(float), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dev_sampleNyArray, sampleNyArray, sampleNum*sizeof(float), cudaMemcpyHostToDevice ) );
		CUDA_SAFE_CALL( cudaMemcpy( dev_sampleDepthArray, sampleDepthArray, sampleNum*sizeof(float), cudaMemcpyHostToDevice ) );

		free(sampleNxArray);	free(sampleNyArray);	free(sampleDepthArray);
	}
}

void LDNIcudaOperation::CopyCUDASolidToCPUSolid(LDNIcudaSolid *cudaSolid, LDNIcpuSolid* &cpuSolid)
{
	float ox,oy,oz,gWidth;	int i,num,res;	short nAxis;
	LDNIcpuRay *rays;	LDNIcpuSample *sampleArray;

	cudaSolid->GetOrigin(ox,oy,oz);		gWidth=cudaSolid->GetSampleWidth();
	res=cudaSolid->GetResolution();

	cpuSolid=new LDNIcpuSolid;			cpuSolid->SetOrigin(ox,oy,oz);		
	cpuSolid->SetSampleWidth(gWidth);	cpuSolid->MallocMemory(res);

	//-----------------------------------------------------------------------------------------
	//	copy the index arrays
	unsigned int *dev_indexArray,*indexArray;
	num=res*res;
	indexArray=(unsigned int *)malloc((num+1)*sizeof(unsigned int));
	for(nAxis=0;nAxis<3;nAxis++) {
		rays=cpuSolid->GetRayArrayPtr(nAxis);
		dev_indexArray=cudaSolid->GetIndexArrayPtr(nAxis);
		CUDA_SAFE_CALL( cudaMemcpy( indexArray, dev_indexArray, (num+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
		for(i=0;i<num;i++) rays[i].sampleIndex=indexArray[i+1];
	}
	free(indexArray);

	//-----------------------------------------------------------------------------------------
	//	copy the sample arrays
	for(nAxis=0;nAxis<3;nAxis++) {
		rays=cpuSolid->GetRayArrayPtr(nAxis);
		int sampleNum=rays[res*res-1].sampleIndex;

		float *sampleNxArray,*sampleNyArray,*sampleDepthArray;
		sampleNxArray=(float*)malloc(sampleNum*sizeof(float));
		sampleNyArray=(float*)malloc(sampleNum*sizeof(float));
		sampleDepthArray=(float*)malloc(sampleNum*sizeof(float));
		float *dev_sampleNxArray=cudaSolid->GetSampleNxArrayPtr(nAxis);
		float *dev_sampleNyArray=cudaSolid->GetSampleNyArrayPtr(nAxis);
		float *dev_sampleDepthArray=cudaSolid->GetSampleDepthArrayPtr(nAxis);

		CUDA_SAFE_CALL( cudaMemcpy( sampleNxArray, dev_sampleNxArray, sampleNum*sizeof(float), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( sampleNyArray, dev_sampleNyArray, sampleNum*sizeof(float), cudaMemcpyDeviceToHost ) );
		CUDA_SAFE_CALL( cudaMemcpy( sampleDepthArray, dev_sampleDepthArray, sampleNum*sizeof(float), cudaMemcpyDeviceToHost ) );

		cpuSolid->MallocSampleMemory(nAxis,sampleNum);
		sampleArray=cpuSolid->GetSampleArrayPtr(nAxis);
		for(i=0;i<sampleNum;i++) {
			sampleArray[i].nx=sampleNxArray[i];
			sampleArray[i].ny=sampleNyArray[i];
			double dd=1.0-sampleArray[i].nx*sampleArray[i].nx-sampleArray[i].ny*sampleArray[i].ny;
			if (dd<0.0) dd=0.0;		if (dd>1.0) dd=1.0;
			if (sampleDepthArray[i]<0) sampleArray[i].nz=-sqrt(dd); else sampleArray[i].nz=sqrt(dd);
			sampleArray[i].depth=fabs(sampleDepthArray[i]);
		}

		free(sampleNxArray);	free(sampleNyArray);	free(sampleDepthArray);
	}
}

void LDNIcudaOperation::_switchSolid(LDNIcudaSolid* solidA, LDNIcudaSolid* solidB)
{
	unsigned int *dev_indexArrayA[3];
	float *dev_sampleNxArrayA[3];
	float *dev_sampleNyArrayA[3];
	float *dev_sampleDepthArrayA[3];
	float originA[3],sampleWidthA;	
	int res,xSampleNum,ySampleNum,zSampleNum;
	float originB[3];	

	dev_indexArrayA[0]=solidA->GetIndexArrayPtr(0);				dev_indexArrayA[1]=solidA->GetIndexArrayPtr(1);				dev_indexArrayA[2]=solidA->GetIndexArrayPtr(2);
	dev_sampleNxArrayA[0]=solidA->GetSampleNxArrayPtr(0);		dev_sampleNxArrayA[1]=solidA->GetSampleNxArrayPtr(1);		dev_sampleNxArrayA[2]=solidA->GetSampleNxArrayPtr(2);
	dev_sampleNyArrayA[0]=solidA->GetSampleNyArrayPtr(0);		dev_sampleNyArrayA[1]=solidA->GetSampleNyArrayPtr(1);		dev_sampleNyArrayA[2]=solidA->GetSampleNyArrayPtr(2);
	dev_sampleDepthArrayA[0]=solidA->GetSampleDepthArrayPtr(0);	dev_sampleDepthArrayA[1]=solidA->GetSampleDepthArrayPtr(1);	dev_sampleDepthArrayA[2]=solidA->GetSampleDepthArrayPtr(2);
	solidA->GetOrigin(originA[0],originA[1],originA[2]);		sampleWidthA=solidA->GetSampleWidth();		res=solidA->GetResolution();
	xSampleNum=solidA->GetSampleNumber(0);	ySampleNum=solidA->GetSampleNumber(1);	zSampleNum=solidA->GetSampleNumber(2);

	solidA->SetIndexArrayPtr(0,solidB->GetIndexArrayPtr(0));	solidA->SetIndexArrayPtr(1,solidB->GetIndexArrayPtr(1));	solidA->SetIndexArrayPtr(2,solidB->GetIndexArrayPtr(2));
	solidA->SetSampleNxArrayPtr(0,solidB->GetSampleNxArrayPtr(0));	solidA->SetSampleNxArrayPtr(1,solidB->GetSampleNxArrayPtr(1));	solidA->SetSampleNxArrayPtr(2,solidB->GetSampleNxArrayPtr(2));
	solidA->SetSampleNyArrayPtr(0,solidB->GetSampleNyArrayPtr(0));	solidA->SetSampleNyArrayPtr(1,solidB->GetSampleNyArrayPtr(1));	solidA->SetSampleNyArrayPtr(2,solidB->GetSampleNyArrayPtr(2));
	solidA->SetSampleDepthArrayPtr(0,solidB->GetSampleDepthArrayPtr(0));	solidA->SetSampleDepthArrayPtr(1,solidB->GetSampleDepthArrayPtr(1));	solidA->SetSampleDepthArrayPtr(2,solidB->GetSampleDepthArrayPtr(2));
	solidB->GetOrigin(originB[0],originB[1],originB[2]);	solidA->SetOrigin(originB[0],originB[1],originB[2]);
	solidA->SetSampleWidth(solidB->GetSampleWidth());	solidA->SetResolution(solidB->GetResolution());
	solidA->SetSampleNumber(0,solidB->GetSampleNumber(0));		solidA->SetSampleNumber(1,solidB->GetSampleNumber(1));		solidA->SetSampleNumber(2,solidB->GetSampleNumber(2));

	solidB->SetIndexArrayPtr(0,dev_indexArrayA[0]);		solidB->SetIndexArrayPtr(1,dev_indexArrayA[1]);		solidB->SetIndexArrayPtr(2,dev_indexArrayA[2]);
	solidB->SetSampleNxArrayPtr(0,dev_sampleNxArrayA[0]);	solidB->SetSampleNxArrayPtr(1,dev_sampleNxArrayA[1]);	solidB->SetSampleNxArrayPtr(2,dev_sampleNxArrayA[2]);
	solidB->SetSampleNyArrayPtr(0,dev_sampleNyArrayA[0]);	solidB->SetSampleNyArrayPtr(1,dev_sampleNyArrayA[1]);	solidB->SetSampleNyArrayPtr(2,dev_sampleNyArrayA[2]);
	solidB->SetSampleDepthArrayPtr(0,dev_sampleDepthArrayA[0]);		solidB->SetSampleDepthArrayPtr(1,dev_sampleDepthArrayA[1]);		solidB->SetSampleDepthArrayPtr(2,dev_sampleDepthArrayA[2]);
	solidB->SetOrigin(originA[0],originA[1],originA[2]);	solidB->SetSampleWidth(sampleWidthA);	solidB->SetResolution(res);
	solidB->SetSampleNumber(0,xSampleNum);		solidB->SetSampleNumber(1,ySampleNum);		solidB->SetSampleNumber(2,zSampleNum);
}

void LDNIcudaOperation::_expansionLDNIcudaSolidByNewBoundingBox(LDNIcudaSolid *cudaSolid, float boundingBox[])
{
	unsigned int sd[3],ed[3],total;		float wx,wy,wz,origin[3],gWidth;
	unsigned int *dev_indexArray;
	float *dev_sampleDepthArray;
	long time=clock();

	cudaSolid->GetOrigin(origin[0],origin[1],origin[2]);
	gWidth=cudaSolid->GetSampleWidth();
	int res=cudaSolid->GetResolution();

	origin[0]=origin[0]-gWidth*0.5f;
	origin[1]=origin[1]-gWidth*0.5f;
	origin[2]=origin[2]-gWidth*0.5f;
	//------------------------------------------------------------------------------
	//	Step 1: determine the number of expansion
	boundingBox[0]=boundingBox[0]-gWidth*2.0f;	
	boundingBox[2]=boundingBox[2]-gWidth*2.0f;	
	boundingBox[4]=boundingBox[4]-gWidth*2.0f;	
	boundingBox[1]=boundingBox[1]+gWidth*2.0f;	
	boundingBox[3]=boundingBox[3]+gWidth*2.0f;	
	boundingBox[5]=boundingBox[5]+gWidth*2.0f;	
	//------------------------------------------------------------------------------
	sd[0]=sd[1]=sd[2]=0;
	if (boundingBox[0]<origin[0]) sd[0]=(unsigned int)((origin[0]-boundingBox[0])/gWidth)+1;
	if (boundingBox[2]<origin[1]) sd[1]=(unsigned int)((origin[1]-boundingBox[2])/gWidth)+1;
	if (boundingBox[4]<origin[2]) sd[2]=(unsigned int)((origin[2]-boundingBox[4])/gWidth)+1;
	//------------------------------------------------------------------------------
	wx=origin[0]+gWidth*(float)(res);
	wy=origin[1]+gWidth*(float)(res);
	wz=origin[2]+gWidth*(float)(res);
	ed[0]=ed[1]=ed[2]=0;
	if (boundingBox[1]>wx) ed[0]=(int)((boundingBox[1]-wx)/gWidth+0.5);
	if (boundingBox[3]>wy) ed[1]=(int)((boundingBox[3]-wy)/gWidth+0.5);
	if (boundingBox[5]>wz) ed[2]=(int)((boundingBox[5]-wz)/gWidth+0.5);
	//------------------------------------------------------------------------------
	total=sd[0]+ed[0];
	if ((sd[1]+ed[1])>total) total=sd[1]+ed[1];
	if ((sd[2]+ed[2])>total) total=sd[2]+ed[2];
	ed[0]=total-sd[0];	ed[1]=total-sd[1];	ed[2]=total-sd[2];

	//------------------------------------------------------------------------------
	//	Step 2: create new index Arrays of LDNISolidNode
	unsigned int newArrsize;
	newArrsize=(unsigned int)(res+total)*(res+total);
	unsigned int *tempIndexArray;	
	CUDA_SAFE_CALL( cudaMalloc( (void**)&tempIndexArray, (newArrsize+1)*sizeof(unsigned int) ) );
	for(short nAxis=0; nAxis<3; nAxis++) {
		dev_indexArray=cudaSolid->GetIndexArrayPtr(nAxis);
		CUDA_SAFE_CALL( cudaMemset( (void*)tempIndexArray, 0, (newArrsize+1)*sizeof(unsigned int) ) );

		//------------------------------------------------------------------
		//	fill the temporary index array by number of samples on each ray
		krLDNIcudaSolid_fillNewIndexBySampleNumber<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(
									tempIndexArray, dev_indexArray, res, res+total, sd[(nAxis+1)%3], sd[(nAxis+2)%3]);

		//------------------------------------------------------------------
		//	scan the index array
		thrust::device_ptr<unsigned int> dev_ptr(tempIndexArray);			//	Wrap raw pointers with dev_ptr
		thrust::exclusive_scan(dev_ptr, dev_ptr+(newArrsize+1), dev_ptr);	//	in-place scan
		
		//------------------------------------------------------------------
		//	update the temporary index array
		cudaFree(dev_indexArray);
		CUDA_SAFE_CALL( cudaMalloc( (void**)&(dev_indexArray), (newArrsize+1)*sizeof(unsigned int) ) );
		cudaSolid->SetIndexArrayPtr(nAxis,dev_indexArray);
		CUDA_SAFE_CALL( cudaMemcpy( dev_indexArray, tempIndexArray, (newArrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToDevice ) );
	}
	cudaFree(tempIndexArray);

	//------------------------------------------------------------------------------
	//	Step 3: update the depth-values of samples when necessary
	origin[0]=origin[0]-gWidth*(float)(sd[0])+gWidth*0.5;
	origin[1]=origin[1]-gWidth*(float)(sd[1])+gWidth*0.5;
	origin[2]=origin[2]-gWidth*(float)(sd[2])+gWidth*0.5;
	cudaSolid->SetOrigin(origin[0],origin[1],origin[2]);
	res+=total;		cudaSolid->SetResolution(res);
	for(short nAxis=0; nAxis<3; nAxis++) {
		if (sd[nAxis]==0) continue;
		float updateDepth=gWidth*(float)sd[nAxis];
		
		dev_indexArray=cudaSolid->GetIndexArrayPtr(nAxis);
		dev_sampleDepthArray=cudaSolid->GetSampleDepthArrayPtr(nAxis);

		unsigned int sampleNum;
		CUDA_SAFE_CALL( cudaMemcpy( &sampleNum, &(dev_indexArray[newArrsize]), sizeof(unsigned int), cudaMemcpyDeviceToHost ) );

		krLDNIcudaSolid_depthSampleAdd<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(dev_sampleDepthArray, updateDepth, sampleNum);
	}

	//------------------------------------------------------------------------------
	//	Step 4: update the boundingBox[] for the sampling of mesh surface bounded by it
	boundingBox[0]=origin[0]-gWidth*0.5;
	boundingBox[2]=origin[1]-gWidth*0.5;
	boundingBox[4]=origin[2]-gWidth*0.5;
	boundingBox[1]=boundingBox[0]+gWidth*((float)res);
	boundingBox[3]=boundingBox[2]+gWidth*((float)res);
	boundingBox[5]=boundingBox[4]+gWidth*((float)res);

	printf("-----------------------------------------------------------------------\n");
	printf("Expanding the working space of existing cuda solid takes: %ld (ms)\n",clock()-time);
	printf("The resolution is extended from %d to %d\n",res-total,res);
	printf("-----------------------------------------------------------------------\n");
}

//--------------------------------------------------------------------------------------------
bool initGLInteroperabilityOnCUDA(int major, int minor) {
    cudaDeviceProp  prop;
    int dev;
    memset( &prop, 0, sizeof( cudaDeviceProp ) );
    prop.major = major;
    prop.minor = minor;
    CUDA_SAFE_CALL( cudaChooseDevice( &dev, &prop ) );
    // tell CUDA which dev we will be using for graphic interop
    // from the programming guide:  Interoperability with OpenGL
    //     requires that the CUDA device be specified by
    //     cudaGLSetGLDevice() before any other runtime calls.

	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

	if (deviceProp.major < 2)
	{
		return false;
	}
	else
	{
		printf("Current device support compute capability 2.0 \n");
	}
    CUDA_SAFE_CALL( cudaGLSetGLDevice( dev ) );
	return true;
}




//--------------------------------------------------------------------------------------------
void LDNIcudaOperation::GetCudaDeviceProperty()
{
    cudaDeviceProp  prop;

    int count;
    CUDA_SAFE_CALL( cudaGetDeviceCount( &count ) );
    for (int i=0; i< count; i++) {
        CUDA_SAFE_CALL( cudaGetDeviceProperties( &prop, i ) );
        printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop.name );
        printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
        printf( "Clock rate:  %d\n", prop.clockRate );
        printf( "Device copy overlap:  " );
        if (prop.deviceOverlap)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n");
        printf( "Kernel execution timeout :  " );
        if (prop.kernelExecTimeoutEnabled)
            printf( "Enabled\n" );
        else
            printf( "Disabled\n" );

        printf( "   --- Memory Information for device %d ---\n", i );
        printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
        printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
        printf( "Max mem pitch:  %ld\n", prop.memPitch );
        printf( "Texture Alignment:  %ld\n", prop.textureAlignment );

        printf( "   --- MP Information for device %d ---\n", i );
        printf( "Multiprocessor count:  %d\n",
                    prop.multiProcessorCount );
        printf( "Shared mem per mp:  %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per mp:  %d\n", prop.regsPerBlock );
        printf( "Threads in warp:  %d\n", prop.warpSize );
        printf( "Max threads per block:  %d\n",
                    prop.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n",
                    prop.maxThreadsDim[0], prop.maxThreadsDim[1],
                    prop.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",
                    prop.maxGridSize[0], prop.maxGridSize[1],
                    prop.maxGridSize[2] );
        printf( "\n" );
    }
}

////////////////////////////////////////////////////////////////////////////////////////
//
//	The following functions are running on the graphics hardware by CUDA
//

__global__ void krLDNIRegularization_RegularizationOnRays(float *devNxArrayPtr, float *devNyArrayPtr, float *devDepthArrayPtr, 
														  unsigned int *devIndexArrayPtr, unsigned int *devIndexArrayPtrRes, int arrsize, float eps)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int stIndex,sampleNum,i,resSampleNum;
	float resNx[MAX_NUM_OF_SAMPLES_ON_RAY],resNy[MAX_NUM_OF_SAMPLES_ON_RAY],resDepth[MAX_NUM_OF_SAMPLES_ON_RAY];

	while(index<arrsize) {
		stIndex=devIndexArrayPtr[index];	sampleNum=devIndexArrayPtr[index+1]-stIndex;

//		if (sampleNum>0) sampleNum=sampleNum-2;

		{
			//------------------------------------------------------------------------------
			//	Eliminating gaps 
			resSampleNum=0;
			if (sampleNum>0) {
				resNx[0]=devNxArrayPtr[stIndex];	resNy[0]=devNyArrayPtr[stIndex];	resDepth[0]=devDepthArrayPtr[stIndex];	resSampleNum++;
				for(i=1;i<sampleNum;i+=2) {
					if (fabs(devDepthArrayPtr[stIndex+i+1])-fabs(devDepthArrayPtr[stIndex+i])<eps)  continue;
					
					resNx[resSampleNum]=devNxArrayPtr[stIndex+i];			
					resNy[resSampleNum]=devNyArrayPtr[stIndex+i];
					resDepth[resSampleNum]=devDepthArrayPtr[stIndex+i];		
					resSampleNum++;

					resNx[resSampleNum]=devNxArrayPtr[stIndex+i+1];			
					resNy[resSampleNum]=devNyArrayPtr[stIndex+i+1];
					resDepth[resSampleNum]=devDepthArrayPtr[stIndex+i+1];		
					resSampleNum++;
				}
				resNx[resSampleNum]=devNxArrayPtr[stIndex+sampleNum-1];		
				resNy[resSampleNum]=devNyArrayPtr[stIndex+sampleNum-1];
				resDepth[resSampleNum]=devDepthArrayPtr[stIndex+sampleNum-1];	

				resSampleNum++;
			}

			//------------------------------------------------------------------------------
			//	Eliminating super-thin sheets
			sampleNum=0;
			for(i=0;i<resSampleNum;i+=2) {
				if (fabs(resDepth[i+1])-fabs(resDepth[i])<eps)  continue;

				devNxArrayPtr[stIndex+sampleNum]=resNx[i];			
				devNyArrayPtr[stIndex+sampleNum]=resNy[i];
				devDepthArrayPtr[stIndex+sampleNum]=resDepth[i];	
				sampleNum++;

				devNxArrayPtr[stIndex+sampleNum]=resNx[i+1];		
				devNyArrayPtr[stIndex+sampleNum]=resNy[i+1];
				devDepthArrayPtr[stIndex+sampleNum]=resDepth[i+1];	
				sampleNum++;
			}
		}


		devIndexArrayPtrRes[index]=sampleNum;

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIRegularization_ResultSampleCollection(float *devNxArrayPtr, float *devNyArrayPtr, float *devDepthArrayPtr, 
															unsigned int *devIndexArrayPtr,
															float *devNxArrayPtrRes, float *devNyArrayPtrRes, float *devDepthArrayPtrRes, 
															unsigned int *devIndexArrayPtrRes, int arrsize)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int st,num,stRes,numRes,k;

	while(index<arrsize) {
		st=devIndexArrayPtr[index];			num=devIndexArrayPtr[index+1]-st;
		stRes=devIndexArrayPtrRes[index];	numRes=devIndexArrayPtrRes[index+1]-stRes;

		if (numRes<=num) {
			for(k=0;k<numRes;k++) {
				devNxArrayPtrRes[stRes+k]=devNxArrayPtr[st+k];
				devNyArrayPtrRes[stRes+k]=devNyArrayPtr[st+k];
				devDepthArrayPtrRes[stRes+k]=devDepthArrayPtr[st+k];
			}
		}
		else {	//	This rarely occurs.
			for(k=0;k<num;k++) {
				devNxArrayPtrRes[stRes+k]=devNxArrayPtr[st+k];
				devNyArrayPtrRes[stRes+k]=devNyArrayPtr[st+k];
				devDepthArrayPtrRes[stRes+k]=devDepthArrayPtr[st+k];
			}
		}

		index += blockDim.x * gridDim.x;
	}
}
#define S_EPS 1.0e-6
__global__ void krLDNIBoolean_SuperUnionOnRays(float *devNxArrayPtr, float *devNyArrayPtr, float *devDepthArrayPtr, unsigned int *devIndexArrayPtr,
											unsigned int *devIndexArrayPtrRes, int arrsize)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int k,st,num,count_start,count_end,count;
	float resNx[MAX_NUM_OF_SAMPLES_ON_RAY],resNy[MAX_NUM_OF_SAMPLES_ON_RAY],resDepth[MAX_NUM_OF_SAMPLES_ON_RAY],s_depth, e_depth;

	while(index<arrsize) {
		st=devIndexArrayPtr[index];	num=devIndexArrayPtr[index+1]-st;
		count_start = 0;
		count_end = 0;
		count = 0;

		if (num%2 == 1)
		{
			
			for(k=0; k <num ; k++)
			{
				devNxArrayPtr[st+k]=0;
				devNyArrayPtr[st+k]=0;
				devDepthArrayPtr[st+k]=0;

				devIndexArrayPtrRes[index]=0;				
			}
		}
		if (num > 0 && num%2==0)
		{			
			resDepth[0]	= s_depth = START_DEPTH(devDepthArrayPtr[st]);
			resNx[0]	= devNxArrayPtr[st];
			resNy[0]	= devNyArrayPtr[st];
			count_start++;
			count++;
			
			e_depth = END_DEPTH(devDepthArrayPtr[num/2+st]);
			count_end++;


			


			for(k=1; k < num/2; k++)
			{
				s_depth = START_DEPTH(devDepthArrayPtr[k+st]);
				if (((fabs(s_depth)- fabs(e_depth))>S_EPS) && (count_start == count_end))
				{
					resDepth[count]	= e_depth;
					resNx[count]	= devNxArrayPtr[st+(k-1)+num/2];
					resNy[count]	= devNyArrayPtr[st+(k-1)+num/2];
					count++;

					resDepth[count] = s_depth;
					resNx[count]	= devNxArrayPtr[st+k];
					resNy[count]	= devNyArrayPtr[st+k];
					count_start++;
					count++;

				}
				//else if (fabs(s_depth) <= fabs(e_depth))
				else if ((fabs(s_depth)- fabs(e_depth))<=S_EPS)
				{
					count_start++;
				}

				e_depth = END_DEPTH(devDepthArrayPtr[num/2+k+st]);
				count_end++;
				
			}

			if ((fabs(e_depth)-fabs(s_depth))<S_EPS) 
			{
				count--;
			}
			else
			{
				resDepth[count]	= e_depth;
				resNx[count]	= devNxArrayPtr[st+(k-1)+num/2];
				resNy[count]	= devNyArrayPtr[st+(k-1)+num/2];
				count++;
			}
			
			devIndexArrayPtrRes[index]=count;

			
			for(k=0; k <count ; k++)
			{
				devNxArrayPtr[st+k]=resNx[k];
				devNyArrayPtr[st+k]=resNy[k];
				devDepthArrayPtr[st+k]=resDepth[k];				
			}


			
		}
		index += blockDim.x * gridDim.x;
	}

}

__global__ void krLDNIBoolean_IdentifyEnterLeaveOnRays(float *devNxArrayPtr, float *devNyArrayPtr, float *devDepthArrayPtr, unsigned int *devIndexArrayPtr, int arrsize)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int k,st,num;
	unsigned int prev_mesh,count;
	float depth, fdepth;
	float resDepth[MAX_NUM_OF_SAMPLES_ON_RAY];

	while(index<arrsize) {
		st=devIndexArrayPtr[index];	num=devIndexArrayPtr[index+1]-st;
		prev_mesh = 0;
		count = 0;
		if (num > 0)
		{
			prev_mesh = floor(fabs(devDepthArrayPtr[st]));
			for(k=0; k<num; k++)
			{				
				depth = devDepthArrayPtr[k+st];
				fdepth = fabs(depth);
				//if (floor(fdepth) != prev_mesh)
				if (fabs(floor(fdepth)-prev_mesh) >= 1.0)
				{
					prev_mesh = floor(fdepth);	count=0;
				}
				if (count%2 == 0)
				{
					fdepth = fdepth - floor(fdepth) + 1;   // all starting pos : 1.xxx
					
				}
				else
				{
					fdepth = fdepth - floor(fdepth) + 2;   // all ending pos : 2.xxx
					
				}

				if (depth < 0) resDepth[k] = -fdepth;
				else resDepth[k] = fdepth;

				count++;

			}
			
			for(k=0; k <num; k++)
			{
				devDepthArrayPtr[st+k]=resDepth[k];
				
			}
		}
		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIBoolean_BooleanOnRays(float *devNxArrayPtrA, float *devNyArrayPtrA, float *devDepthArrayPtrA, unsigned int *devIndexArrayPtrA,
											float *devNxArrayPtrB, float *devNyArrayPtrB, float *devDepthArrayPtrB, unsigned int *devIndexArrayPtrB, 
											unsigned int *devIndexArrayPtrRes, int arrsize, short nOperationType)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int k,stA,stB,numA,numB,numRes,aIndex,bIndex;
	bool last_op,op,insideA,insideB;
	float lastNx,lastNy,lastDepth;
	float resNx[MAX_NUM_OF_SAMPLES_ON_RAY],resNy[MAX_NUM_OF_SAMPLES_ON_RAY],resDepth[MAX_NUM_OF_SAMPLES_ON_RAY];

	while(index<arrsize) {
		stA=devIndexArrayPtrA[index];	numA=devIndexArrayPtrA[index+1]-stA;
		stB=devIndexArrayPtrB[index];	numB=devIndexArrayPtrB[index+1]-stB;

		last_op=insideA=insideB=false;	numRes=0;	
	
		//-------------------------------------------------------------------------------------------------------
		//	Generate the temporary resultant samples
		if (numA>0 && numB>0) {
			aIndex=bIndex=0;	
			while( (aIndex<numA) || (bIndex<numB) ) {	// scaning the samples on solidA and solidB together

				if ((bIndex==numB) || (aIndex<numA && fabs(devDepthArrayPtrA[aIndex+stA])<fabs(devDepthArrayPtrB[bIndex+stB]))) 
				{
					// advancing on ray-A
					lastDepth=devDepthArrayPtrA[aIndex+stA];
					lastNx=devNxArrayPtrA[aIndex+stA];
					lastNy=devNyArrayPtrA[aIndex+stA];
					insideA=!insideA;	aIndex++;
				}
				else {
					// advancing on ray-B
					lastDepth=devDepthArrayPtrB[bIndex+stB];
					lastNx=devNxArrayPtrB[bIndex+stB];
					lastNy=devNyArrayPtrB[bIndex+stB];
					if (nOperationType==2) {lastNx=-lastNx;	lastNy=-lastNy;	lastDepth=-lastDepth;}	// inverse the normal
					insideB=!insideB;	bIndex++;
				}

				switch(nOperationType) {
				case 0:{op=LOGIC_UNION(insideA,insideB); }break;
				case 1:{op=LOGIC_INTER(insideA,insideB); }break;
				case 2:{op=LOGIC_SUBTR(insideA,insideB); }break;
				}

				if (op!=last_op) 
				{
					if (numRes>0 && fabs(fabs(lastDepth)-fabs(resDepth[numRes-1]))<0.00001f) 
						{numRes--;}
					else {
						resDepth[numRes]=lastDepth;	
						resNx[numRes]=lastNx;	resNy[numRes]=lastNy;	
						numRes++;
					}
					last_op=op;
				}
			}
		}
		else if ((numA==0) && (numB>0)) {	// scaning the samples on solidB
			if (nOperationType==0) {
				for(k=0;k<numB;k++) {
					resNx[k]=devNxArrayPtrB[stB+k];	
					resNy[k]=devNyArrayPtrB[stB+k];
					resDepth[k]=devDepthArrayPtrB[stB+k];
				}
				numRes=numB;
			}
			// for "intersect" and "difference", keeping NULL will be fine
		}
		else if ((numA>0) && (numB==0)) {	// scaning the samples on solidA
			if (nOperationType==0 || nOperationType==2) { // union and difference
				for(k=0;k<numA;k++) {
					resNx[k]=devNxArrayPtrA[stA+k];	
					resNy[k]=devNyArrayPtrA[stA+k];
					resDepth[k]=devDepthArrayPtrA[stA+k];
				}
				numRes=numA;
			}
		}

		//-------------------------------------------------------------------------------------------------------
		//	Copy the resultant samples into solidA and solidB
		if (numRes>numA) {
			for(k=0;k<numA;k++) {
				devNxArrayPtrA[stA+k]=resNx[k];
				devNyArrayPtrA[stA+k]=resNy[k];
				devDepthArrayPtrA[stA+k]=resDepth[k];
			}
			for(k=numA;k<numRes;k++) {
				devNxArrayPtrB[stB+k-numA]=resNx[k];	
				devNyArrayPtrB[stB+k-numA]=resNy[k];
				devDepthArrayPtrB[stB+k-numA]=resDepth[k];
			}
		}
		else {
			for(k=0;k<numRes;k++) {
				devNxArrayPtrA[stA+k]=resNx[k];
				devNyArrayPtrA[stA+k]=resNy[k];
				devDepthArrayPtrA[stA+k]=resDepth[k];
			}
		}
		
		devIndexArrayPtrRes[index]=numRes;

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIBoolean_ResultSampleCollection(float *devNxArrayPtrA, float *devNyArrayPtrA, float *devDepthArrayPtrA, unsigned int *devIndexArrayPtrA,
													 float *devNxArrayPtrRes, float *devNyArrayPtrRes, float *devDepthArrayPtrRes, unsigned int *devIndexArrayPtrRes, int arrsize, float width, float gwidth)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int stA,stRes,numRes,k,numA;
	float depth, temp;

	while(index<arrsize) {
		stA=devIndexArrayPtrA[index];		numA=devIndexArrayPtrA[index+1]-stA;
		stRes=devIndexArrayPtrRes[index];	numRes=devIndexArrayPtrRes[index+1]-stRes;

		if (numRes>0) {
			for(k=0;k<numRes;k++) {
				devNxArrayPtrRes[stRes+k]=devNxArrayPtrA[stA+k];
				devNyArrayPtrRes[stRes+k]=devNyArrayPtrA[stA+k];
				depth = devDepthArrayPtrA[stA+k];
				temp = fabs(depth)*width-gwidth*0.5f;
				if (depth < 0)
					devDepthArrayPtrRes[stRes+k]=-temp;
				else
					devDepthArrayPtrRes[stRes+k]=temp;
			}
	
		}

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIBoolean_ResultSampleCollection(float *devNxArrayPtrA, float *devNyArrayPtrA, float *devDepthArrayPtrA, unsigned int *devIndexArrayPtrA,
													 float *devNxArrayPtrB, float *devNyArrayPtrB, float *devDepthArrayPtrB, unsigned int *devIndexArrayPtrB, 
													 float *devNxArrayPtrRes, float *devNyArrayPtrRes, float *devDepthArrayPtrRes, unsigned int *devIndexArrayPtrRes, int arrsize)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int stA,numA,stB,stRes,numRes,k;

	while(index<arrsize) {
		stA=devIndexArrayPtrA[index];		numA=devIndexArrayPtrA[index+1]-stA;
		stRes=devIndexArrayPtrRes[index];	numRes=devIndexArrayPtrRes[index+1]-stRes;

		if (numRes>0) {
			if (numRes>numA) {
				for(k=0;k<numA;k++) {
					devNxArrayPtrRes[stRes+k]=devNxArrayPtrA[stA+k];
					devNyArrayPtrRes[stRes+k]=devNyArrayPtrA[stA+k];
					devDepthArrayPtrRes[stRes+k]=devDepthArrayPtrA[stA+k];
				}
				stB=devIndexArrayPtrB[index];
				for(k=numA;k<numRes;k++) {
					devNxArrayPtrRes[stRes+k]=devNxArrayPtrB[stB+(k-numA)];
					devNyArrayPtrRes[stRes+k]=devNyArrayPtrB[stB+(k-numA)];
					devDepthArrayPtrRes[stRes+k]=devDepthArrayPtrB[stB+(k-numA)];
				}
			}
			else {
				for(k=0;k<numRes;k++) {
					devNxArrayPtrRes[stRes+k]=devNxArrayPtrA[stA+k];
					devNyArrayPtrRes[stRes+k]=devNyArrayPtrA[stA+k];
					devDepthArrayPtrRes[stRes+k]=devDepthArrayPtrA[stA+k];
				}
			}
		}

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNISampling_SortSamples(float *devNxArrayPtr, float *devNyArrayPtr, float *devDepthArrayPtr, 
										   int arrsize, unsigned int *devIndexArrayPtr)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int st,ed,i,j,n;		
	float nx[MAX_NUM_OF_SAMPLES_ON_RAY],ny[MAX_NUM_OF_SAMPLES_ON_RAY],depth[MAX_NUM_OF_SAMPLES_ON_RAY];
	float tempnx,tempny,tempdepth;
//	float auxNx[MAX_NUM_OF_SAMPLES_ON_RAY/2+1],auxNy[MAX_NUM_OF_SAMPLES_ON_RAY/2+1],auxDepth[MAX_NUM_OF_SAMPLES_ON_RAY/2+1];	// for merge-sort
//	int lo,hi,m,k;	// for merge-sort

	while(index<arrsize) {
		st=devIndexArrayPtr[index];		ed=devIndexArrayPtr[index+1];	n=ed-st;

		//-----------------------------------------------------------------------------------------------------------
		//	Download data set
		for(i=0;i<n;i++) nx[i]=devNxArrayPtr[st+i];
		for(i=0;i<n;i++) ny[i]=devNyArrayPtr[st+i];
		for(i=0;i<n;i++) depth[i]=devDepthArrayPtr[st+i];

		//-----------------------------------------------------------------------------------------------------------
		for(i=0;i<n;i++) {
			for(j=i+1;j<n;j++) {
				if (fabs(depth[i])>fabs(depth[j])) {
					tempnx=nx[i];	nx[i]=nx[j];	nx[j]=tempnx;
					tempny=ny[i];	ny[i]=ny[j];	ny[j]=tempny;
					tempdepth=depth[i];	depth[i]=depth[j];	depth[j]=tempdepth;
				}
			}
		}
				

		//-----------------------------------------------------------------------------------------------------------
		//	Upload data set
		for(i=0;i<n;i++) devNxArrayPtr[st+i]=nx[i];	
		for(i=0;i<n;i++) devNyArrayPtr[st+i]=ny[i];	
		for(i=0;i<n;i++) devDepthArrayPtr[st+i]=depth[i];

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNISampling_CopySamples(float *devNxArrayPtr, 
										   float *devNyArrayPtr, float *devDepthArrayPtr, 
										   int n, int arrsize, float width, float sampleWidth, int res, 
										   unsigned int *devIndexArrayPtr)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int arrindex, num, ix, iy;
	float4 rgb;		float temp;

	while(index<arrsize) {
		num=devIndexArrayPtr[index+1]-devIndexArrayPtr[index];
		if (num>=n) {
			arrindex=(int)(devIndexArrayPtr[index])+n-1;

			ix=index%res;	iy=(index/res);
			rgb = tex2D(tex2DFloat4In, ix, iy);

			temp=fabs(rgb.z)*width-sampleWidth*0.5f;
				
			devNxArrayPtr[arrindex]=rgb.x;		// x-component of normal	
			devNyArrayPtr[arrindex]=rgb.y;		// y-component of normal
			if (rgb.z<0) devDepthArrayPtr[arrindex]=-temp; else devDepthArrayPtr[arrindex]=temp;
		}
		index += blockDim.x * gridDim.x;
	}
}


__global__ void krLDNISuperUnion_CopySamples(float *devNxArrayPtr, 
										   float *devNyArrayPtr, float *devDepthArrayPtr, 
										   int n, int arrsize, int res, 
										   unsigned int *devIndexArrayPtr)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int arrindex, num, ix, iy;
	float4 rgb;		//float temp;

	while(index<arrsize) {
		num=devIndexArrayPtr[index+1]-devIndexArrayPtr[index];
		if (num>=n) {
			arrindex=(int)(devIndexArrayPtr[index])+n-1;

			ix=index%res;	iy=(index/res);
			rgb = tex2D(tex2DFloat4In, ix, iy);

			devNxArrayPtr[arrindex]=rgb.x;		// x-component of normal	
			devNyArrayPtr[arrindex]=rgb.y;		// y-component of normal
			devDepthArrayPtr[arrindex]=rgb.z;

			
		}
		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNISampling_CopyIndexAndFindMax(unsigned char *devStencilBufferPtr, unsigned int *devIndexArrayPtr, 
												   unsigned int *devResArrayPtr, int arrsize ) 
{
	__shared__ unsigned int cache[THREADS_PER_BLOCK];
	int tid=threadIdx.x+blockIdx.x*blockDim.x;
	int cacheIndex=threadIdx.x; 

	unsigned int temp=0,temp2;
	while(tid<arrsize) {
		temp2=(unsigned int)(devStencilBufferPtr[tid]);
		devIndexArrayPtr[tid]=temp2;
		temp= MAX(temp, temp2);
		tid += blockDim.x * gridDim.x;
	}

	// set the cache values
	cache[cacheIndex]=temp;

	// synchronize threads in this block
	__syncthreads();

	// for reductions, THREADS_PER_BLOCK must be a power of 2 because of the following code
	int i = blockDim.x/2;
	while (i!=0) {
		if (cacheIndex < i) {cache[cacheIndex] = MAX(cache[cacheIndex], cache[cacheIndex+i]);}
		__syncthreads();
		i /= 2;
	}

	if (cacheIndex==0) devResArrayPtr[blockIdx.x] = cache[0];
}


///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////

bool LDNIcudaOperation::ScaffoldBooleanOperation(LDNIcudaSolid* &outputSolid,QuadTrglMesh *UnitMesh, int UnitNum[], float UnitOff[], int UnitFlip[], int nRes, LDNIcudaSolid* savedSolid)
{
	LDNIcudaSolid *solidB;
	int res=nRes;
	
	float boundingbox[6];
	float UnitWidth[3];

	boundingbox[0]=boundingbox[1]=boundingbox[2]=boundingbox[3]=boundingbox[4]=boundingbox[5]=0;

	UnitMesh->CompBoundingBox(boundingbox);


	UnitWidth[0] = boundingbox[1] - boundingbox[0] ;
	UnitWidth[1] = boundingbox[3] - boundingbox[2] ;
	UnitWidth[2] = boundingbox[5] - boundingbox[4] ;

	boundingbox[1] = boundingbox[1] + (UnitNum[0]-1)*(UnitWidth[0]+UnitOff[0]);
	boundingbox[3] = boundingbox[3] + (UnitNum[1]-1)*(UnitWidth[1]+UnitOff[1]);
	boundingbox[5] = boundingbox[5] + (UnitNum[2]-1)*(UnitWidth[2]+UnitOff[2]);

	float xx=(boundingbox[0]+boundingbox[1])*0.5f;
	float yy=(boundingbox[2]+boundingbox[3])*0.5f;
	float zz=(boundingbox[4]+boundingbox[5])*0.5f;
	float ww=boundingbox[1]-boundingbox[0];
	if ((boundingbox[3]-boundingbox[2])>ww) ww=boundingbox[3]-boundingbox[2];
	if ((boundingbox[5]-boundingbox[4])>ww) ww=boundingbox[5]-boundingbox[4];

	ww=ww*0.55+ww/(float)(res-1)*2.0;

	boundingbox[0]=xx-ww;	boundingbox[1]=xx+ww;
	boundingbox[2]=yy-ww;	boundingbox[3]=yy+ww;
	boundingbox[4]=zz-ww;	boundingbox[5]=zz+ww;


	if (savedSolid!= NULL)
	{		
		_expansionLDNIcudaSolidByNewBoundingBox(savedSolid, boundingbox);
		res = savedSolid->GetResolution();
	}


	//even row + even column
	InstancedBRepToLDNISampling(UnitMesh, outputSolid, boundingbox, res, UnitOff, UnitNum, UnitWidth, UnitFlip, false, true);

	//even row + single column
	InstancedBRepToLDNISampling(UnitMesh, solidB, boundingbox, res, UnitOff, UnitNum, UnitWidth, UnitFlip, false, false);


	printf("-----------------------------------------------------------------------\n");
	printf("Starting to compute Boolean operation\n");
	printf("-----------------------------------------------------------------------\n");
	_booleanOperation(outputSolid, solidB, 0);


	//LDNIcudaSolid *solidA;
	InstancedBRepToLDNISampling(UnitMesh, solidB, boundingbox, res, UnitOff, UnitNum, UnitWidth, UnitFlip, true, true);
	_booleanOperation(outputSolid, solidB, 0);


	InstancedBRepToLDNISampling(UnitMesh, solidB, boundingbox, res, UnitOff, UnitNum, UnitWidth, UnitFlip, true, false);
	_booleanOperation(outputSolid, solidB, 0);
	//even row + single column
	//InstancedBRepToLDNISampling(UnitMesh, solidA, boundingbox, res, UnitOff, UnitNum, UnitWidth, UnitFlip, false, false);





	outputSolid->SetBoundingBox(boundingbox);

	//-----------------------------------------------------------------------------------
	//	Step 4: free the memory
	delete solidB;


	return true;
}


bool LDNIcudaOperation::InstancedBRepToLDNISampling(QuadTrglMesh *mesh, LDNIcudaSolid* &solid, float boundingBox[], int res, float UnitOff[], int UnitNum[], float UnitWidth[], int UnitFlip[], bool bsingleRow, bool bsingleCol)
{
	const bool bCube=true;
	float origin[3],gWidth;		long time=clock(),totalTime=clock();
	int i,nodeNum,faceNum;
	char fileadd[256];


	solid=new LDNIcudaSolid;
	solid->MallocMemory(res);
	gWidth=(boundingBox[1]-boundingBox[0])/(float)res;
	solid->SetSampleWidth(gWidth);
	origin[0]=boundingBox[0]+gWidth*0.5f;
	origin[1]=boundingBox[2]+gWidth*0.5f;
	origin[2]=boundingBox[4]+gWidth*0.5f;
	solid->SetOrigin(origin[0],origin[1],origin[2]);


	//---------------------------------------------------------------------------------
	//	For using OpenGL Shading Language to implement the sampling procedure
	if (glewInit() != GLEW_OK) {printf("glewInit failed. Exiting...\n");	return false;}
	if (glewIsSupported("GL_VERSION_2_0")) {printf("\nReady for OpenGL 2.0\n");} else {printf("OpenGL 2.0 not supported\n"); return false;}
	//-----------------------------------------------------------------------------------------
	GLhandleARB g_programObj, g_vertexShader, g_GeometryShader, g_FragShader;
	const char *VshaderString[1],*GshaderString[1], *FshaderString[1];
	GLint bCompiled = 0, bLinked = 0;
	GLuint vbo, vboInd;
	char str[4096] = "";		
	//-----------------------------------------------------------------------------------------
	//	Step 1: Setup the shaders 
	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"ScaffoldLDNIVertexShader.vert");
	g_vertexShader = glCreateShaderObjectARB( GL_VERTEX_SHADER_ARB );
	unsigned char *ShaderAssembly = _readShaderFile( fileadd );
	VshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_vertexShader, 1, VshaderString, NULL );
	glCompileShaderARB( g_vertexShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_vertexShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {
		glGetInfoLogARB(g_vertexShader, sizeof(str), NULL, str);
		printf("Warning: Vertex Shader Compile Error \n%s\n",str);	return false;
	}
	//-----------------------------------------------------------------------------
	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"ScaffoldLDNIGeometryShader.geo");
	g_GeometryShader = glCreateShaderObjectARB( GL_GEOMETRY_SHADER_EXT );
	ShaderAssembly = _readShaderFile( fileadd );
	GshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_GeometryShader, 1, GshaderString, NULL );
	glCompileShaderARB( g_GeometryShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_GeometryShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {
		glGetInfoLogARB(g_GeometryShader, sizeof(str), NULL, str);
		printf("Warning: Geo Shader Compile Error\n%s\n",str);		return false;
	} 
	//-----------------------------------------------------------------------------
	memset(fileadd,0,256*sizeof(char));
	strcat(fileadd,"ScaffoldLDNIFragmentShader.frag");
	g_FragShader = glCreateShaderObjectARB( GL_FRAGMENT_SHADER_ARB );
	ShaderAssembly = _readShaderFile( fileadd );
	FshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_FragShader, 1, FshaderString, NULL );
	glCompileShaderARB( g_FragShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_FragShader, GL_OBJECT_COMPILE_STATUS_ARB, &bCompiled );
	if (bCompiled  == false) {
		glGetInfoLogARB(g_FragShader, sizeof(str), NULL, str);
		printf("Warning: Vertex Shader Compile Error\n\n");	return false;
	}
	g_programObj = glCreateProgramObjectARB();
	if (glGetError()!=GL_NO_ERROR) printf("Error: OpenGL!\n\n");
	glAttachObjectARB( g_programObj, g_vertexShader );		if (glGetError()!=GL_NO_ERROR) printf("Error: attach Vertex Shader!\n\n");
	glAttachObjectARB( g_programObj, g_GeometryShader );	if (glGetError()!=GL_NO_ERROR) printf("Error: attach Geometry Shader!\n\n");
	glAttachObjectARB( g_programObj, g_FragShader );		if (glGetError()!=GL_NO_ERROR) printf("Error: attach Fragment Shader!\n\n");
	//-----------------------------------------------------------------------------
	//	Configuration setting for geometry shader
	glLinkProgramARB( g_programObj);
	glGetObjectParameterivARB( g_programObj, GL_OBJECT_LINK_STATUS_ARB, &bLinked );
	if( bLinked == false ) {
		glGetInfoLogARB( g_programObj, sizeof(str), NULL, str );
		printf("Linking Fail: %s\n",str);	return false;
	}
	//-----------------------------------------------------------------------------------------
	//	Step 2:  creating vertex and index array buffer
	glGetError();	// for clean-up the error generated before
	nodeNum=mesh->GetNodeNumber();
	faceNum=mesh->GetFaceNumber();
	float* verTex=(float*)malloc(nodeNum*3*sizeof(float));
	memset(verTex,0,nodeNum*3*sizeof(float));
	memcpy(verTex,mesh->GetNodeArrayPtr(),nodeNum*3*sizeof(float));
	int* inDex=(int*)malloc(faceNum*3*sizeof(int));
	memset(inDex,0,faceNum*3*sizeof(int));

	unsigned int* meshptr = mesh->GetFaceTablePtr();
	for(int i=0; i < faceNum; i++)
	{	inDex[3*i] = meshptr[4*i]-1;	inDex[3*i+1] = meshptr[4*i+1]-1;	inDex[3*i+2] = meshptr[4*i+2]-1;
	}
	//memcpy(inDex,mesh->GetFaceTablePtr(),faceNum*3*sizeof(int));

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, nodeNum*3*sizeof(GLfloat), 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, nodeNum*3*sizeof(GLfloat), verTex);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &vboInd);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER_ARB, vboInd);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER_ARB, faceNum*3*sizeof(GL_UNSIGNED_INT), 0, GL_STATIC_DRAW);
	glBufferSubData(GL_ELEMENT_ARRAY_BUFFER_ARB, 0, faceNum*3*sizeof(GL_UNSIGNED_INT), inDex);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER_ARB, 0);

	if (glGetError()!=GL_NO_ERROR) printf("Error: buffer binding!\n\n");
	free(verTex);
	free(inDex);
	//-----------------------------------------------------------------------------------------
	GLint id0,id1,id2,id3,id4,id5,id6;	
	float centerPos[3];
	centerPos[0]=(boundingBox[0]+boundingBox[1])*0.5f;
	centerPos[1]=(boundingBox[2]+boundingBox[3])*0.5f;
	centerPos[2]=(boundingBox[4]+boundingBox[5])*0.5f;

	glUseProgramObjectARB(g_programObj);
	{

		id0 = glGetUniformLocationARB(g_programObj,"Unum");
		glUniform3iARB(id0,UnitNum[0],UnitNum[1],UnitNum[2]);
		id1 = glGetUniformLocationARB(g_programObj,"UOff");
		glUniform3fARB(id1,UnitOff[0],UnitOff[1],UnitOff[2]);
		id2 = glGetUniformLocationARB(g_programObj,"UWidth");
		glUniform3fARB(id2,UnitWidth[0],UnitWidth[1],UnitWidth[2]);
		id3 = glGetUniformLocationARB(g_programObj,"UFlip");
		glUniform3iARB(id3,UnitFlip[0],UnitFlip[1],UnitFlip[2]);
		id4 = glGetUniformLocationARB(g_programObj,"Cent");
		glUniform3fARB(id4,centerPos[0],centerPos[1],centerPos[2]);
		id5 = glGetUniformLocationARB(g_programObj,"bsingleCol");
		glUniform1iARB(id5,bsingleCol);
		id6 = glGetUniformLocationARB(g_programObj,"bsingleRow");
		glUniform1iARB(id6,bsingleRow);


		if (glGetError()!=GL_NO_ERROR) printf("Error: Unit Constant !\n\n");

		_decomposeLDNIByFBOPBO(solid, vbo, vboInd, UnitNum[0]*UnitNum[1]*UnitNum[2], faceNum*3);

	}
	glUseProgramObjectARB(0);

	//-----------------------------------------------------------------------------------------
	//	Step 6:  free the memory
	time=clock();
	//-----------------------------------------------------------------------------------------
	glDeleteBuffers(1, &vboInd);
	glDeleteBuffers(1, &vbo);
	glDeleteObjectARB( g_vertexShader);
	glDeleteObjectARB( g_GeometryShader);
	glDeleteObjectARB( g_FragShader);
	glDeleteObjectARB( g_programObj);
	//------------------------------------------------------------------------
	printf("\nMemory clean-up time is %ld (ms)\n",clock()-time);
	printf("--------------------------------------------------------------\n");
	printf("Total time for sampling is %ld (ms)\n\n",clock()-totalTime);

	return true;
}

void LDNIcudaOperation::_decomposeLDNIByFBOPBO(LDNIcudaSolid *solid, GLuint vbo, GLuint vboI, int instanceCount, int indexCount)
{
	unsigned int n_max,i,n;
	float gWidth,origin[3];
	unsigned int overall_n_max=0;
	long readbackTime=0, sortingTime=0, tempTime;

	cudaEvent_t     startClock, stopClock;
	CUDA_SAFE_CALL( cudaEventCreate( &startClock ) );
	CUDA_SAFE_CALL( cudaEventCreate( &stopClock ) );

	tempTime=clock();
	//------------------------------------------------------------------------
	//	Preparation
	int nRes=solid->GetResolution();		gWidth=solid->GetSampleWidth();
	float width=gWidth*(float)nRes;
	solid->GetOrigin(origin[0],origin[1],origin[2]);
	int arrsize=nRes*nRes;

	//------------------------------------------------------------------------
	//	Step 1: Setup the rendering environment
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_STENCIL_TEST);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glDisable(GL_POLYGON_OFFSET_FILL);
	glDisable(GL_POLYGON_OFFSET_LINE);
	glDisable(GL_BLEND);	
	glDisable(GL_POLYGON_SMOOTH);	// turn off anti-aliasing
	glDisable(GL_POINT_SMOOTH);
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_MAP_COLOR);	glDisable(GL_DITHER);
	glShadeModel(GL_FLAT);
	glDisable(GL_LIGHTING);   glDisable(GL_LIGHT0);
	glDisable(GL_LOGIC_OP);
	glDisable(GL_COLOR_MATERIAL);
	glDisable(GL_ALPHA_TEST);
	glGetError();	// for clean-up the error generated before
	//------------------------------------------------------------------------
	//	create the FBO objects and texture for rendering
	if (glewIsSupported("GL_EXT_framebuffer_object") == 0) printf("Warning: FBO is not supported!\n");
	if (glGetError()!=GL_NO_ERROR) printf("Error: before framebuffer generation!\n");
	//------------------------------------------------------------------------
	GLuint fbo;
	glGenFramebuffersEXT(1, &fbo);
	if (glGetError()!=GL_NO_ERROR) printf("Error: framebuffer generation!\n");
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);
	if (glGetError()!=GL_NO_ERROR) printf("Error: framebuffer binding!\n");
	//------------------------------------------------------------------------
	GLuint tex;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, nRes, nRes, 0, GL_RGBA, GL_FLOAT, 0);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, tex, 0);
	if (glGetError()!=GL_NO_ERROR) printf("Error: attaching texture to framebuffer generation!\n");
	cudaGraphicsResource *sampleTex_resource;
	CUDA_SAFE_CALL( cudaGraphicsGLRegisterImage(&sampleTex_resource, tex, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly) );
	//------------------------------------------------------------------------
	GLuint depth_and_stencil_rb;
	glGenRenderbuffersEXT(1, &depth_and_stencil_rb);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, depth_and_stencil_rb);
	glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_STENCIL_EXT, nRes, nRes);
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depth_and_stencil_rb);
	if (glGetError()!=GL_NO_ERROR) printf("Error: attaching renderbuffer of depth-buffer to framebuffer generation!\n");
	glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, depth_and_stencil_rb);
	if (glGetError()!=GL_NO_ERROR) printf("Error: attaching renderbuffer of stencil-buffer to framebuffer generation!\n");
	//------------------------------------------------------------------------
	GLuint indexPBO;
	glGenBuffers(1,&indexPBO);	//	generation of PBO for index array readback
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, indexPBO);
	glBufferData(GL_PIXEL_PACK_BUFFER_ARB, nRes*nRes*sizeof(unsigned char), NULL, GL_STREAM_READ_ARB);
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	CUDA_SAFE_CALL( cudaGLRegisterBufferObject(indexPBO) );
	//------------------------------------------------------------------------
	if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT)!=GL_FRAMEBUFFER_COMPLETE_EXT) 
		printf("Warning: the setting for rendering on FBO is not correct!\n");
	else
		printf("FBO has been created successfully!\n");
	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0,0,nRes,nRes);
	printf("Preparation time: %ld (ms)\n",clock()-tempTime);

	//------------------------------------------------------------------------
	//	Step 2: Rendering to get the Hermite samples
	for(short nAxis=0; nAxis<3; nAxis++) { 
		//---------------------------------------------------------------------------------------
		//	Rendering step 1: setting the viewing window
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		//---------------------------------------------------------------------------------------
		//	The eye is located at (0, 0, 0), the near clipping plane is at the z=0 plane
		//		the far clipping plane is at the z=(boundingBox[5]-boundingBox[4]) plane
		glOrtho(-width*0.5f,width*0.5f,-width*0.5f,width*0.5f,width*0.5f,-width*0.5f);
		//	Note that:	in "glOrtho(left,right,bottom,top,near,far);"
		//		(left,right,bottom,top) are located at the boundary of pixel instead of
		//		the center of pixels
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();

		//---------------------------------------------------------------------------------------
		//	Rendering step 2: determine the number of layers
		glClearColor( 1.0f, 1.0f, 1.0f, 1.0f );	
		glClearDepth(1.0);
		glClearStencil(0);	glColor3f(1,1,1);
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		glDepthFunc(GL_ALWAYS);
		glStencilFunc(GL_GREATER, 1, 0xff);
		glStencilOp(GL_INCR, GL_INCR, GL_INCR);
		glPushMatrix();
		switch(nAxis) {
			case 0:{glRotatef(-90,0,1,0);	glRotatef(-90,1,0,0); }break;
			case 1:{glRotatef(90,0,1,0);	glRotatef(90,0,0,1);  }break;
		}


		glEnableClientState( GL_VERTEX_ARRAY );		
		glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);	
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, vboI);
		glDrawElementsInstanced(GL_TRIANGLES,indexCount,GL_UNSIGNED_INT, 0 ,instanceCount);
		glDisableClientState( GL_VERTEX_ARRAY );



		glFlush();
		//--------------------------------------------------------------------------------------------------------
		//	reading stencil buffer into the device memory of CUDA
		tempTime=clock();
		glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, indexPBO);
		GLint OldPackAlignment;
		glGetIntegerv(GL_PACK_ALIGNMENT,&OldPackAlignment); 
		glPixelStorei(GL_PACK_ALIGNMENT,1);	// Important!!! Without this, the read-back could be abnormal.
		glReadPixels(0,0,nRes,nRes,GL_STENCIL_INDEX,GL_UNSIGNED_BYTE,0);
		glPixelStorei(GL_PACK_ALIGNMENT,OldPackAlignment);
		//--------------------------------------------------------------------------------------------------------
		unsigned char *devStencilBufferPtr;
		unsigned int *devResArrayPtr;
		unsigned int *devIndexArrayPtr=solid->GetIndexArrayPtr(nAxis);
		CUDA_SAFE_CALL( cudaGLMapBufferObject( (void **)&devStencilBufferPtr, indexPBO) );
		CUDA_SAFE_CALL( cudaMalloc( (void**)&devResArrayPtr, BLOCKS_PER_GRID*sizeof(unsigned int) ) );
		//--------------------------------------------------------------------------------------------------------
		//	building the indexArray on device
		krLDNISampling_CopyIndexAndFindMax<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(devStencilBufferPtr,
			devIndexArrayPtr,devResArrayPtr,arrsize);

		//--------------------------------------------------------------------------------------------------------
		//	read back the max number of layers -- "n_max"
		unsigned int* resArrayPtr;
		resArrayPtr=(unsigned int *)malloc(BLOCKS_PER_GRID*sizeof(unsigned int));
		CUDA_SAFE_CALL( cudaMemcpy( resArrayPtr, devResArrayPtr, BLOCKS_PER_GRID*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
		n_max=0;
		for(i=0;i<BLOCKS_PER_GRID;i++) n_max = MAX(n_max,resArrayPtr[i]);
		cudaFree(devResArrayPtr);		free(resArrayPtr);
		//--------------------------------------------------------------------------------------------------------
		//	read back the number of samples -- "sampleNum"
		unsigned int sampleNum=0;
		tempTime=clock()-tempTime;		//readbackTime+=tempTime;
		printf("Stencil buffer processing time: %ld (ms)\n",tempTime);

		long scanTime=clock();
		//	for debug purpose
		resArrayPtr=(unsigned int *)malloc((arrsize+1)*sizeof(unsigned int));
		CUDA_SAFE_CALL( cudaMemcpy( resArrayPtr, devIndexArrayPtr, (arrsize+1)*sizeof(unsigned int), cudaMemcpyDeviceToHost ) );
		sampleNum=0;
		for(int k=0;k<arrsize;k++) {sampleNum+=resArrayPtr[k];	resArrayPtr[k]=sampleNum;}	
		for(int k=arrsize;k>0;k--) {resArrayPtr[k]=resArrayPtr[k-1];}	
		resArrayPtr[0]=0;
		CUDA_SAFE_CALL( cudaMemcpy( devIndexArrayPtr, resArrayPtr, (arrsize+1)*sizeof(unsigned int), cudaMemcpyHostToDevice ) );
		free(resArrayPtr);
		scanTime=clock()-scanTime;	printf("Scanning time: %ld (ms)\n",scanTime);

		//--------------------------------------------------------------------------------------------------------
		CUDA_SAFE_CALL( cudaGLUnmapBufferObject( indexPBO ) );
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
		printf("n_max=%d   sampleNum=%d\n",n_max,sampleNum);
		if (n_max>overall_n_max) overall_n_max=n_max;
		if (sampleNum==0) continue;

		//---------------------------------------------------------------------------------------
		//	Rendering step 3: decomposing the Layered Depth Images (LDIs) and record its corresponding normals
		solid->MallocSampleMemory(nAxis,sampleNum);	
		float* devNxArrayPtr=solid->GetSampleNxArrayPtr(nAxis);
		float* devNyArrayPtr=solid->GetSampleNyArrayPtr(nAxis);
		float* devDepthArrayPtr=solid->GetSampleDepthArrayPtr(nAxis);
		tempTime=clock();
		for(n=1;n<=n_max;n++) {
			CUDA_SAFE_CALL( cudaGraphicsMapResources( 1, &sampleTex_resource, NULL ) );
			cudaArray *in_array;
			CUDA_SAFE_CALL( cudaGraphicsSubResourceGetMappedArray( &in_array, sampleTex_resource, 0, 0));
			CUDA_SAFE_CALL( cudaBindTextureToArray(tex2DFloat4In, in_array) );
			//--------------------------------------------------------------------------------------------------------
			//	fill the sampleArray on device
			krLDNISampling_CopySamples<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(devNxArrayPtr, devNyArrayPtr, 
				devDepthArrayPtr, n, arrsize, width, gWidth, nRes, devIndexArrayPtr);
			CUDA_SAFE_CALL( cudaGraphicsUnmapResources( 1, &sampleTex_resource, NULL ) );
			if (n==n_max) break;

			glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
			glStencilFunc(GL_GREATER, n+1, 0xff);
			glStencilOp(GL_KEEP, GL_INCR, GL_INCR);
			{
				glEnableClientState( GL_VERTEX_ARRAY );		
				glBindBufferARB(GL_ARRAY_BUFFER_ARB, vbo);	
				glVertexPointer(3, GL_FLOAT, 0, 0);
				glBindBufferARB(GL_ELEMENT_ARRAY_BUFFER_ARB, vboI);
				glDrawElementsInstanced(GL_TRIANGLES,indexCount,GL_UNSIGNED_INT, 0 ,instanceCount);
				glDisableClientState( GL_VERTEX_ARRAY );
			}
			glFlush();
		}
		tempTime=clock()-tempTime;		readbackTime+=tempTime;

		//------------------------------------------------------------------------
		//	Rendering step 4: sorting the samples
		CUDA_SAFE_CALL( cudaEventRecord( startClock, 0 ) );
		CUDA_SAFE_CALL( cudaEventSynchronize( startClock ) );
		krLDNISampling_SortSamples<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(devNxArrayPtr, devNyArrayPtr, 
			devDepthArrayPtr, arrsize, devIndexArrayPtr);
		CUDA_SAFE_CALL( cudaEventRecord( stopClock, 0 ) );
		CUDA_SAFE_CALL( cudaEventSynchronize( stopClock ) );
		float   elapsedTime;
		CUDA_SAFE_CALL( cudaEventElapsedTime( &elapsedTime,
			startClock, stopClock ) );
		//		printf( "Sorting time is:  %3.1f (ms)\n", elapsedTime );		
		sortingTime+=(long)elapsedTime;
	}

	//------------------------------------------------------------------------------------
	//	Step 3: Set the rendering parameters back
	//------------------------------------------------------------------------------------
	//	detach FBO
	glPopAttrib();
	//	release memory for PBO and cuda's map	
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	CUDA_SAFE_CALL( cudaGLUnregisterBufferObject( indexPBO ) );
	glDeleteBuffers(1, &indexPBO);
	CUDA_SAFE_CALL( cudaGraphicsUnregisterResource( sampleTex_resource) );
	//	release memory for the 2D texture
	glBindTexture(GL_TEXTURE_2D, 0);
	glDeleteTextures(1, &tex);
	//	release memory for the frame-buffer object
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glDeleteFramebuffersEXT(1, &fbo);
	//	release memory for the render-buffer object
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
	glDeleteRenderbuffersEXT(1, &depth_and_stencil_rb);
	//------------------------------------------------------------------------------------
	glEnable(GL_POLYGON_OFFSET_FILL);
	glEnable(GL_POLYGON_OFFSET_LINE);
	glEnable(GL_BLEND);	
	glEnable(GL_DITHER);
	glDisable(GL_STENCIL_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_MAP_COLOR);				
	glShadeModel(GL_SMOOTH);   
	glEnable(GL_LIGHTING);  glEnable(GL_LIGHT0);
	//	glEnable(GL_POLYGON_SMOOTH);// adding this will make the invalid display on the Thinkpad laptop	
	glEnable(GL_POINT_SMOOTH);
	//	glEnable(GL_LINE_SMOOTH);	// adding this will make the Compaq laptop's running fail
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

	printf("\nn_max=%ld \n",overall_n_max);
	printf("Texture Size: %f (MB)\n",(float)((float)overall_n_max*(float)nRes*(float)nRes*7.0f)/(1024.0f*1024.0f));
	printf("Readback time: %ld (ms)\nSorting time: %ld (ms)\n", 
		readbackTime, sortingTime);

	CUDA_SAFE_CALL( cudaEventDestroy( startClock ) );
	CUDA_SAFE_CALL( cudaEventDestroy( stopClock ) );
}
