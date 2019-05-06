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


#define _CRT_SECURE_NO_DEPRECATE


#include <string.h>


#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include "../common/GL/glew.h"
#include <omp.h>

#include "PMBody.h"
#include "LDNIcpuOperation.h"

#define MAX_NUM_OF_SAMPLES_ON_RAY	1024

#define MAX(a,b)					(((a)>(b))?(a):(b))
#define MIN(a,b)					(((a)<(b))?(a):(b))
#define GAUSSIAN_FUNC(xx,delta)		(exp(-(xx*xx)/(2.0*delta*delta)))

#define LOGIC_UNION(insideA, insideB)	(insideA || insideB)
#define LOGIC_INTER(insideA, insideB)	(insideA && insideB)
#define LOGIC_SUBTR(insideA, insideB)	(insideA && (!insideB))

extern bool _bExpandableWorkingSpace;


LDNIcpuOperation::LDNIcpuOperation()
{
}

LDNIcpuOperation::~LDNIcpuOperation()
{
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//	Functions for sampling
//

bool LDNIcpuOperation::BRepToLDNISampling(QuadTrglMesh *mesh, LDNIcpuSolid* &solid, float boundingBox[], int res)
{
	const bool bCube=true;
	float origin[3],gWidth;		long time=clock();
	int i,nodeNum;
	char fileadd[256];
	//---------------------------------------------------------------------------------
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

			ww=ww*0.55+ww/(float)(res-1)*2.0f;

			boundingBox[0]=xx-ww;	boundingBox[1]=xx+ww;
			boundingBox[2]=yy-ww;	boundingBox[3]=yy+ww;
			boundingBox[4]=zz-ww;	boundingBox[5]=zz+ww;
		}
	}
	//---------------------------------------------------------------------------------
	solid=new LDNIcpuSolid;
	solid->MallocMemory(res);
	gWidth=(boundingBox[1]-boundingBox[0])/(float)res;
	solid->SetSampleWidth(gWidth);
	origin[0]=boundingBox[0]+gWidth*0.5f;
	origin[1]=boundingBox[2]+gWidth*0.5f;
	origin[2]=boundingBox[4]+gWidth*0.5f;
	solid->SetOrigin(origin[0],origin[1],origin[2]);

	//---------------------------------------------------------------------------------
	//	For using OpenGL Shading Language to implement the sampling procedure
	bool bShader=_shaderInitialization();	
	if (!bShader) return false;
	//-----------------------------------------------------------------------------------------
	int dispListIndex;		GLhandleARB g_programObj, g_vertexShader, g_GeometryShader, g_FragShader;
	GLenum InPrimType=GL_POINTS, OutPrimType=GL_TRIANGLES;		int OutVertexNum=3;
	GLuint vertexTexture;	
	const char *VshaderString[1],*GshaderString[1],*FshaderString[1];
	GLint bVertCompiled = 0, bGeoCompiled = 0, bLinked = 0;
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
	glGetObjectParameterivARB( g_vertexShader, GL_OBJECT_COMPILE_STATUS_ARB, &bVertCompiled );
	if (bVertCompiled  == false) {
		glGetInfoLogARB(g_vertexShader, sizeof(str), NULL, str);
		printf("Warning: Vertex Shader Compile Error\n\n");	return false;
	}
	//-----------------------------------------------------------------------------
	memset(fileadd,0,256*sizeof(char));	
	strcat(fileadd,"sampleLDNIGeometryShader.geo");
	g_GeometryShader = glCreateShaderObjectARB( GL_GEOMETRY_SHADER_EXT );
	ShaderAssembly = _readShaderFile(fileadd );
	GshaderString[0] = (char*)ShaderAssembly;
	glShaderSourceARB( g_GeometryShader, 1, GshaderString, NULL );
	glCompileShaderARB( g_GeometryShader);
	delete ShaderAssembly;
	glGetObjectParameterivARB( g_GeometryShader, GL_OBJECT_COMPILE_STATUS_ARB, &bVertCompiled );
	if (bVertCompiled  == false) {
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
	glGetObjectParameterivARB( g_FragShader, GL_OBJECT_COMPILE_STATUS_ARB, &bVertCompiled );
	if (bVertCompiled  == false) {
		glGetInfoLogARB(g_FragShader, sizeof(str), NULL, str);
		printf("Warning: Vertex Shader Compile Error\n\n");	return false;
	}
	//-----------------------------------------------------------------------------
	glGetError();
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
	if (xF<64) xF=64;
	yF = (int)(nodeNum/(xF-1))+1; 
	printf("Texture Size: xF=%d yF=%d\n",xF,yF);
	float* verTex=(float*)malloc(xF*yF*3*sizeof(float));
	memset(verTex,0,xF*yF*3*sizeof(float));
	i=0;
	float* nodeTable=mesh->GetNodeArrayPtr();
	while(i+xF<nodeNum) {
		memcpy(&(verTex[(i+i/xF)*3]),&(nodeTable[i*3]),xF*3*sizeof(float));
		i += xF;
	}
	if (nodeNum-i>0) {
		memcpy(&(verTex[(i+i/xF)*3]),&(nodeTable[i*3]),(nodeNum-i)*3*sizeof(float));
	}
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
	int faceNum=mesh->GetFaceNumber();		int i1,i2,i3;
	dispListIndex = glGenLists(1);
	glNewList(dispListIndex, GL_COMPILE);
	glBegin(GL_POINTS);
	for(i=0;i<faceNum;i++) {
		mesh->GetFaceNodes(i+1,ver[0],ver[1],ver[2],ver[3]);
		i1=(int)(ver[0])-1;		i1 += i1/xF;
		i2=(int)(ver[1])-1;		i2 += i2/xF;
		i3=(int)(ver[2])-1;		i3 += i3/xF;
		glVertex3i(i1,i2,i3);
		if (mesh->IsQuadFace(i+1)) {glVertex3i((int)(ver[0])-1,(int)(ver[2])-1,(int)(ver[3])-1);}	// one more triangle
	}
	glEnd();
	glEndList();
	clock();
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

	return true;
}

void LDNIcpuOperation::_decomposeLDNIByFBOPBO(LDNIcpuSolid *solid, int displayListIndex)
{
	unsigned int n_max,i,n;
	float gWidth,origin[3];
	unsigned int overall_n_max=0;
	long readbackTime=0,sortingTime=0,tempTime;
	unsigned int sampleNum, arrsize;
	unsigned int *indexArray;		
	unsigned char *stencilArray;	GLfloat *rgbaLayerArray;

	//------------------------------------------------------------------------
	//	Preparation
	int nRes=solid->GetResolution();		gWidth=solid->GetSampleWidth();
	float width=gWidth*(float)nRes;
	solid->GetOrigin(origin[0],origin[1],origin[2]);
	arrsize=nRes*nRes;
	//------------------------------------------------------------------------
	indexArray=(unsigned int *)malloc(arrsize*sizeof(unsigned int));	
	stencilArray=(unsigned char*)malloc(arrsize*sizeof(unsigned char));
	rgbaLayerArray=(GLfloat *)malloc(arrsize*4*sizeof(GLfloat));

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
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F_ARB, nRes, nRes, 0, GL_RGB, GL_FLOAT, 0);
	glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, tex, 0);
	if (glGetError()!=GL_NO_ERROR) printf("Error: attaching texture to framebuffer generation!\n");
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
	GLuint pbo;
	glGenBuffers(1,&pbo);	//	generation of PBO
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, pbo);
	glBufferData(GL_PIXEL_PACK_BUFFER_ARB, nRes*nRes*3*sizeof(float), NULL, GL_STREAM_READ_ARB);
	glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
	//------------------------------------------------------------------------
	if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT)!=GL_FRAMEBUFFER_COMPLETE_EXT) 
		printf("Warning: the setting for rendering on FBO is not correct!\n");
	else
		printf("FBO has been created successfully!\n");
	glPushAttrib(GL_VIEWPORT_BIT);
	glViewport(0,0,nRes,nRes);

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
		//---------------------------------------------------------------------------------------
		tempTime=clock();
		glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
		GLint OldPackAlignment;
		glGetIntegerv(GL_PACK_ALIGNMENT,&OldPackAlignment); 
		glPixelStorei(GL_PACK_ALIGNMENT,1);	// Important!!! Without this, the read-back could be abnormal.
		glReadPixels(0,0,nRes,nRes,GL_STENCIL_INDEX,GL_UNSIGNED_BYTE,stencilArray);
		glPixelStorei(GL_PACK_ALIGNMENT,OldPackAlignment);
		//---------------------------------------------------------------------------------------
		n_max=0;	sampleNum=0;
		memset(indexArray,0,arrsize*sizeof(unsigned int));
		for(i=0;i<arrsize;i++) {
			if (stencilArray[i]==255) {stencilArray[i]=0;printf("*");}	// The result is already unstable.
			sampleNum+=stencilArray[i];
			indexArray[(i+1)%arrsize]=sampleNum;
			if (stencilArray[i]>n_max) n_max=stencilArray[i];
		}
		tempTime=clock()-tempTime;		readbackTime+=tempTime;
		printf("Stencil buffer processing time: %ld (ms)\n",tempTime);
		printf("n_max=%d   sampleNum=%d\n",n_max,sampleNum);
		indexArray[0]=0;  
		if (n_max>overall_n_max) overall_n_max=n_max;
		if (sampleNum==0) continue;
			
		//---------------------------------------------------------------------------------------
		//	Rendering step 3: decomposing the Layered Depth Images (LDIs) and record its corresponding normals
		solid->MallocSampleMemory(nAxis,sampleNum);	
		LDNIcpuSample *sampleArray=solid->GetSampleArrayPtr(nAxis);
		tempTime=clock();		n=1;
		//---------------------------------------------------------------------------------------
		//	PBO based read back
		glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, pbo);
		glReadPixels(0,0,nRes,nRes,GL_RGB,GL_FLOAT,0);
		void* mem = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);   
		memcpy(rgbaLayerArray,mem,nRes*nRes*3*sizeof(float));
		glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
		glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);
		//---------------------------------------------------------------------------------------
		glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		glStencilFunc(GL_GREATER, n+1, 0xff);
		glStencilOp(GL_KEEP, GL_INCR, GL_INCR);
		glCallList(displayListIndex);	glFlush();
		n++;
		while(n<=n_max) {
			//---------------------------------------------------------------------------------------
			//	PBO binding
			glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
			glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, pbo);
			glReadPixels(0,0,nRes,nRes,GL_RGB,GL_FLOAT,0);

			//---------------------------------------------------------------------------------------
			//	fill the information of previous layer (doing in this way will improve the efficiency)
			for(unsigned int index=0;index<arrsize;index++) {
				if (stencilArray[index]<n-1) continue;	// this detection is very important!!!	
				unsigned int arrindex=(indexArray[index]+n-2);	
				sampleArray[arrindex].depth=(float)((double)(fabs(rgbaLayerArray[index*3+2]))*width-0.5*gWidth);	
																		// As the origion is located at the center of a pixel,
																		//	so the depth is shifted with "0.5*gridWidth"
				sampleArray[arrindex].nx=rgbaLayerArray[index*3];		// x-component of normal	
				sampleArray[arrindex].ny=rgbaLayerArray[index*3+1];		// y-component of normal
				float nz=1.0-rgbaLayerArray[index*3]*rgbaLayerArray[index*3]-rgbaLayerArray[index*3+1]*rgbaLayerArray[index*3+1];
				if (nz<0.0) nz=0.0;		if (nz>1.0) nz=1.0;
				if (rgbaLayerArray[index*3+2]<0.0f) nz=-sqrt(nz); else nz=sqrt(nz);
				sampleArray[arrindex].nz=nz;							// z-component of normal
			}

			//---------------------------------------------------------------------------------------
			//	PBO based read back
			void* mem = glMapBuffer(GL_PIXEL_PACK_BUFFER_ARB, GL_READ_ONLY);   
			memcpy(rgbaLayerArray,mem,arrsize*3*sizeof(float));
			glUnmapBuffer(GL_PIXEL_PACK_BUFFER_ARB);
			glBindBuffer(GL_PIXEL_PACK_BUFFER_ARB, 0);

			if (n==n_max) {
				for(unsigned int index=0;index<arrsize;index++) {
					if (stencilArray[index]<n) continue;	// this detection is very important!!!	
					unsigned int arrindex=(indexArray[index]+n-1);
					sampleArray[arrindex].depth=(float)((double)(fabs(rgbaLayerArray[index*3+2]))*width-0.5*gWidth);	
																			// As the origion is located at the center of a pixel,
																			//	so the depth is shifted with "0.5*gridWidth"
					sampleArray[arrindex].nx=rgbaLayerArray[index*3];		// x-component of normal	
					sampleArray[arrindex].ny=rgbaLayerArray[index*3+1];		// y-component of normal
					float nz=1.0-rgbaLayerArray[index*3]*rgbaLayerArray[index*3]-rgbaLayerArray[index*3+1]*rgbaLayerArray[index*3+1];
					if (nz<0.0) nz=0.0;		if (nz>1.0) nz=1.0;
					if (rgbaLayerArray[index*3+2]<0.0f) nz=-sqrt(nz); else nz=sqrt(nz);
					sampleArray[arrindex].nz=nz;							// z-component of normal
				}
				break;
			}

			glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
			glStencilFunc(GL_GREATER, n+1, 0xff);
			glStencilOp(GL_KEEP, GL_INCR, GL_INCR);
			glCallList(displayListIndex);	glFlush();

			n++;
		}
		tempTime=clock()-tempTime;		readbackTime+=tempTime;
		glPopMatrix();

		//------------------------------------------------------------------------
		//	Rendering step 4: sorting the samples
		tempTime=clock();	LDNIcpuRay* rays=solid->GetRayArrayPtr(nAxis);
		for(unsigned int index=0;index<arrsize;index++) {
			unsigned int endIndex=indexArray[(index+1)%arrsize];	
			if ((index+1)==(int)arrsize) endIndex=sampleNum;
			rays[index].sampleIndex=endIndex;
			_sortingSamplesOnRay(sampleArray,indexArray[index],endIndex-indexArray[index]);
		}
		tempTime=clock()-tempTime;		sortingTime+=tempTime;
	}
	free(indexArray);	free(stencilArray);		free(rgbaLayerArray);

	//------------------------------------------------------------------------------------
	//	Step 3: Set the rendering parameters back
	//------------------------------------------------------------------------------------
	//	detach FBO
	glPopAttrib();
	glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
	glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glDeleteBuffers(1, &pbo);
	glDeleteFramebuffersEXT(1, &fbo);
	glDeleteRenderbuffersEXT(1, &depth_and_stencil_rb);
	glDeleteTextures(1, &tex);
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
}

void LDNIcpuOperation::_sortingSamplesOnRay(LDNIcpuSample *sampleArray, unsigned int stIndex, unsigned int size)
{
	unsigned int i,j;	LDNIcpuSample tempSample;

	for(i=0;i<size;i++) {
		for(j=i+1;j<size;j++) {
			if (sampleArray[stIndex+i].depth>sampleArray[stIndex+j].depth) {
				tempSample=sampleArray[stIndex+i];
				sampleArray[stIndex+i]=sampleArray[stIndex+j];
				sampleArray[stIndex+j]=tempSample;
			}
		}
	}
}

bool LDNIcpuOperation::_shaderInitialization()
{
	if(glewInit() != GLEW_OK) {
		printf("glewInit failed. Exiting...\n");
		return false;
	}
	if (glewIsSupported("GL_VERSION_2_0")) {
		printf("\nReady for OpenGL 2.0\n");
		printf("-------------------------------------------------\n");
		printf("GLSL will be used to speed up sampling\n");
	}
	else {
		printf("OpenGL 2.0 not supported\n");
		return false;
	}
	return true;
}

unsigned char* LDNIcpuOperation::_readShaderFile( const char *fileName )
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

void LDNIcpuOperation::_texCalProduct(int in, int &outx, int &outy)
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


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//	Functions for Boolean operations
//

bool LDNIcpuOperation::BooleanOperation(LDNIcpuSolid* &inputSolid, QuadTrglMesh *meshB, short nOperationType)
{
	float boundingBox[6];	LDNIcpuSolid *solidB;

	//-----------------------------------------------------------------------------------
	//	Step 1: converting the mesh surface into a LDNI solid
	
	if ( _bExpandableWorkingSpace ) {
		
		meshB->CompBoundingBox(boundingBox);
		inputSolid->ExpansionByNewBoundingBox(boundingBox);
	}
	int res=inputSolid->GetResolution();
	
	if (nOperationType!=3) {
		LDNIcpuOperation::BRepToLDNISampling( meshB, solidB, boundingBox, res );
		
	}
	else {
		solidB=inputSolid;	inputSolid=0;
		LDNIcpuOperation::BRepToLDNISampling( meshB, inputSolid, boundingBox, res );
		nOperationType=2;
	}
	
	//-----------------------------------------------------------------------------------
	//	Step 2: repair and truncate the sampled LDNI solid into the current working space
	if ( !(_bExpandableWorkingSpace) ) {
		//repair solidB
	}

	//-----------------------------------------------------------------------------------
	//	Step 3: computing the Boolean operation results on LDNIs
	_booleanOperation(inputSolid, solidB, nOperationType);

	//-----------------------------------------------------------------------------------
	//	Step 4: free the memory
	delete solidB;

	return true;
}

bool LDNIcpuOperation::BooleanOperation(QuadTrglMesh *meshA, QuadTrglMesh *meshB, int res, short nOperationType, 
										LDNIcpuSolid* &outputSolid)
{
	float boundingBox[6];	LDNIcpuSolid *solidB;

	//-----------------------------------------------------------------------------------
	//	Step 1: converting mesh surfaces into LDNIs
	_compBoundingCube(meshA, meshB, boundingBox, res);
	if (nOperationType!=3) {
		LDNIcpuOperation::BRepToLDNISampling(meshA, outputSolid, boundingBox, res);
		LDNIcpuOperation::BRepToLDNISampling(meshB, solidB, boundingBox, res);
	}
	else {
		LDNIcpuOperation::BRepToLDNISampling(meshB, outputSolid, boundingBox, res);
		LDNIcpuOperation::BRepToLDNISampling(meshA, solidB, boundingBox, res);
		nOperationType=2;
	}

	//-----------------------------------------------------------------------------------
	//	Step 2: computing the Boolean operation results on LDNIs
	_booleanOperation(outputSolid, solidB, nOperationType);

	//-----------------------------------------------------------------------------------
	//	Step 3: free the memory
	delete solidB;

	return true;
}

void LDNIcpuOperation::_booleanOperation(LDNIcpuSolid* outputSolid, LDNIcpuSolid* solidB, short nOperationType)
{
	int i,j,index,arrsize,stA,numA,stRes,numRes,stB;

	long time=clock();
	int res=outputSolid->GetResolution();
	arrsize=res*res;
	int *resArrayIndex=(int *)malloc((arrsize+1)*sizeof(int));		
	for(short nAxis=0;nAxis<3;nAxis++) {
		//---------------------------------------------------------------------------------------------
		//	Sub-step 1: intialization
		memset(resArrayIndex,0,(res*res+1)*sizeof(int));
		LDNIcpuRay *rayA=outputSolid->GetRayArrayPtr(nAxis);
		LDNIcpuRay *rayB=solidB->GetRayArrayPtr(nAxis);
		LDNIcpuSample *sampleArrA=outputSolid->GetSampleArrayPtr(nAxis);
		LDNIcpuSample *sampleArrB=solidB->GetSampleArrayPtr(nAxis);

		//---------------------------------------------------------------------------------------------
		//	Sub-step 2: computing the result of boolean operation ray by ray
		for(j=0;j<res;j++) {	// row
			for(i=0;i<res;i++) {	// column
				index=i+j*res; 
				_booleanOnRay(rayA, sampleArrA, rayB, sampleArrB, index, resArrayIndex, nOperationType);
			}
		}

		//---------------------------------------------------------------------------------------------
		//	Sub-step 3: compaction of index array
		int sampleNum=0;
		for(int k=0;k<arrsize;k++) {sampleNum+=resArrayIndex[k];	resArrayIndex[k]=sampleNum;}	
		for(int k=arrsize;k>0;k--) {resArrayIndex[k]=resArrayIndex[k-1];}	
		resArrayIndex[0]=0;

		//---------------------------------------------------------------------------------------------
		//	Sub-step 4: collecting the resultant samples into the sampleArray of solidTileA				
		LDNIcpuSample *resSampleArr=(LDNIcpuSample *)malloc(sampleNum*sizeof(LDNIcpuSample));
		for(index=0;index<arrsize;index++) {
			if (index==0) stA=0; else	stA=rayA[index-1].sampleIndex;	
			numA=rayA[index].sampleIndex-stA;
			stRes=resArrayIndex[index];		numRes=resArrayIndex[index+1]-stRes;
			if (numRes==0) continue;
			if (numA==0) {
				if (index==0) stB=0; else stB=rayB[index-1].sampleIndex;
				for(int k=0;k<numRes;k++) {resSampleArr[stRes+k]=sampleArrB[stB+k];}
			}
			else if (numA<numRes) {
				for(int k=0;k<numA;k++) {resSampleArr[stRes+k]=sampleArrA[stA+k];}
				if (index==0) stB=0; else stB=rayB[index-1].sampleIndex;
				for(int k=numA;k<numRes;k++) {resSampleArr[stRes+k]=sampleArrB[stB+(k-numA)];}
			}
			else {
				for(int k=0;k<numRes;k++) {resSampleArr[stRes+k]=sampleArrA[stA+k];}
			}
		}
		outputSolid->SetSampleArrayPtr(nAxis,resSampleArr);	free(sampleArrA);
		//---------------------------------------------------------------------------------------------
		for(index=0;index<arrsize;index++) rayA[index].sampleIndex=resArrayIndex[index+1];
		//printf("num=%d\n",(int)(rayA[arrsize-1].sampleIndex));
	}
	free(resArrayIndex);
	printf("Boolean Operation Time (ms): %ld\n",clock()-time);
}

void LDNIcpuOperation::_booleanOnRay(LDNIcpuRay *rayA, LDNIcpuSample *sampleArrA, 
									 LDNIcpuRay *rayB, LDNIcpuSample *sampleArrB, 
									 int index, int *resArrayIndex, short nOperationType)
{
	int stA,stB,numA,numB,numRes,aIndex,bIndex;
	LDNIcpuSample resSampleArr[MAX_NUM_OF_SAMPLES_ON_RAY];
	LDNIcpuSample lastSample;					bool last_op,op;	
	bool insideA,insideB;

	if (index==0) {stA=stB=0;} else {stA=rayA[index-1].sampleIndex;	stB=rayB[index-1].sampleIndex;}
	numA=rayA[index].sampleIndex-stA;		numB=rayB[index].sampleIndex-stB;

	last_op=false;	insideA=insideB=false;	numRes=0;	

	if (numA>0 && numB>0) {
		aIndex=bIndex=0;	
		while( (aIndex<numA) || (bIndex<numB) ) {	// scaning the samples on solidA and solidB together
			if ((bIndex==numB) || ((aIndex<numA) && (sampleArrA[aIndex+stA].depth<sampleArrB[bIndex+stB].depth))) {
				lastSample=sampleArrA[aIndex+stA];
				insideA=!insideA;		aIndex++;
			}
			else {
				lastSample=sampleArrB[bIndex+stB];
				if (nOperationType==2) {lastSample.nx=-(lastSample.nx);	lastSample.ny=-(lastSample.ny);	lastSample.nz=-(lastSample.nz);}
				insideB=!insideB;	bIndex++;
			}

			switch(nOperationType) {
			case 0:{op=LOGIC_UNION(insideA,insideB); }break;
			case 1:{op=LOGIC_INTER(insideA,insideB); }break;
			case 2:{op=LOGIC_SUBTR(insideA,insideB); }break;
			}

			if (op!=last_op) {
				if (numRes>0 && fabs(lastSample.depth-resSampleArr[numRes-1].depth)<1.0e-5) 
					{numRes--;}
				else 
					{resSampleArr[numRes]=lastSample;	numRes++;}
				last_op=op;
			}
		}
	}
	else if ((numA==0) && (numB>0)) {	// scaning the samples on solidB
		for(int i=0;i<numB;i++) {
			lastSample=sampleArrB[stB+i];
			if (nOperationType==2) {lastSample.nx=-(lastSample.nx);	lastSample.ny=-(lastSample.ny);	lastSample.nz=-(lastSample.nz);}
			insideB=!insideB;
			switch(nOperationType) {
			case 0:{op=LOGIC_UNION(insideA,insideB); }break;
			case 1:{op=LOGIC_INTER(insideA,insideB); }break;
			case 2:{op=LOGIC_SUBTR(insideA,insideB); }break;
			}
			if (op!=last_op) {resSampleArr[numRes]=lastSample;	numRes++;	last_op=op;}
		}
	}
	else if ((numA>0) && (numB==0)) {	// scaning the samples on solidA
		for(int i=0;i<numA;i++) {
			lastSample=sampleArrA[stA+i];		insideA=!insideA;

			switch(nOperationType) {
			case 0:{op=LOGIC_UNION(insideA,insideB); }break;
			case 1:{op=LOGIC_INTER(insideA,insideB); }break;
			case 2:{op=LOGIC_SUBTR(insideA,insideB); }break;
			}
			if (op!=last_op) {resSampleArr[numRes]=lastSample;	numRes++;	last_op=op;}
		}
	}

	//-------------------------------------------------------------------------------------------------------
	//	Copy the resultant samples into solidA and solidB
	if (numA==0) {
		for(int i=0;i<numRes;i++) sampleArrB[stB+i]=resSampleArr[i];
	}
	else if (numRes>numA) {
		for(int i=0;i<numA;i++) sampleArrA[stA+i]=resSampleArr[i];
		for(int i=numA;i<numRes;i++) sampleArrB[stB+(i-numA)]=resSampleArr[i];
	}
	else {
		for(int i=0;i<numRes;i++) sampleArrA[stA+i]=resSampleArr[i];
	}

	resArrayIndex[index]=numRes;
}
	
void LDNIcpuOperation::_compBoundingCube(QuadTrglMesh *meshA, QuadTrglMesh *meshB, float boundingBox[], int res)
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



/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//	Functions for model repair
//




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//	Functions for other solid modeling operations
//

void LDNIcpuOperation::ParallelProcessingNormalVector(LDNIcpuSolid *solid, unsigned int nSupportSize, float normalPara) 
{
	unsigned int *indexArray[3];		float *depthArray[3],*nxArray[3],*nyArray[3];
	int res;
	float ww,origin[3],xx,yy,dbase,radiuSQR;

	//---------------------------------------------------------------------------------------------------------
	//	preparation
	res=solid->GetResolution();		ww=solid->GetSampleWidth();
	solid->GetOrigin(origin[0],origin[1],origin[2]);
	_copyCPUSolidToCUDAEmulArray(solid,indexArray[0],indexArray[1],indexArray[2],nxArray[0],nxArray[1],nxArray[2],
			nyArray[0],nyArray[1],nyArray[2],depthArray[0],depthArray[1],depthArray[2]);
	omp_set_dynamic(8);
	omp_set_num_threads(8);

	//---------------------------------------------------------------------------------------------------------
	//	do the computation
	float deltaC=((float)nSupportSize)*0.5*ww;	// The parameter to control the points to be considered
	float deltaG=normalPara;					// The parameter to control the band width of normal filter
	radiuSQR=ww*(float)nSupportSize;	radiuSQR=radiuSQR*radiuSQR;
	int arrsize=res*res;
	for(short nAxis=0;nAxis<3;nAxis++) {
		//
#pragma omp parallel for schedule(dynamic) 
		for(int index=0;index<arrsize;index++) {
			unsigned int i=((unsigned int)index)%res;	
			unsigned int j=((unsigned int)index)/res;
			for(unsigned int k=indexArray[nAxis][index];k<indexArray[nAxis][index+1];k++) {
				float centerPos[3],centerNv[3];
				float pp[3],normal[3];
				float hh[3],div,tt,ff,gg,dd,depth,lower,upper;
				int iCenter[3],ii,jj,kk,index2;
				int nSearchSize=nSupportSize+1;

				//---------------------------------------------------------------------------------------
				//	Step 1: Get the center point and its normal vector
				centerPos[nAxis]=origin[nAxis]+fabs(depthArray[nAxis][k]);
				centerPos[(nAxis+1)%3]=origin[(nAxis+1)%3]+ww*(float)i;
				centerPos[(nAxis+2)%3]=origin[(nAxis+2)%3]+ww*(float)j;
				centerNv[0]=nxArray[nAxis][k];	centerNv[1]=nyArray[nAxis][k];
				centerNv[2]=1.0f-centerNv[0]*centerNv[0]-centerNv[1]*centerNv[1];
				if (centerNv[2]<0.0) centerNv[2]=0.0;	if (centerNv[2]>1.0) centerNv[2]=1.0;
				centerNv[2]=sqrt(centerNv[2]);
				if (depthArray[nAxis][k]<0.0f) centerNv[2]=-centerNv[2];

				//---------------------------------------------------------------------------------------
				//	Step 2: Compute the filtered normal vector
				hh[0]=centerNv[0];	hh[1]=centerNv[1];	hh[2]=centerNv[2];	div=1.0f;
				iCenter[0]=(int)((centerPos[0]-origin[0])/ww);	
				iCenter[1]=(int)((centerPos[1]-origin[1])/ww);	
				iCenter[2]=(int)((centerPos[2]-origin[2])/ww);
				for(short nDir=0;nDir<3;nDir++) {
					lower=centerPos[nDir]-origin[nDir]-ww*(float)nSearchSize;
					upper=centerPos[nDir]-origin[nDir]+ww*(float)nSearchSize;
					for(ii=iCenter[(nDir+1)%3]-nSearchSize;ii<=iCenter[(nDir+1)%3]+nSearchSize;ii++) {
						if (ii<0 || ii>=res) continue;
						xx=origin[(nDir+1)%3]+ww*(float)ii;
						for(jj=iCenter[(nDir+2)%3]-nSearchSize;jj<=iCenter[(nDir+2)%3]+nSearchSize;jj++) {
							if (jj<0 || jj>=res) continue;
							yy=origin[(nDir+2)%3]+ww*(float)jj;
							dbase=(xx-centerPos[(nDir+1)%3])*(xx-centerPos[(nDir+1)%3])+(yy-centerPos[(nDir+2)%3])*(yy-centerPos[(nDir+2)%3]);
							if (dbase>radiuSQR) continue;

							index2=jj*res+ii;
							for(kk=(int)(indexArray[nDir][index2]);kk<(int)(indexArray[nDir][index2+1]);kk++) {
								depth=fabs(depthArray[nDir][kk]);
								if (depth<lower) continue;
								if (depth>upper) break;

								//---------------------------------------------------------------------------------------
								//	for(each neighbor with pp[] and normal[])	// the result is stored in hh[]
								pp[nDir]=origin[nDir]+fabs(depthArray[nDir][kk]);
								pp[(nDir+1)%3]=origin[(nDir+1)%3]+ww*(float)ii;
								pp[(nDir+2)%3]=origin[(nDir+2)%3]+ww*(float)jj;
								normal[0]=nxArray[nDir][kk];	normal[1]=nyArray[nDir][kk];
								normal[2]=1.0f-normal[0]*normal[0]-normal[1]*normal[1];
								if (normal[2]<0.0) normal[2]=0.0;	if (normal[2]>1.0) normal[2]=1.0;
								normal[2]=sqrt(normal[2]);
								if (depthArray[nDir][kk]<0.0f) normal[2]=-normal[2];
								//---------------------------------------------------------------------------------------
								dd=dbase+(pp[nDir]-centerPos[nDir])*(pp[nDir]-centerPos[nDir]);
								if (dd<=radiuSQR) {
									tt=centerNv[0]*(centerNv[0]-normal[0])
										+centerNv[1]*(centerNv[1]-normal[1])+centerNv[2]*(centerNv[2]-normal[2]);
									dd=sqrt(dd);
									ff=GAUSSIAN_FUNC(dd,deltaC);	
									gg=GAUSSIAN_FUNC(tt,deltaG);

									hh[0]+=ff*gg*normal[0];
									hh[1]+=ff*gg*normal[1];
									hh[2]+=ff*gg*normal[2];
									div+=ff*gg;
								}
							}
						}
					}
				}
				if (fabs(div)>1.0e-5) {
					hh[0]=hh[0]/div; hh[1]=hh[1]/div; hh[2]=hh[2]/div;
				} else {
					hh[0]=centerNv[0];	hh[1]=centerNv[1];	hh[2]=centerNv[2];
				}
				dd=sqrt(hh[0]*hh[0]+hh[1]*hh[1]+hh[2]*hh[2]);
				if (dd<1.0e-5) {
					hh[0]=centerNv[0];	hh[1]=centerNv[1];	hh[2]=centerNv[2];
				} else {
					hh[0]=hh[0]/dd; hh[1]=hh[1]/dd; hh[2]=hh[2]/dd;
				}

				//---------------------------------------------------------------------------------------
				//	Step 3: Update the normal vector in the LDNIcpuSolid
				LDNIcpuSample *cpuSampleArray=solid->GetSampleArrayPtr(nAxis);
				cpuSampleArray[k].nx=hh[0];	
				cpuSampleArray[k].ny=hh[1];	
				cpuSampleArray[k].nz=hh[2];
			}
		}
	}

	//---------------------------------------------------------------------------------------------------------
	//	free the memory
	free(indexArray[0]);	
	free(indexArray[1]);	
	free(indexArray[2]);
	free(nxArray[0]);		
	free(nxArray[1]);		
	free(nxArray[2]);
	free(nyArray[0]);		
	free(nyArray[1]);		
	free(nyArray[2]);
	free(depthArray[0]);	
	free(depthArray[1]);	
	free(depthArray[2]);
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
//	Functions for contouring
//

void LDNIcpuOperation::_copyCPUSolidToCUDAEmulArray(LDNIcpuSolid *cpuSolid, 
													unsigned int* &xIndexArray, unsigned int* &yIndexArray, unsigned int* &zIndexArray,
													float* &xNxArray, float* &yNxArray, float* &zNxArray, 
													float* &xNyArray, float* &yNyArray, float* &zNyArray, 
													float* &xDepthArray, float* &yDepthArray, float* &zDepthArray )
{
	int i,num,res;	short nAxis;
	LDNIcpuRay *rays;	LDNIcpuSample *sampleArray;

	res=cpuSolid->GetResolution();

	//-----------------------------------------------------------------------------------------
	//	copy the index arrays
	unsigned int *indexArray;
	num=res*res;
	for(nAxis=0;nAxis<3;nAxis++) {
		rays=cpuSolid->GetRayArrayPtr(nAxis);
		indexArray=(unsigned int *)malloc((num+1)*sizeof(unsigned int));
		indexArray[0]=0;
		for(i=0;i<num;i++) indexArray[i+1]=rays[i].sampleIndex;
		switch(nAxis) {
			case 0:xIndexArray=indexArray; break;
			case 1:yIndexArray=indexArray; break;
			case 2:zIndexArray=indexArray; break;
		}
	}

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

		switch(nAxis) {
		case 0:{xNxArray=sampleNxArray;		xNyArray=sampleNyArray;		xDepthArray=sampleDepthArray;
			   }break;
		case 1:{yNxArray=sampleNxArray;		yNyArray=sampleNyArray;		yDepthArray=sampleDepthArray;
			   }break;
		case 2:{zNxArray=sampleNxArray;		zNyArray=sampleNyArray;		zDepthArray=sampleDepthArray;
			   }break;
		}
	}
}
