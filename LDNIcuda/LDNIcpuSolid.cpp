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
#include <io.h>
#include <time.h>
#include <math.h>

#include "../common/GL/glew.h"


#include "cuda.h"
#include "cutil.h"
#include "cuda_runtime.h"
#include "cuda_gl_interop.h"

#include "LDNIcpuSolid.h"
#include "LDNIcudaSolid.h"

//#ifdef _DEBUG	// for detecting memory leak, using this - you need to use MFC DLL setting in compiling
//#include <afx.h>         // MFC core and standard components
//#define new DEBUG_NEW
//#endif

//#define MAX(a,b)		((a>b)?a:b)
//#define MAX3(a,b,c)		(MAX(MAX(a,b),MAX(a,c)))

extern GLK _pGLK;

LDNIcpuSolid::LDNIcpuSolid()
{
	m_res=0;
}

LDNIcpuSolid::~LDNIcpuSolid()
{
	FreeMemory();
}

void LDNIcpuSolid::MallocMemory(int res)
{
	int num;

	FreeMemory();
	m_res=res;	num=res*res;
	m_Rays[0]=(LDNIcpuRay*)(malloc(num*sizeof(LDNIcpuRay)));
	m_Rays[1]=(LDNIcpuRay*)(malloc(num*sizeof(LDNIcpuRay)));
	m_Rays[2]=(LDNIcpuRay*)(malloc(num*sizeof(LDNIcpuRay)));
	memset(m_Rays[0],0,num*sizeof(LDNIcpuRay));	memset(m_Rays[1],0,num*sizeof(LDNIcpuRay));	memset(m_Rays[2],0,num*sizeof(LDNIcpuRay));
}

void LDNIcpuSolid::FreeMemory()
{
	if (m_res==0) return;

	int num=m_res*m_res;
	if (m_Rays[0][num-1].sampleIndex!=0) free(m_SampleArray[0]);
	if (m_Rays[1][num-1].sampleIndex!=0) free(m_SampleArray[1]);
	if (m_Rays[2][num-1].sampleIndex!=0) free(m_SampleArray[2]);
	free(m_Rays[0]);	free(m_Rays[1]);	free(m_Rays[2]);
	m_res=0;
}

void LDNIcpuSolid::MallocSampleMemory(short nDir, int sampleNum)
{
	m_SampleArray[nDir]=(LDNIcpuSample*)malloc(sampleNum*sizeof(LDNIcpuSample));
}

int LDNIcpuSolid::GetSampleNumber()
{
	int num=0;

	num=m_Rays[0][m_res*m_res-1].sampleIndex
		+m_Rays[1][m_res*m_res-1].sampleIndex
		+m_Rays[2][m_res*m_res-1].sampleIndex;

	return num;
}

void LDNIcpuSolid::ExpansionByNewBoundingBox(float boundingBox[])
{
	unsigned int sd[3],ed[3],total;		
	int i,j,index,newIndex;
	float wx,wy,wz;		

	m_origin[0]=m_origin[0]-m_sampleWidth*0.5f;
	m_origin[1]=m_origin[1]-m_sampleWidth*0.5f;
	m_origin[2]=m_origin[2]-m_sampleWidth*0.5f;

	//------------------------------------------------------------------------------
	//	Step 1: determine the number of expansion
	boundingBox[0]=boundingBox[0]-m_sampleWidth*2.0;	
	boundingBox[2]=boundingBox[2]-m_sampleWidth*2.0;	
	boundingBox[4]=boundingBox[4]-m_sampleWidth*2.0;	
	boundingBox[1]=boundingBox[1]+m_sampleWidth*2.0;	
	boundingBox[3]=boundingBox[3]+m_sampleWidth*2.0;	
	boundingBox[5]=boundingBox[5]+m_sampleWidth*2.0;	
	//------------------------------------------------------------------------------
	sd[0]=sd[1]=sd[2]=0;
	if (boundingBox[0]<m_origin[0]) sd[0]=(unsigned int)((m_origin[0]-boundingBox[0])/m_sampleWidth)+1;
	if (boundingBox[2]<m_origin[1]) sd[1]=(unsigned int)((m_origin[1]-boundingBox[2])/m_sampleWidth)+1;
	if (boundingBox[4]<m_origin[2]) sd[2]=(unsigned int)((m_origin[2]-boundingBox[4])/m_sampleWidth)+1;
	//------------------------------------------------------------------------------
	wx=m_origin[0]+m_sampleWidth*(float)(m_res);
	wy=m_origin[1]+m_sampleWidth*(float)(m_res);
	wz=m_origin[2]+m_sampleWidth*(float)(m_res);
	ed[0]=ed[1]=ed[2]=0;
	if (boundingBox[1]>wx) ed[0]=(int)((boundingBox[1]-wx)/m_sampleWidth+0.5);
	if (boundingBox[3]>wy) ed[1]=(int)((boundingBox[3]-wy)/m_sampleWidth+0.5);
	if (boundingBox[5]>wz) ed[2]=(int)((boundingBox[5]-wz)/m_sampleWidth+0.5);
	//------------------------------------------------------------------------------
	total=sd[0]+ed[0];
	if ((sd[1]+ed[1])>total) total=sd[1]+ed[1];
	if ((sd[2]+ed[2])>total) total=sd[2]+ed[2];
	ed[0]=total-sd[0];	ed[1]=total-sd[1];	ed[2]=total-sd[2];

	//------------------------------------------------------------------------------
	//	Step 2: create new arrays of LDNISolidNode
	for(short nAxis=0; nAxis<3; nAxis++) {
		int num=(m_res+total)*(m_res+total);
		LDNIcpuRay *newRays=(LDNIcpuRay*)(malloc(num*sizeof(LDNIcpuRay)));
		memset(newRays,0,num*sizeof(LDNIcpuRay));

		for(j=0;j<m_res;j++) {
			for(i=0;i<m_res;i++) {
				index=j*m_res+i;
				newIndex=(j+sd[(nAxis+2)%3])*(m_res+total)+(i+sd[(nAxis+1)%3]);
				newRays[newIndex].rayflag=m_Rays[nAxis][index].rayflag;
				//------------------------------------------------------------------
				//	record the number of samples on each ray
				if (index>0)
					newRays[newIndex].sampleIndex=m_Rays[nAxis][index].sampleIndex-m_Rays[nAxis][index-1].sampleIndex;
				else
					newRays[newIndex].sampleIndex=m_Rays[nAxis][index].sampleIndex;	// actually "m_Rays[nAxis][index].sampleIndex - 0"
			}
		}

		//------------------------------------------------------------------
		//	scan the index array
		int sampleNum=0;
		for(i=0;i<num;i++) {
			sampleNum+=newRays[i].sampleIndex;
			newRays[i].sampleIndex=sampleNum;
		}

		//------------------------------------------------------------------
		free(m_Rays[nAxis]);	m_Rays[nAxis]=newRays;
	}

	//------------------------------------------------------------------------------
	//	Step 3: update the depth-values of samples when necessary
	m_origin[0]=m_origin[0]-m_sampleWidth*(float)(sd[0])+m_sampleWidth*0.5;
	m_origin[1]=m_origin[1]-m_sampleWidth*(float)(sd[1])+m_sampleWidth*0.5;
	m_origin[2]=m_origin[2]-m_sampleWidth*(float)(sd[2])+m_sampleWidth*0.5;
	m_res+=total;
	for(short nAxis=0; nAxis<3; nAxis++) {
		if (sd[nAxis]==0) continue;
		float updateDepth=m_sampleWidth*(float)sd[nAxis];
		int sampleNum=m_Rays[nAxis][m_res*m_res-1].sampleIndex;
		for(i=0;i<sampleNum;i++) m_SampleArray[nAxis][i].depth+=updateDepth;
	}

	//------------------------------------------------------------------------------
	//	Step 4: update the boundingBox[] for the sampling of mesh surface bounded by it
	boundingBox[0]=m_origin[0]-m_sampleWidth*0.5;	
	boundingBox[2]=m_origin[1]-m_sampleWidth*0.5;	
	boundingBox[4]=m_origin[2]-m_sampleWidth*0.5;	
	boundingBox[1]=boundingBox[0]+m_sampleWidth*((float)m_res);
	boundingBox[3]=boundingBox[2]+m_sampleWidth*((float)m_res);
	boundingBox[5]=boundingBox[4]+m_sampleWidth*((float)m_res);
}

bool LDNIcpuSolid::FileSave(char *filename)
{
	FILE *fp;

	if (!(fp=fopen(filename,"w+b"))) {printf("LDB file write error! \n"); return false;}

	fwrite(&m_res,sizeof(int),1,fp);
	fwrite(&m_sampleWidth,sizeof(float),1,fp);
	fwrite(m_origin,sizeof(float),3,fp);

	for(short nDir=0;nDir<3;nDir++) {
		fwrite(m_Rays[nDir],sizeof(LDNIcpuRay),m_res*m_res,fp);
		int sampleNum=m_Rays[nDir][m_res*m_res-1].sampleIndex;
		if (sampleNum!=0) fwrite(m_SampleArray[nDir],sizeof(LDNIcpuSample),sampleNum,fp);
	}

	fclose(fp);

	return true;
}

bool LDNIcpuSolid::FileRead(char *filename)
{
	FILE *fp;	int res;

	if (!(fp=fopen(filename,"r+b"))) {printf("LDB file read error! \n"); return false;}

	fread(&res,sizeof(int),1,fp);
	fread(&m_sampleWidth,sizeof(float),1,fp);
	fread(m_origin,sizeof(float),3,fp);

	MallocMemory(res);
	for(short nDir=0;nDir<3;nDir++) {
		fread(m_Rays[nDir],sizeof(LDNIcpuRay),m_res*m_res,fp);
		int sampleNum=m_Rays[nDir][m_res*m_res-1].sampleIndex;
		if (sampleNum!=0) {
			m_SampleArray[nDir]=(LDNIcpuSample*)(malloc(sampleNum*sizeof(LDNIcpuSample))); 
			fread(m_SampleArray[nDir],sizeof(LDNIcpuSample),sampleNum,fp);
		}
	}

	fclose(fp);

	return true;
}

LDNISolidBody::LDNISolidBody()
{
	m_solid=NULL;		m_cudaSolid=NULL;
	m_range=1.0;
	m_drawListID=-1;	m_Lighting=false;
	m_vboPosition=0;	m_vboNormal=0;
	m_cudaRegistered=false;
}

LDNISolidBody::~LDNISolidBody()
{
	if (m_solid!=NULL) delete m_solid;
	if (m_cudaSolid!=NULL) delete m_cudaSolid;
	

	DeleteVBOGLList();
}

void LDNISolidBody::drawShade()
{
	float gwidth,range,width,scale;	int sx,sy;
	_pGLK.GetScale(scale);
	range=_pGLK.GetRange();
	_pGLK.GetSize(sx,sy);	width=(sx>sy)?sx:sy;

	if (m_solid!=NULL) gwidth=m_solid->GetSampleWidth();
	if (m_cudaSolid!=NULL) gwidth=m_cudaSolid->GetSampleWidth();

	if (m_Lighting) {
		glPointSize((float)(gwidth*1.0*width*0.5/range*scale)*0.866f);
		glEnable(GL_POINT_SMOOTH);	// without this, the rectangule will be displayed for point
	}
	else {
		glDisable(GL_LIGHTING);
		glDisable(GL_POINT_SMOOTH);	// without this, the rectangule will be displayed for point
		glPointSize((float)(gwidth*.01*width*0.5/range*scale)*0.866f);
	}
	
	//------------------------------------------------------------------------------------------------
	//	Call VBO to display the points (m_vboPosition & m_vboNormal)
	if (m_vboPosition!=0 && m_vboNormal!=0) {
		glBindBufferARB( GL_ARRAY_BUFFER_ARB, m_vboNormal );
		glNormalPointer(GL_FLOAT, 0, (char *) NULL);
		glBindBufferARB( GL_ARRAY_BUFFER_ARB, m_vboPosition );
		glVertexPointer(3, GL_FLOAT, 0, (char *) NULL);

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);
		if (m_cudaSolid) 
			glColor3f(174.0f/255.0f,198.0f/255.0f,188.0f/255.0f);
		else if (m_solid) 
			glColor3f(157.0f/255.0f,187.0f/255.0f,97.0f/255.0f);
		glDrawArrays(GL_POINTS, 0, m_vertexNum);
		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}

	if (m_vbo > 0)
	{
		glColor3f(113.0f/255.0f,27.0f/255.0f,228.0f/255.0f);
		//glPointSize(5.0);
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
		glVertexPointer(3, GL_FLOAT, 0, 0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glDrawArrays(GL_POINTS, 0, m_vboSize);
		glDisableClientState(GL_VERTEX_ARRAY);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}

void LDNISolidBody::drawProfile()
{
	if (m_solid==NULL && m_cudaSolid==NULL) {glDeleteLists(m_drawListID, 1); m_drawListID=-1; return;}
	glCallList(m_drawListID);
}

void LDNISolidBody::CompRange()
{
	if (m_solid==NULL && m_cudaSolid==NULL) {m_range=1.0; return;}

	int  i,res;		float ox,oy,oz,ww,xx,yy,zz,d2;	
	int pntIndex[][3]={	{0,0,0},{0,1,0},{1,0,0},{1,1,0}, 
						{0,0,1},{0,1,1},{1,0,1},{1,1,1}	};

	if (m_cudaSolid!=NULL) {		
		m_cudaSolid->GetOrigin(ox,oy,oz);	ww=m_cudaSolid->GetSampleWidth();
		res=m_cudaSolid->GetResolution();
		
		for(i=0;i<8;i++) {
			xx=ox+(float)(pntIndex[i][0])*ww*(float)(res-1);
			yy=oy+(float)(pntIndex[i][1])*ww*(float)(res-1);
			zz=oz+(float)(pntIndex[i][2])*ww*(float)(res-1);

			d2=xx*xx+yy*yy+zz*zz;
			
			if (d2>m_range) m_range=d2;
		}
	}
	
	if (m_solid!=NULL) {
		m_solid->GetOrigin(ox,oy,oz);	ww=m_solid->GetSampleWidth();
		res=m_solid->GetResolution();
		for(i=0;i<8;i++) {
			xx=ox+(float)(pntIndex[i][0])*ww*(float)(res-1);
			yy=oy+(float)(pntIndex[i][1])*ww*(float)(res-1);
			zz=oz+(float)(pntIndex[i][2])*ww*(float)(res-1);

			d2=xx*xx+yy*yy+zz*zz;
			if (d2>m_range) m_range=d2;
		}
	}

	m_range=sqrt(m_range); //Deb
	
}

void LDNISolidBody::DeleteVBOGLList()
{
	if (m_vboPosition!=0) {
		glBindBuffer(1, m_vboPosition);
		glDeleteBuffers(1, &m_vboPosition);
		if (m_cudaRegistered) CUDA_SAFE_CALL( cudaGLUnregisterBufferObject(m_vboPosition) );
		m_vboPosition = 0;
	}
	if (m_vboNormal!=0) {
		glBindBuffer(1, m_vboNormal);
		glDeleteBuffers(1, &m_vboNormal);
		if (m_cudaRegistered) CUDA_SAFE_CALL( cudaGLUnregisterBufferObject(m_vboNormal) );
		m_vboNormal = 0;
	}
	if (m_vbo != 0) 
	{
		glBindBuffer(1, m_vbo);
		glDeleteBuffers(1, &m_vbo);
		//if (m_cudaRegistered) CUDA_SAFE_CALL( cudaGLUnregisterBufferObject(m_vbo) );
		m_vbo = 0;
	}
	m_cudaRegistered=false;
	if (m_drawListID!=-1) {glDeleteLists(m_drawListID, 1);	m_drawListID=-1;}
}

void LDNISolidBody::BuildGLList(bool bWithArrow)
{
	int res,index,i,j;	unsigned si,ei,k;	float ox,oy,oz,xx,yy,zz,x2,y2,z2,ww,nz;
	double arrow=1.0;

	DeleteVBOGLList();
	if (bWithArrow)	m_drawListID = glGenLists(1);

	if (m_cudaSolid!=NULL) {
		m_cudaSolid->BuildVBOforRendering(m_vboPosition, m_vboNormal, m_vertexNum, m_cudaRegistered);

		//--------------------------------------------------------------------------------------
		//	Build the GL List of drawProfile
		if (bWithArrow) {
			unsigned int *indexArray;	float *depthArray,*nxArray,*nyArray;
			m_cudaSolid->GetOrigin(ox,oy,oz);	ww=m_cudaSolid->GetSampleWidth();	res=m_cudaSolid->GetResolution();
			glNewList(m_drawListID, GL_COMPILE);
			glBegin(GL_LINES);
			//--------------------------------------------------------------------------------------
			//	traversal of samples in the LDNIcpuSolid
			for(int nAxis=0;nAxis<3;nAxis++) {
				switch(nAxis) {
				case 0:{glColor3f(1.0,0.0,0.0);   }break;
				case 1:{glColor3f(0.0,1.0,0.0);   }break;
				case 2:{glColor3f(0.0,0.0,1.0);   }break;
				}
				m_cudaSolid->CopyIndexArrayToHost(nAxis,indexArray);
				if (indexArray[res*res]>0) {
					m_cudaSolid->CopySampleArrayToHost(nAxis,depthArray,nxArray,nyArray);
					for(j=0;j<res;j++) {
						for(i=0;i<res;i++) {
							index=j*res+i;
							for(k=indexArray[index];k<indexArray[index+1];k++) {
								switch(nAxis) {
								case 0:{xx=ox+fabs(depthArray[k]);	yy=oy+ww*(float)i;	zz=oz+ww*(float)j;   }break;
								case 1:{xx=ox+ww*(float)j;	yy=oy+fabs(depthArray[k]);	zz=oz+ww*(float)i;   }break;
								case 2:{xx=ox+ww*(float)i;	yy=oy+ww*(float)j;	zz=oz+fabs(depthArray[k]);   }break;
								}
								glVertex3f(xx,yy,zz);

								nz=1.0f-nxArray[k]*nxArray[k]-nyArray[k]*nyArray[k];
								if (nz<0.0) nz=0.0;		if (nz>1.0) nz=1.0;
								nz=sqrt(nz);
								if (depthArray[k]<0.0f) nz=-nz;
								glNormal3f(nxArray[k],nyArray[k],nz);
								x2 = xx + nxArray[k]*ww*arrow;
								y2 = yy + nyArray[k]*ww*arrow;
								z2 = zz + nz*ww*arrow;
								glVertex3d(x2,y2,z2);
							}
						}
					}
					free(depthArray); free(nxArray); free(nyArray);
				}
				free(indexArray);
			}
			//--------------------------------------------------------------------------------------
			glEnd();
			glEndList();
		}

		printf("CUDA-LDNI solid has been processed into a GL-List with points!\n");
	}

	if (m_solid!=NULL) {
		m_solid->GetOrigin(ox,oy,oz);	ww=m_solid->GetSampleWidth();	res=m_solid->GetResolution();

		//--------------------------------------------------------------------------------------
		//	malloc the array for uploading data
		m_vertexNum=m_solid->GetSampleNumber();
		float *pos,*nvec;
		pos=(float*)malloc(sizeof(float)*3*m_vertexNum);
		nvec=(float*)malloc(sizeof(float)*3*m_vertexNum);
		//--------------------------------------------------------------------------------------
		//	traversal of samples in the LDNIcpuSolid
		int ii=0;
		for(int nAxis=0;nAxis<3;nAxis++) {
			LDNIcpuRay *rays=m_solid->GetRayArrayPtr(nAxis);
			if (rays[res*res-1].sampleIndex!=0) {
				LDNIcpuSample *sample=m_solid->GetSampleArrayPtr(nAxis);
				si=0;
				for(j=0;j<res;j++) {
					for(i=0;i<res;i++) {
						index=j*res+i;
						ei=rays[index].sampleIndex;
						for(k=si;k<ei;k++) {
							switch(nAxis) {
							case 0:{xx=ox+sample[k].depth;	yy=oy+ww*(float)i;	zz=oz+ww*(float)j;   }break;
							case 1:{xx=ox+ww*(float)j;	yy=oy+sample[k].depth;	zz=oz+ww*(float)i;   }break;
							case 2:{xx=ox+ww*(float)i;	yy=oy+ww*(float)j;	zz=oz+sample[k].depth;   }break;
							}
							pos[ii]=xx;				pos[ii+1]=yy;				pos[ii+2]=zz;
							nvec[ii]=sample[k].nx;	nvec[ii+1]=sample[k].ny;	nvec[ii+2]=sample[k].nz;
							ii+=3;
							//glNormal3f(sample[k].nx,sample[k].ny,sample[k].nz);
							//glVertex3f(xx,yy,zz);
						}
						si=ei;
					}
				}
			}
		}
		//--------------------------------------------------------------------------------------
		glGenBuffersARB( 1, &m_vboPosition );					// Get A Valid Name
		glBindBufferARB( GL_ARRAY_BUFFER_ARB, m_vboPosition );	// Bind The Buffer
		glBufferDataARB( GL_ARRAY_BUFFER_ARB, m_vertexNum*3*sizeof(float), pos, GL_STATIC_DRAW_ARB );	// upload data
		glGenBuffersARB( 1, &m_vboNormal );						// Get A Valid Name
		glBindBufferARB( GL_ARRAY_BUFFER_ARB, m_vboNormal );	// Bind The Buffer
		glBufferDataARB( GL_ARRAY_BUFFER_ARB, m_vertexNum*3*sizeof(float), nvec, GL_STATIC_DRAW_ARB );	// upload data
		//--------------------------------------------------------------------------------------
		free(pos);	free(nvec);

		//--------------------------------------------------------------------------------------
		//	Build the GL List of drawProfile
		if (bWithArrow) {
			glNewList(m_drawListID, GL_COMPILE);
			glBegin(GL_LINES);
			//--------------------------------------------------------------------------------------
			//	traversal of samples in the LDNIcpuSolid
			for(int nAxis=0;nAxis<3;nAxis++) {
				LDNIcpuRay *rays=m_solid->GetRayArrayPtr(nAxis);
				switch(nAxis) {
				case 0:{glColor3f(1.0,0.0,0.0);   }break;
				case 1:{glColor3f(0.0,1.0,0.0);   }break;
				case 2:{glColor3f(0.0,0.0,1.0);   }break;
				}
				if (rays[res*res-1].sampleIndex!=0) {
					LDNIcpuSample *sample=m_solid->GetSampleArrayPtr(nAxis);
					si=0;
					for(j=0;j<res;j++) {
						for(i=0;i<res;i++) {
							index=j*res+i;
							ei=rays[index].sampleIndex;
							for(k=si;k<ei;k++) {
								switch(nAxis) {
								case 0:{xx=ox+sample[k].depth;	yy=oy+ww*(float)i;	zz=oz+ww*(float)j;   }break;
								case 1:{xx=ox+ww*(float)j;	yy=oy+sample[k].depth;	zz=oz+ww*(float)i;   }break;
								case 2:{xx=ox+ww*(float)i;	yy=oy+ww*(float)j;	zz=oz+sample[k].depth;   }break;
								}
								glVertex3f(xx,yy,zz);
								x2=xx+sample[k].nx*ww*arrow;
								y2=yy+sample[k].ny*ww*arrow;
								z2=zz+sample[k].nz*ww*arrow;
								glVertex3d(x2,y2,z2);
							}
							si=ei;
						}
					}
				}
			}
			//--------------------------------------------------------------------------------------
			glEnd();
			glEndList();
		}
	}
}

