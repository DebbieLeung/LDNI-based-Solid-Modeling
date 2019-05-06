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
#include "scan_common.h"

#define _BISECTION_INTERVAL_SEARCH	true

#define GAUSSIAN_FUNC(xx,delta)		(exp(-(xx*xx)/(2.0f*delta*delta)))


__device__ unsigned int _encodeGlobalIndex(short nAxis, unsigned int dirSampleNum[], unsigned int localIndex)
{
	short i;		unsigned int res=0;
	for(i=0;i<nAxis;i++) res += dirSampleNum[i];
	res += localIndex;

	return res;
}

__device__ unsigned int _decodeGloablIndex(short nAxis, unsigned int dirSampleNum[], unsigned int globalIndex)
{
	short i;		unsigned int res=globalIndex;
	for(i=0;i<nAxis;i++) res -= dirSampleNum[i];
	return res;
}

__device__ bool _JacobianEigensystemSolver(float a[], int n, float v[], float eigenvalues[], float eps, int maxIter)
{
	int i,j,p,q,l;//,u,w,t,s
	float fm,cn,sn,omega,x,y,d;
	
	l=1;
	for (i=0; i<=n-1; i++) { 
		v[i*n+i]=1.0;
		for (j=0; j<=n-1; j++)
			if (i!=j) v[i*n+j]=0.0;
    }

	while (1==1) { 
		fm=0.0;
		for (i=1; i<=n-1; i++)
			for (j=0; j<=i-1; j++) { 
				d=fabs(a[i*n+j]);
				if ((i!=j)&&(d>fm))	{ fm=d; p=i; q=j;}
			}
			if (fm<eps) {for(j=0;j<n;j++) eigenvalues[j]=a[j*n+j];return true;}
			if (l>maxIter)  return false;
			l=l+1;
			x=-a[p*n+q]; y=(a[q*n+q]-a[p*n+p])/2.0;
			omega=x/sqrt(x*x+y*y);
			if (y<0.0) omega=-omega;
			sn=1.0+sqrt(1.0-omega*omega);
			sn=omega/sqrt(2.0*sn);
			cn=sqrt(1.0-sn*sn);
			fm=a[p*n+p];
			a[p*n+p]=fm*cn*cn+a[q*n+q]*sn*sn+a[p*n+q]*omega;
			a[q*n+q]=fm*sn*sn+a[q*n+q]*cn*cn-a[p*n+q]*omega;
			a[p*n+q]=0.0; a[q*n+p]=0.0;
			
			for (j=0; j<=n-1; j++)
				if ((j!=p)&&(j!=q))
				{ 
					//u=p*n+j; w=q*n+j;
					fm=a[p*n+j];
					a[p*n+j]=fm*cn+a[q*n+j]*sn;
					a[q*n+j]=-fm*sn+a[q*n+j]*cn;
				}

			for (i=0; i<=n-1; i++)
				if ((i!=p)&&(i!=q))
				{ 
					//u=i*n+p; w=i*n+q;
					fm=a[i*n+p];
					a[i*n+p]=fm*cn+a[i*n+q]*sn;
					a[i*n+q]=-fm*sn+a[i*n+q]*cn;
				}

			for (i=0; i<=n-1; i++) { 
				//u=i*n+p; w=i*n+q;
				fm=v[i*n+p];
				v[i*n+p]=fm*cn+v[i*n+q]*sn;
				v[i*n+q]=-fm*sn+v[i*n+q]*cn;
			}
    }
	
	for(j=0;j<n;j++) eigenvalues[j]=a[j*n+j];
	return true;
}

#ifdef _BISECTION_INTERVAL_SEARCH
__device__ int _bisectionIntervalSearchOnRay(float *depth, int num, float inputDepth)
{
	inputDepth=fabs(inputDepth);
	if (num==0) return 0;
	if (inputDepth<fabs(depth[0])) return 0;
	if (inputDepth>=fabs(depth[num-1])) return num;

	int st=0,ed=num-1,md;
	while((ed-st)>1) {
		md=(st+ed)>>1;	
		if (inputDepth<fabs(depth[md])) {
			ed=md;
		}
		else {
			st=md;
		}
	}
	return st;
}
#endif

__global__ void krLDNINormalReconstruction_PerSample(unsigned int* xIndexArray, unsigned int* yIndexArray, unsigned int* zIndexArray, 
									float* xNxArray, float* yNxArray, float* zNxArray, float* xNyArray, float* yNyArray, float* zNyArray, 
									float* xDepthArray, float* yDepthArray, float* zDepthArray, float *buffer, 
									int sampleNum, short nAxis, int res, float ww, unsigned int nSupportSize) 
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int i,j,k,kkl,kku,bufferindex,minIndex;
	float centerPos[3],centerNv[3],pp[3],xx,yy,dd,dbase,minValue,lower,upper,nx,ny,nz;
	float covariantM[9],eigenvectors[9],eigenvalues[3];
	int ii,jj,kk,index2,iCenter[3];
	short nDir;

	int nSearchSize=nSupportSize+1;
	float radiuSQR=ww*(float)nSupportSize;	radiuSQR=radiuSQR*radiuSQR;

	while(index<sampleNum) {

		bufferindex=index*3;
		i=(int)(buffer[bufferindex]);	j=(int)(buffer[bufferindex+1]);		k=(int)(buffer[bufferindex+2]);

		//---------------------------------------------------------------------------------------
		//	Step 1: Get the center point and its orientation
		switch(nAxis) {
		case 0:{dd=xDepthArray[index];	centerPos[nAxis]=fabs(dd);  }break;
		case 1:{dd=yDepthArray[index];	centerPos[nAxis]=fabs(dd);  }break;
		case 2:{dd=zDepthArray[index];	centerPos[nAxis]=fabs(dd);  }break;
		}
		centerPos[(nAxis+1)%3]=ww*(float)i;
		centerPos[(nAxis+2)%3]=ww*(float)j;
		centerNv[0]=centerNv[1]=centerNv[2]=0.0;
		if (k%2==0) centerNv[nAxis]=-1.0; /* entering point */ else centerNv[nAxis]=1.0; /* leaving point */
		//---------------------------------------------------------------------------------------
		covariantM[0]=covariantM[1]=covariantM[2]=0.0;
		covariantM[3]=covariantM[4]=covariantM[5]=0.0;
		covariantM[6]=covariantM[7]=covariantM[8]=0.0;
		eigenvectors[0]=eigenvectors[1]=eigenvectors[2]=0.0;
		eigenvectors[3]=eigenvectors[4]=eigenvectors[5]=0.0;
		eigenvectors[6]=eigenvectors[7]=eigenvectors[8]=0.0;
		eigenvalues[0]=eigenvalues[1]=eigenvalues[2]=0.0;

		//---------------------------------------------------------------------------------------
		//	Step 2: Search the related samples close enough and with orientation compatible
		iCenter[0]=(int)(centerPos[0]/ww);
		iCenter[1]=(int)(centerPos[1]/ww);
		iCenter[2]=(int)(centerPos[2]/ww);
		//---------------------------------------------------------------------------------------
		for(nDir=0;nDir<3;nDir++) {
			lower=centerPos[nDir]-ww*(float)nSearchSize;
			upper=centerPos[nDir]+ww*(float)nSearchSize;

			for(ii=iCenter[(nDir+1)%3]-nSearchSize;ii<=iCenter[(nDir+1)%3]+nSearchSize;ii++) {
				if (ii<0 || ii>=res) continue;
				xx=ww*(float)ii;

				for(jj=iCenter[(nDir+2)%3]-nSearchSize;jj<=iCenter[(nDir+2)%3]+nSearchSize;jj++) {
					if (((ii-iCenter[(nDir+1)%3])*(ii-iCenter[(nDir+1)%3])+(jj-iCenter[(nDir+2)%3])*(jj-iCenter[(nDir+2)%3]))
						>=(nSearchSize*nSearchSize)) continue;

					if (jj<0 || jj>=res) continue;
					yy=ww*(float)jj;

					dbase=(xx-centerPos[(nDir+1)%3])*(xx-centerPos[(nDir+1)%3])+(yy-centerPos[(nDir+2)%3])*(yy-centerPos[(nDir+2)%3]);
					if (dbase>radiuSQR) continue;

					index2=jj*res+ii;
					switch(nDir) {
					case 0:{kkl=xIndexArray[index2];	kku=xIndexArray[index2+1];}break;
					case 1:{kkl=yIndexArray[index2];	kku=yIndexArray[index2+1];}break;
					case 2:{kkl=zIndexArray[index2];	kku=zIndexArray[index2+1];}break;
					}
					for(kk=kkl;kk<kku;kk++) {
						switch(nDir) {
						case 0:{dd=xDepthArray[kk];}break;
						case 1:{dd=yDepthArray[kk];}break;
						case 2:{dd=zDepthArray[kk];}break;
						}
						if (fabs(dd)<lower) continue;
						if (fabs(dd)>upper) break;
						if (kk==k) continue;
						if ((nDir==nAxis) && ((kk%2)!=(k%2))) continue;

						//----------------------------------------------------------------------------------------
						//	The position of neighboring points
						pp[nDir]=fabs(dd);
						pp[(nDir+1)%3]=ww*(float)ii;
						pp[(nDir+2)%3]=ww*(float)jj;
						//----------------------------------------------------------------------------------------
						//	Fill the upper covariant matrix
						covariantM[0]+=(pp[0]-centerPos[0])*(pp[0]-centerPos[0]);
						covariantM[1]+=(pp[0]-centerPos[0])*(pp[1]-centerPos[1]);
						covariantM[2]+=(pp[0]-centerPos[0])*(pp[2]-centerPos[2]);
						covariantM[4]+=(pp[1]-centerPos[1])*(pp[1]-centerPos[1]);
						covariantM[5]+=(pp[1]-centerPos[1])*(pp[2]-centerPos[2]);
						covariantM[8]+=(pp[2]-centerPos[2])*(pp[2]-centerPos[2]);
					}
				}
			}
		}

		//---------------------------------------------------------------------------------------
		covariantM[3]=covariantM[1];	covariantM[6]=covariantM[2]; 	covariantM[7]=covariantM[5];	
		if (_JacobianEigensystemSolver(covariantM,3,eigenvectors,eigenvalues,0.00001,100)) {
			minIndex=0;		minValue=fabs(eigenvalues[0]);
			if (fabs(eigenvalues[1])<minValue) {minValue=fabs(eigenvalues[1]);minIndex=1;}
			if (fabs(eigenvalues[2])<minValue) minIndex=2;
			nx=eigenvectors[minIndex];		ny=eigenvectors[3+minIndex];		nz=eigenvectors[6+minIndex];
			dd=sqrt(nx*nx+ny*ny+nz*nz);
			if (dd>1.0e-5) {
				nx=nx/dd; ny=ny/dd; nz=nz/dd; 
				centerNv[0]=nx; centerNv[1]=ny; centerNv[2]=nz;
			}
			if ((centerNv[nAxis]<0.0 && k%2!=0) || (centerNv[nAxis]>0.0 && k%2==0)) {
				centerNv[0]=-centerNv[0]; 	centerNv[1]=-centerNv[1]; 	centerNv[2]=-centerNv[2]; 
			}
		}

		//---------------------------------------------------------------------------------------
		//	Step 3: Update the normal vector in the buffer
		switch(nAxis){
		case 0:dd=xDepthArray[index];break;
		case 1:dd=yDepthArray[index];break;
		case 2:dd=zDepthArray[index];break;
		}
		buffer[bufferindex]=centerNv[0];	buffer[bufferindex+1]=centerNv[1];
		if (centerNv[2]<0.0f) buffer[bufferindex+2]=-fabs(dd); else buffer[bufferindex+2]=fabs(dd);
		//---------------------------------------------------------------------------------------

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNINormalProcessing_OrientationCorrectionByVoting(
									unsigned int* xIndexArray, unsigned int* yIndexArray, unsigned int* zIndexArray,
									float* xNxArray, float* yNxArray, float* zNxArray, float* xNyArray, float* yNyArray, float* zNyArray, 
									float* xDepthArray, float* yDepthArray, float* zDepthArray, float *buffer, 
									int sampleNum, short nAxis, int res, float ww, unsigned int nSupportSize)
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int i,j,kkl,kku,bufferindex,nCompatibleNum,nTotalNum;
	float centerPos[3],centerNv[3],normal[3],xx,yy,dd,dbase,lower,upper;
	int ii,jj,kk,index2,iCenter[3];
	bool bNormalFlip;	short nDir;

	int nSearchSize=nSupportSize+1;
	float radiuSQR=ww*(float)nSupportSize;	radiuSQR=radiuSQR*radiuSQR;

	while(index<sampleNum) {

		bufferindex=index*3;
		i=(int)(buffer[bufferindex]);	j=(int)(buffer[bufferindex+1]);
		nCompatibleNum=0;	nTotalNum=0;

		//---------------------------------------------------------------------------------------
		//	Step 1: Get the center point and its orientation
		bNormalFlip=false;	
		switch(nAxis) {
		case 0:{dd=xDepthArray[index];	centerPos[nAxis]=fabs(dd);  centerNv[0]=xNxArray[index];	centerNv[1]=xNyArray[index];
			   }break;
		case 1:{dd=yDepthArray[index];	centerPos[nAxis]=fabs(dd);  centerNv[0]=yNxArray[index];	centerNv[1]=yNyArray[index];
			   }break;
		case 2:{dd=zDepthArray[index];	centerPos[nAxis]=fabs(dd);  centerNv[0]=zNxArray[index];	centerNv[1]=zNyArray[index];
			   }break;
		}
		if (dd<0.0f) bNormalFlip=true;
		centerPos[(nAxis+1)%3]=ww*(float)i;
		centerPos[(nAxis+2)%3]=ww*(float)j;
		centerNv[2]=1.0f-centerNv[0]*centerNv[0]-centerNv[1]*centerNv[1];
		if (centerNv[2]<0.0f) centerNv[2]=0.0f;	if (centerNv[2]>1.0f) centerNv[2]=1.0f;
		centerNv[2]=sqrt(centerNv[2]);			
		if (bNormalFlip) centerNv[2]=-centerNv[2];

		//---------------------------------------------------------------------------------------
		//	Step 2: Search the related samples close enough and with orientation compatible
		iCenter[0]=(int)(centerPos[0]/ww);
		iCenter[1]=(int)(centerPos[1]/ww);
		iCenter[2]=(int)(centerPos[2]/ww);
		//---------------------------------------------------------------------------------------
		for(nDir=0;nDir<3;nDir++) {
			lower=centerPos[nDir]-ww*(float)nSearchSize;
			upper=centerPos[nDir]+ww*(float)nSearchSize;

			for(ii=iCenter[(nDir+1)%3]-nSearchSize;ii<=iCenter[(nDir+1)%3]+nSearchSize;ii++) {
				if (ii<0 || ii>=res) continue;
				xx=ww*(float)ii;

				for(jj=iCenter[(nDir+2)%3]-nSearchSize;jj<=iCenter[(nDir+2)%3]+nSearchSize;jj++) {
					if (((ii-iCenter[(nDir+1)%3])*(ii-iCenter[(nDir+1)%3])+(jj-iCenter[(nDir+2)%3])*(jj-iCenter[(nDir+2)%3]))
						>=(nSearchSize*nSearchSize)) continue;

					if (jj<0 || jj>=res) continue;
					yy=ww*(float)jj;

					dbase=(xx-centerPos[(nDir+1)%3])*(xx-centerPos[(nDir+1)%3])+(yy-centerPos[(nDir+2)%3])*(yy-centerPos[(nDir+2)%3]);
					if (dbase>radiuSQR) continue;

					index2=jj*res+ii;
					switch(nDir) {
					case 0:{kkl=xIndexArray[index2];	kku=xIndexArray[index2+1];}break;
					case 1:{kkl=yIndexArray[index2];	kku=yIndexArray[index2+1];}break;
					case 2:{kkl=zIndexArray[index2];	kku=zIndexArray[index2+1];}break;
					}
					for(kk=kkl;kk<kku;kk++) {
						switch(nDir) {
						case 0:{dd=xDepthArray[kk];}break;
						case 1:{dd=yDepthArray[kk];}break;
						case 2:{dd=zDepthArray[kk];}break;
						}
						if (fabs(dd)<lower) continue;
						if (fabs(dd)>upper) break;

						switch(nDir) {
						case 0:{normal[0]=xNxArray[kk]; normal[1]=xNyArray[kk];}break;
						case 1:{normal[0]=yNxArray[kk]; normal[1]=yNyArray[kk];}break;
						case 2:{normal[0]=zNxArray[kk]; normal[1]=zNyArray[kk];}break;
						}

						//---------------------------------------------------------------------------------------
						//	for(each neighbor with pp[] and normal[])	// the result is stored in hh[]
//						pp[nDir]=fabs(dd);	pp[(nDir+1)%3]=ww*(float)ii;	pp[(nDir+2)%3]=ww*(float)jj;
						normal[2]=1.0f-normal[0]*normal[0]-normal[1]*normal[1];
						if (normal[2]<0.0f) normal[2]=0.0f;	if (normal[2]>1.0f) normal[2]=1.0f;
						normal[2]=sqrt(normal[2]);
						if (dd<0.0f) normal[2]=-normal[2];

						//---------------------------------------------------------------------------------------
						//	Voting the compatibility with its neighboring samples
						dd=centerNv[0]*normal[0]+centerNv[1]*normal[1]+centerNv[2]*normal[2];
						if (dd>=0.0) nCompatibleNum++;
						nTotalNum++;
					}
				}
			}
		}

		//---------------------------------------------------------------------------------------
		//	Step 3: Update the normal vector of samples
		if ((float)(nCompatibleNum)<0.382*(float)(nTotalNum)) {
			switch(nAxis) {
			case 0:{xDepthArray[index]=-xDepthArray[index];	xNxArray[index]=-xNxArray[index];	xNyArray[index]=-xNyArray[index]; }break;
			case 1:{yDepthArray[index]=-yDepthArray[index];	yNxArray[index]=-yNxArray[index];	yNyArray[index]=-yNyArray[index]; }break;
			case 2:{zDepthArray[index]=-zDepthArray[index];	zNxArray[index]=-zNxArray[index];	zNyArray[index]=-zNyArray[index]; }break;
			}		
		}

		index += blockDim.x * gridDim.x;
	}
}


__global__ void krLDNIBilateralNormalFilter_PerSample(unsigned int* xIndexArray, unsigned int* yIndexArray, unsigned int* zIndexArray, 
									float* xNxArray, float* yNxArray, float* zNxArray, float* xNyArray, float* yNyArray, float* zNyArray, 
									float* xDepthArray, float* yDepthArray, float* zDepthArray, float *buffer, 
									int sampleNum, short nAxis, int res, float ww, unsigned int nSupportSize, float normalPara) 
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int i,j,kkl,kku,bufferindex;//,is,ie,js,je;
	float centerPos[3],centerNv[3],pp[3],normal[3],xx,yy,dd,dbase,lower,upper,tt,ff,gg,weight,hh[3];
	int ii,jj,kk,index2,iCenter[3];
	bool bNormalFlip;	short nDir;

	int nSearchSize=nSupportSize+1;
	float deltaC=((float)nSupportSize)*0.5f*ww;	// The parameter to control the points to be considered
	float deltaG=normalPara;					// The parameter to control the band width of normal filter
	float radiuSQR=ww*(float)nSupportSize;	radiuSQR=radiuSQR*radiuSQR;

	while(index<sampleNum) {
		bufferindex=index*3;
		i=(int)(buffer[bufferindex]);	j=(int)(buffer[bufferindex+1]);		

		//---------------------------------------------------------------------------------------
		//	Step 1: Get the center point and its normal vector
		bNormalFlip=false;	
		switch(nAxis) {
		case 0:{
			dd=xDepthArray[index];
			centerPos[nAxis]=fabs(dd);  
			centerNv[0]=xNxArray[index];		centerNv[1]=xNyArray[index];
			   }break;
		case 1:{
			dd=yDepthArray[index];
			centerPos[nAxis]=fabs(dd);  
			centerNv[0]=yNxArray[index];		centerNv[1]=yNyArray[index];
			   }break;
		case 2:{
			dd=zDepthArray[index];
			centerPos[nAxis]=fabs(dd);  
			centerNv[0]=zNxArray[index];		centerNv[1]=zNyArray[index];
			   }break;
		}
		if (dd<0.0f) bNormalFlip=true;
		centerPos[(nAxis+1)%3]=ww*(float)i;
		centerPos[(nAxis+2)%3]=ww*(float)j;
		centerNv[2]=1.0f-centerNv[0]*centerNv[0]-centerNv[1]*centerNv[1];
		if (centerNv[2]<0.0f) centerNv[2]=0.0f;	if (centerNv[2]>1.0f) centerNv[2]=1.0f;
		centerNv[2]=sqrt(centerNv[2]);			
		if (bNormalFlip) centerNv[2]=-centerNv[2];

		//---------------------------------------------------------------------------------------
		//	Step 2: Compute the filtered normal vector
		hh[0]=centerNv[0];	hh[1]=centerNv[1];	hh[2]=centerNv[2];	weight=1.0f;
		iCenter[0]=(int)(centerPos[0]/ww);	
		iCenter[1]=(int)(centerPos[1]/ww);	
		iCenter[2]=(int)(centerPos[2]/ww);
		//---------------------------------------------------------------------------------------
		for(nDir=0;nDir<3;nDir++) {
			lower=centerPos[nDir]-ww*(float)nSearchSize;
			upper=centerPos[nDir]+ww*(float)nSearchSize;

			for(ii=iCenter[(nDir+1)%3]-nSearchSize;ii<=iCenter[(nDir+1)%3]+nSearchSize;ii++) {
				if (ii<0 || ii>=res) continue;
				xx=ww*(float)ii;

				for(jj=iCenter[(nDir+2)%3]-nSearchSize;jj<=iCenter[(nDir+2)%3]+nSearchSize;jj++) {
					if (((ii-iCenter[(nDir+1)%3])*(ii-iCenter[(nDir+1)%3])+(jj-iCenter[(nDir+2)%3])*(jj-iCenter[(nDir+2)%3]))
						>=(nSearchSize*nSearchSize)) continue;

					if (jj<0 || jj>=res) continue;
					yy=ww*(float)jj;

					dbase=(xx-centerPos[(nDir+1)%3])*(xx-centerPos[(nDir+1)%3])+(yy-centerPos[(nDir+2)%3])*(yy-centerPos[(nDir+2)%3]);
					if (dbase>radiuSQR) continue;

					index2=jj*res+ii;
/*#ifdef _BISECTION_INTERVAL_SEARCH
					switch(nDir) {
					case 0:{
						int stIndex=xIndexArray[index2];
						int edIndex=xIndexArray[index2+1];
						kkl=_bisectionIntervalSearchOnRay(&(xDepthArray[stIndex]),edIndex-stIndex,lower)+stIndex;
						if (kkl>edIndex) kkl=edIndex;
						kku=edIndex;
						   }break;
					case 1:{
						int stIndex=yIndexArray[index2];
						int edIndex=yIndexArray[index2+1];
						kkl=_bisectionIntervalSearchOnRay(&(yDepthArray[stIndex]),edIndex-stIndex,lower)+stIndex;
						if (kkl>edIndex) kkl=edIndex;
						kku=edIndex;
						   }break;
					case 2:{
						int stIndex=zIndexArray[index2];
						int edIndex=zIndexArray[index2+1];
						kkl=_bisectionIntervalSearchOnRay(&(zDepthArray[stIndex]),edIndex-stIndex,lower)+stIndex;
						if (kkl>edIndex) kkl=edIndex;
						kku=edIndex;
						   }break;
					}
#else*/
					switch(nDir) {
					case 0:{kkl=xIndexArray[index2];	kku=xIndexArray[index2+1];}break;
					case 1:{kkl=yIndexArray[index2];	kku=yIndexArray[index2+1];}break;
					case 2:{kkl=zIndexArray[index2];	kku=zIndexArray[index2+1];}break;
					}
//#endif
					for(kk=kkl;kk<kku;kk++) {
						switch(nDir) {
						case 0:{dd=xDepthArray[kk];}break;
						case 1:{dd=yDepthArray[kk];}break;
						case 2:{dd=zDepthArray[kk];}break;
						}
						if (fabs(dd)<lower) continue;
						if (fabs(dd)>upper) break;

						switch(nDir) {
						case 0:{normal[0]=xNxArray[kk]; normal[1]=xNyArray[kk];}break;
						case 1:{normal[0]=yNxArray[kk]; normal[1]=yNyArray[kk];}break;
						case 2:{normal[0]=zNxArray[kk]; normal[1]=zNyArray[kk];}break;
						}

						//---------------------------------------------------------------------------------------
						//	for(each neighbor with pp[] and normal[])	// the result is stored in hh[]
						pp[nDir]=fabs(dd);
						pp[(nDir+1)%3]=ww*(float)ii;
						pp[(nDir+2)%3]=ww*(float)jj;
						normal[2]=1.0f-normal[0]*normal[0]-normal[1]*normal[1];
						if (normal[2]<0.0f) normal[2]=0.0f;	if (normal[2]>1.0f) normal[2]=1.0f;
						normal[2]=sqrt(normal[2]);
						if (dd<0.0f) normal[2]=-normal[2];
						
						//---------------------------------------------------------------------------------------
						dd=dbase+(pp[nDir]-centerPos[nDir])*(pp[nDir]-centerPos[nDir]);
						if (dd<=radiuSQR) 
						{
							tt=centerNv[0]*(centerNv[0]-normal[0])
								+centerNv[1]*(centerNv[1]-normal[1])+centerNv[2]*(centerNv[2]-normal[2]);
							dd=sqrt(dd);
							ff=GAUSSIAN_FUNC(dd,deltaC);	gg=GAUSSIAN_FUNC(tt,deltaG);

							hh[0]=hh[0]+ff*gg*normal[0];	hh[1]=hh[1]+ff*gg*normal[1];	hh[2]=hh[2]+ff*gg*normal[2];
							weight=weight+ff*gg;
						}
					}
				}
			}
		}

		//---------------------------------------------------------------------------------------
		if (fabs(weight)>0.00001f) {
			hh[0]=hh[0]/weight; hh[1]=hh[1]/weight; hh[2]=hh[2]/weight;
		} else {
			hh[0]=centerNv[0];	hh[1]=centerNv[1];	hh[2]=centerNv[2];
		}
		if (fabs(hh[2])<0.00001f) hh[2]=0;
		dd=sqrt(hh[0]*hh[0]+hh[1]*hh[1]+hh[2]*hh[2]);
		if (dd<0.00001f) {
			hh[0]=centerNv[0];	hh[1]=centerNv[1];	hh[2]=centerNv[2];
		} else {
			hh[0]=hh[0]/dd; hh[1]=hh[1]/dd; hh[2]=hh[2]/dd;
		}

		//---------------------------------------------------------------------------------------
		//	Step 3: Update the normal vector in the buffer
		switch(nAxis){
		case 0:dd=xDepthArray[index];break;
		case 1:dd=yDepthArray[index];break;
		case 2:dd=zDepthArray[index];break;
		}
		buffer[bufferindex]=hh[0];	buffer[bufferindex+1]=hh[1];
		if (hh[2]<0.0f) buffer[bufferindex+2]=-fabs(dd); else buffer[bufferindex+2]=fabs(dd);

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNIBilateralNormalFilter_PerRay(unsigned int* xIndexArray, unsigned int* yIndexArray, unsigned int* zIndexArray,
									float* xNxArray, float* yNxArray, float* zNxArray, float* xNyArray, float* yNyArray, float* zNyArray, 
									float* xDepthArray, float* yDepthArray, float* zDepthArray, float *buffer, 
									int arrsize, short nAxis, int res, float ww, float ox, float oy, float oz, unsigned int nSupportSize, float normalPara) 
{
	int index=threadIdx.x+blockIdx.x*blockDim.x;
	int i,j,k,kl,ku,kkl,kku,bufferindex;
	float centerPos[3],origin[3],centerNv[3],pp[3],normal[3],xx,yy,dd,dbase,depth,lower,upper,tt,ff,gg,weight,hh[3];
	int ii,jj,kk,index2,iCenter[3];
	bool bNormalFlip;	short nDir;

	int nSearchSize=nSupportSize+1;
	float deltaC=((float)nSupportSize)*0.5f*ww;	// The parameter to control the points to be considered
	float deltaG=normalPara;					// The parameter to control the band width of normal filter
	float radiuSQR=ww*(float)nSupportSize;	radiuSQR=radiuSQR*radiuSQR;
	origin[0]=ox;	origin[1]=oy;	origin[2]=oz;

	while(index<arrsize) {
		i=index%res;	j=index/res;
		
		switch(nAxis) {
		case 0:{kl=xIndexArray[index];	ku=xIndexArray[index+1];}break;
		case 1:{kl=yIndexArray[index];	ku=yIndexArray[index+1];}break;
		case 2:{kl=zIndexArray[index];	ku=zIndexArray[index+1];}break;
		}
		for(k=kl;k<ku;k++) {
			bufferindex=k*3;
			tt=buffer[bufferindex];	ff=buffer[bufferindex+1];	gg=buffer[bufferindex+2];
			//---------------------------------------------------------------------------------------
			//	Step 1: Get the center point and its normal vector
			bNormalFlip=false;
			switch(nAxis) {
			case 0:{centerPos[nAxis]=origin[nAxis]+fabs(xDepthArray[k]);  
					centerNv[0]=xNxArray[k];	centerNv[1]=xNyArray[k];
					if (xDepthArray[k]<0.0f) bNormalFlip=true;
				   }break;
			case 1:{centerPos[nAxis]=origin[nAxis]+fabs(yDepthArray[k]);  
					centerNv[0]=yNxArray[k];	centerNv[1]=yNyArray[k];
					if (yDepthArray[k]<0.0f) bNormalFlip=true;
				   }break;
			case 2:{centerPos[nAxis]=origin[nAxis]+fabs(zDepthArray[k]);  
					centerNv[0]=zNxArray[k];	centerNv[1]=zNyArray[k];
					if (zDepthArray[k]<0.0f) bNormalFlip=true;
				   }break;
			}
			centerPos[(nAxis+1)%3]=origin[(nAxis+1)%3]+ww*(float)i;
			centerPos[(nAxis+2)%3]=origin[(nAxis+2)%3]+ww*(float)j;
			centerNv[2]=1.0f-centerNv[0]*centerNv[0]-centerNv[1]*centerNv[1];
			if (centerNv[2]<0.0f) centerNv[2]=0.0f;	if (centerNv[2]>1.0f) centerNv[2]=1.0f;
			centerNv[2]=sqrt(centerNv[2]);			
			if (bNormalFlip) centerNv[2]=-centerNv[2];

			//---------------------------------------------------------------------------------------
			//	Step 2: Compute the filtered normal vector
			hh[0]=centerNv[0];	hh[1]=centerNv[1];	hh[2]=centerNv[2];	weight=1.0f;
			iCenter[0]=(int)((centerPos[0]-origin[0])/ww);	
			iCenter[1]=(int)((centerPos[1]-origin[1])/ww);	
			iCenter[2]=(int)((centerPos[2]-origin[2])/ww);

			for(nDir=0;nDir<3;nDir++) {
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
						switch(nDir) {
						case 0:{kkl=xIndexArray[index2];	kku=xIndexArray[index2+1];}break;
						case 1:{kkl=yIndexArray[index2];	kku=yIndexArray[index2+1];}break;
						case 2:{kkl=zIndexArray[index2];	kku=zIndexArray[index2+1];}break;
						}
						for(kk=kkl;kk<kku;kk++) {
							switch(nDir) {
							case 0:{depth=xDepthArray[kk]; normal[0]=xNxArray[kk]; normal[1]=xNyArray[kk];}break;
							case 1:{depth=yDepthArray[kk]; normal[0]=yNxArray[kk]; normal[1]=yNyArray[kk];}break;
							case 2:{depth=zDepthArray[kk]; normal[0]=zNxArray[kk]; normal[1]=zNyArray[kk];}break;
							}
							if (fabs(depth)<lower) continue;
							if (fabs(depth)>upper) break;

							//---------------------------------------------------------------------------------------
							//	for(each neighbor with pp[] and normal[])	// the result is stored in hh[]
							pp[nDir]=origin[nDir]+fabs(depth);
							pp[(nDir+1)%3]=origin[(nDir+1)%3]+ww*(float)ii;
							pp[(nDir+2)%3]=origin[(nDir+2)%3]+ww*(float)jj;
							normal[2]=1.0f-normal[0]*normal[0]-normal[1]*normal[1];
							if (normal[2]<0.0f) normal[2]=0.0f;	if (normal[2]>1.0f) normal[2]=1.0f;
							normal[2]=sqrt(normal[2]);
							if (depth<0.0f) normal[2]=-normal[2];
							
							//---------------------------------------------------------------------------------------
							dd=dbase+(pp[nDir]-centerPos[nDir])*(pp[nDir]-centerPos[nDir]);
							if (dd<=radiuSQR) {
								tt=centerNv[0]*(centerNv[0]-normal[0])
									+centerNv[1]*(centerNv[1]-normal[1])+centerNv[2]*(centerNv[2]-normal[2]);
								dd=sqrt(dd);
								ff=GAUSSIAN_FUNC(dd,deltaC);	gg=GAUSSIAN_FUNC(tt,deltaG);

								hh[0]=hh[0]+ff*gg*normal[0];	hh[1]=hh[1]+ff*gg*normal[1];	hh[2]=hh[2]+ff*gg*normal[2];
								weight=weight+ff*gg;
							}
						}
					}
				}
			}
	
			if (fabs(weight)>0.00001f) {
				hh[0]=hh[0]/weight; hh[1]=hh[1]/weight; hh[2]=hh[2]/weight;
			} else {
				hh[0]=centerNv[0];	hh[1]=centerNv[1];	hh[2]=centerNv[2];
			}
			if (fabs(hh[2])<0.00001f) hh[2]=0;
			dd=sqrt(hh[0]*hh[0]+hh[1]*hh[1]+hh[2]*hh[2]);
			if (dd<0.00001f) {
				hh[0]=centerNv[0];	hh[1]=centerNv[1];	hh[2]=centerNv[2];
			} else {
				hh[0]=hh[0]/dd; hh[1]=hh[1]/dd; hh[2]=hh[2]/dd;
			}

			//---------------------------------------------------------------------------------------
			//	Step 3: Update the normal vector in the buffer
			switch(nAxis) {	
			case 0:depth=xDepthArray[k];break;
			case 1:depth=yDepthArray[k];break;
			case 2:depth=zDepthArray[k];break;
			}
			buffer[bufferindex]=hh[0];	bufferindex++;
			buffer[bufferindex]=hh[1];	bufferindex++;
			if (hh[2]<0.0f) buffer[bufferindex]=-fabs(depth); else buffer[bufferindex]=fabs(depth);
		}

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNINormalProcessing_PreProc(unsigned int* indexArray, float *buffer, int res, int arrsize)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	int i,j,k,kl,ku;

	while(index<arrsize) {
		i=index%res;	j=index/res;
		kl=indexArray[index];	ku=indexArray[index+1];
		for(k=kl;k<ku;k++) {
			buffer[k*3]=i;
			buffer[k*3+1]=j;
			buffer[k*3+2]=k;
		}

		index += blockDim.x * gridDim.x;
	}
}

__global__ void krLDNINormalProcessing_Update(int sampleNum, float *nxArray, float *nyArray, float *depthArray, float *buffer)
{
	int index = threadIdx.x + blockIdx.x*blockDim.x;

	while(index<sampleNum) {
		nxArray[index]=buffer[index*3]; 
		nyArray[index]=buffer[index*3+1];
		depthArray[index]=buffer[index*3+2];

		index += blockDim.x * gridDim.x;
	}
}