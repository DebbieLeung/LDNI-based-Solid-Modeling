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

#include <stdio.h>
#include <stdlib.h>
#include <io.h>
#include <math.h>
#include <string.h>

#include <time.h>
#include "../common/GL/glew.h"
#include "../common/GL/glut.h"
#include <windows.h>


#include "..\GLKLib\GLK.h"
#include "..\GLKLib\GLKCameraTool.h"


#include "LDNIDataBoard.h"
#include "PMBody.h"

#include "LDNIcpuSolid.h"
#include "LDNIcpuOperation.h"

#include "LDNIcudaSolid.h"
#include "LDNIcudaOperation.h"


#define _MENU_QUIT						10001
#define _MENU_FILE_OPEN					10002
#define _MENU_FILE_SAVE					10003
#define _MENU_FILE_SGM					10004
#define _MENU_FILE_OPENALL				10005
	
#define _MENU_VIEW_ISOMETRIC			10101
#define _MENU_VIEW_FRONT				10102
#define _MENU_VIEW_BACK					10103
#define _MENU_VIEW_TOP					10104
#define _MENU_VIEW_BOTTOM				10105
#define _MENU_VIEW_LEFT					10106
#define _MENU_VIEW_RIGHT				10107
#define _MENU_VIEW_ORBITPAN				10108
#define _MENU_VIEW_ZOOMWINDOW			10109
#define _MENU_VIEW_ZOOMIN				10110
#define _MENU_VIEW_ZOOMOUT				10111
#define _MENU_VIEW_ZOOMALL				10112
#define _MENU_VIEW_PROFILE				10113
#define _MENU_VIEW_SHADE				10114
#define _MENU_VIEW_MESH					10115
#define _MENU_VIEW_AXIS					10116
#define _MENU_VIEW_COORD				10117
#define _MENU_VIEW_MESHSMOOTHSHADING	10118
#define _MENU_VIEW_LDNINORMAL			10119
#define _MENU_VIEW_LDNILIGHTING			10120
#define _MENU_VIEW_CLIP					10121


#define	_MENU_MESH_BNDBOXCOMP			10224
#define _MENU_MESH_MODELFLIPPING		10225
#define _MENU_MESH_MODELTRANSF			10226
#define _MENU_MESH_CLEARALL				10227
#define _MENU_MESH_MODELSCALE			10228
#define _MENU_MESH_SHIFTORIGIN			10229
#define _MENU_MESH_SHIFTPOS				10230

#define _MENU_LDNI_SAMPLINGFROMBREP		10301
#define _MENU_LDNI_BOOLEANOPERATION		10302
#define _MENU_LDNI_BILATERAL_NORMALFILTER		10303
#define _MENU_LDNI_CONVERT2CUDASOLID	10304

#define _MENU_CUDA_SAMPLINGFROMBREP		10401
#define _MENU_CUDA_BOOLEANOPERATION		10402
#define _MENU_CUDA_REGULARIZATION		10403
#define _MENU_CUDA_CONTOURING			10404
#define _MENU_CUDA_OFFSETTING			10405
#define _MENU_CUDA_OFFSETTING_SH		10406
#define _MENU_CUDA_OFFSETTING_SHRP		10407
#define _MENU_CUDA_OFFSETTING_SH_SUCC	10408
#define _MENU_CUDA_OFFSETTING_SH_SUCC_NORMAL	10409
#define _MENU_CUDA_BILATERAL_NORMALFILTER		10410
#define _MENU_CUDA_ORIENTEDNORMALRECON	10411
#define _MENU_CUDA_ORIENTEDNORMALRECON2	10412
#define _MENU_CUDA_CONVERT2CPUSOLID		10413

#define _MENU_CUDA_PROPERTY				10499
#define _MENU_CUDA_SCAFFOLDOPERATION	10414
#define _MENU_CUDA_SUPERUNION			10415
#define _MENU_CUDA_FDMCONTOUR			10416
#define _MENU_CUDA_FDMCONTOURSUPT		10418
#define _MENU_CUDA_SLACONTOURSUPT		10419

#define _MAX_INFO 5

GLK _pGLK;

LDNIDataBoard _pDataBoard;
int _pMainWnd;
bool _bExpandableWorkingSpace = true;
char m_info[_MAX_INFO][10];
short inputmesh = 0;

extern void menuEvent(int idCommand);
extern bool initGLInteroperabilityOnCUDA(int major, int minor);


int ControlWidth = 300;
int ControlHeight = 1000;



void menuFuncQuit()
{
	exit(0);
}


void displayCoordinate(int x, int y) 
{
    double wx,wy,wz;

	_pGLK.screen_to_wcl(x, y, wx, wy, wz);
	_pGLK.m_currentCoord[0]=(float)wx;
	_pGLK.m_currentCoord[1]=(float)wy;
	_pGLK.m_currentCoord[2]=(float)wz;

//	printf("(%.2f, %.2f, %.2f)\n",(float)wx,(float)wy,(float)wz);

	_pGLK.refresh();
}


void specialKeyboardFunc(int key, int x, int y)
{
	
	pick_event pe;
	switch(_pGLK.m_mouseState) {
	case 1:{pe.nFlags=GLUT_LEFT_BUTTON;
		   }break;
	case 2:{pe.nFlags=GLUT_MIDDLE_BUTTON;
		   }break;
	case 3:{pe.nFlags=GLUT_RIGHT_BUTTON;
		   }break;
	}
	pe.x=(double)x;	pe.y=(double)y;
	pe.nChar=-key;

	_pGLK.m_nModifier=0;
	switch(glutGetModifiers()) {
	case GLUT_ACTIVE_SHIFT:{_pGLK.m_nModifier=1;}break;
	case GLUT_ACTIVE_CTRL:{_pGLK.m_nModifier=2;	}break;
	case GLUT_ACTIVE_ALT:{_pGLK.m_nModifier=3;	}break;
	}

	if (_pGLK.GetCurrentTool()) _pGLK.GetCurrentTool()->process_event(KEY_PRESS,pe);
}



void keyboardFunc(unsigned char key, int x, int y)
{	
	//------------------------------------------------------------------
	//	Hot Key Processing
	switch(key) {
	case 1:{	// ctrl+a
			menuEvent(_MENU_VIEW_ZOOMALL); return;
		   }break;
	case 12:{	// ctrl+l
			menuEvent(_MENU_VIEW_LDNILIGHTING); return;
		   }break;
	case 15:{	// ctrl+o
			menuEvent(_MENU_FILE_OPEN); return;
		   }break;
	case 18:{	// ctrl+r
			menuEvent(_MENU_VIEW_ORBITPAN); return;
		   }break;
	case 19:{	// ctrl+s
			menuEvent(_MENU_FILE_SAVE); return;
		   }break;
	case 23:{	// ctrl+w
			menuEvent(_MENU_VIEW_ZOOMWINDOW); return;
		   }break;
	case 26:
			{
			_pGLK.SetClipView();
			}break;	
	case 'q': menuFuncQuit(); break;
	}

	pick_event pe;
	switch(_pGLK.m_mouseState) {
	case 1:{pe.nFlags=GLUT_LEFT_BUTTON;
		   }break;
	case 2:{pe.nFlags=GLUT_MIDDLE_BUTTON;
		   }break;
	case 3:{pe.nFlags=GLUT_RIGHT_BUTTON;
		   }break;
	}
	pe.x=(double)x;	pe.y=(double)y;
	pe.nChar=key;

	_pGLK.m_nModifier=0;
	switch(glutGetModifiers()) {
	case GLUT_ACTIVE_SHIFT:{_pGLK.m_nModifier=1;}break;
	case GLUT_ACTIVE_CTRL:{_pGLK.m_nModifier=2;	}break;
	case GLUT_ACTIVE_ALT:{_pGLK.m_nModifier=3;	}break;
	}

	if (_pGLK.GetCurrentTool()) _pGLK.GetCurrentTool()->process_event(KEY_PRESS,pe);
}


void motionFunc(int x, int y)
{
	
	if (_pGLK.m_mouseState==0) return;

	pick_event pe;
	switch(_pGLK.m_mouseState) {
	case 1:{pe.nFlags=GLUT_LEFT_BUTTON;
		   }break;
	case 2:{pe.nFlags=GLUT_MIDDLE_BUTTON;
		   }break;
	case 3:{pe.nFlags=GLUT_RIGHT_BUTTON;
		   }break;
	}
	pe.x=(double)x;
	pe.y=(double)y;

	if (_pGLK.m_bCoordDisp) displayCoordinate(x,y);
	if (_pGLK.GetCurrentTool()) _pGLK.GetCurrentTool()->process_event(MOUSE_MOVE,pe);

	
}

void passiveMotionFunc(int x, int y)
{	
	pick_event pe;
	pe.nFlags=-1;
	pe.x=(double)x;
	pe.y=(double)y;
	if (_pGLK.m_bCoordDisp) displayCoordinate(x,y);
	if (_pGLK.GetCurrentTool()) _pGLK.GetCurrentTool()->process_event(MOUSE_MOVE,pe);
}



void mouseFunc(int button, int state, int x, int y)
{
	
	if (state==GLUT_DOWN) {
		pick_event pe;
		_pGLK.m_nModifier=0;
		switch(glutGetModifiers()) {
		case GLUT_ACTIVE_SHIFT:{_pGLK.m_nModifier=1;}break;
		case GLUT_ACTIVE_CTRL:{_pGLK.m_nModifier=2;	}break;
		case GLUT_ACTIVE_ALT:{_pGLK.m_nModifier=3;	}break;
		}
		if (button==GLUT_LEFT_BUTTON) {pe.nFlags=GLUT_LEFT_BUTTON;_pGLK.m_mouseState=1;}
		if (button==GLUT_MIDDLE_BUTTON) {pe.nFlags=GLUT_MIDDLE_BUTTON;_pGLK.m_mouseState=2;}
		if (button==GLUT_RIGHT_BUTTON) {pe.nFlags=GLUT_RIGHT_BUTTON;_pGLK.m_mouseState=3;}
		pe.x=(double)x;
		pe.y=(double)y;
		if (_pGLK.GetCurrentTool()) _pGLK.GetCurrentTool()->process_event(MOUSE_BUTTON_DOWN,pe);
	}
	else if (state==GLUT_UP) {
		pick_event pe;
		_pGLK.m_nModifier=0;
		switch(glutGetModifiers()) {
		case GLUT_ACTIVE_SHIFT:{_pGLK.m_nModifier=1;}break;
		case GLUT_ACTIVE_CTRL:{_pGLK.m_nModifier=2;	}break;
		case GLUT_ACTIVE_ALT:{_pGLK.m_nModifier=3;	}break;
		}
		if (button==GLUT_LEFT_BUTTON) pe.nFlags=GLUT_LEFT_BUTTON;
		if (button==GLUT_MIDDLE_BUTTON) pe.nFlags=GLUT_MIDDLE_BUTTON;
		if (button==GLUT_RIGHT_BUTTON) pe.nFlags=GLUT_RIGHT_BUTTON;
		pe.x=(double)x;
		pe.y=(double)y;
		if (_pGLK.GetCurrentTool()) _pGLK.GetCurrentTool()->process_event(MOUSE_BUTTON_UP,pe);

		_pGLK.m_mouseState=0;
	}
	
}

void animationFunc()
{
	glutPostRedisplay();
}

void visibleFunc(int visible)
{
	if (visible==GLUT_VISIBLE)
		glutIdleFunc(animationFunc);
	else
		glutIdleFunc(NULL);

}

void displayFunc(void)
{
	_pGLK.refresh();
}

void reshapeFunc(int w, int h) 
{	
	_pGLK.Reshape(w,h);
}

void initFunc()
{
	_pGLK.Initialization();
	_pGLK.SetMesh(false);
	GLKCameraTool *myTool=new GLKCameraTool(&_pGLK,ORBITPAN);
	_pGLK.clear_tools();
	_pGLK.set_tool(myTool);
}

//---------------------------------------------------------------------------------
//	The following functions are for menu processing



void menuFuncFileSave()
{
	char filename[256],name[100],exstr[4],answer[20];
    struct _finddata_t c_file;
    long hFile;

	printf("\nPlease specify the file name for export: ");
	scanf("%s",name);
	printf("\n");

	strcpy(filename,"Data\\");	
	strcat(filename,name);
	
    // Detect whether the file exist.
    if( (hFile = _findfirst( filename, &c_file )) != -1L ) {
		printf( "The file - %s has been found, do you want to overwite it? (y/n)\n", filename);
		scanf("%s",answer);
		if (answer[0]=='n' || answer[0]=='N') return;
	}

	int length=(int)(strlen(filename));
	exstr[0]=filename[length-3];
	exstr[1]=filename[length-2];
	exstr[2]=filename[length-1];
	exstr[3]='\0';

	if (_stricmp(exstr,"obj")==0) {			//.OBJ file
		if (_pDataBoard.m_polyMeshBody!=NULL) {
			QuadTrglMesh *mesh=NULL;
			int num=_pDataBoard.m_polyMeshBody->GetMeshList().GetCount();
			if (num>1) {
			}
			else {
				mesh=(QuadTrglMesh *)(_pDataBoard.m_polyMeshBody->GetMeshList().GetTail());
			}
			if (mesh!=NULL) {
				long time=clock();	printf("Starting to write the OBJ file ...\n");
				mesh->OutputOBJFile(filename);
				printf("Completed and take %ld (ms)\n",clock()-time);
			}
		}
	}
	else if (_stricmp(exstr,"meb")==0) {	//.MEB file
		if (_pDataBoard.m_polyMeshBody!=NULL) {
			QuadTrglMesh *mesh=NULL;
			int num=_pDataBoard.m_polyMeshBody->GetMeshList().GetCount();
			if (num>1) {
			}
			else {
				mesh=(QuadTrglMesh *)(_pDataBoard.m_polyMeshBody->GetMeshList().GetTail());
			}
			if (mesh!=NULL) {
				long time=clock();	printf("Starting to write the MEB file ...\n");
				mesh->OutputMEBFile(filename);
				printf("Completed and take %ld (ms)\n",clock()-time);
			}
		}
	}
	else if (_stricmp(exstr,"ldb")==0) {	//.LDB file
		if (_pDataBoard.m_solidLDNIBody!=NULL) {
			if (_pDataBoard.m_solidLDNIBody->m_solid!=NULL) {
				long time=clock();	printf("Starting to write the LDI file ...\n");
				_pDataBoard.m_solidLDNIBody->m_solid->FileSave(filename);
				printf("Completed and take %ld (ms)\n",clock()-time);
			}
			else {
				printf("Warning: no CPU-LDNI solid is found!\n");
			}
		}
	}
	else if (_stricmp(exstr,"lda")==0) {	//.LDA file
		if (_pDataBoard.m_solidLDNIBody!=NULL) {
			if (_pDataBoard.m_solidLDNIBody->m_cudaSolid!=NULL) {
				long time=clock();	printf("Starting to write the LDA file ...\n");
				_pDataBoard.m_solidLDNIBody->m_cudaSolid->FileSave(filename);
				printf("Completed and take %ld (ms)\n",clock()-time);
			}
			else {
				printf("Warning: no CUDA-LDNI solid is found!\n");
			}
		}
	}
	else if (_stricmp(exstr,"ldn")==0) {	//.LDN file
		if (_pDataBoard.m_solidLDNIBody!=NULL) {
			if (_pDataBoard.m_solidLDNIBody->m_cudaSolid!=NULL) {
				long time=clock();	printf("Starting to write the LDN file ...\n");
				_pDataBoard.m_solidLDNIBody->m_cudaSolid->ExportLDNFile(filename);
				printf("Completed and take %ld (ms)\n",clock()-time);
			}
			else {
				printf("Warning: no cuda-LDNI solid is found!\n");
			}
		}
	}
	else {
		printf("Warning: incorrect file extension, no file is saved!\n");
	}
}

bool fileChosenByList(char directorty[], char filespef2[], char selectedFileName[])
{
    struct _finddata_t c_file;
    long hFile;
	long fileNum=0;
	char filespef[200];
	int colNum=1,colsize;

	colsize=60/colNum-4;
	strcpy(filespef,directorty);
	strcat(filespef,filespef2);	

    if( (hFile = _findfirst( filespef, &c_file )) == -1L ) {  
		printf( "No file is found!\n");
		return false;
	}
	printf( "%*d: %s %*s", 2, fileNum++, c_file.name, colsize-strlen(c_file.name), " ");
    while(_findnext( hFile, &c_file )!=-1L) {
		printf( "%*d: %s %*s", 2, fileNum++, c_file.name, colsize-strlen(c_file.name), " ");
		if ((fileNum%colNum)==0) printf("\n");
	}
	_findclose(hFile);

	int inputNum;	char inputStr[200];
	printf("\nPlease select the file name for import: ");
	scanf("%s",inputStr);	printf("\n");	sscanf(inputStr,"%d",&inputNum);
	if (inputNum<0 || inputNum>=fileNum) {printf("Incorrect Input!!!\n"); return false;}

	fileNum=0;
    if( (hFile = _findfirst( filespef, &c_file )) == -1L ) {return false;}
	if (inputNum!=0) {
		fileNum++;
		while(_findnext( hFile, &c_file )!=-1L) {
			if (fileNum==inputNum) break;
			fileNum++;
		}
	}
	_findclose(hFile);
	strcpy(selectedFileName,c_file.name);
	printf("----------------------------------------\nSelected File: %s\n",selectedFileName);
	return true;
}

void menuFuncFileOpenAll()
{

	WIN32_FIND_DATA FindFileData;
	HANDLE hFind = INVALID_HANDLE_VALUE;

	int i=0;
	int charlength = 256;
	char filedir[256] = {'\0'};
	char filedir2[256] = {'\0'};
	char *filename[50];
	char *foldername[50];
	
	GetCurrentDirectory(256,filedir);
	strcat(filedir, "\\*.*");
	printf("Current Directory : \n%s\n", filedir);
	
	
	hFind = FindFirstFile(filedir, &FindFileData);
	if (hFind == INVALID_HANDLE_VALUE)
	{
		printf ("Invalid file handle (%s). Error is %u.\n", filedir, GetLastError());
		return ;
	} 
	else
	{
		// List all the other files in the directory.
		while (FindNextFile(hFind, &FindFileData) != 0)
		{			
			if (FindFileData.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY)
			{
				charlength = strlen(FindFileData.cFileName);
				foldername[i]=new char[charlength];
				strcpy(foldername[i],FindFileData.cFileName);
				printf("%d) %s \n",i, foldername[i]);
				i++;
			}
		}
		FindClose(hFind);
	}

	int inputNum;	char inputStr[200];
	printf("\nPlease select the file folder for import: ");
	scanf("%s",inputStr);	printf("\n");	sscanf(inputStr,"%d",&inputNum);

	printf("Current Folder : %s\n", foldername[inputNum]);
	GetCurrentDirectory(256,filedir);
	strcat(filedir, "\\");
	strcat(filedir, foldername[inputNum]);
	strcpy(filedir2, filedir);
	strcat(filedir, "\\*.*");
	char exstr[4];
	int l = 0;
	i=0;
	hFind = FindFirstFile(filedir, &FindFileData);
	long time=clock();
	if (hFind == INVALID_HANDLE_VALUE)
	{
		printf ("Invalid file handle (%s). Error is %u.\n", filedir, GetLastError());
		return ;
	} 
	else
	{
		while (FindNextFile(hFind, &FindFileData) != 0)
		{			
			charlength = strlen(FindFileData.cFileName);
			if (charlength < 3) continue;
			filename[i]=new char[256];
			strcpy(filename[i],filedir2);
			strcat(filename[i], "\\");
			strcat(filename[i], FindFileData.cFileName );
			
			
			//printf("%d) %s \n",i, filename[i]);
			charlength = strlen(filename[i]);
				

			exstr[0]=filename[i][charlength-3];
			exstr[1]=filename[i][charlength-2];
			exstr[2]=filename[i][charlength-1];
			exstr[3]='\0';

			if (_stricmp(exstr,"obj")==0) {
				if (!(_pDataBoard.m_polyMeshBody)) 
					_pDataBoard.m_polyMeshBody=new PMBody;
				else
					_pGLK.DelDisplayObj2(_pDataBoard.m_polyMeshBody);

				QuadTrglMesh *mesh=new QuadTrglMesh;
				if (mesh->InputOBJFile(filename[i])) {
					_pDataBoard.m_polyMeshBody->GetMeshList().AddTail(mesh);	
					_pDataBoard.m_polyMeshBody->computeRange();
					mesh->SetMeshId(_pDataBoard.m_polyMeshBody->GetMeshList().GetCount());
					l += mesh->GetFaceNumber();
					printf("OBJ File Import Time (ms): %ld\n",clock()-time); 
				}else 
				{delete mesh;	printf("OBJ file importing fails!\n"); 	return;}
				
				if (_pGLK.GetShading()) _pDataBoard.m_polyMeshBody->BuildGLList(true);
				if (_pGLK.GetMesh()) _pDataBoard.m_polyMeshBody->BuildGLList(false);
				
				_pGLK.AddDisplayObj(_pDataBoard.m_polyMeshBody,true);
				//_pUI.AddControlMesh(_pDataBoard.m_polyMeshBody);
				
			}
			else if (_stricmp(exstr,"stl")==0) {
				
			}
			i++;
		}
		FindClose(hFind);
	}
	printf("--------------------------------------------\n");
	printf("Total Face Num: %d\n",l); 
	printf("Open File Time (ms): %ld\n",clock()-time); time=clock();

}

void menuFuncFileOpen()
{
	char filename[256],name[100],exstr[4];
    struct _finddata_t c_file;
    long hFile;

	if (!fileChosenByList("Data\\","*",name)) return;
	strcpy(filename,"Data\\");	strcat(filename,name);

    // Detect whether the file exist.
    if( (hFile = _findfirst( filename, &c_file )) == -1L ) {
		printf( "The file - %s is not found!\n", filename);
		return;
	}

	int length=(int)(strlen(filename));
	exstr[0]=filename[length-3];
	exstr[1]=filename[length-2];
	exstr[2]=filename[length-1];
	exstr[3]='\0';

	if (_stricmp(exstr,"lda")==0) {	//.lda file
		if (!(_pDataBoard.m_solidLDNIBody)) 
			_pDataBoard.m_solidLDNIBody=new LDNISolidBody;
		else
			_pGLK.DelDisplayObj2(_pDataBoard.m_solidLDNIBody);

		long time=clock();	
		LDNIcudaSolid *solid=new LDNIcudaSolid;
		if (solid->FileRead(filename)) 
			{_pDataBoard.m_solidLDNIBody->m_cudaSolid=solid;}
		else
			{delete solid;	printf("LDB file importing fails!\n"); 	return;}
		printf("LDB File Import Time (ms): %ld\n",clock()-time); 
		_pDataBoard.m_solidLDNIBody->CompRange();
		time=clock();
		_pDataBoard.m_solidLDNIBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay);
		printf("Build point GL List Time (ms): %ld\n",clock()-time); time=clock();
		printf("--------------------------------------------\n");
		_pGLK.AddDisplayObj(_pDataBoard.m_solidLDNIBody, true);
		printf("Refresh time (ms): %ld\n",clock()-time); time=clock();
		printf("Total sample number: %d\n",_pDataBoard.m_solidLDNIBody->m_cudaSolid->GetSampleNumber());
	}
	else if (_stricmp(exstr,"ldb")==0) {	//.ldb file
		if (!(_pDataBoard.m_solidLDNIBody)) 
			_pDataBoard.m_solidLDNIBody=new LDNISolidBody;
		else
			_pGLK.DelDisplayObj2(_pDataBoard.m_solidLDNIBody);

		long time=clock();	
		LDNIcpuSolid *solid=new LDNIcpuSolid;
		if (solid->FileRead(filename)) 
			{_pDataBoard.m_solidLDNIBody->m_solid=solid;}
		else
			{delete solid;	printf("LDB file importing fails!\n"); 	return;}
		printf("LDB File Import Time (ms): %ld\n",clock()-time); 
		_pDataBoard.m_solidLDNIBody->CompRange();
		time=clock();
		_pDataBoard.m_solidLDNIBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay);
		printf("Build point GL List Time (ms): %ld\n",clock()-time); time=clock();
		printf("--------------------------------------------\n");
		_pGLK.AddDisplayObj(_pDataBoard.m_solidLDNIBody, true);
		printf("Refresh time (ms): %ld\n",clock()-time); time=clock();
	}
	else if (_stricmp(exstr,"ldn")==0) {	//.ldn binary file
	}
	else if (_stricmp(exstr,"obj")==0) {	//.obj file
		if (!(_pDataBoard.m_polyMeshBody)) 
			_pDataBoard.m_polyMeshBody=new PMBody;
		else
			_pGLK.DelDisplayObj2(_pDataBoard.m_polyMeshBody);
		
		long time=clock();	QuadTrglMesh *mesh=new QuadTrglMesh;
		if (mesh->InputOBJFile(filename)) {
			_pDataBoard.m_polyMeshBody->GetMeshList().AddTail(mesh);	
			_pDataBoard.m_polyMeshBody->computeRange();
			mesh->SetMeshId(_pDataBoard.m_polyMeshBody->GetMeshList().GetCount());
			
			printf("OBJ File Import Time (ms): %ld\n",clock()-time); 
		}else 
			{delete mesh;	printf("OBJ file importing fails!\n"); 	return;}
		time=clock();	
		if (_pGLK.GetShading()) _pDataBoard.m_polyMeshBody->BuildGLList(true);
		if (_pGLK.GetMesh()) _pDataBoard.m_polyMeshBody->BuildGLList(false);
		printf("--------------------------------------------\n");
		_pGLK.AddDisplayObj(_pDataBoard.m_polyMeshBody,true);		
		printf("Build GL List Time (ms): %ld\n",clock()-time); time=clock();
	}
	else if (_stricmp(exstr,"stl")==0) {	//.stl file
		if (!(_pDataBoard.m_polyMeshBody)) 
			_pDataBoard.m_polyMeshBody=new PMBody;
		else
			_pGLK.DelDisplayObj2(_pDataBoard.m_polyMeshBody);
		
		long time=clock();	QuadTrglMesh *mesh=new QuadTrglMesh;
		if (mesh->InputSTLFile(filename)) {
			_pDataBoard.m_polyMeshBody->GetMeshList().AddTail(mesh);	
			_pDataBoard.m_polyMeshBody->computeRange();
			mesh->SetMeshId(_pDataBoard.m_polyMeshBody->GetMeshList().GetCount());
			
			printf("STL File Import Time (ms): %ld\n",clock()-time);
		}else 
			{delete mesh;	printf("STL file importing fails!\n"); 	return;}
		time=clock();	
		if (_pGLK.GetShading()) _pDataBoard.m_polyMeshBody->BuildGLList(true);
		if (_pGLK.GetMesh()) _pDataBoard.m_polyMeshBody->BuildGLList(false);
		printf("--------------------------------------------\n");
		_pGLK.AddDisplayObj(_pDataBoard.m_polyMeshBody,true);		
		printf("Build GL List Time (ms): %ld\n",clock()-time); time=clock();
	}
	else if (_stricmp(exstr,"meb")==0) {	//.meb file
		if (!(_pDataBoard.m_polyMeshBody)) 
			_pDataBoard.m_polyMeshBody=new PMBody;
		else
			_pGLK.DelDisplayObj2(_pDataBoard.m_polyMeshBody);
		
		long time=clock();	QuadTrglMesh *mesh=new QuadTrglMesh;
		if (mesh->InputMEBFile(filename)) {
			_pDataBoard.m_polyMeshBody->GetMeshList().AddTail(mesh);	
			_pDataBoard.m_polyMeshBody->computeRange();
			printf("MEB File Import Time (ms): %ld\n",clock()-time); 
		}else 
			{delete mesh;	printf("MEB file importing fails!\n"); 	return;}
		time=clock();	
		if (_pGLK.GetShading()) _pDataBoard.m_polyMeshBody->BuildGLList(true);
		if (_pGLK.GetMesh()) _pDataBoard.m_polyMeshBody->BuildGLList(false);
		printf("Build GL List Time (ms): %ld\n",clock()-time); time=clock();
		printf("--------------------------------------------\n");
		_pGLK.AddDisplayObj(_pDataBoard.m_polyMeshBody,true);
		printf("Refresh time (ms): %ld\n",clock()-time); time=clock();
	}

}

void menuFuncMeshBndBoxComp()
{
	if (!(_pDataBoard.m_polyMeshBody)) {
		printf("None mesh is found!\n");	return;
	}

	float bndBox[6];
	_pDataBoard.m_polyMeshBody->CompBoundingBox(bndBox);
	printf("------------------------------------------------------\n");
	printf("The bounding box: \n(%f,%f) (%f,%f) (%f,%f)\n",bndBox[0],bndBox[1],bndBox[2],bndBox[3],bndBox[4],bndBox[5]);
	printf("Dimensions: (%f,%f,%f)\n",bndBox[1]-bndBox[0],bndBox[3]-bndBox[2],bndBox[5]-bndBox[4]);
	printf("Position of Center: (%f,%f,%f)\n",0.5f*(bndBox[1]+bndBox[0]),0.5f*(bndBox[3]+bndBox[2]),0.5f*(bndBox[5]+bndBox[4]));
	printf("------------------------------------------------------\n");
}

void menuFuncMeshModelFlipping()
{
	if (!(_pDataBoard.m_polyMeshBody)) {
		printf("None mesh is found!\n");	return;
	}

	char buf[255];	bool bFlipped=false;
	printf("Please state the flipping direction (x, y or z):\n");
	scanf("%s",buf);
	
	if (buf[0]=='x' || buf[0]=='X') {
		_pDataBoard.m_polyMeshBody->FlipModel(0);	bFlipped=true;
	}
	else if (buf[0]=='y' || buf[0]=='Y') {
		_pDataBoard.m_polyMeshBody->FlipModel(1);	bFlipped=true;
	}
	else if (buf[0]=='z' || buf[0]=='Z') {
		_pDataBoard.m_polyMeshBody->FlipModel(2);	bFlipped=true;
	}
	if (bFlipped) {
		_pDataBoard.m_polyMeshBody->computeRange();
		if (_pGLK.GetShading()) _pDataBoard.m_polyMeshBody->BuildGLList(true);
		if (_pGLK.GetMesh()) _pDataBoard.m_polyMeshBody->BuildGLList(false);
		_pGLK.DelDisplayObj3(_pDataBoard.m_polyMeshBody);
		_pGLK.AddDisplayObj(_pDataBoard.m_polyMeshBody, true);
	}
}

void menuFuncMeshShiftOrigin()
{
	if (!(_pDataBoard.m_polyMeshBody)) {
		printf("None mesh is found!\n");	return;
	}

	_pDataBoard.m_polyMeshBody->ShiftToOrigin();
	_pDataBoard.m_polyMeshBody->computeRange();
	if (_pGLK.GetShading()) _pDataBoard.m_polyMeshBody->BuildGLList(true);
	if (_pGLK.GetMesh()) _pDataBoard.m_polyMeshBody->BuildGLList(false);
	_pGLK.DelDisplayObj3(_pDataBoard.m_polyMeshBody);
	_pGLK.AddDisplayObj(_pDataBoard.m_polyMeshBody, true);
}

void menuFuncMeshShiftPos()
{
	if (!(_pDataBoard.m_polyMeshBody)) {
		printf("None mesh is found!\n");	return;
	}

	_pDataBoard.m_polyMeshBody->ShiftToPosSystem();
	_pDataBoard.m_polyMeshBody->computeRange();
	if (_pGLK.GetShading()) _pDataBoard.m_polyMeshBody->BuildGLList(true);
	if (_pGLK.GetMesh()) _pDataBoard.m_polyMeshBody->BuildGLList(false);
	_pGLK.DelDisplayObj3(_pDataBoard.m_polyMeshBody);
	_pGLK.AddDisplayObj(_pDataBoard.m_polyMeshBody, true);
}

void menuFuncMeshModelScale()
{	
	if (!(_pDataBoard.m_polyMeshBody)) {
		printf("None mesh is found!\n");	return;
	}

	char buf[255];	float sx,sy,sz;
	printf("Please state the scaling vector in the form of sx,sy,sz:\n");
	scanf("%s",buf);
	sscanf(buf,"%f,%f,%f",&sx,&sy,&sz);
	printf("The input is: %f,%f,%f\n",sx,sy,sz);

	_pDataBoard.m_polyMeshBody->Scaling(sx,sy,sz);
	_pDataBoard.m_polyMeshBody->computeRange();
	if (_pGLK.GetShading()) _pDataBoard.m_polyMeshBody->BuildGLList(true);
	if (_pGLK.GetMesh()) _pDataBoard.m_polyMeshBody->BuildGLList(false);
	_pGLK.DelDisplayObj3(_pDataBoard.m_polyMeshBody);
	_pGLK.AddDisplayObj(_pDataBoard.m_polyMeshBody, true);
}

void menuFuncMeshModelTransformation()
{
	if (!(_pDataBoard.m_polyMeshBody)) {
		printf("None mesh is found!\n");	return;
	}

	char buf[255];	float tx,ty,tz;
	printf("Please state the transformation vector in the form of tx,ty,tz:\n");
	scanf("%s",buf);
	sscanf(buf,"%f,%f,%f",&tx,&ty,&tz);
	printf("The input is: %f,%f,%f\n",tx,ty,tz);

	_pDataBoard.m_polyMeshBody->Transformation(tx,ty,tz);
	_pDataBoard.m_polyMeshBody->computeRange();
	if (_pGLK.GetShading()) _pDataBoard.m_polyMeshBody->BuildGLList(true);
	if (_pGLK.GetMesh()) _pDataBoard.m_polyMeshBody->BuildGLList(false);
	_pGLK.DelDisplayObj3(_pDataBoard.m_polyMeshBody);
	_pGLK.AddDisplayObj(_pDataBoard.m_polyMeshBody, true);
}



void menuFuncLDNISampling(bool bCUDA)
{
	if (!(_pDataBoard.m_polyMeshBody)) {printf("None mesh found!\n"); return;}

	int nRes;	char inputStr[200];
	printf("\nPlease specify the resolution for sampling: ");
	scanf("%s",inputStr);	printf("\n");	sscanf(inputStr,"%d",&nRes);
	if (nRes<=0) {printf("Incorrect InputL: %d!!!\n",nRes); return;}
	printf("Sampling Resolution: %d\n",nRes);

	//-----------------------------------------------------------------------------------
	//	Need to release the memory of displaying a object;
	//		otherwise, the sampling may be abnormal if the graphics memory is not enough
	_pDataBoard.m_polyMeshBody->DeleteGLList(true);
	_pDataBoard.m_polyMeshBody->DeleteGLList(false);
	if (_pDataBoard.m_solidLDNIBody!=NULL) 
		_pDataBoard.m_solidLDNIBody->DeleteVBOGLList();

	LDNIcpuSolid *solid;	
	LDNIcudaSolid *cudaSolid;
	float bndBox[6];
	bndBox[0]=bndBox[1]=bndBox[2]=bndBox[3]=bndBox[4]=bndBox[5]=0;
	QuadTrglMesh *mesh=(QuadTrglMesh *)(_pDataBoard.m_polyMeshBody->GetMeshList().GetHead());
	long time=clock();
	if (bCUDA) {
//		LDNIcpuOperation::BRepToLDNISampling(mesh,solid,bndBox,nRes);
//		LDNIcudaOperation::CopyCPUSolidToCUDASolid(solid,cudaSolid);
//		delete solid;
		LDNIcudaOperation::BRepToLDNISampling(mesh,cudaSolid,bndBox,nRes);
	}
	else {
		LDNIcpuOperation::BRepToLDNISampling(mesh,solid,bndBox,nRes);
	}
	printf("--------------------------------------------\n");
	printf("Total Sampling Time: %ld (ms)\n",clock()-time); time=clock();
	printf("--------------------------------------------\n\n");

	if (_pDataBoard.m_solidLDNIBody) _pGLK.DelDisplayObj(_pDataBoard.m_solidLDNIBody);

	_pDataBoard.m_polyMeshBody->GetMeshList().RemoveHead();		delete mesh;
	if (_pDataBoard.m_polyMeshBody->GetMeshList().IsEmpty()) {
		_pGLK.DelDisplayObj(_pDataBoard.m_polyMeshBody);
		_pDataBoard.m_polyMeshBody=NULL;
	}
	else {
		if (_pGLK.GetShading()) _pDataBoard.m_polyMeshBody->BuildGLList(true);
		if (_pGLK.GetMesh()) _pDataBoard.m_polyMeshBody->BuildGLList(false);
	}

	LDNISolidBody *solidBody=new LDNISolidBody;
	if (bCUDA) 
		solidBody->m_cudaSolid=cudaSolid;
	else 
		solidBody->m_solid=solid; 
	solidBody->CompRange();
	time=clock();
	solidBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay);
	printf("Build point GL List Time (ms): %ld\n",clock()-time); time=clock();
	printf("--------------------------------------------\n");
	_pDataBoard.m_solidLDNIBody=solidBody;
	_pGLK.AddDisplayObj(solidBody, true);
	printf("Refresh time (ms): %ld\n",clock()-time); time=clock();
}

//void menuFuncSLAImageGeneration()
//{
//	if (_pDataBoard.m_solidLDNIBody==NULL || _pDataBoard.m_solidLDNIBody->m_cudaSolid==NULL) {
//		printf("None cuda-LDNI-solid found!\n");	return;
//	}
//
//	if (!(_pDataBoard.m_polyMeshBody)) 
//		_pDataBoard.m_polyMeshBody=new PMBody;
//	else
//		_pGLK.DelDisplayObj2(_pDataBoard.m_polyMeshBody);
//
//	char inputStr[200];
//	int dx, dy;
//	float thickness;
//	float range = 1.0;
//	ContourMesh* cmesh = new ContourMesh();
//	_pDataBoard.m_solidLDNIBody->CompRange();
//	range = _pDataBoard.m_solidLDNIBody->GetRange();
//
//	printf("\nPlease specify the binary image Size (x, y): \n");
//	scanf("%s",inputStr);
//	sscanf(inputStr,"%d,%d",&dx,&dy);
//	printf("Please specify the thickness (t): \n");
//	scanf("%s",inputStr);
//	sscanf(inputStr,"%f",&thickness);
//	printf("The Image Size is : (%d, %d) \n", dx,dy);
//	printf("The thickness is : %f \n", thickness);
//	cmesh->SetThickness(thickness);
//
//	LDNIcudaOperation::LDNISLAContouring_Generation( _pDataBoard.m_solidLDNIBody->m_cudaSolid, cmesh, dx, dy, thickness);
//
//	_pGLK.SetContourThickness(cmesh->GetThickness());
//	_pGLK.SetContourLayer(cmesh->GetResolution(1));
//	_pGLK.TransClipPlane(1,cmesh->GetThickness()*0.5);
//	_pGLK.TransClipPlane(0,cmesh->GetThickness()*0.5);
//	_pDataBoard.m_polyMeshBody->GetMeshList().AddTail(cmesh);
//	_pDataBoard.m_polyMeshBody->setRange(range);
//	_pDataBoard.m_polyMeshBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay,true);
//
//
//	_pGLK.DelDisplayObj2(_pDataBoard.m_solidLDNIBody);	
//	delete (_pDataBoard.m_solidLDNIBody);
//
//	_pDataBoard.m_solidLDNIBody=NULL;
//	_pGLK.SetContour(true);
//	_pGLK.AddDisplayObj(_pDataBoard.m_polyMeshBody, true);
//
//	//printf("Build GL List Time (ms): %ld\n",clock()-time); time=clock();
//
//	_pGLK.refresh();
//
//}

void menuFuncSLAContourSupportGeneration()
{
	if (_pDataBoard.m_solidLDNIBody==NULL || _pDataBoard.m_solidLDNIBody->m_cudaSolid==NULL) {
		printf("None cuda-LDNI-solid found!\n");	return;
	}

	if (!(_pDataBoard.m_polyMeshBody)) 
		_pDataBoard.m_polyMeshBody=new PMBody;
	else
		_pGLK.DelDisplayObj2(_pDataBoard.m_polyMeshBody);




	char inputStr[200];
	float samplewidth = 0.0;
	float range = 1.0;
	float thickness = 0.0;
	float anchorRadius = 0.0;
	float threshold = 0.0;
	float cylinderRadius = 2.5;
	float patternThickness = 5.5;

	ContourMesh* cmesh = new ContourMesh();
	ContourMesh* suptmesh = new ContourMesh();

	bool imgDisplay = false;

	_pDataBoard.m_solidLDNIBody->CompRange();
	range = _pDataBoard.m_solidLDNIBody->GetRange();
	printf("ragne %f \n",range);
	printf("\nPlease specify the binary image sampling width: ");
	scanf("%s",inputStr);	printf("\n");	sscanf(inputStr,"%f",&samplewidth);
	printf("The sampling width is :%f \n", samplewidth);
	printf("\nDisplay in binary image? (otherwise in contour) (y/n): ");
	scanf("%s",inputStr);
	if (inputStr[0]=='y' || inputStr[0]=='Y') imgDisplay=true;
	printf("Thickness? : ");
	scanf("%s",inputStr);	sscanf(inputStr,"%f",&thickness);
	printf("Anchor Radius? : ");
	scanf("%s",inputStr);	sscanf(inputStr,"%f",&anchorRadius);
	printf("Threshold? : ");
	scanf("%s",inputStr);	sscanf(inputStr,"%f",&threshold);
	printf("Cylinder Radius? : ");
	scanf("%s",inputStr);	sscanf(inputStr,"%f",&cylinderRadius);
	printf("Pattern Thickness? : ");
	scanf("%s",inputStr);	sscanf(inputStr,"%f",&patternThickness);

	LDNIcudaOperation::LDNISLAContouring_GenerationwithSupporting( _pDataBoard.m_solidLDNIBody->m_cudaSolid, cmesh, suptmesh, samplewidth, imgDisplay, thickness, anchorRadius, threshold, cylinderRadius, patternThickness);

	_pGLK.SetContourThickness(cmesh->GetThickness());
	_pGLK.SetContourLayer(cmesh->GetResolution(1));
	_pGLK.TransClipPlane(1,cmesh->GetThickness()*0.5);
	_pGLK.TransClipPlane(0,cmesh->GetThickness()*0.5);
	_pDataBoard.m_polyMeshBody->GetMeshList().AddTail(cmesh);
	_pDataBoard.m_polyMeshBody->GetMeshList().AddTail(suptmesh);
	_pDataBoard.m_polyMeshBody->setRange(range);
	_pDataBoard.m_polyMeshBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay,true);


	_pGLK.DelDisplayObj2(_pDataBoard.m_solidLDNIBody);	
	delete (_pDataBoard.m_solidLDNIBody);

	_pDataBoard.m_solidLDNIBody=NULL;
	_pGLK.SetContour(true);
	_pGLK.AddDisplayObj(_pDataBoard.m_polyMeshBody, true);

	//printf("Build GL List Time (ms): %ld\n",clock()-time); time=clock();

	_pGLK.refresh();
}

void menuFuncFDMContourSupportGeneration()
{
	if (_pDataBoard.m_solidLDNIBody==NULL || _pDataBoard.m_solidLDNIBody->m_cudaSolid==NULL) {
		printf("None cuda-LDNI-solid found!\n");	return;
	}

	if (!(_pDataBoard.m_polyMeshBody)) 
		_pDataBoard.m_polyMeshBody=new PMBody;
	else
		_pGLK.DelDisplayObj2(_pDataBoard.m_polyMeshBody);




	char inputStr[200];
	float samplewidth = 0.0;
	float range = 1.0;
	ContourMesh* cmesh = new ContourMesh();
	ContourMesh* suptmesh = new ContourMesh();

	_pDataBoard.m_solidLDNIBody->CompRange();
	range = _pDataBoard.m_solidLDNIBody->GetRange();
	printf("ragne %f \n",range);
	printf("\nPlease specify the binary image sampling width: ");
	scanf("%s",inputStr);	printf("\n");	sscanf(inputStr,"%f",&samplewidth);
	//samplewidth = 0.005;
	printf("The sampling width is :%f \n", samplewidth);

	//LDNIcudaOperation::LDNIFDMContouring_Generation( _pDataBoard.m_solidLDNIBody->m_cudaSolid, cmesh, samplewidth);
	LDNIcudaOperation::LDNIFDMContouring_GenerationwithSupporting( _pDataBoard.m_solidLDNIBody->m_cudaSolid, cmesh, suptmesh, samplewidth);

	_pGLK.SetContourThickness(cmesh->GetThickness());
	_pGLK.SetContourLayer(cmesh->GetResolution(1));
	_pGLK.TransClipPlane(1,cmesh->GetThickness()*0.5);
	_pGLK.TransClipPlane(0,cmesh->GetThickness()*0.5);
	_pDataBoard.m_polyMeshBody->GetMeshList().AddTail(cmesh);
	_pDataBoard.m_polyMeshBody->GetMeshList().AddTail(suptmesh);
	_pDataBoard.m_polyMeshBody->setRange(range);
	_pDataBoard.m_polyMeshBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay,true);


	_pGLK.DelDisplayObj2(_pDataBoard.m_solidLDNIBody);	
	delete (_pDataBoard.m_solidLDNIBody);

	_pDataBoard.m_solidLDNIBody=NULL;
	_pGLK.SetContour(true);
	_pGLK.AddDisplayObj(_pDataBoard.m_polyMeshBody, true);

	//printf("Build GL List Time (ms): %ld\n",clock()-time); time=clock();

	_pGLK.refresh();
}

void menuFuncFDMContourGeneration()
{
	if (_pDataBoard.m_solidLDNIBody==NULL || _pDataBoard.m_solidLDNIBody->m_cudaSolid==NULL) {
		printf("None cuda-LDNI-solid found!\n");	return;
	}

	if (!(_pDataBoard.m_polyMeshBody)) 
		_pDataBoard.m_polyMeshBody=new PMBody;
	else
		_pGLK.DelDisplayObj2(_pDataBoard.m_polyMeshBody);


	

	char inputStr[200];
	float samplewidth = 0.0;
	float range = 1.0;
	ContourMesh* cmesh = new ContourMesh();
	_pDataBoard.m_solidLDNIBody->CompRange();
	//cmesh->setRange(_pDataBoard.m_solidLDNIBody->GetRange());
	range = _pDataBoard.m_solidLDNIBody->GetRange();
	printf("ragne %f \n",range);
	printf("\nPlease specify the binary image sampling width: ");
	scanf("%s",inputStr);	printf("\n");	sscanf(inputStr,"%f",&samplewidth);
	printf("The sampling width is :%f \n", samplewidth);
	
	LDNIcudaOperation::LDNIFDMContouring_Generation( _pDataBoard.m_solidLDNIBody->m_cudaSolid, cmesh, samplewidth);
	
	_pGLK.SetContourThickness(cmesh->GetThickness());
	_pGLK.SetContourLayer(cmesh->GetResolution(1));
	_pGLK.TransClipPlane(1,cmesh->GetThickness()*0.5);
	_pGLK.TransClipPlane(0,cmesh->GetThickness()*0.5);
	 _pDataBoard.m_polyMeshBody->GetMeshList().AddTail(cmesh);
	_pDataBoard.m_polyMeshBody->setRange(range);
	 _pDataBoard.m_polyMeshBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay,true);
	

	_pGLK.DelDisplayObj2(_pDataBoard.m_solidLDNIBody);	
	delete (_pDataBoard.m_solidLDNIBody);
	
	_pDataBoard.m_solidLDNIBody=NULL;
	_pGLK.SetContour(true);
	_pGLK.AddDisplayObj(_pDataBoard.m_polyMeshBody, true);

	//printf("Build GL List Time (ms): %ld\n",clock()-time); time=clock();

	_pGLK.refresh();



}





void menuFuncLDNIOffsetting(bool bSpatialHashingBased, bool bRayPacking, bool bSuccessive, bool bWithNormal)
{
	if (_pDataBoard.m_solidLDNIBody==NULL) {printf("None LDNI-solid found!\n");	return;}
	if (_pDataBoard.m_solidLDNIBody->m_cudaSolid==NULL) {printf("None cuda-LDNI-solid found!\n");	return;}

	char buf[20];	float offset,ww;
	ww=_pDataBoard.m_solidLDNIBody->m_cudaSolid->GetSampleWidth(); 
	printf("Offset (in terms of g-width - %lf) = ?",ww);
	scanf("%s",buf);	printf("%s\n",buf);		sscanf(buf,"%f",&offset);

	long time=clock();
	LDNIcudaSolid* newSolid=NULL;
	if (bSpatialHashingBased) {
		if (bSuccessive)
			LDNIcudaOperation::SolidQuickSuccessiveOffsettingBySpatialHashing(
										_pDataBoard.m_solidLDNIBody->m_cudaSolid, newSolid, ww*offset, bWithNormal);
		else {
			if (bWithNormal)
				LDNIcudaOperation::SolidOffsettingBySpatialHashing(
										_pDataBoard.m_solidLDNIBody->m_cudaSolid, newSolid, ww*offset, bRayPacking);
			else
				LDNIcudaOperation::SolidOffsettingWithoutNormal(
										_pDataBoard.m_solidLDNIBody->m_cudaSolid, newSolid, ww*offset, bRayPacking);
		}
	}
	else {
		LDNIcudaOperation::SolidOffsetting(_pDataBoard.m_solidLDNIBody->m_cudaSolid, newSolid, ww*offset);
	}
	printf("---------------------------------------------------------------\n");
	printf("Processing Time for Solid Offsetting is: %ld (ms)\n",clock()-time);	time=clock();
	if (newSolid->GetSampleNumber()==0) {delete newSolid;	newSolid=NULL;}
	if (newSolid!=NULL) 
		{delete (_pDataBoard.m_solidLDNIBody->m_cudaSolid);_pDataBoard.m_solidLDNIBody->m_cudaSolid=newSolid;}
	else 
		{delete (_pDataBoard.m_solidLDNIBody->m_cudaSolid);_pDataBoard.m_solidLDNIBody->m_cudaSolid=NULL;}
	
//	time=clock();
	_pDataBoard.m_solidLDNIBody->CompRange();
	_pDataBoard.m_solidLDNIBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay);
//	printf("Build point GL List Time (ms): %ld\n",clock()-time); time=clock();
	_pGLK.DelDisplayObj2(_pDataBoard.m_solidLDNIBody);
	_pGLK.AddDisplayObj(_pDataBoard.m_solidLDNIBody, true);
}

void menuFuncLDNIContouring()
{
	if (_pDataBoard.m_solidLDNIBody==NULL || _pDataBoard.m_solidLDNIBody->m_cudaSolid==NULL) {
		printf("None cuda-LDNI-solid found!\n");	return;
	}

	if (!(_pDataBoard.m_polyMeshBody)) 
		_pDataBoard.m_polyMeshBody=new PMBody;
	else
		_pGLK.DelDisplayObj2(_pDataBoard.m_polyMeshBody);

	int nRes=600;	char inputStr[200];
	printf("\nThe resolution of LDNI solid is: %d\nPlease specify the resolution for contouring: ",
			_pDataBoard.m_solidLDNIBody->m_cudaSolid->GetResolution());
	scanf("%s",inputStr);	printf("\n");	sscanf(inputStr,"%d",&nRes);
	if (nRes<=0) {printf("Incorrect input resolution: %d!!!\n",nRes); return;}
//	if (nRes>512) {printf("Too large resolution ( larger than 512 ), which cannot fit into the memory of graphics card!!!\n",nRes); return;}
	printf("Sampling Resolution: %d\n",nRes);

	bool bWithSelfIntersectionPreventation=false;
	char answer[20];
	printf( "Do you wish to turn on the function of self-intersection preventation, which may lead to poorer surface quality? (y/n)\n");	
	scanf("%s",answer);
	if (answer[0]=='y' || answer[0]=='Y') bWithSelfIntersectionPreventation=true;

	long time=clock();	QuadTrglMesh *mesh;
	LDNIcudaOperation::LDNIToBRepReconstruction(_pDataBoard.m_solidLDNIBody->m_cudaSolid,mesh,nRes,bWithSelfIntersectionPreventation);
	printf("--------------------------------------------\n");
	printf("Total time of mesh contouring: %ld (ms)\n",clock()-time); time=clock();
	_pDataBoard.m_polyMeshBody->GetMeshList().AddTail(mesh);	
	_pDataBoard.m_polyMeshBody->computeRange();
	if (_pGLK.GetShading()) _pDataBoard.m_polyMeshBody->BuildGLList(true);
	if (_pGLK.GetMesh()) _pDataBoard.m_polyMeshBody->BuildGLList(false);
	_pGLK.AddDisplayObj(_pDataBoard.m_polyMeshBody);
	_pGLK.DelDisplayObj2(_pDataBoard.m_solidLDNIBody);	
	delete (_pDataBoard.m_solidLDNIBody);
	_pDataBoard.m_solidLDNIBody=NULL;
	printf("Build GL List Time (ms): %ld\n",clock()-time); time=clock();

	_pGLK.refresh();
}

void menuFuncLDNIregularization()
{
	if (_pDataBoard.m_solidLDNIBody==NULL || _pDataBoard.m_solidLDNIBody->m_cudaSolid==NULL) {
		printf("None cuda-LDNI-solid found!\n");	return;
	}

	long time=clock();
	LDNIcudaOperation::SolidRegularization(_pDataBoard.m_solidLDNIBody->m_cudaSolid);

	_pDataBoard.m_solidLDNIBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay);
	_pGLK.refresh();
}

void menuFuncLDNINormalRecon(bool bWithOrtVoting)
{
	if (_pDataBoard.m_solidLDNIBody==NULL || _pDataBoard.m_solidLDNIBody->m_cudaSolid==NULL) {
		printf("None cuda-LDNI-solid found!\n");	return;
	}

	long time=clock();
	LDNIcudaOperation::OrientedNormalReconstruction((_pDataBoard.m_solidLDNIBody->m_cudaSolid), 3, bWithOrtVoting);
	printf("------------------------------------------------\n");
	printf("The reconstruction takes: %ld (ms)\n\n",clock()-time);

	_pDataBoard.m_solidLDNIBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay);
	_pGLK.refresh();
}

void menuFuncLDNINormalBilateralFiltering(bool bCUDA)
{
	if (_pDataBoard.m_solidLDNIBody==NULL) {
		printf("None LDNI-solid found!\n");	return;
	}
	if (bCUDA && _pDataBoard.m_solidLDNIBody->m_cudaSolid==NULL) {
		printf("None cuda-LDNI-solid found!\n");	return;
	}
	if ((!bCUDA) && _pDataBoard.m_solidLDNIBody->m_solid==NULL) {
		printf("None cpu-LDNI-solid found!\n");	return;
	}

	long time=clock();
	if (bCUDA) 
		LDNIcudaOperation::ParallelProcessingNormalVector((_pDataBoard.m_solidLDNIBody->m_cudaSolid), 3);
	else
		LDNIcpuOperation::ParallelProcessingNormalVector((_pDataBoard.m_solidLDNIBody->m_solid), 3);
	printf("------------------------------------------------\n");
	printf("The filtering takes: %ld (ms)\n\n",clock()-time);

	_pDataBoard.m_solidLDNIBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay);
	_pGLK.refresh();
}

void menuFuncLDNISuperUnion()
{
	if (!(_pDataBoard.m_polyMeshBody)) {printf("None mesh found!\n"); return;}

	int nRes;	char inputStr[200];
	printf("\nPlease specify the resolution for sampling: ");
	scanf("%s",inputStr);	printf("\n");	sscanf(inputStr,"%d",&nRes);
	if (nRes<=0) {printf("Incorrect InputL: %d!!!\n",nRes); return;}
	printf("Sampling Resolution: %d\n",nRes);

	//-----------------------------------------------------------------------------------
	//	Need to release the memory of displaying a object;
	//		otherwise, the sampling may be abnormal if the graphics memory is not enough
	_pDataBoard.m_polyMeshBody->DeleteGLList(true);
	_pDataBoard.m_polyMeshBody->DeleteGLList(false);
	if (_pDataBoard.m_solidLDNIBody!=NULL) 
		_pDataBoard.m_solidLDNIBody->DeleteVBOGLList();

	
	LDNIcudaSolid *cudaSolid;
	float bndBox[6];
	bndBox[0]=bndBox[1]=bndBox[2]=bndBox[3]=bndBox[4]=bndBox[5]=0;
	_pDataBoard.m_polyMeshBody->CompBoundingBox(bndBox);

	long time=clock();

	
	LDNIcudaOperation::SuperUnionOperation(cudaSolid,&_pDataBoard.m_polyMeshBody->GetMeshList(),bndBox,nRes);

	printf("--------------------------------------------\n");
	printf("Total Sampling Time: %ld (ms)\n",clock()-time); time=clock();
	printf("--------------------------------------------\n\n");

	if (_pDataBoard.m_solidLDNIBody) _pGLK.DelDisplayObj(_pDataBoard.m_solidLDNIBody);

	_pDataBoard.m_polyMeshBody->GetMeshList().RemoveAll();
	if (_pDataBoard.m_polyMeshBody->GetMeshList().IsEmpty()) {
		_pGLK.DelDisplayObj(_pDataBoard.m_polyMeshBody);
		_pDataBoard.m_polyMeshBody=NULL;
	}
	else {
		if (_pGLK.GetShading()) _pDataBoard.m_polyMeshBody->BuildGLList(true);
		if (_pGLK.GetMesh()) _pDataBoard.m_polyMeshBody->BuildGLList(false);
	}

	LDNISolidBody *solidBody=new LDNISolidBody;
	solidBody->m_cudaSolid=cudaSolid;
	solidBody->CompRange();
	time=clock();
	solidBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay);
	printf("Build point GL List Time (ms): %ld\n",clock()-time); time=clock();
	printf("--------------------------------------------\n");
	_pDataBoard.m_solidLDNIBody=solidBody;
	_pGLK.AddDisplayObj(solidBody, true);
	printf("Refresh time (ms): %ld\n",clock()-time); time=clock();
}


void menuFuncLDNIScaffoldConstruction(bool bCUDA)
{
	if (!(_pDataBoard.m_polyMeshBody)) {printf("None mesh found!\n"); return;}

	int nRes;	char inputStr[200];
	int dupNum[3];	float offval[3];	int nFlip[3];
	printf("\nPlease specify the resolution: ");
	scanf("%s",inputStr);	printf("\n");	sscanf(inputStr,"%d",&nRes);
	if (nRes<=0) {printf("Incorrect Input: %d!!!\n",nRes); return;}
	printf("\nPlease specify the repeat number (x y z): ");
	scanf("%d %d %d",&dupNum[0],&dupNum[1],&dupNum[2]);	printf("\n");	printf("Resolution: %d\n",nRes);//sscanf(inputStr,"%d %d %d",&dupNum[0],&dupNum[1],&dupNum[2]);
	if (dupNum[0]<=0 || dupNum[1]<=0 || dupNum[2]<=0) {printf("Incorrect Input: %d %d %d!!!\n",dupNum[0],dupNum[1],dupNum[2]); return;}
	printf("\nPlease specify the offset value (x y z): ");
	scanf("%f %f %f",&offval[0],&offval[1],&offval[2]);	printf("\n"); printf("Resolution: %d\n",nRes);//	sscanf(inputStr,"%f %f %f",&offval[0],&offval[1],&offval[2]);
	printf("\nPlease specify which directions (x y z) want to flip (1/0): ");
	scanf("%d %d %d",&nFlip[0],&nFlip[1],&nFlip[2]);	printf("\n"); printf("Resolution: %d\n",nRes);
	printf("Resolution: %d\n",nRes);
	printf("Repeated Unit: %d %d %d\n",dupNum[0],dupNum[1],dupNum[2]);
	printf("Unit Offset Value: %f %f %f\n",offval[0],offval[1],offval[2]);
	printf("Directions Flip (x y z): %d %d %d\n",nFlip[0],nFlip[1],nFlip[2]);


		
	_pDataBoard.m_polyMeshBody->DeleteGLList(true);
	_pDataBoard.m_polyMeshBody->DeleteGLList(false);
	if (_pDataBoard.m_solidLDNIBody!=NULL) 
		_pDataBoard.m_solidLDNIBody->DeleteVBOGLList();


	
	LDNIcudaSolid *cudaSolid=NULL,*newCudaSolid;
	float bndBox[6];
	bndBox[0]=bndBox[1]=bndBox[2]=bndBox[3]=bndBox[4]=bndBox[5]=0;
	QuadTrglMesh *mesh=(QuadTrglMesh *)(_pDataBoard.m_polyMeshBody->GetMeshList().GetHead());
	
	long time=clock();
	
	LDNIcudaOperation::ScaffoldBooleanOperation(cudaSolid,mesh,dupNum,offval,nFlip,nRes,NULL);
	newCudaSolid=cudaSolid;
	printf("Total Sample Number %d \n", cudaSolid->GetSampleNumber());
	_pDataBoard.m_polyMeshBody->GetMeshList().RemoveHead();
	delete mesh;
	if (_pDataBoard.m_polyMeshBody->GetMeshList().IsEmpty()) {
		_pGLK.DelDisplayObj(_pDataBoard.m_polyMeshBody);
		_pDataBoard.m_polyMeshBody=NULL;
	}
	else {
		if (_pGLK.GetShading()) _pDataBoard.m_polyMeshBody->BuildGLList(true);
		if (_pGLK.GetMesh()) _pDataBoard.m_polyMeshBody->BuildGLList(false);
	}
	
	bool bLight=false;	
	if (_pDataBoard.m_solidLDNIBody!=NULL) {
		bLight=_pDataBoard.m_solidLDNIBody->GetLighting();
		if (bCUDA)
			_pDataBoard.m_solidLDNIBody->m_cudaSolid=NULL;
		else
			_pDataBoard.m_solidLDNIBody->m_solid=NULL;
		_pGLK.DelDisplayObj(_pDataBoard.m_solidLDNIBody);
	}
	LDNISolidBody *solidBody=new LDNISolidBody;
	if (bCUDA) {solidBody->m_cudaSolid=newCudaSolid;} //else {solidBody->m_solid=newSolid;}
	solidBody->SetLighting(bLight);	solidBody->CompRange();
	time=clock();
	solidBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay);
	printf("Build point GL List Time (ms): %ld\n",clock()-time); time=clock();
	_pGLK.AddDisplayObj(solidBody, true);
	_pDataBoard.m_solidLDNIBody=solidBody;

}



void menuFuncLDNIBoolean(bool bCUDA)
{
	if (!(_pDataBoard.m_polyMeshBody)) {
		printf("None mesh found!\n");	return;
	}
	if (_pDataBoard.m_polyMeshBody->GetMeshList().GetCount()<2) {
		if (!(_pDataBoard.m_solidLDNIBody)) {
			printf("Less than two objects!\n");	return;
		}
		else {
			if (bCUDA && _pDataBoard.m_solidLDNIBody->m_cudaSolid==NULL) 
				{printf("No CUDA solid is found!\n");	return;}
			if ((!bCUDA) && _pDataBoard.m_solidLDNIBody->m_solid==NULL) 
				{printf("No CPU solid is found!\n");	return;}
		}
	}

	char answer[10];
	printf( "Choose Operation: (U) union, (I) intersection, (D) difference or (R) inversed-difference\n");
	scanf("%s",answer);
	QuadTrglMesh *meshA=(QuadTrglMesh *)(_pDataBoard.m_polyMeshBody->GetMeshList().GetHead());
	QuadTrglMesh *meshB=(QuadTrglMesh *)(_pDataBoard.m_polyMeshBody->GetMeshList().GetTail());

	LDNIcpuSolid *solid=NULL,*newSolid;
	LDNIcudaSolid *cudaSolid=NULL,*newCudaSolid;
	if (bCUDA) {
		if ((_pDataBoard.m_solidLDNIBody!=NULL) && (_pDataBoard.m_solidLDNIBody->m_cudaSolid!=NULL)) 
			cudaSolid=_pDataBoard.m_solidLDNIBody->m_cudaSolid;
	}
	else {
		if ((_pDataBoard.m_solidLDNIBody!=NULL) && (_pDataBoard.m_solidLDNIBody->m_solid!=NULL)) 
			solid=_pDataBoard.m_solidLDNIBody->m_solid;
	}

	int nRes;	char inputStr[200];
	if (solid==NULL && cudaSolid==NULL) {
		printf("\nPlease specify the resolution for sampling: ");
		scanf("%s",inputStr);	printf("\n");	sscanf(inputStr,"%d",&nRes);
		if (nRes<=0) {printf("Incorrect input resolution: %d!!!\n",nRes); return;}
		printf("Sampling Resolution: %d\n",nRes);
	}

	short nOperatorType;
	if (answer[0]=='u' || answer[0]=='U') // union
		nOperatorType=0;	
	else if (answer[0]=='i' || answer[0]=='I') // intersection
		nOperatorType=1;
	else if (answer[0]=='d' || answer[0]=='D') // difference
		nOperatorType=2;
	else if (answer[0]=='r' || answer[0]=='R') // inversed-difference
		nOperatorType=3;
	else
		{printf("Warning: Incorrect Input!!!\n"); return;}

	//-----------------------------------------------------------------------------------
	//	Need to release the memory of displaying a object;
	//		otherwise, the sampling may be abnormal if the graphics memory is not enough
	_pDataBoard.m_polyMeshBody->DeleteGLList(true);
	_pDataBoard.m_polyMeshBody->DeleteGLList(false);
	if (_pDataBoard.m_solidLDNIBody!=NULL) 
		_pDataBoard.m_solidLDNIBody->DeleteVBOGLList();

	long time=clock();
	if (bCUDA) {
		if (cudaSolid!=NULL) 
			{LDNIcudaOperation::BooleanOperation(cudaSolid,meshB,nOperatorType);	newCudaSolid=cudaSolid;	}
		else {
			LDNIcudaOperation::BooleanOperation(meshA,meshB,nRes,nOperatorType,newCudaSolid);
		}
	}
	else {
		if (solid!=NULL) 
			{LDNIcpuOperation::BooleanOperation(solid,meshB,nOperatorType);		newSolid=solid;	}
		else {
			LDNIcpuOperation::BooleanOperation(meshA,meshB,nRes,nOperatorType,newSolid);
		}
	}
	printf("--------------------------------------------\n");
	printf("Total Processing Time: %ld (ms)\n",clock()-time); time=clock();
	printf("--------------------------------------------\n\n");

	if (meshA!=meshB) {
		_pDataBoard.m_polyMeshBody->GetMeshList().RemoveHead();	
		_pDataBoard.m_polyMeshBody->GetMeshList().RemoveTail();
		delete meshA;	delete meshB;
	}
	else {	// meshA==meshB
		_pDataBoard.m_polyMeshBody->GetMeshList().RemoveHead();
		delete meshA;
	}
	if (_pDataBoard.m_polyMeshBody->GetMeshList().IsEmpty()) {
		_pGLK.DelDisplayObj(_pDataBoard.m_polyMeshBody);
		_pDataBoard.m_polyMeshBody=NULL;
	}
	else {
		if (_pGLK.GetShading()) _pDataBoard.m_polyMeshBody->BuildGLList(true);
		if (_pGLK.GetMesh()) _pDataBoard.m_polyMeshBody->BuildGLList(false);
	}

	bool bLight=false;	
	if (_pDataBoard.m_solidLDNIBody!=NULL) {
		bLight=_pDataBoard.m_solidLDNIBody->GetLighting();
		if (bCUDA)
			_pDataBoard.m_solidLDNIBody->m_cudaSolid=NULL;
		else
			_pDataBoard.m_solidLDNIBody->m_solid=NULL;
		_pGLK.DelDisplayObj(_pDataBoard.m_solidLDNIBody);
	}
	LDNISolidBody *solidBody=new LDNISolidBody;
	if (bCUDA) {solidBody->m_cudaSolid=newCudaSolid;} else {solidBody->m_solid=newSolid;}
	solidBody->SetLighting(bLight);	solidBody->CompRange();
	time=clock();
	solidBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay);
	printf("Build point GL List Time (ms): %ld\n",clock()-time); time=clock();
	_pGLK.AddDisplayObj(solidBody, true);
	_pDataBoard.m_solidLDNIBody=solidBody;
}

void menuFuncLDNIandCudaConversion(bool bCPUtoCUDA) 
{
	if (_pDataBoard.m_solidLDNIBody==NULL) {
		printf("None LDNI-solid found!\n");	return;
	}

	if (bCPUtoCUDA) {
		if (_pDataBoard.m_solidLDNIBody->m_solid==NULL) {printf("None cpu-LDNI-solid found!\n"); return;}
		LDNIcpuSolid *cpuSolid=_pDataBoard.m_solidLDNIBody->m_solid;
		LDNIcudaSolid *cudaSolid;
		LDNIcudaOperation::CopyCPUSolidToCUDASolid(cpuSolid,cudaSolid);
		_pDataBoard.m_solidLDNIBody->m_solid=NULL;	delete cpuSolid;
		_pDataBoard.m_solidLDNIBody->m_cudaSolid=cudaSolid;
	}
	else {
		if (_pDataBoard.m_solidLDNIBody->m_cudaSolid==NULL) {printf("None cuda-LDNI-solid found!\n"); return;}
		LDNIcpuSolid *cpuSolid;
		LDNIcudaSolid *cudaSolid=_pDataBoard.m_solidLDNIBody->m_cudaSolid;
		LDNIcudaOperation::CopyCUDASolidToCPUSolid(cudaSolid,cpuSolid);
		_pDataBoard.m_solidLDNIBody->m_cudaSolid=NULL;	delete cudaSolid;
		_pDataBoard.m_solidLDNIBody->m_solid=cpuSolid;
	}

	_pDataBoard.m_solidLDNIBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay);
	_pGLK.refresh();
}



void menuEvent(int idCommand)
{
	switch(idCommand) {
	case _MENU_QUIT:menuFuncQuit();
		break;

	//--------------------------------------------------------------------
	//	File related
	case _MENU_FILE_OPEN:menuFuncFileOpen();
		break;
	case _MENU_FILE_SAVE:menuFuncFileSave();
		break;
	case _MENU_FILE_OPENALL: menuFuncFileOpenAll();
		break;

	//--------------------------------------------------------------------
	//	View related
	case _MENU_VIEW_CLIP: _pGLK.SetClipView();
		break;
	case _MENU_VIEW_ISOMETRIC:_pGLK.SetViewDirection(VD_ISOMETRICVIEW);
		break;
	case _MENU_VIEW_FRONT:_pGLK.SetViewDirection(VD_FRONTVIEW);
		break;
	case _MENU_VIEW_BACK:_pGLK.SetViewDirection(VD_BACKVIEW);
		break;
	case _MENU_VIEW_TOP:_pGLK.SetViewDirection(VD_TOPVIEW);
		break;
	case _MENU_VIEW_BOTTOM:_pGLK.SetViewDirection(VD_BOTTOMVIEW);
		break;
	case _MENU_VIEW_LEFT:_pGLK.SetViewDirection(VD_LEFTVIEW);
		break;
	case _MENU_VIEW_RIGHT:_pGLK.SetViewDirection(VD_RIGHTVIEW);
		break;
	case _MENU_VIEW_ORBITPAN:{
								GLKCameraTool *myTool=new GLKCameraTool(&_pGLK,ORBITPAN);
								_pGLK.clear_tools();
								_pGLK.set_tool(myTool);
							 }break;
	case _MENU_VIEW_ZOOMWINDOW:{
								GLKCameraTool *myTool=new GLKCameraTool(&_pGLK,ZOOMWINDOW);
								_pGLK.clear_tools();
								_pGLK.set_tool(myTool);
							   }break;
	case _MENU_VIEW_ZOOMIN:_pGLK.zoom(1.5);
		break;
	case _MENU_VIEW_ZOOMOUT:_pGLK.zoom(0.75);
		break;
	case _MENU_VIEW_ZOOMALL:_pGLK.zoom_all_in_view();
		break;
	case _MENU_VIEW_PROFILE:{_pGLK.SetProfile(!(_pGLK.GetProfile())); _pGLK.refresh();
							}break;
	case _MENU_VIEW_SHADE:{
		_pGLK.SetShading(!(_pGLK.GetShading())); 
		if (_pGLK.GetShading()) {
			if (_pDataBoard.m_polyMeshBody!=NULL) _pDataBoard.m_polyMeshBody->BuildGLList(true);
		}
		_pGLK.refresh();
							}break;
	case _MENU_VIEW_MESH:{
		_pGLK.SetMesh(!(_pGLK.GetMesh())); 
		if (_pGLK.GetMesh()) {
			if (_pDataBoard.m_polyMeshBody!=NULL) {
				long time=clock();
				_pDataBoard.m_polyMeshBody->BuildGLList(false);
				printf("Build GL List Time (ms): %ld\n",clock()-time); time=clock();
			}
		}
		_pGLK.refresh();
							}break;
	case _MENU_VIEW_AXIS:{_pGLK.SetAxisDisplay(!(_pGLK.GetAxisDisplay())); _pGLK.refresh();
							}break;
	case _MENU_VIEW_COORD:{_pGLK.m_bCoordDisp=!(_pGLK.m_bCoordDisp); _pGLK.refresh();
							}break;
	case _MENU_VIEW_LDNINORMAL:{
		_pDataBoard.m_bLDNISampleNormalDisplay=!(_pDataBoard.m_bLDNISampleNormalDisplay);
		if (_pDataBoard.m_solidLDNIBody) {
			_pDataBoard.m_solidLDNIBody->BuildGLList(_pDataBoard.m_bLDNISampleNormalDisplay);
			_pGLK.refresh();
		}
							}break;
	case _MENU_VIEW_LDNILIGHTING:{
		if (_pDataBoard.m_solidLDNIBody) {
			bool bLight=_pDataBoard.m_solidLDNIBody->GetLighting();
			_pDataBoard.m_solidLDNIBody->SetLighting(!bLight);	_pGLK.refresh();
		}
							}break;

	//--------------------------------------------------------------------
	//	Mesh related
	case _MENU_MESH_BNDBOXCOMP:menuFuncMeshBndBoxComp();
		break;
	case _MENU_MESH_MODELFLIPPING:menuFuncMeshModelFlipping();
		break;
	case _MENU_MESH_MODELTRANSF:menuFuncMeshModelTransformation();
		break;
	case _MENU_MESH_MODELSCALE:menuFuncMeshModelScale();
		break;
	case _MENU_MESH_SHIFTORIGIN:menuFuncMeshShiftOrigin();
		break;
	case _MENU_MESH_SHIFTPOS:menuFuncMeshShiftPos();
		break;
	case _MENU_MESH_CLEARALL:{	
		char answer[20];
		printf( "Warning: are you sure that you are going to clear all mesh objects? (y/n)\n");	scanf("%s",answer);
		if (answer[0]=='y' || answer[0]=='Y') {_pGLK.DelDisplayObj(_pDataBoard.m_polyMeshBody);	_pDataBoard.m_polyMeshBody=NULL;}
		}break;

	//--------------------------------------------------------------------
	//	LDNI related
	case _MENU_LDNI_SAMPLINGFROMBREP:menuFuncLDNISampling(false);
		break;
	case _MENU_LDNI_BOOLEANOPERATION:menuFuncLDNIBoolean(false);
		break;
	case _MENU_CUDA_CONTOURING:menuFuncLDNIContouring();
		break;
	case _MENU_LDNI_BILATERAL_NORMALFILTER:menuFuncLDNINormalBilateralFiltering(false);
		break;
	case _MENU_LDNI_CONVERT2CUDASOLID:menuFuncLDNIandCudaConversion(true);
		break;
	//--------------------------------------------------------------------
	case _MENU_CUDA_SAMPLINGFROMBREP:menuFuncLDNISampling(true);
		break;
	case _MENU_CUDA_BOOLEANOPERATION:menuFuncLDNIBoolean(true);
		break;
	case _MENU_CUDA_SCAFFOLDOPERATION: menuFuncLDNIScaffoldConstruction(true);
		break;
	case _MENU_CUDA_SUPERUNION: menuFuncLDNISuperUnion();
		break;
	case _MENU_CUDA_REGULARIZATION:menuFuncLDNIregularization();
		break;
	case _MENU_CUDA_OFFSETTING:menuFuncLDNIOffsetting(false, false, false, false);
		break;
	case _MENU_CUDA_OFFSETTING_SH:menuFuncLDNIOffsetting(true, false, false, false);
		break;
	case _MENU_CUDA_OFFSETTING_SHRP:menuFuncLDNIOffsetting(true, true, false, false);
		break;
	case _MENU_CUDA_OFFSETTING_SH_SUCC:menuFuncLDNIOffsetting(true, false, true, false);
		break;
	case _MENU_CUDA_OFFSETTING_SH_SUCC_NORMAL:menuFuncLDNIOffsetting(true, false, true, true);
		break;
	case _MENU_CUDA_FDMCONTOUR: menuFuncFDMContourGeneration();
		break;
	case _MENU_CUDA_FDMCONTOURSUPT: menuFuncFDMContourSupportGeneration();
		break;
	case _MENU_CUDA_SLACONTOURSUPT: menuFuncSLAContourSupportGeneration();
		break;	
	case _MENU_CUDA_ORIENTEDNORMALRECON:menuFuncLDNINormalRecon(false);
		break;
	case _MENU_CUDA_ORIENTEDNORMALRECON2:menuFuncLDNINormalRecon(true);
		break;
	case _MENU_CUDA_BILATERAL_NORMALFILTER:menuFuncLDNINormalBilateralFiltering(true);
		break;
	case _MENU_CUDA_CONVERT2CPUSOLID:menuFuncLDNIandCudaConversion(false);
		break;
	case _MENU_CUDA_PROPERTY:{LDNIcudaOperation::GetCudaDeviceProperty();}
		break;
	
	}
}

int buildPopupMenu (void)
{
	int mainMenu,fileSubMenu,viewSubMenu,ldniSubMenu,meshSubMenu;

	fileSubMenu = glutCreateMenu(menuEvent);
	glutAddMenuEntry("Open\tCtrl+O", _MENU_FILE_OPEN);
	glutAddMenuEntry("Save\tCtrl+S", _MENU_FILE_SAVE);
	glutAddMenuEntry("Open Folder",_MENU_FILE_OPENALL);

	viewSubMenu = glutCreateMenu(menuEvent);
	glutAddMenuEntry("Isometric", _MENU_VIEW_ISOMETRIC);
	glutAddMenuEntry("Front", _MENU_VIEW_FRONT);
	glutAddMenuEntry("Back", _MENU_VIEW_BACK);
	glutAddMenuEntry("Top", _MENU_VIEW_TOP);
	glutAddMenuEntry("Bottom", _MENU_VIEW_BOTTOM);
	glutAddMenuEntry("Left", _MENU_VIEW_LEFT);
	glutAddMenuEntry("Right", _MENU_VIEW_RIGHT);
	glutAddMenuEntry("----",-1);
	glutAddMenuEntry("Orbot and Pan\tCtrl+R",_MENU_VIEW_ORBITPAN);
	glutAddMenuEntry("Zoom Window\tCtrl+W",_MENU_VIEW_ZOOMWINDOW);
	glutAddMenuEntry("Zoom In",_MENU_VIEW_ZOOMIN);
	glutAddMenuEntry("Zoom Out",_MENU_VIEW_ZOOMOUT);
	glutAddMenuEntry("Zoom All\tCtrl+A",_MENU_VIEW_ZOOMALL);
	glutAddMenuEntry("Clip View",_MENU_VIEW_CLIP);
	glutAddMenuEntry("----",-1);
	glutAddMenuEntry("Profile",_MENU_VIEW_PROFILE);
	glutAddMenuEntry("Shade",_MENU_VIEW_SHADE);
	glutAddMenuEntry("Mesh",_MENU_VIEW_MESH);
	glutAddMenuEntry("----",-1);
	glutAddMenuEntry("Axis Frame",_MENU_VIEW_AXIS);
	glutAddMenuEntry("Coordinate",_MENU_VIEW_COORD);
	glutAddMenuEntry("----",-1);
	glutAddMenuEntry("LDNI Sample Normal",_MENU_VIEW_LDNINORMAL);
	glutAddMenuEntry("LDNI Shading with Light\tCtrl+L",_MENU_VIEW_LDNILIGHTING);

	ldniSubMenu = glutCreateMenu(menuEvent);

	meshSubMenu = glutCreateMenu(menuEvent);
	glutAddMenuEntry("Bounding-Box Computation",_MENU_MESH_BNDBOXCOMP);
	glutAddMenuEntry("Model Flipping",_MENU_MESH_MODELFLIPPING);
	glutAddMenuEntry("Model Tranformation",_MENU_MESH_MODELTRANSF);
	glutAddMenuEntry("Model Scaling",_MENU_MESH_MODELSCALE);
	glutAddMenuEntry("----",-1);
	glutAddMenuEntry("Shift To Origin",_MENU_MESH_SHIFTORIGIN);
	glutAddMenuEntry("Shift To Positive Coordinate System",_MENU_MESH_SHIFTPOS);
	glutAddMenuEntry("Clear All",_MENU_MESH_CLEARALL);

	ldniSubMenu = glutCreateMenu(menuEvent);
	glutAddMenuEntry("Sampling (from B-rep)", _MENU_LDNI_SAMPLINGFROMBREP);
	glutAddMenuEntry("Boolean operations", _MENU_LDNI_BOOLEANOPERATION);
	glutAddMenuEntry("Bilateral filtering on Normal", _MENU_LDNI_BILATERAL_NORMALFILTER);
	glutAddMenuEntry("----",-1);
	glutAddMenuEntry("CUDA sampling (from B-rep)", _MENU_CUDA_SAMPLINGFROMBREP);
	glutAddMenuEntry("CUDA Boolean operations", _MENU_CUDA_BOOLEANOPERATION);
	glutAddMenuEntry("CUDA regularization",_MENU_CUDA_REGULARIZATION);
	glutAddMenuEntry("----",-1);
	glutAddMenuEntry("CUDA Offsetting", _MENU_CUDA_OFFSETTING);
	glutAddMenuEntry("CUDA Offsetting (by Spatial Hashing)", _MENU_CUDA_OFFSETTING_SH);
	glutAddMenuEntry("CUDA Offsetting (by Hashing + Packing)", _MENU_CUDA_OFFSETTING_SHRP);
	glutAddMenuEntry("CUDA Scaffold Making", _MENU_CUDA_SCAFFOLDOPERATION);
	glutAddMenuEntry("CUDA Successive Offsetting", _MENU_CUDA_OFFSETTING_SH_SUCC);
	glutAddMenuEntry("CUDA Successive Offsetting (with Normal)", _MENU_CUDA_OFFSETTING_SH_SUCC_NORMAL);
	glutAddMenuEntry("CUDA Super Union",_MENU_CUDA_SUPERUNION);
	glutAddMenuEntry("----",-1);
	glutAddMenuEntry("CUDA Normal Reconstruction", _MENU_CUDA_ORIENTEDNORMALRECON);
	glutAddMenuEntry("CUDA Normal Reconstruction + Voting", _MENU_CUDA_ORIENTEDNORMALRECON2);
	glutAddMenuEntry("CUDA Bilateral filtering on Normal", _MENU_CUDA_BILATERAL_NORMALFILTER);
	glutAddMenuEntry("----",-1);
	glutAddMenuEntry("Convert CUDA solid to mesh", _MENU_CUDA_CONTOURING);
	glutAddMenuEntry("----",-1);
	glutAddMenuEntry("Convert CPU solid to CUDA", _MENU_LDNI_CONVERT2CUDASOLID);
	glutAddMenuEntry("Convert CUDA solid to CPU", _MENU_CUDA_CONVERT2CPUSOLID);
	glutAddMenuEntry("----",-1);
	glutAddMenuEntry("Generate Contour on CUDA (FDM)", _MENU_CUDA_FDMCONTOUR);
	glutAddMenuEntry("Generate Contour and Support on CUDA (FDM)", _MENU_CUDA_FDMCONTOURSUPT);	
	glutAddMenuEntry("Generate Contour and Support on CUDA (SLA)", _MENU_CUDA_SLACONTOURSUPT);
	glutAddMenuEntry("----",-1);
	glutAddMenuEntry("CUDA Device Property",_MENU_CUDA_PROPERTY);	
	
	mainMenu = glutCreateMenu(menuEvent);
	glutAddSubMenu("File", fileSubMenu);
	glutAddSubMenu("View", viewSubMenu);
	glutAddSubMenu("LDNI", ldniSubMenu);
	glutAddSubMenu("Mesh", meshSubMenu);
	glutAddMenuEntry("----",-1);
	glutAddMenuEntry("Quit", _MENU_QUIT);
	
	return mainMenu;
}


//---------------------------------------------------------------------------------
//	The major function of a program
int main(int argc, char *argv[])
{
	if (!initGLInteroperabilityOnCUDA(2, 0)) return 0;	

	glutInit(&argc, argv);
	glutInitWindowPosition(940,20);	

    glutInitDisplayMode(GLUT_DEPTH | GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_STENCIL);

    _pMainWnd=glutCreateWindow("LDNIcuda ver 1");
    glutDisplayFunc(displayFunc);
	glutReshapeWindow(960, 1000);
	
	glutMouseFunc(mouseFunc);
	glutMotionFunc(motionFunc);
	glutPassiveMotionFunc(passiveMotionFunc);
	glutKeyboardFunc(keyboardFunc);
	glutSpecialFunc(specialKeyboardFunc);
    glutReshapeFunc(reshapeFunc);
	glutVisibilityFunc(visibleFunc);

	

	initFunc();	
	_pGLK.SetClearColor(0.35f,0.35f,0.35f);
	_pGLK.SetForegroundColor(1.0f,1.0f,1.0f);
	_pGLK.m_bCoordDisp=false;


	if(glewInit() != GLEW_OK) {
		printf("glewInit failed. Exiting...\n");
		return false;
	}
	if (glewIsSupported("GL_VERSION_2_0")) {
		printf("\nReady for OpenGL 2.0\n");
		printf("-------------------------------------------------\n");		
	}
	else {
		printf("OpenGL 2.0 not supported\n");
		return false;
	}

	
	printf("--------------------------------------------------\n");
	printf("|| Please select the following functions by hot-keys:\n||\n");
	printf("|| Ctrl - O      Open\n");
	printf("|| Ctrl - S      Save\n");
	printf("|| Ctrl - R      Orbit and Pan\n");
	printf("|| Ctrl - W      Zoom Window\n");
	printf("|| Ctrl - A      Zoom All\n");
	printf("||-------------------------------\n");
	printf("|| Ctrl - L      LDNI shading with light\n");
	printf("|| Ctrl - Z      Turn on/off clipping mode\n");
	printf("||-------------------------------\n||\n");
	printf("|| Under clipping mode : \n");
	printf("|| Shift - Arrow left/right   Clip along x-axis \n");
	printf("|| Shift - Arrow up/down      Clip along y-axis \n");	
	printf("|| Shift - Page up/down       Clip along z-axis \n");
	printf("--------------------------------------------------\n");

	buildPopupMenu();
	glutAttachMenu(GLUT_RIGHT_BUTTON);
    glutSwapBuffers();

    glutMainLoop();
	
    return 0;             /* ANSI C requires main to return int. */
}

