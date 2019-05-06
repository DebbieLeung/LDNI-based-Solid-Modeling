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

// GLKObList.h: interface for the GLKObList class.
//
//////////////////////////////////////////////////////////////////////


#ifndef _CW_GLKObject
#define _CW_GLKObject

#define NULL	0

class GLKObject  
{
public:
	GLKObject () {};
};

#endif



#ifndef _CW_GLKObNode
#define _CW_GLKObNode

#define GLKPOSITION		GLKObNode*

class GLKObNode
{
public:
	GLKObNode(GLKObNode* ptrprev=NULL,GLKObNode* ptrnext=NULL)	{next=ptrnext;prev=ptrprev;};

	void InsertAfter(GLKObNode *p) {
		GLKObNode *oldNextNode=next;
		next=p;p->prev=this;
		if (oldNextNode) {oldNextNode->prev=p;p->next=oldNextNode;}
	};
	GLKObNode *DeleteAfter() {
		GLKObNode *tempObj=next;
		if (next==NULL) return NULL;
		next=tempObj->next;
		next->prev=this;
		return tempObj;
	};

	void InsertBefore(GLKObNode *p) {
		GLKObNode *oldPrevNode=prev;
		prev=p;p->next=this;
		if (oldPrevNode) {oldPrevNode->next=p;p->prev=oldPrevNode;}
	};
	GLKObNode *DeleteBefore() {
		GLKObNode *tempObj=prev;
		if (prev==NULL) return NULL;
		prev=tempObj->prev;
		prev->next=this;
		return tempObj;
	};

	GLKObNode *next;
	GLKObNode *prev;
	GLKObject *data;
};

#endif



#ifndef _CW_GLKObList
#define _CW_GLKObList

class GLKObList  
{
public:
	GLKObList();
	virtual ~GLKObList();

	GLKObject* GetHead();
	GLKObject* GetTail();
	GLKPOSITION GetHeadPosition() {return headPos;};
	GLKPOSITION GetTailPosition() {return tailPos;};
	GLKPOSITION FindIndex(int index);
	GLKPOSITION Find(GLKObject* element);

	GLKPOSITION AddHead( GLKObject* newElement );
	void AddHead( GLKObList* pNewList );
	GLKPOSITION AddTail( GLKObject* newElement );
	void AddTail( GLKObList* pNewList );
	GLKObject* RemoveHead();
	GLKObject* RemoveTail();
	GLKObject* RemoveAt(GLKPOSITION rPosition);
	void Remove(GLKObject* element);
	void RemoveAll();
	void RemoveAllWithoutFreeMemory();

	GLKObject* GetNext( GLKPOSITION& rPosition );
	GLKObject* GetPrev( GLKPOSITION& rPosition );

	GLKObject* GetAt( GLKPOSITION rPosition );

	GLKPOSITION InsertBefore( GLKPOSITION rPosition, GLKObject* newElement );
	GLKPOSITION InsertAfter( GLKPOSITION rPosition, GLKObject* newElement );

	void AttachListTail( GLKObList* pNewList );

	int GetCount() {return nCount;};

	bool IsEmpty() {return ((nCount==0)?true:false);};

private:
	GLKPOSITION headPos;
	GLKPOSITION tailPos;

	int nCount;
};

#endif


#ifndef _CW_GLKARRAY
#define _CW_GLKARRAY

class GLKArray  
{
public:
	GLKArray(int sx=50, int increaseStep=50, int type=0);	//	Type:	0 - (void*)
															//			1 - int
															//			2 - float 
															//			3 - double
	~GLKArray();

	int GetSize();

	void* RemoveAt(int i);
	void RemoveAll();
	
	void Add(void* data);
	void Add(int data);
	void Add(float data);
	void Add(double data);

	void SetAt(int i, void* data);
	void SetAt(int i, int data);
	void SetAt(int i, float data);
	void SetAt(int i, double data);

	void InsertAt(int i, void* data);
	void InsertAt(int i, int data);
	void InsertAt(int i, float data);
	void InsertAt(int i, double data);

	void* GetAt(int i);
	int GetIntAt(int i);
	float GetFloatAt(int i);
	double GetDoubleAt(int i);

private:
	void** listType0;
	int* listType1;
	float* listType2;
	double* listType3;

	int size;
	int arraySize;

	int step;
	int m_type;
};

#endif