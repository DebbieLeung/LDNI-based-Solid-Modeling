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


// GLKHeap.h: interface for the GLKHeap class.
//
//////////////////////////////////////////////////////////////////////

#ifndef _CW_GLKHEAPNODE
#define _CW_GLKHEAPNODE

#include "GLKObList.h"

class GLKHeapNode : public GLKObject
{
public:
	GLKHeapNode() {index=0;};
	virtual ~GLKHeapNode() {};
	float GetValue() {return v;};
	void SetValue(float value) {v=value;};

	int index;	//	this is the index for locating HeapNode in a heap

	void* attachedObj;

protected:
	float v;
};

#endif


#ifndef _CW_GLKHEAP
#define _CW_GLKHEAP

class GLKHeap  
{
public:
	GLKHeap(int maxsize, bool minOrMax=true);	//	true - min Heap
												//	false - max Heap
	GLKHeap(GLKHeapNode** arr, int n, bool minOrMax=true);
	virtual ~GLKHeap();

	const GLKHeapNode* operator[] (int i);

	int ListSize();
	bool ListEmpty();
	bool ListFull();

	void SetKetOnMinOrMax(bool flag);
	bool IsKeyOnMinOrMax();		//	true	- Keyed on min value
								//	false	- Keyed in max value
	
	bool Insert(GLKHeapNode* item);
	GLKHeapNode* RemoveTop();
	GLKHeapNode* GetTop();
	void AdjustPosition(GLKHeapNode* item);
	void Remove(GLKHeapNode* item);
	void ClearList();

private:
	bool bMinMax;	//	true	- Keyed on min value
					//	false	- Keyed in max value

	// hlist points at the array which can be allocated by the constructor (inArray == 0)
	//	or passed as a parameter (inArray == 1)
	GLKHeapNode** hlist;

	// amx elements allowed and current size of heap
	int maxheapsize;
	int heapsize;		// identifies end of list

	// utility functions for Delete/Insert to restore heap
	void FilterDown(int i);
	void FilterUp(int i);

	void Expand();
};

#endif
