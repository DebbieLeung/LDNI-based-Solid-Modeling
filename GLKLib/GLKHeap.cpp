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


// GLKHeap.cpp: implementation of the GLKHeap class.
//
//////////////////////////////////////////////////////////////////////

#include "GLKHeap.h"

//////////////////////////////////////////////////////////////////////
// Construction/Destruction
//////////////////////////////////////////////////////////////////////

GLKHeap::GLKHeap(int maxsize, bool minOrMax)
{
	hlist=(GLKHeapNode**) new long[maxsize];
	maxheapsize=maxsize;
	heapsize=0;

	bMinMax=minOrMax;
}

GLKHeap::GLKHeap(GLKHeapNode** arr, int n, bool minOrMax)
{
	int i, currentpos;

	bMinMax=minOrMax;
	hlist=(GLKHeapNode**) new long[n];
	maxheapsize=n;
	heapsize=n;
	for(i=0;i<n;i++) {hlist[i]=arr[i];hlist[i]->index=i;}

	currentpos=(heapsize-2)/2;
	while(currentpos>=0)
	{
		FilterDown(currentpos);
		currentpos--;
	}
}

GLKHeap::~GLKHeap()
{
	ClearList();
}

//////////////////////////////////////////////////////////////////////
// Implementation
//////////////////////////////////////////////////////////////////////

const GLKHeapNode* GLKHeap::operator[] (int i)
{
	return hlist[i];
}

int GLKHeap::ListSize()
{
	return heapsize;
}

bool GLKHeap::ListEmpty()
{
	if (heapsize==0) return true;
	return false;
}

bool GLKHeap::ListFull()
{
	if (heapsize==maxheapsize) return true;
	return false;
}

void GLKHeap::Expand()
{
	GLKHeapNode** new_hlist=(GLKHeapNode**)new long[maxheapsize*2];
	for(int i=0;i<heapsize;i++) new_hlist[i]=hlist[i];
	delete [](GLKHeapNode**)hlist;
	hlist=new_hlist;
	maxheapsize=maxheapsize*2;
}

void GLKHeap::SetKetOnMinOrMax(bool flag)
{
	bMinMax=flag;
}

bool GLKHeap::IsKeyOnMinOrMax()
{
	return bMinMax;
}

void GLKHeap::AdjustPosition(GLKHeapNode* item)
{
	FilterUp(item->index);
	FilterDown(item->index);
}

void GLKHeap::Remove(GLKHeapNode* item)
{
	if (ListEmpty()) return;

	hlist[item->index]=hlist[heapsize-1];	hlist[item->index]->index=item->index;
	heapsize--;

	FilterDown(item->index);
}

bool GLKHeap::Insert(GLKHeapNode* item)
{
	if (ListFull()) Expand();

	hlist[heapsize]=item;	item->index=heapsize;
	FilterUp(heapsize);
	heapsize++;

	return true;
}

GLKHeapNode* GLKHeap::GetTop() 
{
	return (hlist[0]);
}

GLKHeapNode* GLKHeap::RemoveTop()
{
	GLKHeapNode *tempitem;

	if (ListEmpty()) return 0;

	tempitem=hlist[0];
	hlist[0]=hlist[heapsize-1];	hlist[0]->index=0;
	heapsize--;

	FilterDown(0);

	return tempitem;
}

void GLKHeap::ClearList()
{
	if (maxheapsize>0) delete hlist;
	maxheapsize=0;	heapsize=0;
}

void GLKHeap::FilterDown(int i)
{
	int currentpos, childpos;
	GLKHeapNode *target;

	currentpos=i;
	target=hlist[i];

	childpos=2*i+1;

	while (childpos<heapsize)
	{
		if (bMinMax) {
			if ((childpos+1<heapsize) &&
				((hlist[childpos+1]->GetValue())<=(hlist[childpos]->GetValue())))
				childpos=childpos+1;

			if ((target->GetValue())<=(hlist[childpos]->GetValue()))
				break;
			else
			{
				hlist[currentpos]=hlist[childpos];
				hlist[currentpos]->index=currentpos;

				currentpos=childpos;
				childpos=2*currentpos+1;
			}
		}
		else {
			if ((childpos+1<heapsize) &&
				((hlist[childpos+1]->GetValue())>=(hlist[childpos]->GetValue())))
				childpos=childpos+1;

			if ((target->GetValue())>=(hlist[childpos]->GetValue()))
				break;
			else
			{
				hlist[currentpos]=hlist[childpos];
				hlist[currentpos]->index=currentpos;

				currentpos=childpos;
				childpos=2*currentpos+1;
			}
		}
	}

	hlist[currentpos]=target;	target->index=currentpos;
}

void GLKHeap::FilterUp(int i)
{
	int currentpos, parentpos;
	GLKHeapNode *target;

	currentpos=i;
	parentpos=(int)((i-1)/2);
	target=hlist[i];

	while(currentpos!=0)
	{
		if (bMinMax) {
			if ((hlist[parentpos]->GetValue())<=(target->GetValue()))
				break;
			else
			{
				hlist[currentpos]=hlist[parentpos];
				hlist[currentpos]->index=currentpos;
				currentpos=parentpos;
				parentpos=(int)((currentpos-1)/2);
			}
		}
		else {
			if ((hlist[parentpos]->GetValue())>=(target->GetValue()))
				break;
			else
			{
				hlist[currentpos]=hlist[parentpos];
				hlist[currentpos]->index=currentpos;
				currentpos=parentpos;
				parentpos=(int)((currentpos-1)/2);
			}
		}
	}

	hlist[currentpos]=target;	target->index=currentpos;
}
