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

#version 410 compatibility

uniform ivec3 Unum;
uniform vec3 UOff;
uniform vec3 UWidth;
uniform ivec3 UFlip;
uniform vec3 Cent;



in vec3 Vertex;
out ivec3 g_id;
out ivec3 g_flip;

void main( void )
{
	int ix,iy,iz,i;
	
	i = gl_InstanceID/Unum.x;
	ix = gl_InstanceID-i*Unum.x; // ix = id%Unum.x
	
	iy = i/Unum.y;
	iy = i - iy*Unum.y; // iy = (id/Unum.x)%Unum.y
	
	iz = gl_InstanceID/(Unum.x*Unum.y); // iz = id/(Unum.x*Unum.y)
	
	g_id.x = ix;//gl_InstanceID;
	g_id.y = iy;
	g_id.z = iz;
	
	g_flip = UFlip;
	
	
	int remainder = (iy - (g_id.y/2)*2) ^ ((ix - (g_id.x/2)*2)^(iz - (g_id.z/2)*2));
	vec3 pos;
	
	if (remainder==1 && UFlip.x>0)
	{
		pos.x = Vertex.x + (Unum.x-ix-1)*(UWidth.x+UOff.x) ;
		gl_Position.x =  (Cent.x - pos.x) ;
	}
	else 
	{
		gl_Position.x = Vertex.x + ix*(UWidth.x+UOff.x) - Cent.x ;
	}
		
	if (remainder==1 && UFlip.y>0)
	{
		pos.y = Vertex.y + (Unum.y-iy-1)*(UWidth.y+UOff.y) ;
		gl_Position.y =  (Cent.y - pos.y);
	}
	else 
	{
		gl_Position.y = Vertex.y  + iy*(UWidth.y+UOff.y) - Cent.y;
	}
		
		
	if (remainder==1 && UFlip.z>0)
	{	
		pos.z = Vertex.z + (Unum.z-iz-1)*(UWidth.z+UOff.z) ;
		gl_Position.z =  (Cent.z - pos.z);
		
	}
	else
		gl_Position.z = Vertex.z + iz*(UWidth.z+UOff.z) - Cent.z;
		
		


}

	