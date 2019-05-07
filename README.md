# LDNI-based-Solid-Modeling

This project provides a set of solid modeling tools which aims to help processing 3D objects with complex topology and geometry that are widely used in many industrial applications (e.g., microstructure design and manufacturing, biomedical products and applications, jewelry products, reverse engineering).

<p float="middle">
  <img src="http://ldnibasedsolidmodeling.sourceforge.net/image/Mickey%20intersect%20with%20scaffold.jpg" width="100" />
  <img src="http://ldnibasedsolidmodeling.sourceforge.net/image/bunny_dragon_toolpath.jpg" width="100" /> 
  <img src="http://ldnibasedsolidmodeling.sourceforge.net/image/bunny_dragon_toolpath_1.jpg" width="100" />
</p>

The system is completely GPU-based and heavily based on an implicit representation named LDNI.

Features
- Input/Output : Obj files
- Boolean Operations - Union, Intersection and Substraction
- Offset
- Scaffold
- Super Union - union multiple meshes in one operation
- 3D Printing
    - Generate tool path and supporting files for FDM
    - Generate stencil images for SLA

### [Old Project page](http://ldnibasedsolidmodeling.sourceforge.net/) | [Youtube](https://youtu.be/G75mS1VGqx0)

If you use this code, please cite
```text
@article{WANG2010,
title = "Solid modeling of polyhedral objects by Layered Depth-Normal Images on the GPU",
journal = "Computer-Aided Design",
volume = "42",
number = "6",
pages = "535 - 544",
year = "2010",
issn = "0010-4485",
url = "http://www.sciencedirect.com/science/article/pii/S0010448510000278",
author = "Charlie C.L. Wang and Yuen-Shan Leung and Yong Chen",
keywords = "Solid modeler, Complex objects, Layered Depth-Normal Images, GPU",
}
```
