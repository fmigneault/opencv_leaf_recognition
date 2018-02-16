# opencv_leaf_recognition
OpenCV samples for CNN leaf recognition

## Available samples 

* **LeafNet_predict**  
Prediction of leaf image using LeafNet trained with either of Flavia, Foliage, LeafSnap datasets

* **DeepPlant_predict**  
Prediction of leaf image using Deep-Plant trained with MalayaKew and Flavia datasets 

## Installation

Clone the repo and use the provided CMake script to compile C++ codes.

```
git clone https://github.com/fmigneault/opencv_leaf_recognition
cd opencv_leaf_recognition
mkdir build
cd build
cmake ..
make -j4
make install
cd ..
```

Then, launch the desired sample with the corresponding `sh` file.

*__Note__*  
Some paths might require editing in the `sh` files to specify input image or model location on your machine.
