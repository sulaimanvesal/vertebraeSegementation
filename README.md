# Vertebrae and Spine Segmentation
PyTorch code: Spine and Vertebrae Segmentation 

**Dataset:** The “images” folder contains 20 pngs of spine MRI slices. The “masks” folder contains 20 .npy files, where each mask represents the segmentation map of the discs and vertebrae for the corresponding spine image (1.png goes with 1.npy, etc.). 

* Label 1: disc location
* Label 2: Vertebrae
* Label 0: Background

![Spine Image and Mask](imgs/spine.PNG)


## What we’re looking for:
* A data loader capable of reading in the provided dataset in batches
* A script or instructions demonstrating using the data loader to run through 1 epoch of model training with a segmentation network
* Well-organized, easily-understandable, and documented code
* Object oriented programming where appropriate
