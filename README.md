#  Computer vision based recognition of liquid surface and liquid level in transparent glassware vessels (python)
 Get input image containing transperent vessel with liquid (Image)  and a binary mask of the vessel area in the image (VesselMask)
 Return liquid surfaces/levels marked on the image

## Inputs:
  Image: input image
  VesselMask: binary mask with the glassware vessel in in the image

## Output:
  Mask of the liquid surface curves in the image.
  The liquid surface curves marked on the image.
  

![](/Scheme.png)  
For more details see:  [Computer vision-based recognition of liquid surfaces and phase boundaries in transparent vessels, with emphasis on chemistry applications](http://arxiv.org/abs/1404.7174)
For more accurate method of solving the same problem with neural nets see [this work](https://github.com/sagieppel/Detecting-and-segmenting-and-classifying-materials-inside-vessels-in-images-using-convolutional-net)

Tracing the transparent vessel in the image and creating VesselMask can be done using neural network in [here](https://github.com/sagieppel/Detecting-and-segmenting-and-classifying-materials-inside-vessels-in-images-using-convolutional-net)



