import cv2
import scipy.misc as misc
import GetLiquidLevelCurve
Im=misc.imread("Image.png")
Lb=misc.imread("VesselMask.png")
# misc.imshow(Im)
# misc.imshow(Lb*100)
[BinaryCurve, OverLay]= GetLiquidLevelCurve.GetLiquidLevelCurve(Im,Lb)
misc.imshow(OverLay)
misc.imsave("Output.png",OverLay)
