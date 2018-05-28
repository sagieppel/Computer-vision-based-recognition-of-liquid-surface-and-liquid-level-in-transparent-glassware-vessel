import scipy.misc as misc
import cv2
import numpy as np
import time
def GetLiquidLevelCurve(Image,VesselMask,MaxViewAngleDeg=20,Mode="Canny",NormScore=False,MinScore=4,MinRelativeScore =0.7,MaxCurves=4,LineThikness=2,MinWidth=0.25,IgnoreTopBottum=True):
    #Get input image containing transperent vessel with liquid (Image)  and the mask of the vessel area in the image (VesselMask)
    #Return liquid surfaces/levels marked on the image

    #Inputs:
    #Image# input image
    #VesselMaskBinary # binary mask with the tansperent vessel in the iimage marked one

    #Optional inputs:
    # MaxViewAngleDeg = 10# is the maximal angle the liquid suface might be viewed  from in degrees (this limit the possible shape of the liquid surface)
    # Mode = "CannyEdge" # the type of image the scanning will be done on options are "Canny" "Sobel" "Laplacian" and "Greyscale"
    # NormScore = False # Optional parameter regarding how score for liquid surface will be calculated
    # MinScore = -1 # Minimal score for curve to be accepted
    # MinRelativeScore = 0.7 # The minimal score with respect to the highest score that will allow curve to be selected as liquid surfca
    # MaxCurves = 10 # Max number of curves that can be selected as liquid
    # LineThikness = 2 # Basically the resolution in which the scane will be done
    # MinWidth = 0.25 minimal with of the vessel with respect to vessel maximal width  in which the vessel will be scanned (assuming narrow vessel regions correspond to corks)
    # IgnoreTopBottum=False Ignore Curves detected at the vessel to or bottum since they more likely be false alarm

    #MaxViewAngleDeg
    Sy, Sx = VesselMask.shape
    Im=cv2.resize(Image,(Sx,Sy)) #make sure image and Vessel Mask are of the same size
    EllipseMaxRatio=np.sin(np.deg2rad(MaxViewAngleDeg)) # max ratio between the hight and length of the ellipse form by the liquid surface

    kernel = np.ones((3, 3), np.uint8)
    VesselMask = cv2.erode(VesselMask.astype(np.uint8), kernel, iterations=2) # Erode vessel mask to focus trhe scan on the vessel interior
#===========================Get edge image on which the liquid level will be find==========================================================================
    if Mode=="Canny": # Use canny binary edge map
        I = cv2.Canny(Im, 50, 100)
    elif Mode=="Sobel": # Use Sobel edge map
        Grey = Im.mean(axis=2)
        sobelx = cv2.Sobel(Grey, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(Grey, cv2.CV_64F, 0, 1, ksize=5)
        I = np.sqrt(np.square(sobelx) + np.square(sobely))
    elif Mode == "Laplacian":
        Grey = Im.mean(axis=2)
        I = cv2.Laplacian(Grey, cv2.CV_64F)
    elif Mode == "Greyscale":
        I=Im.mean(axis=2)
#===========================================================================================================================
# Get array of left and right edges of the vessel label for each row in the image
    VesselSides=[] # First line row, second line min occupied collumn in row third max occupied column in row
    MaxWidth = -1 # Max width of vessel in pixels
    for i in range(Sy):
         Ln=VesselMask[i,:].nonzero()[0] # Index of none zeros elements in the lin
         if len(Ln)>0:
             x0=Ln.min()
             x1=Ln.max()
             VesselSides.append([i,x0,x1])# The row # Left bound # Right bound
             if x1-x0>MaxWidth: MaxWidth=x1-x0
    MinWidth=MaxWidth*MinWidth # min width of raw that will be scanned (assuming narraw vessel region correspond to corks)
    if LineThikness==-1: LineThikness=np.max((np.int(np.ceil(len(VesselSides)/20)),1)) # Set line thikness to be 5% of the vessel hight in pixels

    if (IgnoreTopBottum): VesselSides=VesselSides[int(MaxWidth*EllipseMaxRatio/2):-int(MaxWidth*EllipseMaxRatio/2)] # Ignore the to and bottum of the vessel region
#============================Scan for optimal curve that represent the vessel boundary================================================================================
  #  VesselSides=VesselSides[200:]

    RSHWCx=np.zeros((len(VesselSides),5)) # Array of scores the various of curves in and their feetnes to liquid surfaces (Fore each 0) row 1) Score  2) Curve Hight 3) Curve Width)
    for c,rw in enumerate(VesselSides):
        Yc=rw[0] # center raw of the elipse
        x0=rw[1] # to left ellispe point
        x1=rw[2] # to right ellipse
        if x1-x0<MinWidth: continue # ignore narraw vessel regions assuming they correspond to cork and funnels
        H=int((x1-x0)*EllipseMaxRatio/2) # calculate liquid surface ellipse max hight
        TopScore = -100
        HTop = -100
        for h in range(H):# Scan varoious of ellipitic curves along each row of the vessel (rw) and score them for how good they represent liquid surface in the image
            #================match Bottum ellispse curve to image and get score==========================================================================
            Elipse=np.zeros((h + LineThikness, x1 - x0 + LineThikness*2))
            cv2.ellipse(Elipse, (int((x1 - x0) / 2), 0), (int((x1 - x0) / 2), h), 0, 0, 180,1, LineThikness)
            W1=I[Yc:Yc + Elipse.shape[0], x0:x0 + Elipse.shape[1]]
            W2=I[Yc + LineThikness:Yc + Elipse.shape[0] + LineThikness, x0:x0 + Elipse.shape[1]]
            if W1.shape==Elipse.shape and W2.shape==Elipse.shape:
                s1 = (W1 * Elipse).sum()
                s2 = (W2 * Elipse).sum() # Caclculate curve score
                if NormScore: TempScore1=np.abs((s1-s2)/(s1+s2))/(x1-x0)#/Elipse.sum()
                else: TempScore1=np.abs(s1-s2)/(x1-x0)#/Elipse.sum() # Caclculate curve score

            else:TempScore1 = -1
            #================match Top curve to image and get score===================================================================================
            Elipse=np.flip(Elipse,axis=0)
            W1 = I[Yc - Elipse.shape[0]:Yc , x0:x0 + Elipse.shape[1]]
            W2 = I[Yc - Elipse.shape[0]- LineThikness:Yc - LineThikness, x0:x0 + Elipse.shape[1]]
            if W1.shape == Elipse.shape and W2.shape == Elipse.shape:
               s1 = (W1 * Elipse).sum()
               s2 = (W2 * Elipse).sum()
               if NormScore:  TempScore2 = np.abs((s1 - s2) / (s1 + s2))/(x1-x0)#/Elipse.sum() # Caclculate curve score
               else: TempScore2 = np.abs(s1 - s2)/(x1-x0)#//Elipse.sum() # Caclculate curve score
            else: TempScore2=-1
            #==================Compare current curve score to top score of curves in this row==================================================================================================
            if np.max((TempScore1,TempScore2))>TopScore:
                     TopScore=np.max((TempScore1,TempScore2))
                     HTop = h

        #=====================================================================================================================================
        RSHWCx[c]=np.array([Yc,TopScore,HTop,x1-x0,int((x1+x0)/2)]) # Write top score for curve in this row and the curve paramters
######################################################################################################################################
    if len(RSHWCx)==0: #if there  is curves that got score ( no vessel in the image or the vessel to small) return empty result
        MarkedIm = Im.copy()  # Orginal input image with liquid surface curves drawn on it in red
        Template = np.zeros(Im[:, :, 0].shape)  # Binary template with liquid surface curves drawn on it
        return Template, MarkedIm
#-----------------------------Go over the scores of various of curves and choose curves which are likely to represent liquid surface--------------------------------------------------------------------------------------------------------------------------
    SortedScores=np.argsort(RSHWCx[:,1])[::-1]
    MaxScore=RSHWCx[SortedScores[0],1]
    MinScore=np.max((MinRelativeScore *MaxScore,MinScore)) # Minimal score for curve to be considered as liquid surface
    LiquidSurfaces=[] #Array that will contain the paramters of the curves that were identifies as liquid surfaces (contain eliptic curve x,y centers hight and width)


    if MaxScore>MinScore:
        LiquidSurfaces.append(RSHWCx[SortedScores[0]]) # Add top ranked curve to liquid surface curves
        Ys = np.ones((MaxCurves+1))*(-1000) # Array with the hight of all cuves selected as liquid surfaces
        Ys[0] = LiquidSurfaces[0][0]
        for ind in SortedScores:
             if RSHWCx[ind,1]>=MinScore:
                 if (abs(RSHWCx[ind][0]-Ys)).min()>EllipseMaxRatio*MaxWidth+LineThikness: # make sure there no overlap between selected curves
                     if len(LiquidSurfaces) == MaxCurves: break
                     LiquidSurfaces.append(RSHWCx[ind])# Add  curve to liquid surface curves
                     Ys[len(LiquidSurfaces)]=RSHWCx[ind][0]

 ##########################draw curves correspond to liquid surfaces on image and binary templat###################################################
    MarkedIm=Im.copy() # Orginal input image with liquid surface curves drawn on it in red
    Template=np.zeros(Im[:,:,0].shape) # Binary template with liquid surface curves drawn on it
    for curve in LiquidSurfaces: # Draw curves on template
        cv2.ellipse(Template, (int(curve[4]), int(curve[0])), (int(curve[3]/2), int(curve[2])), 0, 0, 360, 1, LineThikness)# Draw curves on template
    MarkedIm[:, :, 0][Template > 0] = 255 # mark liquid surface curve templat on image
    MarkedIm[:, :, 1][Template > 0] = 0
    MarkedIm[:, :, 2][Template > 0] = 0
    # misc.imshow(I)
    # I[Template>0]=120
    # misc.imshow(I)

    return Template,MarkedIm




































