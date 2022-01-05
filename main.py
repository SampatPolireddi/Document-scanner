import cv2
import numpy as np
from numpy.lib.type_check import mintypecode

def reorder(myPoints):#Func to arrange the contours points in order
    myPoints=myPoints.reshape((4,2))
    myPointsNew=np.zeros((4,1,2),dtype=np.int32)
    add=myPoints.sum(1)

    myPointsNew[0]=myPoints[np.argmin(add)]
    myPointsNew[3]=myPoints[np.argmax(add)]
    diff=np.diff(myPoints,axis=1)
    myPointsNew[1]=myPoints[np.argmin(diff)]
    myPointsNew[2]= myPoints[np.argmax(diff)]
    return myPointsNew

def drawRectangle(img,biggest,thickness):#Func to draw the rect from the contours
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]), (0, 255, 0), thickness)
    cv2.line(img, (biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]), (0, 255, 0), thickness)
    return img

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):#Func to sharpen the img
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened



path='/home/sampat/Desktop/Doc_scanner/scan1.jpeg'
img=cv2.imread(path)
img=cv2.resize(img,(480,640))
 

img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_blur=cv2.GaussianBlur(img_gray,(5,5),1)
img_canny=cv2.Canny(img_blur,200,200)

ret, img_thresh = cv2.threshold(img_gray, 123, 255, cv2.THRESH_BINARY) #If pixel val>123 then pixel val=255(white)


contours,_=cv2.findContours(img_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) #Finds the contours for the img_tresh
Contours_frame=img.copy()
Contours_frame=cv2.drawContours(Contours_frame,contours,-1,(255,0,255),5) #Draws a purple(255,0,255)line connecting all contours(i.e -1)
                                                                          #5 indicates thickness of the line

Corner_frame=img.copy()
maxArea=0
biggest=[]

for i in contours:
    #print(contours)
    area=cv2.contourArea(i)
    if area>5000: #To avoid small scontours in the img

        peri = cv2.arcLength(i, True)
        edges = cv2.approxPolyDP(i, 0.04 * peri, True) #To find the edges of the img
        if area>maxArea and len(edges) == 4: #Checks if its a rectange 
            biggest=edges
            maxArea=area
# print(biggest)

biggest=reorder(biggest)#To reorder the points in so that the system follows the same order, otherwise we might not get a rectangle if points are missplaced
#print(biggest)

widthImg=480
heightImg=640

if len(biggest) !=0:
    drawRectangle(Corner_frame,biggest,5)
    pts1 = np.float32(biggest) # PREPARE POINTS FOR WARP
    pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # PREPARE POINTS FOR WARP
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warp = cv2.warpPerspective(img, matrix, (widthImg, heightImg)) #Crops the img with the given dimensions



final_img=cv2.cvtColor(img_warp,cv2.COLOR_BGR2GRAY) #Converting cropped img to grayscale to get the scanned doc look
cv2.imwrite('scanned_Img.png',final_img)

sharpened_image= unsharp_mask(final_img) #Sharpening the img
cv2.imwrite('final_sharpened_image.png', sharpened_image)


cv2.imshow('img',img)
cv2.imshow('Gray',img_gray)
cv2.imshow('Blur',img_blur)
cv2.imshow('Canny',img_canny)
cv2.imshow('img_tresh', img_thresh)
cv2.imshow('Contour_frame',Contours_frame)
cv2.imshow('Corner_frame',Corner_frame)
cv2.imshow('warp',img_warp)
cv2.imshow('scanned_img',final_img)
cv2.imshow('final_scanned_sharpened_img',sharpened_image)

cv2.waitKey(0)
cv2.destroyAllWindows()