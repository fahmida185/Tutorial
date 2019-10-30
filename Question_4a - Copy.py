import numpy as np
import cv2
import matplotlib.pyplot as plt 
from scipy.io import loadmat
 
def Question_4_a(img_in, thick, border_colour,font_colour):
    input_img = cv2.imread(img_in)
    input_img= cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) 

    #input_img= cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
    imgray = cv2.cvtColor(input_img,cv2.COLOR_BGR2GRAY)
    ####Finding Out The Contours
    contours, _ = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(input_img.shape, np.uint8)
    largest_areas = sorted(contours, key=cv2.contourArea)
    #######Converting From RGB to BGR Array
    border_colour[0], border_colour[2] = border_colour[2], border_colour[0]
    font_colour[0], font_colour[2] = font_colour[2], font_colour[0]
    ######Contour Filling
    imag10= cv2.drawContours(mask, [largest_areas[-1]], 0, font_colour,-1)
    imag11= cv2.add(input_img , mask)
    #################################### Bordering
    contours_areas = sorted(contours, key=cv2.contourArea)
    #print(contours_areas)
    imag12=cv2.drawContours(image = imag11 ,color =border_colour,contours = contours_areas, 
                            contourIdx = -1, thickness =thick)    
    ###########Remove Largest Contour########Remove Boundary
    bg = np.zeros(imgray.shape)
    biggest = 0
    bigcontour = None
    for contour in contours:
        area = cv2.contourArea(contour) 
        if area > biggest:
            biggest = area
            bigcontour = contour
            
    img_out=cv2.drawContours(imag12, [bigcontour], 0, (255, 255, 255),thick)
    
    return img_out

border_colour=(0,100,0)
font_colour=(50,0,50)
img_out=Question_4_a('sample_letters.png',5,list(border_colour),list(font_colour)) 
input_img = img_out
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) 
plt.imshow(input_img) # display the image