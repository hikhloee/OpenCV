#HW #2-1

import cv2
import numpy as np

img=cv2.imread('landscape.jpg')
cv2.imshow('Original',img)

num_rows,num_cols=img.shape[:2]

def turnaround(event,x,y,flags,param):
    img=cv2.imread('landscape.jpg')
    if event==cv2.EVENT_LBUTTONDOWN:
        rotation_matrix=cv2.getRotationMatrix2D((num_cols*0.6,num_rows*0.6),90,1)
        img=cv2.warpAffine(img,rotation_matrix,(num_cols*2,num_rows*2))
        cv2.imshow('output window',img)
    elif event==cv2.EVENT_RBUTTONDOWN:
        rotation_matrix=cv2.getRotationMatrix2D((num_cols*0.6,num_rows*0.6),-90,1)
        img=cv2.warpAffine(img,rotation_matrix,(num_cols*2,num_rows*2))
        cv2.imshow('output window',img)
        
cv2.namedWindow('output Window')
cv2.setMouseCallback('output Window',turnaround)

while True:
    cv2.imshow("output Window",img)
    cv2.imwrite("output",img)
    c=cv2.waitKey(10)
    if c==27:
        break

cv2.destroyAllWindows()


#HW #2-2
import cv2
import numpy as np


src_points = np.float32([[0,0], [0,0], [0,0], [0,0 ]])  
num_pt = 0

def detect_circle (event, x, y, flags, param): 
    global num_pt, src_points  
    if event == cv2.EVENT_LBUTTONDOWN : 
       cv2.circle(img,(x,y),10,(0,200,0),-1)
       src_points[num_pt][0] = x 
       src_points[num_pt][1] = y 
       num_pt= num_pt+1

    
img=cv2.imread('input.jpg')
img=cv2.resize(img,None,fx=0.2,fy=0.2,interpolation=cv2.INTER_NEAREST)
cv2.imshow('input',img)
rows,cols=img.shape[:2]

cv2.namedWindow('mouse')
cv2.setMouseCallback('mouse',detect_circle)

while True:
    cv2.imshow('mouse',img)
    c=cv2.waitKey(10)
    if c==27:
        break

print(rows) #806
print(cols) #605
pair1_points=np.float32([[0,0],[0,rows-1],[cols-1,rows-1],[cols-1,0]])

perspective_matrix=cv2.getPerspectiveTransform(src_points,pair1_points)
img_output=cv2.warpPerspective(img,perspective_matrix,(cols,rows))

cv2.imshow("Perspective",img_output)
cv2.imwrite('Perspective',img_output)
cv2.waitKey(0)
cv2.destroyAllWindows()
