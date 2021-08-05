import cv2
import numpy as np

image=cv2.imread("lenna.PNG")
cv2.imshow("Input image",image)

height,width=image.shape[:2]
print(height)
print(width)

num=input("yCrCb 을 원하면  y, HSV를 원하면 h 를 입력하시오 : ")

if num=='y':
    image=cv2.cvtColor(image,cv2.COLOR_BGR2YCrCb)
    cv2.imshow('first',image)
    
    im_arr = np.array(image)
    Y_matrix=im_arr[:,:,0]
    Cr_matrix=im_arr[:,:,1]
    Cb_matrix=im_arr[:,:,2]

    #print(Cb_matrix)
    
    for i in range (0,height):
        for j in range (0,width):
            
            if Cb_matrix[i][j]<=127:
                if Cb_matrix[i][j]>=77:
                    if Cr_matrix[i][j]>=133:
                        if Cr_matrix[i][j]<=173:
                            Cb_matrix[i][j]=Cb_matrix[i][j]
                            Cr_matrix[i][j]=Cr_matrix[i][j]
            else:
                Cb_matrix[i][j]=0
                Cr_matrix[i][j]=0
                

    cv2.imshow('1',Y_matrix)
    cv2.imshow('2',Cb_matrix)
    cv2.imshow('3',Cr_matrix)
    
    image_out1=cv2.merge((Y_matrix,Cr_matrix,Cb_matrix))
    cv2.imshow("final",image_out1)
    image_out=cv2.cvtColor(image_out1,cv2.COLOR_YCrCb2BGR)
    cv2.imshow("result",image_out)
if num=='h':
    image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    cv2.imshow('first',image)
    
    im_arr = np.array(image)
    H_matrix=im_arr[:,:,0]
    S_matrix=im_arr[:,:,1]
    V_matrix=im_arr[:,:,2]

    #print(Cb_matrix)
    
    for i in range (0,height):
        for j in range (0,width):
            
            if H_matrix[i][j]<=50:
                if S_matrix[i][j]>=70:
                    if S_matrix[i][j]<=150:
                        if V_matrix[i][j]>=50:
                            if V_matrix[i][j]<=255:
                                H_matrix[i][j]=H_matrix[i][j]
                                S_matrix[i][j]=S_matrix[i][j]
                                V_matrix[i][j]=V_matrix[i][j]
            else:
                H_matrix[i][j]=0
                S_matrix[i][j]=0
                V_matrix[i][j]=0

    cv2.imshow('1',H_matrix)
    cv2.imshow('2',S_matrix)
    cv2.imshow('3',V_matrix)
   
    image_out1=cv2.merge((H_matrix,S_matrix,V_matrix))
    cv2.imshow("final",image_out1)
    image_out=cv2.cvtColor(image_out1,cv2.COLOR_HSV2BGR)
    cv2.imshow("result",image_out)

else:
     print("잘못 입력하셨습니다 “)

    cv2.waitKey(0)
    cv2.destroyAllWindows()




# ※Bonus ( video )

import cv2
import numpy as np

cap = cv2.VideoCapture('kingkong.avi')

while True:
    
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Input',frame)
        imgHSV=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        lowerBound=np.array([0,70,50])
        upperBound=np.array([50,155,255])

        turnout=cv2.inRange(imgHSV,lowerBound,upperBound)
        cv2.imshow('Output', turnout)
        cv2.waitKey(100)
        
    else:
        break
  
cap.release()
cv2.destroyAllWindows()  
