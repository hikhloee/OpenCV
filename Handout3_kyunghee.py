import cv2
import numpy as np
img=cv2.imread("data/car3.PNG")
img=cv2.resize(img,None,fx=1.2,fy=1.2,interpolation=cv2.INTER_NEAREST)
height,width=img.shape[:2]

box1=[]
f_count=0
select =0
plate_width=0

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur=cv2.GaussianBlur(gray,(5,5),0)
edges=cv2.Canny(blur,100,200)

cv2.imshow("edgeimg",edges) # 에지영상
#ret , thresh= cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)
out_img, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
out_img = cv2.drawContours(img, contours, -1, (0,255,0),2) # 외곽선 영상

cv2.imshow("contourimg",out_img)
print(contours)
for cnt in contours:
    area=cv2.contourArea(cnt)
    x,y,w,h=cv2.boundingRect(cnt)
    ratio=float(w)/h

    if (ratio>=0.2) and (ratio<=1.0) and(area>=100) and (area<=600):
        box=cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        box1.append(cv2.boundingRect(cnt))

cv2.imshow("box",box) #글자 외곽선?

for i in range(len(box1)): #버블정렬
    for j in range(len(box1)-(i+1)):
        if box1[j][0]>box1[j+1][0]:
                         temp=box1[j]
                         box1[j]=box1[j+1]
                         box1[j+1]=temp
                         
for m in range(len(box1)):
        count=0
        for n in range(m+1,(len(box1)-1)):
            delta_x=abs(box1[n+1][0]-box1[m][0])
            if delta_x > 150:
                    break
            delta_y =abs(box1[n+1][1]-box1[m][1])
            if delta_x ==0:
                    delta_x=1
            if delta_y ==0:
                    delta_y=1           
            gradient =float(delta_y) /float(delta_x)
            if gradient<0.25:
                count=count+1
               #measure number plate size         
        if count > f_count:
            select = m
            f_count = count;
            plate_width=delta_x
            
print(box1)
finalbox=cv2.rectangle(img,(box1[select][0]-10,box1[select][1]-10),(box1[select][0]+140,box1[select][1]+box1[select][3]+20),(255,0,0),3)
cv2.imshow("final",finalbox)

#number_plate=img[box1[select][1]-10:box1[select][3]+box1[select][1]+20,box1[select][0]-10:140+box1[select][0]]
#cv2.imshow("num_plate",number_plate)


cv2.waitKey(0)
cv2.destroyAllWindows()

#최종 번호판 검출만 하면 끝!
