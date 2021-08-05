import cv2
import numpy as np

    
# Warp img2 to img1 using the homography matrix H
def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0,rows1], [cols1,rows1], [cols1,0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2)
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    translation_dist = [-x_min, -y_min]
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0,0,1]]) 

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
    
    return output_img

if __name__=='__main__':
    img = []
    img1 = cv2.imread('notebook1.jpg', 0)    
    img2 = cv2.imread('notebook2.jpg', 0)
    img3 = cv2.imread('notebook3.jpg', 0)
    img4 = cv2.imread('notebook4.jpg', 0)
    img5 = cv2.imread('notebook5.jpg', 0)
    img=[img1,img2,img3,img4,img5]
    
    min_match_count = 10
    sift = cv2.xfeatures2d.SIFT_create()
    img[0]=cv2.resize(img[0],None,fx=0.1,fy=0.1,interpolation=cv2.INTER_NEAREST)
    img[1]=cv2.resize(img[1],None,fx=0.1,fy=0.1,interpolation=cv2.INTER_NEAREST)
    img[2]=cv2.resize(img[2],None,fx=0.1,fy=0.1,interpolation=cv2.INTER_NEAREST)
    img[3]=cv2.resize(img[3],None,fx=0.1,fy=0.1,interpolation=cv2.INTER_NEAREST)
    img[4]=cv2.resize(img[4],None,fx=0.1,fy=0.1,interpolation=cv2.INTER_NEAREST)

    keypoints= []
    descriptors= []

    keypoints1, descriptors1 = sift.detectAndCompute(img[0], None)
    keypoints2, descriptors2 = sift.detectAndCompute(img[1], None)
    keypoints3, descriptors3 = sift.detectAndCompute(img[2], None)
    keypoints4, descriptors4 = sift.detectAndCompute(img[3], None)
    keypoints5, descriptors5 = sift.detectAndCompute(img[4], None)
    
    keypoints= [keypoints1,keypoints2,keypoints3,keypoints4,keypoints5]
    descriptors=[descriptors1,descriptors2,descriptors3,descriptors4,descriptors5]

    global result   
    result = img[0]    

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)


    for i in range(0,4):
        
        key,des =sift.detectAndCompute(result, None)
        matches = flann.knnMatch(des, descriptors[i+1], k=2)
        cv2.imshow("result",result)
        
        good_matches = []
        for m1,m2 in matches:
            if m1.distance < 0.7*m2.distance:
                good_matches.append(m1)

        if len(good_matches) > min_match_count:
            src_pts = np.float32([ key[good_match.queryIdx].pt for good_match in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ keypoints[i+1][good_match.trainIdx].pt for good_match in good_matches ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0) # 여기까지 3번
            result = warpImages(img[i+1],result,M)
            cv2.imshow('Stitched output', result)

            cv2.waitKey()

        else:
            print ("We don't have enough number of matches between the two images.")
            print ("Found only %d matches. We need at least %d matches." % (len(good_matches), min_match_count))
