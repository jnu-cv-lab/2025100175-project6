import cv2
import numpy as np


path_box = 'C:/python/python4/box.png'
path_scene = 'C:/python/python4/box_in_scene.png'


img1 = cv2.imread(path_box, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(path_scene, cv2.IMREAD_GRAYSCALE)
img2_color = cv2.imread(path_scene)  


nfeatures_list = [500, 1000, 2000]

print(f"{'nfeatures':<10} | {'KP1':<5} {'KP2':<5} | {'Matches':<8} {'Inliers':<8} {'Ratio':<6} | {'Success'}")
print("-" * 70)

for nfeat in nfeatures_list:
 
    orb = cv2.ORB_create(nfeatures=nfeat)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    
    if len(matches) >= 4:  
       
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

       
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

       
        matches_mask = mask.ravel().tolist()
        inliers = sum(matches_mask)
        total_matches = len(matches)
        inlier_ratio = inliers / total_matches if total_matches > 0 else 0

       
        is_success = "Yes" if inliers >= 10 and inlier_ratio > 0.1 else "No"

    else:
        inliers = 0
        total_matches = len(matches)
        inlier_ratio = 0
        is_success = "No"

    
    print(f"{nfeat:<10} | {len(kp1):<5} {len(kp2):<5} | {total_matches:<8} {inliers:<8} {inlier_ratio:.2f}   | {is_success}");import cv2
import numpy as np


path_box = 'C:/python/python4/box.png'
path_scene = 'C:/python/python4/box_in_scene.png'


img1 = cv2.imread(path_box, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(path_scene, cv2.IMREAD_GRAYSCALE)
img2_color = cv2.imread(path_scene) 


nfeatures_list = [500, 1000, 2000]

print(f"{'nfeatures':<10} | {'KP1':<5} {'KP2':<5} | {'Matches':<8} {'Inliers':<8} {'Ratio':<6} | {'Success'}")
print("-" * 70)

for nfeat in nfeatures_list:
    
    orb = cv2.ORB_create(nfeatures=nfeat)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

   
    if len(matches) >= 4: 
    
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

   
        matches_mask = mask.ravel().tolist()
        inliers = sum(matches_mask)
        total_matches = len(matches)
        inlier_ratio = inliers / total_matches if total_matches > 0 else 0

       
        is_success = "Yes" if inliers >= 10 and inlier_ratio > 0.1 else "No"

    else:
        inliers = 0
        total_matches = len(matches)
        inlier_ratio = 0
        is_success = "No"

   
    print(f"{nfeat:<10} | {len(kp1):<5} {len(kp2):<5} | {total_matches:<8} {inliers:<8} {inlier_ratio:.2f}   | {is_success}")