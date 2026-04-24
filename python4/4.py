import cv2
import numpy as np


img1_color = cv2.imread('C:/python/python4/box.png')
img2_color = cv2.imread('C:/python/python4/box_in_scene.png')


img1_gray = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)


orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1_gray, None)
kp2, des2 = orb.detectAndCompute(img2_gray, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)


if len(matches) > 4:
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is not None:
       
        h, w = img1_gray.shape
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)

        
        dst = cv2.perspectiveTransform(pts, H)

        
        img2_result = cv2.polylines(img2_color, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)

      
        print("定位成功！")
        print("单应性矩阵 H:\n", H)

       
        cv2.imshow('Target Localization', img2_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
        cv2.imwrite('C:/python/python4/result_localization.png', img2_result)

    else:
        print("错误：无法计算单应性矩阵 H。")
else:
    print(f"错误：匹配点不足 (只有 {len(matches)} 个)，无法进行定位。")