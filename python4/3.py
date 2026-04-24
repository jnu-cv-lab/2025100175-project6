import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread('C:/python/python4/box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('C:/python/python4/box_in_scene.png', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)


num_initial_matches = 50
if len(matches) > num_initial_matches:
    good_matches_for_ransac = matches[:num_initial_matches]
else:
    good_matches_for_ransac = matches

print(f"参与 RANSAC 计算的初始匹配点数量: {len(good_matches_for_ransac)}")


src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches_for_ransac]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches_for_ransac]).reshape(-1, 1, 2)


M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


matches_mask = mask.ravel().tolist()


total_matches = len(good_matches_for_ransac)
inlier_count = int(np.sum(matches_mask))
inlier_ratio = inlier_count / total_matches if total_matches > 0 else 0


print("-" * 30)
print("任务 3 结果:")
print(f"Homography 矩阵:\n{M}")
print(f"总匹配数量 (参与计算的): {total_matches}")
print(f"RANSAC 内点数量: {inlier_count}")
print(f"内点比例: {inlier_ratio:.2%}")
print("-" * 30)


img_result = cv2.drawMatches(
    img1, kp1, img2, kp2,
    good_matches_for_ransac,
    None,
    matchesMask=matches_mask, 
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


plt.figure(figsize=(15, 8))
plt.imshow(img_result,)
plt.title(f'RANSAC Inlier Matches (Inliers: {inlier_count}, Ratio: {inlier_ratio:.2%})')
plt.axis('off')
plt.show();import cv2
import numpy as np
import matplotlib.pyplot as plt


img1 = cv2.imread('C:/python/python4/box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('C:/python/python4/box_in_scene.png', cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)


num_initial_matches = 50
if len(matches) > num_initial_matches:
    good_matches_for_ransac = matches[:num_initial_matches]
else:
    good_matches_for_ransac = matches

print(f"参与 RANSAC 计算的初始匹配点数量: {len(good_matches_for_ransac)}")


src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches_for_ransac]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches_for_ransac]).reshape(-1, 1, 2)


M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)


matches_mask = mask.ravel().tolist()


total_matches = len(good_matches_for_ransac)
inlier_count = int(np.sum(matches_mask))
inlier_ratio = inlier_count / total_matches if total_matches > 0 else 0


print("-" * 30)
print("任务 3 结果:")
print(f"Homography 矩阵:\n{M}")
print(f"总匹配数量 (参与计算的): {total_matches}")
print(f"RANSAC 内点数量: {inlier_count}")
print(f"内点比例: {inlier_ratio:.2%}")
print("-" * 30)


img_result = cv2.drawMatches(
    img1, kp1, img2, kp2,
    good_matches_for_ransac,
    None,
    matchesMask=matches_mask,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


plt.figure(figsize=(15, 8))
plt.imshow(img_result,)
plt.title(f'RANSAC Inlier Matches (Inliers: {inlier_count}, Ratio: {inlier_ratio:.2%})')
plt.axis('off')
plt.show()