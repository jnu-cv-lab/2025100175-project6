import cv2
import matplotlib.pyplot as plt


img1 = cv2.imread('C:/python/python4/box.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('C:/python/python4/box_in_scene.png', cv2.IMREAD_GRAYSCALE)


orb = cv2.ORB_create(nfeatures=1000)


kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


matches = bf.match(des1, des2)


matches = sorted(matches, key=lambda x: x.distance)


print(f"总匹配数量: {len(matches)}")


num_show = 50

if len(matches) < num_show:
    num_show = len(matches)

img_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:num_show], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


plt.figure(figsize=(12, 6))
plt.imshow(img_matches)
plt.title(f'ORB Matches (Top {num_show})')
plt.axis('off')
plt.show()


cv2.imwrite('C:/python/python4/orb_matches_result.png', img_matches)
print("结果已保存为 orb_matches_result.png")