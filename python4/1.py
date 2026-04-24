import cv2
import os


path_box = 'C:/python/python4/box.png'
path_scene = 'C:/python/python4/box_in_scene.png'

img1 = cv2.imread(path_box)
img2 = cv2.imread(path_scene)


if img1 is None or img2 is None:
    print(f"错误：无法读取图像文件。")
    print(f"请检查路径：{path_box}")
    print(f"请检查路径：{path_scene}")
    exit()


orb = cv2.ORB_create(nfeatures=1000)


img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


kp1, des1 = orb.detectAndCompute(img1_gray, None)


img1_kp_color = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)


img1_result = cv2.drawKeypoints(img1_kp_color, kp1, None, color=(0, 255, 0), flags=0)


img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


kp2, des2 = orb.detectAndCompute(img2_gray, None)


img2_kp_color = cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2BGR)


img2_result = cv2.drawKeypoints(img2_kp_color, kp2, None, color=(0, 255, 0), flags=0)


print(f"图1 (box.png) 关键点数量: {len(kp1)}")
print(f"图2 (box_in_scene.png) 关键点数量: {len(kp2)}")


if des1 is not None:
    print(f"图1 (box.png) 描述子维度: {des1.shape[1]} (通常是32)")
if des2 is not None:
    print(f"图2 (box_in_scene.png) 描述子维度: {des2.shape[1]} (通常是32)")




cv2.imshow('Keypoints - Box', img1_result)
cv2.imshow('Keypoints - Scene', img2_result)


cv2.imwrite('result_box.png', img1_result)
cv2.imwrite('result_scene.png', img2_result)
print("结果已保存为 result_box.png 和 result_scene.png")


cv2.waitKey(0)
cv2.destroyAllWindows()