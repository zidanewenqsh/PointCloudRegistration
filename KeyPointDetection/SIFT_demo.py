import cv2
import matplotlib.pyplot as plt

def show_image(img, title="Image", cmap_type='gray'):
    plt.imshow(img, cmap=cmap_type)
    plt.title(title)
    plt.axis("off")
    plt.show()

# 读取图像
image_path = r"D:\MyProjects\PointCloudRegistration\KeyPointDetection\datas\cat01.jpg"  # 替换为你的图像路径
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 检测SIFT特征点
keypoints, descriptors = sift.detectAndCompute(gray_image, None)

# 绘制关键点
keypoint_image = cv2.drawKeypoints(gray_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 显示图像
show_image(keypoint_image, "SIFT Keypoints")

print(f"Total keypoints detected: {len(keypoints)}")
