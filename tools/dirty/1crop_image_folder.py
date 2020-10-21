import os
import cv2
import imutils
from glob import glob

input_folder = 'data/ytb/raw_image_before_crop/5/'
# 0
# x1, x2 = 67, 1200
# y1, y2 = 16, 655
# 1
# x1, x2 = 156, 968
# y1, y2 = 225, 561
# 2
# x1, x2 = 170, 1109
# y1, y2 = 97, 613
# 3
# x1, x2 = 105, 1147
# y1, y2 = 35, 706
# 4
# x1, x2 = 296, 971
# y1, y2 = 28, 642
# 5
x1, x2 = 25, 1278
y1, y2 = 244, 558
# 5 - 2
# x1, x2 = 195, 1035
# y1, y2 = 188, 577
# 6 delete
# x1, x2 = 273, 946
# y1, y2 = 33, 626
# 7
# x1, x2 = 198, 1134
# y1, y2 = 41, 572
# for folder 6, there are actually 2 configurations

output_folder = 'data/ytb/raw_images/5/'
os.makedirs(output_folder, exist_ok=True)

image_list = sorted(glob(input_folder + '*'))
cnt = 0
for image_name in image_list:
    im = cv2.imread(image_name)
    im = im[y1:y2, x1:x2]
    tar_name = os.path.join(output_folder, f'{cnt:06}.jpg')
    cv2.imwrite(tar_name, im)
    cnt += 1
    print(cnt)
