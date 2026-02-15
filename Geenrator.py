import cv2
from random import uniform, randint
import numpy as np

sign_path = "/home/mik/SchoolProjects/NTO_TOURS/from_zero/Syntetic_data/Signs/bump.jpg"
video_path = "/home/mik/SchoolProjects/NTO_TOURS/from_zero/Syntetic_data/TrashVideo/video.MOV"

video = cv2.VideoCapture(video_path)
sign = cv2.imread(sign_path)

sign_height = sign.shape[0]
sign_width = sign.shape[1]

bg = video.read()[1]
bg = cv2.resize(bg, (640, 640))

bg_height = bg.shape[0]
bg_width = bg.shape[1]

scale_coef = uniform(0.05, 0.25)
sign_new_height = int(scale_coef * sign_height)
sign_new_width = int(scale_coef * sign_width)

sign_resized = cv2.resize(sign, (int(sign_new_width), int(sign_new_height)))

print(sign_new_height, bg_height)
x_offset = randint(0, bg_height - sign_new_height)
y_offset = randint(0, bg_width - sign_new_width)

mask = cv2.inRange(sign_resized, np.zeros(3), np.ones(3))
mask = cv2.bitwise_not(mask)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3)))
mask = cv2.GaussianBlur(mask, (3, 3), 0)
mask = cv2.merge([mask, mask, mask]) / 255.0

bg_to_sign = bg[x_offset:x_offset + sign_new_height, y_offset:y_offset + sign_new_width]
sign_without_black = sign_resized.astype("float32") * mask + bg_to_sign.astype("float32") * (1 - mask)

sign_height = sign_resized.shape[0]
sign_width = sign_resized.shape[1]

bg[x_offset:x_offset + sign_new_height, y_offset:y_offset + sign_new_width] = sign_without_black.astype("uint8")

cv2.imshow("sign", bg)
cv2.waitKey(0)
