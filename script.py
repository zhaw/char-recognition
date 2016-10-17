import cv2
from make_real_data import get_font

while True:
    k = get_font()
    for i in range(5):
        cv2.imwrite('tmp_%d.jpg'%i, k[i])
    x = raw_input()
