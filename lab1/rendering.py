import numpy as np
from PIL import Image
import math as math
import re

def draw_line(img_mat, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True

    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y = y0
    dy = 2*abs(y1 - y0)
    derror = 0
    y_update = 1 if y1 > y0 else -1

    for x in range(int(x0), int(x1)):
        t = (x - x0) / (x1 - x0)
        y = round((1 - t) * y0 + t * y1)
        if (xchange):
            img_mat[x, y] = color
        else:
            img_mat[y, x] = color
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2*(x1 - x0)
        y += y_update

img = np.zeros((1000, 1000,3), dtype=np.uint8)
img[0:1000,0:1000,1]=120

v=[]
with open('C:\\Users\\Admin\\Desktop\\model_1.obj') as file:
    for str in file:
        splitted_str=str.split()
        if (splitted_str[0]=='v'):
            v.append([float(splitted_str[1]), float(splitted_str[2]), float(splitted_str[3])])

for vertex in v:
    img[int(5000*vertex[0]+500),int(5000*vertex[1]+500)]=[255, 255, 255]

f=[]
with open('C:\\Users\\Admin\\Desktop\\model_1.obj') as file:
    for str2 in file:
        splitted_str2=re.split(r'[ /]', str2)

        if (splitted_str2[0]=='f'):
            f.append([int(splitted_str2[1]), int(splitted_str2[4]), int(splitted_str2[7])])
for face in f:
    x0=v[face[0] - 1][0]
    y0 = v[face[0] - 1][1]

    x1=v[face[1] - 1][0]
    y1 = v[face[1] - 1][1]
    draw_line(img, 5000*y0+500, 5000*x0+500, 5000*y1+500, 5000*x1+500, [255,255,255])
    x2=v[face[2]-1][0]
    y2=v[face[2]-1][1]
    draw_line(img, 5000 * y0 + 500, 5000 * x0 + 500, 5000 * y2 + 500, 5000 * x2 + 500, [255, 255, 255])
    draw_line(img, 5000 * y1 + 500, 5000 * x1 + 500, 5000 * y2 + 500, 5000 * x2 + 500, [255, 255, 255])
img = Image.fromarray(img, mode='RGB')
im_rotate = img.rotate(90)
im_rotate.save('image.jpg')