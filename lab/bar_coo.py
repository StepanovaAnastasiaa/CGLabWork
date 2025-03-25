import numpy as np
from PIL import Image
def bar_coor(x0, y0, x1, y1, x2, y2, x, y):
    l0 = ((x-x2)*(y1-y2)-(x1-x2)*(y-y2))/((x0-x2)*(y1-y2)-(x1-x2)*(y0-y2))
    l1 = -((x0-x2)*(y-y2)-(x-x2)*(y0-y2))/((y0-y2)*(x1-x2)-(x0-x2)*(y1-y2))
    l2 = 1-l0-l1
    return[l0, l1, l2]

def draw_triangle(img_mat, x0, y0, x1, y1, x2, y2, color):
    xmin=int(min(x0, x1, x2))
    if(xmin<0):
        xmin=0
    ymin = int(min(y0, y1, y2))
    if (ymin < 0):
        ymin = 0
    xmax = int(max(x0, x1, x2))
    if (xmax > 999):
        xmax = 1000
    ymax = int(max(y0, y1, y2))
    if (ymax > 999):
        ymax = 1000
    print(xmin, ymin, xmax, ymax)
    for i in range(xmax):
        for j in range(ymax):
            if(min(bar_coor(x0,y0,x1,y1,x2,y2, i, j))>0):
                img_mat[i, j]=color

img = np.zeros((1000, 1000,3), dtype=np.uint8)
img[0:1000,0:1000,1]=200

draw_triangle(img, 500, 465.11, 1000, 130.55, 100.87, 670.98, [23,45,100])

img = Image.fromarray(img, mode='RGB')
img.save('triangle.jpg')