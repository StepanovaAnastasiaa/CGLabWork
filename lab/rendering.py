import numpy as np
from PIL import Image
import math as math
import re


def bar_coor(x0, y0, x1, y1, x2, y2, x, y):
    l0 = ((x-x2)*(y1-y2)-(x1-x2)*(y-y2))/((x0-x2)*(y1-y2)-(x1-x2)*(y0-y2))
    l1 = -((x0-x2)*(y-y2)-(x-x2)*(y0-y2))/((y0-y2)*(x1-x2)-(x0-x2)*(y1-y2))
    l2 = 1-l0-l1
    return[l0, l1, l2]

def draw_triangle(img_mat, z_buffer, x0, y0, x1, y1, x2, y2, color):
    xmin=int(min(x0, x1, x2))
    if(xmin<0):
        xmin=0
    ymin = int(min(y0, y1, y2))
    if (ymin < 0):
        ymin = 0
    xmax = int(max(x0, x1, x2))
    if (xmax > 9999):
        xmax = 10000
    ymax = int(max(y0, y1, y2))
    if (ymax > 9999):
        ymax = 10000

    for i in range(xmin, xmax):
        for j in range(ymin, ymax):
            lambdas = bar_coor(x0, y0, x1, y1, x2, y2, i, j)
            if all(l > 0 for l in lambdas):
                # Вычисляем z-координату
                z_hat = lambdas[0] * z0 + lambdas[1] * z1 + lambdas[2] * z2

                # Проверка на z-буфер
                if z_hat < z_buffer[i, j]:
                    img_mat[i, j] = color
                    z_buffer[i, j] = z_hat
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


def compute_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    # Определяем два вектора из вершин треугольника
    vec_a = np.array([x1 - x2, y1 - y2, z1 - z2])
    vec_b = np.array([x1 - x0, y1 - y0, z1 - z0])
    # Вычисляем нормаль как векторное произведение
    normal = np.cross(vec_a, vec_b)
    return normal


img = np.zeros((10000, 10000,3), dtype=np.uint8)
img[0:10000,0:10000,1]=120
z_buffer = np.full((10000, 10000), np.inf)  # Инициализация z-буфера

v=[]
with open('C:\\Users\\Admin\\Desktop\\model_1.obj') as file:
    for str in file:
        splitted_str=str.split()
        if (splitted_str[0]=='v'):
            v.append([float(splitted_str[1]), float(splitted_str[2]), float(splitted_str[3])])

for vertex in v:
    img[int(10000*vertex[0]+5000),int(10000*vertex[1]+5000)]=[255, 255, 255]

f=[]
with open('C:\\Users\\Admin\\Desktop\\model_1.obj') as file:
    for str2 in file:
        splitted_str2=re.split(r'[ /]', str2)

        if (splitted_str2[0]=='f'):
            f.append([int(splitted_str2[1]), int(splitted_str2[4]), int(splitted_str2[7])])
light_direction = np.array([0, 0, 1])
light_magnitude = np.linalg.norm(light_direction)

for face in f:
    x0=v[face[0] - 1][0]
    y0 = v[face[0] - 1][1]
    z0 = v[face[0] - 1][2]

    x1=v[face[1] - 1][0]
    y1 = v[face[1] - 1][1]
    z1 = v[face[1] - 1][2]
    # draw_line(img, 30000*y0+5000, 30000*x0+5000, 30000*y1+5000, 30000*x1+5000, [255,255,255])
    x2=v[face[2]-1][0]
    y2=v[face[2]-1][1]
    z2 = v[face[2] - 1][2]
    # draw_line(img, 30000 * y0 + 5000, 30000 * x0 + 5000, 30000 * y2 + 5000, 30000 * x2 + 5000, [255, 255, 255])
    # draw_line(img, 30000 * y1 + 5000, 30000 * x1 + 5000, 30000 * y2 + 5000, 30000 * x2 + 5000, [255, 255, 255])
    normal = compute_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)

    # Вычисляем косинус угла падения
    cos_angle = np.dot(normal, light_direction)
    normal_magnitude = np.linalg.norm(normal)

    # Отрисовываем треугольник только если cos_angle < 0
    if cos_angle / (normal_magnitude * light_magnitude) < 0:
        color_value = int(-255 * (cos_angle / (normal_magnitude * light_magnitude)))
        #color_value = max(0, min(255, color_value))
        color = (color_value, 255, 255)

        draw_triangle(img, z_buffer, 10000*x0+5000,10000*y0+5000, 10000*x1+5000, 10000*y1+5000, 10000*x2+5000, 10000*y2+5000, color)
img = Image.fromarray(img, mode='RGB')
im_rotate = img.rotate(90)
im_rotate.save('image.jpg')