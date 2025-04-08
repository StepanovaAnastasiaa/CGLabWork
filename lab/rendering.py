import numpy as np
from PIL import Image
import math as math
import re

w=8000
h=8000
def rot_trans(ver, alpha, beta, gamma, tx, ty, tz):
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)
    # Матрица поворота вокруг X
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(alpha), math.sin(alpha)],
        [0, -math.sin(alpha), math.cos(alpha)]
    ])

    # Матрица поворота вокруг Y
    Ry = np.array([
        [math.cos(beta), 0, math.sin(beta)],
        [0, 1, 0],
        [-math.sin(beta), 0, math.cos(beta)]
    ])

    # Матрица поворота вокруг Z
    Rz = np.array([
        [math.cos(gamma), math.sin(gamma), 0],
        [-math.sin(gamma), math.cos(gamma), 0],
        [0, 0, 1]
    ])

    # Общая матрица поворота
    R = np.dot(Rx, np.dot(Ry, Rz))

    # Применяем преобразование к каждой вершине
    transformed = np.dot(ver, R.T) + np.array([tx, ty, tz])
    return transformed


def bar_coor(x0, y0, x1, y1, x2, y2, x, y):
    l0 = ((x-x2)*(y1-y2)-(x1-x2)*(y-y2))/((x0-x2)*(y1-y2)-(x1-x2)*(y0-y2))
    l1 = -((x0-x2)*(y-y2)-(x-x2)*(y0-y2))/((y0-y2)*(x1-x2)-(x0-x2)*(y1-y2))
    l2 = 1-l0-l1
    return[l0, l1, l2]

def draw_triangle(img_mat, z_buffer, x0, y0, x1, y1, x2, y2, color):
    xmin = math.floor(min(x0, x1, x2))
    if xmin < 0:
        xmin = 0
    ymin = math.floor(min(y0, y1, y2))
    if ymin < 0:
        ymin = 0
    xmax = math.ceil(max(x0, x1, x2))
    if xmax > w-1:
        xmax = w
    ymax = math.ceil(max(y0, y1, y2))
    if ymax > h-1:
        ymax = h

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


img = np.zeros((h, w, 3), dtype=np.uint8)
img[0:w,0:h,1]=120
z_buffer = np.full((w, h), np.inf)  # Инициализация z-буфера

ver=[]
with open('C:\\Users\\Admin\\Desktop\\model_1.obj') as file:
    for str in file:
        splitted_str=str.split()
        if (splitted_str[0]=='v'):
            ver.append([float(splitted_str[1]), float(splitted_str[2]), float(splitted_str[3])])

v = rot_trans(ver, 0, 180, 90, 0.05, 0.011, 1)
m=7000
u0=w/2
v0=h/2
for vertex in v:
    img[int((m*vertex[0])/vertex[2]+u0),int((m*vertex[1])/vertex[2]+v0)]=[255, 255, 255]

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
    x2=v[face[2]-1][0]
    y2=v[face[2]-1][1]
    z2 = v[face[2] - 1][2]
    normal = compute_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)

    # Вычисляем косинус угла падения
    cos_angle = np.dot(normal, light_direction)
    normal_magnitude = np.linalg.norm(normal)

    # Отрисовываем треугольник только если cos_angle < 0
    if cos_angle / (normal_magnitude * light_magnitude) < 0:
        color_value = int(-255 * (cos_angle / (normal_magnitude * light_magnitude)))
        color = (color_value, 255, 255)

        draw_triangle(img, z_buffer, m*x0/z0+u0,m*y0/z0+v0, m*x1/z1+u0, m*y1/z1+v0, m*x2/z2+u0, m*y2/z2+v0, color)
img = Image.fromarray(img, mode='RGB')

img.save('image.jpg')