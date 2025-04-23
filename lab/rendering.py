import numpy as np
from PIL import Image
import math as math

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

def draw_triangle(img_mat, z_buffer, x0, y0, z0, u0, v0, x1, y1,  z1, u1, v1, x2, y2, z2, u2, v2, i0, i1, i2, texture):
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
                    # Интерполяция текстурных координат
                    u = lambdas[0] * u0 + lambdas[1] * u1 + lambdas[2] * u2
                    v = lambdas[0] * v0 + lambdas[1] * v1 + lambdas[2] * v2

                    # Получение цвета из текстуры (с проверкой границ)
                    tex_x = min(max(0, int(u * (texture.width - 1))), texture.width - 1)
                    tex_y = min(max(0, int((1-v) * (texture.height - 1))), texture.height - 1)
                    tex_color = texture.getpixel((tex_x, tex_y))

                    color_value = int( -255*(lambdas[0]*i0+lambdas[1]*i1+lambdas[2]*i2))

                    color = (max(0,min((tex_color[0]/255)*color_value, 255)), max(0,min((tex_color[1]/255)*color_value, 255)), max(0,min((tex_color[2]/255)*color_value, 255)))
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


def quaternion_multiply(q1, q2):
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2
    return np.array([
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2
    ])

def cvant(ver, alpha, beta, gamma, tx, ty, tz):
    alpha = np.radians(alpha)/2
    beta = np.radians(beta)/2
    gamma = np.radians(gamma)/2

    qx = np.array([math.cos(alpha), math.sin(alpha), 0, 0])
    qy = np.array([math.cos(beta), 0, math.sin(beta), 0])
    qz = np.array([math.cos(gamma), 0, 0, math.sin(gamma)])

    q = quaternion_multiply(qx, quaternion_multiply(qy, qz))

    q = q / np.linalg.norm(q)

    a, b, c, d = q
    R = np.array([
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d]
    ])
    transform = np.dot(ver, R.T) + np.array([tx, ty, tz])
    return transform
img = np.zeros((h, w, 3), dtype=np.uint8)
img[0:w,0:h,1]=120
z_buffer = np.full((w, h), np.inf)  # Инициализация z-буфера

texture = Image.open('C:\\Users\\Admin\\Desktop\\bunny-atlas.jpg').convert('RGB')
def draw_model(img, z_buffer, texture, w, h, m, alpha, beta, gamma, tx, ty, tz):
    ver=[]
    with open('C:\\Users\\Admin\\Desktop\\model_1.obj') as file:
        for str in file:
            splitted_str=str.split()
            if (splitted_str[0]=='v'):
                k=[]
                for g in splitted_str[1:]:
                    k.append(float(g))
                ver.append(k)

    #v = rot_trans(ver, alpha, beta, gamma, tx, ty, tz)
    v=cvant(ver, alpha, beta, gamma, tx, ty, tz)
    #m=7000
    u0=w/2
    v0=h/2
    for vertex in v:
        img[int((m*vertex[0])/vertex[2]+u0),int((m*vertex[1])/vertex[2]+v0)]=[255, 255, 255]

    f=[]
    vt = []
    vt_num = []
    with open('C:\\Users\\Admin\\Desktop\\model_1.obj') as file:
        for str2 in file:
            splitted_str2=str2.split()
            if (splitted_str2[0] == 'f'):
                res_f = []
                res_vt = []
                for sp in splitted_str2[1:]:
                    s = sp.split('/')
                    res_f.append(int(s[0])-1)
                    res_vt.append(int(s[1])-1)

                f.append(res_f)
                vt_num.append(res_vt)
            if (splitted_str2[0] == 'vt'):
                vt.append([float(splitted_str2[1]), float(splitted_str2[2])])


    light_direction = np.array([0, 0, 1])
    light_magnitude = np.linalg.norm(light_direction)

    f_n = [[0] * 2 for i in range(len(f))]
    for f1 in f:
        x0 = v[f1[0]][0]
        y0 = v[f1[0]][1]
        z0 = v[f1[0]][2]

        x1 = v[f1[1]][0]
        y1 = v[f1[1]][1]
        z1 = v[f1[1]][2]
        x2 = v[f1[2]][0]
        y2 = v[f1[2]][1]
        z2 = v[f1[2]][2]
        normal = compute_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        normal = normal/np.linalg.norm(normal)
        f_n[f1[0]][1]+= normal
        f_n[f1[1]][1] += normal
        f_n[f1[2]][1] += normal
        f_n[f1[0]][0] += 1
        f_n[f1[1]][0] += 1
        f_n[f1[2]][0] += 1
    for i in range(len(f_n)):
        if f_n[i][0]!=0:
            f_n[i][1]/= f_n[i][0]
    for face, vn in zip(f, vt_num):

        x0=v[face[0]][0]
        y0 = v[face[0]][1]
        z0 = v[face[0]][2]
        tu0 = vt[vn[0]][0]
        tv0 = vt[vn[0]][1]

        x1=v[face[1]][0]
        y1 = v[face[1]][1]
        z1 = v[face[1]][2]
        tu1 = vt[vn[1]][0]
        tv1 = vt[vn[1]][1]

        x2=v[face[2]][0]
        y2=v[face[2]][1]
        z2 = v[face[2]][2]
        tu2 = vt[vn[2]][0]
        tv2 = vt[vn[2]][1]
        normal = compute_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
        normal = normal / np.linalg.norm(normal)
        i0 = np.dot(f_n[face[0]][1], light_direction)/ (np.linalg.norm(f_n[face[0]][1]) * np.linalg.norm(light_direction))
        i1 = np.dot(f_n[face[1]][1], light_direction) / (np.linalg.norm(f_n[face[1]][1]) * np.linalg.norm(light_direction))
        i2 = np.dot(f_n[face[2]][1], light_direction) / ( np.linalg.norm(f_n[face[2]][1]) * np.linalg.norm(light_direction))
    # Вычисляем косинус угла падения
        cos_angle = np.dot(normal, light_direction)
        normal_magnitude = np.linalg.norm(normal)

        # Отрисовываем треугольник только если cos_angle < 0
        if cos_angle / (normal_magnitude * light_magnitude) < 0:

            draw_triangle(img, z_buffer, m*x0/z0+u0,m*y0/z0+v0, z0, tu0, tv0, m*x1/z1+u0, m*y1/z1+v0, z1, tu1, tv1, m*x2/z2+u0, m*y2/z2+v0, z2, tu2, tv2, i0, i1, i2, texture)
    img = Image.fromarray(img, mode='RGB')

    img.save('image.jpg')

draw_model(img, z_buffer, texture, w, h, 7000, 0, 180, 270, 0.03, 0.014, 0.6)
draw_model(img, z_buffer, texture, w, h, 7000, 150, 160, 90, 0.1, 0.02, 0.5)
