import numpy as np
from PIL import Image
def bar_coor(x0, y0, x1, y1, x2, y2, x, y):
    l0 = ((x-x2)*(y1-y2)-(x1-x2)*(y-y2))/((x0-x2)*(y1-y2)-(x1-x2)*(y0-y2))
