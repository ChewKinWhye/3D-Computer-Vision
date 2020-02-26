import cv2
import k3d
import numpy as np
import scipy
import matplotlib.pyplot as plt

print('Hello world')

x = np.random.randn(100, 3).astype(np.float32)
colors = k3d.helpers.map_colors((np.sum(x**3-.1*x**2, axis=1)),
                                k3d.colormaps.basic_color_maps.WarmCool, [-2, .1]).astype(np.uint32)
point_size = 0.2
plot = k3d.plot()
plt_points = k3d.points(position=x, point_size=0.2, color=colors)
plot += plt_points
plt_points.shader='3d'
plot.display()


