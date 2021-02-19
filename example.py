import tps
import numpy as np
import matplotlib.pyplot as plt

samp = np.linspace(-2, 2, 4)
xx, yy = np.meshgrid(samp, samp)

# make source surface, get uniformed distributed control points
source_xy = np.stack([xx, yy], axis=2).reshape(-1, 2)

# make deformed surface
yy[:, [0, 3]] *=2
deform_xy = np.stack([xx, yy], axis=2).reshape(-1, 2)

# get coefficient, use class
trans = tps.TPS(source_xy, deform_xy)

# make other points a left-bottom to upper-right line on source surface
samp2 = np.linspace(-1.8, 1.8, 10)
test_xy = np.tile(samp2, [2, 1]).T

# get transformed points
transformed_xy = trans(test_xy)

plt.scatter(source_xy[:,0], source_xy[:,1])
plt.scatter(test_xy[:,0], test_xy[:,1])
plt.show()

plt.scatter(deform_xy[:,0], deform_xy[:,1])
plt.scatter(transformed_xy[:,0], transformed_xy[:,1])
plt.show()
