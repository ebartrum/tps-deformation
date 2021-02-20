import tps
import numpy as np
import matplotlib.pyplot as plt
import torch

#TODO: this needs to work on a batch

samp = np.linspace(-2, 2, 4)
xx, yy = np.meshgrid(samp, samp)

# make source surface, get uniformed distributed control points
source_xy = torch.from_numpy(np.stack([xx, yy], axis=2).reshape(-1, 2))

# make deformed surface
yy[:, [0, 3]] *=2
deform_xy = torch.from_numpy(np.stack([xx, yy], axis=2).reshape(-1, 2))

# get coefficient, use class
deform_xy = torch.stack(2*[deform_xy])
source_xy = torch.stack(2*[source_xy])
trans = tps.TPS(deform_xy, source_xy)

# make other points a left-bottom to upper-right line on source surface
samp2 = torch.from_numpy(np.linspace(-1.8, 1.8, 10))
test_xy = torch.from_numpy(np.tile(samp2, [2, 1]).T)

# get transformed points
test_xy = torch.stack(2*[test_xy])
transformed_xy = trans(test_xy)

# plot points
plt.scatter(deform_xy[0,:,0], deform_xy[0,:,1])
plt.scatter(test_xy[0,:,0], test_xy[0,:,1])
plt.show()
plt.scatter(source_xy[0,:,0], source_xy[0,:,1])
plt.scatter(transformed_xy[0,:,0], transformed_xy[0,:,1])
plt.show()
plt.scatter(deform_xy[1,:,0], deform_xy[1,:,1])
plt.scatter(test_xy[1,:,0], test_xy[1,:,1])
plt.show()
plt.scatter(source_xy[1,:,0], source_xy[1,:,1])
plt.scatter(transformed_xy[1,:,0], transformed_xy[1,:,1])
plt.show()
