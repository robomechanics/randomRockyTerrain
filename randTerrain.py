import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise1,pnoise2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.ndimage import gaussian_filter

def randomSteps(xPoints,yPoints,numCells,noiseScale):
	centersX = np.random.uniform(size=numCells,low=np.min(xPoints),high=np.max(xPoints))
	centersY = np.random.uniform(size=numCells,low=np.min(yPoints),high=np.max(yPoints))
	centersZ = np.array([pnoise2(centersX[i]*noiseScale,centersY[i]*noiseScale) for i in range(len(centersX))])
	xPointsMatrix = np.matmul(np.matrix(xPoints).transpose(),np.ones((1,numCells)))
	yPointsMatrix = np.matmul(np.matrix(yPoints).transpose(),np.ones((1,numCells)))
	centersXMatrix = np.matmul(np.matrix(centersX).transpose(),np.ones((1,len(xPoints)))).transpose()
	centersYMatrix = np.matmul(np.matrix(centersY).transpose(),np.ones((1,len(yPoints)))).transpose()
	xDiff = xPointsMatrix - centersXMatrix
	yDiff = yPointsMatrix - centersYMatrix
	distMatrix = np.multiply(xDiff,xDiff)+np.multiply(yDiff,yDiff)
	correspondingCell = np.argmin(distMatrix,axis=1)
	return centersZ[correspondingCell]

mapWidth = 10
mapHeight = 10
numCells = 20
gridX = np.linspace(0,mapWidth,1000)
gridY = np.linspace(0,mapHeight,1000)
gridX,gridY = np.meshgrid(gridX,gridY)
gridX = gridX.reshape(-1)
gridY = gridY.reshape(-1)
gridZ = randomSteps(gridX,gridY,numCells,10)
gridX = gridX.reshape((1000,1000))
gridY = gridY.reshape((1000,1000))
gridZ = gridZ.reshape((1000,1000))
gridZ = 3*gaussian_filter(gridZ, sigma=10)
#gridHeightsmooth(gridZ,1)
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(gridX, gridY, gridZ, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_zlim(-1,9)
plt.show()