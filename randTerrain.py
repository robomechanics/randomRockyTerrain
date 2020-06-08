import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise1,pnoise2
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mpl_toolkits import mplot3d
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.ndimage import gaussian_filter
from stl import mesh

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

def generateTerrainMesh(xGrid,yGrid,zGrid):
	bottomHeight = np.min(zGrid)-0.1
	zGrid[0,:] = bottomHeight
	zGrid[-1,:] = bottomHeight
	zGrid[:,0] = bottomHeight
	zGrid[:,-1] = bottomHeight
	numTriangles = 2*(xGrid.shape[0]-1)*(xGrid.shape[1]-1)+2
	data = np.zeros(numTriangles,dtype=mesh.Mesh.dtype)
	triangleCounter = 0
	for i in range(xGrid.shape[0]-1):
		for j in range(xGrid.shape[1]-1):
			data['vectors'][triangleCounter] = np.array([[xGrid[i,j],yGrid[i,j],zGrid[i,j]],
														[xGrid[i,j+1],yGrid[i,j+1],zGrid[i,j+1]],
														[xGrid[i+1,j+1],yGrid[i+1,j+1],zGrid[i+1,j+1]]])
			triangleCounter+=1
			data['vectors'][triangleCounter] = np.array([[xGrid[i,j],yGrid[i,j],zGrid[i,j]],
														[xGrid[i+1,j],yGrid[i+1,j],zGrid[i+1,j]],
														[xGrid[i+1,j+1],yGrid[i+1,j+1],zGrid[i+1,j+1]]])
			triangleCounter+=1
	data['vectors'][triangleCounter] = np.array([[xGrid[0,0],yGrid[0,0],zGrid[0,0]],
												[xGrid[0,-1],yGrid[0,-1],zGrid[0,-1]],
												[xGrid[-1,-1],yGrid[-1,-1],zGrid[-1,-1]]])
	triangleCounter+=1
	data['vectors'][triangleCounter] = np.array([[xGrid[0,0],yGrid[0,0],zGrid[0,0]],
												[xGrid[-1,0],yGrid[-1,0],zGrid[-1,0]],
												[xGrid[-1,-1],yGrid[-1,-1],zGrid[-1,-1]]])
	terrainMesh = mesh.Mesh(data, remove_empty_areas=False)
	terrainMesh.save('terrain.stl')
	return terrainMesh


def plotTerrainMesh(terrainMesh):
	fig = plt.figure()
	ax = mplot3d.Axes3D(fig)
	# Load the STL files and add the vectors to the plot
	ax.add_collection3d(mplot3d.art3d.Poly3DCollection(terrainMesh.vectors))

	# Auto scale to the mesh size
	scale = terrainMesh.points.flatten()
	ax.auto_scale_xyz(scale, scale, scale)

	# Show the plot to the screen
	plt.show()


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
terrainMesh = generateTerrainMesh(gridX,gridY,gridZ)
#plotTerrainMesh(terrainMesh)
#gridHeightsmooth(gridZ,1)
"""
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(gridX, gridY, gridZ, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.set_zlim(-1,9)
plt.show()"""