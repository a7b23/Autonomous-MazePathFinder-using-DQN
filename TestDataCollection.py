import cv2
import numpy as np
import random
import CheckingConnectivity
import os
import shutil

numberOfBlocks = 75
gridSize = [25,25,3]
totalGames = 20

points = []
def assignBlocks(grid,xlimit,limit,numberOfBlocks,flag) :
	x1 = random.sample(range(1,limit-1),numberOfBlocks)
	for i in range(numberOfBlocks) :
		grid[int(x1[i]/xlimit)][x1[i]%xlimit][0] = 0
		grid[int(x1[i]/xlimit)][x1[i]%xlimit][1] = 0
		grid[int(x1[i]/xlimit)][x1[i]%xlimit][2] = 0	
	while not CheckingConnectivity.Checking(grid[:,:,0]) :
		print 'not Valid Connection'
		for i in range(numberOfBlocks) :
			grid[int(x1[i]/xlimit)][x1[i]%xlimit][0] = 255
			grid[int(x1[i]/xlimit)][x1[i]%xlimit][1] = 255
			grid[int(x1[i]/xlimit)][x1[i]%xlimit][2] = 255
		x1 = random.sample(range(1,limit-1),numberOfBlocks)
		for i in range(numberOfBlocks) :
			grid[int(x1[i]/xlimit)][x1[i]%xlimit][0] = 0
			grid[int(x1[i]/xlimit)][x1[i]%xlimit][1] = 0
			grid[int(x1[i]/xlimit)][x1[i]%xlimit][2] = 0	

	for i in x1 :
		flag[i] = 1


folderName = 'testImages'

if os.path.exists(folderName):
	shutil.rmtree(folderName)

os.mkdir(folderName)


for t in range(totalGames) :
	if t%10 == 0 :
		print('%d steps reached'%t)
	a = np.zeros(gridSize)
	x,y,z = a.shape

	for i in range(x) :
		for j in range(y) :
			for k in range(z) :
				a[i][j][k] = 255


	#cv2.imshow('image',a)
	

	limit = x*y
	flag = np.zeros([limit])
	assignBlocks(a,x,limit,numberOfBlocks,flag)
	for i in range(limit) :
		if not flag[i] == 0 :
			#points[t][i] = -100
			points.append(-100)
		elif i== limit-1 :
			points.append(100)
		else :
			points.append(0)		
	#points[t][limit-1] = 100		
	a[x-1][y-1][0] = 40
	a[x-1][y-1][1] = 222
	a[x-1][y-1][2] = 20

	for i in range(x) :
		for j in range(y) :
			a[i][j][0] = 223
			a[i][j][1] = 27
			a[i][j][2] = 20
			cv2.imwrite(os.path.join(folderName,"image_"+str(t)+"_"+str(gridSize[0]*i+j)+".png"),a)
			if flag[i*x+j] == 0 : 
				a[i][j][0] = 255
				a[i][j][1] = 255
				a[i][j][2] = 255
			else :
				a[i][j][0] = 0
				a[i][j][1] = 0
				a[i][j][2] = 0	
	
	
np.savetxt('testPoints.txt',points)
