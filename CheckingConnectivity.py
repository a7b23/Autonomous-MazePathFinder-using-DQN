import numpy as np

def Valid(i,limit,flag,value) :
	if i>=0 and i<=limit-1 and value[i] == 255 and flag[i] == 0 :
		return True 
	return False	

def Checking(graph) :
	stack = []
	x,y = graph.shape
	graph1 = np.reshape(graph,-1)
	limit = len(graph1)
	flag = np.zeros([len(graph1)])

	stack.append(0)
	flag[0] = 1
	while (len(stack)>0 and not flag[limit-1] == 1) :
		element = stack.pop()
		#print element
		'''
		if Valid(element-1,limit,flag,graph1) and not element%x == 0 :
			stack.append(element-1)
			flag[element-1] = 1
		'''	
		if Valid(element+1,limit,flag,graph1) and not (element+1)%x == 0:
			stack.append(element+1)
			flag[element+1] = 1
		'''	
		if Valid(element-x,limit,flag,graph1) :
			stack.append(element-x)
			flag[element-x] = 1
		'''	
		if Valid(element+x,limit,flag,graph1) :
			stack.append(element+x)
			flag[element+x] = 1

	if flag[limit-1] == 1 :
		return True
	return False			

						

'''
a = np.zeros([25,25])

a[0][1] = 1
a[1][1] = 1
a[2][0] = 1
if Checking(a) :
	print 'True'
else :
	print 'False'
'''		