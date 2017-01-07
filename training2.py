import numpy as np
import tensorflow as tf
import cv2
import ReCalculateTargets2

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.05)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.05, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

imageSize = [25,25,3]
batchSize = 50
games = 200
limit = imageSize[0]*imageSize[1]
totalImages = games*imageSize[0]*imageSize[1]

#gamesArr = np.arange(totalImages)

start = 0
def randomiseGames() :
	global gamesArr
	np.random.shuffle(gamesArr)

def getInput(index) :
	global gamesArr
	global imageSize
	temp = imageSize[0]*imageSize[1]
	img = cv2.imread('trainImages/image_'+str(int(index/temp))+'_'+str(index%temp)+'.png')
	return img
def getRightOutput(index,points,target) :
	global gamesArr
	global imageSize

	pos = index%limit
	if (pos+1)%imageSize[0] == 0 or points[index+1] == -100:
		output = -100

	elif points[index+1] == 100 :
		#print 'hi'
		output = 100	

	else :

		output = points[index+1] - points[index]+	max(target[index+1])
	#if index == 13 :
	#	print('the output for index 14 is %g'%output)	
	return output

def getBottomOutput(index,points,target) :
	global gamesArr
	global limit
	global imageSize
	
	pos = index%limit
	if (pos+imageSize[0]>=limit) or (points[index+imageSize[0]] == -100) :
		#print 'hi'
		output = -100
	elif points[index+imageSize[0]] == 100 :
		#print 'hi'
		output = 100	
	else :
		output = points[index+imageSize[0]] - points[index]+	max(target[index+imageSize[0]])	

	return output	


def getBatchInputOutput(batchSize,points,target) :	
	global start
	first = start
	start = start+batchSize
	global totalImages
	global gamesArr
	if start > len(gamesArr) :
		print 'randomise called again'

		randomiseGames()
		first = 0 
		start = batchSize
	end = start
	inputs = []
	output = []
	
	inputIndices = gamesArr[first:end]	
	for i in range(batchSize) :
		inputs.append(getInput(inputIndices[i]))
	#points = np.genfromtxt('points.txt')
	#target = np.genfromtxt('Targets.txt')
	#print inputIndices
	for i in range(batchSize) :

		output1 = getRightOutput(inputIndices[i],points,target)
		output2 = getBottomOutput(inputIndices[i],points,target)
		output.append([output1,output2])

	return np.array(inputs),np.array(output)

steps = 50
count = 0
restoreCheckpointFile = 'NewCheckpoints/Checkpoint3.ckpt'
saveCheckpointFile = 'NewCheckpoints/Checkpoint2.ckpt'
while count<steps :

	print('count is %d'%count)
	gamesArr = []
	points = np.genfromtxt('pointsNew.txt')
	target = np.genfromtxt('Targets200_New.txt')

	for i in range(totalImages) :
		if points[i] == 0 :
			gamesArr.append(i)

	randomiseGames()

		

	outputLength = 2
	epochs = 4
	learningRate = 0.001
	lr_decay_step = 2000
	lr_decay_rate = 0.9

	#restoreCheckpointFile = 'ModelCheckpoints/initialCheckpoint.ckpt'

	iterations = int(epochs*(len(gamesArr)/batchSize))
	with tf.Graph().as_default() :

		x = tf.placeholder(tf.float32, shape=[None, imageSize[0], imageSize[1], imageSize[2]])
		y = tf.placeholder(tf.float32,shape= [None,outputLength])
		W_conv1 = weight_variable([2, 2, 3, 10])
		b_conv1 = bias_variable([10])

		h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)

		W_conv2 = weight_variable([2, 2, 10, 20])
		b_conv2 = bias_variable([20])

		h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

		h_conv2_flat = tf.reshape(h_conv2, [-1, 25*25*20])
		W_fc1 = weight_variable([25*25*20, 100])
		b_fc1 = bias_variable([100])
		h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

		W_fc2 = weight_variable([100, outputLength])
		b_fc2 = bias_variable([outputLength])

		y_out=tf.matmul(h_fc1, W_fc2) + b_fc2




		loss = tf.reduce_mean(tf.square(y-y_out),1)
		avg_loss = tf.reduce_mean(loss)

		global_step = tf.Variable(0, trainable=False)
		lr = tf.train.exponential_decay(learningRate,
										global_step,
										lr_decay_step,
										lr_decay_rate,
										staircase=True)

		train_step = tf.train.AdamOptimizer(lr).minimize(avg_loss)

		sess = tf.Session()
		saver = tf.train.Saver()
		saver.restore(sess, restoreCheckpointFile)
		print('Model restored from file %s'%restoreCheckpointFile)

		for i in range(iterations) :
			inputs,outputs = getBatchInputOutput(batchSize,points,target)

			if i%50 == 0 :
				losses = sess.run(avg_loss,feed_dict={x: inputs, y: outputs})
				print('%d steps reached and the loss is %g'%(i,losses))
				
			sess.run(train_step,feed_dict={x: inputs, y: outputs})

			if i == iterations-1 :
				save_path = saver.save(sess, saveCheckpointFile)
				print("Model saved in file: %s" % save_path)
			
	ReCalculateTargets2.CalculateTargets(saveCheckpointFile)
	tempor = restoreCheckpointFile
	restoreCheckpointFile = saveCheckpointFile
	saveCheckpointFile = tempor
	count+=1


			


