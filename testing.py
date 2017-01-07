import numpy as np
import tensorflow as tf
import cv2
import os
import shutil

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def getBatchInput(inputs,start,batchSize) :
	first = start
	start = start + batchSize
	end = start
	return inputs[first:end],start

outputFolder = 'testOutput'

if os.path.exists(outputFolder):
	shutil.rmtree(outputFolder)

os.mkdir(outputFolder)


imageSize = [25,25,3]

learningRate = 0.001
lr_decay_rate = 0.9
lr_decay_step = 2000

checkpointFile = 'NewCheckpoints/Checkpoint2.ckpt'
x = tf.placeholder(tf.float32, shape=[None, imageSize[0], imageSize[1], imageSize[2]])
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

W_fc2 = weight_variable([100, 2])
b_fc2 = bias_variable([2])

y_out=tf.matmul(h_fc1, W_fc2) + b_fc2

loss = tf.reduce_mean(tf.square(y_out),1)
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
saver.restore(sess, checkpointFile)


testImages = 20
limit = imageSize[0]*imageSize[1]
point = np.genfromtxt('testPoints.txt')

for i in range(testImages) :
	index = 0
	outputImage = cv2.imread('testImages/image_'+str(i)+'_'+str(index)+'.png')
	while not index == limit-1 :
		print('index is %d'%index)
		inputImages = cv2.imread('testImages/image_'+str(i)+'_'+str(index)+'.png')
		
		inputs = []
		inputs.append(inputImages)
		inputs = np.array(inputs)
		output = sess.run(y_out,feed_dict={x: inputs})
		print output[0]
		if output[0][0] > output[0][1] :
			if (index+1)%imageSize[0] == 0 :
				print 'wrong grid entered'
				break

			index = index+1

		else :
			index = index+25

		if index>=limit or point[i*limit+index] == -100 :
			print 'wrong grid entered'
			break
		else :
			outputImage[int(index/imageSize[0])][index%imageSize[0]][0] = 223
			outputImage[int(index/imageSize[0])][index%imageSize[0]][1] = 27
			outputImage[int(index/imageSize[0])][index%imageSize[0]][2] = 20
			


	if index == limit-1 :
		cv2.imwrite(os.path.join(outputFolder,'image_'+str(i)+'.png'),outputImage)



					

