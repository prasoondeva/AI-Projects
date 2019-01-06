import pdb
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from load_dataset import mnist

def one_hot(y,n):
	y_one_hot = np.zeros((y.shape[1],n))
	for i in range(y.shape[1]):
		y_one_hot[i,int(y[0,i])] = 1
	return y_one_hot

# Loading the data
trainX, trainY, testX, testY = mnist(ntrain=60000,ntest=10000,onehot=False,subset=True,digit_range=[0,10],shuffle=True)
trainX = trainX.T
testX = testX.T

n_input = 28 * 28
n_classes = 10
batch_size = 128
n_epoch = 200
learning_rate = 0.01

'''
As we have multiple classes we first need to perform one hot encoding of our labels
'''
train_Y_one_hot = one_hot(trainY, n_classes)
test_Y_one_hot = one_hot(testY, n_classes)


features = tf.placeholder(dtype = tf.float32 , shape = [None,n_input])  # placeholder for the input features
labels = tf.placeholder(dtype = tf.float32, shape = [None,n_classes]) # placeholder for the labels

'''
The line below will generate a hidden layers which has features as it's input, 128 neurons and sigmoid as the 
activation function.
'''

hidden_layer_1 = tf.contrib.layers.fully_connected(features, 512 , activation_fn = tf.nn.sigmoid)
hidden_layer_2 = tf.contrib.layers.fully_connected(hidden_layer_1, 256 , activation_fn = tf.nn.sigmoid)
hidden_layer_3 = tf.contrib.layers.fully_connected(hidden_layer_2, 128 , activation_fn = tf.nn.sigmoid)
hidden_layer_4 = tf.contrib.layers.fully_connected(hidden_layer_3, 32 , activation_fn = tf.nn.sigmoid)
prev_hidden_layer = tf.contrib.layers.fully_connected(hidden_layer_4, 16 , activation_fn = tf.nn.sigmoid)
'''
Define your Hidden Layers Here. Your last hidden layer should have varianle name "prev_hidden_layer"
'''
logits  = tf.contrib.layers.fully_connected(prev_hidden_layer , n_classes, activation_fn = None) # Defining the final layer.

softmax_op = tf.nn.softmax(logits) # Calculating the softmax activation.
preds = tf.argmax(softmax_op,axis = 1) # Computing the final predictions.


'''
The line below calculates the cross entropy loss for mutliclass predictions.
'''
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels,logits = logits))

correct_prediction = tf.equal(tf.argmax(logits,1) , tf.argmax(labels,1)) #Comparing network predictons with the actual class labels.
accuracy = tf.reduce_mean(tf.cast(correct_prediction , tf.float32)) # Computing the accuracy ( How many correct prediction / Total predictions to make)


optimizer = tf.train.RMSPropOptimizer(learning_rate)

'''
This operations does all the important work from calculating the gradients to updating the parameters.
'''

train_step = optimizer.minimize(loss)

train_losses = []
test_losses = []
test_accuracies = []

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for ii in range(n_epoch):
		for j in range(trainX.shape[1] // batch_size):
			batch_x = trainX[batch_size*j : batch_size * j + batch_size, :]
			batch_y = train_Y_one_hot[batch_size*j : batch_size * j + batch_size, :]
			sess.run(train_step, feed_dict = {features: batch_x , labels : batch_y})
		if ii % 10 == 0:
			'''
			For Every 10th epoch get the training loss and tesing loss and store them.
			You do it for all the the data points in your training and testing sets, not for batches.
			'''
			train_loss = sess.run(loss,feed_dict = {features : trainX , labels : train_Y_one_hot})
			train_acc = sess.run(accuracy , feed_dict = {features : trainX , labels : train_Y_one_hot})
			test_loss = sess.run(loss , feed_dict = {features : testX , labels : test_Y_one_hot})
			test_acc = sess.run(accuracy, feed_dict={features: testX, labels: test_Y_one_hot})
			print("Epoch : {}, Training Loss : {}, Training Accuracy : {}".format(ii,train_loss,train_acc))
			train_losses.append(train_loss)
			test_losses.append(test_loss)
			test_accuracies.append(test_acc)

	test_accuracy = sess.run(accuracy , feed_dict = {features : testX , labels : test_Y_one_hot}) # Get the test accuracy by evaluating the accuracy tensor with test data and test labels.

'''
The following code generates the plot for epochs vs training loss and epoch vs testing loss.
You will need to note the test accuracy and generate a plot for architecture vs test accuracy. 
'''
print("Testing accuracy ",test_accuracy)

X_axis = range(1,n_epoch + 1 ,10)
plt.plot(X_axis,train_losses,"-",color = "blue")
plt.plot(X_axis,test_losses,"--",color = "red")
plt.legend(["Training Loss","Testing Loss"])
plt.show()

plt.plot(X_axis,test_accuracies,"-",color = "red")
plt.legend(["Testing Accuracy"])
plt.show()