import matplotlib.pyplot as plt
import numpy as np
import random 



class PerceptronLearningAlgo():
	"""
	Perception Learning Algorithm

	h(x) = sign(w0 + w1*x1 + w2*x2 + ... + wd*xd)
	Activation 		:sign function
	Optimization	:Mean Squared Error/Stochastic Gradient Descent

	input:
	alpha			:Learning rate (Hyperparameter)
	n_epochs		:Number of Epochs
	X_inp			:n X d data of real numbers. n - number of input value, d - dimension of each input value.
	Y_out			:corresponding output for the input {-1, 1}

	Methods:
	train 			:Train the perceptron
	test 			:Testing the perceptron
	crossValidate	:Evaluate perceptron using k-fold cross validation

	"""
	def __init__(self, alpha = 0.01, n_epochs = 1000):
		self.alpha = alpha
		self.n_epochs = n_epochs
		self.total_cost = []

	def sign(self, X, weights, bias):
		"""
		Weighted sum and activation function

		Activation Function: Sign
		"""
		sum = np.dot(X, weights) + bias
		return 1 if sum>=0 else -1

	def train(self, X_inp, Y_out, epochLog = False):
		#Initializing weights and bias
		self.weights = np.random.uniform(0, 1, X_inp.shape[1])
		self.bias = np.random.uniform(0,1,1)
		self.epochLog = epochLog

		for epoch in range(self.n_epochs): 
			epoch_cost = 0
			for x_inp, y_out in zip(X_inp, Y_out):
				predict = self.sign(x_inp, self.weights, self.bias)
				#Mean Squared Error
				cost = np.square(y_out - predict)
				epoch_cost += float(cost/len(X_inp))

				#Updating weights and bias
				update = self.alpha*(y_out - predict)
				self.weights += update*x_inp
				self.bias += update
			self.total_cost.append(epoch_cost)

			if self.epochLog:
				print("Epoch: {:04}\tLoss: {:06.5f}".format((epoch+1), epoch_cost), end='')
		print("Equation:{:.2f} + {:.2f}(X0) + {:.2f}(X1)".format(float(self.bias), self.weights[0], self.weights[1]))

		return self 

	def test(self, X_inp, Y_out, Testing = False, filename = None):
		"""
		Testing the perceptron on the unseen data

		Accuracy Score: 100*sum([h(x) = y(i)])/m
		h(x) 			:prediction of perceptron
		y(i)			:corresponding actual value
		m 				:total number of test values
		[.]				:indicator function (which is 1 if condition is true else false)
		"""
		if Testing:
			if filename is None:
				raise Exception("Weight file is missing. Please input the weight file.")
			else:
				bias, weights = dataProcessing.loadWeights(filename)
		else:
			bias = self.bias
			weights = self.weights
		score = 0
		for x_inp, y_out in zip(X_inp, Y_out):
			predict = self.sign(x_inp, weights, bias)
			if predict == y_out:
				score += 1
			else:
				score +- 0
		accuracy = 100*score/len(X_inp)
		print("Accuracy: {:.2f}".format(accuracy))
		return accuracy

	
	def crossValidate(self, n_folds, X_inp, Y_out):
		"""
		Evaluate the performance of perceptron by k-fold cross validation
		"""
		dataset_split = dataProcessing.splitData(n_folds, X_inp, Y_out)
		scores = []
		
		for i in range(len(dataset_split)):
			# one fold data for testing
			testset = dataset_split[i]
			test_x = testset[:,0:testset.shape[1]-1]
			test_y = testset[:,-1]

			# Remaining folds of data for training
			trainset = np.delete(dataset_split, i, 0)
			trainset = np.concatenate(trainset, axis = 0)
			train_x = trainset[:,0:trainset.shape[1]-1]
			train_y = trainset[:,-1]

			self.train(train_x, train_y)
			accuracy = self.test(test_x, test_y)
			scores.append(accuracy)
		return scores

				

class dataProcessing():

	@staticmethod
	def visualizeData(X_inp, Y_out, weights = None, bias = None):
		"""
		Visualizes 2-d Data according to the class (positive or negative)
		"""
		plot1 = plt.figure(1)
		for i in range(X_inp.shape[0]):
			if Y_out[i] == 1:
				color = 'r'
				marker = '+'
				label = 'class 1'
				class1 = plt.scatter(X_inp[i][0], X_inp[i][1], marker = marker, c = color, label = label);
			else:
				color = 'b'
				marker = '*'
				label = 'class 2'
				class2 = plt.scatter(X_inp[i][0], X_inp[i][1], marker = marker, c = color, label = label);
		plt.xlabel('X0')
		plt.ylabel('X1')
		
		if weights is None and bias is None:
			plt.legend(handles = [class1, class2])
			plt.grid()
			plt.show()
		else:
			#plots the 2D line classifying the two classes
			slope = -(bias[0]/weights[1])/(bias[0]/weights[0])
			intercept = -bias[0]/weights[1]
			
			x = np.linspace(np.amin(X_inp[:,:2]), np.amax(X_inp[:,:2]))
			y = (slope*x) + intercept
			label = f'y={slope}x + {intercept}'
			line, = plt.plot(x, y, '-k', label = label)
			plt.legend(handles = [class1, class2, line])
			plt.grid()
			plt.show()


	@staticmethod
	def ErrorPlot(costArray):
		"""
		Plot Mean Square Error vs Epoch
		"""
		plot2 = plt.figure(2)
		plt.plot(costArray)
		plt.xlabel('Epochs')
		plt.ylabel('MSE')
		plt.show()
		
	@staticmethod
	def readData():
		"""
		Read data (training or testing) from the console and split to input and output data.
		"""
		n, d = map(int, input("Enter the number of data points and dimension").split())
		data = np.array([input().strip().split() for _ in range(n)], float)
		X_inp = data[:,0:d]
		Y_out = data[:,-1]
		return X_inp, Y_out

	@staticmethod
	def readFromFile(filename):
		"""
		Read data (training or testing) from the file and split to input and output data. 
		"""
		data = np.genfromtxt(filename, dtype = float)
		X_inp = data[:,0:data.shape[1]-1]
		Y_out = data[:,-1]
		return X_inp, Y_out

	@staticmethod
	def writeToFile(filename, weights,bias):
		"""
		Save the weights to a file
		"""
		f = open(filename, 'w')
		print(bias[0], file=f)
		for i in range(len(weights)):
			print(weights[i], file=f)					
		f.close()

	@staticmethod
	def loadWeights(filename):
		"""
		Load bias and weights from a file
		"""
		data = np.genfromtxt(filename, dtype = float)
		bias = [data[0]]
		weights = data[1:]
		return bias, weights

	@staticmethod
	def splitData(n_folds, X_inp, Y_out):
		"""
		Splits the dataset n folds
		"""
		data = np.column_stack((X_inp, Y_out))
		dataset_split = np.array_split(data, n_folds)
		return dataset_split



def main():
	
	# Load the training and testing data
	train_filename 	= 'data1.txt'
	test_filename 	= 'data2.txt'
	output_filename = 'weights.txt'

	train_x, train_y = dataProcessing.readFromFile(train_filename)
	test_x, test_y 	= dataProcessing.readFromFile(test_filename)

	#Training the Perceptron
	perceptron = PerceptronLearningAlgo()
	perceptron.train(train_x, train_y)
	dataProcessing.ErrorPlot(perceptron.total_cost)
	dataProcessing.visualizeData(train_x, train_y, perceptron.weights, perceptron.bias)

	#Testing the Perceptron
	perceptron.test(test_x, test_y)	
	dataProcessing.writeToFile(output_filename, perceptron.weights, perceptron.bias)
	
	#k-fold Cross Validation of the model
	perceptron = PerceptronLearningAlgo()
	scores = perceptron.crossValidate(3, test_x, test_y)
	print(scores)



if __name__ == '__main__':
	main()






