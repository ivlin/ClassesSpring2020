import math
import numpy
import argparse 

def DEBUG(s):
	if IS_DEBUG:
		print(s)

class DecisionStump:
	boundary = None
	dim = None
	
	def __init__(self):
		pass

	def predict(self, data_in):
		return data_in[self.dim] >= self.boundary
		
	def update(self, data_in, data_out, dist):
		opt_dim = None
		opt_bound = None
		opt_err = None
		training_set_dist = numpy.array([1/len(data_in) for i in data_in])

		for dim in range(len( data_in[0] )):
			coordinates = sorted( [ (data_in[i][dim], data_out[i], dist[i]) for i in range(len(data_in)) ] )
			coordinates.append((coordinates[-1][0]+1))

			err = 0
			for i in range(len(data_out)):
				if data_out[i] == 1:
					err += dist[i]
			
			if opt_err == None or err < opt_err:
				opt_err = err
				opt_bound = coordinates[0][0]-1
				opt_dim = dim

			for cor in range(len(data_in)):
				err -= coordinates[cor][1] * coordinates[cor][2] #data_out[cor] * dist[cor]
				if err < opt_err and coordinates[cor][0] != coordinates[cor+1][0]:
					opt_err = err
					opt_bound = (coordinates[cor][0] + coordinates[cor+1][0])/2
					opt_dim = dim
					
		self.dim = opt_dim
		self.boundary = opt_bound
		return opt_dim, opt_bound

class AdaBoost:
	def __init__(self):
		pass

	def update(self, weak_learner, data_in, data_out, max_iters=None):
		dist = numpy.array([1/len(data_in) for i in data_in])
		while max_iters is None or max_iters > 0:
			a, b = weak_learner.update(data_in, data_out, dist)
			
			cur_err = 0
			for i in range(len(data_in)):
				cur_err += dist[i] * (weak_learner.predict(data_in[i]) != data_out[i])
			DEBUG(cur_err)
			weights = 1/2*math.log(1/cur_err-1)
			
			new_dist = []
			for i in range(len(dist)):
				new_dist.append(data_out[i] == weak_learner.predict(data_in[i]))
			
			dist = [ dist[i]*math.exp(-1*weights*(data_out[i] != weak_learner.predict(data_in[i]))) for i in range(len(dist)) ]

			DEBUG(new_dist)
			
			sum_dist = sum(dist)
			dist = [ dist[i]/sum_dist for i in range(len(dist)) ]

			if max_iters is not None:
				max_iters -= 1
			DEBUG(dist)

class Tester:
	data_in = None
	data_out = None

	def __init__(self, filename=None):
		if filename:
			self.parse_file(filename)

	#Reads the data from a file and converts it to floats. It assumes binary files and will convert output of 0 to -1
	def parse_file(self, filename):
		with open(filename, "r") as f:
			f.readline()
			raw_string = [line.strip().split(",") for line in f]
			clean_input = []
			clean_output = []
			for line in raw_string:
				clean_input.append([float(item) for item in line[:-1]])
				clean_output.append(0 if line[-1]=="0" else 1)
			print("Dataset size: %d lines, %d input dimensions"%(len(clean_output), len(clean_input[0])))
			
			self.data_in = clean_input
			self.data_out = clean_output

			return clean_input, clean_output
		return [],[]


if __name__=="__main__":
	global IS_DEBUG 

	parser = argparse.ArgumentParser(description="Run perceptron learning algorithm")
	parser.add_argument("--dataset", dest='fname', action="store", help="Path to datafile")
	parser.add_argument("--mode", dest='mode', action="store", help="ERM or 10-fold")
	parser.add_argument("-d", dest='IS_DEBUG', default=False, action="store_true", help="Print debug statements")

	args = parser.parse_args()
	IS_DEBUG = args.IS_DEBUG


	d = DecisionStump()
	a = AdaBoost()
	t = Tester()

	data_in, data_out = t.parse_file(args.fname)
	
	a.update(d, data_in[:50], data_out[:50], 50)