import argparse
import numpy

#IS_DEBUG = False
def DEBUG(s):
	if IS_DEBUG:
		print(s)

class Perceptron:
	''' '''
	weights = []

	def __init__(self, dimensions):
		self.weights = numpy.array([0 for i in range(dimensions+1)])

	def predict(self, inp):
		return numpy.dot(inp, self.weights)

	''' 
		Update takes an input vector. It automatically adds the bias term to the input.
	'''
	def update_iter(self, inp, out, max_iters=None):
		inp=[numpy.append(row, 1) for row in inp]
		y = self.predict(inp)

		best_weights = None
		best_err = None

		cur_iter = 1
		count = -1
		while max_iters and max_iters > 0 and count != 0:
			DEBUG("Iteration %d"%(cur_iter))
			count=0
			new_weights = self.weights
			for row in range(len(y)):
				if y[row]*out[row] <= 0:
					count+=1
					new_weights = new_weights + inp[row] * out[row]
			DEBUG("\tCount of incorrect guesses: %d"%(count))
			
			if best_weights is None or count<best_err:
				best_weights = self.weights
				best_err = count
			self.weights = new_weights
			
			y = self.predict(inp)
			max_iters -= 1
			cur_iter += 1
		self.weights = best_weights
		return self.weights

	def empirical_err(self, testin, testout):
		testin = [numpy.append(row, 1) for row in testin]
		y = self.predict(testin)
		err = 0
		for row in range(len(y)):
			if y[row]*testout[row] <= 0:
				err+=1
		err = err / len(y)
		return err

	def __str__(self):
		out = "Weights"
		for i in range(len(self.weights)):
			out += "\n" + str(self.weights[i])
		return out

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
				clean_output.append(-1 if line[-1]=="0" else 1)
			print("Dataset size: %d lines, %d input dimensions"%(len(clean_output), len(clean_input[0])))
			
			self.data_in = clean_input
			self.data_out = clean_output

			return clean_input, clean_output
		return [],[]

	def run_erm(self, learner, inp=None, out=None):
		if inp is None:
			inp = self.data_in
			out = self.data_out
		learner.update_iter(numpy.array(inp), out, 1000)
		DEBUG(learner)
		DEBUG("Empirical error: %f"%(learner.empirical_err(inp,out)))
		return learner, learner.empirical_err(inp,out)

	def run_kfolds(self, learners, inp=None, out=None, k=10):
		if inp is None:
			inp = self.data_in
			out = self.data_out

		input_sets = [[] for fold in range(k)]
		output_sets = [[] for fold in range(k)]
		for row in range(len(inp)):
			input_sets[row%k].append(inp[row])
			output_sets[row%k].append(out[row])

		results = []
		for learner in range(len(learners)):
			training_in = []
			training_out = []
			for i in range(len(input_sets)):
				if i!= learner:
					training_in += input_sets[i]
			for i in range(len(output_sets)):
				if i!= learner:
					training_out += output_sets[i]
			results.append( self.run_erm(learners[learner], training_in, training_out)[0] )
		
		sum_err = 0
		for perceptron in range(len(results)):
			print("Perceptron %d"%perceptron)
			print(results[perceptron])
			err = results[perceptron].empirical_err(input_sets[perceptron], output_sets[perceptron])
			sum_err += err
			print("Error: %f"%(err))
			print()
		print("Overall Error (Averaged Across Folds) : %f"%(sum_err/k))

if __name__=="__main__":
	global IS_DEBUG 

	parser = argparse.ArgumentParser(description="Run perceptron learning algorithm")
	parser.add_argument("--dataset", dest='fname', action="store", help="Path to datafile")
	parser.add_argument("--mode", dest='mode', action="store", help="ERM or 10-fold")
	parser.add_argument("-d", dest='IS_DEBUG', default=False, action="store_true", help="Print debug statements")

	args = parser.parse_args()
	IS_DEBUG = args.IS_DEBUG

	t = Tester()
	
	inp, out = t.parse_file(args.fname)

	p = Perceptron(len(inp[0]))
	p_multi = [Perceptron(len(inp[0])) for i in range(10)]

	#t.run_erm(p)
	t.run_kfolds(p_multi)