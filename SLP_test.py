import numpy as np

class Perceptron(object):
    def __init__(self, wvect, input_size, learn_rate = 0.1, epochs = 100):
        self.wvect = wvect
        self.epochs = epochs
        self.learn_rate = learn_rate
    
    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.wvect.T.dot(x)
        a = self.activation(z)
        return a

    def weight_update(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                e = d[i] - y
                self.wvect = self.wvect + self.learn_rate * e * x
    
    def test(self, X):
      for i in range(X.shape[0]):
        x = np.insert(X[i], 0, 1)
        y = self.predict(x)
        print('x1:{}, x2:{}, Output:{}'.format(x[1], x[2], y))

if __name__ == '__main__':
  AND_X = np.array([
      [0.001, 0.97],
      [0.002, 0.97],
      [0.01, 0.98],
      [0.97, 0.002],
      [0.98, 0.002],
      [0.95, 0.01],
      [0.97, 0.99],
      [0.95, 0.96],
      [0.94, 0.95],
      [0.001, 0.002],
      [0.002, 0.001],
      [0.032, 0.012]
  ])

  OR_X = np.array([
      [0.001, 0.97],
      [0.002, 0.97],
      [0.01, 0.98],
      [0.97, 0.002],
      [0.98, 0.002],
      [0.95, 0.01],
      [0.97, 0.99],
      [0.95, 0.96],
      [0.94, 0.95],
      [0.001, 0.002],
      [0.002, 0.001],
      [0.032, 0.012]
  ])

  test_X = np.array([
      [0.04, 0.90],
      [0.02, 0.96],
      [0.09, 0.90],
      [0.94, 0.02],
      [0.908, 0.02],
      [0.925, 0.1],
      [0.92, 0.94],
      [0.96, 0.92],
      [0.90, 0.89],
      [0.1, 0.12],
      [0.02, 0.1],
      [0.02, 0.1]
  ])
  
  AND_Y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
  OR_Y = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])

  input_size = 2
  AND_perceptron = Perceptron(input_size = input_size, wvect = np.ones(input_size+1))
  AND_perceptron.weight_update(AND_X, AND_Y)
  print('Weights for AND perceptron are : {}\n'.format(AND_perceptron.wvect))
  print('Predictions for test data are:')
  AND_perceptron.test(test_X)

  input_size = 2
  OR_perceptron = Perceptron(input_size = input_size, wvect = np.ones(input_size+1))
  OR_perceptron.weight_update(OR_X, OR_Y)
  print('\nWeights for OR perceptron are : {}\n'.format(OR_perceptron.wvect))
  print('Predictions for test data are:')
  OR_perceptron.test(test_X)
