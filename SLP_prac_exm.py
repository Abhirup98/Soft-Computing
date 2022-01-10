import numpy as np

class Perceptron(object):
    def __init__(self, w_vect, inp_size, lr = 0.1, epochs = 100):
        self.w_vect = w_vect
        self.lr = lr
        self.epochs = epochs
    
    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        z = self.w_vect.T.dot(x)
        a = self.activation(z)
        return a

    def weight_update(self, X, d):
        for _ in range(self.epochs):
            for i in range(d.shape[0]):
                x = np.insert(X[i], 0, 1)
                y = self.predict(x)
                e = d[i] - y
                self.w_vect = self.w_vect + self.lr * e * x

    def test(self, X):
        for i in range(X.shape[0]):
            x = np.insert(X[i], 0, 1)
            y = self.predict(x)
            print('x1:{}, x2:{}, Output:{}'.format(x[1], x[2], y))

if __name__ == '__main__':
  AND_X = np.array([
      [0.021, 0.943],
      [0.092, 0.061],
      [0.937, 0.929],
      [0.032, 0.989],
      [0.954, 0.985],
      [0.021, 0.956],
      [0.947, 0.032],
      [0.952, 0.051],
      [0.955, 0.946],
      [0.11, 0.02],
      [0.984, 0.072],
      [0.092, 0.082]
  ])

  test_X = np.array([
      [0.14, 0.90],
      [0.12, 0.906],
      [0.19, 0.99],
      [0.94, 0.12],
      [0.908, 0.12],
      [0.925, 0.11],
      [0.902, 0.904],
      [0.906, 0.982],
      [0.890, 0.889],
      [0.15, 0.192],
      [0.12, 0.19],
      [0.12, 0.18]
  ])

  AND_Y = np.array([0,0,1,0,1,0,0,0,1,0,0,0])
  

  inp_size = 2
  AND_percept = Perceptron(inp_size = inp_size, w_vect = np.ones(inp_size+1))
  AND_percept.weight_update(AND_X, AND_Y)
  print('Weights for AND perceptron are : {}\n'.format(AND_percept.w_vect))
  print('Predictions for the test data are:')
  AND_percept.test(test_X)

  
