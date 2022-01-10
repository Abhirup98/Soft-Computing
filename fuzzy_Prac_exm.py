import itertools
import numpy as np

class FuzzySet:
  
    def __init__(self, size, data, degree):
        self.size = size
        self.data = data
        self.memb_degree = degree

    def display(self):
      for (a, b) in zip(self.data, self.memb_degree):
        print('(' + str(a) + ', ' + str('%.3f'%b) + ')')

    def algb_product(self,obj):
          if isinstance(obj,FuzzySet) and self.size==obj.size:
            mul = []
            for (a, b) in zip(self.memb_degree, obj.memb_degree):
                mul.append(a*b)

            result = FuzzySet(self.size, self.data, mul)
            result.display()
          else:
              print('error')

    def multiplication(self, multiplication):
      mul = []
      for i in self.memb_degree:
          mul.append(i*multiplication)

      result = FuzzySet(self.size, self.data, mul)
      result.display()

    def algb_sum(self,obj):
          if isinstance(obj,FuzzySet) and self.size==obj.size:
            sum = []
            for (a, b) in zip(self.memb_degree, obj.memb_degree):
                sum.append(a + b - a*b)

            result = FuzzySet(self.size, self.data, sum)
            result.display()
          else:
              print('error')

    def algb_diff(self,obj):
          if isinstance(obj,FuzzySet) and self.size==obj.size:
            diff = []
            c = [1-i for i in obj.memb_degree]
            for (a, b) in zip(self.memb_degree, c):
                if a < b: diff.append(a)
                else: diff.append(b)

            result = FuzzySet(self.size, self.data, diff)
            result.display()
          else:
              print('error')

    def cartesian(self, obj):
      cart = []
      for i in self.memb_degree:
        for j in obj.memb_degree:
          if i < j: cart.append(i)
          else: cart.append(j)
      print(np.array(cart).reshape(self.size, obj.size))

if __name__=='__main__':

    data1 = ['x1', 'x2', 'x3', 'x4']
    memb1 = [0.25, 0.8, 0.1, 0.45]
    data2 = ['x1', 'x2', 'x3', 'x4']
    memb2 = [.8,.95,.85,.78]

    set1 = FuzzySet(4, data1, memb1)
    set2 = FuzzySet(4, data2, memb2)

    print('The algebraic product of fuzzy set 1 and 2 is:')
    set1.algb_product(set2)

    print('The multiplication of fuzzy set 1 is:')
    set1.multiplication(4)

    print('The algebraic sum of fuzzy set 1 and 2 is:')
    set1.algb_sum(set2)

    print('The algebraic difference of fuzzy set 1 and 2 is:')
    set1.algb_diff(set2)

    print('The cartesian product of fuzzy set 1 and 2 is:')
    set1.cartesian(set2)


    


              
