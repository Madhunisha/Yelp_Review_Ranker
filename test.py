from _ctypes import sizeof

__author__ = 'Nikhil'

import numpy

mat = numpy.zeros((5,6))
mat[0][2] =3
num = [1,2,3,4,5]

def three(num):
    return 1,2,3

print len(num)
print 1,2,3

mat[1][2:5] = three(num)
use = 5
review = 2.0

print mat

print float(use/review)