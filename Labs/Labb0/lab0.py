import numpy as np
import time

"""
def sum_trad():
    start = time.time()
    X = range(10000000)
    Y = range(10000000)
    Z = []
    for i in range(len(X)):
        Z.append(X[i]+Y[i])
    return time.time() - start

def sum_numpy():
    start = time.time()
    X = np.arange(10000000)
    Y = np.arange(10000000)
    Z = X + Y
    return time.time() - start

print('time sum:', sum_trad(), '  time sum numpy:', sum_numpy()) 
"""

arr = np.array([2, 6, 5, 9], float)
print(arr)
print(type(arr))

arr = np.array([1, 2, 3], float)
arr.tolist()
list(arr)

