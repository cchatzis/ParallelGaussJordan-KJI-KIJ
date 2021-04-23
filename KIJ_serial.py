import numpy as np
from functions import *
import sys
from mpi4py import MPI

# Setup matrix A

np.random.seed(4)

A = np.random.randint(-10,10,size=(int(sys.argv[1]),int(sys.argv[1]))).astype(float)

# Setup vector b

b = np.random.randint(-10,10,size=(int(sys.argv[1]),1)).astype(float)

Afull = np.hstack((A,b))

# Setup pivot vector

pivot = np.arange(0,int(sys.argv[1]),1,dtype=int)

# Our goal is to solve Ax=b using the GJ parallel method

n = Afull.shape[0]

start = MPI.Wtime() # Start timer

for k in range(0,n):

    # Find pivot of this column

    pivot_pos = k

    for i in range(k+1,Afull.shape[0]):

        if abs(Afull[pivot[pivot_pos]][k]) <= abs(Afull[pivot[i]][k]):

            pivot_pos = i
    
    # Update pivot vector

    _ = pivot[pivot_pos]
    pivot[pivot_pos] = pivot[k]
    pivot[k] = _

    # Update k-th row

    for j in range(k+1,Afull.shape[1]):  # only those after the active columns are required to be updated

        Afull[pivot[k]][j] = Afull[pivot[k]][j] / Afull[pivot[k]][k]

    # Update A

    for i in range(0,Afull.shape[0]):

        if i != k:

            for j in range(k+1,Afull.shape[1]):

                Afull[pivot[i]][j] = Afull[pivot[i]][j] - Afull[pivot[i]][k] * Afull[pivot[k]][j]

    # Print new matrix

    # print_matrix(Afull,pivot)

end = MPI.Wtime() # Stop timer

elapsed_time = end - start

print_solution(Afull,pivot)

print(f"EXEC_TIME: {elapsed_time}s")