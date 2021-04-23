########################################################
# Import necessary libraries
########################################################

from mpi4py import MPI
import numpy as np
import math
from functions import *
import sys

if int(sys.argv[3]) == 1: # DEBUG MODE ENABLED
    
    np.seterr(all='print')

########################################################
# MPI Initialization - Global Variables
########################################################

comm = MPI.COMM_WORLD

p = comm.Get_size()
myrank = comm.Get_rank()

if myrank == 0:

    print(f"Processors: {p}")

########################################################
# Problem declaration - Matrix Initilization
########################################################

# Setup matrix A

np.random.seed(4)

A = np.random.randint(-10,10,size=(int(sys.argv[2]),int(sys.argv[2]))).astype(float)

n = A.shape[0]

# Setup vector b

b = np.random.randint(-10,10,size=(int(sys.argv[2]),1)).astype(float)

Afull = np.hstack((A,b))

if myrank == 0:

    print(f"Size of Afull: {Afull.shape}")

# Setup pivot vector

pivot = np.arange(0,int(sys.argv[2]),1,dtype=int)

# Divide Matrix in columns for each processor available

method = int(sys.argv[1])

column_mapping = [ [] for _ in range(p) ]

if method == 0: # block mapping method

    if myrank == 0:

        print("Block mapping method selected...")

    size_of_block = math.floor(n/p) 

    for col in range(0,Afull.shape[1]):

        processor_no = col // size_of_block

        if processor_no >= p:

            column_mapping[-1].append(col)

        else:

            column_mapping[processor_no].append(col)

else: # Shuffle method

    if myrank == 0:

        print("Shuffle method selected...")

    for col in range(0,Afull.shape[1]):

        column_mapping[col%p].append(col)

if myrank == 0:
    print(column_mapping)

########################################################
# Gauss Jordan KIJ Parallel method
########################################################

start = MPI.Wtime() # Start timer

for k in range(0,n):

    # Given a column of the matrix there are 3 possibilities:
    # a) This column is in my area of responsibility -> Find pivoting row, compute active columns -> send columns to "next" processors
    # b) This column is in a previous processor's responsibility -> Receive new pivot, compute active columns -> send columns to "next" processors
    # c) This coulmn is in a next processor's responsibility -> Exit

    if k in column_mapping[myrank]: # a) This column is in my area of responsibility

        # Find pivot of this column

        pivot_pos = k

        for i in range(k+1,Afull.shape[0]):

            if abs(Afull[pivot[pivot_pos]][k]) <= abs(Afull[pivot[i]][k]):

                pivot_pos = i
        
        # Update pivot vector

        _ = pivot[pivot_pos]
        pivot[pivot_pos] = pivot[k]
        pivot[k] = _

        # Send updated pivot array to next processors

        for proc in range(0,p):

            if max(column_mapping[proc]) > k and proc != myrank:

                comm.Send(pivot,dest=proc,tag=11)

                if int(sys.argv[3]) == 1: # DEBUG MODE ENABLED

                    print(f"{myrank} just sent {pivot} to {proc}")

        # Update k row in the columns im responsible for

        for col in column_mapping[myrank]:

            if col > k: # only those after the active columns are required to be updated

                Afull[pivot[k]][col] = Afull[pivot[k]][col] / Afull[pivot[k]][k]

        # Update my columns

        for i in range(0,Afull.shape[0]):

            if i != k: # no need to update the column that the pivot is in

                for j in column_mapping[myrank]:

                    if j > k: # update columns right of the active column

                        Afull[pivot[i]][j] = Afull[pivot[i]][j] - Afull[pivot[i]][k] * Afull[pivot[k]][j]

        # Send my columns to the next processors

        for next_proc in range(0,p):

            if max(column_mapping[next_proc]) >= k and next_proc != myrank:

                comm.Send(Afull[:,column_mapping[myrank]],dest=next_proc,tag=11)

                if int(sys.argv[3]) == 1: # DEBUG MODE ENABLED
                    
                    print(f"{myrank} just sent {Afull[:,column_mapping[myrank]]} to {next_proc}")

        # Receive updated columns from other processors

        for prev_proc in range(0,p):

            if max(column_mapping[prev_proc]) >= k and prev_proc != myrank:

                _ = np.zeros((len(column_mapping[prev_proc]),n))

                comm.Recv(_,source=prev_proc,tag=11)

                _ = np.transpose(_)

                if int(sys.argv[3]) == 1: # DEBUG MODE ENABLED
                    
                    print(f"{myrank} just received {_} from {prev_proc}")

                # Update Afull

                for i in range(0,len(column_mapping[prev_proc])):

                    Afull[:,column_mapping[prev_proc][i]] = _[:,i]
                
    else:

        if method == 1 and (max(column_mapping[myrank]) < k):
            
            # This statement asks whether this processor is responsible for any nodes "right of" the active.
            # If not, the it is no longer needed and the process should exit.

            exit()

        src = 0

        for index in range(0,p):

            if k in column_mapping[index]:
                src = index
                break

        if method == 0 and (index > myrank): # <-- c) This coulmn is in a next processor's responsibility

            # In the block method, if a processor of higher order is responsible for the active column
            # then this processor should no longer contibute and exit, as it's columns are not updated
            # in the next steps.

            exit()

        # VV b) This column is in a previous processor's responsibility VV

        # Receive updated pivot array

        comm.Recv(pivot,source=src,tag=11)

        if int(sys.argv[3]) == 1: # DEBUG MODE ENABLED
            
            print(f"{myrank} just received {pivot} from {src}")

        # Update k row in the columns im responsible for

        for col in column_mapping[myrank]:

            if col > k: # only those after the active columns are required to be updated

                Afull[pivot[k]][col] = Afull[pivot[k]][col] / Afull[pivot[k]][k]

        # Update my columns

        for i in range(0,Afull.shape[0]):

            if i != k:

                for j in column_mapping[myrank]:

                    if j > k:

                        Afull[pivot[i]][j] = Afull[pivot[i]][j] - Afull[pivot[i]][k] * Afull[pivot[k]][j]

        # Send my columns to the next processors

        for next_proc in range(0,p):

            if max(column_mapping[next_proc]) >= k and next_proc != myrank:

                comm.Send(Afull[:,column_mapping[myrank]],dest=next_proc,tag=11)

                if int(sys.argv[3]) == 1: # DEBUG MODE ENABLED
                    
                    print(f"{myrank} just sent {Afull[:,column_mapping[myrank]]} to {next_proc}")

        # Receive columns from previous processor (if they exist)

        for prev_proc in range(0,p):

            if max(column_mapping[prev_proc]) >= k and prev_proc != myrank:

                _ = np.zeros((len(column_mapping[prev_proc]),n))

                comm.Recv(_,source=prev_proc,tag=11)

                _ = np.transpose(_)

                if int(sys.argv[3]) == 1: # DEBUG MODE ENABLED
                    
                    print(f"{myrank} just received {_} from {prev_proc}")

                # Update Afull

                for i in range(0,len(column_mapping[prev_proc])):

                    Afull[:,column_mapping[prev_proc][i]] = _[:,i]

end = MPI.Wtime() # Stop timer

elapsed_time = end - start

proc = 0

for i in range(1,p):

    if Afull.shape[1]-1 in column_mapping[i]:

        proc = i

if myrank == proc:

    print_solution(Afull,pivot)

    print(f"EXEC_TIME: {elapsed_time}s")