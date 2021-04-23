def print_matrix(A,pivot):
    '''
    Prints a matrix (numpy array) according to the given pivot vector.
    '''

    print("=========================================")

    for i in range(0,A.shape[0]):

        print('  ', end = "")

        for j in range(0,A.shape[1]):

            if j == A.shape[1]-1:

                print(f"| {A[pivot[i]][j]}", end="")

            else:

                print(A[pivot[i]][j], end = "\t")

        print('\n')

    print("=========================================")

def print_solution(A,pivot):
    '''
    Prints the last column of given numpy array according to the given pivot vector.
    '''

    print("=========================================")

    print("[", end ="")

    for i in range(0,A.shape[0]):

        if i != A.shape[0] - 1:

            print(f"{A[pivot[i]][A.shape[0]]}", end=", ")

        else:

            print(f"{A[pivot[i]][A.shape[0]]}", end=" ")

    print("]")

    print("=========================================")