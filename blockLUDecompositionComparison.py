import numpy as np
import scipy.sparse
import scipy
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt

#function that does LU decomposition w/o pivoting 
def luWithoutPivot(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy().astype(float)
    for k in range(n):
        if abs(U[k, k]) < 1e-8:
            print("Zero pivot error")
        for i in range(k+1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    return L, U

dimensions = []
block_times = []
gauss_times = []
table = []

#need to download these matrices from sparse.tamu.edu before running the code
matrixNames = ['bcsstk24.mtx', '1138_bus.mtx', 'bcsstk03.mtx', 'arc130.mtx']
for name in matrixNames:
    print(name)
    aMatrix = scipy.io.mmread(name).tocsc()  # You can change the matrix code here
    A = aMatrix.toarray()
    n = A.shape[0]
    if n % 2 != 0:
        n = n - 1
        A = A[:n, :n]
    print(f"Matrix shape: {A.shape}")

    #splits A into 4 blocks
    sizeOfBlock = n // 2
    A11 = A[:sizeOfBlock, :sizeOfBlock]
    A12 = A[:sizeOfBlock, sizeOfBlock:]
    A21 = A[sizeOfBlock:, :sizeOfBlock]
    A22 = A[sizeOfBlock:, sizeOfBlock:]
    #generates a random b matrix
    b = np.random.rand(n)
    b1 = b[:sizeOfBlock]
    b2 = b[sizeOfBlock:]
    start = time.time()
    # Block LU decomposition to solve for x as described in paper
    L11, U11 = luWithoutPivot(A11)
    U12 = np.linalg.solve(L11, A12) #solves A12  = L11U12 for U12
    L21 = np.linalg.solve(U11.T, A21.T).T #solves A21  = L21U11 for L21
    Schurs_complement = A22 - L21 @ U12 #Schur's complement
    L22, U22 = luWithoutPivot(Schurs_complement)
    #concatenates blocks together
    L = np.block([[L11, np.zeros((sizeOfBlock, n-sizeOfBlock))],[L21, L22]])
    U = np.block([[U11, U12],[np.zeros((n-sizeOfBlock, sizeOfBlock)), U22]])
    #solves Ly = b for y using forward substitution
    y1 = np.linalg.solve(L11, b1)
    y2 = np.linalg.solve(L22, b2 - L21 @ y1)
    y = np.concatenate([y1, y2])
    #solves Ux = y for x using backwards substitution
    x2 = np.linalg.solve(U22, y2)
    x1 = np.linalg.solve(U11, y1 - U12 @ x2)
    xBlock = np.concatenate([x1, x2])
    blockLUTime = time.time() - start

    # normal LU decomposition to solve for x
    start = time.time()
    L_normal, U_normal = luWithoutPivot(A)
    yNormal = np.linalg.solve(L_normal, b)
    xNormal = np.linalg.solve(U_normal, yNormal)
    normalLUTime = time.time() - start
    nopivot_error = np.linalg.norm(A @ xNormal - b)

    absolute_error = abs(blockLUTime-normalLUTime)
    relative_error = abs((blockLUTime - normalLUTime)/normalLUTime)
    if np.allclose(A @ xBlock, b) and np.allclose(A @ xNormal, b):
        print(f'Normal LU decomposition Time: {normalLUTime} seconds')
        print(f'Block LU decomposition Finished in {blockLUTime} seconds')
        print(f'time difference between Normal and Block LU decomposition Methods: {blockLUTime-normalLUTime} seconds')
        print(f'Relative time difference between Normal and Block LU decomposition Methods: {abs((blockLUTime - normalLUTime)/normalLUTime)} seconds')
    else:
        print("Incorrect Solution")
    print("\n")
    dimensions.append(n)
    block_times.append(blockLUTime)
    gauss_times.append(normalLUTime)
    table.append([name, n, blockLUTime, normalLUTime, absolute_error, relative_error ])
    
# Graph the results
plt.figure(figsize=(8,5))
plt.plot(dimensions, block_times, marker='o', color='blue', label='Block LU')
plt.plot(dimensions, gauss_times, marker='s', color='red', label='Normal LU')
plt.xlabel('Matrix Dimension (n)')
plt.ylabel('Performance Time (seconds)')
plt.title('Block LU vs Normal LU Decomposition Times')
plt.legend()
plt.grid(True)
plt.show()

#print out table
header = ["Matrix", "Size", "Block Time (s)", "Normal Time (s)", "Absolute Error (s)", "Relative Error"]
print("{:<18} {:<8} {:<16} {:<16} {:<18} {:<16}".format(*header))
for list in table:
    print("{:<18} {:<8} {:<16.5f} {:<16.5f} {:<18.5f} {:<16.5f}".format(
        list[0], list[1], list[2], list[3], list[4], list[5]
    ))