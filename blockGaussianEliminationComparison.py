import numpy as np
import scipy.io
import scipy.sparse
import scipy.sparse.linalg
import time
import matplotlib.pyplot as plt

dimensions = []
block_times = []
gauss_times = []
table = []

matrixNames = ['bcsstk24.mtx', '1138_bus.mtx',  'bcsstk03.mtx', 'arc130.mtx']
for name in matrixNames:
    print(name)
    # Load A matrix in
    A = scipy.io.mmread(name).tocsc() #download a matrix from dataset to be able to do this
    size = A.shape[0]
    print("Matrix dimensions of "+str(A.shape))
    # Create a random b matrix
    b = np.random.rand(size)
    #Block Gaussian Elimination
    # first partitioning A into 4 blocks that are roughly the same size
    block_size = size // 2
    A11 = A[:block_size, :block_size]
    A12 = A[:block_size, block_size:]
    A21 = A[block_size:, :block_size]
    A22 = A[block_size:, block_size:]
    b1 = b[:block_size]
    b2 = b[block_size:]
    start = time.time()
    # Solve using blocks as discussed
    A11_inv = scipy.sparse.linalg.inv(A11)
    Schurs_complement = A22 - A21 @ A11_inv @ A12
    x2 = np.linalg.solve(Schurs_complement.toarray(), b2-A21 @ A11_inv @ b1)
    x1 = np.linalg.solve(A11.toarray(), b1 - A12 @ x2)
    solutionBlockGaussian = np.concatenate([x1, x2])
    solutionBlockGaussianTime = time.time() - start

    # Normal Gaussian Elimination using built-in np.linalg
    start = time.time()
    solutionNormalGaussian = np.linalg.solve(A.toarray(), b)
    solutionNormalGaussianTime = time.time() - start

    absolute_error = abs(solutionNormalGaussianTime-solutionBlockGaussianTime)
    relative_error = abs((solutionNormalGaussianTime-solutionBlockGaussianTime)/solutionBlockGaussianTime)
    
    #print results
    if np.allclose(A @ solutionNormalGaussian, b) and np.allclose(A @ solutionBlockGaussian, b):
        print(f'Normal Gaussian Elimination Time: {solutionNormalGaussianTime} seconds')
        print(f'Block Gaussian Elimination Finished in {solutionBlockGaussianTime} seconds')
        print(f'time difference between Normal and Block Gaussian Elimination Methods: {solutionNormalGaussianTime-solutionBlockGaussianTime} seconds')
        print(f'Relative time difference between Normal and Block Gaussian Elimination Methods: {abs((solutionNormalGaussianTime-solutionBlockGaussianTime)/solutionBlockGaussianTime)} seconds')
        print("\n")
    else:
        print("Solutions aren't valid")
        
    dimensions.append(size)
    block_times.append(solutionBlockGaussianTime)
    gauss_times.append(solutionNormalGaussianTime)
    table.append([name, size, solutionBlockGaussianTime,solutionNormalGaussianTime, absolute_error, relative_error ])
    
# Graph results
plt.figure(figsize=(8,5))
plt.plot(dimensions, block_times, marker='o', color='blue', label='Block Gaussian')
plt.plot(dimensions, gauss_times, marker='s', color='red', label='Normal Gaussian')
plt.xlabel('Matrix Dimension')
plt.ylabel('Performance Time (seconds)')
plt.title('Block Gaussian vs Normal Gaussian Times')
plt.legend()
plt.grid(True)
plt.show()

# Print table
header = ["Matrix", "Size", "Block Time (s)", "Normal Time (s)", "Absolute Error (s)", "Relative Error"]
print("{:<18} {:<8} {:<16} {:<16} {:<18} {:<16}".format(*header))
for row in table:
    print("{:<18} {:<8} {:<16.5f} {:<16.5f} {:<18.5f} {:<16.5f}".format(
        row[0], row[1], row[2], row[3], row[4], row[5]
    ))
    