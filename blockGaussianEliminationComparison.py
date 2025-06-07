import numpy as np
import scipy.io
import scipy.sparse
import scipy.sparse.linalg
import time

matrixNames = ['bcsstk24.mtx', '1138_bus.mtx', 'bcsstk03.mtx', 'arc130.mtx']
for name in matrixNames:
    print(name)
    # Load A matrix in
    A = scipy.io.mmread(name).tocsc() #download a matrix from dataset to be able to do this
    size = A.shape[0]
    print("Matrix dimensions of "+str(A.shape))

    # Create a random b matrix
    b = np.random.rand(size)

    # Normal Gaussian Elimination using built-in np.linalg
    start = time.time()
    solutionNormalGaussian = np.linalg.solve(A.toarray(), b)
    solutionNormalGaussianTime = time.time() - start

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

    #print results
    if np.allclose(A @ solutionNormalGaussian, b) and np.allclose(A @ solutionBlockGaussian, b):
        print(f'Normal Gaussian Elimination Time: {solutionNormalGaussianTime} seconds')
        print(f'Block Gaussian Elimination Finished in {solutionBlockGaussianTime} seconds')
        print(f'time difference between Normal and Block Gaussian Elimination Methods: {solutionNormalGaussianTime-solutionBlockGaussianTime} seconds')
        print(f'Relative time difference between Normal and Block Gaussian Elimination Methods: {abs((solutionNormalGaussianTime-solutionBlockGaussianTime)/solutionBlockGaussianTime)} seconds')
        print("\n")
    else:
        print("Solutions aren't valid")

    

    
