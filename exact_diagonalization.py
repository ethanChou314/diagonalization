import numpy as np


def normalize(v):
	return v / np.sqrt(np.dot(v, v))


def tridiag_matrix(alpha, beta):
	"""
    Reconstructs a real symmetric tridiagonal matrix T from diagonal 
    and off-diagonal components.

    Parameters:
        alpha (1D array): Diagonal elements of the tridiagonal matrix (length n).
        beta  (1D array): Off-diagonal elements (length n), where beta[i] 
        				  corresponds to the entries at positions [i+1, i] and [i, i+1].

    Returns:
        T (2D array): Symmetric tridiagonal matrix of shape (n, n) with 'alpha' 
        			  on the diagonal and 'beta' on the first sub- and 
        			  super-diagonals.
    """
	if alpha.ndim != 1:
		raise ValueError("alpha must be a rank 1 array")

	if beta.ndim != 1:
		raise ValueError("beta must be a rank 1 array")

	if alpha.shape != beta.shape:
		raise ValueError("dimension mismatch between 'alpha' and 'beta'.")

	n = len(alpha)

	T = np.zeros((n, n))

	for i in range(n):
        T[i, i] = alpha[i]
        if i > 0:  # exclude start
            T[i, i-1] = beta[i-1]
        if i < n - 1:  # exclude end
            T[i, i+1] = beta[i]

	return T


def lanczos(A, b):
	"""
	Performs the Lanczos algorithm.

	Parameters:
	    A (2D array[float]): Symmetric (or Hermitian) matrix to 
	    								be diagonalized (shape: n x n).
	    b (1D array[float]): Starting vector as initial guess (shape: n,). 
	    								Should not be the zero vector.

	Returns:
	    Q (2D array[float]): Orthonormal basis vectors of the Krylov subspace (n, m).
	    alpha (1D array[float]): Diagonal elements of the tridiagonal matrix T (length m).
	    beta (1D array[float]): Off-diagonal elements of T (length m).
	"""
	# get the size of the matrix:
	m = len(A)

	if len(b) != m:
		raise ValueError(f"Dimension mismatch. Got 'b' as vector of size {len(b)}. " + \
						 f"Expected size {m}")

	# initialize variables:
	Q = np.zeros((m, m))
	alpha = np.zeros(m)
	beta = np.zeros(m)
	beta_im1 = 0

	# set q0 = 0 implicitly (not stored), and q1 = b / ||b||:
	Q[:, 0] = normalize(b).flatten()
	q_im1 = np.zeros((m, 1))  # q0
	
	# iteration starts here:
	for i in range(m):
		q_i = Q[:, i].reshape(-1, 1)
		w = A @ q_i
		alpha[i] = float(q_i.T @ w)
		v = w - beta_im1 * q_im1 - alpha[i] * q_i
		q_ip1 = normalize(v)

		if i < m - 1:
			Q[:, i + 1] = q_ip1.flatten()

		beta[i] = float(q_ip1.T @ v)

		# for next iteration:
		q_im1 = q_i
		beta_im1 = beta[i]

	return Q, alpha, beta


def exact_diagonalization(A):
	"""
	Performs approximate diagonalization using the Lanczos algorithm.
	(A ~ Q * T * Q.T)
	
	Parameters:
		A (ndarray): Real symmetric matrix to diagonalize (n x n)
	
	Returns:
		Q (ndarray): Orthonormal basis of Krylov subspace (n x k)
		T (ndarray): Tridiagonal matrix (k x k)
		Q.T (ndarray): Transpose of Q
	"""
	# error checking:
	if not isinstance(A, np.ndarray):
		raise ValueError("'A' must be a numpy array.")

	if A.ndim != 2:
		raise ValueError("'A' must be a 2D matrix.")

	if A.shape[0] != A.shape[1]:
		raise ValueError("'A' is not a square matrix.")

	# prepare parameters:
	A = A.astype(np.float64)  # type cast
	m = len(A)  # size of matrix
	b = normalize(np.random.rand(m))  # randomize a normalized rank 1 array
	
	# diagonalize:
	Q, alpha, beta = lanczos(A, b)
	T = tridiag_matrix(alpha, beta)

	return Q, T, Q.T
