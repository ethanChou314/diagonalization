import numpy as np


def normalize(v):
	norm = np.linalg.norm(v)
	if np.isclose(norm, 0.):
		raise ValueError("Cannot normalize zero vector.")
	return v / norm


def tridiag_matrix(alpha, beta):
	if alpha.ndim != 1 or beta.ndim != 1:
		raise ValueError("alpha and beta must be 1D arrays")
	if len(beta) != len(alpha) - 1:
		raise ValueError("beta must be one element shorter than alpha")

	n = len(alpha)
	T = np.zeros((n, n))
	for i in range(n):
		T[i, i] = alpha[i]
		if i > 0:
			T[i, i - 1] = beta[i - 1]
		if i < n - 1:
			T[i, i + 1] = beta[i]
	return T


def lanczos(A, b, n, m):
	if b.shape[0] != n:
		raise ValueError(f"Expected b of length {n}, got {len(b)}")

	Q = np.zeros((n, m))
	alpha = np.zeros(m)
	beta = np.zeros(m - 1)

	Q[:, 0] = normalize(b)
	q_im1 = np.zeros((n, 1))

	for i in range(m):
		q_i = Q[:, i].reshape(-1, 1)
		w = A @ q_i
		alpha[i] = float(q_i.T @ w)
		v = w - alpha[i] * q_i - (beta[i - 1] * q_im1 if i > 0 else 0)

		if i < m - 1:
			beta[i] = np.linalg.norm(v)
			q_ip1 = normalize(v)
			Q[:, i + 1] = q_ip1.flatten()
			q_im1 = q_i

	return Q, alpha, beta


def exact_diagonalization(A, m):
	A = np.asarray(A, dtype=float)
	n = A.shape[0]

	if A.ndim != 2 or n != A.shape[1]:
		raise ValueError("A must be square")
	if not (0 < m <= n):
		raise ValueError("Invalid value of m")

	b = normalize(np.random.rand(n))
	Q, alpha, beta = lanczos(A, b, n, m)
	T = tridiag_matrix(alpha, beta)

	return Q, T, Q.T