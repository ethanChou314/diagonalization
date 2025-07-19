import numpy as np


def project(a, u):
	"""
	a: a (n,) vector
	u: a (n,) unit vector that a will be projected onto
	"""
	if not np.isclose(np.linalg.norm(u), 1.0):
		raise ValueError("'u' is not a unit vector.")

	return a.dot(u) * u


def normalize(v):
	return v / np.linalg.norm(v)


def gs_process(A):
	ncol = A.shape[1]
	Q = np.empty_like(A, dtype=float)
	for i in range(ncol):  # iterate over columns
		q = A[:, i].copy()
		for j in range(i):
			q -= project(A[:, i], Q[:, j])
		Q[:, i] = normalize(q)
	return Q


def qr_decomposition(A):
	"""
	Performs QR decomposition on A, where R is 
	the upper triangular matrix.
	"""
	A = np.asarray(A, dtype=float)

	if (A.ndim != 2):
		raise ValueError("Expected A to be a 2D array.")

	if (A.shape[0] != A.shape[1]):
		raise ValueError("Dimension mismatch: 'A' must be a square matrix.")

	Q = gs_process(A)
	R = Q.T @ A
	return Q, R
