import numpy as np
from numpy.linalg import det, inv, eig, matrix_rank

def matrix_addition(matrix_a, matrix_b):
    """Add two matrices."""
    try:
        a = np.array(matrix_a)
        b = np.array(matrix_b)
        if a.shape != b.shape:
            raise ValueError("Matrices must have the same dimensions for addition")
        return (a + b).tolist()
    except Exception as e:
        raise ValueError(f"Error in matrix addition: {str(e)}")

def matrix_subtraction(matrix_a, matrix_b):
    """Subtract matrix_b from matrix_a."""
    try:
        a = np.array(matrix_a)
        b = np.array(matrix_b)
        if a.shape != b.shape:
            raise ValueError("Matrices must have the same dimensions for subtraction")
        return (a - b).tolist()
    except Exception as e:
        raise ValueError(f"Error in matrix subtraction: {str(e)}")

def matrix_multiplication(matrix_a, matrix_b):
    """Multiply two matrices."""
    try:
        a = np.array(matrix_a)
        b = np.array(matrix_b)
        if a.shape[1] != b.shape[0]:
            raise ValueError("Number of columns in first matrix must equal number of rows in second matrix")
        return np.matmul(a, b).tolist()
    except Exception as e:
        raise ValueError(f"Error in matrix multiplication: {str(e)}")

def scalar_multiplication(matrix, scalar):
    """Multiply a matrix by a scalar."""
    try:
        a = np.array(matrix)
        return (scalar * a).tolist()
    except Exception as e:
        raise ValueError(f"Error in scalar multiplication: {str(e)}")

def determinant(matrix):
    """Calculate the determinant of a matrix."""
    try:
        a = np.array(matrix)
        if a.shape[0] != a.shape[1]:
            raise ValueError("Matrix must be square to calculate determinant")
        return float(det(a))
    except Exception as e:
        raise ValueError(f"Error calculating determinant: {str(e)}")

def inverse_matrix(matrix):
    """Calculate the inverse of a matrix."""
    try:
        a = np.array(matrix)
        if a.shape[0] != a.shape[1]:
            raise ValueError("Matrix must be square to calculate inverse")
        if abs(det(a)) < 1e-10:
            raise ValueError("Matrix is singular, inverse does not exist")
        return inv(a).tolist()
    except Exception as e:
        raise ValueError(f"Error calculating inverse: {str(e)}")

def transpose_matrix(matrix):
    """Calculate the transpose of a matrix."""
    try:
        a = np.array(matrix)
        return a.T.tolist()
    except Exception as e:
        raise ValueError(f"Error calculating transpose: {str(e)}")

def eigenvalues_eigenvectors(matrix):
    """Calculate eigenvalues and eigenvectors of a matrix."""
    try:
        a = np.array(matrix)
        if a.shape[0] != a.shape[1]:
            raise ValueError("Matrix must be square to calculate eigenvalues and eigenvectors")
        eigenvalues, eigenvectors = eig(a)
        return eigenvalues.tolist(), eigenvectors.tolist()
    except Exception as e:
        raise ValueError(f"Error calculating eigenvalues and eigenvectors: {str(e)}")

import numpy as np
from numpy.linalg import matrix_rank

import numpy as np
from numpy.linalg import svd

def matrix_rank(matrix):
    """Calculate the rank of a matrix with enhanced numerical stability."""
    try:
        a = np.array(matrix, dtype=np.float64)
        
        # Convert very small numbers to zero
        a[np.abs(a) < 1e-12] = 0
        
        # Manual SVD-based rank calculation
        s = svd(a, compute_uv=False)
        # Tolerance based on matrix size and data type
        tol = max(a.shape) * np.spacing(np.linalg.norm(s, np.inf))
        rank = np.sum(s > tol)
        return int(rank)
    except Exception as e:
        raise ValueError(f"Rank calculation error: {str(e)}")

def adjoint_matrix(matrix):
    """Calculate the adjoint (adjugate) of a matrix."""
    try:
        a = np.array(matrix)
        if a.shape[0] != a.shape[1]:
            raise ValueError("Matrix must be square to calculate adjoint")
        
        # Adjoint = Determinant * Inverse (if determinant is non-zero)
        det_a = det(a)
        if abs(det_a) < 1e-10:
            # For a singular matrix, we calculate the adjoint differently
            n = a.shape[0]
            adjoint = np.zeros((n, n))
            
            for i in range(n):
                for j in range(n):
                    # Get minor by removing row i and column j
                    minor = np.delete(np.delete(a, i, axis=0), j, axis=1)
                    adjoint[j, i] = ((-1) ** (i + j)) * det(minor)
            
            return adjoint.tolist()
        else:
            # For non-singular matrix: adjoint = determinant * inverse
            return (det_a * inv(a)).tolist()
    except Exception as e:
        raise ValueError(f"Error calculating adjoint: {str(e)}")

def matrix_trace(matrix):
    """Calculate the trace of a matrix (sum of diagonal elements)."""
    try:
        a = np.array(matrix)
        if a.shape[0] != a.shape[1]:
            raise ValueError("Matrix must be square to calculate trace")
        return float(np.trace(a))
    except Exception as e:
        raise ValueError(f"Error calculating trace: {str(e)}")