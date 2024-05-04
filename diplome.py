
import  numpy as np

def jacobi(A, b, x0, tol=1e-6, max_iter=100):
  """
  Решает систему линейных уравнений Ax = b методом Якоби.

  Args:
    A: Матрица коэффициентов (numpy.ndarray).
    b: Вектор правой части (numpy.ndarray).
    x0: Начальное приближение решения (numpy.ndarray).
    tol: Точность решения (float).
    max_iter: Максимальное число итераций (int).

  Returns:
    numpy.ndarray: Решение системы уравнений.
  """
  n = len(A)
  x = x0.copy()
  for _ in range(max_iter):
    x_new = np.zeros_like(x)
    for i in range(n):
      s = 0
      for j in range(n):
        if i != j:
          s += A[i, j] * x[j]
      x_new[i] = (b[i] - s) / A[i, i]
    if np.linalg.norm(x_new - x) < tol:
      return x_new
    x = x_new
  raise ValueError("Метод Якоби не сошелся")


def gauss_seidel(A, b, x0, tol=1e-6, max_iter=100):
  """
  Решает систему линейных уравнений Ax = b методом Гаусса-Зейделя.

  Args:
    A: Матрица коэффициентов (numpy.ndarray).
    b: Вектор правой части (numpy.ndarray).
    x0: Начальное приближение решения (numpy.ndarray).
    tol: Точность решения (float).
    max_iter: Максимальное число итераций (int).

  Returns:
    numpy.ndarray: Решение системы уравнений.
  """
  n = len(A)
  x = x0.copy()
  for _ in range(max_iter):
    for i in range(n):
      s = 0
      for j in range(n):
        if i != j:
          s += A[i, j] * x[j]
      x[i] = (b[i] - s) / A[i, i]
    if np.linalg.norm(x - x0) < tol:
      return x
    x0 = x.copy()
  raise ValueError("Метод Гаусса-Зейделя не сошелся")



def sor(A, b, x0, omega, tol=1e-6, max_iter=100):
  """
  Решает систему линейных уравнений Ax = b методом верхней релаксации (SOR).

  Args:
    A: Матрица коэффициентов (numpy.ndarray).
    b: Вектор правой части (numpy.ndarray).
    x0: Начальное приближение решения (numpy.ndarray).
    omega: Параметр релаксации (float).
    tol: Точность решения (float).
    max_iter: Максимальное число итераций (int).

  Returns:
    numpy.ndarray: Решение системы уравнений.
  """
  n = len(A)
  x = x0.copy()
  for _ in range(max_iter):
    for i in range(n):
      s = 0
      for j in range(n):
        if i != j:
          s += A[i, j] * x[j]
      x[i] = (1 - omega) * x[i] + omega * (b[i] - s) / A[i, i]
    if np.linalg.norm(x - x0) < tol:
      return x
    x0 = x.copy()
  raise ValueError("Метод SOR не сошелся")




A = np.array([[4, 1, 1], [1, 5, 2], [1, 2, 6]])
b = np.array([7, -8, 6])
x0 = np.zeros(3)

x_jacobi = jacobi(A, b, x0)
x_gauss_seidel = gauss_seidel(A, b, x0)
x_sor = sor(A, b, x0, omega=1.1)

print("Решение методом Якоби:", x_jacobi)
print("Решение методом Гаусса-Зейделя:", x_gauss_seidel)
print("Решение методом SOR:", x_sor)

