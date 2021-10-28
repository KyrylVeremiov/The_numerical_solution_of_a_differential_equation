# Решение краевой задачи
# -u'' + qu = f на x на (a1,b1); y на (a2,b2)
# phi1=u(x,a2)=x; phi2=u(a1,y)=y^3+1+y
# phi3=u(x,b2)=8x^2+x+2x; phi4=(u-du/dx)(b1,y)=-1-y

# Решается модельная задача с известным
# решением u(x)= x^2y^3+x+xy; q(x)= 1
# В этом случае: f(x)=x^2*y^3+x+xy-2y^3-6yx^2


# Здесь q(x) >= 0 на (a,b)
# %%
import matplotlib.pyplot as plt
import numpy as np

a1 = 0
b1 = 1
a2 = 0
b2 = 2

# кол-во промежутков по Ox
n = 50
# шаг по Ox
h1 = (b1 - a1) / n

# кол-во промежутков по Oy
m = 100
# шаг по Oy
h2 = (b2 - a2) / m

q = 1


def f(x, y):
    return x ** 2 * (y ** 3) + x + x * y - 2 * (y ** 3) - 6 * y * (x ** 2)


def phi1(x, y):
    return x


def phi2(x, y):
    return y ** 3 + 1 + y


def phi3(x, y):
    return 8 * (x ** 2) + x + 2 * x


def phi4(x, y):
    return -1 - y


def phi(x, y):
    if y == a2:
        return phi1(x, y)
    elif x == b1:
        return phi2(x, y)
    elif y == b2:
        return phi3(x, y)
    elif x == a1:
        return phi4(x, y)
    else:
        raise IOError("Error in phi")


# %%
# формирование матрицы A и правой части b СЛАУ Au=b
A = np.zeros((n * (m - 1), n * (m - 1)))
B = np.zeros((n * (m - 1), 1))

a = 2 / (h1 * h1) + 2 / (h2 * h2)
b = -1 / (h1 * h1)
c = -1 / (h2 * h2)
d = 1 / h1

for i in range(n * (m - 1)):
    x = a1 + (i % n) * h1
    y = a2 + (int(i / n) + 1) * h2
    if i % n == 0:
        A[i, i] = 1 + d
        A[i, i + 1] = -d
        B[i] = phi(x, y)
    else:
        A[i, i] = a + q
        A[i, i - 1] = b
        if i % n == n - 1:
            B[i] = f(x, y) - b * phi(b1, y)
        else:
            A[i, i + 1] = b
            B[i] = f(x, y)

        if i + n < n * (m - 1):
            A[i, i + n] = c
        else:
            B[i] -= c * phi(x, b2)
        if i - n > 0:
            A[i, i - n] = c
        else:
            B[i] -= c * phi(x, a2)

U = np.linalg.solve(A, B)
u = np.zeros((m - 1, n))
for i in range(m - 1):
    for j in range(n):
        u[i, j] = U[i * n + j]

# print(u)
Phi2 = np.array([[phi2(a1, a2 + i * h2) for i in range(m - 1)]]).T
# print(Phi2)
u = np.c_[u, Phi2]
# The resulting function
u = np.r_[np.array([[phi1(a1 + i * h1, a2) for i in range(n + 1)]]), u, np.array(
    [[phi3(a1 + i * h1, a2) for i in range(n + 1)]])]
print("The resulting function", u)


# %% Check
def u_t(x, y):
    return x ** 2 * (y ** 3) + x + x * y


u_t_d = np.zeros((m + 1, n + 1))

for i in range(m + 1):
    for j in range(n + 1):
        u_t_d[i, j] = u_t(a1 + h1 * j, a2 + h2 * i)

# print(u_t_d)

MSE = (abs(u_t_d - u) ** 2).sum() / ((n + 1) * (m + 1))
print("MSE: ", MSE)

X = np.linspace(a1, b1, n + 1)
Y = np.linspace(a2, b2, m + 1)
X, Y = np.meshgrid(X, Y)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# The resulting function
ax.plot_surface(X, Y, u, cmap='plasma')
plt.title("Numerical_solution")
plt.savefig("Numerical_solution.png")
plt.show()

# Original function
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, u_t(X, Y), cmap='plasma')
plt.title("Original function")
plt.savefig("Original_function.png")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, u_t(X, Y) - u, cmap='plasma')
plt.title("Residues")
plt.savefig("Residues.png")
plt.show()
