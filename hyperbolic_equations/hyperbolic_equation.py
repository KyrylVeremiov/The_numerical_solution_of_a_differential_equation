# Numerical solution of a equation
# -d^2u/dt^2=d^2u/dx^2 with x on (a,b) and t>=0
# phi0(t)=u(a,t)=t^2+a^2+a; phi1(t)=u(b,t)=t^2+b^2+b
# psi(x)=u(x,0)=x^2+x; psi_h(x)=du/dt(x,0)=0; psi_xx(x)=d^2u/dx^2(x,0)=2

# Solution is u(x)= x^2+t^2+x
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

a = 0
b = 1
T0 = 0
T = 4

# number of spans by Ox
# n = 4
n = 100
# шаг по Ox
h = (b - a) / n

# number of spans by Oy
# m = 16
m = 400
# шаг по Oy
tau = T / m

gamma = tau * tau / (h * h)


def phi0(t):
    return t * t + a * a + a


def phi1(t):
    return t * t + b * b + b


def psi(x):
    return x + x * x


def psi_h(x):
    return 0


def psi_xx(x):
    return 2


assert gamma <= 1, "gamma should be less than 1"

print("gamma= ", gamma, "<= 1")

# %%

u = np.zeros((m, n - 1))
u = np.r_[np.array([[psi(a + i * h) for i in range(1, n)]]), u]

u = np.c_[np.array([[phi0(i * tau) for i in range(m + 1)]]).T, u, np.array(
    [[phi1(i * tau) for i in range(m + 1)]]).T]

for j in range(1, n):
    u[1, j] = (-psi_xx(a + h * j) * tau / 2 + psi_h(a + h * j)) * tau \
              + u[0, j]

for i in range(1, m):
    for j in range(1, n):
        u[i + 1, j] = 2 * u[i, j] - u[i - 1, j] + \
                      gamma * (u[i, j - 1] - 2 * u[i, j] + u[i, j + 1])

print("The resulting function", u)


# %% Check
def u_t(x, t):
    return x * x + t * t + x


u_t_d = np.zeros((m + 1, n + 1))

for i in range(m + 1):
    for j in range(n + 1):
        u_t_d[i, j] = u_t(a + h * j, tau * i)

# print(u_t_d)

MSE = (abs(u_t_d - u) ** 2).sum() / ((n + 1) * (m + 1))
print("MSE: ", MSE)

X = np.linspace(a, b, n + 1)
Tt = np.linspace(T0, T, m + 1)
X, Tt = np.meshgrid(X, Tt)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# The resulting function
ax.plot_surface(X, Tt, u, cmap='plasma')
plt.title("Numerical_solution")
plt.savefig("Numerical_solution.png")
plt.show()

# Original function
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Tt, u_t(X, Tt), cmap='plasma')
plt.title("Original function")
plt.savefig("Original_function.png")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Tt, abs(u_t(X, Tt) - u), cmap='plasma')
plt.title("Residues")
plt.savefig("Residues.png")
plt.show()

pd.DataFrame(u).to_csv("Numerical_solution.csv")
