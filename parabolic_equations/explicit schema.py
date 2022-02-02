# Numerical solution of a equation
# -du/dt=d^2u/dx^2 + f(x,t) with x on (a,b) and t>=0
# mu0=u(x,0)=2x; mu1=u(a,t)=3t
# mu2=u(b,t)=t^2+3t+2

# Solution is u(x)= x^3*t^2+3t+2x
# So f(x)=3+2*t*(x^3)-6x*(t^2)
# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

a = 0
b = 1
T = 10

# кол-во узлов по Ox
n = 50
# шаг по Ox
h = (b - a) / n

# кол-во узлов по Ot
m = 100000
# шаг по Oy
tau = T / m


def f(x, t):
    return 3 + 2 * (x ** 3) * t - 6 * x * (t ** 2)


def mu0(x):
    return 2 * x


def mu1(t):
    return 3 * t


def mu2(t):
    return t ** 2 + 3 * t + 2



assert tau <= 0.5 * h * h, "tay should be less than 0.5*h*h"

print("tay= ", tau, "<= 0.5*h*h=", 0.5 * h * h)

u = np.zeros((m, n - 1))
u = np.r_[np.array([[mu0(a + i * h) for i in range(1, n)]]), u]

u = np.c_[np.array([[mu1(i * tau) for i in range(m + 1)]]).T, u, np.array(
    [[mu2(i * tau) for i in range(m + 1)]]).T]

for i in range(1, m + 1):
    for j in range(1, n):
        u[i, j] = u[i - 1, j] + tau * (
                    (u[i - 1, j - 1] - 2 * u[i - 1, j] + u[i - 1, j + 1]) / (h * h) + f(a + j * h, i * tau))
        # print((u[i-1,j-1]-2*u[i-1,j]+u[i-1,j+1])/(h*h))

print("The resulting function", u)


# %% Check
def u_t(x, t):
    return (x ** 3) * (t ** 2) + 3 * t + 2 * x


u_t_d = np.zeros((m + 1, n + 1))

for i in range(m + 1):
    for j in range(n + 1):
        u_t_d[i, j] = u_t(a + j * h, i * tau)

# print(u_t_d)

MSE = (abs(u_t_d - u) ** 2).sum() / ((n + 1) * (m + 1))
print("MSE: ", MSE)

X = np.linspace(a, b, n + 1)
Tt = np.linspace(0, T, m + 1)
X, Tt = np.meshgrid(X, Tt)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# The resulting function
ax.plot_surface(X, Tt, u, cmap='plasma')
plt.title("Numerical solution of explicit schema")
plt.savefig("Numerical_solution_explicit_schema.png")
plt.show()

# Original function
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Tt, u_t(X, Tt), cmap='plasma')
plt.title("Original function of explicit schema")
plt.savefig("Original_function_explicit_schema.png")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Tt, abs(u_t(X, Tt) - u), cmap='plasma')
plt.title("Residues of explicit schema")
plt.savefig("Residues_explicit_schema.png")
plt.show()

pd.DataFrame(u).to_csv("Numerical_solution_explicit_schema.csv")
