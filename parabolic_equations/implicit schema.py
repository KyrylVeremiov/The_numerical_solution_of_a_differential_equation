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
# T=1
T = 10

# кол-во узлов по Ox
# n=3
n = 50
# шаг по Ox
h = (b - a) / (n+1)

# кол-во узлов по Ot
# m=4
m = 1000
# шаг по Oy
tau = T / m


def f(x, t):
    return 3+2*(x**3)*t-6*x*(t**2)


def mu0(x):
    return 2*x


def mu1(t):
    return 3*t


def mu2(t):
    return t**2+3*t+2



def matrix_portrait(M, caption):
    for i in range(len(M)):
        for j in range(len(M)):
            if M[i,j]!=0:
                plt.plot(i+1,j+1,'b*')
    plt.title(caption)
    plt.show()

#%%
# Решение явной схемой

gamma= tau / (h * h)
u = np.zeros((m, n))
u = np.r_[np.array([[mu0(a + i * h) for i in range(1,n+1)]]),u]

u = np.c_[np.array([[mu1(i * tau) for i in range(m + 1)]]).T, u, np.array(
    [[mu2(i * tau) for i in range(m + 1)]]).T]

for i in range(1,m+1):
    t= i * tau
    A=np.zeros((n,n))
    B=np.zeros((n,1))
    for j in range(1,n+1):
        x = a + ((j-1) % n+1) * h
        A[j-1,j-1]=-(1+2*gamma)
        B[j-1]=-(tau * f(x, (i - 1) * tau) + u[i - 1, j])
        if j-1==0:
            B[j-1]-=gamma*u[i,j-1]
        else:
            A[j-1,j-2]=gamma
        if j== n:
            B[j-1]-=gamma*u[i,j+1]
        else:
            A[j-1,j]=gamma
    U=np.linalg.solve(A, B)
    for j in range(n):
        u[i,j+1] = U[j]

print("The resulting function", u)


# %% Check
def u_t(x, t):
    return (x**3)*(t**2)+3*t+2*x


u_t_d = np.zeros((m+1, n+2))

for i in range(m+1):
    for j in range(n+2):
        u_t_d[i, j] = u_t(a + j * h, i * tau)

# print(u_t_d)

MSE = (abs(u_t_d - u) ** 2).sum() / ((n+2) * (m+1))
print("MSE: ", MSE)

X = np.linspace(a, b, n + 2)
Tt = np.linspace(0, T, m + 1)
X, Tt = np.meshgrid(X, Tt)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# The resulting function
ax.plot_surface(X, Tt, u, cmap='plasma')
plt.title("Numerical solution of implicit schema")
plt.savefig("Numerical_solution_implicit_schema.png")
plt.show()

# Original function
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Tt, u_t(X, Tt), cmap='plasma')
plt.title("Original function of implicit schema")
plt.savefig("Original_function_implicit_schema.png")
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Tt, abs(u_t(X, Tt) - u), cmap='plasma')
plt.title("Residues of implicit schema")
plt.savefig("Residues_implicit_schema.png")
plt.show()

pd.DataFrame(u).to_csv("Numerical_solution_implicit_schema.csv")