import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parametre
L = 1
T = 0.1
n = 50
m = 300
h = L / n
k = T / m
alpha = k / h**2

x = np.linspace(0, L, n+1)
t = np.linspace(0, T, m+1)

# Initialbetingelse
def initial():
    return np.sin(np.pi * x)

# Eksplisitt Euler
def eksplisitt_euler():
    u = np.zeros((n+1, m+1))
    u[:,0] = initial()
    for j in range(m):
        for i in range(1, n):
            u[i,j+1] = u[i,j] + alpha * (u[i+1,j] - 2*u[i,j] + u[i-1,j])
    return u

# Implisitt Euler
def implisitt_euler():
    u = np.zeros((n+1, m+1))
    u[:,0] = initial()
    A = np.zeros((n-1, n-1))
    for i in range(n-1):
        A[i,i] = 1 + 2 * alpha
        if i > 0:
            A[i,i-1] = -alpha
        if i < n-2:
            A[i,i+1] = -alpha
    for j in range(m):
        b = u[1:n,j]
        u[1:n,j+1] = np.linalg.solve(A, b)
    return u

# Crank-Nicolson
def crank_nicolson():
    u = np.zeros((n+1, m+1))
    u[:,0] = initial()
    A = np.zeros((n-1, n-1))
    B = np.zeros((n-1, n-1))
    for i in range(n-1):
        A[i,i] = 1 + alpha
        B[i,i] = 1 - alpha
        if i > 0:
            A[i,i-1] = -alpha/2
            B[i,i-1] = alpha/2
        if i < n-2:
            A[i,i+1] = -alpha/2
            B[i,i+1] = alpha/2
    for j in range(m):
        b = B @ u[1:n,j]
        u[1:n,j+1] = np.linalg.solve(A, b)
    return u

# Analytisk løsning
def analytisk_løsning():
    U = np.zeros((n+1, m+1))
    for j in range(m+1):
        U[:,j] = np.exp(-np.pi**2 * t[j]) * np.sin(np.pi * x)
    return U

# Beregner alle løsninger
Ue = eksplisitt_euler()
Ui = implisitt_euler()
Uc = crank_nicolson()
Ua = analytisk_løsning()

# Lager animasjon
fig, ax = plt.subplots()
line_e, = ax.plot(x, Ue[:,0], label="Eksplisitt", linestyle='--')
line_i, = ax.plot(x, Ui[:,0], label="Implisitt", linestyle='-.')
line_c, = ax.plot(x, Uc[:,0], label="Crank-Nicolson", linestyle=':')
line_a, = ax.plot(x, Ua[:,0], label="Analytisk", linestyle= "dotted", color="black")
ax.set_ylim(-0.1, 1.1)
ax.set_title("Sammenligning av løsninger")
ax.set_xlabel("x")
ax.set_ylabel("u(x,t)")
ax.legend()
ax.grid()

def update(j):
    line_e.set_ydata(Ue[:,j])
    line_i.set_ydata(Ui[:,j])
    line_c.set_ydata(Uc[:,j])
    line_a.set_ydata(Ua[:,j])
    ax.set_title(f"Sammenligning av løsninger, t = {t[j]:.3f}")
    return line_e, line_i, line_c, line_a

ani = FuncAnimation(fig, update, frames=m+1, interval=40, blit=False)
plt.show()
