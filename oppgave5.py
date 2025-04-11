"""

Tiril Mathiassen
Matte 2 oblig - Oppgave 5: Løse varmelikningen med implisitt skjema

"""

# Importerer nødvendige biblioteker 
import numpy as np
import matplotlib.pyplot as plt

# Parametre
L = 1
T = 0.1
n = 50
m = 500
h = L / n
k = T / m
alpha = k / h**2  # Alltid stabil

# Gitter
x = np.linspace(0, L, n+1)
t = np.linspace(0, T, m+1)

# Initialisering
u = np.zeros((n+1, m+1))
u[:,0] = np.sin(np.pi * x)

# Fra randkrav:
u[0,:] = 0
u[-1,:] = 0

# Koefisientmatrise A (tridiagonal)
A = np.zeros((n-1, n-1))
for i in range(n-1):
    A[i,i] = 1 + 2*alpha
    if i > 0:
        A[i,i-1] = -alpha
    if i < n-2:
        A[i,i+1] = -alpha

# Implisitt tidsstepping
for j in range(m):
    b = u[1:n,j]  # høyresiden i systemet (uten randbetingelser)
    u[1:n,j+1] = np.linalg.solve(A, b)

# Plot sluttresultat
plt.plot(x, u[:,0], label="t=0")
plt.plot(x, u[:,int(m/2)], label=f"t={T/2}")
plt.plot(x, u[:,-1], label=f"t={T}")
plt.legend()
plt.title("Implisitt Euler for varmelikningen")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.grid(True)
plt.show()

