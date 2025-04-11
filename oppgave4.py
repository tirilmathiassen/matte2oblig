"""

Tiril Mathiassen
Matte 2 oblig - Oppgave 4: Løse varmelikningen med eksplisitt skjema

"""

# Importerer nødvendige biblioteker 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parametre
L = 1             # Lengde på stang
T = 0.1           # Sluttid
n = 50            # Antall punkt i rom
m = 500           # Antall punkt i tid
h = 0.2        # Romsteg
k = 0.0001         # Tidssteg
alpha = k / h**2  # Må være <= 0.5 

# Gitter
x = np.linspace(0, L, n+1)
t = np.linspace(0, T, m+1)

# Initialbetingelse
u = np.zeros((n+1, m+1))
u[:,0] = np.sin(np.pi * x)  # Initial: u(x,0) = sin(pi x)

# Fra randkrav:
u[0,:] = 0
u[-1,:] = 0

# Eksplisitt skjema
for j in range(0, m):
    for i in range(1, n):
        u[i,j+1] = u[i,j] + alpha * (u[i+1,j] - 2*u[i,j] + u[i-1,j])

# Animasjon
fig, ax = plt.subplots()
line, = ax.plot(x, u[:,0])
ax.set_ylim(-1, 1)

def animate(j):
    line.set_ydata(u[:,j])
    return line,

ani = animation.FuncAnimation(fig, animate, frames=m, interval=20)
plt.show()
