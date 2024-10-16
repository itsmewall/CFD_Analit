import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# Parâmetros do domínio
nx = 200  # Número de pontos em x (aumente para maior resolução)
ny = 161  # Número de pontos em y
nt = 50  # Número de passos de tempo
nit = 50  # Iterações para resolver a pressão
c = 1     # Velocidade característica

dx = 2.0 / (nx - 1)  # Espaçamento em x
dy = 1.0 / (ny - 1)  # Espaçamento em y

x = np.linspace(-1.0, 1.0, nx)
y = np.linspace(-0.5, 0.5, ny)
X, Y = np.meshgrid(x, y)

# Parâmetros do fluido
rho_fluido = 1.0     # Densidade do fluido
nu = 0.1      # Viscosidade cinemática
dt = 0.001    # Passo de tempo

# Parâmetros do modelo k-ω
k = np.ones((ny, nx)) * 0.1  # Energia cinética turbulenta inicial
omega = np.ones((ny, nx)) * 0.1  # Dissipação específica inicial
beta_star = 0.09  # Constante do modelo k-omega

# Função para gerar o perfil NACA 0012
def naca0012(x):
    t = 0.12  # Espessura máxima (12% da corda)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * (x) -
                  0.3516 * (x) ** 2 + 0.2843 * (x) ** 3 - 0.1015 * (x) ** 4)
    return yt

# Gerar o contorno superior e inferior do aerofólio NACA 0012
x_airfoil = np.linspace(0, 1.0, 200)
y_upper = naca0012(x_airfoil)  # Contorno superior
y_lower = -naca0012(x_airfoil)  # Contorno inferior

# Função para a etapa de pressão do método SIMPLE
def simple_pressure_correction(p, u, v, dx, dy, rho, nit):
    pn = np.empty_like(p)
    for _ in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy ** 2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx ** 2) /
                         (2 * (dx ** 2 + dy ** 2)) -
                         dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) *
                         ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                          (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)))

        # Condições de contorno para a pressão
        p[:, -1] = p[:, -2]     # dp/dx = 0 na saída
        p[:, 0] = p[:, 1]       # dp/dx = 0 na entrada
        p[0, :] = p[1, :]       # dp/dy = 0 na parede inferior
        p[-1, :] = p[-2, :]     # dp/dy = 0 na parede superior
    return p

# Função para calcular o modelo k-omega
def solve_k_omega(k, omega, u, v, dx, dy, dt, beta_star):
    # Gerar e dissipar energia cinética turbulenta
    dkdx = (k[1:-1, 2:] - k[1:-1, :-2]) / (2 * dx)
    dkdy = (k[2:, 1:-1] - k[:-2, 1:-1]) / (2 * dy)

    # Ajustar as bordas para garantir que as dimensões sejam consistentes
    u_mid = u[1:-1, 1:-1]
    v_mid = v[1:-1, 1:-1]
    k_mid = k[1:-1, 1:-1]
    omega_mid = omega[1:-1, 1:-1]

    # Dissipação específica de turbulência (omega)
    omega_new = omega_mid + dt * (
        - u_mid * dkdx - v_mid * dkdy
        + beta_star * (k_mid * omega_mid)
    )

    k_new = k_mid + dt * (
        - u_mid * dkdx - v_mid * dkdy
    )

    # Atualizando os valores de k e omega para o domínio completo
    k[1:-1, 1:-1] = k_new
    omega[1:-1, 1:-1] = omega_new

    return k, omega

# Inicializar campos de velocidade e pressão
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))

# Loop de tempo
for n in range(nt):
    un = u.copy()
    vn = v.copy()

    # Correção de pressão usando SIMPLE
    p = simple_pressure_correction(p, u, v, dx, dy, rho_fluido, nit)

    # Atualizar velocidades com a pressão corrigida
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     dt * ((p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)) +
                     nu * dt * ((un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) / dx ** 2 +
                                (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1]) / dy ** 2))

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     dt * ((p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)) +
                     nu * dt * ((vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) / dx ** 2 +
                                (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1]) / dy ** 2))

    # Resolver o modelo k-omega para turbulência
    k, omega = solve_k_omega(k, omega, u, v, dx, dy, dt, beta_star)

    # Feedback no terminal
    if (n + 1) % 50 == 0:
        print(f"Passo de tempo {n + 1}/{nt} concluído")

print("Simulação concluída!")

# Visualização do campo de velocidade
plt.figure(figsize=(12, 6))
masked_velocity = np.sqrt(u ** 2 + v ** 2)
plt.contourf(X, Y, masked_velocity, levels=100, cmap=cm.jet)
plt.colorbar(label='Magnitude da Velocidade')

plt.plot(x_airfoil, naca0012(x_airfoil), 'k', linewidth=2)
plt.plot(x_airfoil, -naca0012(x_airfoil), 'k', linewidth=2)

plt.title('Campo de Velocidade ao Redor do Aerofólio NACA 0012')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Visualização do campo de pressão
plt.figure(figsize=(12, 6))
plt.contourf(X, Y, p, levels=100, cmap=cm.jet)
plt.colorbar(label='Pressão')

plt.plot(x_airfoil, naca0012(x_airfoil), 'k', linewidth=2)
plt.plot(x_airfoil, -naca0012(x_airfoil), 'k', linewidth=2)

plt.title('Campo de Pressão ao Redor do Aerofólio NACA 0012')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
