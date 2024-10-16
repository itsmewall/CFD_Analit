import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# Parâmetros do domínio
nx = 100  # Número de pontos em x (aumente para maior resolução)
ny = 161  # Número de pontos em y
nt = 5000  # Número de passos de tempo
nit = 10  # Iterações para resolver a pressão
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

# Definir a densidade e massa do aerofólio
densidade_aerofolio = 1000  # kg/m² (densidade uniforme)
corda_aerofolio = 0.40  # Corda do aerofólio (m)

# Função para gerar o perfil NACA 0012
def naca0012(x):
    t = 0.12  # Espessura máxima (12% da corda)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * (x) -
                  0.3516 * (x) ** 2 + 0.2843 * (x) ** 3 - 0.1015 * (x) ** 4)
    return yt

# Gerar o contorno superior e inferior do aerofólio NACA 0012
x_airfoil = np.linspace(0, corda_aerofolio, 200)
y_upper = naca0012(x_airfoil)  # Contorno superior
y_lower = -naca0012(x_airfoil)  # Contorno inferior

# Calcular a área do aerofólio (2D) usando integração trapezoidal simples
def calcular_area(x_airfoil, y_upper, y_lower):
    area = 0.0
    for i in range(1, len(x_airfoil)):
        dx = x_airfoil[i] - x_airfoil[i - 1]
        avg_height = (y_upper[i] - y_lower[i] + y_upper[i - 1] - y_lower[i - 1]) / 2
        area += dx * avg_height
    return area

# Calculando a massa do aerofólio
area_aerofolio = calcular_area(x_airfoil, y_upper, y_lower)
massa_aerofolio = densidade_aerofolio * area_aerofolio

# Imprimir a massa do aerofólio
print(f"Área do aerofólio: {area_aerofolio:.4f} m²")
print(f"Massa do aerofólio: {massa_aerofolio:.2f} kg")

# Criar máscara para o aerofólio usando o Método da Fronteira Imersa
airfoil_mask = np.zeros((ny, nx), dtype=bool)
for i in range(nx):
    for j in range(ny):
        xi = x[i]
        yi = y[j]
        if 0 <= xi <= corda_aerofolio:
            y_upper = naca0012(xi)
            y_lower = -y_upper
            if y_lower <= yi <= y_upper:
                airfoil_mask[j, i] = True

# Função para atualizar o termo de fonte b
def build_up_b(b, u, v, dx, dy, dt, rho_fluido):
    b[1:-1, 1:-1] = (rho_fluido * (1 / dt *
                            ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                             (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) -
                            ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)) ** 2 -
                            2 * ((u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                                 (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)) -
                            ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) ** 2))
    return b

# Função para resolver a equação de Poisson para a pressão
def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy ** 2 +
                          (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx ** 2) /
                         (2 * (dx ** 2 + dy ** 2)) -
                         dx ** 2 * dy ** 2 / (2 * (dx ** 2 + dy ** 2)) *
                         b[1:-1, 1:-1])

        # Condições de contorno para a pressão
        p[:, -1] = p[:, -2]     # dp/dx = 0 na saída
        p[:, 0] = p[:, 1]       # dp/dx = 0 na entrada
        p[0, :] = p[1, :]       # dp/dy = 0 na parede inferior
        p[-1, :] = p[-2, :]     # dp/dy = 0 na parede superior

    return p

# Inicializar campos de velocidade e pressão
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))

# Loop de tempo (simulação básica de fluxo ao redor do aerofólio)
for n in range(nt):
    un = u.copy()
    vn = v.copy()

    b = build_up_b(b, u, v, dx, dy, dt, rho_fluido)
    p = pressure_poisson(p, dx, dy, b)

    # Atualizar velocidades u e v
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                     (un[1:-1, 1:-1] - un[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                     (un[1:-1, 1:-1] - un[:-2, 1:-1]) -
                     dt / (2 * rho_fluido * dx) *
                     (p[1:-1, 2:] - p[1:-1, :-2]) +
                     nu * (dt / dx ** 2 *
                           (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) +
                           dt / dy ** 2 *
                           (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                     un[1:-1, 1:-1] * dt / dx *
                     (vn[1:-1, 1:-1] - vn[1:-1, :-2]) -
                     vn[1:-1, 1:-1] * dt / dy *
                     (vn[1:-1, 1:-1] - vn[:-2, 1:-1]) -
                     dt / (2 * rho_fluido * dy) *
                     (p[2:, 1:-1] - p[:-2, 1:-1]) +
                     nu * (dt / dx ** 2 *
                           (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, :-2]) +
                           dt / dy ** 2 *
                           (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[:-2, 1:-1])))

    # Aplicar condições de contorno
    u[:, 0] = 1  # u = 1 na entrada
    u[0, :] = 0
    u[-1, :] = 0
    v[:, 0] = 0
    v[:, -1] = 0

    # Aplicar condições de não deslizamento no aerofólio
    u[airfoil_mask] = 0
    v[airfoil_mask] = 0

    # Feedback no terminal
    if (n + 1) % 50 == 0:
        print(f"Passo de tempo {n + 1}/{nt} concluído")

print("Simulação concluída.")

# Visualização do campo de velocidade
plt.figure(figsize=(12, 6))

# Aplicar máscara para não mostrar dados dentro do aerofólio
masked_velocity = np.sqrt(u ** 2 + v ** 2)
masked_velocity[airfoil_mask] = np.nan  # Não mostrar valores dentro do aerofólio

plt.contourf(X, Y, masked_velocity, levels=100, cmap=cm.jet)
plt.colorbar(label='Magnitude da Velocidade')

# Desenhar o contorno do aerofólio corretamente
plt.plot(x_airfoil, naca0012(x_airfoil), 'k', linewidth=2)
plt.plot(x_airfoil, -naca0012(x_airfoil), 'k', linewidth=2)

plt.title('Campo de Velocidade ao Redor do Aerofólio NACA 0012')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Visualização do campo de pressão
plt.figure(figsize=(12, 6))

# Aplicar máscara para não mostrar dados dentro do aerofólio
masked_pressure = p.copy()
masked_pressure[airfoil_mask] = np.nan  # Não mostrar valores dentro do aerofólio

plt.contourf(X, Y, masked_pressure, levels=100, cmap=cm.jet)
plt.colorbar(label='Pressão')

# Desenhar o contorno do aerofólio corretamente
plt.plot(x_airfoil, naca0012(x_airfoil), 'k', linewidth=2)
plt.plot(x_airfoil, -naca0012(x_airfoil), 'k', linewidth=2)

plt.title('Campo de Pressão ao Redor do Aerofólio NACA 0012')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
