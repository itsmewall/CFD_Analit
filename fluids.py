import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.path import Path
import imageio

# Parâmetros do domínio
nx = 401  # Aumentado para melhorar a resolução espacial
ny = 201
Lx = 2.0
Ly = 1.0
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

# Propriedades do fluido
rho = 1.0    # Densidade
nu = 0.001   # Viscosidade cinemática ajustada para fluxo viscoso

# Ângulo de ataque
alpha = 14  # Em graus
alpha_rad = np.radians(alpha)

# Inicialização das variáveis
u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))  # Termo fonte para a equação de Poisson da pressão

# Condições de entrada (perfil de velocidade)
u_in = np.cos(alpha_rad) * 1.0
v_in = np.sin(alpha_rad) * 1.0
u[:, 0] = u_in
v[:, 0] = v_in

# Função para gerar o perfil NACA 0012
def naca0012(x):
    t = 0.12
    return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 +
                    0.2843 * x**3 - 0.1015 * x**4)

# Coordenadas do aerofólio
x_airfoil = np.linspace(0, 1.0, 400)
y_upper = naca0012(x_airfoil)
y_lower = -naca0012(x_airfoil)

# Rotacionando o aerofólio pelo ângulo de ataque
x_center = 0.5
y_center = 0.0
cos_alpha = np.cos(alpha_rad)
sin_alpha = np.sin(alpha_rad)

x_airfoil_rotated = cos_alpha * (x_airfoil - x_center) - sin_alpha * (y_upper - y_center) + x_center
y_upper_rotated = sin_alpha * (x_airfoil - x_center) + cos_alpha * (y_upper - y_center) + y_center
y_lower_rotated = sin_alpha * (x_airfoil - x_center) + cos_alpha * (y_lower - y_center) + y_center

# Criando uma máscara para o aerofólio no domínio
X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(-Ly/2, Ly/2, ny))
airfoil_coords = np.concatenate((np.vstack((x_airfoil_rotated, y_upper_rotated)).T,
                                 np.vstack((x_airfoil_rotated[::-1], y_lower_rotated[::-1])).T))
airfoil_path = Path(airfoil_coords)
points = np.vstack((X.flatten(), Y.flatten())).T
airfoil_mask = airfoil_path.contains_points(points).reshape(X.shape)

# Aplicando condição de não deslizamento no aerofólio
u[airfoil_mask] = 0
v[airfoil_mask] = 0

# Parâmetros do critério CFL
CFL = 0.01  # Reduzido para melhorar a estabilidade

# Configuração para salvar os frames
frames = []
fig, ax = plt.subplots(figsize=(12,6))

# Loop principal de tempo
nt = 100000  # Aumentado para prolongar a simulação
frame_interval = 100

for n in range(nt):
    un = u.copy()
    vn = v.copy()

    # Ajuste do passo de tempo baseado no critério CFL
    u_max = np.max(np.abs(un)) + 1e-5
    v_max = np.max(np.abs(vn)) + 1e-5
    dt = CFL * min(dx / u_max, dy / v_max)

    # Construir o termo fonte b
    b[1:-1,1:-1] = (rho * (1/dt *
        ((un[1:-1,2:] - un[1:-1,0:-2]) / (2*dx) +
         (vn[2:,1:-1] - vn[0:-2,1:-1]) / (2*dy)) -
        ((un[1:-1,2:] - un[1:-1,0:-2]) / (2*dx))**2 -
        2 * ((un[2:,1:-1] - un[0:-2,1:-1]) / (2*dy) *
             (vn[1:-1,2:] - vn[1:-1,0:-2]) / (2*dx)) -
        ((vn[2:,1:-1] - vn[0:-2,1:-1]) / (2*dy))**2))

    # Resolver a equação de Poisson para a pressão
    pn = p.copy()
    for _ in range(100):  # Aumentado para melhorar a convergência
        pn = p.copy()
        p[1:-1,1:-1] = (((pn[1:-1,2:] + pn[1:-1,0:-2]) * dy**2 +
                         (pn[2:,1:-1] + pn[0:-2,1:-1]) * dx**2) /
                        (2 * (dx**2 + dy**2)) -
                        dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1,1:-1])

        # Condições de contorno para a pressão
        p[:, -1] = p[:, -2]   # dp/dx = 0 na saída
        p[:, 0] = p[:, 1]     # dp/dx = 0 na entrada
        p[-1, :] = p[-2, :]   # dp/dy = 0 no topo
        p[0, :] = p[1, :]     # dp/dy = 0 na base

        # Condições no aerofólio
        p[airfoil_mask] = 0

    # Atualizar as velocidades usando esquemas upwind de segunda ordem
    # Termos convectivos
    u_convective = ((un[1:-1,1:-1] * (un[1:-1,1:-1] - un[1:-1,0:-2]) / dx) +
                    (vn[1:-1,1:-1] * (un[1:-1,1:-1] - un[0:-2,1:-1]) / dy))

    v_convective = ((un[1:-1,1:-1] * (vn[1:-1,1:-1] - vn[1:-1,0:-2]) / dx) +
                    (vn[1:-1,1:-1] * (vn[1:-1,1:-1] - vn[0:-2,1:-1]) / dy))

    # Termos difusivos
    u_diffusive = nu * ((un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,0:-2]) / dx**2 +
                        (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[0:-2,1:-1]) / dy**2)

    v_diffusive = nu * ((vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,0:-2]) / dx**2 +
                        (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[0:-2,1:-1]) / dy**2)

    # Atualização das velocidades
    u[1:-1,1:-1] = (un[1:-1,1:-1] -
                    dt * u_convective +
                    dt * u_diffusive -
                    dt / (rho * 2 * dx) * (p[1:-1,2:] - p[1:-1,0:-2]))

    v[1:-1,1:-1] = (vn[1:-1,1:-1] -
                    dt * v_convective +
                    dt * v_diffusive -
                    dt / (rho * 2 * dy) * (p[2:,1:-1] - p[0:-2,1:-1]))

    # Aplicando condições de contorno
    u[0, :] = u_in   # Velocidade na base
    u[-1, :] = u_in  # Velocidade no topo
    u[:, 0] = u_in   # Entrada
    u[:, -1] = u[:, -2]  # Saída (du/dx = 0)

    v[0, :] = v_in
    v[-1, :] = v_in
    v[:, 0] = v_in
    v[:, -1] = v[:, -2]  # Saída (dv/dx = 0)

    # Condições no aerofólio
    u[airfoil_mask] = 0
    v[airfoil_mask] = 0

    # Verificação de NaNs
    if np.isnan(u).any() or np.isnan(v).any() or np.isnan(p).any():
        print(f"NaNs encontrados na iteração {n}")
        break

    # Impressão dos cálculos no terminal
    if n % 500 == 0:
        max_u = np.max(np.abs(u))
        max_v = np.max(np.abs(v))
        max_p = np.max(np.abs(p))
        print(f"Iteração {n}: max_u = {max_u:.5f}, max_v = {max_v:.5f}, max_p = {max_p:.5f}")

    # Captura de frames para o GIF
    if n % frame_interval == 0:
        ax.clear()
        magnitude = np.sqrt(u**2 + v**2)
        contour = ax.contourf(X, Y, magnitude, levels=100, cmap=cm.jet)
        ax.plot(x_airfoil_rotated, y_upper_rotated, 'k', linewidth=2)
        ax.plot(x_airfoil_rotated, y_lower_rotated, 'k', linewidth=2)
        ax.set_title(f'Simulação de Fluxo ao Redor do Aerofólio (α = {alpha}°)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.tight_layout()

        # Salvar o frame atual
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

# Criar o GIF ao final da simulação
imageio.mimsave('simulacao_aerofolio.gif', frames, fps=30)

print("Simulação concluída. O GIF foi salvo como 'simulacao_aerofolio.gif'.")
