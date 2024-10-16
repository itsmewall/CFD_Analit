import numpy as np
import matplotlib.pyplot as plt

# Função para gerar o perfil NACA 0012
def naca0012(x, c=1.0):
    t = 0.12  # Espessura máxima
    return 5 * t * (0.2969 * np.sqrt(x/c) - 0.1260 * (x/c) - 0.3516 * (x/c)**2 + 0.2843 * (x/c)**3 - 0.1015 * (x/c)**4)

# Parâmetros do aerofólio
N_panels = 10000  # Número de painéis (divisões)
alpha = 10 * np.pi / 180  # Ângulo de ataque (em radianos)
V_inf = 20  # Velocidade do fluxo livre (m/s)
rho = 1.225  # Densidade do ar (kg/m³)

# Gerar o contorno do perfil NACA 0012
x_panel = np.linspace(0, 1, N_panels + 1)  # Painéis definidos de 0 a 1 (com N_panels+1 pontos)
y_upper = naca0012(x_panel)
y_lower = -naca0012(x_panel)

# Coordenadas dos pontos de controle e dos painéis (usamos a média entre pontos adjacentes)
X_control = (x_panel[:-1] + x_panel[1:]) / 2
Y_control = (y_upper[:-1] + y_lower[1:]) / 2

# Montar sistema de equações para vorticidade usando Vortex Panel Method
def build_matrix(X_control, Y_control, x_panel, y_upper, y_lower, alpha):
    A = np.zeros((N_panels, N_panels))
    rhs = np.zeros(N_panels)
    for i in range(N_panels):
        for j in range(N_panels):
            if i == j:
                A[i, j] = np.pi  # Evitar singularidade (auto-interação)
            else:
                dx = X_control[i] - X_control[j]
                dy = Y_control[i] - Y_control[j]
                A[i, j] = (dy * np.cos(alpha) - dx * np.sin(alpha)) / (dx**2 + dy**2)
        rhs[i] = -V_inf * np.sin(alpha)
    return A, rhs

# Resolver a vorticidade
A, rhs = build_matrix(X_control, Y_control, x_panel, y_upper, y_lower, alpha)
gamma = np.linalg.solve(A, rhs)

# Calcular a sustentação total do perfil
L_total = np.sum(gamma) * rho * V_inf
print(f"Sustentação Total: {L_total:.2f} N")

# Visualizar a distribuição de vorticidade (circulação)
plt.plot(X_control, gamma)
plt.title('Distribuição de Circulação ao Longo do Perfil (Vortex Panel Method)')
plt.xlabel('Posição ao Longo do Perfil')
plt.ylabel('Circulação (gamma)')
plt.grid(True)
plt.show()
