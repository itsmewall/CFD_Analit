import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.path import Path
import imageio
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
from PIL import Image, ImageTk
import time

def run_simulation():
    start_time = time.time()
    
    nx = int(entry_nx.get())
    ny = int(entry_ny.get())
    alpha = int(entry_alpha.get())
    nt = int(entry_nt.get())
    nu = float(entry_nu.get())
    Lx = 2.0
    Ly = 1.0
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    alpha_rad = np.radians(alpha)

    rho = 1.0
    CFL = 0.1

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))

    u_in = np.cos(alpha_rad) * 1.0
    v_in = np.sin(alpha_rad) * 1.0
    u[:, 0] = u_in
    v[:, 0] = v_in

    def naca0012(x):
        t = 0.12
        return 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 +
                        0.2843 * x**3 - 0.1015 * x**4)

    x_airfoil = np.linspace(0, 1.0, 400)
    y_upper = naca0012(x_airfoil)
    y_lower = -naca0012(x_airfoil)

    x_center = 0.5
    y_center = 0.0
    cos_alpha = np.cos(alpha_rad)
    sin_alpha = np.sin(alpha_rad)

    x_airfoil_rotated = cos_alpha * (x_airfoil - x_center) - sin_alpha * (y_upper - y_center) + x_center
    y_upper_rotated = sin_alpha * (x_airfoil - x_center) + cos_alpha * (y_upper - y_center) + y_center
    y_lower_rotated = sin_alpha * (x_airfoil - x_center) + cos_alpha * (y_lower - y_center) + y_center

    X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(-Ly/2, Ly/2, ny))
    airfoil_coords = np.concatenate((np.vstack((x_airfoil_rotated, y_upper_rotated)).T,
                                     np.vstack((x_airfoil_rotated[::-1], y_lower_rotated[::-1])).T))
    airfoil_path = Path(airfoil_coords)
    points = np.vstack((X.flatten(), Y.flatten())).T
    airfoil_mask = airfoil_path.contains_points(points).reshape(X.shape)

    u[airfoil_mask] = 0
    v[airfoil_mask] = 0

    frames = []
    fig, ax = plt.subplots(figsize=(12,6))

    frame_interval = 200
    progress_bar['maximum'] = nt

    for n in range(nt):
        un = u.copy()
        vn = v.copy()

        u_max = np.max(np.abs(un)) + 1e-5
        v_max = np.max(np.abs(vn)) + 1e-5
        dt = CFL * min(dx / u_max, dy / v_max)

        b[1:-1,1:-1] = (rho * (1/dt *
            ((un[1:-1,2:] - un[1:-1,0:-2]) / (2*dx) +
             (vn[2:,1:-1] - vn[0:-2,1:-1]) / (2*dy)) - 
            ((un[1:-1,2:] - un[1:-1,0:-2]) / (2*dx))**2 -
            2 * ((un[2:,1:-1] - un[0:-2,1:-1]) / (2*dy) *
                 (vn[1:-1,2:] - vn[1:-1,0:-2]) / (2*dx)) - 
            ((vn[2:,1:-1] - vn[0:-2,1:-1]) / (2*dy))**2))

        pn = p.copy()
        for _ in range(100):
            pn = p.copy()
            p[1:-1,1:-1] = (((pn[1:-1,2:] + pn[1:-1,0:-2]) * dy**2 +
                             (pn[2:,1:-1] + pn[0:-2,1:-1]) * dx**2) /
                            (2 * (dx**2 + dy**2)) - 
                            dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1,1:-1])

            p[:, -1] = p[:, -2]
            p[:, 0] = p[:, 1]
            p[-1, :] = p[-2, :]
            p[0, :] = p[1, :]
            p[airfoil_mask] = 0

        u[1:-1,1:-1] = (un[1:-1,1:-1] - dt * ((p[1:-1,2:] - p[1:-1,0:-2]) / (2*dx)))
        v[1:-1,1:-1] = (vn[1:-1,1:-1] - dt * ((p[2:,1:-1] - p[0:-2,1:-1]) / (2*dy)))
        u[airfoil_mask] = 0
        v[airfoil_mask] = 0

        if n % frame_interval == 0:
            ax.clear()
            magnitude = np.sqrt(u**2 + v**2)
            contour = ax.contourf(X, Y, magnitude, levels=100, cmap=cm.jet)
            ax.plot(x_airfoil_rotated, y_upper_rotated, 'k', linewidth=2)
            ax.plot(x_airfoil_rotated, y_lower_rotated, 'k', linewidth=2)
            ax.set_title(f'Simulação de Fluxo (α = {alpha}°)')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.tight_layout()
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            frames.append(image)

        progress_bar['value'] = n
        root.update_idletasks()

    imageio.mimsave('Wally_Simu.gif', frames, fps=30)
    end_time = time.time()
    elapsed_time = end_time - start_time
    messagebox.showinfo("Concluído", f"Simulação concluída em {elapsed_time:.2f} segundos. O GIF foi salvo como 'Wally_Simu.gif'.")
    display_gif('Wally_Simu.gif')

def display_gif(gif_path):
    gif_window = tk.Toplevel(root)
    gif_window.title("Resultado da Simulação")
    gif_label = tk.Label(gif_window)
    gif_label.pack()
    gif_image = Image.open(gif_path)
    gif_frames = [ImageTk.PhotoImage(gif_image.copy()) for frame in range(gif_image.n_frames)]

    def animate(index):
        gif_label.config(image=gif_frames[index])
        gif_window.after(50, animate, (index + 1) % len(gif_frames))

    animate(0)

root = tk.Tk()
root.title("Simulador de Fluxo ao Redor do Aerofólio")

tk.Label(root, text="Número de Pontos em X:").pack()
entry_nx = tk.Entry(root)
entry_nx.insert(0, "401")
entry_nx.pack()

tk.Label(root, text="Número de Pontos em Y:").pack()
entry_ny = tk.Entry(root)
entry_ny.insert(0, "201")
entry_ny.pack()

tk.Label(root, text="Ângulo de Ataque (graus):").pack()
entry_alpha = tk.Entry(root)
entry_alpha.insert(0, "0")
entry_alpha.pack()

tk.Label(root, text="Número de Iterações:").pack()
entry_nt = tk.Entry(root)
entry_nt.insert(0, "15000")
entry_nt.pack()

tk.Label(root, text="Viscosidade Cinemática (nu):").pack()
entry_nu = tk.Entry(root)
entry_nu.insert(0, "0.001")
entry_nu.pack()

progress_bar = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=300, mode='determinate')
progress_bar.pack(pady=10)

tk.Button(root, text="Executar Simulação", command=run_simulation).pack()
root.mainloop()
