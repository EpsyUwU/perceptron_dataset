import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from matplotlib.animation import FuncAnimation, PillowWriter
from pandastable import Table


class Perceptron:
    def __init__(self, input_dim, learning_rate=0.01):
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn(1)
        self.learning_rate = learning_rate
        self.history = {'weights': [], 'bias': [], 'mse': [], 'y_pred': []}

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def train(self, X, y, epochs, error_threshold):
        for epoch in range(epochs):
            for i in range(X.shape[0]):
                y_pred = self.predict(X[i])
                error = y[i] - y_pred
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error
            mse = np.mean((self.predict(X) - y) ** 2)
            y_pred_full = self.predict(X)
            self.history['weights'].append(self.weights.copy())
            self.history['bias'].append(self.bias.copy())
            self.history['mse'].append(mse)
            self.history['y_pred'].append(y_pred_full.copy())
            if mse <= error_threshold:
                print(f"Epoch {epoch}, Error: {mse:.4f}")
                break
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Error: {mse:.4f}")

        final_epoch, final_mse = epoch, mse
        print(f"Final Epoch {final_epoch}, Final Error: {final_mse:.4f}")

    def plot_weights_evolution(self):
        epochs = len(self.history['weights'])
        fig, ax = plt.subplots()

        for i in range(len(self.weights)):
            weights_evolution = [self.history['weights'][e][i] for e in range(epochs)]
            ax.plot(range(epochs), weights_evolution, label=f'Weight {i+1}')

        ax.plot(range(epochs), self.history['bias'], label='Bias', linestyle='--')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')
        ax.set_title('Evolución de los Pesos y el Sesgo')
        ax.legend()
        plt.grid(True)

        # Guardar la gráfica como imagen temporal
        plot_filename = 'weights_evolution_plot.png'
        plt.savefig(plot_filename)
        plt.close()

        return plot_filename

    def plot_mse_evolution(self):
        epochs = len(self.history['mse'])
        fig, ax = plt.subplots()
        ax.plot(range(epochs), self.history['mse'])
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE')
        ax.set_title('Evolución del Error Cuadrático Medio')
        plt.grid(True)

        # Guardar la gráfica como imagen temporal
        plot_filename = 'mse_plot.png'
        plt.savefig(plot_filename)
        plt.close()

        return plot_filename

    def generate_y_animation(self, X, y_true, fps=10):
        fig, ax = plt.subplots()

        def update(frame):
            epoch = frame
            mse = self.history['mse'][frame]
            y_pred_full = self.history['y_pred'][frame]
            ax.clear()
            ax.plot(range(len(y_true)), y_true, marker='o', color='blue', linestyle='-', label='Y Deseado')
            ax.plot(range(len(y_true)), y_pred_full[:len(y_true)], marker='x', color='orange', linestyle='--',
                    label='Y Aproximado')
            ax.set_title(f'Epoch {epoch}, Error: {mse:.4f}')
            ax.legend()

        anim = FuncAnimation(fig, update, frames=len(self.history['y_pred']), repeat=False)

        # Guardar la animación como GIF temporal
        gif_filename = 'y_evolution.gif'
        anim.save(gif_filename, writer='pillow', fps=fps)

        plt.close(fig)  # Cerrar la figura después de guardar la animación

        return gif_filename

    def get_final_weights_table(self):
        final_weights = np.append(self.weights, self.bias)
        index_names = [f'Weight {i+1}' for i in range(len(final_weights) - 1)] + ['Bias']
        weight_names = ['Weight 1', 'Weight 2', 'Weight 3', 'Weight 4', 'Bias']
        df = pd.DataFrame({'Weight Name': weight_names, 'Value': final_weights})
        return df

    def plot_y_comparison(self, X, y_true):
        y_pred_final = self.history['y_pred'][-1]
        fig, ax = plt.subplots()
        ax.plot(range(len(y_true)), y_true, marker='o', color='blue', linestyle='-', label='Y Deseado')
        ax.plot(range(len(y_true)), y_pred_final[:len(y_true)], marker='x', color='orange', linestyle='--',
                label='Y Aproximado')
        ax.set_title('Comparación de Y Deseado vs Y Aproximado')
        ax.set_xlabel('Índice de Muestra')
        ax.set_ylabel('Valor')
        ax.legend()
        plt.grid(True)

        # Guardar la gráfica como imagen temporal
        plot_filename = 'y_comparison_plot.png'
        plt.savefig(plot_filename)
        plt.close()

        return plot_filename



def train_perceptron():
    learning_rate = float(entry_learning_rate.get())
    error_threshold = float(entry_error_threshold.get())
    epochs = int(entry_epochs.get())

    global perceptron
    input_dim = X_train.shape[1]
    perceptron = Perceptron(input_dim, learning_rate)
    perceptron.train(X_train, y_train, epochs, error_threshold)

    # Habilitar los botones después del entrenamiento
    btn_plot_weights.config(state=tk.NORMAL)
    btn_plot_mse.config(state=tk.NORMAL)
    btn_show_gif.config(state=tk.NORMAL)
    btn_show_table.config(state=tk.NORMAL)
    btn_show_y_comparison.config(state=tk.NORMAL)
    messagebox.showinfo("Entrenamiento Completado", "El perceptrón ha sido entrenado con éxito.")


def show_weights_evolution():
    global perceptron

    if not hasattr(perceptron, 'history'):
        messagebox.showwarning("Error", "Primero debes entrenar el perceptrón.")
        return

    # Limpiar cualquier contenido previo en weight_frame
    for widget in weight_frame.winfo_children():
        widget.destroy()

    # Mostrar la imagen del gráfico de evolución de los pesos y el sesgo
    plot_filename = perceptron.plot_weights_evolution()

    # Cargar la imagen en el Label correspondiente
    plot_image = tk.PhotoImage(file=plot_filename)
    plot_label = ttk.Label(weight_frame, image=plot_image)
    plot_label.image = plot_image
    plot_label.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)


def show_mse_plot():
    global perceptron

    if not hasattr(perceptron, 'history'):
        messagebox.showwarning("Error", "Primero debes entrenar el perceptrón.")
        return

    # Limpiar cualquier contenido previo en weight_frame
    for widget in weight_frame.winfo_children():
        widget.destroy()

    # Mostrar la imagen del gráfico de MSE en el Label correspondiente
    plot_filename = perceptron.plot_mse_evolution()

    # Cargar la imagen en el Label correspondiente
    plot_image = tk.PhotoImage(file=plot_filename)
    plot_label = ttk.Label(weight_frame, image=plot_image)
    plot_label.image = plot_image
    plot_label.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)


def show_gif():
    global perceptron

    if not hasattr(perceptron, 'history'):
        messagebox.showwarning("Error", "Primero debes entrenar el perceptrón.")
        return

    # Limpiar cualquier contenido previo en weight_frame
    for widget in weight_frame.winfo_children():
        widget.destroy()

    # Generar la animación Y Deseado vs Y Aproximado
    gif_filename = perceptron.generate_y_animation(X_train, y_train)

    # Cargar el GIF en el Label correspondiente
    gif_image = tk.PhotoImage(file=gif_filename)
    gif_label = ttk.Label(weight_frame, image=gif_image)
    gif_label.image = gif_image
    gif_label.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)

def show_weights_table():
    global perceptron

    if not hasattr(perceptron, 'weights'):
        messagebox.showwarning("Error", "Primero debes entrenar el perceptrón.")
        return

    # Obtener la tabla de pesos finales
    df = perceptron.get_final_weights_table()

    # Crear una ventana nueva para mostrar la tabla
    table_window = tk.Toplevel(root)
    table_window.title("Pesos Finales del Perceptrón")

    # Crear la tabla con PandasTable
    table = Table(table_window, dataframe=df, showtoolbar=True, showstatusbar=True)
    table.show()

def show_y_comparison():
    global perceptron

    if not hasattr(perceptron, 'history'):
        messagebox.showwarning("Error", "Primero debes entrenar el perceptrón.")
        return

    # Limpiar cualquier contenido previo en weight_frame
    for widget in weight_frame.winfo_children():
        widget.destroy()

    # Mostrar la imagen de la comparación Y Deseado vs Y Aproximado
    plot_filename = perceptron.plot_y_comparison(X_train, y_train)

    # Cargar la imagen en el Label correspondiente
    plot_image = tk.PhotoImage(file=plot_filename)
    plot_label = ttk.Label(weight_frame, image=plot_image)
    plot_label.image = plot_image
    plot_label.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)


# Cargar los datos desde un archivo Excel
df = pd.read_excel("data.xlsx", skiprows=1)
X = df[['x1', 'x2', 'x3', 'x4']].values
y = df["y"].values

# Normalizar los datos
scaler = StandardScaler()
X_train, y_train = scaler.fit_transform(X), y

# Configurar la interfaz gráfica
root = tk.Tk()
root.title("Perceptrón - Interfaz Gráfica")
root.geometry('900x600')

# Frame para los campos de entrada y los botones
left_frame = ttk.Frame(root)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

# Campos de entrada para Learning Rate, Error Threshold y Epochs
label_learning_rate = ttk.Label(left_frame, text="Learning Rate:")
label_learning_rate.pack(pady=5)
entry_learning_rate = ttk.Entry(left_frame)
entry_learning_rate.pack(pady=5)

label_error_threshold = ttk.Label(left_frame, text="Error Threshold:")
label_error_threshold.pack(pady=5)
entry_error_threshold = ttk.Entry(left_frame)
entry_error_threshold.pack(pady=5)

label_epochs = ttk.Label(left_frame, text="Epochs:")
label_epochs.pack(pady=5)
entry_epochs = ttk.Entry(left_frame)
entry_epochs.pack(pady=5)

# Botón para entrenar el perceptrón
btn_train = ttk.Button(left_frame, text="Entrenar Perceptrón", command=train_perceptron)
btn_train.pack(pady=10)

# Frame para mostrar la imagen del gráfico de evolución de pesos y sesgo
weight_frame = ttk.Frame(root)
weight_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Botón para mostrar gráfico de evolución de pesos y sesgo
btn_plot_weights = ttk.Button(left_frame, text="Mostrar Evolución de Pesos y Sesgo", command=show_weights_evolution)
btn_plot_weights.pack(pady=10)
btn_plot_weights.config(state=tk.DISABLED)

# Botón para mostrar gráfico de MSE
btn_plot_mse = ttk.Button(left_frame, text="Mostrar Gráfico de MSE", command=show_mse_plot)
btn_plot_mse.pack(pady=10)
btn_plot_mse.config(state=tk.DISABLED)

btn_show_table = ttk.Button(left_frame, text="Mostrar Tabla de Pesos Finales", command=show_weights_table)
btn_show_table.pack(pady=10)
btn_show_table.config(state=tk.DISABLED)

btn_show_y_comparison = ttk.Button(left_frame, text="Mostrar Comparación de Y", command=show_y_comparison)
btn_show_y_comparison.pack(pady=10)
btn_show_y_comparison.config(state=tk.DISABLED)

btn_show_gif = ttk.Button(left_frame, text="Mostrar Gif", command=show_gif)
btn_show_gif.pack(pady=10)
btn_show_gif.config(state=tk.DISABLED)

root.mainloop()
