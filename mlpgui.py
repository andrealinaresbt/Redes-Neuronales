import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import pickle
import numpy as np

# Asume que tienes tu clase MLP definida en otro archivo llamado mlp_module.py
from mlp import MLP  # Asegúrate de ajustar esto si está en el mismo archivo

class MLPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MLP Interfaz Gráfica")
        self.network = None

        # Botones principales
        tk.Button(root, text="1. Crear nuevo MLP", command=self.create_network).pack(pady=5)
        tk.Button(root, text="2. Cargar MLP desde archivo", command=self.load_network).pack(pady=5)
        tk.Button(root, text="3. Entrenar MLP", command=self.train_network).pack(pady=5)
        tk.Button(root, text="4. Realizar predicciones", command=self.predict).pack(pady=5)
        tk.Button(root, text="5. Guardar MLP en archivo", command=self.save_network).pack(pady=5)
        tk.Button(root, text="6. Salir", command=root.quit).pack(pady=5)

    def create_network(self):
        try:
            input_size = simpledialog.askinteger("Crear MLP", "Tamaño de entrada:")
            hidden_sizes = simpledialog.askstring("Crear MLP", "Tamaños de capas ocultas (separados por comas):")
            output_size = simpledialog.askinteger("Crear MLP", "Tamaño de salida:")
            learning_rate = simpledialog.askfloat("Crear MLP", "Tasa de aprendizaje:")

            hidden_sizes = [int(x) for x in hidden_sizes.split(',')]
            self.network = MLP(input_size, hidden_sizes, output_size, learning_rate)
            messagebox.showinfo("Éxito", "Red creada exitosamente.")
        except Exception as e:
            messagebox.showerror("Error", f"Error al crear red: {e}")

    def load_network(self):
        filename = filedialog.askopenfilename(title="Seleccionar archivo .pkl", filetypes=[("Pickle files", "*.pkl")])
        if filename:
            try:
                with open(filename, 'rb') as file:
                    self.network = pickle.load(file)
                messagebox.showinfo("Cargado", "Red cargada exitosamente.")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")

    def train_network(self):
        if not self.network:
            messagebox.showwarning("Advertencia", "Primero crea o carga una red.")
            return

        try:
            X_file = filedialog.askopenfilename(title="Seleccionar archivo de entrada (X)")
            y_file = filedialog.askopenfilename(title="Seleccionar archivo de salida (y)")

            X = np.loadtxt(X_file)
            y = np.loadtxt(y_file)

            epochs = simpledialog.askinteger("Entrenamiento", "Número de épocas:")

            self.network.train(X, y, epochs)
            messagebox.showinfo("Éxito", "Entrenamiento completado.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo entrenar: {e}")

    def predict(self):
        if not self.network:
            messagebox.showwarning("Advertencia", "Primero crea o carga una red.")
            return

        try:
            X_file = filedialog.askopenfilename(title="Seleccionar archivo de entrada para predicción")
            X = np.loadtxt(X_file)
            predictions = self.network.predict(X)

            result_window = tk.Toplevel(self.root)
            result_window.title("Resultados de la predicción")

            tk.Label(result_window, text="Predicciones:").pack()
            tk.Text(result_window, width=50, height=10).pack()
            text_box = result_window.children['!text']
            text_box.insert('1.0', str(predictions))
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo predecir: {e}")

    def save_network(self):
        if not self.network:
            messagebox.showwarning("Advertencia", "Primero crea o carga una red.")
            return

        filename = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")])
        if filename:
            try:
                with open(filename, 'wb') as file:
                    pickle.dump(self.network, file)
                messagebox.showinfo("Guardado", "Red guardada correctamente.")
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo guardar: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = MLPApp(root)
    root.mainloop()
