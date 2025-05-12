import numpy as np
import pickle  # Importar pickle para la serialización
import matplotlib.pyplot as plt

# Inicializar listas para almacenar las precisiones por época
train_accuracies = []
test_accuracies = []


class MLP:
    def __init__(self, input_neurons, output_neurons, hidden_layers, neurons_per_hidden_layer, weights=None, biases=None):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.hidden_layers = hidden_layers
        self.neurons_per_hidden_layer = neurons_per_hidden_layer

        # Inicialización de pesos y sesgos
        layer_sizes = [self.input_neurons] + [self.neurons_per_hidden_layer] * self.hidden_layers + [self.output_neurons]
        
        if weights is None or biases is None:
            self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
            self.biases = [np.random.randn(1, layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]
        else:
            self.weights = weights
            self.biases = biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.layer_outputs = [inputs]
        for i in range(self.hidden_layers + 1):
            net_input = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            output = self.sigmoid(net_input)
            self.layer_outputs.append(output)
        return self.layer_outputs[-1]

    def backward(self, inputs, expected_output, learning_rate):
        deltas = []
        output = self.layer_outputs[-1]
        error = expected_output - output
        delta = error * self.sigmoid_derivative(output)
        deltas.insert(0, delta)

        for i in range(self.hidden_layers, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.sigmoid_derivative(self.layer_outputs[i])
            deltas.insert(0, delta)

        for i in range(self.hidden_layers + 1):
            self.weights[i] += np.dot(self.layer_outputs[i].T, deltas[i]) * learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate

    def train(self, data, labels, epochs, learning_rate=0.5, test_data=None, test_labels=None):
        train_accuracies = []  # Para almacenar la precisión de cada época
        test_accuracies = []   # Para almacenar la precisión de prueba de cada época

        for epoch in range(epochs):
            total_error = 0  # Para almacenar el error total de la época
            for inputs, expected_output in zip(data, labels):
                inputs = inputs.reshape(1, -1)
                expected_output = expected_output.reshape(1, -1)

                self.forward(inputs)
                self.backward(inputs, expected_output, learning_rate)

                # Calcula el error cuadrático medio
                error = np.mean((expected_output - self.layer_outputs[-1])**2)
                total_error += error

            # Calcula precisión de entrenamiento
            train_accuracy = self.calculate_accuracy(data, labels)
            train_accuracies.append(train_accuracy)

            # Calcula precisión de prueba si se proporcionaron datos de prueba
            if test_data is not None and test_labels is not None:
                test_accuracy = self.calculate_accuracy(test_data, test_labels)
                test_accuracies.append(test_accuracy)

            # Imprimir el error al final de cada época
            print(f"Época {epoch + 1}/{epochs} - Error total: {total_error}")
            print(f"Precisión de entrenamiento: {train_accuracy}")
            if test_data is not None and test_labels is not None:
                print(f"Precisión de prueba: {test_accuracy}")

        # Después de todas las épocas, graficar los resultados
        import matplotlib.pyplot as plt
        plt.plot(range(1, epochs + 1), train_accuracies, label='Precisión de Entrenamiento')
        if test_data is not None and test_labels is not None:
            plt.plot(range(1, epochs + 1), test_accuracies, label='Precisión de Prueba')

        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.title('Precisión por Época')
        plt.legend()
        plt.show()

    def calculate_accuracy(self, data, labels):
        # Aquí va el cálculo de precisión de la red (por ejemplo, comparación de etiquetas predichas y reales)
        correct = 0
        total = len(data)
        for inputs, expected_output in zip(data, labels):
            prediction = self.predict(inputs)
            if np.round(prediction) == expected_output:
                correct += 1
        return correct / total

    def predict(self, inputs):
        inputs = inputs.reshape(1, -1)
        return self.forward(inputs)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'input_neurons': self.input_neurons,
                'output_neurons': self.output_neurons,
                'hidden_layers': self.hidden_layers,
                'neurons_per_hidden_layer': self.neurons_per_hidden_layer,
                'weights': self.weights,
                'biases': self.biases
            }, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return cls(
            input_neurons=data['input_neurons'],
            output_neurons=data['output_neurons'],
            hidden_layers=data['hidden_layers'],
            neurons_per_hidden_layer=data['neurons_per_hidden_layer'],
            weights=data['weights'],
            biases=data['biases']
        )



def load_data(input_file, output_file=None):
    inputs = np.loadtxt(input_file, delimiter=',')  # Cargar entradas
    if output_file:
        outputs = np.loadtxt(output_file, delimiter=',')  # Cargar salidas
    else:
        outputs = None
    return inputs, outputs


def main():
    while True:
        print("--- Menú ---")
        print("1. Crear un nuevo perceptrón multicapa")
        print("2. Cargar un perceptrón desde archivo")
        print("3. Salir")
        option = input("Seleccione una opción: ")

        if option == "1":
            input_neurons = int(input("Número de neuronas de entrada: "))
            output_neurons = int(input("Número de neuronas de salida: "))
            hidden_layers = int(input("Número de capas ocultas: "))
            neurons_per_hidden_layer = int(input("Número de neuronas por capa oculta: "))

            # Archivos
            train_file = input("Archivo de entrenamiento (con .txt): ")
            output_train_file = input("Archivo de salidas esperadas a los datos de entrenamiento (con .txt): ")
            test_file = input("Archivo de datos de prueba (con .txt): ")
            output_test_file = input("Archivo de salidas esperadas a los datos de prueba (con .txt): ")
            epochs = int(input("Número de épocas: "))

            # Cargar datos
            train_data, train_labels = load_data(train_file, output_train_file)
            test_data, test_labels = load_data(test_file, output_test_file)

            # Crear el MLP
            mlp = MLP(input_neurons, output_neurons, hidden_layers, neurons_per_hidden_layer)
            mlp.train(train_data, train_labels, epochs)

            predictions = []
            for inputs in test_data:
                prediction = mlp.predict(inputs)
                predictions.append(float(prediction[0][0]))  # Accede al valor dentro de la matriz

            print("Predicciones de prueba:", predictions)

            

            # Guardar el perceptrón entrenado
            save_option = input("¿Quieres guardar el perceptrón entrenado? (s/n): ")
            if save_option.lower() == 's':
                save_file = input("Introduce el nombre del archivo para guardar el perceptrón (ej. perceptron1.pkl): ")
                mlp.save(save_file)

        elif option == "2":
            filename = input("Introduce el nombre del archivo de perceptrón (ej. perceptron1.pkl): ")
            try:
                mlp = MLP.load(filename)
                print(f"Perceptrón cargado desde {filename}")

                # Opción para realizar predicciones o seguir entrenando
                action = input("¿Deseas hacer predicciones (1) o seguir entrenando (2)? ")
                if action == "1":
                    test_file = input("Introduce el archivo de prueba: ")
                    test_data, _ = load_data(test_file, None)
                    predictions = [mlp.predict(inputs) for inputs in test_data]
                    
                    print("Predicciones de prueba:")
                    for i, pred in enumerate(predictions):
                        valor_decimal = float(pred[0][0])  # extrae el número de array([[x]])
                        clase = 1 if valor_decimal >= 0.5 else 0
                        print(f"Entrada {i+1}: Valor = {valor_decimal:.4f}, Clase = {clase}")


                elif action == "2":
                    train_file = input("Introduce el archivo de entrenamiento: ")
                    output_train_file = input("Introduce el archivo de salidas esperadas: ")
                    train_data, train_labels = load_data(train_file, output_train_file)
                    epochs = int(input("Número de épocas: "))
                    mlp.train(train_data, train_labels, epochs)

                    # Guardar el perceptrón nuevamente
                    save_option = input("¿Quieres guardar el perceptrón entrenado? (s/n): ")
                    if save_option.lower() == 's':
                        save_file = input("Introduce el nombre del archivo para guardar el perceptrón (ej. perceptron1.pkl): ")
                        mlp.save(save_file)
            except FileNotFoundError:
                print(f"No se encontró el archivo {filename}. Por favor, verifica el nombre del archivo.")

        elif option == "3":
            print("Saliendo...")
            break
           

if __name__ == "__main__":
    main()

