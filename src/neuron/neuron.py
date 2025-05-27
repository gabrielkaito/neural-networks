import numpy as np

class Neuronio:
    def __init__(self, activation_function='linear'):
        self.net = 0
        self.saida = 0
        self.erro = 0
        self.activation_function = activation_function

    def activation(self, x):
        # Funções de ativação
        if self.activation_function == 'linear':
            return x / 10
        elif self.activation_function == 'logistica':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'hiperbolica':
            return np.tanh(x)
        else:
            raise ValueError("Função de ativação inválida")

    def activation_derivative(self):
        # Derivadas das funções de ativação usando self.saida
        if self.activation_function == 'linear':
            return 1 / 10
        elif self.activation_function == 'logistica':
            return self.saida * (1 - self.saida)
        elif self.activation_function == 'hiperbolica':
            return 1 - self.saida ** 2
        else:
            raise ValueError("Função de ativação inválida")
