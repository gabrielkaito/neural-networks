from neuron.neuron import Neuronio
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

class Rede:
    def __init__(self, qtdEntradas, qtdCamadaOculta, qtdSaida, dataframe, n, epocas, erroMinimo, funcao):
        self.qtdEntradas = qtdEntradas
        self.qtdCamadaOculta = qtdCamadaOculta
        self.qtdSaida = qtdSaida
        self.funcao = funcao
        self.n = n
        self.epocas = epocas
        self.erroMinimo = erroMinimo
        self.dataframe = dataframe

        self.neuroniosOculta = [Neuronio(funcao) for _ in range(qtdCamadaOculta)]
        self.neuroniosSaida = [Neuronio(funcao) for _ in range(qtdSaida)]

        self.pesosEntrada = np.random.uniform(-1, 1, (qtdEntradas, qtdCamadaOculta))
        self.pesosOculta = np.random.uniform(-1, 1, (qtdCamadaOculta, qtdSaida))

        self.treinar()

    def forward(self, entrada):
        for i, neuronio in enumerate(self.neuroniosOculta):
            neuronio.net = np.dot(entrada, self.pesosEntrada[:, i])
            neuronio.saida = neuronio.activation(neuronio.net)
        saidas_oculta = np.array([n.saida for n in self.neuroniosOculta])

        for i, neuronio in enumerate(self.neuroniosSaida):
            neuronio.net = np.dot(saidas_oculta, self.pesosOculta[:, i])
            neuronio.saida = neuronio.activation(neuronio.net)
        saidas_saida = np.array([n.saida for n in self.neuroniosSaida])

        return saidas_oculta, saidas_saida

    def treinar(self):
        for epoca in range(self.epocas):
            erro_total = 0

            for _, linha in self.dataframe.iterrows():
                entrada = np.array(linha.iloc[:self.qtdEntradas])
                saida_esperada = np.array(linha.iloc[self.qtdEntradas:])

                saidas_oculta, saidas_rede = self.forward(entrada)

                erros = saida_esperada - saidas_rede
                erro_total += np.mean(erros ** 2)

                deltas_saida = []
                for i, neuronio in enumerate(self.neuroniosSaida):
                    erro = erros[i]
                    derivada = neuronio.activation_derivative()
                    delta = erro * derivada
                    deltas_saida.append(delta)
                    for j in range(self.qtdCamadaOculta):
                        self.pesosOculta[j][i] += self.n * delta * self.neuroniosOculta[j].saida

                for i, neuronio in enumerate(self.neuroniosOculta):
                    soma = sum(deltas_saida[k] * self.pesosOculta[i][k] for k in range(self.qtdSaida))
                    derivada = neuronio.activation_derivative()
                    delta = soma * derivada
                    for j in range(self.qtdEntradas):
                        self.pesosEntrada[j][i] += self.n * delta * entrada[j]

            erro_medio = erro_total / len(self.dataframe)
            print(f"Época {epoca + 1:02d} - Erro médio: {erro_medio:.6f}")

            if np.isnan(self.pesosEntrada).any() or np.isnan(self.pesosOculta).any() or \
               np.isinf(self.pesosEntrada).any() or np.isinf(self.pesosOculta).any():
                print("Pesos explodiram! Abortando treinamento...")
                break

            if erro_medio <= self.erroMinimo:
                print("Critério de parada atingido.")
                break

        print("Treinamento finalizado.\n")

    def testar(self, dataframe_teste):
        y_true = []
        y_pred = []

        for _, linha in dataframe_teste.iterrows():
            entrada = np.array(linha.iloc[:self.qtdEntradas])
            saida_esperada = np.array(linha.iloc[self.qtdEntradas:])

            _, saidas_rede = self.forward(entrada)

            y_true.append(np.argmax(saida_esperada))
            y_pred.append(np.argmax(saidas_rede))

        print("\nMatriz de Confusão:")
        print(confusion_matrix(y_true, y_pred))
        print("\nRelatório de Classificação:")
        print(classification_report(y_true, y_pred))
        return y_true, y_pred
