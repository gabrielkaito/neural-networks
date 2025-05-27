import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk
from utils.mlp import mlp
from utils.normalize_data import normalize_data
import math
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def upload_csv_treino():
    global treino_path
    file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file:
        treino_path = file
        csvPath.config(text=file, fg="white")


def upload_csv_teste():
    global teste_path
    file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file:
        teste_path = file
        testCsvPath.config(text=file, fg="white")


def executar_rede_neural():
    global y_true_result, y_pred_result
    if not treino_path or not teste_path:
        print("Selecione os dois arquivos CSV.")
        return

    # Carrega os dados
    treino_data = pd.read_csv(treino_path)
    teste_data = pd.read_csv(teste_path)

    df_treino, entradas, saidas, scaler = normalize_data(treino_data)
    df_teste, entradas_teste, saidas_teste, _ = normalize_data(teste_data, scaler)

    qtd_neuronios_oculta = neuroniosOculta.get().strip()

    if not qtd_neuronios_oculta:
        qtdNeuroniosOculta = math.floor((entradas + saidas) / 2)
        neuroniosOculta.delete(0, tk.END)  # limpa o campo
        neuroniosOculta.insert(0, str(qtdNeuroniosOculta))  # insere o valor calculado
    else:
        qtdNeuroniosOculta = int(qtd_neuronios_oculta)


    entradaLabel.config(text=str(entradas))
    saidaLabel.config(text=str(saidas))


    y_true_result, y_pred_result = mlp(df_treino, df_teste, int(iteracoes.get()), qtdNeuroniosOculta, entradas, saidas, float(erro.get()), float(n.get()), funcao.get())
    btnConfusao.pack(pady=10)

    # Exibe os dados de teste na tabela
    for widget in frameTable.winfo_children():
        widget.destroy()

    borderedFrame = tk.Frame(frameTable, bd=1, relief="solid")
    borderedFrame.pack(expand=True, fill="both")

    treeScrollY = tk.Scrollbar(borderedFrame)
    treeScrollY.pack(side="right", fill="y")

    treeScrollX = tk.Scrollbar(borderedFrame, orient="horizontal")
    treeScrollX.pack(side="bottom", fill="x")

    tree = ttk.Treeview(
        borderedFrame,
        columns=list(teste_data.columns),
        show="headings",
        yscrollcommand=treeScrollY.set,
        xscrollcommand=treeScrollX.set
    )
    tree.pack(expand=True, fill="both")

    treeScrollY.config(command=tree.yview)
    treeScrollX.config(command=tree.xview)

    for col in teste_data.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=100)

    for _, row in teste_data.iterrows():
        tree.insert("", "end", values=list(row))



treino_path = None
teste_path = None


# Janela principal
window = tk.Tk()
window.title("Rede Neural - FIPP")
window.geometry("800x600")
window.configure(bg="white")

# Cabeçalho
frameHeader = tk.Frame(window, width=800, height=60, bg="white")
frameHeader.pack(pady=10)
tk.Label(frameHeader, text="INTELIGÊNCIA ARTIFICIAL", font=("Arial", 14, "bold"), bg="white").pack()
tk.Label(frameHeader, text="REDES NEURAIS", font=("Arial", 14, "bold"), fg="green", bg="white").pack()

# Inputs gerais
frameInputs = tk.Frame(window, width=800, height=180)
frameInputs.pack(padx=10, pady=5, fill="x")
frameInputs.grid_columnconfigure((0, 1, 2), weight=1, uniform="col")

# ----- Neurônios -----
frameNeuron = tk.Frame(frameInputs, padx=10, pady=5, bd=1, relief="solid")
frameNeuron.grid(row=0, column=0, sticky="nsew")
tk.Label(frameNeuron, text="Configurar número de neurônios:", font=("Arial", 9, "bold")).grid(column=0, row=0, columnspan=2, sticky="w", pady=(0,10))

tk.Label(frameNeuron, text="Camada de Entrada:").grid(row=1, column=0, sticky="e", pady=2)
entradaLabel = tk.Label(frameNeuron, width=7, relief="sunken", bg="white", anchor="w")
entradaLabel.grid(column=1, row=1, sticky="w")

tk.Label(frameNeuron, text="Camada de Saída:").grid(row=2, column=0, sticky="e", pady=2)
saidaLabel = tk.Label(frameNeuron, width=7, relief="sunken", bg="white", anchor="w")
saidaLabel.grid(column=1, row=2, sticky="w")


tk.Label(frameNeuron, text="Camada Oculta:").grid(row=3, column=0, sticky="e", pady=2)
neuroniosOculta = tk.Entry(frameNeuron, width=7)
neuroniosOculta.grid(column=1, row=3, sticky="w")

# ----- Variáveis -----
frameVariables = tk.Frame(frameInputs, padx=10, pady=5, bd=1, relief="solid")
frameVariables.grid(row=0, column=1, sticky="nsew")
tk.Label(frameVariables, text="Variáveis:", font=("Arial", 9, "bold")).grid(column=0, row=0, columnspan=2, sticky="w", pady=(0,10))

tk.Label(frameVariables, text="Valor do Erro:").grid(column=0, row=1, sticky="e", pady=2)
erro = tk.Entry(frameVariables, width=7)
erro.grid(column=1, row=1, sticky="w")

tk.Label(frameVariables, text="Número de Iterações:").grid(column=0, row=2, sticky="e", pady=2)
iteracoes = tk.Entry(frameVariables, width=7)
iteracoes.grid(column=1, row=2, sticky="w")

tk.Label(frameVariables, text="N:").grid(column=0, row=3, sticky="e", pady=2)
n = tk.Entry(frameVariables, width=7)
n.grid(column=1, row=3, sticky="w")

# ----- Funções -----
frameFunctions = tk.Frame(frameInputs, padx=10, pady=5, bd=1, relief="solid")
frameFunctions.grid(row=0, column=2, sticky="nsew")
tk.Label(frameFunctions, text="Função de transferência:", font=("Arial", 9, "bold")).grid(column=0, row=0, sticky="w", pady=(0,10))

funcao = tk.StringVar(value="linear")  # valor inicial definido

tk.Radiobutton(frameFunctions, text="Linear", variable=funcao, value="linear").grid(column=0, row=1, sticky="w", pady=2)
tk.Radiobutton(frameFunctions, text="Logística", variable=funcao, value="logistica").grid(column=0, row=2, sticky="w", pady=2)
tk.Radiobutton(frameFunctions, text="Hiperbólica", variable=funcao, value="hiperbolica").grid(column=0, row=3, sticky="w", pady=2)


# ----- CSV -----
frameCsv = tk.Frame(window, width=800, height=60, bg="green")
frameCsv.pack(pady=10, fill="x")
frameCsv.grid_columnconfigure(0, weight=1)
frameCsv.grid_columnconfigure(1, weight=4)
frameCsv.grid_columnconfigure(2, weight=1)
frameCsv.grid_columnconfigure(3, weight=4)

# Botão e label para treino
buttonLoadCsv = tk.Button(frameCsv, text="Carregar Arquivo Treino", command=upload_csv_treino)
buttonLoadCsv.grid(row=0, column=0, padx=10, pady=10, sticky="w")

csvPath = tk.Label(frameCsv, text="Nenhum arquivo selecionado", bg="green", fg="white", anchor="w")
csvPath.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

# 🔽 Novo botão e label para teste
buttonLoadTestCsv = tk.Button(frameCsv, text="Carregar Arquivo Teste", command=upload_csv_teste)
buttonLoadTestCsv.grid(row=0, column=2, padx=10, pady=10, sticky="w")

buttonExecutar = tk.Button(window, text="Executar Rede Neural", bg="blue", fg="white", command=executar_rede_neural)
buttonExecutar.pack(pady=10)

def abrir_matriz_confusao():


    cm = confusion_matrix(y_true_result, y_pred_result)
    nova_janela = tk.Toplevel(window)
    nova_janela.title("Matriz de Confusão")
    nova_janela.geometry("400x400")

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de Confusão")

    canvas = FigureCanvasTkAgg(fig, master=nova_janela)
    canvas.draw()
    canvas.get_tk_widget().pack(expand=True, fill='both')

btnConfusao = tk.Button(window, text="Ver Matriz de Confusão", bg="orange", command=abrir_matriz_confusao)
btnConfusao.pack_forget()  # começa invisível


testCsvPath = tk.Label(frameCsv, text="Nenhum arquivo selecionado", bg="green", fg="white", anchor="w")
testCsvPath.grid(row=0, column=3, padx=10, pady=10, sticky="ew")

# ----- TABELA CSV -----
frameTable = tk.Frame(window)
frameTable.pack(expand=True, fill="both", padx=10, pady=5)

window.mainloop()
