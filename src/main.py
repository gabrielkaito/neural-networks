import pandas as pd
import tkinter as tk
from tkinter import filedialog, ttk
from utils.normalize_data import normalize_data

def upload_csv():
    file = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])

    if file:
        csvPath.config(text=file, fg="white")
        data = pd.read_csv(file)

        # Normaliza os dados
        data = normalize_data(data)

        # Limpar tabela anterior
        for widget in frameTable.winfo_children():
            widget.destroy()

        # Frame com borda para a tabela
        borderedFrame = tk.Frame(frameTable, bd=1, relief="solid")
        borderedFrame.pack(expand=True, fill="both")

        # Scrollbars
        treeScrollY = tk.Scrollbar(borderedFrame)
        treeScrollY.pack(side="right", fill="y")

        treeScrollX = tk.Scrollbar(borderedFrame, orient="horizontal")
        treeScrollX.pack(side="bottom", fill="x")

        # Criar Treeview
        tree = ttk.Treeview(
            borderedFrame,
            columns=list(data.columns),
            show="headings",
            yscrollcommand=treeScrollY.set,
            xscrollcommand=treeScrollX.set
        )
        tree.pack(expand=True, fill="both")

        # Configurar Scroll
        treeScrollY.config(command=tree.yview)
        treeScrollX.config(command=tree.xview)

        # Cabeçalhos
        for col in data.columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center", width=100)

        # Dados
        for _, row in data.iterrows():
            tree.insert("", "end", values=list(row))
        print(data)


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
tk.Entry(frameNeuron, width=7).grid(column=1, row=1, sticky="w")

tk.Label(frameNeuron, text="Camada de Saída:").grid(row=2, column=0, sticky="e", pady=2)
tk.Entry(frameNeuron, width=7).grid(column=1, row=2, sticky="w")

tk.Label(frameNeuron, text="Camada Oculta:").grid(row=3, column=0, sticky="e", pady=2)
tk.Entry(frameNeuron, width=7).grid(column=1, row=3, sticky="w")

# ----- Variáveis -----
frameVariables = tk.Frame(frameInputs, padx=10, pady=5, bd=1, relief="solid")
frameVariables.grid(row=0, column=1, sticky="nsew")
tk.Label(frameVariables, text="Variáveis:", font=("Arial", 9, "bold")).grid(column=0, row=0, columnspan=2, sticky="w", pady=(0,10))

tk.Label(frameVariables, text="Valor do Erro:").grid(column=0, row=1, sticky="e", pady=2)
tk.Entry(frameVariables, width=7).grid(column=1, row=1, sticky="w")

tk.Label(frameVariables, text="Número de Iterações:").grid(column=0, row=2, sticky="e", pady=2)
tk.Entry(frameVariables, width=7).grid(column=1, row=2, sticky="w")

tk.Label(frameVariables, text="N:").grid(column=0, row=3, sticky="e", pady=2)
tk.Entry(frameVariables, width=7).grid(column=1, row=3, sticky="w")

# ----- Funções -----
frameFunctions = tk.Frame(frameInputs, padx=10, pady=5, bd=1, relief="solid")
frameFunctions.grid(row=0, column=2, sticky="nsew")
tk.Label(frameFunctions, text="Função de transferência:", font=("Arial", 9, "bold")).grid(column=0, row=0, sticky="w", pady=(0,10))

function = tk.StringVar(value="lienar")  # valor inicial definido

tk.Radiobutton(frameFunctions, text="Linear", variable=function, value="lienar").grid(column=0, row=1, sticky="w", pady=2)
tk.Radiobutton(frameFunctions, text="Logística", variable=function, value="logistic").grid(column=0, row=2, sticky="w", pady=2)
tk.Radiobutton(frameFunctions, text="Hiperbólica", variable=function, value="hyperbolic").grid(column=0, row=3, sticky="w", pady=2)


# ----- CSV -----
frameCsv = tk.Frame(window, width=800, height=60, bg="green")
frameCsv.pack(pady=10, fill="x")
frameCsv.grid_columnconfigure(0, weight=1)
frameCsv.grid_columnconfigure(1, weight=4)

buttonLoadCsv = tk.Button(frameCsv, text="Carregar Arquivo", command=upload_csv)
buttonLoadCsv.grid(row=0, column=0, padx=10, pady=10, sticky="w")

csvPath = tk.Label(frameCsv, text="Nenhum arquivo selecionado", bg="green", fg="white", anchor="w")
csvPath.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

# ----- TABELA CSV -----
frameTable = tk.Frame(window)
frameTable.pack(expand=True, fill="both", padx=10, pady=5)

window.mainloop()
