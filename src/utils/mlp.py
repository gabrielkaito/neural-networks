from network.network import Rede


def mlp(df_treino, df_teste, numeroIteracoes, qtd_oculta, qtd_entrada, qtd_saida, erroMinimo, n, funcao):

    print(df_treino)
    print(df_teste)

    rede = Rede(qtd_entrada, qtd_oculta, qtd_saida, df_treino, n, numeroIteracoes, erroMinimo, funcao)
    y_true, y_pred = rede.testar(df_teste)
    return y_true, y_pred
