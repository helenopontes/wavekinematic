import os
import pandas as pd
import pandas as to_csv
import pandas as DataFrame

print ("\tESCOLHA UMA DAS OPÇÕES ABAIXO:\n")
print ("\tEscolha '0' para ver os dados completos.\n")
print ("\tEscolha '1' se deseja ver os dados de um ANO específico.\n")
print ("\tEscolha '2' para ver estatísticas.\n")

df = pd.read_csv("lista.csv", delimiter=";", names=["CÓDIGO MUNICÍPIO","ANO","HOMICÍDIOS","IDADE MÉDIA","PROPORÇÃO NEGROS"])
maximo = df.max()
minimo = df.min()
descricao = df.describe()
media = df.mean()

ano = int (input("Informe um ANO: "))

imprimir_ano = df.loc[(df["ANO"] == ano)]
excel = pd.ExcelWriter('NOME_DO_ARQUIVO.xlsx', engine='xlsxwriter')
imprimir_ano.to_excel(excel, sheet_name='Dados de um ANO específico')
excel.save()

nova_tabela = df.pivot_table(index="ANO")
print(imprimir_ano)
print (nova_tabela)