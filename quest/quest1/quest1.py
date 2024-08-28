import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, pd, plt


@app.cell
def __(mo):
    mo.md(
        """
        # Questionário 1

        Primeiro questionário de Introdução a Ciência de Dados (SME0828)
        """
    )
    return


@app.cell
def __(pd):
    iris_with_errors = pd.read_csv('data/iris-with-errors.csv')
    iris_with_errors
    return iris_with_errors,


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## Exercício 1

        Achar o formato da coluna após limpar duplicadas e dados inválidos
        """
    )
    return


@app.cell
def __(iris_with_errors):
    ex1 = iris_with_errors.copy(deep=True)
    return ex1,


@app.cell
def __(ex1):
    ex1
    return


@app.cell
def __(ex1, np):
    # Limpeza dos dados. Por razões que me são misteriosas, a ordem importa

    ex1.drop_duplicates(inplace=True)

    ex1.replace(to_replace="?", value = np.nan, inplace=True)
    ex1.dropna(inplace=True)

    ex1.drop(columns=["petal_width", 'species'], inplace=True)
    return


@app.cell
def __(ex1):
    ex1
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Exercício 2

        Encontrar a média de cada atributo onde forem encontrados os valores ```?``` e ```nan```
        """
    )
    return


@app.cell
def __(iris_with_errors):
    # Limpeza dos dados

    ex2 = iris_with_errors.copy(deep=True)
    ex2
    return ex2,


@app.cell
def __(ex2, np):
    ex2.replace("?", np.nan, inplace=True)
    ex2.drop(columns=['species'], inplace=True)
    return


@app.cell
def __(ex2, np):
    # Criando a array equivalente e achando as médias 

    ex2_array = np.array(ex2[ex2.any(axis=1)], dtype=float) # Precisamos converter pra float
    ex3_array = np.array(ex2_array, copy=True)
    averages = np.nanmean(ex2_array, axis=0)
    return averages, ex2_array, ex3_array


@app.cell
def __(averages, ex2_array, np):
    # Substituindo as médias na array.
    # Literalmente idêntico ao do professor.

    for i in np.arange(0, ex2_array.shape[0]):
        for j in np.arange(0, ex2_array.shape[1]):
            if np.isnan(ex2_array[i,j]) == True:
                ex2_array[i,j] = averages[j]
    return i, j


@app.cell
def __(ex2_array, np):
    medians = np.median(ex2_array, axis=0)
    medians
    return medians,


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Exercício 3

        Montar box plot de todas variáveis e observar outliers
        """
    )
    return


@app.cell
def __(iris_with_errors):
    ex3 = iris_with_errors.copy(deep = True)
    return ex3,


@app.cell
def __(ex2_array, ex3):
    ex3["sepal_length"] = ex2_array[:,0]
    ex3["sepal_width"] = ex2_array[:,1]
    ex3["petal_length"] = ex2_array[:,2]
    ex3["petal_width"] = ex2_array[:,3]
    return


@app.cell
def __(ex3):
    ex3
    return


@app.cell
def __(ex3, plt):
    plt.figure(figsize=(1,1))
    fig, axs = plt.subplots(2, 2)

    axs[0,0].boxplot(ex3["sepal_length"])
    axs[0,0].set_title("Sepal length")

    axs[1,0].boxplot(ex3["sepal_width"])
    axs[1,0].set_title("Sepal width")

    axs[0,1].boxplot(ex3["petal_length"])
    axs[0,1].set_title("Petal length")

    axs[1,1].boxplot(ex3["petal_width"])
    axs[1,1].set_title("Petal width")


    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                        hspace=0.4, wspace=0.3)

    plt.show()
    return axs, fig


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Exercício 4

        Calcular a correlação das variáveis de Advertising.csv
        """
    )
    return


@app.cell
def __(pd):
    advertising = pd.read_csv("data/Advertising.csv")
    #advertising.drop(columns=[0])
    advertising.drop(columns=["Unnamed: 0"], inplace=True)
    return advertising,


@app.cell
def __(advertising):
    advertising
    return


@app.cell
def __(advertising):
    corr = advertising.corr()
    return corr,


@app.cell
def __(corr, plt):
    plt.figure(figsize=(5,4))
    # imshow é usado para mostrar imagens
    plt.imshow(corr, cmap='Blues', interpolation='none', aspect='auto')
    # mostra a barra lateral de cores
    plt.colorbar()
    # inclui o nome das variáveis
    plt.xticks(range(len(corr)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr)), corr.columns);
    plt.suptitle('Correlação entre variáveis', fontsize=15, fontweight='bold')
    plt.grid(False)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
