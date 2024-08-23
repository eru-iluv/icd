import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        # Questionário 1

        Primeiro questionário de Introdução a Ciência de Dados (SME0828)
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    return mo, np, pd


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


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Exercício 2

        Encontrar a média de cada atributo onde forem encontrados os valores ```?``` e ```nan```
        """
    )
    return


@app.cell
def __(iris_with_errors, np):
    # Limpeza dos dados

    ex2 = iris_with_errors.copy(deep=True)
    ex2.replace("?", np.nan, inplace=True)
    ex2.drop(columns=['species'], inplace=True)
    return ex2,


@app.cell
def __(ex2, np):
    # Criando a array equivalente e achando as médias 

    ex2_array = np.array(ex2[ex2.isna().any(axis=1)], dtype=float) # Precisamos converter pra float
    averages = np.nanmean(ex2_array, axis=0)
    return averages, ex2_array


@app.cell
def __(ex2_array):
    ex2_array
    return


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


if __name__ == "__main__":
    app.run()
