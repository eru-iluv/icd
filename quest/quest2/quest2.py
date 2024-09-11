import marimo

__generated_with = "0.8.3"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn
    return mo, np, pd, plt, seaborn


@app.cell
def __(mo):
    mo.md(
        r"""
        # Questionário 2

        Análise e exploração dos dados de expectativa de vida em diversos países. Retirado do:

        [github de Aishwarya Ramakrishnan](https://gist.github.com/aishwarya8615/89d9f36fc014dea62487f7347864d16a)
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        """
        ## Exercício 1

        Construir um gráfico de setores para variável status. Qual a porcentagem de países desenvolvidos?
        """
    )
    return


@app.cell
def __(pd):
    url = "https://gist.github.com/aishwarya8615/89d9f36fc014dea62487f7347864d16a"

    life_expectancy = pd.read_html(url)[0]
    life_expectancy.drop(columns=["Unnamed: 0"], inplace=True)
    return life_expectancy, url


@app.cell
def __(life_expectancy):
    life_expectancy
    return


@app.cell
def __(life_expectancy, np):
    labels, counts = np.unique(life_expectancy["Status"], return_counts=True)
    return counts, labels


@app.cell
def __(counts, labels, plt):
    explode = (0, 0.04)
    fig1, ax1 = plt.subplots(figsize=(10,6))
    ax1.pie(counts, explode=explode, labels=labels, autopct="%1.1f%%")
    ax1.axis('equal')
    plt.show()
    return ax1, explode, fig1


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Exercício 2

        Considerando a base anterior, construa um histograma para variável life expectancy
        """
    )
    return


@app.cell
def __(life_expectancy, plt):
    fig = plt.figure(figsize=(6,4))

    num_bins = 10 
    n, bins, patches =  plt.hist(life_expectancy["Life_expectancy"], num_bins,
        facecolor="blue", alpha=0.5, density=True, edgecolor='black')
    plt.xlabel("Expectativa de vida", fontsize=15)
    plt.ylabel("Frequência")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()
    return bins, fig, n, num_bins, patches


@app.cell
def __(life_expectancy):
    var_life_expectancy = life_expectancy['Life_expectancy'].var()
    mean_life_expectancy = life_expectancy['Life_expectancy'].mean();
    return mean_life_expectancy, var_life_expectancy


@app.cell(hide_code=True)
def __(mean_life_expectancy, mo, var_life_expectancy):
    mo.md("""A expectativa de vida nos dados apresentou um valor médio de {media:.1f} e uma variância de {var:.1f}""".format(media = mean_life_expectancy, var =  var_life_expectancy))
    return


@app.cell
def __(life_expectancy):
    country_mask = (life_expectancy["Country"] == 'Angola') | (life_expectancy["Country"] == 'Finland') | (life_expectancy["Country"] == 'Ireland') | (life_expectancy["Country"] == 'Netherlands') | (life_expectancy["Country"] == 'Zambia')
    return country_mask,


@app.cell
def __(country_mask, life_expectancy):
    df_selected_contries = life_expectancy[country_mask]
    return df_selected_contries,


@app.cell
def __(df_selected_contries):
    df_selected_contries
    return


@app.cell
def __(df_selected_contries, np, plt):
    countries = np.unique(df_selected_contries["Country"])

    fig2, axs2 = plt.subplots(figsize=(7,7))

    plt.title("Evolution of life expectancy in 5 countries.", fontsize=20)

    for country in countries:
        country_data = df_selected_contries[df_selected_contries['Country'] == country]
        plt.plot(country_data['Year'], country_data['Life_expectancy'], label=country)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlabel("Year", fontsize=15)
    plt.ylabel("Life expectancy", fontsize=15)
    plt.legend()
    plt.show()
    return axs2, countries, country, country_data, fig2


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Vemos que os países com maior e menor expectativa de vida são Países Baixos e Angola, respectivamente.""")
    return


@app.cell
def __(mo):
    mo.md(
        """
        ## Exercício 4

        Fazer um boxplot para variável Schooling e ver qual apresenta a maior mediana para escolaridade.
        """
    )
    return


@app.cell
def __(df_selected_contries, plt, seaborn):
    fig3, axs3 = plt.subplots(figsize=(7,7))

    plt.title("Evolution of life expectancy in 5 countries.", fontsize=20)

    seaborn.boxplot(x='Country',y='Schooling', data=df_selected_contries )

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlabel("Country", fontsize=15)
    plt.ylabel("Schooling", fontsize=15)
    plt.show()
    return axs3, fig3


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Notamos que a Irlanda apresenta a maior mediana para expectativa de vida.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Exercício 5

        Construir a matriz de correlação e ver a menos relacionada.
        """
    )
    return


@app.cell
def __():
    import requests
    from bs4 import BeautifulSoup
    return BeautifulSoup, requests


@app.cell
def __(BeautifulSoup, np, pd, requests):
    url_ex5 = "https://www.worldometers.info/world-population/population-by-country/"
    # Send a GET request to the URL
    response = requests.get(url_ex5)
    # Parse the HTML content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')
    # Find the table containing the population data
    table = soup.find('table', {'id': 'example2'})
    # Convert the table to a Pandas DataFrame
    df = pd.read_html(str(table))[0]

    df.drop(columns=["Country (or dependency)"], inplace=True)
    df.replace("N.A.", np.nan, inplace=True)
    df.dropna(inplace=True)
    df['Urban  Pop %'] = df['Urban  Pop %'].str.rstrip('%').astype('float') / 100.0
    df['Yearly  Change'] = df['Yearly  Change'].str.rstrip('%').astype('float') / 100.0
    df['World  Share'] = df['World  Share'].str.rstrip('%').astype('float') / 100.0

    # Print the DataFrame
    df.head()
    return df, response, soup, table, url_ex5


@app.cell(hide_code=True)
def __(df, plt):
    corr = df.corr()
    #Plot Correlation Matrix using Matplotlib
    plt.figure(figsize=(7, 5))
    plt.imshow(corr, cmap='Blues', interpolation='none', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr)), corr.columns);
    plt.suptitle('Correlation between variables', fontsize=15, fontweight='bold')
    plt.grid(False)
    plt.show()
    return corr,


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""As váriaveis menos relacionadas são a taxa de fertilidade com a idade média""")
    return


if __name__ == "__main__":
    app.run()
