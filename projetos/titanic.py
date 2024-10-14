import marimo

__generated_with = "0.9.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""# Projeto Titanic""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Preparação de terreno:

        Nesta seção, são importados os packages que utilizaremos nos projetos.

        Também criamos os `TypeAlias` que nos serão úteis e produzimos a paleta que será usada nos gráficos.
        """
    )
    return


@app.cell(hide_code=True)
def __():
    import marimo as mo
    return (mo,)


@app.cell
def __():
    import random
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    from aquarel import load_theme
    import seaborn as sns
    import typing
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import KNNImputer
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans

    theme = load_theme("boxy_light")
    theme.apply()
    # ... plotting code here
    theme.apply_transforms()
    return (
        KMeans,
        KNNImputer,
        PCA,
        StandardScaler,
        load_theme,
        matplotlib,
        np,
        pd,
        plt,
        random,
        sns,
        theme,
        typing,
    )


@app.cell(hide_code=True)
def __(matplotlib, pd):
    DataFrame = pd.core.frame.DataFrame
    Figure = matplotlib.figure.Figure
    Axes: matplotlib.axes.Axes
    return Axes, DataFrame, Figure


@app.cell(hide_code=True)
def __():
    pallete_survived = {
        0: "#e97777",
        1: "#478c8c",
    }
    return (pallete_survived,)


@app.cell
def __(mo):
    mo.md(r"""## Importação e transformação dos dados""")
    return


@app.cell
def __(DataFrame, pd, random):
    # Define a semente aleatória (Para termos resultados reproduzíveis)
    random.seed(42)

    # Importa os DataFrames train e test
    train: DataFrame = pd.read_csv("data/titanic/train.csv", header=(0))
    test: DataFrame = pd.read_csv("data/titanic/test.csv", header=(0))

    print("Número de linhas e colunas no conjunto de treinamento:", train.shape)
    print("Número de linhas e colunas no conjunto de teste:", test.shape)
    attributes = list(train.columns)
    train.head(10)

    train_original = train

    # Limpeza de colunas que não nos interessam
    train.drop(
        columns=[
            "Cabin",
            "Name",
            "Ticket",
            "PassengerId",
        ],
        inplace=True,
    )

    test.drop(
        columns=[
            "Cabin",
            "Name",
            "Ticket",
            "PassengerId",
        ],
        inplace=True,
    )


    # Cria colunas que podem ser úteis
    for df in [train, test]:
        df["IsKid"] = df["Age"] < 18

    train = pd.get_dummies(train, drop_first=True)
    test = pd.get_dummies(test, drop_first=True)
    return attributes, df, test, train, train_original


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Podemos observar os primeiros dados do treino.""")
    return


@app.cell
def __(train):
    train.head(20)
    return


@app.cell
def __(mo):
    mo.md(
        r"""
        Vamos converter os dados para matrizes de `numpy` e padronizá-los para facilitar a manipulação.

        Ademais, usamos o `KNNImputer` para preenches os valores que estariam vazios.
        """
    )
    return


@app.cell
def __(KNNImputer, StandardScaler, np, test, train):
    data_train = train.to_numpy()
    nrow, ncol = data_train.shape
    y_train = data_train[:, 0]
    y_train = y_train.astype(int)
    X_train = data_train[:, 1:ncol]

    data_test = test.to_numpy()
    X_test = data_test

    scaler = StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    imputer = KNNImputer(n_neighbors=4)

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    print("Dados transformados:")
    print("Media: ", np.mean(X_train, axis=0))
    print("Desvio Padrao:", np.std(X_train, axis=0))

    print("Dados transformados:")
    print("Media: ", np.mean(X_test, axis=0))
    print("Desvio Padrao:", np.std(X_test, axis=0))
    return (
        X_test,
        X_train,
        data_test,
        data_train,
        imputer,
        ncol,
        nrow,
        scaler,
        y_train,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Análise exploratória dos dados""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Balanceamento das classes""")
    return


@app.cell
def __(np, pallete_survived, plt, y_train):
    _classes = y_train
    cl = np.unique(_classes)
    ncl = np.zeros(len(cl))
    for i in np.arange(0, len(cl)):
        a = _classes == cl[i]
        ncl[i] = len(_classes[a])

    numbers = np.arange(0, len(cl))
    # Uma pequena dica visual do significado das classes
    for _i in range(0,2):
        plt.bar(numbers[_i], ncl[_i], alpha=0.75, color=pallete_survived[_i])

    plt.xticks(numbers, cl)
    plt.title("Número de elementos em cada classe")
    plt.show()
    return a, cl, i, ncl, numbers


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Vemos então que as classes estão mais ou menos balanceados. Podemos ter que prestar atenção nisso.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Correlação entre as varariáveis""")
    return


@app.cell
def __(plt, train):
    corr = train.corr()
    # Plot Correlation Matrix using Matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(corr, cmap="RdBu", interpolation="none", aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation="vertical")
    plt.yticks(range(len(corr)), corr.columns)
    plt.suptitle("Correlação entre variáveis", fontsize=18, fontweight="bold")
    plt.grid(False)
    plt.show()
    return (corr,)


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Algumas variáveis são bem correlacionadas, mas, como ser criança ou não costuma a ser importante em procedimentos de resgate, e é interessante separar classe e tarifa, vamos mantê-las.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Relação entre gênero e sobrevivência.""")
    return


@app.cell
def __(pallete_survived, plt, sns, train_original):
    sns.violinplot(
        x="Sex", y="Age", hue="Survived", data=train_original, split=True, palette=pallete_survived
    )
    plt.title("Gráfico de violino entre gênero, idade e sobrevivência", fontsize=18, fontweight="bold", pad=20)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Vemos que, para homens, existe uma relação muito forte entre idade e sobrevivência é muito forte. Já para mulheres, isso não é verdade.""")
    return


@app.cell
def __(plt, sns, train_original):
    group = train_original.groupby(["Pclass", "Survived"])
    pclass_survived = group.size().unstack().astype(float)
    for _i in range(1, 4):
        pclass_survived.loc[_i] = (
            pclass_survived.loc[_i] / pclass_survived.loc[_i].sum()
        )
    pclass_survived
    plt.title(
        "Porcentagem de sobreviventes por classe",
        fontweight="bold",
        fontsize=18,
        pad=30,
    )
    sns.heatmap(pclass_survived, annot=True, fmt=".0%", cmap="PiYG")
    return group, pclass_survived


@app.cell
def __(pd, sns, train_original):
    # Divide Fare into 4 bins
    train_original['Fare_Range'] = pd.qcut(train_original['Fare'], 4)

    # Barplot - Shows approximate values based on the height of bars.
    sns.barplot(x ='Fare_Range', y ='Survived', hue="Fare_Range", data = train_original)
    return


@app.cell
def __(pallete_survived, sns, train_original):
    # Countplot
    sns.catplot(x ='Embarked', hue ='Survived', palette=pallete_survived,
    kind ='count', col ='Pclass', data = train_original)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Separação das classes""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Para começar, podemos criar uma ideia de como a divisão de classes se parecerá através da aplicação do método SVD de decomposição da classe em seus autovalores principais.""")
    return


@app.cell
def __(PCA, X_train, np, pallete_survived, plt, y_train):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X_train)

    _classes = np.unique(y_train)

    _aux = 0
    plt.figure(figsize=(8,5))
    for c in _classes:
        if c == 1:
            lb = 'Sobreviveu'
        else:
            lb = 'Morreu'
        nodes = np.where(y_train == c)
        plt.scatter(pca_result[nodes,0], pca_result[nodes,1], s=50, 
                    label = lb, color=pallete_survived[_aux])
        _aux = _aux + 1
    plt.legend()
    plt.xlabel("First component", fontsize=20)
    plt.ylabel("Second component", fontsize=20)
    plt.xticks(color='k', size=20)
    plt.yticks(color='k', size=20)
    plt.show()
    return c, lb, nodes, pca, pca_result


@app.cell
def __(mo):
    mo.md(r"""Vemos aqui que o nosso dataset tem uma grande correlação inerente, e o agrupamento em clusters não será muito útil para explicar a (não) sobrevivência de certos grupos.""")
    return


@app.cell
def __(KMeans, pallete_survived, pca_result, plt, y_train):
    _k = 2 # numero de clusters a serem identificados
    # define o método com k clusters
    kmeans = KMeans(n_clusters=_k) 
    # realiza o ajuste considerando os dados X
    kmeans.fit(pca_result)
    # faz a predição, identificando os clusters
    y_km = kmeans.fit_predict(pca_result)

    plt.figure(figsize=(15,5))
    # mostra o primeiro gráfico
    plt.subplot(1, 2, 2)
    plt.scatter(pca_result[:,0], pca_result[:,1], c=y_km, cmap='viridis', s=50)
    plt.title('K-means')
    centers = kmeans.cluster_centers_
    # mostra os centróides obtidos
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, alpha=0.9); 

    # mostra o segundo gráfico
    plt.subplot(1,2, 1)
    plt.scatter(pca_result[:,0], pca_result[:,1], c=y_train, pallete=pallete_survived, s=50)
    plt.title('Dados originais')
    plt.show()
    return centers, kmeans, y_km


if __name__ == "__main__":
    app.run()
