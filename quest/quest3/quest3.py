import marimo

__generated_with = "0.8.11"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_validate
    from sklearn.preprocessing import StandardScaler
    from mlxtend.plotting import plot_decision_regions
    from sklearn.linear_model import LogisticRegression
    return (
        KNeighborsClassifier,
        LogisticRegression,
        StandardScaler,
        cross_validate,
        mo,
        np,
        pd,
        plot_decision_regions,
        plt,
        train_test_split,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Questionário 3

        Classificação através dos K vizinhos mais próximos.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Ex. 1

        Considere as bases Iris e Vehicle. Em um mesmo gráfico, mostre a acurácia em função de k para o metódo de K vizinhos.
        """
    )
    return


@app.cell
def __(pd):
    df_iris = pd.read_csv("data/iris.csv")
    df_vehicle = pd.read_csv("data/Vehicle.csv")
    return df_iris, df_vehicle


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Sempre importante visualizar as tabelas.""")
    return


@app.cell
def __(df_iris):
    df_iris.head()
    return


@app.cell
def __(df_vehicle):
    df_vehicle.head()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Construiremos as variárveis $X$ e $y$ como ```array``` do ```numpy```. A seguir, vem a normalização, para eliminar o efeito de escala.""")
    return


@app.cell
def __(StandardScaler, df_iris, df_vehicle):
    iris = df_iris.to_numpy()
    iris_y = iris[:, -1]
    iris_X = iris[:, 0:-1]

    vehicle = df_vehicle.to_numpy()
    vehicle_y = vehicle[:, -1]
    vehicle_X = vehicle[:, 0:-1]

    i_scaler = StandardScaler()
    i_scaler.fit(iris_X)
    iris_X = i_scaler.transform(iris_X)

    v_scaler = StandardScaler()
    v_scaler.fit(vehicle_X)
    vehicle_X = v_scaler.transform(vehicle_X)
    return (
        i_scaler,
        iris,
        iris_X,
        iris_y,
        v_scaler,
        vehicle,
        vehicle_X,
        vehicle_y,
    )


@app.cell
def __(iris_X, iris_y, train_test_split, vehicle_X, vehicle_y):
    p = 0.7

    iX_train, iX_test, iy_train, iy_test = train_test_split(
        iris_X, iris_y, train_size=p, random_state=42
    )

    vX_train, vX_test, vy_train, vy_test = train_test_split(
        vehicle_X, vehicle_y, train_size=p, random_state=42
    )
    return (
        iX_test,
        iX_train,
        iy_test,
        iy_train,
        p,
        vX_test,
        vX_train,
        vy_test,
        vy_train,
    )


@app.cell
def __(
    KNeighborsClassifier,
    cross_validate,
    iX_train,
    iy_train,
    plt,
    vX_train,
    vy_train,
):
    _k_values = []
    iris_scores, vehicle_scores = [], []

    nkf = 5

    for _k in range(1, 20):
        iris_model = KNeighborsClassifier(n_neighbors=_k)
        vehicle_model = KNeighborsClassifier(n_neighbors=_k)
        i_cv = cross_validate(iris_model, iX_train, iy_train, cv=nkf)
        v_cv = cross_validate(vehicle_model, vX_train, vy_train, cv=nkf)

        iris_scores.append(i_cv["test_score"].mean())
        vehicle_scores.append(v_cv["test_score"].mean())
        _k_values.append(_k)

    plt.plot(_k_values, iris_scores, "-o", label="Iris")
    plt.plot(_k_values, vehicle_scores, "-o", label="Vehicle")
    plt.title("Accuracy of KNN Classifier for two separete datasets")

    plt.ylabel("Accuracy", fontsize=15)
    plt.xlabel("k", fontsize=15)

    plt.xticks(_k_values)

    plt.legend()
    return (
        i_cv,
        iris_model,
        iris_scores,
        nkf,
        v_cv,
        vehicle_model,
        vehicle_scores,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Ex. 2 

        Considere os dados gerados com o código abaixo e obtenha as regiões de separação usando
        o método k-vizinhos para diferentes valores de k. Compare com as regiões obtidas usando o
        método regressão logística.
        """
    )
    return


@app.cell
def __(plt):
    # Código dado no exercício
    from sklearn import datasets

    plt.figure(figsize=(6, 4))
    n_samples = 1000

    data = datasets.make_moons(n_samples=n_samples, noise=0.9)

    X = data[0]
    y = data[1]

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", s=50, alpha=0.7)
    return X, data, datasets, n_samples, y


@app.cell
def __(KNeighborsClassifier, X, n_samples, plot_decision_regions, plt, y):
    _k_values = [1, 5, 10, 20, int(n_samples / 2)]
    for _k in _k_values:
        model = KNeighborsClassifier(n_neighbors=_k, metric="euclidean")
        model.fit(X, y)

        plot_decision_regions(X, y, clf=model, legend=2)
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.title("Decision Regions: k = " + str(_k))

        plt.show()
    return model,


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Ex. 3

        Classifique as bases Iris e Vehicle usando regressão logística e compare com o método k-vizinhos.
        """
    )
    return


@app.cell
def __(
    LogisticRegression,
    iX_test,
    iX_train,
    iy_test,
    iy_train,
    vX_test,
    vX_train,
    vy_test,
    vy_train,
):
    i_logistical_model = LogisticRegression(solver="lbfgs", max_iter=1000)
    v_logistical_model = LogisticRegression(solver="lbfgs", max_iter=1000)


    i_logistical_model.fit(iX_train, iy_train)
    v_logistical_model.fit(vX_train, vy_train)

    i_logistical_score = i_logistical_model.score(iX_test, iy_test)
    v_logistical_score = v_logistical_model.score(vX_test, vy_test)

    print("Iris Logistical Score: {:2.2%}".format(i_logistical_score))
    print("Vehicle Logistical Score: {:2.2%}".format(v_logistical_score))
    return (
        i_logistical_model,
        i_logistical_score,
        v_logistical_model,
        v_logistical_score,
    )


@app.cell(hide_code=True)
def __(
    i_logistical_score,
    iris_scores,
    mo,
    v_logistical_score,
    vehicle_scores,
):
    mo.md(
        f"""
        Vemos que, para essas duas bases de dados, a Regresão Logística alcançou resultados bem superiores ao método de K vizinhos.
        Para a base de dados da íris, essa diferença foi de {abs(max(iris_scores) - i_logistical_score):2.2%}. Já para os veículos, a diferença foi de surpreendentes {abs(max(vehicle_scores) - v_logistical_score):2.2%}. 

        Wow!

        """
    )
    return


if __name__ == "__main__":
    app.run()
