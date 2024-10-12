import marimo

__generated_with = "0.8.15"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def __():
    import marimo as mo
    from sklearn import linear_model as lm
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score
    from aquarel import load_theme
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.model_selection import GridSearchCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_validate
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import KFold

    theme = load_theme("boxy_light")
    theme.apply()
    # ... plotting code here
    theme.apply_transforms()
    return (
        GridSearchCV,
        KFold,
        KNeighborsClassifier,
        LabelEncoder,
        LogisticRegression,
        PolynomialFeatures,
        StandardScaler,
        accuracy_score,
        cross_validate,
        lm,
        load_theme,
        mo,
        np,
        pd,
        plt,
        r2_score,
        sns,
        theme,
        train_test_split,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Questionário 4

        Teste de modelo e sobreajuste.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Ex 1.

        Considere a base BostonHousing. Compare o coeficiente $R2$ obtido através de
        regressão linear múltipla, Lasso e ridge regression. Para os métodos Lasso e ridge regression, faça um gráfico de $R2 \times \alpha$ conforme feito no notebook da aula.
        """
    )
    return


@app.cell
def __(pd, train_test_split):
    df = pd.read_csv("data/BostonHousing.csv")
    _data = df.to_numpy()
    Xb = _data[:, :-1]
    yb = _data[:, -1]
    p = 0.3
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        Xb, yb, test_size=p, random_state=42
    )
    return Xb, Xb_test, Xb_train, df, p, yb, yb_test, yb_train


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Regressão Linear Múltipla""")
    return


@app.cell
def __(Xb_test, Xb_train, lm, yb_train):
    _model = lm.LinearRegression()
    _model.fit(Xb_train, yb_train)

    y_reg_pred = _model.predict(Xb_test)
    return y_reg_pred,


@app.cell
def __(np, plt, y_reg_pred, yb_test):
    _fig = plt.figure()
    _l = plt.plot(y_reg_pred, yb_test, "bo")
    plt.setp(_l, markersize=10)
    plt.setp(_l, markerfacecolor="C0")

    plt.ylabel("y", fontsize=15)
    plt.xlabel("Prediction", fontsize=15)

    # mostra os valores preditos e originais
    _xl = np.arange(
        min(yb_test), 1.2 * max(yb_test), (max(yb_test) - min(yb_test)) / 10
    )
    _yl = _xl
    plt.plot(_xl, _yl, "r--")

    plt.show()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Método de Lasso""")
    return


@app.cell
def __(Xb_test, Xb_train, lm, yb_train):
    _model = lm.Lasso(alpha=1)
    _model.fit(Xb_train, yb_train)

    y_lasso_pred = _model.predict(Xb_test)
    return y_lasso_pred,


@app.cell
def __(np, plt, y_lasso_pred, yb_test):
    _fig = plt.figure()
    _l = plt.plot(y_lasso_pred, yb_test, "bo")
    plt.setp(_l, markersize=10)
    plt.setp(_l, markerfacecolor="C0")

    plt.ylabel("y", fontsize=15)
    plt.xlabel("Prediction", fontsize=15)

    # mostra os valores preditos e originais
    _xl = np.arange(
        min(yb_test), 1.2 * max(yb_test), (max(yb_test) - min(yb_test)) / 10
    )
    _yl = _xl
    plt.plot(_xl, _yl, "r--")
    plt.text(40, 55, f"$\\alpha = 1$", fontsize=12)

    plt.show()
    return


@app.cell
def __(Xb_test, Xb_train, lm, np, plt, r2_score, yb_test, yb_train):
    _vR2 = []
    _valpha = []
    for _alpha in np.arange(0.01, 5, 0.01):
        _lasso = lm.Lasso(alpha=_alpha)
        _lasso.fit(
            Xb_train, yb_train
        )  # Fit a ridge regression on the training data
        _y_ridge_pred = _lasso.predict(
            Xb_test
        )  # Use this model to predict the test data
        _r2 = r2_score(yb_test, _y_ridge_pred)
        _vR2.append(_r2)
        _valpha.append(_alpha)
    plt.plot(_valpha, _vR2, "--r")
    plt.title(
        "R2 entre $y$ e $\\hat{y}$ do método de Lasso\nem função de $\\alpha$"
    )
    plt.xlabel("$\\alpha$", fontsize=15)
    plt.ylabel("R2", fontsize=15)
    plt.show()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Método de Ridge""")
    return


@app.cell
def __(Xb_test, Xb_train, lm, yb_train):
    _model = lm.Ridge(alpha=0.2)
    _model.fit(Xb_train, yb_train)

    y_ridge_pred = _model.predict(Xb_test)
    return y_ridge_pred,


@app.cell
def __(np, plt, y_ridge_pred, yb_test):
    _fig = plt.figure()
    _l = plt.plot(y_ridge_pred, yb_test, "bo")
    plt.setp(_l, markersize=10)
    plt.setp(_l, markerfacecolor="C0")

    plt.ylabel("y", fontsize=15)
    plt.xlabel("Prediction", fontsize=15)

    # mostra os valores preditos e originais
    _xl = np.arange(
        min(yb_test), 1.2 * max(yb_test), (max(yb_test) - min(yb_test)) / 10
    )
    _yl = _xl
    plt.plot(_xl, _yl, "r--")
    plt.title("$y_ \\times \\hat{y}$\n Método de Ridge")
    plt.text(40, 55, f"$\\alpha = 1$", fontsize=12)
    plt.show()
    return


@app.cell
def __(Xb_test, Xb_train, lm, np, plt, r2_score, yb_test, yb_train):
    _vR2 = []
    _valpha = []
    for _alpha in np.arange(0.01, 4, 0.01):
        _lasso = lm.Ridge(alpha=_alpha)
        _lasso.fit(
            Xb_train, yb_train
        )  # Fit a ridge regression on the training data
        _y_ridge_pred = _lasso.predict(
            Xb_test
        )  # Use this model to predict the test data
        _r2 = r2_score(yb_test, _y_ridge_pred)
        _vR2.append(_r2)
        _valpha.append(_alpha)
    plt.plot(_valpha, _vR2, "--r")
    plt.title(
        "R2 entre $y$ e $\\hat{y}$ do método de Lasso\nem função de $\\alpha$"
    )
    plt.xlabel("$\\alpha$", fontsize=15)
    plt.ylabel("R2", fontsize=15)
    plt.show()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Ex 2.

        Determine as variáveis que mais influenciam o preço de imóveis em Boston usando
        Lasso.
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Podemos usar o fato de que a qualidade das predições piora com o aumento de $\alpha$ para escolher um valor adequado para o hiper parâmetro. No caso, usamos $0.2$, onde temos uma mudança no declive da curva.""")
    return


@app.cell
def __(Xb_train, lm, yb_train):
    # Fazemos a regressão
    lasso = lm.Lasso(alpha=0.2)
    lasso.fit(Xb_train, yb_train)

    # Lemos quais são as váriaveis usadas
    _fileName = "data/BostonHousing.csv"
    _fileHandle = open(_fileName, "r")
    _cabeçalho = _fileHandle.readline()  # Lemos o cabeçalho
    _cabeçalho = _cabeçalho.replace('"', "")
    _fileHandle.close()
    _cabeçalho = _cabeçalho.split(",")[:-1]
    print(
        "\n".join(
            [
                f"{_cabeçalho[i]}: {lasso.coef_[i]:.2f}"
                for i in range(len(_cabeçalho))
            ]
        )
    )
    1
    return lasso,


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        Vemos que os paramentros mais influentes são `rm` (número de comodos por habitação), `dis` (distância de centros de emprego) e `ptratio` (número de alunos por professor).

        Dificilmente surpreendente. Como gostamos!
        """
    )
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Ex 3. 

        Considere os dados gerados com o código a seguir. Usando regularização, ajuste o
        grau do polinômio que define o modelo mais adequado.
        """
    )
    return


@app.cell
def __(np, plt):
    """Código do ex.3"""

    np.random.seed(10)


    # função para gerar os dados
    def function(x):
        y = x**4 + x**9
        return y


    # training set
    N_train = 20
    sigma = 0.2
    x_train = np.linspace(0, 1, N_train)
    y_train = function(x_train) + np.random.normal(0, sigma, N_train)
    x_train = x_train.reshape(len(x_train), 1)
    fig = plt.figure(figsize=(8, 4))
    plt.scatter(
        x_train,
        y_train,
        facecolor="blue",
        edgecolor="b",
        s=100,
        label="training data",
    )
    # test set
    N_test = 20
    x_test = np.linspace(0, 1, N_test)
    y_test = function(x_test) + np.random.normal(0, sigma, N_test)
    x_test = x_test.reshape(len(x_test), 1)
    # Curva teorica
    xt = np.linspace(0, 1, 100)
    yt = function(xt)
    plt.plot(xt, yt, "-r", label="Theoretical curve")
    plt.legend(fontsize=15)
    plt.show()
    return (
        N_test,
        N_train,
        fig,
        function,
        sigma,
        x_test,
        x_train,
        xt,
        y_test,
        y_train,
        yt,
    )


@app.cell
def __(PolynomialFeatures, lm, np, plt, x_test, x_train, y_test, y_train):
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean(np.square(y_true - y_pred)))


    poly15 = PolynomialFeatures(degree=15)
    X_train = poly15.fit_transform(x_train)
    X_test = poly15.transform(x_test)

    _alphaV = np.arange(0.0002, 0.002, 0.0001)

    rmseV = []

    for _i in range(len(_alphaV)):
        _alpha = _alphaV[_i]
        _lasso = lm.Lasso(alpha=_alpha, max_iter=5000)
        _lasso.fit(X_train, y_train)
        _y_pred = _lasso.predict(X_test)
        rmseV.append(rmse(y_test, _y_pred))
    plt.figure()
    plt.plot(_alphaV, rmseV, marker="o")
    plt.xlabel("$\\alpha$", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.title("RMSE vs $\\alpha$", fontsize=18)
    plt.show()
    return X_test, X_train, poly15, rmse, rmseV


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Temos então que o menor erro ocorre para $\alpha = 0.009$. Usando esse valor, obtemos então os coeficientes""")
    return


@app.cell
def __(X_test, X_train, lm, plt, x_test, xt, y_train, yt):
    _alpha = 0.009
    _lasso = lm.Lasso(alpha=_alpha, max_iter=5000)
    _lasso.fit(X_train, y_train)
    _y_pred = _lasso.predict(X_test)


    lasso_coef = _lasso.coef_

    _fig, _axs = plt.subplots(2, 1, figsize=(10, 10))
    _axs[1].bar(range(len(lasso_coef)), lasso_coef)
    _axs[1].set_xticks(range(len(lasso_coef)))
    _axs[1].set_xlim([0, 10])

    _axs[1].set_title("Lasso Regression Coefficients")

    _axs[0].plot(xt, yt, "k:", label="Theoretical curve")
    _axs[0].plot(x_test, _y_pred, "-r", label="Predicted curve")
    _axs[0].legend(fontsize=15)

    plt.show()
    return lasso_coef,


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Vemos então que o programa resume os dois graus do polinômio em um único, de expoente 5.""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Ex 4.

        Realize a classificação da base Vehicles usando validação cruzada e o método
        grid_search para escolher os melhores hiperparâmetros do modelo regressão logística e
        knn.
        """
    )
    return


@app.cell
def __(LabelEncoder, StandardScaler, pd, train_test_split):
    df_vehicle = pd.read_csv("data/Vehicle.csv")

    vehicle = df_vehicle.to_numpy()
    vehicle_y = vehicle[:, -1]
    vehicle_X = vehicle[:, 0:-1]

    # Pré processamento
    v_scaler = StandardScaler()
    v_scaler.fit(vehicle_X)
    vehicle_X = v_scaler.transform(vehicle_X)

    label_encoder = LabelEncoder()
    vehicle_y = label_encoder.fit_transform(vehicle_y)

    # Divisão dos train/test
    vX_train, vX_test, vy_train, vy_test = train_test_split(
        vehicle_X, vehicle_y, train_size=0.7, random_state=42
    )
    return (
        df_vehicle,
        label_encoder,
        vX_test,
        vX_train,
        v_scaler,
        vehicle,
        vehicle_X,
        vehicle_y,
        vy_test,
        vy_train,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### K-Vizinhos

        Achando os melhores parâmetros
        """
    )
    return


@app.cell
def __(GridSearchCV, KNeighborsClassifier, np, vX_train, vy_train):
    _grid_params = {
        "n_neighbors": np.arange(1, 10),
        "weights": ["uniform", "distance"],
        "metric": ["minkowski", "euclidean", "manhattan"],
    }
    _gs = GridSearchCV(
        KNeighborsClassifier(), _grid_params, verbose=0, cv=5, n_jobs=-1
    )
    _g_res = _gs.fit(vX_train, vy_train)

    print("Acurácia: ", _g_res.best_score_)
    print("Melhores hiperparâmetros: ", _g_res.best_params_)
    return


@app.cell
def __(
    KNeighborsClassifier,
    accuracy_score,
    vX_test,
    vX_train,
    vy_test,
    vy_train,
):
    _best_model = KNeighborsClassifier(
        n_neighbors=7, metric="manhattan", weights="distance"
    )

    _best_model.fit(vX_train, vy_train)
    _y_pred = _best_model.predict(vX_test)
    print("Accuracy:", accuracy_score(_y_pred, vy_test))
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Regressão Logística

        Achando os melhores parâmetros
        """
    )
    return


@app.cell
def __(GridSearchCV, LogisticRegression, vX_train, vy_train):
    _grid_params = {
        "penalty": [None, "l2"],
        "solver": ["saga", "lbfgs"],
        "class_weight": [None, "balanced"],
    }
    _gs = GridSearchCV(
        LogisticRegression(max_iter=1000),
        _grid_params,
        verbose=0,
        cv=5,
        n_jobs=-1,
    )
    _g_res = _gs.fit(vX_train, vy_train)


    print("Acurácia: ", _g_res.best_score_)
    print("Melhores hiperparâmetros: ", _g_res.best_params_)
    return


@app.cell
def __(
    LogisticRegression,
    accuracy_score,
    vX_test,
    vX_train,
    vy_test,
    vy_train,
):
    _best_model = LogisticRegression(
        solver="saga", class_weight=None, penalty=None, max_iter=5000
    )
    _best_model.fit(vX_train, vy_train)
    _y_pred = _best_model.predict(vX_test)
    print("Accuracy:", accuracy_score(_y_pred, vy_test))
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ## Ex 5.

        Verifique se o número de folds, usado na validação cruzada, influencia na classificação
        da base winequality-red. Use o modelo de regressão logística.
        """
    )
    return


@app.cell
def __(pd, train_test_split, v_scaler):
    df_wine = pd.read_csv("data/winequality-red.csv")

    wine = df_wine.to_numpy()
    wine_y = wine[:, -1]
    wine_X = wine[:, 0:-1]

    # Pré processamento
    v_scaler.fit(wine_X)
    wine_X = v_scaler.transform(wine_X)

    # Divisão dos train/test
    wX_train, wX_test, wy_train, wy_test = train_test_split(
        wine_X, wine_y, train_size=0.7, random_state=42
    )
    return (
        df_wine,
        wX_test,
        wX_train,
        wine,
        wine_X,
        wine_y,
        wy_test,
        wy_train,
    )


@app.cell(hide_code=True)
def __(GridSearchCV, LogisticRegression, wX_train, wy_train):
    _grid_params = {
        "penalty": [None, "l2"],
        "solver": ["saga", "lbfgs"],
        "class_weight": [None, "balanced"],
    }
    _gs = GridSearchCV(
        LogisticRegression(max_iter=1000),
        _grid_params,
        verbose=0,
        cv=5,
        n_jobs=-1,
    )
    _g_res = _gs.fit(wX_train, wy_train)


    print("Acurácia: ", _g_res.best_score_)
    print("Melhores hiperparâmetros: ", _g_res.best_params_)
    return


@app.cell
def __(
    KFold,
    LogisticRegression,
    accuracy_score,
    np,
    wX_test,
    wX_train,
    wy_test,
    wy_train,
):
    Ns = 10  # number of executions
    mean_vacc = []
    for _n in range(2, Ns + 1):
        cv = KFold(n_splits=_n)
        vacc = []
        for train_index, test_index in cv.split(wX_train, wy_train):
            _x_train, _x_test = wX_train[train_index], wX_train[test_index]
            _y_train, _y_test = wy_train[train_index], wy_train[test_index]
            _model = LogisticRegression(
                solver="saga", penalty="l2", max_iter=1000
            )
            _model.fit(_x_train, _y_train)
            _y_pred = _model.predict(wX_test)
            _score = accuracy_score(_y_pred, wy_test)
            vacc.append(_score)

        mean_vacc.append(np.mean(vacc))
    return Ns, cv, mean_vacc, test_index, train_index, vacc


@app.cell
def __(Ns, mean_vacc, plt):
    plt.plot(range(2, Ns + 1), mean_vacc)
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""Podemos ver que a influência do número de folds parece ser MUITO pequena. Da ordem da terceira casa decimal na acurácia.""")
    return


if __name__ == "__main__":
    app.run()
