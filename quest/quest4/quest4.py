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

    theme = load_theme("boxy_light")
    theme.apply()
    # ... plotting code here
    theme.apply_transforms()
    return (
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


@app.cell(hide_code=True)
def __(pd, train_test_split):
    df = pd.read_csv('data/BostonHousing.csv')
    _data = df.to_numpy()
    Xb = _data[:,:-1]
    yb = _data[:,-1]
    p = 0.3
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size = p, random_state = 42)
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


@app.cell(hide_code=True)
def __(np, plt, y_reg_pred, yb_test):
    _fig = plt.figure()
    _l = plt.plot(y_reg_pred, yb_test, 'bo')
    plt.setp(_l, markersize=10)
    plt.setp(_l, markerfacecolor='C0')

    plt.ylabel("y", fontsize=15)
    plt.xlabel("Prediction", fontsize=15)

    # mostra os valores preditos e originais
    _xl = np.arange(min(yb_test), 1.2*max(yb_test),(max(yb_test)-min(yb_test))/10)
    _yl = _xl
    plt.plot(_xl, _yl, 'r--')

    plt.show()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Método de Lasso""")
    return


@app.cell
def __():
    return


@app.cell(hide_code=True)
def __(mo):
    alpha = mo.ui.slider(0.001, 1, value=1, step=0.1, label='alpha')

    mo.md(
        f"""
        $\\alpha$

        {alpha}
        """
    )
    return alpha,


@app.cell
def __(Xb_test, Xb_train, alpha, lm, yb_train):
    _model = lm.Lasso(alpha=alpha.value)
    _model.fit(Xb_train, yb_train)

    y_lasso_pred = _model.predict(Xb_test)
    return y_lasso_pred,


@app.cell(hide_code=True)
def __(alpha, np, plt, y_lasso_pred, yb_test):
    _fig = plt.figure()
    _l = plt.plot(y_lasso_pred, yb_test, 'bo')
    plt.setp(_l, markersize=10)
    plt.setp(_l, markerfacecolor='C0')

    plt.ylabel("y", fontsize=15)
    plt.xlabel("Prediction", fontsize=15)

    # mostra os valores preditos e originais
    _xl = np.arange(min(yb_test), 1.2*max(yb_test),(max(yb_test)-min(yb_test))/10)
    _yl = _xl
    plt.plot(_xl, _yl, 'r--')
    plt.text(40,55,f"$\\alpha = {alpha.value}$", fontsize=12)

    plt.show()
    return


@app.cell(hide_code=True)
def __(Xb_test, Xb_train, lm, np, plt, r2_score, yb_test, yb_train):
    _vR2 = []
    _valpha = []
    for _alpha in np.arange(0.01,5,0.01):
        _lasso = lm.Lasso(alpha = _alpha)
        _lasso.fit(Xb_train, yb_train)             # Fit a ridge regression on the training data
        _y_ridge_pred = _lasso.predict(Xb_test)           # Use this model to predict the test data
        _r2 = r2_score(yb_test, _y_ridge_pred)
        _vR2.append(_r2)
        _valpha.append(_alpha)
    plt.plot(_valpha, _vR2, '-r', ls='--')
    plt.title("R2 entre $y$ e $\\hat{y}$ do método de Lasso\nem função de $\\alpha$")
    plt.xlabel("$\\alpha$", fontsize=15)
    plt.ylabel("R2", fontsize=15)
    plt.show()
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""### Método de Ridge""")
    return


@app.cell(hide_code=True)
def __(Xb_test, Xb_train, lm, yb_train):
    _model = lm.Ridge(alpha=0.2)
    _model.fit(Xb_train, yb_train)

    y_ridge_pred = _model.predict(Xb_test)
    return y_ridge_pred,


@app.cell(hide_code=True)
def __(alpha, np, plt, y_ridge_pred, yb_test):
    _fig = plt.figure()
    _l = plt.plot(y_ridge_pred, yb_test, 'bo')
    plt.setp(_l, markersize=10)
    plt.setp(_l, markerfacecolor='C0')

    plt.ylabel("y", fontsize=15)
    plt.xlabel("Prediction", fontsize=15)

    # mostra os valores preditos e originais
    _xl = np.arange(min(yb_test), 1.2*max(yb_test),(max(yb_test)-min(yb_test))/10)
    _yl = _xl
    plt.plot(_xl, _yl, 'r--')
    plt.title("$y_ \\times \\hat{y}$\n Método de Ridge")
    plt.text(40,55,f"$\\alpha = {alpha.value}$", fontsize=12)
    plt.show()
    return


@app.cell(hide_code=True)
def __(Xb_test, Xb_train, lm, np, plt, r2_score, yb_test, yb_train):
    _vR2 = []
    _valpha = []
    for _alpha in np.arange(0.01,4,0.01):
        _lasso = lm.Ridge(alpha = _alpha)
        _lasso.fit(Xb_train, yb_train)             # Fit a ridge regression on the training data
        _y_ridge_pred = _lasso.predict(Xb_test)           # Use this model to predict the test data
        _r2 = r2_score(yb_test, _y_ridge_pred)
        _vR2.append(_r2)
        _valpha.append(_alpha)
    plt.plot(_valpha, _vR2, '-r', ls='--')
    plt.title("R2 entre $y$ e $\\hat{y}$ do método de Lasso\nem função de $\\alpha$")
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
    mo.md(r"""Podemos usar o fato de que a qualidade das predições piora com o aumento de $\alpha$ para escolher um valor adequado para o hiper parâmetro. No caso, usamos 2, onde temos uma mudança no declive da curva.""")
    return


@app.cell
def __(Xb_train, lm, yb_train):
    # Fazemos a regressão
    lasso = lm.Lasso(alpha = 0.2)
    lasso.fit(Xb_train, yb_train)

    # Lemos quais são as váriaveis usadas
    _fileName = "data/BostonHousing.csv"
    _fileHandle = open(_fileName, 'r')
    _cabeçalho = _fileHandle.readline()  # Lemos o cabeçalho
    _cabeçalho = _cabeçalho.replace('"','')
    _fileHandle.close()
    _cabeçalho = _cabeçalho.split(',')[:-1]
    print("\n".join([f"{_cabeçalho[i]}: {lasso.coef_[i]:.2f}" for i in range(len(_cabeçalho))]))
    1
    return lasso,


if __name__ == "__main__":
    app.run()
