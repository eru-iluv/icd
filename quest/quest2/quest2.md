---
title: Quest2
marimo-version: 0.8.3
width: medium
---

```{.python.marimo}
import marimo as mo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
```

# Questionário 2

Análise e exploração dos dados de expectativa de vida em diversos países. Retirado do:

[github de Aishwarya Ramakrishnan](https://gist.github.com/aishwarya8615/89d9f36fc014dea62487f7347864d16a)
<!---->
## Exercício 1

Construir um gráfico de setores para variável status. Qual a porcentagem de países desenvolvidos?

```{.python.marimo}
url = "https://gist.github.com/aishwarya8615/89d9f36fc014dea62487f7347864d16a"

life_expectancy = pd.read_html(url)[0]
life_expectancy.drop(columns=["Unnamed: 0"], inplace=True)
```

```{.python.marimo}
life_expectancy
```

```{.python.marimo}
labels, counts = np.unique(life_expectancy["Status"], return_counts=True)
```

```{.python.marimo}
explode = (0, 0.04)
fig1, ax1 = plt.subplots(figsize=(10,6))
ax1.pie(counts, explode=explode, labels=labels, autopct="%1.1f%%")
ax1.axis('equal')
plt.show()
```

## Exercício 2

Considerando a base anterior, construa um histograma para variável life expectancy

```{.python.marimo}
fig = plt.figure(figsize=(6,4))

num_bins = 10 
n, bins, patches =  plt.hist(life_expectancy["Life_expectancy"], num_bins,
    facecolor="blue", alpha=0.5, density=True, edgecolor='black')
plt.xlabel("Expectativa de vida", fontsize=15)
plt.ylabel("Frequência")
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.show()
```

```{.python.marimo}
var_life_expectancy = life_expectancy['Life_expectancy'].var()
mean_life_expectancy = life_expectancy['Life_expectancy'].mean();
```

```{.python.marimo hide_code="true"}
mo.md("""A expectativa de vida nos dados apresentou um valor médio de {media:.1f} e uma variância de {var:.1f}""".format(media = mean_life_expectancy, var =  var_life_expectancy))
```

```{.python.marimo}
country_mask = (life_expectancy["Country"] == 'Angola') | (life_expectancy["Country"] == 'Finland') | (life_expectancy["Country"] == 'Ireland') | (life_expectancy["Country"] == 'Netherlands') | (life_expectancy["Country"] == 'Zambia')
```

```{.python.marimo}
df_selected_contries = life_expectancy[country_mask]
```

```{.python.marimo}
df_selected_contries
```

```{.python.marimo}
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
```

Vemos que os países com maior e menor expectativa de vida são Países Baixos e Angola, respectivamente.
<!---->
## Exercício 4

Fazer um boxplot para variável Schooling e ver qual apresenta a maior mediana para escolaridade.

```{.python.marimo}
fig3, axs3 = plt.subplots(figsize=(7,7))

plt.title("Evolution of life expectancy in 5 countries.", fontsize=20)

seaborn.boxplot(x='Country',y='Schooling', data=df_selected_contries )

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.xlabel("Country", fontsize=15)
plt.ylabel("Schooling", fontsize=15)
plt.show()
```

Notamos que a Irlanda apresenta a maior mediana para expectativa de vida.
<!---->
## Exercício 5

Construir a matriz de correlação e ver a menos relacionada.

```{.python.marimo}
import requests
from bs4 import BeautifulSoup
```

```{.python.marimo}
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
```

```{.python.marimo hide_code="true"}
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
```

As váriaveis menos relacionadas são a taxa de fertilidade com a idade média