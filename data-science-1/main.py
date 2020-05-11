#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF


# In[2]:


# %matplotlib inline

# from IPython.core.pylabtools import figsize


# figsize(12, 8)

sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[3]:


np.random.seed(42)
    
dataframe = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[4]:


# Sua análise da parte 1 começa aqui.
dataframe.head()


# In[5]:


# Sua análise da parte 1 começa aqui.
dataframe.describe()


# In[6]:


dataframe.normal.plot(kind='hist')


# In[7]:


dataframe.binomial.plot(kind='hist')


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[8]:


def q1():
    quartis = [np.quantile(dataframe['normal'], i) - np.quantile(dataframe['binomial'], i) for i in [0.25, 0.5, 0.75]]
    return tuple(np.round(quartis, 3))


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# **Sim, era esperado valores dessa magnitude, tendo em vista que as amostras foram geradas com médias e desvio padrão semelhantes semelhantes.**

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[9]:


def q2():
    z_inf = dataframe['normal'].mean() - dataframe['normal'].std()
    z_sup = dataframe['normal'].mean() + dataframe['normal'].std()

    ecdf = ECDF(dataframe['normal'])
    
    answer =  np.round(ecdf(z_sup) - ecdf(z_inf),3)
   
    return float(answer)


# In[10]:


q2()


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# **Sim, por se tratar de uma distribuição normal, e o intervalo de um desvio padrão da média é corresponde a aproximadamente a 68% da proporção dos dados, provavelmente os outros intervalos seguiram a mesma propriedade da curva normal, 95% e 99% respectivamente**

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[11]:


def q3():
    diff_mean = dataframe['binomial'].mean() - dataframe['normal'].mean()
    diff_var = dataframe['binomial'].var() - dataframe['normal'].var()
    return np.round(diff_mean, 3) , np.round(diff_var, 3)


# In[12]:


q3()


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# 
# **Sim, as amostras foram geradas com médias e desvios semelhantes, conforma a lei dos grandes números, comm a amostra grande o suficiente, a média sera próxima da população.**
# 
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?
# 
# **Aumentando o n, as médias vão tendem a se igualarem conforme a lei dos grandes número (Se aproximando do valor da população) e dimunindo tendem a se afastar.**

# ## Parte 2

# ### _Setup_ da parte 2

# In[13]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[14]:


# Sua análise da parte 2 começa aqui.
stars.head()


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[15]:


false_pulsar = stars[stars.target == False] ['mean_profile']
false_pulsar_mean_profile_standardized = (false_pulsar - false_pulsar.mean())/false_pulsar.std()
ecdf = ECDF(false_pulsar_mean_profile_standardized)


# In[16]:


def q4():
    quantis = [sct.norm.ppf(i, 0, 1) for i in [0.8, 0.9, 0.95]]
    return tuple(np.round(ecdf(quantis), 3))


# In[17]:


q4()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# 
#     **Sim, como foi feito a padronização das distribuições, elas ficaram mais 'normatizadas' e como consequência tem os quartis próximos da distribuição normal gerada**
# 
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# 
#     **Que a mesma é normalmente distribuída**
# 

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[18]:


def q5():
    dist = false_pulsar_mean_profile_standardized
    return tuple(np.round([np.quantile(dist, i) - sct.norm.ppf(i, 0, 1) for i in [0.25, 0.5, 0.75]], 3))


# In[19]:


q5()


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# 
#     **Sim, como visto anteriormente, a `false_pulsar_mean_profile_standardized` corresponde aproximadamente uma distribuição normal, então a diferença entre o real o teórico é muito pequeno**
#     
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# 
#     **Confirma estatisticamente, que ela é normalmente distribuída**
# 
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
# 

# In[20]:


false_pulsar_mean_profile_standardized.plot(kind='hist')

