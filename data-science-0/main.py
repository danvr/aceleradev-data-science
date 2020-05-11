#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[3]:


def q1():
    # Retorne aqui o resultado da questão 1.
    return black_friday.shape


# In[4]:


q1()


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[5]:


def q2():
    # Retorne aqui o resultado da questão 2.
    return black_friday.query('Gender == "F" & Age == "26-35"').shape[0]


# In[6]:


q2()


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[7]:


def q3():
    # Retorne aqui o resultado da questão 3.
    return black_friday.User_ID.nunique()


# In[8]:


q3()


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[9]:


def q4():
    # Retorne aqui o resultado da questão 4.
    return len(black_friday.dtypes.unique())


# In[10]:


q4()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[11]:


def q5():
    # Retorne aqui o resultado da questão 5.
    total_nan = black_friday.shape[0] - black_friday.dropna().shape[0] 
    data_size = black_friday.shape[0]
    return total_nan/data_size


# In[12]:


q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[13]:


def q6():
    # Retorne aqui o resultado da questão 6.
    null_array = black_friday.isna().sum()
    max_null_index = null_array.argmax()
    colmun_with_max_null = null_array[max_null_index]
    return int(colmun_with_max_null)


# In[14]:


q6()


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[15]:


def q7():
    # Retorne aqui o resultado da questão 7.
    return black_friday['Product_Category_3'].value_counts().index[0]


# In[16]:


q7()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[17]:


def q8():
    # Retorne aqui o resultado da questão 8.
    variable_min = black_friday['Purchase'].min()
    variable_max = black_friday['Purchase'].max()
    variable_normalized = (black_friday['Purchase'] - variable_min) / (variable_max - variable_min)
    return float(variable_normalized.mean())


# In[18]:


q8()


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[19]:


def q9():
    # Retorne aqui o resultado da questão 9.
    variable_mean = black_friday['Purchase'].mean()
    variable_std = black_friday['Purchase'].std()
    variable_normalized = (black_friday['Purchase'] - variable_mean) / variable_std
    return len (variable_normalized[variable_normalized.between(-1,1)])


# In[20]:


q9()


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[21]:


def q10():
    # Retorne aqui o resultado da questão 10.
    product2_bool = black_friday['Product_Category_2'].isna()
    product3_bool = black_friday['Product_Category_2'].isna()
    bool_check  = product2_bool.equals(product3_bool)
    return bool_check


# In[22]:


q10()

