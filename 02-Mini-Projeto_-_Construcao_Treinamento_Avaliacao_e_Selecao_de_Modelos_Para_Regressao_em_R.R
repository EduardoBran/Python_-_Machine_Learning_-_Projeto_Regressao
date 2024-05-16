####  Big Data Real-Time Analytics com Python e Spark  ####

# Configurando o diretório de trabalho
setwd("~/Desktop/DataScience/CienciaDeDados/2.Big-Data-Real-Time-Analytics-com-Python-e-Spark/9.Machine_Learning_em_Python_-_MiniProjeto_Regressao_-_Empresa_Ecommerce")
getwd()



## Importando Pacotes
library(readxl)         # carregar arquivos
library(dplyr)          # manipula dados
library(tidyr)          # manipula dados (funcao pivot_longer)
library(ROSE)           # balanceamento de dados
library(ggplot2)        # gera gráficos
library(patchwork)      # unir gráficos
library(corrplot)       # mapa de correlação

library(caret)          # pacote preProcess para normalização / facilitar a validação cruzada / seleção de hiperparâmetros
library(pROC)           # Para ROC e AUC

library(glmnet)         # algoritmo para regressão logística com regularização
library(randomForest)   # algoritmo de ML
library(class)          # algortimo KNN
library(rpart)          # algoritmo árvore de decisão (Decision Tree)
library(rpart.plot)
library(e1071)          # algoritmo SVM





############################  Modelagem Preditiva para Empresa de Ecommerce  ############################  

## Etapas:

# - Construção, Treinamento, Avaliação e Seleção de Modelos para Regressão

## Introdução:

# - Vamos trabalhar agora em nosso segundo Mini-Projeto de Machine Learning, cujo objetivo é trazer para você um passo a passo completo do processo de construção,
#   treinamento, avaliação e seleção de modelos para regressão. Todo o processo será demonstrado de uma ponta a outra, desde a definição do problema de negócio, até a
#   interpretação do modelo e entrega do resultado ao tomador de decisão.

## Contexto:

# - Uma empresa de e-commerce comercializa produtos através de seu web site e de sua app para dispositivos móveis. Para efetuar uma compra, um cliente realiza um
#   cadastro no portal (usando web site ou app). Cada vez que o cliente realiza o login, o sistema registra o tempo que fica logado, seja na app ou no web site.
# - Para cada cliente, a empresa mantém o registro de vendas com o total gasto por mês.
# - A empresa gostaria de aumentar as vendas, mas o orçamento permite investir somente no web site ou na app neste momento. O foco é melhorar a experiência do
#   cliente durante a navegação no sistema, aumentando o tempo logado, aumentando o engajamento e, consequentemente, aumentando as vendas.

## Objetivo:

# - O objetivo deste projeto é ajudar a empresa a tomar uma decisão sobre onde investir (web site ou app) para melhorar a experiência do cliente, aumentando o tempo
#   logado e, assim, o engajamento e as vendas.
# - Vamos construir, treinar, avaliar e selecionar modelos de regressão para prever o valor total gasto pelo cliente em um mês, com base no tempo logado e outras
#   variáveis.

## Dados:

# - Os dados representam um mês de operação do portal de e-commerce. As colunas do conjunto de dados são auto-explicativas e descritas a seguir:

# -> tempo_cadastro_cliente          : Tempo que o cliente está cadastrado, convertido de meses para anos (float64).
# -> numero_medio_cliques_por_sessao : Número médio de cliques (ou toques) em cada sessão (float64).
# -> tempo_total_logado_app          : Tempo total logado na app em minutos (float64).
# -> tempo_total_logado_website      : Tempo total logado no web site em minutos (float64).
# -> valor_total_gasto               : Valor total gasto pelo cliente em um mês em R$ (float64).




#### Carregando os Dados















