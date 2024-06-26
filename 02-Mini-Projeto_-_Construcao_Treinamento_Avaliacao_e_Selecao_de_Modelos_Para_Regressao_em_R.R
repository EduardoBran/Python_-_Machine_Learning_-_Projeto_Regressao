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
library(GGally)         # gera gráficos
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
library(Metrics)




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


## IMPORTANTE:

# - Analisando nossos dados, definimos a variável "valor_total_gasto" como nossa variável alvo.
# - Para este projeto, não queremos apenas prever o valor; queremos estudar o relacionamento entre as variáveis.
# - Isso será uma entrega adicional do modelo de Machine Learning.

# - Internamente, os modelos de Machine Learning fazem relacionamentos entre os dados. Portanto, é exatamente esse relacionamento que será usado para resolver o
#   problema de negócio.
# - Vamos criar um modelo que será capaz de realizar previsões. No entanto, nosso foco não será as previsões, mas sim o relacionamento que o modelo encontrou durante
#   seu aprendizado.
# - Esta é uma característica/propriedade dos modelos de Machine Learning que nos permite resolver ainda mais problemas de negócio.




#### Carregando os Dados
df <- data.frame(read.csv("dados/dataset.csv", stringsAsFactors = FALSE))

dim(df)
names(df)
head(df)



#### Análise Exploratória

# Tipo de dados
str(df)


## Realizando Análise Inicial (Sumário Estatístico, Veriricação de Valores NA, '' e especiais)

analise_inicial <- function(dataframe_recebido) {
  # Sumário
  cat("\n\n####  DIMENSÕES  ####\n\n")
  print(dim(dataframe_recebido))
  cat("\n\n\n####  INFO  ####\n\n")
  print(str(dataframe_recebido))
  cat("\n\n\n####  SUMÁRIO  ####\n\n")
  print(summary(dataframe_recebido))
  cat("\n\n\n####  VERIFICANDO QTD DE LINHAS DUPLICADAS  ####\n\n")
  print(sum(duplicated(dataframe_recebido)))
  cat("\n\n\n####  VERIFICANDO VALORES NA  ####\n\n")
  valores_na <- colSums(is.na(dataframe_recebido))
  if(any(valores_na > 0)) {
    cat("\n-> Colunas com valores NA:\n\n")
    print(valores_na[valores_na > 0])
  } else {
    cat("\n-> Não foram encontrados valores NA.\n")
  }
  cat("\n\n\n####  VERIFICANDO VALORES VAZIOS ''  ####\n\n")
  valores_vazios <- sapply(dataframe_recebido, function(x) sum(x == "", na.rm = TRUE)) # Adicionando na.rm = TRUE
  if(any(valores_vazios > 0, na.rm = TRUE)) { # Tratamento de NA na condição
    cat("\n-> Colunas com valores vazios \"\":\n\n")
    print(valores_vazios[valores_vazios > 0])
  } else {
    cat("\n-> Não foram encontrados valores vazios \"\".\n")
  }
  cat("\n\n\n####  VERIFICANDO VALORES COM CARACTERES ESPECIAIS  ####\n\n")
  caracteres_especiais <- sapply(dataframe_recebido, function(x) {
    sum(sapply(x, function(y) {
      if(is.character(y) && length(y) == 1) {
        any(charToRaw(y) > 0x7E | charToRaw(y) < 0x20)
      } else {
        FALSE
      }
    }))
  })
  if(any(caracteres_especiais > 0)) {
    cat("\n-> Colunas com caracteres especiais:\n\n")
    print(caracteres_especiais[caracteres_especiais > 0])
  } else {
    cat("\n-> Não foram encontrados caracteres especiais.\n")
  }
}

analise_inicial(df)
rm(analise_inicial)



## CORRELAÇÃO

# Verificando Correlação (Tabela)
cor(df, use = "complete.obs")

# Verificando Correlação (através de matriz de gráficos de dispersão (scatter plots))
pairs(df, main = "Matriz de Gráficos de Dispersão")

# Verificando Correlação através de um Mapa de Calor
corrplot(cor(df, use = "complete.obs"),
         method = "color",
         type = "upper",
         addCoef.col = 'springgreen2',
         tl.col = "black",
         tl.srt = 45)  




## Análise 1 - Relação Entre Tempo no Web Site e Valor Gasto

# Criando o gráfico de dispersão com histogramas marginais
ggplot(data = df, aes(x = tempo_total_logado_website, y = valor_total_gasto)) +
  geom_point(color = "blue") +
  geom_smooth(method = 'lm', col = 'red') +  # linha de tendência
  ggtitle("Relação Entre Tempo no Web Site e Valor Gasto") +
  theme_minimal() +
  theme(text = element_text(size = 16))

# Detalhe dos Gráficos:

# - O gráfico de dispersão sugere que não há uma correlação evidente entre tempo_total_logado_website e valor_total_gasto, pois os pontos estão espalhados de forma
#   aleatória e não seguem uma tendência clara.
# - Vamos confirmar calculando o coeficiente de correlação entre elas abaixo.

# Correlação (tabela)
cor(df[, c("tempo_total_logado_website", "valor_total_gasto")], use = "complete.obs")


## Conclusão:

# - Não parece haver correlação entre o tempo logado no web site e o valor gasto pelos clientes.





## Análise 2 - Relação Entre Tempo na App e Valor Gasto

# Criando o gráfico de dispersão com histogramas marginais
ggplot(data = df, aes(x = tempo_total_logado_app, y = valor_total_gasto)) +
  geom_point(color = "darkgreen") +
  geom_smooth(method = 'lm', col = 'red') +  # linha de tendência
  ggtitle("Relação Entre Tempo na App e Valor Gasto") +
  theme_minimal() +
  theme(text = element_text(size = 16))

# Detalhe dos Gráficos:

# - O gráfico de dispersão mostra uma correlação moderada positiva entre tempo_total_logado_app e valor_total_gasto. Os pontos no gráfico tendem a formar um padrão
#   mais concentrado, sugerindo que, à medida que o tempo logado no aplicativo aumenta, o valor total gasto também tende a aumentar.

# Correlação (tabela)
cor(df[, c("tempo_total_logado_app", "valor_total_gasto")], use = "complete.obs")

## Conclusão:

# - Há uma correlação moderada positiva entre o tempo total logado no aplicativo e o valor total gasto pelos clientes, com um coeficiente de correlação de 0.499328.
#   Isso indica que aumentar o tempo que os clientes passam no aplicativo pode estar associado a um aumento no valor gasto.





## Análise 3 - Relação Entre Tempo na App e Tempo de Cadastro

# Criando o gráfico de dispersão com histogramas marginais
ggplot(data = df, aes(x = tempo_total_logado_app, y = tempo_cadastro_cliente)) +
  geom_point(color = "darkgreen") +
  geom_smooth(method = 'lm', col = 'red') +  # linha de tendência
  ggtitle("Relação Entre Tempo na App e Tempo de Cadastro") +
  theme_minimal() +
  theme(text = element_text(size = 16))


# Detalhe dos Gráficos:

# - O gráfico de dispersão mostra que não há uma correlação clara entre tempo_total_logado_app e tempo_cadastro_cliente. Os pontos estão espalhados de forma
#   relativamente uniforme e aleatória, sem formar um padrão específico.

# Correlação (tabela)
cor(df[, c("tempo_total_logado_app", "tempo_cadastro_cliente")], use = "complete.obs")

## Conclusão:

# - Não parece haver uma correlação significativa entre o tempo total logado no aplicativo e o tempo de cadastro do cliente, com um coeficiente de correlação
#   de 0.029143. Isso indica que o tempo que os clientes passam no aplicativo não está diretamente relacionado com o tempo que eles estão cadastrados na plataforma.





## Análise 4 - Relação Entre Tempo de Cadastro e Valor Gasto

# Criando o gráfico de dispersão com histogramas marginais
ggplot(data = df, aes(x = tempo_cadastro_cliente, y = valor_total_gasto)) +
  geom_point(color = "darkgreen") +
  geom_smooth(method = 'lm', col = 'red') +  # linha de tendência
  ggtitle("Relação Entre Tempo de Cadastro e Valor Gasto") +
  theme_minimal() +
  theme(text = element_text(size = 16))

# Detalhe dos Gráficos:

# - O gráfico de dispersão mostra uma clara tendência positiva entre tempo_cadastro_cliente e valor_total_gasto. Conforme o tempo de cadastro do cliente aumenta,
#   o valor total gasto também tende a aumentar. A linha de tendência azul confirma essa correlação positiva, indicando uma relação linear significativa entre as
#   duas variáveis.

# Correlação (tabela)
cor(df[, c("tempo_cadastro_cliente", "valor_total_gasto")], use = "complete.obs")

## Conclusão:

# - Há uma forte correlação positiva (coeficiente de correlação de 0.809084) entre o tempo de cadastro do cliente e o valor total gasto. Isso sugere que clientes que
#   estão cadastrados há mais tempo tendem a gastar mais. Esta informação é valiosa para a empresa, indicando que investimentos em estratégias de retenção de clientes
#   podem ser eficazes para aumentar as vendas.





## Análise 5 - Relação Entre Tempo Logado na App e Tempo Logado no Web Site

# Criando o gráfico de dispersão com histogramas marginais
ggplot(data = df, aes(x = tempo_total_logado_app, y = tempo_total_logado_website)) +
  geom_point(color = "darkgreen") +
  geom_smooth(method = 'lm', col = 'red') +  # linha de tendência
  ggtitle("Relação Entre Tempo Logado na App e Tempo Logado no Web Site") +
  theme_minimal() +
  theme(text = element_text(size = 16))

# Detalhe dos Gráficos:

# - O gráfico de dispersão entre tempo_total_logado_app e tempo_total_logado_website mostra pontos distribuídos de maneira bastante aleatória, sem uma tendência clara.
#   Isso sugere que não há uma correlação forte entre essas duas variáveis.

# Correlação (tabela)


## Conclusão:

# - A correlação entre tempo_total_logado_app e tempo_total_logado_website é muito baixa (coeficiente de correlação de 0.082388). Isso indica que o tempo que os
#   clientes passam logados no app não está fortemente relacionado com o tempo que passam logados no website. Essa informação pode sugerir que os usuários tendem a
#   utilizar um dos canais (app ou website) mais frequentemente do que o outro, sem uma relação significativa entre os tempos de uso.







#### Pré-Processamento de Dados Para Construção de Modelos de Machine Learning

##  Dividindo os dados em treino e teste
set.seed(100)
indices <- createDataPartition(df$valor_total_gasto, p = 0.70, list = FALSE)
dados_treino <- df[indices, ]
dados_teste <- df[-indices, ]
rm(indices)


## Padronização

# Padronizado Dados de Treino
summary(dados_treino)

# Calculando a média e o desvio padrão dos dados de treino 
treino_mean <- sapply(dados_treino[, -which(names(dados_treino) == "valor_total_gasto")], mean, na.rm = TRUE)
treino_std <- sapply(dados_treino[, -which(names(dados_treino) == "valor_total_gasto")], sd, na.rm = TRUE)

# Exibindo a média e o desvio padrão
print(treino_mean)
print(treino_std)

# Padronizando todas as variáveis, exceto 'Target'
dados_treino[, names(treino_mean)] <- sweep(dados_treino[, names(treino_mean)], 2, treino_mean, "-")
dados_treino[, names(treino_std)] <- sweep(dados_treino[, names(treino_std)], 2, treino_std, "/")

summary(dados_treino)


# Padronizado Dados de Teste
summary(dados_teste)

# Padronizando os dados de teste usando a média e desvio padrão dos dados de treino
dados_teste[, names(treino_mean)] <- sweep(dados_teste[, names(treino_mean)], 2, treino_mean, "-")
dados_teste[, names(treino_std)] <- sweep(dados_teste[, names(treino_std)], 2, treino_std, "/")

summary(dados_teste)
rm(treino_mean, treino_std)



#### Construindo Modelos de Machine Learning

# Nesta etapa do projeto, desenvolveremos e avaliaremos três diferentes modelos de machine learning para identificar qual deles apresenta o melhor desempenho para
# o nosso conjunto de dados.

# Abaixo estão os modelos que serão implementados e testados:
  
# Modelo 1 -> Regressão Linear (Benchmark) - Utilizado como linha de base devido à sua simplicidade e eficácia em problemas de regressão. Este modelo ajudará a 
#             estabelecer uma base para a performance que esperamos superar com técnicas mais complexas.

# Modelo 2 -> Regressão Ridge - Um modelo de regressão que adiciona uma penalização L2 ao cálculo dos coeficientes, ajudando a reduzir o sobreajuste (overfitting) e
#             melhorando a capacidade de generalização do modelo.

#Modelo 3 -> Regressão LASSO - Um modelo de regressão que aplica uma penalização L1, forçando a soma dos valores absolutos dos coeficientes a serem menores que um 
#            valor fixo. Isso pode resultar na eliminação de algumas variáveis irrelevantes, tornando o modelo mais interpretável e eficiente.

# Cada modelo será treinado utilizando o mesmo conjunto de dados, permitindo uma comparação justa de sua eficácia. A avaliação de cada modelo incluirá métricas como 
# erro médio absoluto (MAE), erro quadrático médio (MSE) e o coeficiente de determinação (R²), entre outras, dependendo das especificidades de nosso problema e dados.




## Cria um dataframe para receber as métricas de cada modelo
df_modelos <- data.frame()



### Modelo 1 com Regressão Linear (Benchmark)

## Versão 1

# - Nesta versão, criamos e treinamos um modelo de Regressão Linear para prever o valor total gasto pelos clientes.
# - Iremos avaliar o modelo usando várias métricas de desempenho, visualizamos os coeficientes das variáveis preditoras e realizamos a análise de resíduos para
#   verificar o ajuste do modelo.
# - As métricas de desempenho serão salvas em um dataframe para futura comparação com outros modelos.


# Criação do modelo de Regressão Linear
modelo_v1_RL <- train(valor_total_gasto ~ ., data = dados_treino, method = "lm")

# Visualizando coeficientes das variáveis preditoras
coeficientes <- summary(modelo_v1_RL$finalModel)$coefficients
df_coef <- as.data.frame(coeficientes)
colnames(df_coef) <- c("Coeficiente", "Erro Padrão", "Valor T", "P-valor")
df_coef


## Interpretação do resultado dos coeficientes das variáveis preditoras:

# - Os coeficientes indicam a magnitude e a direção da influência de cada variável preditora no valor alvo (valor total gasto).
#   Por exemplo:
#      -> tempo_cadastro_cliente          : Cada ano adicional de cadastro aumenta o valor total gasto em média em 62.77 unidades.
#      -> numero_medio_cliques_por_sessao : Cada clique adicional por sessão aumenta o valor total gasto em média em 25.85 unidades.
#      -> tempo_total_logado_app          : Cada minuto adicional logado no app aumenta o valor total gasto em média em 38.69 unidades.
#      -> tempo_total_logado_website      : Cada minuto adicional logado no website aumenta o valor total gasto em média em 0.47 unidades.

# Assumindo que todas as outras variáveis permaneçam constantes.

## Previsões

# Previsões com dados de teste
pred_v1 <- predict(modelo_v1_RL, dados_teste)

# Imprime as 10 primeiras previsões
print(pred_v1[1:10])

# Plot das previsões vs valores reais
ggplot() +
  geom_point(aes(x = dados_teste$valor_total_gasto, y = pred_v1), color = 'skyblue') +
  labs(x = 'Valor Real de Y', y = 'Valor Previsto de Y') +
  theme_minimal()

# A partir do gráfico de dispersão, podemos ver que há uma correlação muito forte entre os y's previstos e os y's reais nos dados do teste. 
# Isso significa que temos um modelo muito bom.


## Avaliação do Modelo

# Métricas

# Valor médio gasto pelos clientes
valor_medio <- mean(df$valor_total_gasto)
print(paste('Valor médio gasto pelos clientes:', valor_medio))

# Valor mínimo
valor_minimo <- min(df$valor_total_gasto)
print(paste('Valor mínimo gasto pelos clientes:', valor_minimo))

# Valor máximo
valor_maximo <- max(df$valor_total_gasto)
print(paste('Valor máximo gasto pelos clientes:', valor_maximo))

# MAE - Erro Médio Absoluto
mae <- MAE(pred_v1, dados_teste$valor_total_gasto)
print(paste('MAE - Erro Médio Absoluto:', mae))

# MSE - Erro Quadrático Médio
mse <- mean((pred_v1 - dados_teste$valor_total_gasto)^2)
print(paste('MSE - Erro Quadrático Médio:', mse))

# RMSE - Raiz Quadrada do Erro Quadrático Médio
rmse <- sqrt(mse)
print(paste('RMSE - Raiz Quadrada do Erro Quadrático Médio:', rmse))


# - O RMSE prevê que, em média, as previsões do nosso modelo (de valores gastos) estão erradas em aproximadamente 9.35, que é um valor pequeno comparado ao
#   valor médio gasto por cliente.


# Coeficiente R2
r2 <- R2(pred_v1, dados_teste$valor_total_gasto)
print(paste('Coeficiente R2:', r2))

# Variância Explicada
evs <- var(pred_v1) / var(dados_teste$valor_total_gasto)
print(paste('Variância Explicada:', evs))


# - O coeficiente R2 de aproximadamente 98.48% indica que nosso modelo de regressão linear é muito bom.
#   Ele é capaz de explicar quase toda a variação nos dados.
# - A variância explicada de 100.98% sugere que nosso modelo está superestimando a variância dos dados de teste, o que pode indicar um ligeiro overfitting.
# - Apesar disso, o modelo mostra uma excelente performance na previsão dos valores gastos pelos clientes.
#   Será que conseguimos melhorar essa performance com outros modelos?


## Análise de Resíduos

# Calculando os resíduos
residuos <- dados_teste$valor_total_gasto - pred_v1

# Plotando o histograma e a linha de densidade corretamente
ggplot() +
  geom_histogram(aes(x = residuos, y = after_stat(density)), bins = 40, fill = 'red', color = 'black', alpha = 0.7) +
  geom_density(aes(x = residuos), color = 'blue', linewidth = 1) +
  labs(x = 'Resíduos', y = 'Densidade') +
  theme_minimal()


# Os resíduos são aproximadamente normalmente distribuídos, o que indica um bom ajuste do modelo.


## Salvando Dados do Modelo em um Dataframe

# Salve as métricas em df_modelos
df_modelos <- rbind(df_modelos, data.frame(
  Nome_do_Modelo = 'Modelo 1 RL',
  Nome_do_Algoritmo = 'Regressão Linear (Benchmark)',
  MAE = mae,
  MSE = mse,
  RMSE = rmse,
  Coeficiente_R2 = r2,
  Variancia_Explicada = evs
))

# Visualizando Dataframe
print(df_modelos)

rm(modelo_v1_RL, coeficientes, df_coef, pred_v1, valor_minimo, valor_maximo, valor_medio, mae, mse, rmse, r2, evs, residuos)





### Modelo 2 com Regressão Ridge

## Versão 1

# - Nesta versão, criamos e treinamos um modelo de Regressão Ridge para prever o valor total gasto pelos clientes.


# Criar matriz de preditores e vetor de resposta
X <- as.matrix(dados_treino[, -ncol(dados_treino)])
y <- dados_treino$valor_total_gasto

# Criação do modelo de Regressão Ridge usando glmnet
modelo_v2_RR <- glmnet(X, y, alpha = 0)

# Escolher o melhor lambda com validação cruzada
cv_modelo_v2_RR <- cv.glmnet(X, y, alpha = 0)
cv_modelo_v2_RR

# Melhor lambda
lambda_best <- cv_modelo_v2_RR$lambda.min
lambda_best

# Visualizando coeficientes das variáveis preditoras
coeficientes <- coef(modelo_v2_RR, s = lambda_best)
df_coef <- as.data.frame(as.matrix(coeficientes))
colnames(df_coef) <- c("Coeficiente")
print(df_coef)


## Interpretação do resultado dos coeficientes das variáveis preditoras:

# - Os coeficientes indicam a magnitude e a direção da influência de cada variável preditora no valor alvo (valor total gasto).
#   Por exemplo:
#      -> tempo_cadastro_cliente          : Cada ano adicional de cadastro aumenta o valor total gasto em média em 58.24 unidades.
#      -> numero_medio_cliques_por_sessao : Cada clique adicional por sessão aumenta o valor total gasto em média em 24.00 unidades.
#      -> tempo_total_logado_app          : Cada minuto adicional logado no app aumenta o valor total gasto em média em 35.91 unidades.
#      -> tempo_total_logado_website      : Cada minuto adicional logado no website aumenta o valor total gasto em média em 0.63 unidades.

# Assumindo que todas as outras variáveis permaneçam constantes.


## Previsões

# Previsões com dados de teste
X_teste <- as.matrix(dados_teste[, -ncol(dados_teste)])
pred_v2 <- predict(modelo_v2_RR, s = lambda_best, newx = X_teste)

# Imprime as 10 primeiras previsões
print(pred_v2[1:10])

# Plot das previsões vs valores reais
ggplot() +
  geom_point(aes(x = dados_teste$valor_total_gasto, y = pred_v2), color = 'skyblue') +
  labs(x = 'Valor Real de Y', y = 'Valor Previsto de Y') +
  theme_minimal()

# A partir do gráfico de dispersão, podemos ver que há uma correlação muito forte entre os y's previstos e os y's reais nos dados do teste. 
# Isso significa que temos um modelo muito bom.


## Avaliação do Modelo

# Métricas

# Valor médio gasto pelos clientes
valor_medio <- mean(df$valor_total_gasto)
print(paste('Valor médio gasto pelos clientes:', valor_medio))

# Valor mínimo
valor_minimo <- min(df$valor_total_gasto)
print(paste('Valor mínimo gasto pelos clientes:', valor_minimo))

# Valor máximo
valor_maximo <- max(df$valor_total_gasto)
print(paste('Valor máximo gasto pelos clientes:', valor_maximo))

# MAE - Erro Médio Absoluto
mae <- mean(abs(pred_v2 - dados_teste$valor_total_gasto))
print(paste('MAE - Erro Médio Absoluto:', mae))

# MSE - Erro Quadrático Médio
mse <- mean((pred_v2 - dados_teste$valor_total_gasto)^2)
print(paste('MSE - Erro Quadrático Médio:', mse))

# RMSE - Raiz Quadrada do Erro Quadrático Médio
rmse <- sqrt(mse)
print(paste('RMSE - Raiz Quadrada do Erro Quadrático Médio:', rmse))

# - O RMSE prevê que, em média, as previsões do nosso modelo (de valores gastos) estão erradas em aproximadamente 9.74, que é um valor pequeno comparado ao
#   valor médio gasto por cliente.

# Coeficiente R2
r2 <- cor(pred_v2, dados_teste$valor_total_gasto)^2
print(paste('Coeficiente R2:', r2))

# Variância Explicada
evs <- var(pred_v2) / var(dados_teste$valor_total_gasto)
print(paste('Variância Explicada:', evs))

# - O coeficiente R2 de aproximadamente 98.48% indica que nosso modelo de regressão Ridge é muito bom.
#   Ele é capaz de explicar quase toda a variação nos dados.
# - A variância explicada de 86.94% sugere que nosso modelo está estimando bem a variância dos dados de teste, mas não tão bem quanto o R2.
# - O modelo mostra uma excelente performance na previsão dos valores gastos pelos clientes.
#   Será que conseguimos melhorar essa performance com outros modelos?

## Análise de Resíduos

# Calculando os resíduos
residuos <- dados_teste$valor_total_gasto - pred_v2

# Plotando o histograma e a linha de densidade corretamente
ggplot() +
  geom_histogram(aes(x = residuos, y = after_stat(density)), bins = 40, fill = 'red', color = 'black', alpha = 0.7) +
  geom_density(aes(x = residuos), color = 'blue', linewidth = 1) +
  labs(x = 'Resíduos', y = 'Densidade') +
  theme_minimal()

# Os resíduos são aproximadamente normalmente distribuídos, o que indica um bom ajuste do modelo.

## Salvando Dados do Modelo em um Dataframe

# Salve as métricas em df_modelos
modelo_v2 <- setNames(data.frame(
  'Modelo 2 RR', 'Regressão Ridge', mae, mse, rmse, r2, evs
), names(df_modelos))

# Adicionando o novo dataframe ao dataframe existente
df_modelos <- rbind(df_modelos, modelo_v2)

# Visualizando o dataframe resultante
print(df_modelos)

rm(modelo_v2_RR, coeficientes, df_coef, pred_v2, valor_minimo, valor_maximo, valor_medio, mae, mse, rmse, r2, evs, residuos, X, y, X_teste,
   cv_modelo_v2_RR, lambda_best, modelo_v2)





### Modelo 3 com Regressão Lasso

## Versão 1

# - Nesta versão, criamos e treinamos um modelo de Regressão Lasso para prever o valor total gasto pelos clientes.


# Criação e Treinamento do Modelo

# Criar matriz de preditores e vetor de resposta
X <- as.matrix(dados_treino[, -ncol(dados_treino)])
y <- dados_treino$valor_total_gasto

# Criação do modelo de Regressão Lasso usando glmnet
modelo_v3_LA <- glmnet(X, y, alpha = 1)

# Escolher o melhor lambda com validação cruzada
cv_modelo_v3_LA <- cv.glmnet(X, y, alpha = 1)
print(cv_modelo_v3_LA)

# Melhor lambda
lambda_best <- cv_modelo_v3_LA$lambda.min
print(lambda_best)

# Visualizando coeficientes das variáveis preditoras
coeficientes <- coef(modelo_v3_LA, s = lambda_best)
df_coef <- as.data.frame(as.matrix(coeficientes))
colnames(df_coef) <- c("Coeficiente")
print(df_coef)

## Interpretação do resultado dos coeficientes das variáveis preditoras:

# - Os coeficientes indicam a magnitude e a direção da influência de cada variável preditora no valor alvo (valor total gasto).
#   Por exemplo:
#      -> tempo_cadastro_cliente          : Cada ano adicional de cadastro aumenta o valor total gasto em média em 62.86 unidades.
#      -> numero_medio_cliques_por_sessao : Cada clique adicional por sessão aumenta o valor total gasto em média em 25.18 unidades.
#      -> tempo_total_logado_app          : Cada minuto adicional logado no app aumenta o valor total gasto em média em 37.62 unidades.
#      -> tempo_total_logado_website      : Cada minuto adicional logado no website aumenta o valor total gasto em média em 0 unidade.
# Assumindo que todas as outras variáveis permaneçam constantes.

## Previsões

# Previsões com dados de teste
X_teste <- as.matrix(dados_teste[, -ncol(dados_teste)])
pred_v3 <- predict(modelo_v3_LA, s = lambda_best, newx = X_teste)

# Imprime as 10 primeiras previsões
print(pred_v3[1:10])

# Plot das previsões vs valores reais
ggplot() +
  geom_point(aes(x = dados_teste$valor_total_gasto, y = pred_v3), color = 'green') +
  labs(x = 'Valor Real de Y', y = 'Valor Previsto de Y') +
  theme_minimal()

# A partir do gráfico de dispersão, podemos ver que há uma correlação muito forte entre os y's previstos e os y's reais nos dados do teste. 
# Isso significa que temos um modelo muito bom.

## Avaliação do Modelo

# Métricas

# Valor médio gasto pelos clientes
valor_medio <- mean(df$valor_total_gasto)
print(paste('Valor médio gasto pelos clientes:', valor_medio))

# Valor mínimo
valor_minimo <- min(df$valor_total_gasto)
print(paste('Valor mínimo gasto pelos clientes:', valor_minimo))

# Valor máximo
valor_maximo <- max(df$valor_total_gasto)
print(paste('Valor máximo gasto pelos clientes:', valor_maximo))

# MAE - Erro Médio Absoluto
mae <- mean(abs(pred_v3 - dados_teste$valor_total_gasto))
print(paste('MAE - Erro Médio Absoluto:', mae))

# MSE - Erro Quadrático Médio
mse <- mean((pred_v3 - dados_teste$valor_total_gasto)^2)
print(paste('MSE - Erro Quadrático Médio:', mse))

# RMSE - Raiz Quadrada do Erro Quadrático Médio
rmse <- sqrt(mse)
print(paste('RMSE - Raiz Quadrada do Erro Quadrático Médio:', rmse))

# Coeficiente R2
r2 <- cor(pred_v3, dados_teste$valor_total_gasto)^2
print(paste('Coeficiente R2:', r2))

# Variância Explicada
evs <- var(pred_v3) / var(dados_teste$valor_total_gasto)
print(paste('Variância Explicada:', evs))

# - O coeficiente R2 de aproximadamente 98.11% indica que nosso modelo de regressão Lasso é muito bom.
#   Ele é capaz de explicar quase toda a variação nos dados.
# - A variância explicada de 98.16% sugere que nosso modelo está estimando bem a variância dos dados de teste.
# - O modelo mostra uma excelente performance na previsão dos valores gastos pelos clientes.
#   Será que conseguimos melhorar essa performance com outros modelos?

## Análise de Resíduos

# Calculando os resíduos
residuos <- dados_teste$valor_total_gasto - pred_v3

# Plotando o histograma e a linha de densidade corretamente
ggplot() +
  geom_histogram(aes(x = residuos, y = after_stat(density)), bins = 40, fill = 'red', color = 'black', alpha = 0.7) +
  geom_density(aes(x = residuos), color = 'blue', linewidth = 1) +
  labs(x = 'Resíduos', y = 'Densidade') +
  theme_minimal()

# Os resíduos são aproximadamente normalmente distribuídos, o que indica um bom ajuste do modelo.

## Salvando Dados do Modelo em um Dataframe

# Salve as métricas em df_modelos
modelo_v3 <- setNames(data.frame(
  'Modelo 3 LA', 'Regressão Lasso', mae, mse, rmse, r2, evs
), names(df_modelos))

# Adicionando o novo dataframe ao dataframe existente
df_modelos <- rbind(df_modelos, modelo_v3)

# Visualizando o dataframe resultante
print(df_modelos)

rm(modelo_v3_LA, coeficientes, df_coef, pred_v3, valor_minimo, valor_maximo, valor_medio, mae, mse, rmse, r2, evs, residuos, X, y, X_teste,
   cv_modelo_v3_LA, lambda_best, modelo_v3)



#### Seleção do Melhor Modelo

# Visualizando Modelos
df_modelos

# Seleção do Melhor Modelo (criterio: menor valor de RMSE)
melhor_modelo <- df_modelos[which.min(df_modelos$RMSE),]

# Exibindo o melhor modelo
print(melhor_modelo)


## Explicação da Escolha do Melhor Modelo

# - O Modelo 2 apresentou uma taxa de erro (RMSE) levemente maior e pode ser descartado.
# - A análise comparativa dos modelos de regressão nos mostrou que o Modelo 3 - Regressão Lasso teve o menor RMSE (9.313104), indicando que ele é ligeiramente
#   melhor que os outros modelos em termos de previsão.
# - No entanto, considerando a simplicidade e a performance praticamente equivalente, optamos pelo Modelo 1 - Regressão Linear como o melhor modelo, devido à
#   sua simplicidade e facilidade de interpretação.

#  -> Com base nas explicações acima, nós iremos escolher o Modelo 1 RL.


# Criação do modelo de Regressão Linear
modelo_v1_RL <- train(valor_total_gasto ~ ., data = dados_treino, method = "lm")

# Visualizando coeficientes das variáveis preditoras
coeficientes <- summary(modelo_v1_RL$finalModel)$coefficients
df_coef <- as.data.frame(coeficientes)
colnames(df_coef) <- c("Coeficiente", "Erro Padrão", "Valor T", "P-valor")
df_coef


## Interpretação do resultado dos coeficientes das variáveis preditoras:

# - Os coeficientes indicam a magnitude e a direção da influência de cada variável preditora no valor alvo (valor total gasto).
#   Por exemplo:
#      -> tempo_cadastro_cliente          : Cada ano adicional de cadastro aumenta o valor total gasto em média em 62.77 unidades.
#      -> numero_medio_cliques_por_sessao : Cada clique adicional por sessão aumenta o valor total gasto em média em 25.85 unidades.
#      -> tempo_total_logado_app          : Cada minuto adicional logado no app aumenta o valor total gasto em média em 38.69 unidades.
#      -> tempo_total_logado_website      : Cada minuto adicional logado no website aumenta o valor total gasto em média em 0.47 unidades.

# Assumindo que todas as outras variáveis permaneçam constantes.

rm(melhor_modelo, modelo_v1_RL, coeficientes, df_coef)




#### Respondendo Problema de Negócio

## Com base nos coeficientes do melhor modelo de regressão, podemos inferir que:
  
# - Tempo Logado no App: O tempo logado no app tem a maior influência positiva no valor total gasto pelo cliente. Cada minuto adicional logado no app resulta em
#   um aumento médio de 38.57 unidades no valor total gasto. Portanto, investir em melhorias no app para aumentar o tempo logado dos clientes pode ser uma
#   estratégia eficaz para aumentar as vendas.

# - Número Médio de Cliques por Sessão: A segunda maior influência positiva é o número médio de cliques por sessão. Cada clique adicional por sessão aumenta o
#   valor total gasto em média em 26.24 unidades. Isso sugere que melhorar a usabilidade e a experiência do usuário para facilitar a navegação e incentivar
#   cliques pode também aumentar as vendas.

# - Tempo Logado no Website: Embora tenha uma influência positiva, o impacto do tempo logado no website é muito menor (0.68 unidades por minuto logado).
#   Isso indica que, em comparação ao app, o website tem uma influência significativamente menor no valor total gasto pelos clientes.

# - Tempo de Cadastro do Cliente: Clientes com mais tempo de cadastro tendem a gastar mais, com cada ano adicional de cadastro aumentando o valor total gasto
#   em 63.74 unidades. Isso pode indicar a importância de manter clientes a longo prazo e a eficácia de estratégias de fidelização.




#### Recomendação:

## Dado que o tempo logado no app tem o maior impacto no valor total gasto pelo cliente, a empresa deve considerar investir em melhorias no aplicativo, como:
  
#  -> Melhorar a usabilidade e a interface do app para tornar a experiência do usuário mais intuitiva e agradável.
#  -> Adicionar funcionalidades que incentivem os usuários a passar mais tempo no app.
#  -> Implementar estratégias de engajamento, como notificações push personalizadas, para manter os usuários ativos no app.
#  -> Analisar e otimizar o conteúdo e as ofertas disponíveis no app para maximizar a interação e o engajamento do usuário.

# Ao focar nessas áreas, a empresa pode aumentar o tempo logado dos clientes no app, o que, por sua vez, deverá aumentar o valor total gasto e, consequentemente,
# as vendas.

