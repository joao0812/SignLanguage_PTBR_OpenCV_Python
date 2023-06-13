import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PICKLE_DIR = './dataSets/dataSetHalfBodyHand.pickle'

data_dict = pickle.load(open(DATA_PICKLE_DIR, 'rb'))

data = data_dict['data']
labels = np.asarray(data_dict['labels'])

df = pd.DataFrame(data)
df = df.loc[:, :41]
#print(df[:5])

# Colocar as variáveis do train_test_split em um .pkl
x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.25, shuffle=True, stratify=labels)

print(x_train.shape)
print(y_train.shape)

model_mlp = MLPClassifier(max_iter=900, verbose=True, tol=0.0000100, hidden_layer_sizes=(26,26))
#verbose=False (valor default): Nenhum detalhe será impresso durante o treinamento.

#verbose=True: Informações de progresso serão impressas durante o treinamento, como o número de iterações e o valor da função de perda.

# tol: valor de tolerância de perdas aceita, por default o valor é de 0.000100

#hidden_layer_sizes (default: 100): Esse parâmetro define a arquitetura da rede neural especificando o número de neurônios em cada camada oculta. Pode ser uma tupla que representa o número de neurônios em cada camada ou uma única inteiro para especificar o mesmo número de neurônios em todas as camadas ocultas. -> hidden_layer_sizes = (Camada de entrada - Camada de saída) /2 (inicialmente se utiliza essa fórmula como ponto de partida, mas não é uma regra)

#activation: Esse parâmetro define a função de ativação a ser usada nas camadas ocultas da MLP. As opções incluem "identity" (função identidade), "logistic" (função sigmoide logística), "tanh" (função tangente hiperbólica) e "relu" (função de ativação retificada linear - valor Default).

#solver: Esse parâmetro define o algoritmo de otimização usado para ajustar os pesos e viéses da MLP. Pode ser "adam" (que usa o otimizador Adam - é o valor Default), "lbfgs" (que usa o algoritmo de otimização de Quase-Newton limitado), ou "sgd" (que usa o gradiente descendente estocástico).

#alpha: Esse parâmetro controla a regularização L2 (termo de penalidade) aplicada aos pesos da rede para evitar o sobreajuste (overfitting). Quanto maior o valor de alpha, mais forte é a regularização.

#learning_rate: Esse parâmetro controla a taxa de aprendizado usada no algoritmo de otimização. Pode ser "constant" (taxa de aprendizado constante), "invscaling" (taxa de aprendizado que diminui gradualmente ao longo do tempo) ou "adaptive" (taxa de aprendizado adaptativa que ajusta automaticamente com base na estabilidade da perda).

#max_iter: Esse parâmetro define o número máximo de iterações (épocas) que o algoritmo de otimização será executado durante o treinamento.

#random_state: Esse parâmetro define a semente utilizada pelo gerador de números aleatórios para garantir a reprodutibilidade dos resultados.

model_mlp.fit(x_train, y_train)

y_predict = model_mlp.predict(x_test)

score = accuracy_score(y_test, y_predict)
print(f'{round((score*100), 2)}% de precisão') 

# Mostra a matrix confusão do modelo MLP gerado
#cm = confusion_matrix(y_test, y_predict, labels=model_mlp.classes_)
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels= model_mlp.classes_)
#disp.plot()
#plt.show()

#print(classification_report(y_test, y_predict))

f = open(f"./models/MPLClassifier.p", 'wb')
pickle.dump({'model_mlp': model_mlp}, f) # Salvando o modelo gerado
f.close()