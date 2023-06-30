import os, pickle

import numpy as np
import pandas as pd
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#DATA_PICKLE_DIR = './dataSets/data.pickle'
DATA_PICKLE_DIR = './dataSets/dataSetHalfBodyHand.pickle'

data_dict = pickle.load(open(DATA_PICKLE_DIR, 'rb'))

#print(data_dict.keys())
col = []
# Convertendo os dados em um array com numpy
data_list = data_dict['data']
data = np.asarray([np.asarray(d) for d in data_dict['data']])
labels = np.asarray(data_dict['labels'])

for i in range(len(data[0])//2):
    col.append(f'x{i}')
    col.append(f'y{i}')

#print(len(data_list[0]))

df = pd.DataFrame(data_list)
df = df.loc[:, :41]
print(df)

#print(data.shape)
#print(labels)

# Agora vamos dividir nossos dados em dados para treinamento e dados para teste. Se test_size = 0.25, significa que 75% dos dados serão usados para treino e 25% para teste, para testar esse treinamento, x são os dados/features e y as classes/target
x_train, x_test, y_train, y_test = train_test_split(df, labels, test_size=0.25, shuffle=True, stratify=labels)
# shuffle: Um booleano que indica se os dados devem ser embaralhados antes da divisão ou não. O padrão é True, o que significa que os dados são embaralhados.
# stratify: Um array ou série contendo os rótulos/targets. Se especificado, a divisão será estratificada, ou seja, a proporção de rótulos em cada conjunto (treinamento e teste) será a mesma que a proporção de rótulos/classes no conjunto de dados completo.
#print(x_train)
#print(y_train)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test) # 99.58% de precisão

print(f'{round((score*100), 2)}% de precisão') 

f = open('./models/RandomForest.p', 'wb')
pickle.dump({'model': model}, f) # Salvando o modelo gerado
f.close()