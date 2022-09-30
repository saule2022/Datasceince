import pandas as pd
import pathlib

filepath = pathlib.Path(r'Users/daivadaugelaite/pythonProject/messages/spamCollection.csv')
df1 = pd.read_csv(r'Users/daivadaugelaite/pythonProject/messages/spamCollection.csv',
                  encoding='utf-8',
                  sep='\t',
                  header=None,
                  names=['label', 'text'])
with open(filepath.as_posix(), 'r', encoding='utf-8') as f:
    data = f.read().split('\n')

tag = [row.split('\t')[0] for row in data]
sms = [row.split('\t')[1] for row in data]
data2 = [row.split('\t') for row in data]
df = pd.DataFrame(data=data2, colums=['label', 'text'])