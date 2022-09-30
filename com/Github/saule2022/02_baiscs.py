import pandas as pd
import pathlib

dir_path = pathlib.Path(r'C:\DATA\Hyarchis_DS_Academy\02\Tennis-Major-Tournaments-Match-Statistics')
proj_path = pathlib.Path(r'C:\Users\ArturasKatvickis\PycharmProjects\HyaDS\intro\sandbox')

name_map = dict()
with open(proj_path.joinpath('players_name_map.csv').as_posix(), 'r') as f:
    data = f.read().split('\n')
for row in data[1:]:
    names = row.split(',')
    name_map[names[0]] = names[1]


def map_names(name):
    new_name = name
    if name in name_map:
        new_name = name_map[name]
    return new_name


columns_names=['Player1', 'Player2', 'Round', 'Result', 'FNL1', 'FNL2', 'FSP.1', 'FSW.1', 'SSP.1', 'SSW.1',
               'ACE.1', 'DBF.1', 'WNR.1', 'UFE.1', 'BPC.1', 'BPW.1', 'NPA.1', 'NPW.1', 'TPW.1', 'ST1.1', 'ST2.1',
               'ST3.1', 'ST4.1', 'ST5.1', 'FSP.2', 'FSW.2', 'SSP.2', 'SSW.2', 'ACE.2', 'DBF.2', 'WNR.2', 'UFE.2',
               'BPC.2', 'BPW.2', 'NPA.2', 'NPW.2', 'TPW.2', 'ST1.2', 'ST2.2', 'ST3.2', 'ST4.2', 'ST5.2']

df = pd.DataFrame(columns=columns_names)
for file in dir_path.glob('*.csv'):
    df_temp = pd.read_csv(file.as_posix(), header=0, names=columns_names, skiprows=1)
    df_temp['File'] = file.name
    print(file.name)
    df_temp['P1'] = df_temp['Player1'].apply(map_names)
    df_temp['P2'] = df_temp['Player2'].apply(map_names)
    df = df.append(df_temp, ignore_index=True)

df1 = df[['File', 'P1', 'FSP.1', 'FSW.1', 'SSP.1', 'SSW.1', 'ACE.1',
          'DBF.1', 'WNR.1', 'UFE.1', 'BPC.1', 'BPW.1', 'NPA.1', 'NPW.1',
          'TPW.1', 'ST1.1', 'ST2.1', 'ST3.1', 'ST4.1', 'ST5.1']]\
    .rename(columns = {'P1' : 'Player', 'FSP.1' : 'FSP', 'FSW.1' : 'FSW',
                       'SSP.1' : 'SSP', 'SSW.1' : 'SSW', 'ACE.1' : 'ACE', 'DBF.1' : 'DBF',
                       'WNR.1' : 'WNR', 'UFE.1' : 'UFE', 'BPC.1' : 'BPC', 'BPW.1' : 'BPW',
                       'NPA.1' : 'NPA', 'NPW.1' : 'NPW', 'TPW.1' : 'TPW', 'ST1.1' : 'ST1',
                       'ST2.1' : 'ST2', 'ST3.1' : 'ST3', 'ST4.1' : 'ST4', 'ST5.1' : 'ST5'})
df2 = df[['File', 'P2', 'FSP.2', 'FSW.2', 'SSP.2', 'SSW.2', 'ACE.2',
          'DBF.2', 'WNR.2', 'UFE.2', 'BPC.2', 'BPW.2', 'NPA.2', 'NPW.2',
          'TPW.2', 'ST1.2', 'ST2.2', 'ST3.2', 'ST4.2', 'ST5.2']]\
    .rename(columns = {'P2' : 'Player', 'FSP.2' : 'FSP', 'FSW.2' : 'FSW',
                       'SSP.2' : 'SSP', 'SSW.2' : 'SSW', 'ACE.2' : 'ACE', 'DBF.2' : 'DBF',
                       'WNR.2' : 'WNR', 'UFE.2' : 'UFE', 'BPC.2' : 'BPC', 'BPW.2' : 'BPW',
                       'NPA.2' : 'NPA', 'NPW.2' : 'NPW', 'TPW.2' : 'TPW', 'ST1.2' : 'ST1',
                       'ST2.2' : 'ST2', 'ST3.2' : 'ST3', 'ST4.2' : 'ST4', 'ST5.2' : 'ST5'})
df = df1.append(df2, ignore_index=True)

rez = df.groupby(by='Player').mean().reset_index()
rez.fillna(value=0, inplace=True)
rez.to_csv(proj_path.joinpath('players2013_withoutNaN.csv').as_posix(), index=False)