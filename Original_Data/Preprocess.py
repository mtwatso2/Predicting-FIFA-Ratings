# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:57:32 2020

@author: MWatson717
"""

import pandas as pd

standard = pd.read_csv('standardStats.csv')

standard.isnull().sum()

st2 = standard[standard.Pos != 'GK']

st2.isnull().sum()

st2.loc[135]

st2.loc[135, 'Age'] = 15
st2.loc[135, 'Born'] = 2013

st2 = st2.dropna()


shooting = pd.read_csv('shooting.csv')

shooting.isnull().sum()

sh2 = shooting[shooting.Pos != 'GK']

sh2.isnull().sum().sum()

sh2 = sh2.fillna({'SoT%':0, 'G/Sh':0, 'G/SoT':0})

sh2 = sh2.dropna()


st2.columns

sh2.columns

cols = ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Gls', 'PK', 'PKatt']

data = pd.merge(left = st2, right = sh2, how = "left", left_on = cols, right_on = cols)

data.isnull().sum()


passing = pd.read_csv('passing.csv')

passing.isnull().sum()

p2 = passing[passing.Pos != 'GK']

p2.isnull().sum()

p2 = p2.drop(['Cmp%', 'Cmp%_S', 'Cmp%_M', 'Cmp%_L'], axis = 1)

p2.isnull().sum().sum()


data.columns

p2.columns

cols2 = ['Rk', 'Player', 'Nation', 'Pos', 'Squad', '90s', 'Ast']

data = pd.merge(left = data, right = p2, how = "left", left_on = cols2, right_on = cols2)

data.isnull().sum().sum()


misc = pd.read_csv('misc.csv')

misc.isnull().sum()

m2 = misc[misc.Pos != 'GK']

m2.isnull().sum()

m2 = m2.drop(['Succ%', 'Tkl%'], axis = 1)

m2.isnull().sum().sum()


data.columns

m2.columns

cols3 = ['Rk', 'Player', 'Nation', 'Pos', 'Squad', '90s', 'CrdY', 'CrdR']

data = pd.merge(left = data, right = m2, how = 'left', left_on = cols3, right_on = cols3)

data.isnull().sum().sum()


data['Player'] = data['Player'].str.split('\\').str[0]



fifa19 = pd.read_csv('Fifa19Ranks.csv')

fifa19 = fifa19[['Name', 'Rtg']]

fifa19 = fifa19.rename(columns = {'Name': 'Player', 'Rtg': 'Rating19'})

data = pd.merge(left = data, right = fifa19, how = 'left', on = 'Player')

data.isnull().sum().sum()



fifa20 = pd.read_csv('fifa20.csv')

fifa20 = fifa20[['Name', 'Rtg']]

fifa20_EFL = pd.read_csv('fifa20_2.csv')

fifa20_EFL = fifa20_EFL[['Name', 'Rtg']]

fifa20 = pd.concat([fifa20, fifa20_EFL], axis = 0)

fifa20 = fifa20.rename(columns = {'Name': 'Player', 'Rtg': 'Rating20'})


data = pd.merge(left = data, right = fifa20, how = 'left', on = 'Player')

data.isnull().sum()

na = data[data['Rating19'].isnull()].index.tolist()
na

data.iloc[[26]]
data.loc[26, 'Rating19'] = 86

data.iloc[[34]]
data.loc[34, 'Rating19'] = 75

data.iloc[[56]]
data.loc[56, 'Rating19'] = 74

data.iloc[[58]]
data.loc[58, 'Rating19'] = 74

data.iloc[[63]]
data.loc[63, 'Rating19'] = 76

data.iloc[[64]]
data.loc[64, 'Rating19'] = 77

data.iloc[[66]]
data.loc[66, 'Rating19'] = 79

data.iloc[[107]]
data.loc[107, 'Rating19'] = 84

data.iloc[[110]]
data.loc[110, 'Rating19'] = 81

data.iloc[[114]]
data.loc[114, 'Rating19'] = 65

data.iloc[[125]]
data.loc[125, 'Rating19'] = 61

data.iloc[[129]]
data.loc[129, 'Rating19'] = 74

data.iloc[[130]]
data.loc[130, 'Rating19'] = 59

data.iloc[[154]]
data.loc[154, 'Rating19'] = 82

data.iloc[[167]]
data.loc[167, 'Rating19'] = 66

data.iloc[[176]]                                    #Rhys Healey: Not in FUT, but in Career mode (used Futwiz)
data.loc[176, 'Rating19'] = 63

data.iloc[[181]]
data.loc[181, 'Rating19'] = 84

data.iloc[[187]]
data.loc[187, 'Rating19'] = 77

data.iloc[[192]]
data.loc[192, 'Rating19'] = 76

data.iloc[[194]]                                    #listed as Vincente Iborra, only as Iborra on Fifa
data.loc[194, 'Rating19'] = 81

data.iloc[[207]]
data.loc[207, 'Rating19'] = 56

data.iloc[[239]]
data.loc[239, 'Rating19'] = 80                      #Accent on E not present in fifa for Erik Lamela

data.iloc[[252]]
data.loc[252, 'Rating19'] = 77

data.iloc[[271]]
data.loc[271, 'Rating19'] = 74

data.iloc[[282]]
data.loc[282, 'Rating19'] = 81

data.iloc[[304]]
data.loc[304, 'Rating19'] = 77

data.iloc[[306]]
data.loc[306, 'Rating19'] = 83

data.iloc[[312]]
data.loc[312, 'Rating19'] = 82                      #Lucas Moura, goes by Lucas in FIFA

data.iloc[[320]]
data.loc[320, 'Rating19'] = 76

data.iloc[[321]]
data.loc[321, 'Rating19'] = 82

data.iloc[[330]]
data.loc[330, 'Rating19'] = 64                      # Edward Nketiah vs Eddie 

data.iloc[[343]]
data.loc[343, 'Rating19'] = 76

data.iloc[[344]]
data.loc[344, 'Rating19'] = 84

data.iloc[[365]]
data.loc[365, 'Rating19'] = 60

data.iloc[[372]]
data.loc[372, 'Rating19'] = 69

data.iloc[[383]]
data.loc[383, 'Rating19'] = 58

data.iloc[[387]]
data.loc[387, 'Rating19'] = 65

data.iloc[[389]]
data.loc[389, 'Rating19'] = 71

data.iloc[[401]]
data.loc[401, 'Rating19'] = 82

data.iloc[[421]]
data.loc[421, 'Rating19'] = 77

data.iloc[[435]]
data.loc[435, 'Rating19'] = 76

data.iloc[[449]]
data.loc[449, 'Rating19'] = 75

data.iloc[[476]]
data.loc[476, 'Rating19'] = 61

data.iloc[[489]]
data.loc[489, 'Rating19'] = 75



na = data[data['Rating20'].isnull()].index.tolist()
na[:11]

data.iloc[[11]]
data.loc[11, 'Rating20'] = 65

data.iloc[[14]]
data.loc[14, 'Rating20'] = 76

data.iloc[[17]]
data.loc[17, 'Rating20'] = 82

data.iloc[[23]]
data.loc[23, 'Rating20'] = 75

data.iloc[[25]]
data.loc[25, 'Rating20'] = 69

data.iloc[[26]]
data.loc[26, 'Rating20'] = 78

data.iloc[[27]]
data.loc[27, 'Rating20'] = 78

data.iloc[[34]]
data.loc[34, 'Rating20'] = 76

data.iloc[[39]]
data.loc[39, 'Rating20'] = 74

data.iloc[[53]]
data.loc[53, 'Rating20'] = 72

data.iloc[[56]]                             #Retired after 18-19 season

data.iloc[[58]]                             #Retired

data.iloc[[63]]
data.loc[63, 'Rating20'] = 76

data.iloc[[64]]
data.loc[64, 'Rating20'] = 80

data.iloc[[88]]
data.loc[88, 'Rating20'] = 74

data.iloc[[90]]                             #Retired

data.iloc[[94]]
data.loc[94, 'Rating20'] = 68

data.iloc[[96]]
data.loc[96, 'Rating20'] = 79

data.iloc[[98]]
data.loc[98, 'Rating20'] = 75

data.iloc[[101]]
data.loc[101, 'Rating20'] = 74

data.iloc[[104]]
data.loc[104, 'Rating20'] = 76

data.iloc[[105]]
data.loc[105, 'Rating20'] = 78

data.iloc[[107]]
data.loc[107, 'Rating20'] = 81

data.iloc[[109]]
data.loc[109, 'Rating20'] = 73

data.iloc[[110]]
data.loc[110, 'Rating20'] = 80

data.iloc[[111]]
data.loc[111, 'Rating20'] = 70

data.iloc[[113]]                        #Plays in qatar: not in FIFA

data.iloc[[114]]
data.loc[114, 'Rating20'] = 68

data.iloc[[128]]
data.loc[128, 'Rating20'] = 74

data.iloc[[129]]
data.loc[129, 'Rating20'] = 74

data.iloc[[130]]                        #Not in FIFA 20, drop

data.iloc[[131]]                       
data.loc[131, 'Rating20'] = 77

data.iloc[[132]]
data.loc[132, 'Rating20'] = 76

data.iloc[[136]]
data.loc[136, 'Rating20'] = 81

data.iloc[[137]]
data.loc[137, 'Rating20'] = 76

data.iloc[[152]]
data.loc[152, 'Rating20'] = 76

data.iloc[[153]]                       
data.loc[153, 'Rating20'] = 68

data.iloc[[154]]
data.loc[154, 'Rating20'] = 83

data.iloc[[155]]
data.loc[155, 'Rating20'] = 74

data.iloc[[171]]                        #Plays in QATAR, not in FIFA

data.iloc[[173]]
data.loc[173, 'Rating20'] = 68

data.iloc[[175]]                       
data.loc[175, 'Rating20'] = 91

data.iloc[[176]]
data.loc[176, 'Rating20'] = 63

data.iloc[[179]]
data.loc[179, 'Rating20'] = 78

data.iloc[[180]]
data.loc[180, 'Rating20'] = 82

data.iloc[[181]]                       
data.loc[181, 'Rating20'] = 87

data.iloc[[182]]
data.loc[182, 'Rating20'] = 85

data.iloc[[183]]
data.loc[183, 'Rating20'] = 75

data.iloc[[187]]
data.loc[187, 'Rating20'] = 78

data.iloc[[191]]
data.loc[191, 'Rating20'] = 66

data.iloc[[192]]                       
data.loc[192, 'Rating20'] = 77

data.iloc[[194]]
data.loc[194, 'Rating20'] = 78

data.iloc[[202]]
data.loc[202, 'Rating20'] = 75

data.iloc[[203]]
data.loc[203, 'Rating20'] = 70

data.iloc[[210]]
data.loc[210, 'Rating20'] = 74

data.iloc[[212]]                       
data.loc[212, 'Rating20'] = 76

data.iloc[[224]]
data.loc[224, 'Rating20'] = 75

data.iloc[[225]]
data.loc[225, 'Rating20'] = 73

data.iloc[[229]]
data.loc[229, 'Rating20'] = 83

data.iloc[[232]]
data.loc[232, 'Rating20'] = 81

data.iloc[[235]]                        #Plays in Serbia, not in FIFA    

data.iloc[[239]]
data.loc[239, 'Rating20'] = 80

data.iloc[[246]]
data.loc[246, 'Rating20'] = 78

data.iloc[[249]]
data.loc[249, 'Rating20'] = 75

data.iloc[[252]]
data.loc[252, 'Rating20'] = 76

data.iloc[[253]]
data.loc[253, 'Rating20'] = 76

data.iloc[[258]]
data.loc[258, 'Rating20'] = 76

data.iloc[[260]]
data.loc[260, 'Rating20'] = 72

data.iloc[[263]]
data.loc[263, 'Rating20'] = 85

data.iloc[[271]]
data.loc[271, 'Rating20'] = 74

data.iloc[[274]]                        #Plays in Serbia, not in FIFA

data.iloc[[275]]                        #Duplicated somehow, remove as well

data.iloc[[282]]
data.loc[282, 'Rating20'] = 82

data.iloc[[302]]
data.loc[302, 'Rating20'] = 81

data.iloc[[303]]
data.loc[303, 'Rating20'] = 79

data.iloc[[304]]
data.loc[304, 'Rating20'] = 76

data.iloc[[306]]
data.loc[306, 'Rating20'] = 83

data.iloc[[307]]
data.loc[307, 'Rating20'] = 76

data.iloc[[310]]
data.loc[310, 'Rating20'] = 78

data.iloc[[315]]
data.loc[315, 'Rating20'] = 71

data.iloc[[320]]
data.loc[320, 'Rating20'] = 74

data.iloc[[321]]
data.loc[321, 'Rating20'] = 78

data.iloc[[322]]
data.loc[322, 'Rating20'] = 73

data.iloc[[332]]
data.loc[332, 'Rating20'] = 75

data.iloc[[334]]
data.loc[334, 'Rating20'] = 77

data.iloc[[337]]
data.loc[337, 'Rating20'] = 75

data.iloc[[338]]
data.loc[338, 'Rating20'] = 73

data.iloc[[343]]
data.loc[343, 'Rating20'] = 77

data.iloc[[344]]
data.loc[344, 'Rating20'] = 84

data.iloc[[352]]
data.loc[352, 'Rating20'] = 77

data.iloc[[357]]
data.iloc[[358]]
data.iloc[[359]]
data.iloc[[360]]        #All the same player? Plays in cyprus, not in FIFA
 
data.iloc[[364]]
data.loc[364, 'Rating20'] = 83

data.iloc[[365]]
data.loc[365, 'Rating20'] = 61

data.iloc[[370]]
data.loc[370, 'Rating20'] = 73

data.iloc[[372]]
data.loc[372, 'Rating20'] = 69

data.iloc[[381]]
data.loc[381, 'Rating20'] = 79

data.iloc[[385]]
data.loc[385, 'Rating20'] = 65

data.iloc[[389]]                #Plays in Cyprus, not in FIFA

data.iloc[[391]]
data.loc[391, 'Rating20'] = 82

data.iloc[[399]]
data.loc[399, 'Rating20'] = 76

data.iloc[[400]]
data.loc[400, 'Rating20'] = 72

data.iloc[[401]]
data.loc[401, 'Rating20'] = 78

data.iloc[[407]]
data.loc[407, 'Rating20'] = 78

data.iloc[[411]]
data.loc[411, 'Rating20'] = 70

data.iloc[[417]]
data.loc[417, 'Rating20'] = 80

data.iloc[[419]]
data.loc[419, 'Rating20'] = 72

data.iloc[[421]]
data.loc[421, 'Rating20'] = 76

data.iloc[[422]]                        #Plays in Egypt, not in FIFA

data.iloc[[432]]
data.loc[432, 'Rating20'] = 77

data.iloc[[433]]
data.loc[433, 'Rating20'] = 78

data.iloc[[435]]
data.loc[435, 'Rating20'] = 76

data.iloc[[438]]
data.loc[438, 'Rating20'] = 73

data.iloc[[450]]
data.loc[450, 'Rating20'] = 80

data.iloc[[451]]                        #Plays in Ecuador, not in FIFA

data.iloc[[456]]
data.loc[456, 'Rating20'] = 74

data.iloc[[458]]
data.loc[458, 'Rating20'] = 73

data.iloc[[468]]
data.loc[468, 'Rating20'] = 72

data.iloc[[474]]                    #Plays in Cyprus, not in FIFA

data.iloc[[476]]
data.loc[476, 'Rating20'] = 67

data.iloc[[477]]
data.loc[477, 'Rating20'] = 60

data.iloc[[489]]
data.loc[489, 'Rating20'] = 73

data.iloc[[490]]
data.loc[490, 'Rating20'] = 79

data.iloc[[492]]
data.loc[492, 'Rating20'] = 71

na = data[data['Rating20'].isnull()].index.tolist()



data = data.drop(data.index[na])

data = data.reset_index()
data = data.drop(['index'], axis = 1)

data.isnull().sum().sum()


data['Player'].describe()


data = data.drop_duplicates(subset = 'Player')

varsToDrop = ['Rk', 'Born', 'Gls.1', 'Ast.1', 'G+A', 'G-PK', 'G+A-PK', '90s', 'SoT%', 'Sh/90', 'SoT/90', 'G/Sh', 'G/SoT', 'Cmp', 'Att_x']


data = data.drop(varsToDrop, axis = 1)

data.columns

data = data.rename(columns = {'Att_y': 'Dr_Att', 'Att.1': 'D_Cont'})

import matplotlib.pyplot as plt

plt.scatter('Rating19', 'Rating20', data = data)
plt.plot('Rating19', 'Rating19', data = data, color = 'red')
plt.title("Rating in FIFA 19 vs Rating in FIFA 20")
plt.xlabel("Rating in FIFA 19")
plt.ylabel("Rating in FIFA 20")


plt.scatter('Age', 'Rating19', data = data)
plt.title("Age in FIFA 19 vs Rating in FIFA 20")
plt.xlabel("Age in FIFA 19")
plt.ylabel("Rating in FIFA 19")

import seaborn as sn

corrMatrix = data.corr()
sn.heatmap(corrMatrix)


data['Pos'].describe()

data['Pos'].value_counts()


positions = ['FWMF', 'FWDF', 'MFFW', 'MFDF', 'DFMF', 'DFFW']
new = ['FW', 'FW', 'MF', 'MF', 'DF', 'DF']

data = data.replace(positions, new)

data['Pos'].value_counts()

data.boxplot(column = 'Rating19', by = 'Pos')
plt.title("Boxplot of Rating 19 by Position")
plt.suptitle("")
plt.xlabel("Position")
plt.ylabel("Rating")
plt.show()

data.boxplot(column = 'Rating20', by = 'Pos')
plt.title("Boxplot of Rating 20 by Position")
plt.suptitle("")
plt.xlabel("Position")
plt.ylabel("Rating")
plt.show()

import matplotlib.pyplot as plt
from pandas.plotting import table
desc = data['Rating20'].describe()
plot = plt.subplot(111, frame_on=False)
plot.xaxis.set_visible(False) 
plot.yaxis.set_visible(False) 
table(plot, desc,loc='upper right')
plt.savefig('rating20.png')


varsdrop = ['Player', 'Nation', 'Pos', 'Squad']

data2 = data.drop(varsdrop, axis = 1)

fifaR = data2

colsR = fifaR.columns.to_list()

colsR = colsR[-1:] + colsR[:-1]

fifaR = fifaR[colsR]

fifaR.to_csv('fifaR.csv', index = False)

fifaC = data2

fifaC['Dif'] = fifaC['Rating20'] < fifaC['Rating19']

fifaC['Dif'] *= 1

fifaC['Dif'].value_counts().plot(kind = 'bar')

fifaC = fifaC.drop(['Rating20'], axis = 1)

colsC = fifaC.columns.to_list()

colsC = colsC[-1:] + colsC[:-1]

fifaC = fifaC[colsC]

fifaC.to_csv('fifaC.csv', index = False)




