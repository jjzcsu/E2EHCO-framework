import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_data_collection(path):
    input_data = []
    with open(path, "r") as f:
        for line in f.readlines():
            temp_data = []
            line = line.strip('\n')
            data = line.split(' ')
            try:
                temp_data.append(float(data[2].strip("\t")))
                temp_data.append(float(data[3].strip("\t")))
                if data[4] == '':
                    temp_data.append(float(data[5].strip("\t")))
                else:
                    temp_data.append(float(data[4].strip("\t")))
                input_data.append(temp_data)
            except:
                continue
    f.close()
    input_data_1 = np.array(input_data)
    return input_data_1

location_1 = []
for i in range(102):
    path = "KAIST/KAIST_30sec_%03d.txt"%(int(i))
    data = get_data_collection(path)
    # location.append([data[0, 1], data[0, 2]])
    location_1.append([data[0, 1], data[0, 2]])
print(location_1)

location_2 = []
for i in range(102):
    path = "KAIST/KAIST_30sec_%03d.txt"%(int(i))
    data = get_data_collection(path)
    # location.append([data[0, 1], data[0, 2]])
    location_2.append([data[145, 1], data[145, 2]])
print(location_2)

df1 = pd.DataFrame(location_1, columns = ['A','B'])
df2 = pd.DataFrame(location_2, columns = ['A','B'])

# sns.kdeplot(df1['A'],df1['B'],cmap = 'Greens',
#             shade = True,shade_lowest=False)
# sns.kdeplot(df2['A'],df2['B'],cmap = 'Blues',
#             shade = True,shade_lowest=False)


sns.kdeplot(df1['A'],df1['B'],cmap = 'Greens',
            shade = False, shade_lowest=False)
sns.kdeplot(df2['A'],df2['B'],cmap = 'Reds',
            shade = False, shade_lowest=False)

# rs1 = np.random.RandomState(2)
# rs2 = np.random.RandomState(5)
# df1 = pd.DataFrame(rs1.randn(100,2)+2,columns = ['A','B'])
# df2 = pd.DataFrame(rs2.randn(100,2)-2,columns = ['A','B'])
# # 创建数据
#
# sns.kdeplot(df1['A'],df1['B'],cmap = 'Greens',
#             shade = True,shade_lowest=False)
# sns.kdeplot(df2['A'],df2['B'],cmap = 'Blues',
#             shade = True,shade_lowest=False)

plt.show()