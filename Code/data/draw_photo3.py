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

location = []
for i in range(102):
    path = "KAIST/KAIST_30sec_%03d.txt"%(int(i))
    data = get_data_collection(path)
    # location.append([data[0, 1], data[0, 2]])
    location.append([data[145, 1], data[145, 2]])
print(location)


# rs = np.random.RandomState(2)  # 设定随机数种子
# df = pd.DataFrame(rs.randn(100,2),
#                  columns = ['A','B'])


df = pd.DataFrame(location,
                 columns = [r'$u^{x}_{n}(t)$',r'$u^{y}_{n}(t)$'])

sns.kdeplot(df[r'$u^{x}_{n}(t)$'],df[r'$u^{y}_{n}(t)$'],
           cbar = True,    # 是否显示颜色图例
           shade = True,   # 是否填充
           # cmap = 'Reds',  # 设置调色盘
            cmap = 'Blues',
           shade_lowest=False,  # 最外围颜色是否显示
           n_levels = 10   # 曲线个数（如果非常多，则会越平滑）
           )
# 两个维度数据生成曲线密度图，以颜色作为密度衰减显示

# sns.rugplot(df['A'], color="g", axis='x',alpha = 0.5)
# sns.rugplot(df['B'], color="r", axis='y',alpha = 0.5)


sns.rugplot(df[r'$u^{x}_{n}(t)$'], axis='x',alpha = 0.5)
sns.rugplot(df[r'$u^{y}_{n}(t)$'], axis='y',alpha = 0.5)
plt.savefig('./posation2.png')
plt.show()