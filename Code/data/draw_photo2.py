import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


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


# 0-1648
# # input_data = get_data_collection("KAIST/KAIST_30sec_057.txt")
# input_data = get_data_collection("KAIST/KAIST_30sec_000.txt")
# x = input_data[0:200, 1]
# y = input_data[0:200, 2]
# t = input_data[0:200, 0]
#
# # input_data1 = get_data_collection("KAIST/KAIST_30sec_009.txt")
# input_data1 = get_data_collection("KAIST/KAIST_30sec_001.txt")
# x1 = input_data1[0:200, 1]
# y1 = input_data1[0:200, 2]
# t1 = input_data1[0:200, 0]
#
# # input_data2 = get_data_collection("KAIST/KAIST_30sec_055.txt")
# input_data2 = get_data_collection("KAIST/KAIST_30sec_007.txt")
# x2 = input_data2[0:200, 1]
# y2 = input_data2[0:200, 2]
# t2 = input_data2[0:200, 0]
#
fig = plt.figure()
ax = fig.gca(projection='3d')
#ax.set_title("3D gamer location track map",fontsize=13, fontweight='bold')
ax.set_xlabel(r"$u^{x}_{n}(t)$",fontsize=13, fontweight='bold')
ax.set_ylabel(r"$u^{y}_{n}(t)$",fontsize=13, fontweight='bold')
ax.set_zlabel(r"$t$",fontsize=13, fontweight='bold')

xlist = []
ylist = []
tlist = []

input_data_list = ["KAIST/KAIST_30sec_000.txt", "KAIST/KAIST_30sec_001.txt", "KAIST/KAIST_30sec_007.txt",
                   "KAIST/KAIST_30sec_011.txt"]

for index, i in enumerate(input_data_list):
    data = get_data_collection(i)
    ax.plot(data[0:200, 1], data[0:200, 2], data[0:200, 0], label="UE " + str(index+1))
    ylist.append(data[0:200, 2])
    tlist.append(data[0:200, 0])



# ax.plot(x, y, t, c='cornflowerblue', label='UE 1')
# ax.plot(x1, y1, t1, c='seagreen', label='UE 2')
# ax.plot(x2, y2, t2, c='indianred', label='UE 3')

# ax.plot(x, y, t, label='UE 1')
# ax.plot(x1, y1, t1, label='UE 2')
# ax.plot(x2, y2, t2, label='UE 3')

plt.legend(loc=0, numpoints=1)
leg = plt.gca().get_legend()
ltext = leg.get_texts()
plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细
plt.savefig("location.pdf", format="pdf", bbox_inches = 'tight')
# plt.savefig("location.png", bbox_inches = 'tight')
# plt.savefig('./filename.svg', format='svg')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
plt.savefig('./posation3.png')
plt.show()
