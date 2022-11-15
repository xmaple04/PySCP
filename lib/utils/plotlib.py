import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# from matplotlib.backends.backend_pdf import PdfPages

plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams.update({'font.size': 12})
# config = {"mathtext.fontset": 'stix'}
# rcParams.update(config)
SimSun = FontProperties(fname='D:/ProgramData/Miniconda3/Lib/site-packages/matplotlib/mpl-data/fonts/ttf/SimSun.ttf')  # 字体放在该目录下

if __name__ == '__main__':
    # 全屏
    plt.switch_backend('QT5Agg')
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    linestyle_tuple = [
        ('loosely dotted', (0, (1, 10))),
        ('dotted', (0, (1, 1))),
        ('densely dotted', (0, (1, 1))),

        ('loosely dashed', (0, (5, 10))),
        ('dashed', (0, (5, 5))),
        ('densely dashed', (0, (5, 1))),

        ('loosely dashdotted', (0, (3, 10, 1, 10))),
        ('dashdotted', (0, (3, 5, 1, 5))),
        ('densely dashdotted', (0, (3, 1, 1, 1))),

        ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
        ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
        ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

    # 中英    # 参见https://blog.csdn.net/Mr_sdhm/article/details/111498153
    plt.xlabel(u"时间", fontproperties=SimSun)
    plt.legend([u"实际数据", u'简化模型'], prop={'family': 'SimSun'})

    # 空心
    # markerfacecolor='none'

    # 中文
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 隐藏坐标轴
    ax = plt.gca()
    ax.axes.get_yaxis().set_visible(False)

    # 保存图片，不要留白
    plt.savefig('template.jpg', dpi=900, bbox_inches='tight')


    # 遍历对象的所有属性
    class CC:
        def __init__(self):
            self.a = 2
            self.b = 3

        def print(self, name):
            for property, value in vars(self).items():
                print(property, ':', value)


    # 获得句柄，legend
    a, = plt.plot(1, 2)
    b, = plt.plot(2, 3)
    plt.legend([a, b], ['a', 'b'])
