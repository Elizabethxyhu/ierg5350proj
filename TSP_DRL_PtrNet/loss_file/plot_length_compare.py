import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
pred_length=np.load('C:\\Users\\ChengYuansen\\Desktop\\2020-2021_1\\IERG5230\\TSP_DRL_PtrNet-master-course\\loss_file\\list_pred_ave_len_node20_8000_iteration.npy')
pred_length_wo=np.load('C:\\Users\\ChengYuansen\\Desktop\\2020-2021_1\\IERG5230\\TSP_DRL_PtrNet-master-course\\loss_file\\list_pred_ave_len_node20_wo.npy')
#pred_length=np.load('C:\\Users\\ChengYuansen\\Desktop\\2020-2021_1\\IERG5230\\TSP_DRL_PtrNet-master\\plot_data\\node5\\list_pred_ave_len.npy')
#length_diff=np.load('C:\\Users\\ChengYuansen\\Desktop\\2020-2021_1\\IERG5230\\TSP_DRL_PtrNet-master\\plot_data\\node5\\list_ave_len_diff.npy')
#opt_length = opt_length[:5000]
#length_diff = length_diff[:5000]

pred_length = pred_length * 110
pred_length_wo = pred_length_wo * 110
x=np.arange(1,len(pred_length)+1)

#创建图和小图
fig = plt.figure(figsize = (7,5))
ax1 = fig.add_subplot(1, 1, 1)

'''
#先画出整体的loss曲线
p2 = pl.plot(x, opt_length,'r-', label = u'Optimal_length')
pl.legend()
'''
#显示图例
p3 = pl.plot(x,pred_length, 'b-', label = u'P+O TSP20 w/ TSP5 Pre-training')
pl.legend()

p9 = pl.plot(x,pred_length_wo, 'y-', label = u'P+O TSP20 w/o TSP5 Pre-training')
pl.legend()

'''
p4 = pl.plot(x, length_diff,'y-', label = u'Length_difference')
pl.legend()
'''

'''
#画出随机选择的结果

p5 = plt.axhline(y=550, color='r', linestyle='-', label = u'Random_Select')
#p5 = pl.plot(500, 'y-', label = u'Predicted_length')
pl.legend()


#画出linear regression 的结果
p6 = plt.axhline(y=549.3, color='y', linestyle='-', label = u'Linear_Regression')
#p5 = pl.plot(500, 'y-', label = u'Predicted_length')
pl.legend()

#画出SPO tree 的结果
p7 = plt.axhline(y=530.6, color='g', linestyle='-', label = u'SPO_tree')
#p5 = pl.plot(500, 'y-', label = u'Predicted_length')
pl.legend()

#画出SPO forest 的结果
p8 = plt.axhline(y=511.38, color='k', linestyle='-', label = u'SPO_forest')
#p5 = pl.plot(500, 'y-', label = u'Predicted_length')
pl.legend()

'''

pl.xlabel(u'epoch')
pl.ylabel(u'length')
plt.title('Tour length - P+O TSP20')


###放大图片
### plot the box
##tx0 = 0
##tx1 = 80
###设置想放大区域的横坐标范围
##ty0 = -83
##ty1 = 43
###设置想放大区域的纵坐标范围
##sx = [tx0,tx1,tx1,tx0,tx0]
##sy = [ty0,ty0,ty1,ty1,ty0]
##pl.plot(sx,sy,"purple")
##
##axins2 = ax1.inset_axes((0.5, 0.1, 0.45, 0.45))
##axins2.plot(x,actor_loss , color='red', ls='-')
##axins2.plot(x,critic_loss , color='blue', ls='-')
##axins2.axis([0,80,-83,43])

#axins = inset_axes(ax1, width=1.5, height=1.5, loc=4, bbox_to_anchor=(1,2,1,2))
#loc是设置小图的放置位置，可以有"lower left,lower right,upper right,upper left,upper #,center,center left,right,center right,lower center,center"

plt.savefig("train_results_length_TSP20.png")
pl.show()
