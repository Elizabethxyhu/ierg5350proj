import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
actor_loss=np.load('C:\\Users\\ChengYuansen\\Desktop\\2020-2021_1\\IERG5230\\TSP_DRL_PtrNet-master-course\\loss_file\\actor_loss_node5.npy')
critic_loss=np.load('C:\\Users\\ChengYuansen\\Desktop\\2020-2021_1\\IERG5230\\TSP_DRL_PtrNet-master-course\\loss_file\\critic_loss_node5.npy')
actor_loss = actor_loss[:1000]
critic_loss = critic_loss[:1000]
x=np.arange(1,len(actor_loss)+1)


#创建图和小图
fig = plt.figure(figsize = (7,5))
ax1 = fig.add_subplot(1, 1, 1)


#先画出整体的loss曲线
p2 = pl.plot(x, actor_loss,'r-', label = u'Actor_loss')
pl.legend()
#显示图例
p3 = pl.plot(x,critic_loss, 'b-', label = u'Critic_loss')
pl.legend()


pl.xlabel(u'epoch')
pl.ylabel(u'loss')
plt.title('Trianning loss for actor and critic networks - P+O TSP5')


#放大图片
# plot the box
tx0 = 0
tx1 = 50
#设置想放大区域的横坐标范围
ty0 = -25
ty1 = 25
#设置想放大区域的纵坐标范围
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
pl.plot(sx,sy,"purple")

axins2 = ax1.inset_axes((0.45, 0.07, 0.4, 0.4))
axins2.plot(x,actor_loss , color='red', ls='-')
axins2.plot(x,critic_loss , color='blue', ls='-')
axins2.axis([0,50,-25,25])

#axins = inset_axes(ax1, width=1.5, height=1.5, loc=4, bbox_to_anchor=(1,2,1,2))
#loc是设置小图的放置位置，可以有"lower left,lower right,upper right,upper left,upper #,center,center left,right,center right,lower center,center"

plt.savefig("train_results_loss_5.png")
pl.show()
