import matplotlib.pyplot as plt
import numpy as np


x=np.linspace(-1,1,50)
y1=x*x

y2=x**3





plt.figure()
plt.plot(x,y1)



plt.figure()
plt.plot(x,y1)
plt.plot(x,y2,c='red',ls='--',lw=10)
plt.xlim(-1,2)
plt.xlim(0,3)
plt.xlabel('i am x')
plt.ylabel('i am y')

new_ticks=np.linspace(-1,2,5)


print(new_ticks)

plt.xticks(new_ticks)
# 这里告诉我们matplotlib是支持latex的
plt.yticks([-2,-1.8,-1,1.22,3,
            ],
           [r'$really\ bad$',r'$bad$',r'$5_1$',r'$good$',r'$really\ good$'
            ])



plt.show()
