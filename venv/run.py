import numpy as np
import matplotlib.pyplot as plt
import matplotlib
print(plt)

x = np.arange(0, 6, 0.1)
y = np.sin(x)


plt.plot(x, y)

plt.show()
