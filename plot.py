import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

nash_price = 1.4729273733327568
coop_price = 1.9249689958811602

nash = 0.22292696
coop = 0.33749046

r = np.load("rewards.npy")
print(r[0, :10])
r[0] = (r[0] - nash) / (coop - nash)
r[1] = (r[1] - nash) / (coop - nash)

plt.plot(pd.Series(r[0]).rolling(100).mean())
plt.plot(pd.Series(r[1]).rolling(100).mean())

plt.show()
