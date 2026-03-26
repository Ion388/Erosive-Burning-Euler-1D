import numpy as np
import pymsis

dates = np.arange(np.datetime64("2025-10-28T00:00"), np.datetime64("2025-11-04T00:00"), np.timedelta64(30, "m"))
# geomagnetic_activity=-1 is a storm-time run
data = pymsis.calculate(dates, 0, 0, 400, geomagnetic_activity=-1)

# Plot the data
import matplotlib.pyplot as plt
# Total mass density over time
plt.plot(dates, data[:, 0, 0, 0, 0])
plt.show()