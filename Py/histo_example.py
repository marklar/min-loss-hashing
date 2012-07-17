#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

mu = 100
sigma = 2
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(
    x,          # the data points.
    20,         # the number of bins.
    normed=1,   # normalize points (x) to make pdf (prob dens fn).
                # integral of the histo: 1.0.
    facecolor='g',
    alpha=0.35  # percent opacity.
    )

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(96, .20, '$\\mu={0},\ \\sigma={1}$'.format(mu, sigma))
plt.axis([
    mu-(sigma*4), mu+(sigma*4), # x-axis range
    0, 0.3                      # y-axis range
    ])
plt.grid(True)
plt.show()
