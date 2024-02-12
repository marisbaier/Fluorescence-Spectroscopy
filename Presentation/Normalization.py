'''
Semesterproject Fluorescence Spectroscopy
'''
import numpy as np
import matplotlib.pyplot as plt

n=150
Y_log_before = np.load('/home/tmp/baierluc/1_Y.npy')[:n]
Y_log = Y_log_before.reshape(256*256*n)
Y_log = Y_log[Y_log>0.2]
Y = np.exp(Y_log)-1

Y_log = Y_log_before.reshape(256*256*n)
Y_log = Y_log[Y_log>0.2]
Y = np.exp(Y_log)-1

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Intensity Distribution of Y and log(1+Y) from 150 samples')
fig.subplots_adjust(hspace=0.5)

ax1.hist(Y, bins=100, label='Relative Frequency from 150 samples', density=True, color=(0.04706, 0.13725, 0.26667), alpha=0.8)
ax1.set_xlabel('Intensity')
ax1.set_ylabel('Rel. Frequency')
ax1.grid(True)
#ax1.legend()

ax2.hist(Y_log, bins=100, label='Relative Frequency from 150 samples', density=True, color=(0.04706, 0.13725, 0.26667), alpha=0.8)
ax2.set_xlabel('Intensity after log(1+Y)')
ax2.set_ylabel('Rel. Frequency')
ax2.grid(True)
#ax2.legend()

plt.savefig('Presentation/Log_Normalization.png')

fig.clear()
rows = 3
columns = 5
fig, ax = plt.subplots(rows,columns, )
#fig.suptitle(f'{rows*columns} samples before any normalization.')

for i,axs in enumerate(ax.flatten()):
    axslol = axs.imshow(np.exp(Y_log_before[i])-1, vmin=0, vmax=1024)
    axs.axis('off')
#fig.tight_layout()
#fig.colorbar(axslol, ax=ax[-1], shrink=0.3)
fig.savefig('Presentation/Images_before_Normalization.png') 

for i,axs in enumerate(ax.flatten()):
    axslol = axs.imshow(Y_log_before[i], vmin=0, vmax=np.log(1024))
    axs.axis('off')

fig.savefig('Presentation/Images_after_Normalization.png')
