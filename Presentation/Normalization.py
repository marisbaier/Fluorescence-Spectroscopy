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

""" fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2, figsize=(7,5), dpi=300)
fig.suptitle('Log(1+Y) Normalization.')
fig.subplots_adjust(hspace=0.5, wspace=0.1)

ax1.hist(Y, bins=100, label='Relative Frequency from 150 samples', density=True, color=(0.04706, 0.13725, 0.26667), alpha=0.8)
ax1.set_xlabel('Intensity')
ax1.set_ylabel('Rel. Frequency from 150 samples')
ax1.grid(True)
#ax1.legend()

ax3.hist(Y_log, bins=100, label='Relative Frequency from 150 samples', density=True, color=(0.04706, 0.13725, 0.26667), alpha=0.8)
ax3.set_xlabel('Intensity after log(1+Y)')
ax3.set_ylabel('Rel. Frequency from 150 samples')
ax3.grid(True)
#ax2.legend()

ax2.imshow(np.exp(Y_log_before[1])-1, vmin=0, vmax=1024)
ax2.set_title('Before log(1+Y)')
ax2.axis('off')

ax4.imshow(Y_log_before[1], vmin=0, vmax=np.log(1024))
ax4.set_title('After log(1+Y)')
ax4.axis('off') """

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
