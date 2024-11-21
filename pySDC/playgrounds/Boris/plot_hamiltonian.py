import numpy as np
import matplotlib.pyplot as plt
import pdb
import time as tm
import matplotlib.pyplot as _plt
_plt.rcParams['figure.figsize'] = 7.44, 6.74
_plt.rc('font', size=15)
_plt.rcParams['lines.linewidth'] = 2
_plt.rcParams['axes.titlesize'] = 20
_plt.rcParams['axes.labelsize'] = 25
_plt.rcParams['xtick.labelsize'] = 18
_plt.rcParams['ytick.labelsize'] = 18
_plt.rcParams['xtick.major.pad'] = 5
_plt.rcParams['ytick.major.pad'] = 5
_plt.rcParams['axes.labelpad'] = 6
# _plt.rcParams['markers.fillstyle'] = 'none'
_plt.rcParams['lines.markersize'] = 3.0
_plt.rcParams['lines.markeredgewidth'] = 1.2
_plt.rcParams['mathtext.fontset'] = 'cm'
_plt.rcParams['mathtext.rm'] = 'serif'
# time_RKN1 = np.array([])
# time_SDC2 = np.array([])
# time_SDC3 = np.array([])
# time_SDC4 = np.array([])

# Ham_RKN1=np.array([])
# Ham_SDC2=np.array([])
# Ham_SDC3=np.array([])
# Ham_SDC4=np.array([])
Name=('SDC25G', 'SDC35G', 'SDC45G')
Hamiltonian={ 'SDC25G':np.array([]), 'SDC35G':np.array([]), 'SDC45G':np.array([])}
# time={'RKNG':np.array([]) }#, 'SDC25G':np.array([]), 'SDC35G':np.array([])}
# tm.ctime()

# for ii in Name:

#     filename='data/Energy_' + ii +'new.txt'

#     np.save('data/Ham' + ii + '.npy', Hamiltonian[ii])
#     np.save('time' + ii + '.npy', time[ii])
#    # print(ii)
#     tm.ctime()

    # pdb.set_trace()
time=1e+6
tn=2*np.pi/10
t=np.arange(0, time+tn, tn)
t_len=len(t)
RKNnew=np.loadtxt('data/Energy_RKNGnew.txt', delimiter='*')
SDC2new=np.loadtxt('data/Energy_SDC25Gnew.txt', delimiter='*')
SDC3new=np.loadtxt('data/Energy_SDC35Gnew.txt', delimiter='*')
SDC4new=np.loadtxt('data/Energy_SDC45Gnew.txt', delimiter='*')
RKN=RKNnew[:,1]
SDC2=SDC2new[:,1]
SDC3=SDC3new[:,1]
SDC4=SDC4new[:,1]
step=3000

#SDC12=np.load('energy1million.npy')
#pdb.set_trace()
plt.loglog(t[:t_len:step], RKN[:t_len:step], label='RKN-4', marker='.', linestyle=' ')
plt.loglog(t[:t_len:step], SDC2[:t_len:step], label='K=2', marker='s', linestyle=' ')
plt.loglog(t[:t_len:step], SDC3[:t_len:step], label='K=3', marker='*', linestyle=' ')
plt.loglog(t[:t_len:step], SDC4[:t_len:step], label='K=4', marker='H', linestyle=' ')
plt.xlabel('$\omega \cdot t$')
plt.ylabel('$\Delta H^{\mathrm{(rel)}}$')
plt.ylim(1e-11, 1e-3 +0.001)
plt.legend(fontsize=15)
plt.tight_layout()
