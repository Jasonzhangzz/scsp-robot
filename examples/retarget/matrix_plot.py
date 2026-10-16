"""Plot the common-clock errors saved by matrix_report.py."""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root):
    root=Path(root)
    methods=[('fa_cost','FA cost','#d95f02'),('comfree_model','ComFree prediction','#1b9e77'),('dexforge','DexForge','#7570b3'),('current','Current','#e7298a')]
    fig,axes=plt.subplots(2,2,figsize=(11,7),sharex=True,constrained_layout=True)
    heights,haxes=plt.subplots(2,1,figsize=(9,6),sharex=True,constrained_layout=True)
    for row,executor in enumerate(['mujoco','comfree']):
        reference=None
        for method,label,color in methods:
            a=np.loadtxt(root/executor/method/'pose_errors.csv',delimiter=',',skiprows=1)
            for column in range(2):axes[row,column].plot(a[:,0],a[:,column+1],label=label,color=color,linewidth=1.7)
            haxes[row].plot(a[:,0],a[:,5],label=label,color=color,linewidth=1.7)
            reference=a
        haxes[row].plot(reference[:,0],reference[:,12],label='Demonstration',color='black',linewidth=1.6,linestyle='--')
        haxes[row].set_ylabel(executor+'\nObject z (m)')
        for ax in [*axes[row],haxes[row]]:
            ax.axvline(71/30,color='0.5',linestyle=':',linewidth=1);ax.grid(alpha=.2)
        axes[row,0].set_ylabel(executor+'\nPosition error (mm)');axes[row,1].set_ylabel(executor+'\nRotation error (deg)')
    for ax in axes[-1]:ax.set_xlabel('Simulation time (s)')
    haxes[-1].set_xlabel('Simulation time (s)')
    axes[0,0].legend(fontsize=9);haxes[0].legend(fontsize=9,ncol=3)
    fig.savefig(root/'tracking_errors.png',dpi=180);heights.savefig(root/'object_height.png',dpi=180)
    plt.close(fig);plt.close(heights)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();plot(a.root)
