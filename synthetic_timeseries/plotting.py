from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt 
from stingray import AveragedCrossspectrum, AveragedPowerspectrum, Lightcurve
from typing import Optional, Tuple, Callable, Union
import numpy as np
from matplotlib import gridspec

def make_cs(lc1:Lightcurve,lc2:Lightcurve,**cs_kwargs)->AveragedCrossspectrum:

    return AveragedCrossspectrum.from_lightcurve(lc1,lc2,**cs_kwargs)

def make_ps(lc:Lightcurve,**cs_kwargs)->AveragedPowerspectrum:
    
    return AveragedPowerspectrum.from_lightcurve(lc,**cs_kwargs)

def plot_simulated_lag(lc1: Lightcurve,
                       lc2: Lightcurve,
                       target_lag:np.ndarray,
                       freq: Optional[np.ndarray] = None,
                       plotlinear: Optional[bool] = True,
                       ploterrors: Optional[bool] = True,
                       **stingray_kwargs)->Tuple[Figure,Tuple[Axes,Axes,Axes]]:
    
    pds1 = make_ps(lc1, **stingray_kwargs)
    pds2 = make_ps(lc2, **stingray_kwargs)
    cs = make_cs(lc1,lc2,**stingray_kwargs)

    lag,lag_err = cs.phase_lag()
    
    pds1_reb = pds1.rebin_log(0.1)
    pds2_reb = pds2.rebin_log(0.1)

    fig,(ax1,ax2,ax3) = plt.subplots(nrows=3,figsize=(8.5,9.5))
    fig.tight_layout(h_pad=2.,pad=3.3,w_pad = 8)
    
    if ploterrors:
        ax1.errorbar(cs.freq,lag,yerr=lag_err,capsize=2,marker='o',label='Calculated phase lag',fmt='')
    else:
        ax1.scatter(cs.freq,lag,label='Calculated phase lag')
    
    if plotlinear:
        ax1.set(xscale='linear',xlabel='Frequency (Hz)',ylabel='Phase lag',ylim=(-np.pi,np.pi),xlim=(10**-3,None))
        ax2.set_xscale('log')
    else:
        ax1.sharex(ax2)
        ax1.set(xscale='log',xlabel='Frequency (Hz)',ylabel='Phase lag',ylim=(-np.pi,np.pi),xlim=(10**-3,None))

    if freq is None:
        freq = cs.freq
    
    ax1.plot(freq,target_lag,label='Target',color='red')
    ax1.legend(loc='upper left')

    ax2.set(xlabel='Frequency (Hz)',ylabel='Power * Freq',yscale='log')
    ax2.step(pds1_reb.freq,pds1_reb.power * pds1_reb.freq,label='PDS 1')
    ax2.step(pds2_reb.freq,pds2_reb.power * pds2_reb.freq,label='PDS 2')
    ax2.legend(loc='upper right')
    
    ax3.set(xlabel='Time (s)',ylabel='Counts/sec')
    ax3.plot(lc1.time,lc1.countrate,label='LC 1')
    ax3.plot(lc2.time,lc2.countrate,label='LC 2')
    ax3.legend(loc='upper right')

    return (fig,(ax1,ax2,ax3))

def plot_coh_and_lag(lc1: Lightcurve,
                     lc2: Lightcurve,
                     target_lag:Optional[Union[np.ndarray,float,Callable]]=None,
                     target_coherence:Optional[Union[np.ndarray,float,Callable]] = None,
                     freq: Optional[np.ndarray] = None,
                     plotlinear: Optional[bool] = True,
                     ploterrors: Optional[bool] = False,
                     plotsegments: Optional[bool] = False,
                     title: Optional[str] = '',
                     **stingray_kwargs)->Tuple[Figure,Tuple[Axes,Axes,Axes]]:
    
    # pds1 = make_ps(lc1, **stingray_kwargs)
    # pds2 = make_ps(lc2, **stingray_kwargs)
    cs = make_cs(lc1,lc2, **stingray_kwargs)


    plt.rcParams['text.usetex'] = True
    lag_plot_pad = 0.4

    lag,lag_err = cs.phase_lag()

    if freq is None:
        freq = cs.freq

    # Inspect the inputs of target_lag and target_coherence
    if isinstance(target_lag, (float, int)):
        target_lag = np.ones_like(freq) * target_lag
    elif isinstance(target_lag, (list,np.ndarray)):
        pass
    else:
        target_lag = target_lag(freq)

    if isinstance(target_coherence, (float, int)):
        target_coherence = np.ones_like(freq) * target_coherence
    elif isinstance(target_coherence, (list,np.ndarray)):
        pass
    else:
        target_coherence = target_coherence(freq)
    

    coh = (cs.unnorm_power * np.conj(cs.unnorm_power)).real / (cs.pds1.unnorm_power * cs.pds2.unnorm_power)
    
    pds1_reb = cs.pds1.rebin_log(0.1)
    pds2_reb = cs.pds2.rebin_log(0.1)

    fig = plt.figure(figsize=(10, 9.5))  # Adjust width for side panel
    gs = gridspec.GridSpec(4, 2, width_ratios=[4, 1], wspace=0.05, hspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])
    ax4 = fig.add_subplot(gs[3, 0])

    if plotsegments:
        # Side panel for histograms
        ax1_hist = fig.add_subplot(gs[0, 1], sharey=ax1)
        ax2_hist = fig.add_subplot(gs[1, 1], sharey=ax2)
        ax1_hist.hist(lag, bins=30, orientation='horizontal', color='gray', alpha=0.7)
        ax1_hist.set_xlabel('Count')
        ax1_hist.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Hide y-axis ticks
        ax2_hist.hist(coh, bins=30, orientation='horizontal', color='gray', alpha=0.7)
        ax2_hist.set_xlabel('Count')
        ax2_hist.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)  # Hide y-axis ticks

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    if plotlinear:
        ax1.sharex(ax2)
        ax1.set(xscale='linear',xlabel='Frequency (Hz)',ylabel='Phase lag',ylim=(-np.pi-lag_plot_pad,np.pi+lag_plot_pad),xlim=(min(cs.freq),max(cs.freq)))
    else:
        ax2.sharex(ax1)
        ax3.sharex(ax1)
        ax1.set(xscale='log',xlabel='Frequency (Hz)',ylabel='Phase lag',ylim=(-np.pi-lag_plot_pad,np.pi+lag_plot_pad),xlim=(min(cs.freq),max(cs.freq))) 
    
    ax2.scatter(cs.freq,coh,label='Coherence')
    ax2.set(ylabel='Coherence',xlabel='Frequency (Hz)', ylim=(-.1,1.1))

    if target_coherence is not None:
        ax2.plot(freq,target_coherence,color='red',label='Target')

    ax2.legend(loc='upper right')
    
    if ploterrors:
        ax1.errorbar(cs.freq,lag,yerr=lag_err,capsize=2,marker='o',label='Calculated phase lag',linestyle='')
    else:
        ax1.scatter(cs.freq,lag,label='Calculated phase lag')

    ax1.set_yticks(np.array([-1,-1/2,0,1/2,1])*np.pi, labels=[r'$-\pi$',r'$-\pi/2$','0',r'$\pi/2$',r'$\pi$'])
    
    ax3.set_xscale('log')
    
    ax1.plot(freq,target_lag,label='Target',color='red')
    ax1.plot(freq,target_lag+2*np.pi,color='red')
    ax1.plot(freq,target_lag-2*np.pi,color='red')
    ax1.legend(loc='upper right')

    ax3.set(xlabel='Frequency (Hz)',ylabel='Power * Freq',yscale='log')
    ax3.step(pds1_reb.freq,pds1_reb.power * pds1_reb.freq,label='PDS 1')
    ax3.step(pds2_reb.freq,pds2_reb.power * pds2_reb.freq,label='PDS 2')
    ax3.legend(loc='upper right')
    
    ax4.set(xlabel='Time (s)',ylabel='Counts/sec')
    ax4.plot(lc1.time,lc1.countrate,label='LC 1')
    ax4.plot(lc2.time,lc2.countrate,label='LC 2')
    ax4.legend(loc='upper right')

    if len(title) > 0:
        fig.suptitle(title)

    if plotsegments:
        segsize = stingray_kwargs['segment_size']
        nsegs = int(np.ceil(np.max(lc1.time)/segsize))
        seg_boundaries = np.arange(nsegs)*segsize
        ax4.vlines(seg_boundaries,ax4.get_ylim()[0],ax4.get_ylim()[1],colors='red',linestyles='dotted')

    return (fig,(ax1,ax2,ax3))

def plot_coh_lag_deviations(lc1: Lightcurve,
                            lc2: Lightcurve,
                            target_lag:Optional[Union[np.ndarray,float,Callable]]=None,
                            target_coherence:Optional[Union[np.ndarray,float,Callable]] = None,
                            plotlinear: Optional[bool] = True,
                            plotylog: Optional[bool] = False,
                            title: Optional[str] = '',
                            **stingray_kwargs):
    
    cs = make_cs(lc1,lc2, **stingray_kwargs)


    plt.rcParams['text.usetex'] = True
    lag_plot_pad = 0.4

    lag,_ = cs.phase_lag()

    # Inspect the inputs of target_lag and target_coherence
    if isinstance(target_lag, (float, int)):
        target_lag = np.ones_like(cs.freq) * target_lag
    elif isinstance(target_lag, (list,np.ndarray)):
        if len(target_lag) != len(cs.freq):
            raise ValueError('Target lag array must of same length as cross spectrum')
    else:
        target_lag = target_lag(cs.freq)

    if isinstance(target_coherence, (float, int)):
        target_coherence = np.ones_like(cs.freq) * target_coherence
    elif isinstance(target_coherence, (list,np.ndarray)):
        if len(target_coherence) != len(cs.freq):
            raise ValueError('Target coherence array must of same length as cross spectrum')
    else:
        target_coherence = target_coherence(cs.freq)
    

    coh = (cs.unnorm_power * np.conj(cs.unnorm_power)).real / (cs.pds1.unnorm_power * cs.pds2.unnorm_power)
    
    window = 20 # window over which to calculate the running average
    running_avg = np.convolve(target_coherence - coh, np.ones(window)/window, mode='valid')
    lag_running_av = np.convolve(target_lag - lag, np.ones(window)/window, mode='valid')


    fig,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=4,figsize=(8.5,9.5))
    fig.tight_layout(h_pad=2.,pad=3.3,w_pad = 8)

    ax2.sharex(ax1)
    ax3.sharex(ax1)
    ax4.sharex(ax1)

    ax4.set_ylabel('Frequency (Hz)')
    
    ax1.set(ylabel='Phase lag',ylim=(-np.pi-lag_plot_pad,np.pi+lag_plot_pad),xlim=(min(cs.freq),max(cs.freq)))
    ax2.set(ylabel='Phase lag deviation',xlim=(min(cs.freq),max(cs.freq)))
    ax3.set(ylabel='Coherence')
    ax4.set(ylabel='Coherence deviation',xlabel='Frequency (Hz)')

    if plotlinear:
        ax1.set(xscale='linear')
    else:
        ax1.set(xscale='log') 

    if plotylog:
        ax2.set_yscale('log')
        ax4.set_yscale('log')
        ax2.scatter(cs.freq,np.abs(target_lag-lag))
        ax4.scatter(cs.freq,np.abs(target_coherence-coh))
        ax4.plot(cs.freq[window-1:], np.abs(running_avg), color='orange', label='Running Average')
        ax2.plot(cs.freq[window-1:],np.abs(lag_running_av),color='orange',label='Running Average')
    else:
        ax2.scatter(cs.freq,target_lag-lag)
        ax4.scatter(cs.freq,target_coherence-coh)
        ax4.plot(cs.freq[window-1:], running_avg, color='orange', label='Running Average')
        ax2.plot(cs.freq[window-1:],lag_running_av,color='orange',label='Running Average')

    ax3.scatter(cs.freq,coh,label='Calculated')
    ax3.plot(cs.freq,target_coherence,color='red',label='Target')
    
    ax1.scatter(cs.freq,lag,label='Calculated')
    ax1.plot(cs.freq,target_lag,color='red',label='Target')

   

    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    ax3.legend(loc='upper right')
    ax4.legend(loc='upper right')

    if len(title) > 0:
        fig.suptitle(title)

    return (fig,(ax1,ax2,ax3))