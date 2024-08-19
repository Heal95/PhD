#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from scipy import signal
import os

def ormsby(duration, dt, f, return_t=False):
    """
    The Ormsby wavelet requires four frequencies which together define a
    trapezoid shape in the spectrum. The Ormsby wavelet has several sidelobes,
    unlike Ricker wavelets.
    Args:
        duration (float): The length in seconds of the wavelet.
        dt (float): The sample interval in seconds (usually 0.001, 0.002,
            or 0.004).
        f (ndarray): Sequence of form (f1, f2, f3, f4), or list of lists of
            frequencies, which will return a 2D wavelet bank.
        f1 = low-cut frequency
        f2 = low-pass frequency (roll-off)
        f3 = high-pass frequency (roll-off)
        f4 = high-cut frequency
    Returns:
        ndarray: A vector containing the Ormsby wavelet, or a bank of them.
    """
    f = np.asanyarray(f).reshape(-1, 1)
    try:
        f2, f1, f4, f3 = f
    except ValueError:
        raise ValueError("The last dimension must be 4")

    def numerator(f, t):
        return (np.sinc(f * t)**2) * (np.pi * (f ** 2))

    pf43 = f4 - f3
    pf21 = f2 - f1

    t = np.arange(-duration/2, duration/2, dt)
    w = (numerator(f1, t)/pf21) - (numerator(f2, t)/pf21) - (numerator(f3, t)/pf43) + (numerator(f4, t)/pf43)
    w = np.squeeze(w) / np.amax(w)
    t = np.arange(0, duration, dt)
    
    if return_t:
        return w, t
    else:
        return w

def strong_motion_duration(trace, filters, s_filters, save_fig=False, fig_folder='./'):
    """
    Strong motion duration of the record according to the paper by Novikova and Trifunac (1995).
    Args:
        trace (obspy Trace object): The input record.
        filters (ndarray): Sequence of form (f1, f2, f3, f4), or list of lists of
            frequencies, used for Ormsby filtering.
        s_filters (ndarray): List of smoothing filters
        save_fig (boolean): True/False to save figure
        fig_folder (string): Path to where the figures are saved 
    Returns:
        list: A vector containing strong motion duration in seconds for each filter/channel.
    """
    
    Ttr = trace.times()
    Vtr = trace.data
    dt = trace.stats.delta
    durations = []
    for m in range(0,len(filters)):
        filt = filters[m]
        oms,t = ormsby(len(Vtr)*dt,dt,filt,True)
        s = np.convolve(Vtr, oms,'same')*dt 
        window = signal.tukey(len(Vtr),0.25)
        s *= window

        # integral of the squared signal values
        I = 0
        data = s*s
        integral = []
        for l in range(0,len(Ttr)):
            Ii = data[l]*dt
            I += Ii
            integral.append(I)

        # normalize the values of integrals    
        integral = np.asarray(integral)
        integral /= max(abs(integral))

        # integral smooting using LF filter
        fs = 1/dt
        fc = s_filters[m]  # Cut-off frequency of the filter
        wl = fc / (fs / 2) # Normalize the frequency
        bl, al = signal.butter(4, wl, 'low')
        Ism = signal.filtfilt(bl, al, integral)
        m += 1

        # derivative of the smoothed integral
        D = np.gradient(Ism)
        D /= max(abs(D))

        # defining the strong motion portions of the record
        suma = max(Ism); alpha = 0;
        while suma/max(Ism) > 0.9:
            threshold = 0.1 + alpha
            x = np.empty(len(D))
            x.fill(threshold*max(D)) # threshold
            times_t = []; pom = True;
            for l in range(0,len(Ttr)):
                if D[l]>=x[l]:
                    if pom:
                        times_t.append(Ttr[l])
                        pom = False
                    if D[l+1]<x[l+1]:
                        times_t.append(Ttr[l])
                        pom = True
            suma = 0            
            for l in range(0,len(times_t),2):
                k1 = int(round(times_t[l]/dt))
                k2 = int(round(times_t[l+1]/dt))
                suma += Ism[k2] - Ism[k1]
                alpha += 0.001

        # calculation of the duration
        dur = 0
        for l in range(0,len(times_t),2):
            dur += times_t[l+1]-times_t[l]
        durations.append(dur)
        # plotting steps of the method for each of the channels
        fig = plt.figure(figsize=(36, 6))
        plt.rcParams.update({'font.size': 20})
        #fig.suptitle(sta[j]+', component: '+comp)
        plt.subplot(1, 3, 1)
        plt.rcParams.update({'font.size': 20})
        plt.grid(True)
        plt.title('Ormsby filter = '+str(filt)+' Hz')
        plt.plot(Ttr,Vtr,'b-',label='measured data')
        plt.plot(Ttr,s,'r-', label='filtered data')
        for xc in times_t:
            plt.axvline(x=xc,ls='--',c='k')
        axes = plt.gca()
        for l in range(0,len(times_t),2):
            x1 = times_t[l]
            x2 = times_t[l+1]
            if l==0:
                axes.axvspan(x1, x2, alpha=0.3, color='grey',label='strong motion')
            else:
                axes.axvspan(x1, x2, alpha=0.3, color='grey')
        plt.xlim(min(Ttr),max(Ttr))
        plt.legend(loc='best')

        plt.subplot(1, 3, 2)
        plt.rcParams.update({'font.size': 20})
        plt.grid(True)
        plt.title('Duration = ' + str(dur)+' s')
        plt.plot(Ttr,integral,'b-',label='integral $\int$ f$^{2}$(t) dt')
        plt.plot(Ttr,Ism,'r-',label='smoothed integral')
        for xc in times_t:
            plt.axvline(x=xc,ls='--',c='k')
        axes = plt.gca()
        for l in range(0,len(times_t),2):
            x1 = times_t[l]
            x2 = times_t[l+1]
            if l==0:
                axes.axvspan(x1, x2, alpha=0.3, color='grey',label='strong motion')
            else:
                axes.axvspan(x1, x2, alpha=0.3, color='grey')
        plt.xlim(min(Ttr),max(Ttr))
        plt.legend(loc='best')

        plt.subplot(1, 3, 3)
        plt.rcParams.update({'font.size': 20})
        plt.grid(True)
        plt.title('Duration = ' + str(dur)+' s')
        plt.plot(Ttr,D,'b-',label='derivate')
        plt.plot(Ttr,x,'k--',label='threshold level')
        axes = plt.gca()
        axes.fill_between(Ttr, x, D, where=D > x, color='grey', alpha=0.3,label='strong motion')
        plt.xlim(min(Ttr),max(Ttr))
        plt.legend(loc='best')
        plt.show()
        if save_fig:
            fig.savefig(os.path.join(fig_folder,'duration'+str(filt)+'.png'))

    return durations