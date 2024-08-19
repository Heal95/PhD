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

def scaling_laws(Mw, scaling_law, fault, mu=30*pow(10,9), dyne=False):
    """
    Fault dimensions, slip, rupture area and seismic moment from
    scaling law.
    Args:
        Mw (float): Moment magnitude of the event.
        scaling_law (string): Either WC (Wells & Coppersmith, 1994) 
        or S (Strasser et.al, 2010) scaling law.
        fault (string): Type of fault: SS (strike-slip), R (reverse), N (normal).
        mu (float): Shear modulus.
        dyne (boolean): Output M0 in dyne cm (True) or Nm (False).
    Returns:
        L (float): Surface rupture length.
        W (float): Downdip rupture width.
        A (float): Area of the rupture.
        M0 (float): Seismic moment in Nm or dyne cm.
        S (float): Average slip on the fault.
    """
    
    # calculating surface rupture length L and downdip rupture width W
    if scaling_law == 'WC': # Wells & Coppersmith earthquake scaling factors, 1994
        if fault == 'SS': # strike-slip fault
            L = pow(10,-2.57+0.62*Mw)
            W = pow(10,-0.76+0.27*Mw)
        elif fault == 'R': # reverse fault
            L = pow(10,-2.42+0.58*Mw)
            W = pow(10,-1.61+0.41*Mw)
        elif fault == 'N': # normal fault
            L = pow(10,-1.88+0.50*Mw)
            W = pow(10,-1.14+0.35*Mw)
        else: # general fault
            L = pow(10,-2.44+0.59*Mw)    
            W = pow(10,-1.01+0.32*Mw)
    elif scaling_law == 'S': # Strasser et.al earthquake scaling factors, 2010
        L = pow(10,-2.477+0.585*Mw)   
        W = pow(10,-0.882+0.351*Mw)

    A = L*W # area of the rupture
    M0 = pow(10,1.5*Mw+9.1) # seismic moment in Nm
    S = M0/(A*1000000)/mu #slip
    if dyne:
        M0 = pow(10,1.5*(Mw+10.7)) # seismic moment in dyne cm

    return(L, W, A, M0, S)

def dist2coord_deg(lon_P0,lat_P0,Lx,Ly):
    """
    Coordinate of the unknown point P from distance between
    P and known point P0 on Earth.
    Args:
        lon_P0 (float): Longitude of the known point.
        lat_P0 (float): Latitude of the known point.
        Lx (float): Distance in kilometers along x (UTM).
        Ly (float): Distance in kilometers along y (UTM).
    Returns:
        lon_P (float): Longitude of the unknown point.
        lat_P (float): Latitude of the unknown point.
    """

    Re = 6371 # Earth radius in km 

    lat_P_tmp = (180*Ly)/(np.pi*Re)
    lat_P = lat_P0 + lat_P_tmp; # latitude of the unknown point P
    
    R = (np.sqrt(pow(Lx,2)+pow(Ly,2)))/Re # distance of the point P and P0 in radians
    a = np.cos(R)-np.sin(lat_P*np.pi/180)*np.sin(lat_P0*np.pi/180)
    b = np.cos(lat_P*np.pi/180)*np.cos(lat_P0*np.pi/180)
    lon_P_tmp = np.arccos(a/b)*180/np.pi#/180

    if Lx<0:
        lon_P_tmp = lon_P_tmp*(-1)
    lon_P = lon_P0 + lon_P_tmp # longitude of the unknown point P
    
    return(lon_P, lat_P);

def LatLonR2xyz(file_in):
    """
    Conversion of the latitude, longitude and R to x, y and z
    coordinates.
    Args:
        file_in (ndarray): List containing original coordinates.
    Returns:
        file_out (ndarray): List containing converted coordinates.
    """

    lon = file_in[0]
    lat = file_in[1]
    depth = file_in[2]
    Re = 6371
    erre = Re - depth

    x1 = erre*np.sin((90-lat)*np.pi/180)*np.cos(lon*np.pi/180)
    y1 = erre*np.sin((90-lat)*np.pi/180)*np.sin(lon*np.pi/180)
    z1 = erre*np.cos((90-lat)*np.pi/180)

    file_out = [x1,y1,z1]

    return(file_out);

def patchfault(m,i,j):
    """
    Creates patch model for a given finite-fault parameters.
    Args:
        m (ndarray): List containing fault parameters: L, W, depth_le, 
        dip, strike, W/2*np.cos(strike*np.pi/180), 0
        i (int): Number of patches along fault length.
        j (int): Number of patches along fault width.
    Returns:
        pm (ndarray): List defining patch model.
    """

    # set constants
    dip = m[3]*np.pi/180
    strike = -m[4]*np.pi/180
    sin_dip = np.sin(dip)
    cos_dip = np.cos(dip)
    iw = m[0]/i # length
    jw = m[1]/j # width
    iss = np.asarray([np.linspace(1,i,i)])
    jss = np.asarray([np.linspace(1,j,j)]).T
    n = i*j
    
    c1 = -m[1]*cos_dip
    c2 = 0.5*(m[0] + iw)
    c3 = m[2]-j*jw*sin_dip

    # calculate midpoints and depths of the patches
    p = np.dot(cos_dip*(jw*jss-m[1]),np.ones((1,i)))
    q = np.dot(np.ones((j,1)),(iw*iss)-0.5*(m[0]+iw))
    r = np.dot(m[2]-jw*sin_dip*(j-jss),np.ones((1,i)))
    p = p.T; 
    q = q.T; 
    r = r.T;
    mp = [p.ravel(), q.ravel(), r.ravel()]
    mp = np.asarray(mp).T
    
    # adjust midpoints for strike
    R = np.asarray([(np.cos(strike),-np.sin(strike),0),
         (np.sin(strike),np.cos(strike),0),
         (0,0,1)])
    mp = np.dot(mp,R.T).T
    
    # adjust midpoints for offset from origin
    mp[:][0] = mp[:][0] + m[5]; 
    mp[:][1] = mp[:][1] + m[6]; 
    
    # form patch models
    pm = [];
    pm.append((np.ones((n,1))*iw).ravel())
    pm.append((np.ones((n,1))*jw).ravel())
    pm.append((mp[:][2]).ravel())
    pm.append((np.ones((n,1))*m[3]).ravel())
    pm.append((np.ones((n,1))*m[4]).ravel())
    pm.append((mp[:][0]).ravel())
    pm.append((mp[:][1]).ravel())
    pm = np.asarray(pm)

    return pm;

def cmt2finite_faults(lonc,latc,depthc,strike,dip,Mw,ii,jj,H):
    """
    Creates finite-fault model for a source.
    Args:
        lonc (float): Longitude of the hypocenter.
        latc (float): Latitude of the hypocenter.
        depthc (float): Depth of the hypocenter in km.
        strike (float): Strike of the event.
        dip (float): Dip of the event.
        Mw (float): Moment magnitude of the event.
        ii (int): Number of patches along fault length.
        jj (int): Number of patches along fault width.
        H (int): Rupture model: 1 (left lateral), 2 (right lateral),
        3 (bilateral).
    Returns:
        coords_time (ndarray): List defining finite-fault model.
    """

    L, W, A, M0, S = scaling_laws(Mw,'WC','R')
    depth_le = depthc + W*np.sin(dip*np.pi/180)/2
    m = [L, W, depth_le, dip, strike, W/2*np.cos(strike*np.pi/180), 0]
    pm = patchfault(m,ii,jj)
    lon_P = []; lat_P = []; coords_time = []; 
    for i in range(0,ii*jj):
        lo_P,la_P = dist2coord_deg(lonc,latc,pm[5][i],pm[6][i])
        lon_P.append(float(lo_P))
        lat_P.append(float(la_P))
    coords_time.append(lon_P)
    coords_time.append(lat_P)
    coords_time.append(pm[2][::-1])
    coords_time.append(np.ones(len(lon_P)))
    
    # H is the time rupture mode: 1 = left lateral, 2 = right lateral, 3 = bilateral
    if H == 1:
        index = 2*jj+jj/2 # index of the starting point (middle left)
    elif H == 3:
        index = jj*(ii/2)+jj/2 # index of the starting point (centroid)
    elif H == 2:
        index = jj*(ii-3)+jj/2 # index of the starting point (middle right)
        
    velocity_rupture = 3 * 0.8 # km/s
    nucl_pt_lld = []; 
    nucl_pt_lld.append(lon_P[int(index)])
    nucl_pt_lld.append(lat_P[int(index)])
    nucl_pt_lld.append(pm[2][int(index)])
    nucl_pt_xyz = LatLonR2xyz(nucl_pt_lld)
    
    dist = [];
    for j in range(0,len(lon_P)):
        actual_pt_lld = [];
        actual_pt_lld.append(lon_P[j])
        actual_pt_lld.append(lat_P[j])
        actual_pt_lld.append(pm[2][j]) 
        actual_pt_xyz = LatLonR2xyz(actual_pt_lld)
        dist.append(np.sqrt(pow(actual_pt_xyz[0]-nucl_pt_xyz[0],2) + pow(actual_pt_xyz[1]-nucl_pt_xyz[1],2) + pow(actual_pt_xyz[2]-nucl_pt_xyz[2],2))) 
        coords_time[3][j] = dist[j]/velocity_rupture
    
    return coords_time;