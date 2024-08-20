#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from obspy import read
from scipy import signal
import os, sys

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

def dist2coord_deg(lon_P0, lat_P0, Lx, Ly):
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

def patchfault(m, i, j):
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

def cmt2finite_faults(lonc, latc, depthc, strike, dip, Mw, ii, jj, H):
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

def sdr2Hcmt(Mw, S, D, R):
    """
    Converts moment tensor components from XYZ to TPR coordinates
    (Harvard CMTSOLUTION format)
    Args:
        Mw (float): Moment magnitude of the event.
        S (float): Strike of the fault.
        D (float): Dip of the fault.
        R (float): Rake of the fault.
    Returns:
        float: Moment tensor components in TPR coordinates.
    """

    # PI / 180 to convert degrees to radians
    d2r =  np.pi/180

    # convert to radians
    S *= d2r
    D *= d2r
    R *= d2r

    # Earthquake magnitude
    M0 = pow(10,1.5*(Mw+10.7)); # seismic moment in dyne cm

    Mxx = -M0 * (np.sin(D) * np.cos(R) * np.sin(2*S) + np.sin(2*D) * np.sin(R) * np.sin(S)*np.sin(S))
    Myy =  M0 * (np.sin(D) * np.cos(R) * np.sin(2*S) - np.sin(2*D) * np.sin(R) * np.cos(S)*np.cos(S))
    Mzz = -1.0 * (Mxx + Myy)
    Mxy =  M0 * (np.sin(D) * np.cos(R) * np.cos(2*S) + 0.5 * np.sin(2*D) * np.sin(R) * np.sin(2*S))
    Mxz = -M0 * (np.cos(D) * np.cos(R) * np.cos(S) + np.cos(2*D) * np.sin(R) * np.sin(S))
    Myz = -M0 * (np.cos(D) * np.cos(R) * np.sin(S) - np.cos(2*D) * np.sin(R) * np.cos(S))

    # also convert to Harvard CMTSOLUTION format
    Mtt = Mxx
    Mpp = Myy
    Mrr = Mzz
    Mtp = Mxy * (-1)
    Mrt = Mxz
    Mrp = Myz * (-1)
    
    return(Mtt, Mpp, Mrr, Mtp, Mrt, Mrp);

def convolve(timeval, sem, hd_triangle, sdm_triangle, triangle=False):
    """
    Translated Specfem3D code from FORTRAN90 to Python by H. Latečki, 
    for Gaussian or triangle convolution
    Args:
        timeval (ndarray): Time values of the signal.
        sem (ndarray): Values of the signal.
        hd_triangle (float): Half duration of the triangle.
        sdm_triangle (float): Used to define a Gaussian with the 
        right exponent to mimic a triangle of equivalent half huration.
        triangle (boolean): Set True for triangle convolution;
        False for Gaussian convolution.
        datasave (boolean): Set True to save convolved signal.
    Returns:
        ndarray: Values of the convolved signal.
    """

    alpha = sdm_triangle/hd_triangle
    timeval = np.asarray(timeval)
    sem = np.asarray(sem)
    sem_fil = np.zeros(len(sem))
    nlines = len(sem)
    
    # compute the time step
    dt = timeval[1] - timeval[0]
    
    if triangle == True:
        N_j = np.ceil(hd_triangle/dt)
    else:
        N_j = np.ceil(1.5*hd_triangle/dt)
    sem_fil = []

    for i in range(0,nlines-1):
        a = 0
        for j in range(-int(N_j),int(N_j)):
            if i>j and (i-j)<=(nlines-1):
                tau_j = j*dt 
                # convolve with triangle
                if triangle==True: 
                    height = 1/ hd_triangle
                    if abs(tau_j)>hd_triangle:
                        source = 0
                    elif tau_j<0:
                        t1 = -N_j*dt
                        displ1 = 0
                        t2 = 0
                        displ2 = height
                        gamma = (tau_j - t1)/(t2 - t1)
                        source= (1 - gamma)*displ1 + gamma*displ2
                    elif tau_j>=0:
                        t1 = 0
                        displ1 = height
                        t2 = +N_j*dt
                        displ2 = 0
                        gamma = (tau_j - t1)/(t2 - t1)
                        source= (1 - gamma)*displ1 + gamma*displ2
                # convolve with Gaussian
                else:
                    exponentval = (alpha*alpha)*(tau_j*tau_j)
                    if exponentval<50:
                        source = alpha*np.exp(-exponentval)/np.sqrt(np.pi)
                    else:
                        source = 0
        
                a += sem[i-j]*source*dt
        sem_fil.append(a)
    sem_fil.append(a)
    
    # plot convolved and unconvolved signal
    fig = plt.figure(figsize=(18, 18))
    plt.rcParams.update({'font.size': 20})
    plt.grid(True)
    plt.title(file)
    plt.plot(timeval, sem,'cornflowerblue',label='before convolution')
    plt.plot(timeval, sem_fil,'tab:red',label='after convolution')
    plt.legend(loc='upper right')
    
    return sem_fil

def gaussian_taper(x, sigma=2.0):
    """
    Gaussian tapering function.
    Args:
        x (ndarray): Values to taper.
        sigma (float): Standard deviation.
    Returns:
        ndarray: Gaussian tapered values.
    """
    return np.exp(-(x**2)/(2 * sigma**2))

def calculate_energy_contributions(center_point, all_points, radius, in_percent, total_energy):
    """
    Calculates energy contributions based on a distance from the center 
    point. Within the radius, values follow Gaussian distrubution of distances. 
    Outside the radius, values exponentially decrease by the square of the distance.
    Args:
        center_point (ndarray): Coordinates of the center point
        (latitude, longitude, altitude in meters).
        all_points (ndarray): Coordinates of all data points
        (latitude, longitude, altitude in meters).
        radius (float): Radius in km from center point. 
        in_percent (float): Value between 0 and 1 determening how
        much energy is contained within radius from center point.
        total_energy (float): Total energy.
    Returns:
        contributions (ndarray): List of energy contributions and
        coordinates of the data points.
    """
    contributions = [];
    total_energy_within = total_energy * in_percent
    total_energy_outside = total_energy * (1-in_percent)
    
    for point in all_points:
        distance = np.linalg.norm(point - center_point)
        if distance<=radius:
            energy_contribution = total_energy_within * gaussian_taper(distance)
        else:
            energy_contribution = total_energy_outside * pow(distance,-2)
        
        contributions.append((energy_contribution, point))

    total_contributions = sum(contribution[0] for contribution in contributions)
    normalization_factor = total_energy / total_contributions if total_contributions != 0 else 1
    # percent output
    contributions = [((energy*normalization_factor), point) for energy, point in contributions]
    
    return contributions

def poly_taper(timeval, sem, taper_start, taper_end, order=4)
    """
    Calculates energy contributions based on a distance from the center 
    point. Within the radius, values follow Gaussian distrubution of distances. 
    Outside the radius, values exponentially decrease by the square of the distance.
    Args:
        timeval (ndarray): Time values of the signal.
        sem (ndarray): Values of the signal.
        taper_start (float): Start of the tapering interval (in seconds) 
        taper_end (float): End of the tapering interval (in seconds).
        order (int): Order of the polynominal function.
    Returns:
        tapered (ndarray): Tapered values of the signal.
    """
    delta = timeval[1] - timeval[0]
    orig = np.asarray(sem)
    time = np.asarray(timeval)
    fs = 1/delta  # Sampling frequency in Hz

    # Calculate corresponding sample indices
    start_index = int(taper_start * fs)
    end_index = int(taper_end * fs)

    # Define the length of the taper in seconds
    taper_length = taper_end - taper_start

    # Calculate the number of samples corresponding to taper length
    taper_samples = taper_length * fs

    # Create a tapering window using a polynomial function
    taper_window = np.zeros_like(time)
    taper_window[start_index:end_index] = np.power((np.arange(taper_samples-1)/float(taper_samples-1)),order)

    # Apply the tapering window to the data
    tapered = orig * taper_window

    # Set the values of the tapered data after the tapering interval to be the same as the original data
    tapered[end_index:] = orig[end_index:]

    return tapered
