#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')

def plot_sgram(T,V,comp):
    """
    Plots timeseries.
    Args:
        T (ndarray): List of the times of the timeseries.
        V (ndarray): List of the timeseries values.
        comp (string): Component name.
    """

    fig = plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 14})
    plt.plot(T,V,'r')
    plt.ylabel('timeseries, component 'comp)
    plt.xlabel('time')
    plt.xlim(min(T),max(T))
    plt.ylim(-1.05*max(abs(V)),1.05*max(abs(V)))
    plt.grid()
    fig.savefig('./'+comp+'/'+stations[k]+'.png',bbox_inches='tight')
    plt.close(fig)
    plt.cla()
    plt.clf()

    return

F = open("./STATIONS_FINAL.csv", 'r')
stations = []; latitude = []; longitude = []

for line in F: 
    sta, lat, lon = line.split(',')
    stations.append(sta)
    latitude.append(float(lat))
    longitude.append(float(lon))
F.close()

for k in range(0,len(stations)):
    V1 = []; V2 = []; V3 = []; T = [];

    F = open("./files/"+stations[k]+".txt", 'r')
    lines = F.readlines()
    for i in range(5,len(lines)): 
        t, vz, vn, ve = lines[i].split(',')
        T.append(float(t))
        V1.append(float(vz))
        V2.append(float(vn))
        V3.append(float(ve))
    F.close()
    
    V1 = np.asarray(V1)
    V2 = np.asarray(V2)
    V3 = np.asarray(V3)
    
    plot_sgram(T,V1,'Z')
    plot_sgram(T,V2,'N')
    plot_sgram(T,V3,'E')