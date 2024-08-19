#!/usr/bin/env python
# coding: utf-8

from obspy import read
import modules as m

# Example Obspy trace
st = read("https://examples.obspy.org/loc_RJOB20050831023349.z")
trZ = st[0]

# List of filters for Ormsby filtering 
filters = [[0.08,0.1,0.15,0.17], [0.15,0.17,0.27,0.3], [0.27,0.3,0.45,0.5],\
          [0.45,0.5,0.8,0.9],[0.80,0.90,1.30,1.50],[1.30,1.50,1.90,2.20],\
          [1.90,2.20,2.80,3.50],[2.80,3.50,5.00,6.00],[5.00,6.00,8.50,10.00]]
# List of smoothing filters
smoothing_filt = [0.06, 0.11, 0.14, 0.17,0.20,0.23,0.26,0.28,0.30]

# Path to output folder 
#folder = './test'

durations = m.strong_motion_duration(trace=trZ,filters=filters,s_filters=smoothing_filt)#,save_fig=True,fig_folder=folder
#print(durations)