# -*- coding: utf-8 -*-
"""
This Script reads the excel file: humanValues.xlsx
plots it, using pandas, then saves the plot in 
a png file using matplotlib.pyplot
"""



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def msd_straight_forward(r):
    shifts = np.arange(len(r))
    msds = np.zeros(shifts.size)    

    for i, shift in enumerate(shifts):
        diffs = r[:-shift if shift else None] - r[shift:]
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()

    return msds
    
def compute_msd(trajectory, t_step, coords=['x', 'y']):

    tau = trajectory['t'].copy()
    shifts = np.floor(tau / t_step).astype(np.int)
    msds = np.zeros(shifts.size)
    msds_std = np.zeros(shifts.size)

    for i, shift in enumerate(shifts):
        diffs = trajectory[coords] - trajectory[coords].shift(-shift)
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()
        msds_std[i] = sqdist.std()

    msds = pd.DataFrame({'msds': msds, 'tau': tau, 'msds_std': msds_std})
    return msds

os.chdir("E:\Code\pythonPractice\MSD")
df = pd.read_excel("E:\\Code\pythonPractice\\MSD\\file.xls")
#Filter for entries that have Particle ID = 2, then draw x1 and y1 columns
#toPlot = df[df['Particle ID'] == 2][[' X1 (pixels)', ' Y1 (pixels)']]
#plot = toPlot.plot(x=" X1 (pixels)", y=" Y1 (pixels)", kind="scatter")
#fig = plot.get_figure()
#fig.savefig("particlePlot.png")


#Group Data by id
grouped = df[[' X1 (pixels)', ' Y1 (pixels)',' X2 (pixels)', ' Y2 (pixels)', ' Particle Size (nm)']].groupby(df['Particle ID'])
counts = grouped.size() #Creates Series with sizes for each group
validData = []
lowestTime = 5000
dt = .04
msdData = []
for name, group in grouped:
    if counts[name] > 20 and group.iloc[0][' Particle Size (nm)'] > 50:
        #Group passes -> Do things
        #print("Group %s is valid data" %name)
        validData.append({'id': "%s" %name, 'data': group})
        r = group[[' X1 (pixels)', ' Y1 (pixels)']]
        rSize = r.size/2
        maxTime = dt*rSize
        if maxTime < lowestTime:
            lowestTime = maxTime
        t = np.linspace(0,maxTime,rSize)
        traj = pd.DataFrame({'t':t,'x':r[' X1 (pixels)'], 'y':r[' Y1 (pixels)']})
        msds = compute_msd(traj, t_step=dt)
        msdData.append({'id':"%s" %name, 'data':msds})
    #else:
        #Discarded  Data
        #print("Group %s is INVALID data" %name)
msdAvgs = pd.DataFrame(columns=['t', 'msdAvg'])
for time in np.arange(0, lowestTime, dt):
    sum = 0.0
    avg = 0.0
    for point in msdData:
        step = time % .04
        print(point['data']['msds'].head())
        #step is off, due to mis-assigned indices (indices carried from initial import)
        plus = point['data'].get_value(step, 'msds')
        # sum += point.get('data')['msds'][step]
    #avg = sum % len(msdData)
    #msdAvgs.append({'t': time, 'msdAvg': avg})
        
        
    



