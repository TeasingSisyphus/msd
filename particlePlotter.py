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
from sklearn import datasets, linear_model


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
dataDir = "E:\\Code\\pythonPractice\\MSD\\data\\"
dataList = os.listdir(dataDir)
dataFileCount = len(dataList)

for file in dataList:
    filePath = dataDir + file

    
    df = pd.read_excel(filePath)
    #Filter for entries that have Particle ID = 2, then draw x1 and y1 columns
    #toPlot = df[df['Particle ID'] == 2][[' X1 (pixels)', ' Y1 (pixels)']]
    #plot = toPlot.plot(x=" X1 (pixels)", y=" Y1 (pixels)", kind="scatter")
    #fig = plot.get_figure()
    #fig.savefig("particlePlot.png")
    
    
    #Group Data by id
    grouped = df[[' X1 (pixels)', ' Y1 (pixels)',' X2 (pixels)', ' Y2 (pixels)', ' Particle Size (nm)']].groupby(df['Particle ID'])
    counts = grouped.size() #Creates Series with sizes for each group
    validData = []
    lowestTime = 1000
    msdRowLimit = 15
    dt = .04
    msdData = []
    for name, group in grouped:
        if counts[name] > 100 and group.iloc[0][' Particle Size (nm)'] > 50:
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
    #msdAvgs = pd.DataFrame(columns=['t', 'msdAvg'])
    #for time in np.arange(0, lowestTime, dt):
    times = []
    avgs = []
    #For each time interval, calculate average MSD across all particles
    for index in range(0, msdRowLimit):
        sum = 0.0
        avg = 0.0
        for point in msdData:
            #step = time % .04
            #print(point['data'])
            diff = point['data']['msds'].iloc[index]
            sum = sum + diff
    
        avg = sum/len(msdData)
        time = index * dt
        times.append(time)
        avgs.append(avg)
        print("Time: %s Avg: %s" %(time, avg))
    
    #msdAvgs = pd.Series(data=avgs, index=times)
    msdAvgs = pd.DataFrame(data=avgs, index=times, columns=["Avg MSD"])
    msdAvgs.reset_index(inplace=True)
    msdAvgs.columns = ["Delta Time in Seconds", "Avg MSD"]
    print(msdAvgs)
    scatterPlot = msdAvgs.plot(kind="scatter", x="Delta Time in Seconds", y="Avg MSD", title="Average MSD across %s" %filePath)
    scatterPlot.set_xlabel("Delta Time in Seconds")
    scatterPlot.set_ylabel("Avg MSD")
    fig = scatterPlot.get_figure()
    fig.savefig("./output/scatterplots/MSD Graph of %s.png" %file)
    msdAvgs.columns = ['a','b']
    x = msdAvgs['a'].values
    y = msdAvgs['b'].values
    length = len(x)
    msdAvgs.columns = ["Delta Time in Seconds", "Avg MSD"]
    x = x.reshape(length,1)
    y= y.reshape(length,1)

    regr = linear_model.LinearRegression()
    fitXY = regr.fit(x,y)
    regPlot = plt.plot(x, regr.predict(x), color='blue',linewidth=2)
    regFig = scatterPlot.get_figure()
    regFig.savefig("./output/regplots/Regression MSD Graph of %s.png" %file)
        
        
    



