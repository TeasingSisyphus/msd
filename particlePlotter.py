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
from scipy import stats 
import math

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

def save(x,y,sampleName,fileName,dataDir):
    #Calculate polynomial regression
    print("calculating regression plots")
    x1 = x.reshape(1,len(x))
    y1 = y.reshape(1,len(y))
    x2 = x1[0]
    y2 = y1[0]
    m,b,rval,pval,stderr = stats.linregress(x2,y2)
    plt.clf()
    p1,r1,a1,b1,c1 = np.polyfit(x2,y2,1, full=True)
    p2,r2,a2,b2,c2 = np.polyfit(x2,y2,2, full=True)    
    plot1 =plt.plot(x2,y2,'o', label="AVG MSD Degree Reg", linewidth=2)
    plt.plot(x2,np.polyval(p1,x2), 'm--')
    plt.plot(x2, np.polyval(p2,x2), 'b:')
    plt.xlabel("Time in Seconds")
    plt.ylabel("Avg MSD") 
    fig2=plot1[0].figure   
    fig2.text(.18, .7, "r2_lin value: %s" %rval)
    fig2.text(.18,.75, "quad residual: %s" %r2)
    fig2.suptitle("AVG MSD %s Reg" %sampleName)
    fig2.savefig("./output/%s degree Reg MSD.png" %  fileName)
    print("file '%s' saved to '%s' directory " %(fileName, dataDir+'/output/'))

def go(dataDir,dataList,inputsAnalysis):

    print("starting analysis with %s parameters." %(inputsAnalysis))
    minSize=inputsAnalysis[0]
    minCount = inputsAnalysis[1]
    msdRowLimit = inputsAnalysis[2]
    dt = inputsAnalysis[3]
    for fileName in dataList:
        print("working on file %s" %fileName)
        filePath = dataDir + fileName
        tmp = fileName.split(' ')
        sampleName = tmp[0]
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
        msdData = []
        acceptedParticles = []
        for name, group in grouped:
           
            if counts[name] > float(minCount) and group.iloc[0][' Particle Size (nm)'] > float(minSize):
                #Group passes -> Do things
                #print("Group %s is valid data" %name)
                acceptedParticles.append(name)
                validData.append({'id': "%s" %name, 'data': group})
                r = group[[' X1 (pixels)', ' Y1 (pixels)']]
                rSize = r.size/2
                maxTime = dt*rSize
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
        print("The accepted particle id's from file %s are: %s" %(fileName, acceptedParticles))
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
            
        
        #msdAvgs = pd.Series(data=avgs, index=times)
        msdAvgs = pd.DataFrame(data=avgs, index=times, columns=["Avg MSD"])
        msdAvgs.reset_index(inplace=True)
        msdAvgs.columns = ["Delta Time in Seconds", "Avg MSD"]

        #Format data for regression
        msdAvgs.columns = ['a','b']
        x = msdAvgs['a'].values
        y = msdAvgs['b'].values
        length = len(x)
        msdAvgs.columns = ["Delta Time in Seconds", "Avg MSD"]
        x = x.reshape(length,1)
        y= y.reshape(length,1)
        save(x,y,sampleName,fileName,dataDir)

           
  

        

print("starting")
os.chdir("C:\\Users\\ashis\\Documents\\code\\msd\\")
dataDir = "C:\\Users\\ashis\\Documents\\code\\msd\\data\\"
dataList = os.listdir(dataDir)
dataFileCount = len(dataList)


    
#Command line UI
print("There are %s input data files in the input directory" %dataFileCount)
res = input("Do you want to continue with analysis? (Answer with yes or no)  :  ")

if res.lower() == "yes":
    res2 = input("please the follwoing information (comma-separated): minimum particle size, minimum count per particle : ")
    inputs = res2.split(",")
    res3 = input("you have selected %s d-nm, %s counts as your minimun critera. Do you want to continue?   " %(inputs[0],inputs[1]))
    if res3.lower() == "yes":
        minSize = inputs[0]
        minCount = inputs[1]
    
        res4 = input("Analysis parameters: delta-t, total time (0.1-1 sec)")
        input3 = res4.split(",")
        dt= float(input3[0])
        msdTotalTime = float(input3[1])
        msdRowLimit = math.ceil(msdTotalTime/dt)
        msdRowLimit = int(msdRowLimit)
        inputsAnalysis = [ minSize, minCount,msdRowLimit, dt]
        go(dataDir, dataList, inputsAnalysis)
    else:
        res4 = input("Do you want to change your inputs?")
        if res4.lower() == "yes":
            inputs2 = input("please the follwoing information (comma-separated): minimum particle size, minimum count per particle , total row limit  :  ")
        else:
            print("Thank you for using our MSDplotter")
elif res.lower() == "justgo":
    
    inputsAnalysis= [50,100,14,.04]
    
    minSize = inputsAnalysis[0]
    minCount = inputsAnalysis[1]
    msdRowLimit = inputsAnalysis[2]
    dt = inputsAnalysis[3]
    go(dataDir, dataList, inputsAnalysis)
else:
    print("Thank you for browising MSDplotter")            



