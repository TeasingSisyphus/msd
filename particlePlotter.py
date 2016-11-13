# -*- coding: utf-8 -*-
"""
This Script reads the excel file: humanValues.xlsx
plots it, using pandas, then saves the plot in 
a png file using matplotlib.pyplot
"""



import os
import pandas
import matplotlib.pyplot as plt

os.chdir("E:\Code\pythonPractice\MSD")
df = pandas.read_excel("E:\\Code\pythonPractice\\MSD\\file.xls")
#Filter for entries that have Particle ID = 2, then draw x1 and y1 columns
toPlot = df[df['Particle ID'] == 2][[' X1 (pixels)', ' Y1 (pixels)']]
plot = toPlot.plot(x=" X1 (pixels)", y=" Y1 (pixels)", kind="scatter")
fig = plot.get_figure()
fig.savefig("particlePlot.png")


#Group Data by id
grouped = df[[' X1 (pixels)', ' Y1 (pixels)',' X2 (pixels)', ' Y2 (pixels)', ' Particle Size (nm)']].groupby(df['Particle ID'])
counts = grouped.size() #Creates Series with sizes for each group
validData = []
for name, group in grouped:
    if counts[name] > 20 and group.iloc[0][' Particle Size (nm)'] > 50:
        #Group passes -> Do things
        print("Group %s is valid data" %name)
        validData.append({"%s" %name: group})
    else:
        #Discarded  Data
        print("Group %s is INVALID data" %name)

        
    