import sys
import numpy as np
import math

lines = sys.stdin.readlines()

markers = ["Abstract test results:\n", "UDP test results:\n", "OMIM test results:\n"]

for marker in markers:
    ct = 0
    means = np.zeros(7)
    vars = np.zeros(7)

    for i,l in enumerate(lines):
        if i>0 and lines[i-1]==marker:
            ct+=1
            means += 100*np.array(map(float,l.strip().replace('\\','').split('&')))
    means/=ct
        
    for i,l in enumerate(lines):
        if i>0 and lines[i-1]==marker:
            vars += (means-100*np.array(map(float,l.strip().replace('\\','').split('&')))) ** 2
            #print l.replace('&','\t').replace('\\','')
    #print np.sqrt(vars/(ct-1))
    stds = np.sqrt(vars/(ct))
    for i,x in enumerate(means):
        print '%.1f $\pm$ %.1f &' % (x,stds[i]),
    print ""

r5 = []
r1 = []
for l in lines:
    if l.startswith("R@5"):
        r5.append(100*float(l.split("::")[-1]))
    if l.startswith("R@1"):
        r1.append(100*float(l.split("::")[-1]))
        
print '%.1f $\pm$ %.1f & %.1f $\pm$ %.1f' % (np.mean(r1),np.std(r1),np.mean(r5),np.std(r5)),
print ""

