import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
import csv

#=========data-settings============
dataname = str(sys.argv[1])
pp = PdfPages(dataname +'.pdf')
#=============================

row_data = np.genfromtxt(dataname+'.csv', delimiter=',')
avg_auc = row_data[-1]
ensemble_auc = row_data[0:len(row_data)]
base = np.genfromtxt(dataname+'_base.csv', delimiter=',')

plt.figure()
plt.boxplot(ensemble_auc)
plt.plot(1,base,'rs')
plt.plot(1,avg_auc,'bo')
#plt.xlim([0.0, 1.0])
#plt.ylim([0, 1])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
#plt.legend(loc="lower right")
plt.savefig(pp, format='pdf')
#plt.show()
pp.close()
