# python
#stacked barplot

import matplotlib.pyplot as plt
from tabulate import tabulate

#set grey scale style

plt.style.use('grayscale')
#set characters parameters

SMALL_SIZE = 5
MEDIUM_SIZE = 8
BIGGER_SIZE = 10

#plt.rc('font', size=SMALL_SIZE) 
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

#get data tabulate (also in latex format)
print(tabulate(data, headers='keys', tablefmt='psql'))

+-------------------------------+---------------------+---------------------+
|                               |   A                           B
|-------------------------------+---------------------+---------------------|
| Actin filaments               |                   0 |                   0 |
| Cell Junctions                |                   0 |                   0 |
| Cytokinetic bridge            |                   0 |                   0 |
| Cytoplasmic bodies            |                   0 |                   0 |
| Cytosol                       |                   0 |                  16 |
| Endoplasmic reticulum         |                   0 |                   4 |
| Focal adhesion sites          |                   0 |                   0 |

print(tabulate(data, headers='keys', tablefmt='latex'))
'''
-h, --help show this message
-1, --header use the first row of data as a table header
-o FILE, --output FILE print table to FILE (default: stdout)
-s REGEXP, --sep REGEXP use a custom column separator (default: whitespace)
-F FPFMT, --float FPFMT floating point number format (default: g)
-f FMT, --format FMT set output table format; supported formats:
 "plain"
- "simple"
- "grid"
- "fancy_grid"
- "pipe"
- "orgtbl"
- "jira"
- "presto"
- "psql"
- "rst"
- "mediawiki"
- "moinmoin"
- "youtrack"
- "html"
- "latex"
- "latex_raw"
- "latex_booktabs"
- "textile"

#-------------------------------
#single stacked barplot
#-------------------------------

data.plot.bar(stacked=True,width=0.1)
#insert legend
plt.legend(loc='lower right') #legend_loc best, upper left, upper right, lower left, lower right
#set y limits step 1,100,1000
plt.yticks(np.arange(0, sum(data.iloc[0])+step, step))
#set x label rotation (eg 0,30,45,90)
plt.xticks(rotation=0)
#save fig with transparent background
plt.savefig('out.png', bbox_inches='tight',transparent=True)
#show graph 
plt.show()

data.columns=[['PnuclRNAcyto','PcytoRNANucl']]
print(tabulate(data, headers='keys', tablefmt='psql'))
print(tabulate(data, headers='keys', tablefmt='latex'))
ax = data[['PnuclRNAcyto','PcytoRNANucl']].plot.bar(rot=45)
ax.legend(loc=2) 
plt.show()

#---------------------------
#multiclass barplot
#---------------------------
data.columns=[['A','B']]
#prinnt data
print(tabulate(data, headers='keys', tablefmt='psql'))
print(tabulate(data, headers='keys', tablefmt='latex'))
#set xlabel rotation (eg.45)
ax = data[['A','B']].plot.bar(rot=45)
#set legend position 1,2,3,4 
ax.legend(loc=2) 
#save fig with transparent background
plt.savefig('out.png', bbox_inches='tight',transparent=True)
plt.show()


