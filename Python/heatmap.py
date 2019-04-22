
import cPickle
import numpy as np
import codecs
import pandas as pd
import matplotlib as mpl
#from matplotlib import pyplot as plt
import seaborn as sns
scores = {'pass': 1,'warn': 0,'fail': -1}
df= pd.read_csv('/home/bonafede/CRG/with_ste/rna_seq/results/quality_control/all_mod_scores.csv',sep=",",low_memory=False)
df.index=df['Unnamed: 0']
del df['Unnamed: 0']

#cmap2 = mpl.colors.ListedColormap(sns.cubehelix_palette(n_colors=3, start=0, rot=0.4, gamma=1, hue=0.8, light=0.85, dark=0.15, reverse=False))
df.rename(index={0:'Date'}
cmap2 = mpl.colors.ListedColormap(['red','orange',"#2ecc71"])
heatmap=sns.heatmap(df,linewidths=.5,vmin=-1, vmax=1,cmap=cmap2)
plt.show()


#addition elements
cbar_kws={"orientation": "horizontal"})






#create a color palette with the same number of colors as unique values in the Source column
network_pal = sns.light_palette('red', 3)

#Create a dictionary where the key is the category and the values are the
#colors from the palette we just created
network_lut = dict(zip([0,1,-1], network_pal))

#get the series of all of the categories
networks = df['Per tile_sequence quality']

#map the colors to the series. Now we have a list of colors the same
#length as our dataframe, where unique values are mapped to the same color
network_colors = pd.Series(networks).map(network_lut)

#plot the heatmap with the 16S and ITS categories with the network colors
#defined by Source column
sns.clustermap(df[['16S', 'ITS']], row_colors=network_colors, cmap='BuGn_r')


