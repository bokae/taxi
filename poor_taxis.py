
# coding: utf-8

# In[1]:


from visualize2 import ResultParser
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider
from geometry import City
get_ipython().run_line_magic('matplotlib', 'inline')

# collecting result data
rp = ResultParser('0831_base')
df = rp.prepare_all_data(force=True)


# In[2]:


def plotting_percentile(p,row):
    """
    Where are the taxis from the dataset row, that have less income than the p-th percentile?
    """
    
    row = df.loc[row]
    thres = np.percentile(row['trip_avg_price'],p)
    position = [row['position'].iloc[i] for i,price in enumerate(row['trip_avg_price']) if price<thres]
    
    
    print("TAXI POSITIONS AT END OF SIMULATION")
    
    fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(15,5))
    # taxis
    ax[0].hist2d(
        [x[0] for x in position],[x[1] for x in position],
        range = [[0,row['n']],[0,row['m']]],
        bins = [row['n'],row['m']],
        cmap = 'magma'
    )
    ax[0].axis('equal')
    ax[0].set_xlim(0,row['n'])
    ax[0].set_ylim(0,row['m'])
    ax[0].set_title('Taxis below the %dth percentile' % p)
    
    if not hasattr(row['request_destination_distributions'],'dtype'):
        del row['request_destination_distributions']
        
    # request origin and destination distributions
    C = City(**row)
    
    o = []
    d = []
    for i in range(100000):
        ox,oy,dx,dy = C.create_one_request_coord()
        o.append([ox,oy])
        d.append([dx,dy])

    ax[1].hist2d(np.array(o)[:,0],np.array(o)[:,1],
        range = [[0,row['n']],[0,row['m']]],
        bins = [row['n'],row['m']]
    )
    ax[1].axis('equal')
    ax[1].set_xlim(0,row['n'])
    ax[1].set_ylim(0,row['m'])
    ax[1].set_title('Request origin distribution, R=%.2f' % row['R'])
              
    plt.hist2d(np.array(d)[:,0],np.array(d)[:,1],
        range = [[0,row['n']],[0,row['m']]],
        bins = [row['n'],row['m']]
    )
    ax[2].axis('equal')
    ax[2].set_xlim(0,row['n'])
    ax[2].set_ylim(0,row['m'])
    ax[2].set_title('Request destination distribution, R=%.2f' % row['R'])
    plt.show()
    
    print("INCOME VS TAXI ")
    
    fig,axn = plt.subplots(nrows=1,ncols=4,figsize=(20,5))
    for i,ratio in enumerate(['ratio_cruising','ratio_waiting', 'ratio_serving', 'ratio_to_request']):
        axn[i].plot(row['trip_avg_price'],row[ratio],'bo')
        axn[i].set_title(ratio)
        axn[i].set_ylim(0,1)
    plt.show()


# In[3]:


interact(plotting_percentile,
         p = IntSlider(min=0,max=100, value=100,step=5),
         row = IntSlider(min=0,max=df.index.max(),value=0))

