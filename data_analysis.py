import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy import interpolate
from scipy.interpolate import griddata
import matplotlib
import numpy

SAVE_FILE = r'C:\Users\Usuario\Documents\POLI\TCC\cargobikeresults.xlsx'

def get_src():
    '''

    Returns 3 pandas DataFrame, from each Setup
    -------
    src1 : pandas DataFrame
        Setup 1.
    src2 : pandas DataFrame
        Setup 2.
    src3 : pandas DataFrame
        Setup 3.

    '''
    src1 = pd.read_excel(SAVE_FILE, sheet_name = 'ar_32setup1') #Setup 
    src2 = pd.read_excel(SAVE_FILE, sheet_name = 'ar_32setup2') #Setup 2
    src3 = pd.read_excel(SAVE_FILE, sheet_name = 'ar_32setup3') #Setup 3

    src1 = treatment_src(src1)
    src2 = treatment_src(src2)
    src3 = treatment_src(src3)
    
    return src1, src2, src3
        
def treatment_src(src):
    '''

    Parameters
    ----------
    src : pandas DataFrame

    Returns
    -------
    src : pandas DataFrame
        Return DF after setting up some data treatments.

    '''
    src = src[~src['lowest stable speed'].isnull()]
    
    src['max speed'] = src['lowest stable speed'] + src['stable speed range'] #Setup 2 e 3
    src = src[src['max speed'] < 40] #m/s
    
    src = src.rename(columns = {'bike floor height':'Floor Height (m)',
                         'cargo space': 'Cargo Space (m)',
                         'steering angle': 'Steering Angle (º)',
                         'lowest stable speed': 'Lowest Speed (m/s)',
                         'stable speed range': 'Stable Range (m/s)',
                         'max speed': 'Max Speed (m/s)'})
    
    src['Weight Capacity (kg)'] = src['Cargo Space (m)']*43.4
    
    return src
    
def get_plot_3d_all(src1, src2, src3, cte, var, h, z):    
    '''
    Plot 3D Graphs for all 3 Setups

    Parameters
    ----------
    src1 : pandas DataFrame
    
    src2 : pandas DataFrame

    src3 : pandas DataFrame

    cte : STRING
        String do valor constante.
    var : STRING
        String dos valores variáveis.
    h : FLOAT
        Float do valor constante.
    z : STRING
        String do output

    Returns
    -------
    None.

    '''

    df1 = src1[(src1[cte] == h)] 
    x1_1 = np.linspace(df1[var[0]].min(), df1[var[0]].max(), len(df1[var[0]].unique()))
    y1_1 = np.linspace(df1[var[1]].min(), df1[var[1]].max(), len(df1[var[1]].unique()))
    x2_1, y2_1 = np.meshgrid(x1_1, y1_1)
    z2_1 = griddata((df1[var[0]], df1[var[1]]), df1[z], (x2_1, y2_1), method='cubic')
    tck1 = interpolate.bisplrep(y2_1, x2_1, z2_1, s=0.2)
    znew1 = interpolate.bisplev(y2_1[:,0], x2_1[0,:], tck1)
    
    df2 = src2[(src2[cte] == h)] 
    x1_2 = np.linspace(df2[var[0]].min(), df2[var[0]].max(), len(df2[var[0]].unique()))
    y1_2 = np.linspace(df2[var[1]].min(), df2[var[1]].max(), len(df2[var[1]].unique()))
    x2_2, y2_2 = np.meshgrid(x1_2, y1_2)
    z2_2 = griddata((df2[var[0]], df2[var[1]]), df2[z], (x2_2, y2_2), method='cubic')
    tck2 = interpolate.bisplrep(y2_2, x2_2, z2_2, s=0.2)
    znew2 = interpolate.bisplev(y2_2[:,0], x2_2[0,:], tck2)
    
    df3 = src3[(src3[cte] == h)] 
    x1_3 = np.linspace(df3[var[0]].min(), df3[var[0]].max(), len(df3[var[0]].unique()))
    y1_3 = np.linspace(df3[var[1]].min(), df3[var[1]].max(), len(df3[var[1]].unique()))
    x2_3, y2_3 = np.meshgrid(x1_3, y1_3)
    z2_3 = griddata((df3[var[0]], df3[var[1]]), df3[z], (x2_3, y2_3), method='cubic')
    tck3 = interpolate.bisplrep(y2_3, x2_3, z2_3, s=0.2)
    znew3 = interpolate.bisplev(y2_3[:,0], x2_3[0,:], tck3)
    
    '''========================================================'''
    if z == 'Stable Range (m/s)':
        vmax = 18
        vmin = 2
    elif z == 'Lowest Speed (m/s)':
        vmax = 12
        vmin = 4
    elif z == 'Max Speed (m/s)':
        vmax = 22
        vmin = 6
    
    fig = plt.figure(figsize=plt.figaspect(0.25))
    
    ''' ==== First Subplot ===== '''
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf = ax1.plot_surface(x2_1, y2_1, znew1, rstride=1, cstride=1, cmap=cm.coolwarm, vmax = vmax, vmin = vmin,
                           linewidth=0.1, antialiased=True, alpha=None)
    ax1.title.set_text('Setup 1')
    ax1.set_zlim(vmin, vmax - 2)
    
    ''' ==== Second Subplot ===== '''
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    surf = ax2.plot_surface(x2_2, y2_2, znew2, rstride=1, cstride=1, cmap=cm.coolwarm, vmax = vmax, vmin = vmin,
                           linewidth=0.1, antialiased=True, alpha=None)
    
    ax2.title.set_text('Setup 2')
    ax2.set_zlim(vmin, vmax - 2)
    
    ''' ==== Third Subplot ===== '''
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    surf = ax3.plot_surface(x2_3, y2_3, znew3, rstride=1, cstride=1, cmap=cm.coolwarm, vmax = vmax, vmin = vmin,
                           linewidth=0.1, antialiased=True)    
    ax3.title.set_text('Setup 3')
    ax3.set_zlim(vmin, vmax - 2)
    
    ''' ------------------------- '''
    ax1.zaxis.set_major_locator(LinearLocator(10))
    ax1.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    ax2.zaxis.set_major_locator(LinearLocator(10))
    ax2.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    ax3.zaxis.set_major_locator(LinearLocator(10))
    ax3.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
    
    ax1.set_xlabel(str(var[0]))
    ax1.set_ylabel(str(var[1]))
    ax2.set_xlabel(str(var[0]))
    ax2.set_ylabel(str(var[1]))
    ax3.set_xlabel(str(var[0]))
    ax3.set_ylabel(str(var[1]))
    
    ax1.yaxis._axinfo['label']['space_factor'] = 3.0
    ax2.yaxis._axinfo['label']['space_factor'] = 3.0
    ax3.yaxis._axinfo['label']['space_factor'] = 3.0
    
    if z == 'Lowest Speed (m/s)':
        ax3.set_zlabel('Velocidade mínima (m/s)', rotation = 0)
    elif z == 'Stable Range (m/s)':
        ax3.set_zlabel('Range estável (m/s)', rotation = 0)
    
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    colorbar = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm.coolwarm)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.01, 0.7])
    fig.colorbar(colorbar, shrink=1, aspect=15, orientation="vertical", pad=0.25, cax = cbar_ax)

    filename = '3D_' + z[:-6] + '_' + cte[:5] + '_' + str(h) + '_' + var[0][:5] + '-' + var[1][:5] + '.png'
    plt.savefig(filename, dpi=300)
    plt.show()
    
def get_plot_2d_all(src1, src2, src3, cte, var, h, z, SA):
    '''
    Plot 2D Graphs for all 3 Setups

    Parameters
    ----------
    src1 : pandas DataFrame
    
    src2 : pandas DataFrame

    src3 : pandas DataFrame

    cte : STRING
        String do valor constante.
    var : STRING
        String dos valores variáveis.
    h : FLOAT
        Float do valor constante.
    z : STRING
        String do output
    SA : list
        Contains var[1] simulation values

    Returns
    -------
    None.

    '''
    # SA (Steering Angle) is a table
    colors = ['c', 'm', 'y', 'b', 'g' 'r']
    
    vmin = min(min(src1[z]), min(src2[z]), min(src3[z]))
    vmax = min(max(max(src1[z]), max(src2[z]), max(src3[z])), 20)
    for j in range(3):
        if j == 0:
            src = src1
        elif j == 1:
            src = src2
        elif j == 2:
            src = src3
            
        src = src[(src[cte] == h)]
        
        if var[0] == 'Cargo Space (m)':
            src[var[0]] = src[var[0]].round(2)
        
        for i in SA:
            df = src[src[var[0]] == i]
            
            x = df[var[1]]
            y = df[z]

            '''==== Regressão quadrática ===='''
            mymodel = numpy.poly1d(numpy.polyfit(x, y, 2))
            myline = numpy.linspace(x.min(), x.max(), 100)
            plt.plot(myline, mymodel(myline), colors[j], label = 'Setup ' + str(j + 1) )
            print('Setup ' + str(j) + ': ')
            print(np.poly1d(mymodel))
            
            '''==== Regressão Linear ====='''
            # slope, intercept, r_value, p_value, std_err = stats.linregress(x.astype(float), y.astype(float))
            # line = slope*x+intercept
            # print('Regressão Linear para '+str(var[0])+' = ' + str(i) + ' e ' + str(cte) + '= ' + str(h) + ': \
            #       y = ' + str(slope) + '*x + ' + str(intercept))
            # plt.plot(x, line, colors[count], label= str(var[0])+': ' + str(i))
            # plt.scatter(x,y, color="k", s=3.5)
            
            plt.legend(fontsize=10, loc = 'upper left')
                    
        ax = plt.gca()
        ax.set_ylim([vmin, vmax])
        
        plt.xlabel(str(var[1]), fontsize=12)
        
    if z == 'Lowest Speed (m/s)':
        plt.ylabel('Minimum Stable Speed (m/s)', fontsize=12)
        plt.title(f'{cte}: {h}; {var[0]}: {SA[0]}')
    elif z == 'Stable Range (m/s)':
        plt.ylabel('Stable Speed Range (m/s)', fontsize=12)
        plt.title(f'{cte}: {h}; {var[0]}: {SA[0]}')
    elif z == 'Max Speed (m/s)':
        plt.ylabel('Max Stable Speed (m/s)', fontsize=12)
        plt.title(f'{cte}: {h}; {var[0]}: {SA[0]}')
        
    filename = '2D-' + z[:-6] + '_' + cte[:5] + '_' + str(h) + '_' + var[0][:5] + '-' + var[1][:5] + '.png'
    plt.savefig(filename, dpi=400)
    plt.show()
    
def get_plot_2d(src, cte, var, h, z, SA):
    '''
    Plot 2D Graphs for a specific Setup

    Parameters
    ----------
    src : pandas DataFrame
    
    cte : STRING
        String do valor constante.
    var : STRING
        String dos valores variáveis.
    h : FLOAT
        Float do valor constante.
    z : STRING
        String do output
    SA : list
        Contains var[1] simulation values
        
    Returns
    -------
    None.

    '''
    # SA (Steering Angle) is a table
    colors = ['c', 'm', 'y', 'b', 'r']
    
    vmin = min(src1[z])
    vmax = min(max(src1[z]), 20)

    src = src[(src[cte] == h)]
    
    if var[0] == 'Cargo Space (m)':
        src[var[0]] = src[var[0]].round(2)
    
    count = 0
    for i in SA:
        df = src[src[var[0]] == i]
        
        x = df[var[1]]
        y = df[z]

        '''==== Regressão quadrática ===='''
        mymodel = numpy.poly1d(numpy.polyfit(x, y, 2))
        myline = numpy.linspace(x.min(), x.max(), 100)
        plt.plot(myline, mymodel(myline), colors[count], label = var[0] + ' ' + str(i) )
        print(var[0] + ' ' + str(i) + ': ')
        print(np.poly1d(mymodel))
        
        '''==== Regressão Linear ====='''
        # slope, intercept, r_value, p_value, std_err = stats.linregress(x.astype(float), y.astype(float))
        # line = slope*x+intercept
        # print('Regressão Linear para '+str(var[0])+' = ' + str(i) + ' e ' + str(cte) + '= ' + str(h) + ': \
        #       y = ' + str(slope) + '*x + ' + str(intercept))
        # plt.plot(x, line, colors[count], label= str(var[0])+': ' + str(i))
        # plt.scatter(x,y, color="k", s=3.5)
        
        plt.legend(fontsize=10, loc = 'upper left')
                    
        ax = plt.gca()
        ax.set_ylim([vmin, vmax])
        
        plt.xlabel(str(var[1]), fontsize=12)
        count += 1
        
    if z == 'Lowest Speed (m/s)':
        plt.ylabel('Minimum Stable Speed (m/s)', fontsize=12)
        plt.title(f'Setup 1; {cte}: {h};')
    elif z == 'Stable Range (m/s)':
        plt.ylabel('Stable Speed Range (m/s)', fontsize=12)
        plt.title(f'Setup 1; {cte}: {h};')
    elif z == 'Max Speed (m/s)':
        plt.ylabel('Max Stable Speed (m/s)', fontsize=12)
        plt.title(f'Setup 1; {cte}: {h};')
        
    filename = '2D-' + z[:-6] + '_' + cte[:5] + '_' + str(h) + '_' + var[0][:5] + '-' + var[1][:5] + '.png'
    plt.savefig(filename, dpi=400)
    plt.show()

''' ======= PARAMS =======================

        0.12 < 'Floor Height (m)' < 0.28
        0.4 < 'Cargo Space (m)' < 1.4
        15 < 'Steering Angle (º)' < 28
        
        z = ['Lowest Speed (m/s)', 'Stable Range (m/s)', 'Max Speed (m/s)']
        var/cte = ['Cargo Space (m)', 'Steering Angle (º)', 'Floor Height (m)']
'''

''' Possíveis valores de SA '''
SA = [28, 24, 20, 16] # Para LAMBDA
SA = [0.12, 0.16, 0.20, 0.24] # Para Floor Height


''' ====== INIT ======== '''
src1, src2, src3 = get_src()


''' ====== GRAFICOS 1 - 3D ======== '''
get_plot_3d_all(src1, src2, src3, cte = 'Floor Height (m)', 
                var = ['Cargo Space (m)', 'Steering Angle (º)'], 
                h = 0.15, 
                z = 'Lowest Speed (m/s)')

get_plot_3d_all(src1, src2, src3, cte = 'Floor Height (m)', 
                var = ['Cargo Space (m)', 'Steering Angle (º)'], 
                h = 0.15, 
                z = 'Stable Range (m/s)')


''' ======= GRAFICOS 2 - 2D ======== '''
get_plot_2d_all(src1, src2, src3, cte = 'Floor Height (m)', 
                var = ['Steering Angle (º)', 'Cargo Space (m)'], 
                h = 0.15, 
                z = 'Lowest Speed (m/s)',
                SA = [20])

get_plot_2d_all(src1, src2, src3, cte = 'Floor Height (m)', 
                var = ['Steering Angle (º)', 'Cargo Space (m)'], 
                h = 0.15, 
                z = 'Stable Range (m/s)',
                SA = [20])

''' ======== IMPACTO ALTURA ========= '''
get_plot_3d_all(src1, src2, src3, cte = 'Steering Angle (º)', 
                var = ['Cargo Space (m)', 'Floor Height (m)'], 
                h = 20, 
                z = 'Lowest Speed (m/s)')

get_plot_2d_all(src1, src2, src3, cte = 'Steering Angle (º)', 
                var = ['Cargo Space (m)', 'Floor Height (m)'], 
                h = 20, 
                z = 'Lowest Speed (m/s)',
                SA = [0.8]) 

get_plot_3d_all(src1, src2, src3, cte = 'Steering Angle (º)', 
                var = ['Cargo Space (m)', 'Floor Height (m)'], 
                h = 20, 
                z = 'Stable Range (m/s)')

get_plot_2d_all(src1, src2, src3, cte = 'Steering Angle (º)', 
                var = ['Cargo Space (m)', 'Floor Height (m)'], 
                h = 20, 
                z = 'Stable Range (m/s)',
                SA = [0.8]) 

get_plot_2d_all(src1, src2, src3, cte = 'Steering Angle (º)', 
                var = ['Cargo Space (m)', 'Floor Height (m)'], 
                h = 20, 
                z = 'Max Speed (m/s)',
                SA = [0.8]) 

''' ======== ESTUDO SA ============ '''
get_plot_2d(src1, cte = 'Floor Height (m)', 
                var = ['Steering Angle (º)', 'Cargo Space (m)'], 
                h = 0.15, 
                z = 'Stable Range (m/s)',
                SA = [20, 19, 18, 17, 16])

''' ======== ESTUDO WEIGHT CAPACITY ========== '''
get_plot_2d_all(src1, src2, src3, cte = 'Floor Height (m)', 
               var = ['Steering Angle (º)', 'Weight Capacity (kg)'], 
               h = 0.15, 
               z = 'Max Speed (m/s)',
               SA = [16])

