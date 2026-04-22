from unittest import result
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from tqdm import tqdm
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
import os as os
import joblib
from matplotlib.patches import BoxStyle
from pynvml import *
from scipy.stats import percentileofscore
from mpl_toolkits.mplot3d import Axes3D
import subprocess
from sklearn.decomposition import PCA
import psutil
import copy
from scipy.spatial import ConvexHull
from sklearn.base import clone
import threading
from sklearn.manifold import TSNE
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import optuna
import gc
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
import pickle
import time
import math
import resource
from sklearn.linear_model import Ridge
import numpy as np
import statistics
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit
from matplotlib import ticker as plticker
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import shap
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel, Matern, RationalQuadratic
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRegressor
from tqdm import tqdm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import MaxNLocator 
import matplotlib.ticker as plticker 
import matplotlib.patches as patches
from sklearn.neural_network import MLPRegressor
import sklearn.gaussian_process as gp
from sklearn.utils import shuffle
from sklearn.impute import KNNImputer
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import FixedLocator
import matplotlib
import random
from sklearn.model_selection import KFold
import geopandas as gpd
from shapely.geometry import MultiPolygon
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

matplotlib.rcParams['axes.linewidth'] = 1
matplotlib.rcParams['text.color'] = '#343a40'
matplotlib.rcParams['axes.labelcolor'] = '#343a40'
matplotlib.rcParams['xtick.color'] = '#343a40'
matplotlib.rcParams['ytick.color'] = '#343a40'
matplotlib.rcParams['axes.edgecolor'] = '#343a40' 
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams["font.family"] = "Liberation Sans"

def MPE(actual, prediction):
    error = []
    for i in range(len(prediction)):
        if actual[i] != 0:
            error.append((abs((actual[i]-prediction[i]))/abs(actual[i])*100))
        else:
            error.append((abs((actual[i]-prediction[i]))/abs(prediction[i])*100))
    x = 0
    for i in error:
        x += i
    x = x/len(prediction)
    mpe = round(x,2)
    return(mpe)

_profiler_state = {}
def get_gpu_stats():
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,nounits,noheader'],
            stderr=subprocess.DEVNULL
        )
        usage = result.decode('utf-8').strip().split('\n')
        gpu_stats = []
        for i, line in enumerate(usage):
            parts = line.strip().split(', ')
            if len(parts) != 3:
                continue
            util, mem_used, mem_total = map(int, parts)
            stat = f"GPU {i}: Utilization: {util}%, Memory Used: {mem_used} MB / {mem_total} MB"
            print(stat)
            gpu_stats.append(stat)
        return gpu_stats
    except subprocess.CalledProcessError as e:
        print(f"Failed to fetch GPU stats: {e}")
        return []

def _monitor_cpu(interval=1.0):
    pid = os.getpid()
    process = psutil.Process(pid)
    cpu_usage = []

    def record_usage():
        while not _profiler_state.get('stop_monitoring', True) is True:
            cpu_percent = process.cpu_percent(interval=None) / psutil.cpu_count()
            cpu_usage.append(cpu_percent)
            time.sleep(interval)

    _profiler_state['cpu_usage'] = []
    _profiler_state['stop_monitoring'] = False
    t = threading.Thread(target=record_usage)
    t.start()
    _profiler_state['cpu_thread'] = t
    _profiler_state['cpu_usage'] = cpu_usage

def start_profiling():
    _profiler_state['gpu_before'] = get_gpu_stats()
    _profiler_state['start_wall'] = time.time()
    _profiler_state['start_usage'] = resource.getrusage(resource.RUSAGE_SELF)
    _monitor_cpu(interval=0.5)

def end_profiling(save_path=None, note=None):  # default values none if nothing given, just called ()
    _profiler_state['stop_monitoring'] = True
    _profiler_state['cpu_thread'].join()

    gpu_after = get_gpu_stats()
    end_wall = time.time()
    end_usage = resource.getrusage(resource.RUSAGE_SELF)

    user_time = end_usage.ru_utime - _profiler_state['start_usage'].ru_utime
    sys_time = end_usage.ru_stime - _profiler_state['start_usage'].ru_stime
    total_user = user_time + sys_time
    wall_time = end_wall - _profiler_state['start_wall']
    cpu_usages = _profiler_state['cpu_usage']

    avg_cpu = sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0
    max_cpu = max(cpu_usages) if cpu_usages else 0
    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)

    output = []

    if note:
        output.append("---- NOTE ----")
        output.append(note)
        output.append("")

    output.append("---- GPU Stats (Before Training) ----")
    output.extend(_profiler_state['gpu_before'])
    output.append("")
    output.append("---- GPU Stats (After Training) ----")
    output.extend(gpu_after)
    output.append("")
    output.append("---- CPU Timing Stats ----")
    output.append(f"User CPU time    : {user_time:.4f} s")
    output.append(f"System CPU time  : {sys_time:.4f} s")
    output.append(f"Total CPU time   : {total_user:.4f} s")
    output.append(f"Wall clock time  : {wall_time:.4f} s")
    output.append("")
    output.append("---- CPU Utilization Stats ----")
    output.append(f"Avg CPU usage    : {avg_cpu:.2f}%")
    output.append(f"Max CPU usage    : {max_cpu:.2f}%")
    output.append(f"Logical cores    : {logical_cores}")
    output.append(f"Physical cores   : {physical_cores}")

    # Print all profiling output
    print("\n".join(output))

    # Save to file only if path is provided
    if save_path:
        with open(save_path, "w") as f:
            f.write("\n".join(output))

        
def average(min, max):
    mid = (min+max)/2
    if max-min <=1:
        mini = np.round(min,2)
        maxi = np.round(max,2)
        mid = np.round(mid,2)
    if max-min >1:
        mini = np.round(min,1)
        maxi = np.round(max,1)
        mid = np.round(mid,1)
    return(mini, mid, maxi)


#Density Plot
def get_limits(info):
    return info['limits']

def get_scale(info):
    return info['scale']

def get_label(info):
    return info['label']

def get_inter(info):
    return info['inter']

def get_den_scale(info):
    return info['den_scale_bool'], info['den_scale']  # den_scale always = 1

def make_den_plot(info, ytest, ytest_pred,R2_Train, R2_test):
    tt_s = 0
    den_scale_bool, den_scale = get_den_scale(info)
    rng = get_limits(info)
    scale = get_scale(info)   # scale always = 1
    label = get_label(info)
    inter = get_inter(info)

    fig,[ax] = panel(1,1,dpi=500)
    plt.sca(ax)

    def put_legend():
        ax.plot([], [],'k',ls='none', mew=0,label='Train (MAPE ={:.2f})'.format(R2_Train))
        #ax.plot([], [],'k',ls='none', mew=0,label='Val (R$^2$={:.3f})'.format(R2_val))
        ax.plot([], [],'k',ls='none', mew=0,label='Test (MAPE ={:.2f})'.format(R2_test))
    put_legend()

    if tt_s:

        ax.plot(ytest/scale, ytest_pred/scale, 'or',mec='k')
        ax.plot(rng,rng,'-',c='k')
        xlabel(label.replace('*','')+'Plant data')
        ylabel(label.replace('*','')+'Predicted')
        loc = plticker.MultipleLocator(base=inter)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        legend_on(loc=4) #Changed loc=4 to loc=None


    else:
        s= 70
        im_data, xedges, yedges= np.histogram2d(ytest.ravel()/scale, ytest_pred.ravel()/scale, bins=(s,s), range=np.array([rng,rng]), density=0)

        if den_scale_bool:  # what is density scale bool in info?
            mask = im_data > (im_data.mean()+den_scale*im_data.std())
            im_data[mask] = im_data.mean()+den_scale*im_data.std()

            im_data = im_data.astype(int)
        else:
            im_data = im_data.astype(int)

        ocean = cm.get_cmap('gist_heat', 256) ## Throwing error
        c_data = ocean(np.linspace(0, 1, 256))
        mycm = ListedColormap(c_data[::-1,:])
        cb = ax.imshow(im_data, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap = mycm)

        # plt.title('CO')
        xlim(rng)
        ylim(rng)
        ax.plot(rng,rng,':',c='gray')
        xlabel(label.replace('*',''))
#         ylabel('Predicted '+label.replace('*',''))
        ylabel(label.replace('*',''))
        loc = plticker.MultipleLocator(base=inter)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        legend_on(loc=4) #Changed loc=4 to loc=None
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        # cax.set_yscale('log')
        plt.colorbar(cb, cax=cax, label="Number of data points per pixel")
        cbax = plt.gca()
        
        cbax.yaxis.set_major_locator(MaxNLocator(integer=True))



        e = (ytest.ravel()-ytest_pred.ravel())/scale

        std = e.std()

        ins_ax = inset([0.05,0.67,0.3,0.3], ax=ax)

        plt.sca(ins_ax)

        range = [-4*std, 4*std]
        _ = ins_ax.hist(e,bins=50,fc='k', density=False, range=range)  # changed it to True ##########CHECK

        rectangle(-1.65*std,1.65*std,0,_[0].max()*1.2,ax=ins_ax,color='r')

    
        ins_ax.yaxis.set_label_position("right")
        ins_ax.yaxis.tick_right()
    
        # ins_ax.set_xticks([-0.01,0,0.01])           # These two lines are for setting x tick lables of inset graph
        # ins_ax.set_xticklabels(['-0.01','0','0.01'], fontdict={'fontsize':10,'fontweight':'bold' })   #

        xlim(range)

        ylabel('Frequency', fontdict={'size':15} )
        
        if len(label.split('('))>1:
            unit = "("+"(".join(label.split('(')[1:])
        else:
            unit = ""
        xlabel(r'$\varepsilon $ ') # + unit

    return fig

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
def get_score(base='Training', actual=0,predicted=0):
    r2 = r2_score(actual,predicted)
    mae = mean_absolute_error(actual,predicted)
    mse = mean_squared_error(actual,predicted)
    mape = mean_absolute_percentage_error(actual,predicted)
    print(base,r2.round(2),mae.round(2), (mape*100).round(2))
    return (r2.round(2), ((mae)).round(2),(mape*100).round(2))

def train_test_score(model, X_train, X_test, y_train, y_test):
    _, _ = get_score_parity('Training',actual=y_train,predicted=model.predict(X_train))
    _, _ = get_score_parity('Test',actual=y_test,predicted=model.predict(X_test))

def get_score_custom(actual=0,predicted=0):
    r2 = r2_score(actual,predicted)
    mae = mean_absolute_error(actual,predicted)
    mape = mean_absolute_percentage_error(actual,predicted)
    return [r2.round(3),mae.round(3),mape.round(3)]

def filter_df(df):
    print(f'original length = {len(df)}')
    quantile_limit_values = []
    variable = df

    for column in variable:
        q_low = variable[column].quantile(0.0001)   # move dot 2 steps ahead, i.e., quantile(0.0001) = 0.01 %
        q_hi  = variable[column].quantile(0.9999)
        quantile_limit_values.append((q_low,q_hi))


    count = 0
    for column in variable:
        q_low, q_hi = quantile_limit_values[count]
        variable = variable[(variable[column] < q_hi) & (variable[column] > q_low)]
        count +=1
        
    filtered = variable
    filtered.reset_index(inplace=False)
    # filtered.drop(columns=['index'],inplace=True)
    print(f'length filtered = {len(filtered)}')
    return(filtered)

def filter_df_quantile(df,min, max):
    print(f'original length = {len(df)}')
    quantile_limit_values = []
    variable = df

    for column in variable:
        q_low = variable[column].quantile(min)   # move dot 2 steps ahead, i.e., quantile(0.0001) = 0.01 %
        q_hi  = variable[column].quantile(max)
        quantile_limit_values.append((q_low,q_hi))


    count = 0
    for column in variable:
        q_low, q_hi = quantile_limit_values[count]
        variable = variable[(variable[column] < q_hi) & (variable[column] > q_low)]
        count +=1
        
    filtered = variable
    filtered.reset_index(inplace=False)
    # filtered.drop(columns=['index'],inplace=True)
    print(f'length filtered = {len(filtered)}')
    return(filtered)

    
def filter_nestor(df):
    ds4 = df.copy()
    ds4.drop(ds4[ds4['Raw meal mass flow rate (to preheater)'] < 421.52].index, inplace = True)
    ds4.drop(ds4[ds4['Raw meal mass flow rate (to preheater)'] > 517.28].index, inplace = True)
    ds4.drop(ds4[ds4['Fuel consumption (calciner)'] < 16.78].index, inplace = True)
    ds4.drop(ds4[ds4['Fuel consumption (calciner)'] > 30.28].index, inplace = True)
    ds4.drop(ds4[ds4['Fuel consumption (kiln)'] < 8.94].index, inplace = True)
    ds4.drop(ds4[ds4['Fuel consumption (kiln)'] > 15.36].index, inplace = True)
    ds4.drop(ds4[ds4['O2 content in the raw gas (preheater outlet)'] < 1.5].index, inplace = True)
    ds4.drop(ds4[ds4['O2 content in the raw gas (preheater outlet)'] > 3.5].index, inplace = True)
    ds4.drop(ds4[ds4['O2 content in the kiln gas (to preheater) at kiln inlet chamber'] < 5.23].index, inplace = True)
    ds4.drop(ds4[ds4['O2 content in the kiln gas (to preheater) at kiln inlet chamber'] > 12.57].index, inplace = True)
    ds4.drop(ds4[ds4['Raw meal temperature (to preheater)'] < 62.2].index, inplace = True)
    ds4.drop(ds4[ds4['Raw meal temperature (to preheater)'] > 116.26].index, inplace = True)
    ds4.drop(ds4[ds4['Solids outlet temperature (kiln inlet)'] < 957.58].index, inplace = True)
    ds4.drop(ds4[ds4['Solids outlet temperature (kiln inlet)'] > 1222.2].index, inplace = True)
    ds4.drop(ds4[ds4['Hot meal temperature of lowest cyclone'] < 852.34 ].index, inplace = True)
    ds4.drop(ds4[ds4['Hot meal temperature of lowest cyclone'] > 1039.73].index, inplace = True)
    ds4.drop(ds4[ds4['Exit temperature from each preheater cyclone (Stage 4)'] < 710].index, inplace = True)
    ds4.drop(ds4[ds4['Exit temperature from each preheater cyclone (Stage 4)'] > 900].index, inplace = True)
    ds4.drop(ds4[ds4['Total cooling air'] < 400000].index, inplace = True)
    ds4.drop(ds4[ds4['Total cooling air'] < 375000].index, inplace = True)
    ds4.drop(ds4[ds4['I Air (including transport air)'] > 1750].index, inplace = True)
    ds4.drop(ds4[ds4['I Air (including transport air)'] < 1000].index, inplace = True)
    ds4.drop(ds4[ds4['Exit temperature from each preheater cyclone (Stage 1A)'] < 362].index, inplace = True)
    ds4.drop(ds4[ds4['Exit temperature from each preheater cyclone (Stage 1A)'] > 390].index, inplace = True)
    ds4.drop(ds4[ds4['Exit temperature from each preheater cyclone (Stage 2)'] < 512].index, inplace = True)
    ds4.drop(ds4[ds4['Exit temperature from each preheater cyclone (Stage 2)'] > 550].index, inplace = True)
    ds4.drop(ds4[ds4['Preheater gas outlet temperature'] > 360].index, inplace = True)
    ds4.drop(ds4[ds4['Preheater gas outlet temperature'] < 336].index, inplace = True)
    ds4 = ds4.dropna(axis=0, how='any')
    ds4.drop(columns = ['Exit temperature from each preheater cyclone (Stage 5)',
    'Flue gas outlet temperature (calciner outlet)',
    # 'Flue gas O2 content (calciner outlet)',
    # 'Clinker outlet temperature'
    ], inplace = True)
    # unreliable variables:

    return(ds4)

def moving_average_n_window(arr, n):
    if n % 2 == 0:
        raise ValueError("Window size n must be odd.")
    result = arr.copy()
    half = n // 2
    for i in range(half, len(arr) - half):
        window = arr[i - half:i + half + 1]
        result[i] = sum(window) / n
    return result

def find_max(numbers):
    largest_number = numbers[0]
    for number in numbers:
        if number > largest_number:
            largest_number = number
    return(largest_number)

def array_to_list(x):
    mad = []
    for i in range(len(x)):
        mad.append(x[i][0])
    return(mad)

def parity_error(model, X,y, unit, title, model_name):
    # mean = df['alite_pred'].to_list()
    d = 0.4

    matplotlib.rcParams['axes.linewidth'] = 1
    style = 'normal'
    s = 20
    s1 = 13
    w =1

    train_actual = y_train[list(y_train.columns)[0]].to_list()
    test_actual = y_test[list(y_test.columns)[0]].to_list()

# Log tranformation:
#     train_actual = array_to_list(np.expm1(y_train).values)
#     test_actual = array_to_list(np.expm1(y_test).values)
#     test_pred = list(np.expm1(model.predict(X_test)))
#     train_pred = list(np.expm1(model.predict(X_train)))

    if model_name in ['Linear regression','Ridge']:
           test_pred = array_to_list(model.predict(X_test))
           train_pred = array_to_list(model.predict(X_train))
           mape_train, r2_train = get_score('Training',actual=y_train,predicted=model.predict(X_train))
           mape_test, r2_test = get_score('Test',actual=y_test,predicted=model.predict(X_test))
    elif model_name in ['SVR', 'GPR', 'NN']:   
           test_pred_s = model.predict(X_test_s).reshape(-1,1)
           train_pred_s = model.predict(X_train_s).reshape(-1,1)
    
           mape_train,r2_train = get_score('Training',actual=y_train,predicted= scaler.inverse_transform(train_pred_s) )
           mape_test, r2_test = get_score('Test',actual=y_test,predicted=scaler.inverse_transform(test_pred_s) )
           train_pred = array_to_list(scaler.inverse_transform(train_pred_s))
           test_pred = array_to_list(scaler.inverse_transform(test_pred_s)) 
    else:
           test_pred = list(model.predict(X_test))
           train_pred = list(model.predict(X_train))
           mape_train,r2_train = get_score('Training',actual=y_train,predicted=model.predict(X_train))
           mape_test, r2_test = get_score('Test',actual=y_test,predicted=model.predict(X_test))
       
           # log transform:
       #     mape_train,r2_train = get_score('Training',actual = np.expm1(y_train) ,predicted = np.expm1(model.predict(X_train)))
       #     mape_test, r2_test = get_score('Test',actual = np.expm1(y_test),predicted=np.expm1(model.predict(X_test)))
       

    minimum = int(y.describe().T[['min', 'max']].values[0][0].round(0))
    maximum = int(y.describe().T[['min', 'max']].values[0][1].round(0))
    min = minimum
    max = maximum
   
    fig, ax = plt.subplots(figsize=(4.5,4.5))
     
    _= plt.scatter(train_actual,train_pred,color = ['white'] , s=25, linewidth = 1,edgecolors='darkgreen', marker = '^')
    _= plt.scatter(train_actual,train_pred,color = ['darkgreen'] , s=25,alpha = 0.1, marker ='^' )

    _= plt.scatter(test_actual,test_pred,color = ['white'] , s = 20, linewidth = 1,edgecolors='black', marker= 'o')
    _= plt.scatter(test_actual,test_pred,color = ['black'] , s=20,alpha = 0.1, marker = 'o')

    _= plt.plot([minimum, minimum],[minimum, minimum], color = 'white', lw = 0)
    _= plt.plot([minimum, maximum],[minimum, maximum], color = 'black', lw = 1, ls = '--')

    # X = [min,round(((min+max)/2),0),max]
    X = [min, ((min+max)/2),max]
    _= plt.xticks(X,X,rotation = 0, fontweight = style, fontsize = s)
    _= plt.yticks(X, X,rotation = 0, fontweight = style, fontsize = s)

    _= ax.tick_params('both', length=10, width=w, which='major')
    _= ax.tick_params('both', length=5, width=w, which='minor')

    _= plt.xlabel(f'Plant data ({unit})', fontsize= s, fontweight = 'normal')
    _= plt.ylabel(f'Predicted ({unit})', fontsize= s, fontweight = 'normal')
    _= plt.title(f'{title}', fontsize= s, fontweight = 'normal')

    box = ('$MAE_{Train}$ ='+f' {mape_train}'
           +'\n'+ '$MAE_{Test}$ ='+f' {mape_test}')
    _= ax.text( 0.538,0.95,box, transform=ax.transAxes, linespacing=1 ,fontsize=14,verticalalignment='top', fontweight='normal',
                bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0', alpha =0.65))

    # inset test
    left, bottom, width, height = [0.15, 0.64, 0.22, 0.22]
    ax2 = fig.add_axes([left, bottom, width, height])
    p = np.array(test_pred)
    a = np.array(test_actual)
    std=np.std(p - a)
    mean=np.mean(p - a)
    xx=mean-2*std
    yy = 0
    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='white',range = [-4*std, 4*std] )
    rec = patches.Rectangle((xx,yy),4*std, find_max(y1), facecolor ='red', alpha = 0.16)
    ax2.add_patch(rec)
    _= plt.axvline(x = xx, color = 'red', lw =0.6, ls = '--')
    _= plt.axvline(x = xx+4*std, color = 'red', lw =0.6, ls = '--')
    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='black',range = [-4*std, 4*std], alpha = 0.78)

    ax2.set_ylim(0,find_max(y1))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    xlabel_ax2 = ax2.get_xticklabels()
    ax2.yaxis.tick_right()
    # ax2.yaxis.set_tick_params(labelsize =10,weight = 'normal')
    _= plt.xticks(fontweight = style, fontsize = s1)
    _= plt.yticks(fontweight = style, fontsize = s1)
    _= ax2.xaxis.set_tick_params(labelsize =s1)  #11.8
    # _= ax2.yaxis.set_tick_params(labelsize =12)
    _= ax2.tick_params('both', length=10, width=w, which='major')
    _= ax2.tick_params('both', length=5, width=w, which='minor')
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('\u03BD',fontsize=s1, fontweight = style, labelpad = 0)
    ax2.set_xlabel(r'$\epsilon$',fontsize=s1, fontweight = style, labelpad = 0)



    # incet train
    left, bottom, width, height = [0.65, 0.19, 0.22, 0.22]
    ax2 = fig.add_axes([left, bottom, width, height])
    p = np.array(train_pred)
    a = np.array(train_actual)
    std=np.std(p - a)
    mean=np.mean(p - a)
    xx=mean-2*std
    yy = 0
    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='white',range = [-4*std, 4*std] )
    rec = patches.Rectangle((xx,yy),4*std, find_max(y1), facecolor ='red', alpha = 0.16)
    ax2.add_patch(rec)
    _= plt.axvline(x = xx, color = 'red', lw =0.6, ls = '--')
    _= plt.axvline(x = xx+4*std, color = 'red', lw =0.6, ls = '--')
    _=y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='darkgreen',range = [-4*std, 4*std], alpha = 1)

    _=ax2.set_ylim(0,find_max(y1))
    _=ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    _=xlabel_ax2 = ax2.get_xticklabels()
    _=ax2.yaxis.tick_left()
    # ax2.yaxis.set_tick_params(labelsize =10,weight = 'normal')
    _= plt.xticks(fontweight = style, fontsize = s1)
    _= plt.yticks(fontweight = style, fontsize = s1)
    _= ax2.xaxis.set_tick_params(labelsize =s1)  #11.8
    _= ax2.yaxis.set_tick_params(labelsize =s1)
    _= ax2.tick_params('both', length=10, width=w, which='major')
    _= ax2.tick_params('both', length=5, width=w, which='minor')
    _=ax2.yaxis.set_label_position("left")
    _=ax2.set_ylabel('\u03BD',fontsize=s1, fontweight = style, labelpad = 0)
    # _=ax2.set_xlabel(r'$\epsilon$',fontsize=s1, fontweight = style, labelpad = 0)
    # plt.savefig('/media/m3rg2000/mounted/Junaid/Heidelberg/HB_clk_phases/saved_figures/alite_1.tiff',format='tiff', dpi=1200)
    _=plt.show()
    
    
def error_parity_nox(ac, pred):    
    get_score('train+test',actual=ac,predicted= pred) 
    min = 100
    max = 1000
    minimum = min
    maximum = max
    unit = 'PPM'
    fig, ax = plt.subplots(figsize=(4.5,4.5))

    matplotlib.rcParams['axes.linewidth'] = 1
    style = 'normal'
    s = 20
    s1 = 13
    w =1

    _= plt.scatter(ac, pred,color = ['white'] , s=25, linewidth = 1,edgecolors='darkgreen', marker = '^')
    _= plt.scatter(ac, pred,color = ['darkgreen'] , s=25,alpha = 0.1, marker ='^' )

    _= plt.plot([minimum, minimum],[minimum, minimum], color = 'white', lw = 0)
    _= plt.plot([minimum, maximum],[minimum, maximum], color = 'black', lw = 1, ls = '--')

    # X = [min,round(((min+max)/2),0),max]
    X = [min, ((min+max)/2),max]
    _= plt.xticks(X,X,rotation = 0, fontweight = style, fontsize = s)
    _= plt.yticks(X, X,rotation = 0, fontweight = style, fontsize = s)

    _= ax.tick_params('both', length=10, width=w, which='major')
    _= ax.tick_params('both', length=5, width=w, which='minor')

    _= plt.xlabel(f'Plant data ({unit})', fontsize= s, fontweight = 'normal')
    _= plt.ylabel(f'Predicted ({unit})', fontsize= s, fontweight = 'normal')
    _=plt.show()
    
    
def error_parity_co(ac, pred): 
    get_score('train+test',actual=ac,predicted= pred)    
    min = 27
    max = 1370
    minimum = min
    maximum = max
    unit = 'co'
    fig, ax = plt.subplots(figsize=(4.5,4.5))

    matplotlib.rcParams['axes.linewidth'] = 1
    style = 'normal'
    s = 20
    s1 = 13
    w =1

    _= plt.scatter(ac, pred,color = ['white'] , s=25, linewidth = 1,edgecolors='darkgreen', marker = '^')
    _= plt.scatter(ac, pred,color = ['darkgreen'] , s=25,alpha = 0.1, marker ='^' )

    _= plt.plot([minimum, minimum],[minimum, minimum], color = 'white', lw = 0)
    _= plt.plot([minimum, maximum],[minimum, maximum], color = 'black', lw = 1, ls = '--')

    # X = [min,round(((min+max)/2),0),max]
    X = [min, ((min+max)/2),max]
    _= plt.xticks(X,X,rotation = 0, fontweight = style, fontsize = s)
    _= plt.yticks(X, X,rotation = 0, fontweight = style, fontsize = s)

    _= ax.tick_params('both', length=10, width=w, which='major')
    _= ax.tick_params('both', length=5, width=w, which='minor')

    _= plt.xlabel(f'Plant data ({unit})', fontsize= s, fontweight = 'normal')
    _= plt.ylabel(f'Predicted ({unit})', fontsize= s, fontweight = 'normal')
    _=plt.show()

def error_parity_co2(ac, pred): 
    get_score('train+test',actual=ac,predicted= pred)    
    min = 170
    max = 336
    minimum = min
    maximum = max
    unit = 'co2'
    fig, ax = plt.subplots(figsize=(4.5,4.5))

    matplotlib.rcParams['axes.linewidth'] = 1
    style = 'normal'
    s = 20
    s1 = 13
    w =1

    _= plt.scatter(ac, pred,color = ['white'] , s=25, linewidth = 1,edgecolors='darkgreen', marker = '^')
    _= plt.scatter(ac, pred,color = ['darkgreen'] , s=25,alpha = 0.1, marker ='^' )

    _= plt.plot([minimum, minimum],[minimum, minimum], color = 'white', lw = 0)
    _= plt.plot([minimum, maximum],[minimum, maximum], color = 'black', lw = 1, ls = '--')

    # X = [min,round(((min+max)/2),0),max]
    X = [min, ((min+max)/2),max]
    _= plt.xticks(X,X,rotation = 0, fontweight = style, fontsize = s)
    _= plt.yticks(X, X,rotation = 0, fontweight = style, fontsize = s)

    _= ax.tick_params('both', length=10, width=w, which='major')
    _= ax.tick_params('both', length=5, width=w, which='minor')

    _= plt.xlabel(f'Plant data ({unit})', fontsize= s, fontweight = 'normal')
    _= plt.ylabel(f'Predicted ({unit})', fontsize= s, fontweight = 'normal')
    _=plt.show()

def get_score_parity(base='Training', actual=0,predicted=0):
    r2 = r2_score(actual,predicted)
    mae = mean_absolute_error(actual,predicted)
    mse = mean_squared_error(actual,predicted)
    mape = mean_absolute_percentage_error(actual,predicted)
    print(base,r2.round(2),mae.round(2), (mape*100).round(2))
    return ((mae).round(2), r2.round(2))


def parity(model,X_train, y_train,X_test, y_test, y, unit, title, model_name):

    # mean = df['alite_pred'].to_list()
    d = 0.4

    matplotlib.rcParams['axes.linewidth'] = 1
    style = 'normal'
    s = 20
    s1 = 13
    w =1

    train_actual = y_train[list(y_train.columns)[0]].to_list()
    test_actual = y_test[list(y_test.columns)[0]].to_list()

# Log tranformation:
#     train_actual = array_to_list(np.expm1(y_train).values)
#     test_actual = array_to_list(np.expm1(y_test).values)
#     test_pred = list(np.expm1(model.predict(X_test)))
#     train_pred = list(np.expm1(model.predict(X_train)))

    if model_name in ['Linear regression','Ridge']:
           test_pred = array_to_list(model.predict(X_test))
           train_pred = array_to_list(model.predict(X_train))
           mape_train, r2_train = get_score_parity('Training',actual=y_train,predicted=model.predict(X_train))
           mape_test, r2_test = get_score_parity('Test',actual=y_test,predicted=model.predict(X_test))
    elif model_name in ['SVR', 'GPR', 'NN']:   
           test_pred_s = model.predict(X_test_s).reshape(-1,1)
           train_pred_s = model.predict(X_train_s).reshape(-1,1)
    
           mape_train,r2_train = get_score_parity('Training',actual=y_train,predicted= scaler.inverse_transform(train_pred_s) )
           mape_test, r2_test = get_score_parity('Test',actual=y_test,predicted=scaler.inverse_transform(test_pred_s) )
           train_pred = array_to_list(scaler.inverse_transform(train_pred_s))
           test_pred = array_to_list(scaler.inverse_transform(test_pred_s)) 
    else:
           test_pred = list(model.predict(X_test))
           train_pred = list(model.predict(X_train))
           mape_train,r2_train = get_score_parity('Training',actual=y_train,predicted=model.predict(X_train))
           mape_test, r2_test = get_score_parity('Test',actual=y_test,predicted=model.predict(X_test))
       
           # log transform:
       #     mape_train,r2_train = get_score('Training',actual = np.expm1(y_train) ,predicted = np.expm1(model.predict(X_train)))
       #     mape_test, r2_test = get_score('Test',actual = np.expm1(y_test),predicted=np.expm1(model.predict(X_test)))
       

    minimum = int(y.describe().T[['min', 'max']].values[0][0].round(0))
    maximum = int(y.describe().T[['min', 'max']].values[0][1].round(0))
    min = minimum
    max = maximum
   
    fig, ax = plt.subplots(figsize=(4.5,4.5))
     
    _= plt.scatter(train_actual,train_pred,color = ['white'] , s=25, linewidth = 1,edgecolors='darkgreen', marker = '^')
    _= plt.scatter(train_actual,train_pred,color = ['darkgreen'] , s=25,alpha = 0.1, marker ='^' )

    _= plt.scatter(test_actual,test_pred,color = ['white'] , s = 20, linewidth = 1,edgecolors='black', marker= 'o')
    _= plt.scatter(test_actual,test_pred,color = ['black'] , s=20,alpha = 0.1, marker = 'o')

    _= plt.plot([minimum, minimum],[minimum, minimum], color = 'white', lw = 0)
    _= plt.plot([minimum, maximum],[minimum, maximum], color = 'black', lw = 1, ls = '--')

    # X = [min,round(((min+max)/2),0),max]
    X = [min, ((min+max)/2),max]
    _= plt.xticks(X,X,rotation = 0, fontweight = style, fontsize = s)
    _= plt.yticks(X, X,rotation = 0, fontweight = style, fontsize = s)

    _= ax.tick_params('both', length=10, width=w, which='major')
    _= ax.tick_params('both', length=5, width=w, which='minor')

    _= plt.xlabel(f'Plant data ({unit})', fontsize= s, fontweight = 'normal')
    _= plt.ylabel(f'Predicted ({unit})', fontsize= s, fontweight = 'normal')
    _= plt.title(f'{title}', fontsize= s, fontweight = 'normal')

    box = ('$MAE_{Train}$ ='+f' {mape_train}'
           +'\n'+ '$MAE_{Test}$ ='+f' {mape_test}')
    # _= ax.text( 0.538,0.95,box, transform=ax.transAxes, linespacing=1 ,fontsize=14,verticalalignment='top', fontweight='normal',
    #             bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0', alpha =0.65))

    # inset test
    left, bottom, width, height = [0.15, 0.64, 0.22, 0.22]
    ax2 = fig.add_axes([left, bottom, width, height])
    p = np.array(test_pred)
    a = np.array(test_actual)
    std=np.std(p - a)
    mean=np.mean(p - a)
    xx=mean-2*std
    yy = 0
    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='white',range = [-4*std, 4*std] )
    rec = patches.Rectangle((xx,yy),4*std, find_max(y1), facecolor ='red', alpha = 0.16)
    ax2.add_patch(rec)
    _= plt.axvline(x = xx, color = 'red', lw =0.6, ls = '--')
    _= plt.axvline(x = xx+4*std, color = 'red', lw =0.6, ls = '--')
    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='black',range = [-4*std, 4*std], alpha = 0.78)

    ax2.set_ylim(0,find_max(y1))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    xlabel_ax2 = ax2.get_xticklabels()
    ax2.yaxis.tick_right()
    # ax2.yaxis.set_tick_params(labelsize =10,weight = 'normal')
    _= plt.xticks(fontweight = style, fontsize = s1)
    _= plt.yticks(fontweight = style, fontsize = s1)
    _= ax2.xaxis.set_tick_params(labelsize =s1)  #11.8
    # _= ax2.yaxis.set_tick_params(labelsize =12)
    _= ax2.tick_params('both', length=10, width=w, which='major')
    _= ax2.tick_params('both', length=5, width=w, which='minor')
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('\u03BD',fontsize=s1, fontweight = style, labelpad = 0)
    ax2.set_xlabel(r'$\epsilon$',fontsize=s1, fontweight = style, labelpad = 0)



    # # incet train
    # left, bottom, width, height = [0.65, 0.19, 0.22, 0.22]
    # ax2 = fig.add_axes([left, bottom, width, height])
    # p = np.array(train_pred)
    # a = np.array(train_actual)
    # std=np.std(p - a)
    # mean=np.mean(p - a)
    # xx=mean-2*std
    # yy = 0
    # y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='white',range = [-4*std, 4*std] )
    # rec = patches.Rectangle((xx,yy),4*std, find_max(y1), facecolor ='red', alpha = 0.16)
    # ax2.add_patch(rec)
    # _= plt.axvline(x = xx, color = 'red', lw =0.6, ls = '--')
    # _= plt.axvline(x = xx+4*std, color = 'red', lw =0.6, ls = '--')
    # _=y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='darkgreen',range = [-4*std, 4*std], alpha = 1)

    # _=ax2.set_ylim(0,find_max(y1))
    # _=ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    # _=xlabel_ax2 = ax2.get_xticklabels()
    # _=ax2.yaxis.tick_left()
    # # ax2.yaxis.set_tick_params(labelsize =10,weight = 'normal')
    # _= plt.xticks(fontweight = style, fontsize = s1)
    # _= plt.yticks(fontweight = style, fontsize = s1)
    # _= ax2.xaxis.set_tick_params(labelsize =s1)  #11.8
    # _= ax2.yaxis.set_tick_params(labelsize =s1)
    # _= ax2.tick_params('both', length=10, width=w, which='major')
    # _= ax2.tick_params('both', length=5, width=w, which='minor')
    # _=ax2.yaxis.set_label_position("left")
    # _=ax2.set_ylabel('\u03BD',fontsize=s1, fontweight = style, labelpad = 0)
    # # _=ax2.set_xlabel(r'$\epsilon$',fontsize=s1, fontweight = style, labelpad = 0)
    # # plt.savefig('/media/m3rg2000/mounted/Junaid/Heidelberg/HB_clk_phases/saved_figures/alite_1.tiff',format='tiff', dpi=1200)
    _=plt.show()
    
    
def shap_plot(name, top_fe, new_captions, empty, X_test,shap_values):    
    sz = 15
    _= shap.summary_plot(shap_values,show = False, 
                    plot_type='bar',
                    max_display = top_fe,
                    color_bar = False,
                    color= "firebrick",
                    feature_names = empty,
                    features = X_test,      #Changed from X_train to X_test_scx as we are explaining all the predictions in test set
                    plot_size=[17,7]  # Change this size as per requirement
                    )
    _= plt.ylabel('Features', fontsize = sz, fontweight = 'normal')
    fig=plt.gca()
    _= fig.spines['top'].set_visible(True) 
    _= fig.spines['right'].set_visible(True)
    _= fig.spines['left'].set_visible(True)
    # plt.xlim(-3, 3)
    visible_ticks = {
    "top": True,
    }
    _= plt.tick_params(axis="x", which="both", **visible_ticks)
    _= plt.tick_params(axis="y", which="both", **visible_ticks)
    _= plt.tick_params(axis="y", which="major", direction = 'in', left = True)
    _= plt.tick_params(axis="y", which="major", direction = 'in', right = True)
    _= plt.xlabel('mean |SHAP values| for '+name, fontsize= sz, fontweight= 'normal')
    _= plt.xticks(fontsize = sz, fontweight= 'normal')
    _= plt.yticks(fontsize = sz, fontweight= 'normal')
    _= plt.tick_params(which = 'major', width= 1, length =6)
    _= plt.tick_params(which = 'minor', width= 1, length =3)
    # plt.savefig('/media/m3rg2000/New Volume/Junaid/Heidelberg/figures_ppt/shap_bar_nox.tiff',format='tiff', dpi=1200)
    plt.show()


    #Beeswarm code 
    sz = 15
    _= shap.summary_plot(shap_values, show = False, 
                    plot_type='violin',
                    max_display = top_fe,
                    color_bar = True,
                    features = X_test,     #Changed the features to X_test_scx instead of X_train
                    feature_names = new_captions,
                    plot_size=[17,7])

    _= plt.ylabel('Features', fontsize = sz, fontweight = 'normal')
    fig=plt.gca()
    _= fig.spines['top'].set_visible(True) 
    _= fig.spines['right'].set_visible(True)
    _= fig.spines['left'].set_visible(True)

    visible_ticks = {
    "top": True,
    }
    _= plt.tick_params(axis="x", which="both", **visible_ticks)
    _= plt.tick_params(axis="y", which="both", **visible_ticks)
    _= plt.tick_params(axis="y", which="major", direction = 'in', left = True)
    _= plt.tick_params(axis="y", which="major", direction = 'in', right = True)
    _= plt.xlabel('SHAP value for '+name, fontsize= sz, fontweight= 'normal')
    _= plt.xticks(fontsize = sz, fontweight= 'normal')
    _= plt.yticks(fontsize = sz, fontweight= 'normal')
    _= plt.tick_params(which = 'major', width= 1, length =6)
    _= plt.tick_params(which = 'minor', width= 1, length =3)
    plt.show()
        

def timeline_plot(sequence, label, index_plot, month_plot):
    s = 19                                      
    matplotlib.rcParams['axes.linewidth'] = 1
    w =1
    values_raw = np.array(sequence, dtype=np.float64) 
    figure, ax = plt.subplots(figsize=(12,5))  #dimgray
    _= ax.set_xticks([i for i in index_plot], labels = month_plot , rotation = 90, fontsize= 18, fontweight= 'normal') # ['2020\nFeb', '2020\nApril', '2020\nJune', '2020\nAug', '2020\nOct', '2020\nDec','2021\nFeb', '2021\nApril', '2021\nJune', '2021\nAug', '2021\nOct', '2021\nDec']
    # _= plt.yticks([0,5,10],[0,5,10], rotation = 0, fontsize=s, fontweight= 'normal')
    _= ax.tick_params(which = 'major', width= 1, length =10,)
    _= ax.tick_params(which = 'minor', width= 1, length =5)
    _= ax.tick_params(axis='x', direction ='in', which = 'both')
    _= ax.tick_params(axis='y', direction ='in', which = 'both')
    _= ax.tick_params(axis='y', which='major', right= True)
    _= ax.tick_params(axis='y', which='minor', right= True)
    _= ax.tick_params(axis='y', which='major', left= True)
    _= ax.tick_params(axis='y', which='minor', left= True)
    _= ax.tick_params(axis='x', which='major', top = True)
    _= ax.tick_params(axis='x', which='minor', top = False)
    _= ax.tick_params(axis='x', which='minor', bottom = False)
    _= ax.set_ylabel(label, fontsize=s, fontweight='normal')
    _= ax.set_xlabel('Time (Months)', fontsize=s, fontweight='normal')
    # _= ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    _= ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.scatter([i+1 for i in range(len(values_raw))],values_raw,color = 'black',s = 5)   #, alpha = 1, edgecolors='black', linewidth =1
    plt.show()


def parity_test_only(unit, title, model_name, y, ac, test_pred):

    d = 0.4

    matplotlib.rcParams['axes.linewidth'] = 1
    style = 'normal'
    s = 20
    s1 = 13
    w =1
    if model_name in ['Linear regression','Ridge']:
            test_pred = array_to_list(model.predict(X_test))
            train_pred = array_to_list(model.predict(X_train))
            mape_train, r2_train = get_score('Training',actual=y_train,predicted=model.predict(X_train))
            mape_test, r2_test = get_score('Test',actual=y_test,predicted=model.predict(X_test))
    elif model_name in ['SVR', 'GPR', 'NN']:   
            test_pred_s = model.predict(X_test_s).reshape(-1,1)
            train_pred_s = model.predict(X_train_s).reshape(-1,1)

            mape_train,r2_train = get_score('Training',actual=y_train,predicted= scaler.inverse_transform(train_pred_s) )
            mape_test, r2_test = get_score('Test',actual=y_test,predicted=scaler.inverse_transform(test_pred_s) )
            train_pred = array_to_list(scaler.inverse_transform(train_pred_s))
            test_pred = array_to_list(scaler.inverse_transform(test_pred_s)) 
    else:
            mape_test, r2_test = get_score('Test',actual=y_test,predicted=test_pred)
    minimum = int(y.describe().T[['min', 'max']].values[0][0].round(0))
    maximum = int(y.describe().T[['min', 'max']].values[0][1].round(0))
    min = minimum
    max = maximum

    fig, ax = plt.subplots(figsize=(4.5,4.5))

    _= plt.scatter(test_actual,test_pred,color = ['white'] , s = 20, linewidth = 1,edgecolors='black', marker= 'o')
    _= plt.scatter(test_actual,test_pred,color = ['black'] , s=20,alpha = 0.1, marker = 'o')

    _= plt.plot([minimum, minimum],[minimum, minimum], color = 'white', lw = 0)
    _= plt.plot([minimum, maximum],[minimum, maximum], color = 'black', lw = 1, ls = '--')

    # X = [min,round(((min+max)/2),0),max]
    X = [min, ((min+max)/2),max]
    _= plt.xticks(X,X,rotation = 0, fontweight = style, fontsize = s)
    _= plt.yticks(X, X,rotation = 0, fontweight = style, fontsize = s)

    _= ax.tick_params('both', length=10, width=w, which='major')
    _= ax.tick_params('both', length=5, width=w, which='minor')

    _= plt.xlabel(f'Plant data ({unit})', fontsize= s, fontweight = 'normal')
    _= plt.ylabel(f'Predicted ({unit})', fontsize= s, fontweight = 'normal')
    _= plt.title(f'{title}', fontsize= s, fontweight = 'normal')

    box = ('$MAE_{Test}$ ='+f' {mape_test}')
    _= ax.text( 0.538,0.95,box, transform=ax.transAxes, linespacing=1 ,fontsize=14,verticalalignment='top', fontweight='normal',
                bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0', alpha =0.65))

    # inset test
    left, bottom, width, height = [0.15, 0.64, 0.22, 0.22]
    ax2 = fig.add_axes([left, bottom, width, height])
    rect = plt.Rectangle(
      (left, bottom),
      width, height,
      transform=fig.transFigure,
      color='white',
      zorder=4)
    fig.patches.append(rect)
  
    p = np.array(test_pred)
    a = np.array(test_actual)
    std=np.std(p - a)
    mean=np.mean(p - a)
    xx=mean-2*std
    yy = 0
    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='white',range = [-4*std, 4*std] )
    rec = patches.Rectangle((xx,yy),4*std, find_max(y1), facecolor ='red', alpha = 0.16)
    ax2.add_patch(rec)
    _= plt.axvline(x = xx, color = 'red', lw =0.6, ls = '--')
    _= plt.axvline(x = xx+4*std, color = 'red', lw =0.6, ls = '--')
    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='black',range = [-4*std, 4*std], alpha = 0.78)

    ax2.set_ylim(0,find_max(y1))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    xlabel_ax2 = ax2.get_xticklabels()
    ax2.yaxis.tick_right()
    # ax2.yaxis.set_tick_params(labelsize =10,weight = 'normal')
    _= plt.xticks(fontweight = style, fontsize = s1)
    _= plt.yticks(fontweight = style, fontsize = s1)
    _= ax2.xaxis.set_tick_params(labelsize =s1)  #11.8
    # _= ax2.yaxis.set_tick_params(labelsize =12)
    _= ax2.tick_params('both', length=10, width=w, which='major')
    _= ax2.tick_params('both', length=5, width=w, which='minor')
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('\u03BD',fontsize=s1, fontweight = style, labelpad = 0)
    ax2.set_xlabel(r'$\epsilon$',fontsize=s1, fontweight = style, labelpad = 0)
    _=plt.show()


def best_model_tr_ts_score(best_model,X_train_s, X_test_s, scaler, y_train, y_test ):
    test_pred_s =best_model.predict(X_test_s).reshape(-1,1)
    train_pred_s =best_model.predict(X_train_s).reshape(-1,1)
    get_score('Training',actual=y_train,predicted=scaler.inverse_transform(train_pred_s))
    get_score('Test',actual=y_test,predicted=scaler.inverse_transform(test_pred_s))

def k_fold_scores(model, X_train, y_train):   
    kf = KFold(n_splits=4)
    train_fold_results = []
    val_fold_results = []

    for train_index, val_index in kf.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]
        best_model = clone(model)
        best_model.fit(X_train_fold, y_train_fold)
        y_val_pred = best_model.predict(X_val_fold)
        y_train_pred = best_model.predict(X_train_fold)

        train_fold_r2 = r2_score(y_train_fold, y_train_pred)
        train_fold_mae = mean_absolute_error(y_train_fold, y_train_pred)
        train_fold_mape = np.round((mean_absolute_percentage_error(y_train_fold,y_train_pred ))*100,2)

        val_fold_r2 = r2_score(y_val_fold, y_val_pred)
        val_fold_mae = mean_absolute_error(y_val_fold, y_val_pred)
        val_fold_mape = np.round((mean_absolute_percentage_error(y_val_fold,y_val_pred ))*100,2)

        train_fold_results.append((train_fold_r2, train_fold_mae, train_fold_mape))
        val_fold_results.append((val_fold_r2, val_fold_mae, val_fold_mape))
    avg_train_results = [round(x,2) for x in list(np.mean(np.array(train_fold_results), axis=0))]  # Average values of R2, MAE and MAPE on train folds
    avg_val_results =  [round(x,2) for x in list(np.mean(np.array(val_fold_results), axis=0))]    # Average values of R2, MAE and MAPE on val folds
    print(f"Avg Training:  R^2 = {avg_train_results[0]}, MAE = {avg_train_results[1]}, MAPE = {avg_train_results[2]}%")
    print(f"Avg Val:  R^2 = {avg_val_results[0]}, MAE = {avg_val_results[1]}, MAPE = {avg_val_results[2]}%")
    return(avg_train_results, avg_val_results)

def parity_train_test_no_model(name, label, train_actual, test_actual, train_pred,test_pred, y, fig_path=None):
    matplotlib.rcParams['axes.linewidth'] = 1
    marker_alpha = 0.055
    marker_lw = 1.1
    
    train_col = '#3c6e71'
    test_col = '#353535'
    parity_col = '#7b2d26'
    parity_error_alpha = 0.09

    rect_col = '#a3a380'
    rect_col_alpha = 0.4
    alpha_hist = 0.9

    style = 'normal'
    s = 25
    s1 = 15
    w =1
    minimum = int(min(y+test_pred+train_pred))
    maximum = int(max(y+test_pred+train_pred))
    # minimum = 180
    # maximum = 369
    
    fig, ax = plt.subplots(figsize=(5,5))
    # _=plt.ylim(minimum, maximum)
    # _=plt.xlim(minimum, maximum)
    _= plt.title(name, fontsize= s, fontweight = 'normal')
    _= plt.plot([minimum, maximum],[minimum, maximum], color = parity_col, lw = 1.2, ls = (5,(10,3))) #ls = '--')
    x1 = np.array([minimum, maximum])
    
    _= plt.plot(x1, 0.9*x1   , color = parity_col, lw = 0.7, ls = 'dotted', zorder= 10)
    _= plt.plot(x1, 1.1*x1  , color = parity_col,  lw = 0.7, ls = 'dotted', zorder= 10)
    _= plt.fill_between(x1, 0.9*x1, 1.1*x1 ,alpha = parity_error_alpha, color = parity_col, zorder= 10) #-10

    _= plt.scatter(train_actual,train_pred,color = ['white'] , s=35, linewidth = marker_lw,edgecolors=train_col, marker = '^' ,rasterized=True)
    _= plt.scatter(train_actual,train_pred,color = [train_col] , s=35,alpha = marker_alpha, marker ='^',rasterized=True)

    _= plt.scatter(test_actual,test_pred,color = ['white'] , s = 35, linewidth = marker_lw,edgecolors=test_col, marker= 'o', rasterized=True)
    _= plt.scatter(test_actual,test_pred,color = [test_col] , s=35,alpha = marker_alpha, marker = 'o',rasterized=True)

    _= plt.text(0.84,0.68,'- 10%',  transform=ax.transAxes,fontsize= s1, fontweight ='normal', zorder= 10, color = parity_col)  #(from left, from bottom)
    _= plt.text(0.6,0.77, '+ 10%', transform=ax.transAxes, fontsize= s1, fontweight ='normal', zorder= 10, color = parity_col)

    X1 = [minimum, ((minimum+maximum)/2),maximum]
    _= plt.xticks(X1, X1,rotation = 0, fontweight = style, fontsize = s)
    _= plt.yticks(X1,X1,rotation = 0, fontweight = style, fontsize = s)
    _= ax.tick_params('both', length=9, width=w, which='major')
    _= ax.tick_params('both', length=4.5, width=w, which='minor')
    _= plt.tick_params(axis="both", which="both", direction = 'in', right = True, left = True, top = True)
    _= plt.xlabel('Measured '+ label, fontsize= s, fontweight = 'normal')
    _= plt.ylabel('Predicted '+ label, fontsize= s, fontweight = 'normal')
    _= ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # 2 = one subdivision
    _= ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    # leg= plt.legend(labelspacing = 0, loc = 'upper right', frameon = True,prop = {'weight' : 'normal', 'size' : 14.5},
    #                 borderpad=0.2,   # Adjust this to control space between the text and the border    #upper right
    #                 handlelength=0.5,  # Adjust this to shorten the space before the legend text
    #                 handletextpad=0.5 ) # Adjust this to remove space between the legend marker and text)

      # incet train
    left, bottom, width, height = [0.67, 0.19, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    rect = plt.Rectangle(
      (left, bottom),
      width, height,
      transform=fig.transFigure,
      color='white',
      zorder=4)
    fig.patches.append(rect)
    
    p = np.array(train_pred)
    a = np.array(train_actual)
    std=np.std(p - a)
    mean=np.mean(p - a)
    xx=mean-2*std
    yy = 0
    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='white',range = [-4*std, 4*std], rasterized=True)
    rec = patches.Rectangle((xx,yy),4*std, 1.15*find_max(y1), facecolor = rect_col, alpha = rect_col_alpha)
    ax2.add_patch(rec)

    # _= plt.axvline(x = xx, color = 'red', lw =0.6, ls = '--')
    # _= plt.axvline(x = xx+4*std, color = 'red', lw =0.6, ls = '--')

    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color = train_col ,range = [-4*std, 4*std], alpha = alpha_hist, ec= 'none') #ec= 'grey'

    ax2.set_ylim(0,1.1*find_max(y1))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    xlabel_ax2 = ax2.get_xticklabels()
    _=ax2.yaxis.tick_left()
    # ax2.yaxis.set_tick_params(labelsize =10,weight = 'normal')
    _= plt.xticks(fontweight = style, fontsize = s1)
    _= plt.yticks(fontweight = style, fontsize = s1)
    _= ax2.xaxis.set_tick_params(labelsize =12)  #11.8
    _= ax2.yaxis.set_tick_params(labelsize =12)
    _= ax2.tick_params('both', length=7.5, width=w, which='major', direction = 'in')
    _= ax2.tick_params('both', length=5, width=w, which='minor', direction = 'in')
    _=ax2.yaxis.set_label_position("left")
    _=ax2.set_ylabel('\u03BD',fontsize=s1, fontweight = style, labelpad = 0, fontname='DejaVu Sans')
    _=ax2.set_xlabel(r'$\epsilon$',fontsize=s1+2, fontweight = style, labelpad = 0)
    _= ax2.xaxis.set_minor_locator(AutoMinorLocator(1))  # 2 = one subdivision
    _= ax2.yaxis.set_minor_locator(AutoMinorLocator(1))
    
    # # incet test
    left, bottom, width, height = [0.15, 0.66, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    rect = plt.Rectangle(
      (left, bottom),
      width, height,
      transform=fig.transFigure,
      color='white',
      zorder=4)
    fig.patches.append(rect)
    p = np.array(test_actual)
    a = np.array(test_pred)
    std=np.std(p - a)
    mean=np.mean(p - a)
    xx=mean-2*std
    yy = 0
    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='white',range = [-4*std, 4*std], rasterized=True )
    rec = patches.Rectangle((xx,yy),4*std, 1.15*find_max(y1), facecolor = rect_col, alpha = rect_col_alpha)
    ax2.add_patch(rec)

    # _= plt.axvline(x = xx, color = 'red', lw =0.6, ls = '--')
    # _= plt.axvline(x = xx+4*std, color = 'red', lw =0.6, ls = '--')

    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color=test_col,range = [-4*std, 4*std], alpha = alpha_hist, ec= 'none') #grey

    ax2.set_ylim(0,1.1*np.max(y1))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    xlabel_ax2 = ax2.get_xticklabels()
    ax2.yaxis.tick_right()
    # ax2.yaxis.set_tick_params(labelsize =10,weight = 'normal')
    _= plt.xticks(fontweight = style, fontsize = s1)
    _= plt.yticks(fontweight = style, fontsize = s1)
    _= ax2.xaxis.set_tick_params(labelsize =12)  # x tick labels
    _= ax2.yaxis.set_tick_params(labelsize =12) # y tick labels
    _= ax2.tick_params('both', length=7.5, width=w, which='major', direction = 'in')
    _= ax2.tick_params('both', length=5, width=w, which='minor', direction = 'in')
    ax2.yaxis.set_label_position("right")
    _=ax2.set_ylabel(r"$\nu$",fontsize=s1, fontweight = style, labelpad = 0, fontname='DejaVu Sans')
    _=ax2.set_xlabel(r'$\epsilon$',fontsize=s1+2, fontweight = style, labelpad = 0)
    _= ax2.xaxis.set_minor_locator(AutoMinorLocator(1))  # 2 = one minor tick
    _= ax2.yaxis.set_minor_locator(AutoMinorLocator(1))
    if fig_path != None:
        _= plt.savefig(fig_path, bbox_inches='tight', transparent = True, dpi = 1000)  
    _=plt.show()
    
    
    
def parity_train_test(model, name, label, X_train, X_test, y_train, y_test,avg_train_results,y, fig_path=None):
    matplotlib.rcParams['axes.linewidth'] = 1
    marker_alpha = 0.055
    marker_lw = 1.1
    
    train_col = '#3c6e71'
    test_col = '#353535'
    parity_col = '#7b2d26'
    parity_error_alpha = 0.09

    rect_col = '#a3a380'
    rect_col_alpha = 0.4
    alpha_hist = 0.9

    test_actual = y_test[y_test.columns[0]].to_list()
    train_actual = y_train[y_train.columns[0]].to_list()
    if name in ['Linear regression', 'Ridge','SVR','GPR', 'NN']:
        test_pred = array_to_list(model.predict(X_test))
        train_pred = array_to_list(model.predict(X_train))
        [r2_train, mae_train, mape_train] = avg_train_results
        r2_test , mae_test, mape_test = get_score('Test',actual=y_test,predicted=model.predict(X_test))
 
    else: # Lasso, random forest, XGboost
        test_pred =  list(model.predict(X_test))
        train_pred = list(model.predict(X_train))
        [r2_train, mae_train, mape_train] = avg_train_results
        r2_test , mae_test, mape_test   = get_score('Test',actual=y_test,predicted=model.predict(X_test))
    test_results = [r2_test,mae_test, mape_test]
    style = 'normal'
    s = 25
    s1 = 15
    w =1
    minimum = int(min(array_to_list(y.values)+test_pred+train_pred).round(0))
    maximum = int(max(array_to_list(y.values)+test_pred+train_pred).round(0))
    # minimum = 180
    # maximum = 369
    
    fig, ax = plt.subplots(figsize=(5,5))
    # _=plt.ylim(minimum, maximum)
    # _=plt.xlim(minimum, maximum)
    _= plt.title(name, fontsize= s, fontweight = 'normal')
    _= plt.plot([minimum, maximum],[minimum, maximum], color = parity_col, lw = 1.2, ls = (5,(10,3))) #ls = '--')
    x1 = np.array([minimum, maximum])
    
    _= plt.plot(x1, 0.9*x1   , color = parity_col, lw = 0.7, ls = 'dotted', zorder= 10)
    _= plt.plot(x1, 1.1*x1  , color = parity_col,  lw = 0.7, ls = 'dotted', zorder= 10)
    _= plt.fill_between(x1, 0.9*x1, 1.1*x1 ,alpha = parity_error_alpha, color = parity_col, zorder= 10) #-10

    _= plt.scatter(train_actual,train_pred,color = ['white'] , s=35, linewidth = marker_lw,edgecolors=train_col, marker = '^' ,label = 'MAPE$_{train}$: '+f'{mape_train:.2f}'+'%' ,rasterized=True)
    _= plt.scatter(train_actual,train_pred,color = [train_col] , s=35,alpha = marker_alpha, marker ='^',rasterized=True)

    _= plt.scatter(test_actual,test_pred,color = ['white'] , s = 35, linewidth = marker_lw,edgecolors=test_col, marker= 'o', label = 'MAPE$_{test}$: '+f'{mape_test:.2f}' +'%',rasterized=True)
    _= plt.scatter(test_actual,test_pred,color = [test_col] , s=35,alpha = marker_alpha, marker = 'o',rasterized=True)

    _= plt.text(0.84,0.68,'- 10%',  transform=ax.transAxes,fontsize= s1, fontweight ='normal', zorder= 10, color = parity_col)  #(from left, from bottom)
    _= plt.text(0.6,0.77, '+ 10%', transform=ax.transAxes, fontsize= s1, fontweight ='normal', zorder= 10, color = parity_col)

    X1 = [minimum, ((minimum+maximum)/2),maximum]
    _= plt.xticks(X1, X1,rotation = 0, fontweight = style, fontsize = s)
    _= plt.yticks(X1,X1,rotation = 0, fontweight = style, fontsize = s)
    _= ax.tick_params('both', length=9, width=w, which='major')
    _= ax.tick_params('both', length=4.5, width=w, which='minor')
    _= plt.tick_params(axis="both", which="both", direction = 'in', right = True, left = True, top = True)
    _= plt.xlabel('Measured '+ label, fontsize= s, fontweight = 'normal')
    _= plt.ylabel('Predicted '+ label, fontsize= s, fontweight = 'normal')
    _= ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # 2 = one subdivision
    _= ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    leg= plt.legend(labelspacing = 0, loc = 'upper right', frameon = True,prop = {'weight' : 'normal', 'size' : 14.5},
                    borderpad=0.2,   # Adjust this to control space between the text and the border    #upper right
                    handlelength=0.5,  # Adjust this to shorten the space before the legend text
                    handletextpad=0.5 ) # Adjust this to remove space between the legend marker and text)
    # transparent frameon
    # frame = leg.get_frame()
    # frame.set_facecolor("none")   
    # frame.set_edgecolor('#343a40')  

      # incet train
    left, bottom, width, height = [0.67, 0.19, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    p = np.array(train_pred)
    a = np.array(train_actual)
    std=np.std(p - a)
    mean=np.mean(p - a)
    xx=mean-2*std
    yy = 0
    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='white',range = [-4*std, 4*std], rasterized=True)
    rec = patches.Rectangle((xx,yy),4*std, 1.15*find_max(y1), facecolor = rect_col, alpha = rect_col_alpha)
    ax2.add_patch(rec)

    # _= plt.axvline(x = xx, color = 'red', lw =0.6, ls = '--')
    # _= plt.axvline(x = xx+4*std, color = 'red', lw =0.6, ls = '--')

    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color = train_col ,range = [-4*std, 4*std], alpha = alpha_hist, ec= 'none') #ec= 'grey'

    ax2.set_ylim(0,1.1*find_max(y1))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    xlabel_ax2 = ax2.get_xticklabels()
    _=ax2.yaxis.tick_left()
    # ax2.yaxis.set_tick_params(labelsize =10,weight = 'normal')
    _= plt.xticks(fontweight = style, fontsize = s1)
    _= plt.yticks(fontweight = style, fontsize = s1)
    _= ax2.xaxis.set_tick_params(labelsize =12)  #11.8
    _= ax2.yaxis.set_tick_params(labelsize =12)
    _= ax2.tick_params('both', length=7.5, width=w, which='major', direction = 'in')
    _= ax2.tick_params('both', length=5, width=w, which='minor', direction = 'in')
    _=ax2.yaxis.set_label_position("left")
    _=ax2.set_ylabel('\u03BD',fontsize=s1, fontweight = style, labelpad = 0, fontname='DejaVu Sans')
    _=ax2.set_xlabel(r'$\epsilon$',fontsize=s1+2, fontweight = style, labelpad = 0)
    _= ax2.xaxis.set_minor_locator(AutoMinorLocator(1))  # 2 = one subdivision
    _= ax2.yaxis.set_minor_locator(AutoMinorLocator(1))
    
    # # incet test
    left, bottom, width, height = [0.15, 0.66, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height])
    p = np.array(test_actual)
    a = np.array(test_pred)
    std=np.std(p - a)
    mean=np.mean(p - a)
    xx=mean-2*std
    yy = 0
    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='white',range = [-4*std, 4*std], rasterized=True )
    rec = patches.Rectangle((xx,yy),4*std, 1.15*find_max(y1), facecolor = rect_col, alpha = rect_col_alpha)
    ax2.add_patch(rec)

    # _= plt.axvline(x = xx, color = 'red', lw =0.6, ls = '--')
    # _= plt.axvline(x = xx+4*std, color = 'red', lw =0.6, ls = '--')

    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color=test_col,range = [-4*std, 4*std], alpha = alpha_hist, ec= 'none') #grey

    ax2.set_ylim(0,1.1*np.max(y1))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    xlabel_ax2 = ax2.get_xticklabels()
    ax2.yaxis.tick_right()
    # ax2.yaxis.set_tick_params(labelsize =10,weight = 'normal')
    _= plt.xticks(fontweight = style, fontsize = s1)
    _= plt.yticks(fontweight = style, fontsize = s1)
    _= ax2.xaxis.set_tick_params(labelsize =12)  # x tick labels
    _= ax2.yaxis.set_tick_params(labelsize =12) # y tick labels
    _= ax2.tick_params('both', length=7.5, width=w, which='major', direction = 'in')
    _= ax2.tick_params('both', length=5, width=w, which='minor', direction = 'in')
    ax2.yaxis.set_label_position("right")
    _=ax2.set_ylabel(r"$\nu$",fontsize=s1, fontweight = style, labelpad = 0, fontname='DejaVu Sans')
    _=ax2.set_xlabel(r'$\epsilon$',fontsize=s1+2, fontweight = style, labelpad = 0)
    _= ax2.xaxis.set_minor_locator(AutoMinorLocator(1))  # 2 = one minor tick
    _= ax2.yaxis.set_minor_locator(AutoMinorLocator(1))
    if fig_path != None:
        _= plt.savefig(fig_path, bbox_inches='tight', transparent = True, dpi = 1000)  
    _=plt.show()
    return(test_results)

def temporal_discont_plot(sequence, label, index_plot, month_plot, missing_index, fig_path):                                   
    matplotlib.rcParams['axes.linewidth'] = 0.5
    s = 13
    w = 0.5
    values_raw = np.array(sequence, dtype=np.float64) 
    figure, ax = plt.subplots(figsize=(9,4))  
    _= ax.spines['top'].set_visible(False)
    _= ax.spines['right'].set_visible(False)
    _= ax.spines['left'].set_position(('outward', 8))
    _= ax.spines['bottom'].set_position(('outward', 10))

    _= ax.set_xticks(index_plot[0::2], labels = month_plot[0::2] , rotation = 0, fontsize= s, fontweight= 'normal') # ['2020\nFeb', '2020\nApril', '2020\nJune', '2020\nAug', '2020\nOct', '2020\nDec','2021\nFeb', '2021\nApril', '2021\nJune', '2021\nAug', '2021\nOct', '2021\nDec']
    #index_plot[0::2] start at 0 index, step by 2
    _= ax.set_xticks(index_plot[1::2], minor=True)
    _= plt.yticks(rotation = 0, fontsize=s, fontweight= 'normal')
    
    _= ax.tick_params(which = 'major', width= w, length = 8)
    _= ax.tick_params(which = 'minor', width= w, length = 4)
    
    _= ax.tick_params(axis='x', direction ='in', which = 'both')
    _= ax.tick_params(axis='y', direction ='in', which = 'both')
    
    _= ax.tick_params(axis='y', which='both', right= False)
    _= ax.tick_params(axis='x', which='both', top = False)
    
    _= ax.set_ylabel(label, fontsize=s, fontweight='normal')
    _= ax.set_xlabel('Time (Months)', fontsize= s, fontweight='normal')

    _= ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    
    ax.scatter([i for i in range(len(values_raw))],values_raw, marker = '.', color = '#736f72', alpha = 0.09,s = 20, rasterized=True)  
    plt.vlines(x=missing_index, ymin=min(values_raw), ymax=max(values_raw), color= '#b7b7a4' #b6a39e',
           ,linestyle='solid' , linewidth=0.3, alpha = 0.1, rasterized = True)
    if fig_path != None:          
        _= plt.savefig(fig_path, bbox_inches='tight', transparent = True, dpi = 1000)  
    _=plt.show()
    
def temporal_discont_small(sequence, label, index_plot, month_plot, missing_index, fig_path):                                   
    matplotlib.rcParams['axes.linewidth'] = 0.5
    s = 11.5
    w = 0.5
    values_raw = np.array(sequence, dtype=np.float64) 
    figure, ax = plt.subplots(figsize=(7.51,3.6))  
    _= ax.spines['top'].set_visible(False)
    _= ax.spines['right'].set_visible(False)
    _= ax.spines['left'].set_position(('outward', 8))
    _= ax.spines['bottom'].set_position(('outward', 10))

    _= ax.set_xticks(index_plot[0::2], labels = month_plot[0::2] , rotation = 0, fontsize= s, fontweight= 'normal') # ['2020\nFeb', '2020\nApril', '2020\nJune', '2020\nAug', '2020\nOct', '2020\nDec','2021\nFeb', '2021\nApril', '2021\nJune', '2021\nAug', '2021\nOct', '2021\nDec']
    #index_plot[0::2] start at 0 index, step by 2
    _= ax.set_xticks(index_plot[1::2], minor=True)
    _= plt.yticks(rotation = 0, fontsize=s, fontweight= 'normal')
    
    _= ax.tick_params(which = 'major', width= w, length = 8)
    _= ax.tick_params(which = 'minor', width= w, length = 4)
    
    _= ax.tick_params(axis='x', direction ='in', which = 'both')
    _= ax.tick_params(axis='y', direction ='in', which = 'both')
    
    _= ax.tick_params(axis='y', which='both', right= False)
    _= ax.tick_params(axis='x', which='both', top = False)
    
    _= ax.set_ylabel(label, fontsize=s, fontweight='normal')
    _= ax.set_xlabel('Time (Months)', fontsize= s, fontweight='normal')

    _= ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    # _= ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    
    ax.scatter([i for i in range(len(values_raw))],values_raw, marker = '.', color = '#736f72', alpha = 0.09,s = 20, rasterized=True)  
    plt.vlines(x=missing_index, ymin=min(values_raw), ymax=max(values_raw), color= '#b7b7a4' #b6a39e',
           ,linestyle='solid' , linewidth=0.3, alpha = 0.1, rasterized = True)
    if fig_path != None:          
        _= plt.savefig(fig_path, bbox_inches='tight', transparent = True, dpi = 1000)  
    _=plt.show()  

    

def log_search_space(results, file_note=None, log_file_path=None):
    df = pd.DataFrame({
        'iteration': list(range(len(results['params']))),
        'train_mae': -1 * results['mean_train_score'],
        'val_mae': -1 * results['mean_test_score'],
        'params': results['params']
    })

    df_sorted = df.sort_values(by='val_mae', ascending=True).reset_index(drop=True)

    if log_file_path is not None:
        if file_note is None:
            file_note = ''
        with open(log_file_path, 'w') as f:
            f.write(file_note + "\n\n")
            f.write(
                'THESE ARE SCORES FOR SINGLE FOLD, WILL BE DIFFERENT FROM AVG SCORES ON K FOLDS CALCULATED BY K_FOLD_SCORE FUNCTION\n\n\n'
            )
            for _, row in df_sorted.iterrows():
                # Remove any "model__regressor__" or "model__" prefix
                cleaned_params = {
                    k.replace("model__regressor__", "").replace("model__", ""): v
                    for k, v in row['params'].items()
                }

                params_str = ", ".join(
                    f'{k}= "{v}"' if isinstance(v, str) else f'{k}= {v}'
                    for k, v in cleaned_params.items()
                )

                f.write(f"Original Iteration {row['iteration']}:\n")
                f.write(f"Val   MAE: {row['val_mae']:.4f}\n")
                f.write(f"Train MAE: {row['train_mae']:.4f}\n")
                f.write(params_str + "\n\n")



                
def RandomSearchCV_pipeline(regressor, hyperparameter_grid, X_train, y_train, iterations, verbose=2):
    X_train_grid, X_val, y_train_grid, y_val = train_test_split(X_train, y_train, 
                                                                test_size=0.25, random_state = 100, 
                                                                stratify=None)
    X_combined = np.concatenate((X_train_grid, X_val), axis=0)
    y_combined = np.concatenate((y_train_grid, y_val), axis=0)
    test_fold = [-1] * len(X_train_grid) + [0] * len(X_val)
    ps = PredefinedSplit(test_fold)
    n_iter_search = iterations
    random_search = RandomizedSearchCV(
        estimator=regressor,
        param_distributions=hyperparameter_grid,
        n_iter=n_iter_search,
        cv=ps, 
        scoring='neg_mean_absolute_error',
        verbose=verbose,
        random_state= 100,
        return_train_score=True 
    )
    random_search.fit(X_combined, y_combined)
    best_regressor = random_search.best_estimator_
    # print('best params:'+f'{random_search.best_params_}')
    return(best_regressor,random_search)

def save_model(model, model_path =None):
    with open(model_path, 'wb') as f:   # save best model
        pickle.dump(model,f)
        
def load_model(model_path):
    model = pickle.load(open(model_path, 'rb'))
    return(model)

def flatten_grid(grid, step_name='model__regressor'):
    return {f"{step_name}__{k}": v for k, v in grid.items()}


def model_size(model_path):
    model_size = os.path.getsize(model_path)/(1024*1024)  # Convert bytes to MB
    print(f"Model size: {model_size:.2f} MB")
    
    
def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # minus one for clockwise labels
            self.set_theta_zero_location('N')
            # Add this line to make angles increase clockwise
            self.set_theta_direction(-1)

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor= '#343a40' )
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
            

                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

def spine_param(ax, lw, col):
    spine = ax.spines['polar']
    spine.set_linewidth(lw)
    spine.set_edgecolor(col) #'#343a40'

    
def save_fig(path):
    _= plt.savefig(path, bbox_inches='tight', transparent = True, dpi =1000)  
    
def MLP_training_loss_seeds(seeds, base_path, title, fig_path):  
    loss_curves = []
    s = 17                                 
    matplotlib.rcParams['axes.linewidth'] = 0.5
    w =0.5
    for model_no in tqdm(range(seeds)):
        nn = pickle.load(open(base_path + f'{model_no}.pkl', 'rb'))
        loss_curves.append(nn.loss_curve_)
    # Pad shorter loss curves with NaN so they can be handled properly
    max_len = max(len(lc) for lc in loss_curves)
    loss_curves_padded = np.array([np.pad(lc, (0, max_len - len(lc)), constant_values=np.nan) for lc in loss_curves])
    # Compute mean and std ignoring NaNs
    mean_loss = np.nanmean(loss_curves_padded, axis=0)
    std_factor = 3
    std_loss = np.nanstd(loss_curves_padded, axis=0)* std_factor
    # Plotting
    fig, ax = plt.subplots(figsize=(5,5))
    _= ax.spines['top'].set_visible(False)
    _= ax.spines['right'].set_visible(False)
    _= ax.spines['left'].set_position(('outward', 20))
    _= ax.spines['bottom'].set_position(('outward', 20))
    _= ax.tick_params(which = 'major', width= w, length = 8)
    _= ax.tick_params(which = 'minor', width= w, length = 4)
    _= ax.tick_params(axis='x', direction ='in', which = 'both')
    _= ax.tick_params(axis='y', direction ='in', which = 'both')
    _= ax.tick_params(axis='y', which='both', right= False)
    _= ax.tick_params(axis='x', which='both', top = False)
    _= ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    _= ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    _= plt.xticks(fontsize=s, fontweight='normal')
    _= plt.yticks(fontsize=s, fontweight='normal')

    x = np.arange(len(mean_loss))
    _ = plt.plot(x, mean_loss, label='Mean training loss (MSE)', color='#484a47')
    _ = plt.fill_between(x, mean_loss - std_loss, mean_loss + std_loss, color='indianred', alpha=0.3, label=f'±{std_factor} Std Dev')
    _ = plt.xlabel("Iterations", fontsize=s, fontweight='normal')
    _ = plt.ylabel("Training loss (MSE)",fontsize=s, fontweight='normal')
    _ = plt.title(title,fontsize=s, fontweight='normal')
    _= ax.xaxis.grid(True, linestyle="solid", which = 'major' , color="darkgray", linewidth=0.1, alpha=0.7)
    _= ax.yaxis.grid(True, linestyle="solid", which = 'major' , color="darkgray", linewidth=0.1, alpha=0.7)
    _ = plt.legend(
        loc='upper right',         # anchor point on the legend box
        bbox_to_anchor=(1,0.97),     # position of that anchor in Axes coords
        frameon=True,
        facecolor='white',
        edgecolor='gray',
        fontsize=s-2,
        labelspacing=0.3,
        handlelength=0.6,
        borderpad=0.25,
        framealpha=0.5
    )
    save_fig(fig_path)
    _ = plt.show()
    
    
def parity_train_test_no_model_s_font(label, train_actual, test_actual, train_pred,test_pred, y, fig_path=None):
    matplotlib.rcParams['axes.linewidth'] = 0.5
    marker_alpha = 0.055
    marker_lw = 1.1
    
    train_col ='#bc8034'
    test_col = '#353535'
    parity_col = 'mediumturquoise'
    parity_error_alpha = 0.09

    rect_col = '#a3a380'
    rect_col_alpha = 0.4
    alpha_hist = 0.9

    style = 'normal'
    s = 20
    s1 = 15
    w =0.5
    minimum = int(min(y+test_pred+train_pred))
    maximum = int(max(y+test_pred+train_pred))
    # minimum = 180
    # maximum = 369
    
    fig, ax = plt.subplots(figsize=(5,5))
    # _=plt.ylim(minimum, maximum)
    # _=plt.xlim(minimum, maximum)
    _= plt.plot([minimum, maximum],[minimum, maximum], color = parity_col, lw = 1.2, ls = (5,(10,3))) #ls = '--')
    x1 = np.array([minimum, maximum])
    
    # _= plt.plot(x1, 0.9*x1   , color = parity_col, lw = 0.7, ls = 'dotted', zorder= 10)
    # _= plt.plot(x1, 1.1*x1  , color = parity_col,  lw = 0.7, ls = 'dotted', zorder= 10)
    # _= plt.fill_between(x1, 0.9*x1, 1.1*x1 ,alpha = parity_error_alpha, color = parity_col, zorder= 10) #-10

    _= plt.scatter(train_actual,train_pred,color = ['white'] , s=35, linewidth = marker_lw,edgecolors=train_col, marker = '^' ,rasterized=True, label = 'Train')
    _= plt.scatter(train_actual,train_pred,color = [train_col] , s=35,alpha = marker_alpha, marker ='^',rasterized=True)
    _= plt.scatter(test_actual,test_pred,color = ['white'] , s = 35, linewidth = marker_lw,edgecolors=test_col, marker= 'o', rasterized=True, label = 'Test')
    _= plt.scatter(test_actual,test_pred,color = [test_col] , s=35,alpha = marker_alpha, marker = 'o',rasterized=True)

    # _= plt.text(0.84,0.68,'- 10%',  transform=ax.transAxes,fontsize= s1, fontweight ='normal', zorder= 10, color = parity_col)  #(from left, from bottom)
    # _= plt.text(0.6,0.77, '+ 10%', transform=ax.transAxes, fontsize= s1, fontweight ='normal', zorder= 10, color = parity_col)

    X1 = [minimum, ((minimum+maximum)/2),maximum]
    _= plt.xticks(X1, X1,rotation = 0, fontweight = style, fontsize = s)
    _= plt.yticks(X1,X1,rotation = 0, fontweight = style, fontsize = s)
    _= ax.tick_params('both', length=9, width=w, which='major')
    _= ax.tick_params('both', length=4.5, width=w, which='minor')
    _= plt.tick_params(axis="both", which="both", direction = 'in', right = True, left = True, top = True)
    _= plt.xlabel('Measured '+ label, fontsize= s, fontweight = 'normal')
    _= plt.ylabel('Predicted '+ label, fontsize= s, fontweight = 'normal')
    _= ax.xaxis.set_minor_locator(AutoMinorLocator(2))  # 2 = one subdivision
    _= ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    leg = plt.legend(
        ncol=2,
        labelspacing=0,
        loc='upper right',          # Will be ignored if bbox_to_anchor is used
        frameon=True,               # Must be True for background fill to show
        prop={'weight': 'normal', 'size': s1-3},
        borderpad=0.2,
        handlelength=0.5,
        handletextpad=0.5,
        bbox_to_anchor=(0.85, 0.97),  # (x, y) coordinates relative to axes
    )
    # Set legend facecolor (fill) and transparency (alpha)
    frame = leg.get_frame()
    frame.set_facecolor('white')     # Fill color
    frame.set_edgecolor('gray')      # Optional border color
    frame.set_alpha(1)             # Transparency (0 = fully transparent, 1 = opaque)


      # incet train
    left, bottom, width, height = [0.67, 0.19, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height], zorder = 5)
    rect = plt.Rectangle(
      (left, bottom),
      width, height,
      transform=fig.transFigure,
      color='white',
      zorder= 4)
    fig.patches.append(rect)
    
    p = np.array(train_pred)
    a = np.array(train_actual)
    std=np.std(p - a)
    mean=np.mean(p - a)
    xx=mean-2*std
    yy = 0
    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='white',range = [-4*std, 4*std], rasterized=True)
    rec = patches.Rectangle((xx,yy),4*std, 1.15*find_max(y1), facecolor = rect_col, alpha = rect_col_alpha)
    ax2.add_patch(rec)

    # _= plt.axvline(x = xx, color = 'red', lw =0.6, ls = '--')
    # _= plt.axvline(x = xx+4*std, color = 'red', lw =0.6, ls = '--')

    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color = train_col ,range = [-4*std, 4*std], alpha = alpha_hist, ec= 'none') #ec= 'grey'

    ax2.set_ylim(0,1.1*find_max(y1))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    xlabel_ax2 = ax2.get_xticklabels()
    _=ax2.yaxis.tick_left()
    # ax2.yaxis.set_tick_params(labelsize =10,weight = 'normal')
    _= plt.xticks(fontweight = style, fontsize = s1)
    _= plt.yticks(fontweight = style, fontsize = s1)
    _= ax2.xaxis.set_tick_params(labelsize =s1-3)  #11.8
    _= ax2.yaxis.set_tick_params(labelsize =s1-3)
    _= ax2.tick_params('both', length=7.5, width=w, which='major', direction = 'in')
    _= ax2.tick_params('both', length=5, width=w, which='minor', direction = 'in')
    _=ax2.yaxis.set_label_position("left")
    _=ax2.set_ylabel('\u03BD',fontsize=s1, fontweight = style, labelpad = 0, fontname='DejaVu Sans')
    _=ax2.set_xlabel(r'$\epsilon$',fontsize=s1+2, fontweight = style, labelpad = 0)
    _= ax2.xaxis.set_minor_locator(AutoMinorLocator(1))  # 2 = one subdivision
    _= ax2.yaxis.set_minor_locator(AutoMinorLocator(1))
    
    # # incet test
    left, bottom, width, height = [0.15, 0.66, 0.2, 0.2]
    ax2 = fig.add_axes([left, bottom, width, height], zorder = 5)
    rect = plt.Rectangle(
      (left, bottom),
      width, height,
      transform=fig.transFigure,
      color='white',
      zorder = 4)
    fig.patches.append(rect)
    
    p = np.array(test_actual)
    a = np.array(test_pred)
    std=np.std(p - a)
    mean=np.mean(p - a)
    xx=mean-2*std
    yy = 0
    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color='white',range = [-4*std, 4*std], rasterized=True )
    rec = patches.Rectangle((xx,yy),4*std, 1.15*find_max(y1), facecolor = rect_col, alpha = rect_col_alpha)
    ax2.add_patch(rec)

    # _= plt.axvline(x = xx, color = 'red', lw =0.6, ls = '--')
    # _= plt.axvline(x = xx+4*std, color = 'red', lw =0.6, ls = '--')

    y1,x,_ = ax2.hist(p - a, bins = int(1+3.3*np.log(len(p-a))) ,color=test_col,range = [-4*std, 4*std], alpha = alpha_hist, ec= 'none') #grey

    ax2.set_ylim(0,1.1*np.max(y1))
    ax2.xaxis.set_major_locator(plt.MaxNLocator(3))
    xlabel_ax2 = ax2.get_xticklabels()
    ax2.yaxis.tick_right()
    # ax2.yaxis.set_tick_params(labelsize =10,weight = 'normal')
    _= plt.xticks(fontweight = style, fontsize = s1)
    _= plt.yticks(fontweight = style, fontsize = s1)
    _= ax2.xaxis.set_tick_params(labelsize =12)  # x tick labels
    _= ax2.yaxis.set_tick_params(labelsize =12) # y tick labels
    _= ax2.tick_params('both', length=7.5, width=w, which='major', direction = 'in')
    _= ax2.tick_params('both', length=5, width=w, which='minor', direction = 'in')
    ax2.yaxis.set_label_position("right")
    _=ax2.set_ylabel(r"$\nu$",fontsize=s1, fontweight = style, labelpad = 0, fontname='DejaVu Sans')
    _=ax2.set_xlabel(r'$\epsilon$',fontsize=s1+2, fontweight = style, labelpad = 0)
    _= ax2.xaxis.set_minor_locator(AutoMinorLocator(1))  # 2 = one minor tick
    _= ax2.yaxis.set_minor_locator(AutoMinorLocator(1))
    if fig_path != None:
        _= plt.savefig(fig_path, bbox_inches='tight', transparent = True, dpi = 1000)  
    _=plt.show()
    
def col_compare(col1, col2, col3):
    fig, ax = plt.subplots(figsize=(6, 6))
    # Plot three overlapping circles
    circle1 = plt.Circle((-0.3, 0), 0.6, color=col1)
    circle2 = plt.Circle((0.3, 0), 0.6, color=col2)
    circle3 = plt.Circle((0, 0.4), 0.6, color=col3)
    for c in [circle1, circle2, circle3]:
        ax.add_artist(c)

    # Add text labels centered on each circle
    ax.text(-0.8, 0, 'col1', color='red', fontsize=12, ha='center', va='center')
    ax.text(0.8, 0, 'col2', color='red', fontsize=12, ha='center', va='center')
    ax.text(0, 0.8, 'col3', color='red', fontsize=12, ha='center', va='center')

    # Formatting
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.show()
    
def array_scale(arr):
    scaled_arr = (raw_nox - arr.min()) / (arr.max() - arr.min())
    return(scaled_arr)

# =====================================================================
#  HELPER: Extract and organize plotting configuration from info{}
# =====================================================================
def get_config(info):
    return {
        "limits": info.get("limits"),
        "axis_limits": info.get("axis_limits", None),
        "label": info.get("label", ""),
        "title": info.get("title", None),
        "n_major_ticks": info.get("n_major_ticks", 6),
        "clip_std_multiplier": info.get("cmap_scale", 1),
        "margin": info.get("margin", 0.0),
        "tick_decimals": info.get("tick_decimals", 5),
        "fig_path": info.get("fig_path", None),
        "show_cbar": info.get("show_cbar", True),
        "tick_labels": info.get("tick_labels", None) 
    }
# =====================================================================
#  MAIN: 2D Density Plot + Residual Inset
# =====================================================================
def make_density_plot(info, y_true, y_pred):
    cfg = get_config(info)
    s = 14             # main axis font size
    s1 = 9           # inset font size
    w = 0.5            # tick width

    # ---------- FIGURE ----------
    fig, ax = plt.subplots(figsize=(2.7, 2.7))
    matplotlib.rcParams['axes.linewidth'] = 0.5
    _= ax.spines['top'].set_visible(False)
    _= ax.spines['right'].set_visible(False)
    _= ax.spines['left'].set_position(('outward', 15))
    _= ax.spines['bottom'].set_position(('outward', 15))
    for spine in ax.spines.values():
        spine.set_linewidth(w)

    # ---------- HISTOGRAM ----------
    bins = 100
    hist_data, xedges, yedges = np.histogram2d(
        y_true.ravel(),
        y_pred.ravel(),
        bins=(bins, bins),
        range=np.array([cfg["limits"], cfg["limits"]]),
        density=False
    )

    clip_value = hist_data.mean() + cfg["clip_std_multiplier"] * hist_data.std()
    hist_data = np.minimum(hist_data, clip_value)

    # ---------- COLORMAP ----------
    base = plt.cm.get_cmap("viridis_r", 256)
    colors = base(np.linspace(0, 1, 256))
    colors[0] = [1, 1, 1, 1]
    cmap = ListedColormap(colors)
    # ___________________2D historgram density map
    im = ax.imshow(
        hist_data.astype(int),
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        cmap=cmap,
        origin="lower",
        aspect="auto"
    )
    im.set_rasterized(True)

    # ---------- LABELS ----------
    clean_label = cfg["label"].replace('*', '')
    ax.set_xlabel('Measured ' + clean_label, fontsize=s-2)
    ax.set_ylabel('Predicted ' + clean_label, fontsize=s-2)

    # ---------- title ----------
    ax.set_title(cfg['title'], fontsize=s-1, pad = 11)
    
    # ---------- AXIS LIMITS ----------
    if cfg["axis_limits"] is not None:
        axis_min_raw, axis_max_raw = cfg["axis_limits"]
    else:
        axis_min_raw, axis_max_raw = cfg["limits"]
        
        # ---------- MAJOR TICKS ----------
    if cfg["tick_labels"] is not None:
        # User-specified tick labels
        tick_positions = cfg["tick_labels"]
    else:
        # Automatically generate based on number of ticks
        n = cfg["n_major_ticks"]
        tick_positions = np.linspace(axis_min_raw, axis_max_raw, n)

    # Apply margin outside the tick range
    margin = cfg["margin"]
    axis_min = tick_positions[0] - margin
    axis_max = tick_positions[-1] + margin

    # Apply limits
    ax.set_xlim([axis_min, axis_max])
    ax.set_ylim([axis_min, axis_max])

    # Draw reference y = x line
    ax.plot([axis_min, axis_max], [axis_min, axis_max], ":", c="gray")

    # Set tick locations
    ax.xaxis.set_major_locator(plticker.FixedLocator(tick_positions))
    ax.yaxis.set_major_locator(plticker.FixedLocator(tick_positions))

    # Set tick labels (if provided)
    if cfg["tick_labels"] is not None:
        ax.set_xticklabels(cfg["tick_labels"], fontsize = s)
        # ax.set_yticklabels(cfg["tick_labels"], fontsize = s, rotation = 90)
        ax.set_yticklabels(cfg["tick_labels"], fontsize=s)
        for label in ax.get_yticklabels():
            label.set_rotation(90)
            label.set_va('center')
            # label.set_ha('center')


    margin = cfg["margin"]
    axis_min = tick_positions[0] - margin
    axis_max = tick_positions[-1] + margin

    ax.set_xlim([axis_min, axis_max])
    ax.set_ylim([axis_min, axis_max])

    # ---------- REFERENCE LINE ----------
    ax.plot([axis_min, axis_max], [axis_min, axis_max], ":", c="gray")

    # ---------- TICKS (MAJOR) ----------
    ax.xaxis.set_major_locator(plticker.FixedLocator(tick_positions))
    ax.yaxis.set_major_locator(plticker.FixedLocator(tick_positions))

    dec = cfg["tick_decimals"]
    fmt = "%d" if dec == 0 else "%." + str(dec) + "f"
    ax.xaxis.set_major_formatter(plticker.FormatStrFormatter(fmt))
    ax.yaxis.set_major_formatter(plticker.FormatStrFormatter(fmt))

    ax.tick_params('both', length=8, width=w, which='major', direction='in')
    ax.tick_params('both', right=False, left=True, top=False)

    # ---------- TICKS (MINOR) ----------
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params('both', which='minor', length=4.5, width=w, direction='in')

    # ---------- COLORBAR ----------
    if cfg["show_cbar"]:
        divider = make_axes_locatable(ax)
        # Attach the colorbar betolow the main plot
        cax = divider.append_axes("bottom", size="8%", pad=0.85)  # <-- controls position
        # Create horizontal colorbar
        cb = plt.colorbar(im, cax=cax, orientation='horizontal')
        # Set colorbar label and label style
        cb.set_label("Points per pixel", fontsize=s-2, labelpad=6)
        
        cb.ax.tick_params(
            labelsize=s-2,       # 🔹 font size for tick labels
            width=w,             # tick line width
            length=6,            # tick length
            direction='in',      # tick direction ('in', 'out', 'inout')
            bottom=True,         # ticks on bottom
            top=False            # no ticks on top
        )
        
        # # Set custom tick labels
        # # cb.set_ticks([0, 20, 40, 60, 80, 100])  # tick positions
        
        # Set Min, mid and max tick labels
        vmin, vmax = im.get_clim()     # color scale limits from imshow
        mid = 0.5 * (vmin + vmax)      # midpoint

        cb.set_ticks([vmin, mid, vmax])  # three main ticks
        cb.ax.set_xticklabels(
            [f"{vmin:.0f}", f"{mid:.0f}", f"{vmax:.0f}"]  # format as integers
        )

    # =====================================================================
    # RESIDUAL INSET
    # =====================================================================
    residuals = (y_pred - y_true)
    std = residuals.std()

    # inset placement (same as parity plot)
    left, top, width, height = [0.11, 0.88, 0.23, 0.23]
    bottom = top - height
    ax2 = fig.add_axes([left, bottom, width, height], zorder=5)

    # background rectangle
    rect = plt.Rectangle((left, bottom), width, height,
                         transform=fig.transFigure,
                         color='white', zorder=4)
    fig.patches.append(rect)

    # inset histogram
    inset_range = [-4 * std, 4 * std]
    y1, x1, _ = ax2.hist(residuals, bins=50, fc='#353535', range=inset_range, alpha=0.9)

    ax2.axvspan(-1.65 * std, 1.65 * std, color= '#a3a380', alpha=0.4)
    ax2.set_xlim(inset_range)
    ax2.set_ylim(0, 1.1 * np.max(y1))

    # inset labels
    ax2.set_xlabel(r"$\varepsilon$", fontsize=s1+2)
    ax2.set_ylabel(r"$\nu$", fontsize=s1+2)

    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    # inset ticks
    ax2.tick_params('both', length=5.8, width=w, which='major', direction='in')
    ax2.tick_params('both', length=5, width=w, which='minor', direction='in')

    ax2.xaxis.set_major_locator(MaxNLocator(3))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(1))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(1))
    ax2.tick_params(labelsize=s1)

    # ---------- SAVE FIGURE ----------
    if cfg["fig_path"] is not None:
        plt.savefig(cfg["fig_path"], dpi=1000,
                    bbox_inches='tight', transparent=True)

    return fig
