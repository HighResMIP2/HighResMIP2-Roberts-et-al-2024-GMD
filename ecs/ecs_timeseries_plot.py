import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from scipy.io import readsav
from scipy.stats import t
import scipy
import iris
import iris.coord_categorisation as icc
import os

workdir = './'

def calc_ecs(dT,dN):
    m,c,r,p,stderr=scipy.stats.linregress(dT,dN)
    lambda4x=m
    F4x=c
    ecs=-0.5*F4x/lambda4x
    ecs_95_interval=calc_95_interval(dT,stderr)
    return ecs,m,c,ecs_95_interval

def calc_95_interval(dT, stderr):
    # Two-sided inverse Students t-distribution
    # p - probability, df - degrees of freedom
    #from scipy.stats import t
    tinv = lambda p, df: abs(t.ppf(p/2, df))
    if dT.size<3:
        gradient_95_interval = 0
    else:
        ts = tinv(0.05, dT.size-2)
        gradient_95_interval = ts*stderr
    return gradient_95_interval

def calc_ecs_timeseries(delta, exclude):
    yr=delta[0].coord('year').points
    yr_length = len(yr) - exclude
    dT=delta[0].data[exclude:]
    dN=delta[1].data[exclude:]
    ECS=np.zeros(yr_length)
    lamb=np.zeros(yr_length)
    F=np.zeros(yr_length)
    ECS_interval=np.zeros(yr_length)
    #r2=np.zeros(yr_length)
    for t in np.arange(exclude,yr_length-1):
        ecs,m,c,ecs_interval=calc_ecs(dT[:t+1],dN[:t+1])
        ECS[t]=ecs
        lamb[t]=m
        F[t]=c
        ECS_interval[t]=ecs_interval
        #r2[t]=r
    #plt.plot(np.arange(exclude,len(yr)-1), ECS[:-1])
    #plt.xlabel('Years')
    #plt.ylabel('ECS')
    #plt.show()
    return ECS, ECS_interval


def load_and_calc_delta(runinfo_suite, indir):

    data = {}
    cntl_suite = runinfo_suite['suites'][0]
    pert_suite = runinfo_suite['suites'][1]
    start_year = runinfo_suite['start_year']
    end_year = runinfo_suite['end_year']
    
    for item in zip(['pert', 'cntl'], [pert_suite, cntl_suite]):
        suite = item[1]
        runtype = item[0]
        fname = os.path.join(indir, '{suite}_{var}_annual_{startyear}-{endyear}_globalmean.nc')
        data[('tas', runtype)]=iris.load_cube(fname.format(suite=suite, var='tas', startyear=start_year, endyear=end_year))
        data[('rlut', runtype)]=iris.load_cube(fname.format(suite=suite, var='rlut', startyear=start_year, endyear=end_year))
        data[('rsut', runtype)]=iris.load_cube(fname.format(suite=suite, var='rsut', startyear=start_year, endyear=end_year))
        data[('rsdt', runtype)]=iris.load_cube(fname.format(suite=suite, var='rsdt', startyear=start_year, endyear=end_year))
        data[('dnet', runtype)] = data[('rsdt', runtype)] - data[('rlut', runtype)] - data[('rsut', runtype)]
  
    dtas=data[('tas', 'pert')][:149].copy()
    dnet=data[('dnet', 'pert')][:149].copy()
    years_pert = dtas.coord('year').points
    years_cntl = data[('tas', 'cntl')].coord('year').points
    print('cntl ',len(years_cntl))
    print('pert ',len(years_pert))
    for iy, year in enumerate(years_pert):
        index = np.where(years_cntl == year)[0]
        if len(index) > 0:
            if index < len(data[('tas', 'cntl')].data):
                print('index ',index)
                print('year ',year, years_cntl[index[0]])
                dtas.data[iy] = data[('tas', 'pert')].data[iy] - data[('tas', 'cntl')].data[index[0]]
                dnet.data[iy]=  data[('dnet', 'pert')].data[iy] - data[('dnet', 'cntl')].data[index[0]]

    dT=dtas
    dNET=dnet
  
    return(dT, dNET)

def plot_data(runs_info, data_ecs, data_ecs_interval):
    # do not plot first 12 years
    data_offset = 12

    plt.figure(figsize=(12,7))

    for resol in runs_info:
        ecs = data_ecs[resol]
        ecs_interval = data_ecs_interval[resol]

        plt.plot(np.arange(data_offset,len(ecs)-1), ecs[data_offset:-1], label=resol, c=runs_info[resol]['colour'])

        plt.fill_between(np.arange(data_offset,len(ecs)-1), ecs[data_offset:-1]-ecs_interval[data_offset:-1], ecs[data_offset:-1]+ecs_interval[data_offset:-1], alpha=0.2, color=runs_info[resol]['colour'])

    plt.xlabel('Years')
    plt.ylabel('Effective Climate Sensitivity (K)')
    plt.legend()
    plt.grid()
    figname = 'EERIE_N96O1_diff_baseline_EffCS_evolution_comparison.png'
    plt.savefig(figname[:-4]+'.pdf')
    plt.savefig(figname)
    plt.show()

def set_runid_info(title, suites, resol, branch, start_year, end_year, model_cntl, calendar, colour):
    runid_info = {'title': title, 'suites': suites, 'resol': resol, 'branch': branch, 'start_year': start_year, 'end_year': end_year, 'model_cntl': model_cntl, 'calendar': calendar, 'colour': colour}
    return runid_info

def work(runs_info):

    data = {}
    data_ecs = {}
    data_ecs_interval = {}

    for resol in runs_info:
        # load data
        #data[resol] = load_and_calc_delta(runs_info[resol]['suites'][1], runs_info[resol]['suites'][0], workdir)
        data[resol] = load_and_calc_delta(runs_info[resol], workdir)
        #calculate EffCS and confidence interval 
        data_ecs[resol], data_ecs_interval[resol] = calc_ecs_timeseries(data[resol], 0)

    plot_data(runs_info, data_ecs, data_ecs_interval)

if __name__ == '__main__':
    runs_info = {}

    # the first suite name (run identifier) is the control run, the second the 4xCO2 run

    # N96 piControl & 4xCO2 EERIE
    title = 'HadGEM3-GC5-EERIE'
    suites = ['u-cy163', 'u-db365']
    resol = 'LL-piControl-ENS1'
    branch = 1850
    start_year = 1850
    end_year = 2001
    model_cntl = 'piControl'
    calendar='gregorian'
    colour='blue'
    runs_info[resol] = set_runid_info(title, suites, resol, branch, start_year, end_year, model_cntl, calendar, colour)

    # N96 control-1950 & 4xCO2-1950 EERIE
    title = 'HadGEM3-GC5-EERIE'
    suites = ['u-de062', 'u-de066']
    resol = 'LL-control-1950-ENS1'
    branch = 1950
    start_year = 1950
    end_year = 2100
    model_cntl = 'control-1950'
    calendar='gregorian'
    colour='red'
    runs_info[resol] = set_runid_info(title, suites, resol, branch, start_year, end_year, model_cntl, calendar, colour)

    work(runs_info)
