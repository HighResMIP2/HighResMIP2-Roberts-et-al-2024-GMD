'''
This code is part of HighResMIP2-Roberts-et-al-2024-GMD, and plots the 
illustrative Effective Climate Sensitivity (Fig. 5) from three
ensemble members of a piControl and parallel abrupt 4xCO2 experiment, 
and three members from a HighResMIP control-1950 and parallel abrupt 
4xCO2 experiment.
The plot of ECS vs time shows the fit for ECS for all data up to the time 
(year) value shown, showing how this tends to converge to a constant value
after typically ~100 years or so. It also shows a spread based on the 95%
interval from a two-sides Student t-test to indicate the uncertainty
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnchoredText
from scipy.io import readsav
from scipy.stats import t
import scipy
import os
import xarray as xr

workdir = './data/'

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
    yr=delta[0]['year'].values
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
    '''
    Load the data from the netcdf files, for each variable.
    For each pair of ensemble members, subtract the values to produce a 
    difference
    '''

    data = {}
    cntl_suite = runinfo_suite['suites'][0]
    pert_suite = runinfo_suite['suites'][1]
    start_year = runinfo_suite['start_year']
    end_year = runinfo_suite['end_year']
    branch_year = runinfo_suite['branch']
    
    for item in zip(['pert', 'cntl'], [pert_suite, cntl_suite]):
        suite = item[1]
        runtype = item[0]
        print('suite, run ',item)
        if runtype == 'pert':
            startyear = start_year
            endyear = end_year
        else:
            startyear = branch_year
            endyear = branch_year + (end_year-start_year)
        fname = os.path.join(indir, '{suite}_{var}_annual_{startyear}-{endyear}_globalmean.nc')
        #fname = os.path.join(indir, '{suite}_{var}_annual.nc')
        nc = xr.open_dataset(fname.format(suite=suite, var='tas', startyear=startyear, endyear=endyear))
        data[('tas', runtype)]=nc['tas']

        nc = xr.open_dataset(fname.format(suite=suite, var='rlut', startyear=startyear, endyear=endyear))
        data[('rlut', runtype)]=nc['rlut']

        nc = xr.open_dataset(fname.format(suite=suite, var='rsut', startyear=startyear, endyear=endyear))
        data[('rsut', runtype)]=nc['rsut']

        nc = xr.open_dataset(fname.format(suite=suite, var='rsdt', startyear=startyear, endyear=endyear))
        data[('rsdt', runtype)]=nc['rsdt']

        data[('dnet', runtype)] = data[('rsdt', runtype)] - data[('rlut', runtype)] - data[('rsut', runtype)]
        print('fname ',fname.format(suite=suite, var='tas', startyear=startyear, endyear=endyear))
  
    dtas=data[('tas', 'pert')][:149].copy()
    dnet=data[('dnet', 'pert')][:149].copy()

    # calculate the difference, year by year, of the 4xCO2 and control runs
    # Note that, usually for piControl simulations, the net TOA would not 
    # be a difference (since the piControl run should have TOA=0), but this
    # may not be true for the control-1950.
    dtas.data = data[('tas', 'pert')].data[:149] - data[('tas', 'cntl')].data[:149]
    dnet.data = data[('dnet', 'pert')].data[:149] - data[('dnet', 'cntl')].data[:149]

    dT=dtas
    dNET=dnet

    return(dT, dNET)

def plot_data(runs_info, data_ecs, data_ecs_interval):
    # do not plot first 12 years of data, due to the uncertain fit
    data_offset = 12

    plt.figure(figsize=(12,7))
    plt.rcParams.update({'font.size': 14})

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
        data[resol] = load_and_calc_delta(runs_info[resol], workdir)

        #calculate EffCS and confidence interval 
        data_ecs[resol], data_ecs_interval[resol] = calc_ecs_timeseries(data[resol], 0)

    plot_data(runs_info, data_ecs, data_ecs_interval)

if __name__ == '__main__':
    runs_info = {}

    # the ens_info_1850 dictionary describes the piControl and abrupt 4xCO2
    # simulations, each as a pair, for three ensemble members
    # Some of the ensemble members are branched off the control at different
    # times, and so the branch values indicate this, to ensure that the
    # corresponding years are subtracted from the parallel simulations

    # N96 piControl & 4xCO2 EERIE
    title = 'HadGEM3-GC5-EERIE'
    ens_info_1850 = {'resol': ['LL-piControl-1850-ENS1', 'LL-piControl-1850-ENS2', 'LL-piControl-1850-ENS3'], 'suites': [['u-cy163', 'u-db365'], ['u-cy163', 'u-dh625'], ['u-cy163', 'u-dh626']], 'start_year': [1850, 1850, 1851], 'end_year': [2001, 2050, 2050], 'branch': [1850, 1925, 1990] } #, 'LL-piControl-ENS2', 'LL-piControl-ENS3' #, 'LL-piControl-ENS2', 'LL-piControl-ENS3'
    model_cntl = 'piControl'
    calendar='gregorian'
    colours=['blue','slateblue','darkblue']
    for ir, resol in enumerate(ens_info_1850['resol']):
        suites =  ens_info_1850['suites'][ir]
        branch = ens_info_1850['branch'][ir]
        start_year = ens_info_1850['start_year'][ir]
        end_year = ens_info_1850['end_year'][ir]
        runs_info[resol] = set_runid_info(title, suites, resol, branch, start_year, end_year, model_cntl, calendar, colours[ir])

    # the ens_info_1950 dictionary describes the control-1950 and abrupt 4xCO2
    # simulations, each as a pair, for three ensemble members

    # N96 control-1950 & 4xCO2-1950 EERIE
    title = 'HadGEM3-GC5-EERIE'
    ens_info_1950 = {'resol': ['LL-control-1950-ENS1', 'LL-control-1950-ENS2', 'LL-control-1950-ENS3'], 'suites': [['u-de062', 'u-de066'], ['u-de062', 'u-dh631'], ['u-de062', 'u-dh632']], 'start_year': [1950, 1995, 2030], 'end_year': [2100, 2145, 2180], 'branch': [1950, 1995, 2030] } #'LL-control-1950-ENS2', #, 'LL-control-1950-ENS2', 'LL-control-1950-ENS3'
    model_cntl = 'control-1950'
    calendar='gregorian'
    colours=['red', 'salmon', 'tomato']
    for ir, resol in enumerate(ens_info_1950['resol']):
        suites =  ens_info_1950['suites'][ir]
        branch = ens_info_1950['branch'][ir]
        start_year = ens_info_1950['start_year'][ir]
        end_year = ens_info_1950['end_year'][ir]
        runs_info[resol] = set_runid_info(title, suites, resol, branch, start_year, end_year, model_cntl, calendar, colours[ir])

    work(runs_info)
