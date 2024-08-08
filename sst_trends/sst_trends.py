'''
Create trends of SST over period 1980-2015
Based on code at: https://github.com/cl3225/Sobel-et-al-2023-PNAS/blob/main/sobel_etal_2023_code.ipynb

'''
import os
import numpy as np
import xarray as xr
from scipy import stats
import subprocess
#import xesmf as xe
#import gcsfs
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import cartopy as cart
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

years = range(1980, 2016)

obs_datasets = ['HadISST1', 'HighResMIP', 'ESA_CCI']
obs_variable = ['sst', 'sst', 'analysed_sst']
labels = ['(a)', '(b)', '(c)']

current_dir = os.getcwd()
DIR_IN = current_dir
trends = {}
period = str(years[0])+'-'+str(years[-1])
trends['ESA_CCI'] = os.path.join(DIR_IN, 'ESA_trend_'+period+'.nc')
trends['HighResMIP'] = os.path.join(DIR_IN, 'HighResMIP_trend_'+period+'.nc')
trends['HadISST1'] = os.path.join(DIR_IN, 'HadISST_trend_'+period+'.nc')

def linregress_1d(x, y):
        # Wrapper around scipy linregress to use in apply_ufunc
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return np.array([slope, intercept, r_value, p_value, std_err])

def linregress_xr(da,dim='time'):
        NT = len(da[dim])
        da[dim] = np.arange(0,NT)
        stats = xr.apply_ufunc(linregress_1d, da[dim], da,
                               input_core_dims=[[dim], [dim]],
                               output_core_dims=[["parameter"]],
                               vectorize=True,
                               dask="parallelized",
                               output_dtypes=['float32'],
                               output_sizes={"parameter": 5},
        )
        ds = ((NT-1)*stats[:,:,0]).to_dataset(name='trend')
        ds['pvalue'] = stats[:,:,3]
        return ds

def plotMap_myproj(MyFigsize,MyFontsize,londis,latdis,lon1,lon2,lat1,lat2,map_proj=ccrs.PlateCarree(central_longitude=180),ax_list=None,fig_list=None):
        """
        map_proj = ccrs.PlateCarree(central_longitude=180)
        data_crs = ccrs.PlateCarree()
        ax.plot(iblon[:,iSlandfall[0]],iblat[:,iSlandfall[0]],transform=ccrs.PlateCarree())
        note that the logitude and latitude here can only be from -180 to 180
        """

        if ((lon1 >=180) & (lon2 >180)):
                lon1 = lon1-360
                lon2 = lon2-360
        if ax_list is None:
            fig = plt.figure(figsize=MyFigsize)
            ax = fig.add_subplot(
                3, 1, 1,
                projection=map_proj)
        else:
            fig = fig_list
            ax = fig.add_subplot(
                3,1, ax_list,
                projection=map_proj)
            
        ax.coastlines(resolution='110m')
        #gl = ax.gridlines(draw_labels=True)
        gl = ax.gridlines(draw_labels=True,
                linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        #gl.ylabels_right = False
        gl.xlocator = mticker.FixedLocator(np.arange(-180,185,londis))
        gl.ylocator = mticker.FixedLocator(range(-90, 90,latdis))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabels_top = False
        #ax.xaxis.label.set(fontsize=5)
        #ax.yaxis.label.set(fontsize=5)
        ax.add_feature(cart.feature.BORDERS)
        ax.set_extent([lon1,lon2,lat1,lat2],crs = ccrs.PlateCarree())
        return fig,ax

def plot_TP(vmax=1., vmin=-1.):
        alpha = 0.5
        interval = 5
        lon1, lon2 = 10,350
        lat1, lat2 = -60, 60
        xlong = np.arange(lon1,lon2+interval,interval)
        xlat = np.arange(lat1,lat2+interval,interval)
        MyFigsize,MyFontsize = (6,7),8
        londis, latdis = 30, 15
        my_proj = ccrs.PlateCarree(central_longitude=180)
        data_ccrs = ccrs.PlateCarree()

        for ids, dset in enumerate(obs_datasets):
            fignum = labels[ids]
            name = dset
            # read SST trend data from relevant file
            print('ids ',ids, dset)
            ds = xr.open_dataset(trends[dset])
            ds_stats = ds
            if ids == 0:
                ax_list = None
                fig_list = None
            else:
                ax_list = ids+1
                fig_list = fig

            fig,ax = plotMap_myproj(MyFigsize,MyFontsize,londis,latdis,lon1,lon2,lat1,lat2,my_proj,ax_list,fig_list)
            cm = plt.get_cmap("RdYlBu_r")
            an_image = ds.trend.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(),
                        vmin=vmin, vmax=vmax, cmap=cm, add_colorbar=False, rasterized=True);
            cbar = plt.colorbar(an_image, shrink=0.6, pad=.03)
            #cbar.set_label(r'SST trend anomaly ($\degree C$)')
            cbar.ax.tick_params(labelsize=6)

            CS = ds.trend.plot.contour(ax=ax, transform=ccrs.PlateCarree(),vmin=-2,vmax=2,colors='k',levels=21,linewidths=0.8)
            ax.clabel(CS, inline=1, fontsize=8, fmt='%1.1f')
            ax.add_feature(cfeature.LAND, zorder=5, color='lightgrey', edgecolor='black', alpha=1.0)

            #ds.pvalue.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), levels=[alpha,1.0],colors='none',hatches=['..',None],extend='both',add_colorbar=0)
            ax.coastlines(linewidth=2,zorder=6);

            ax.set_extent((120, 290, -30, 30), crs=ccrs.PlateCarree())
            ax.spines['top'].set_visible(False)
            ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=False)
            title = name
            name1 = name.replace(' ','_')
            yval = [0.9, 0.58, 0.28]
            plt.title(f'{fignum} ',fontsize=15,pad=0,loc='left');
        fig.subplots_adjust(bottom=0.05, left=0.1, right=0.98, top=0.95, wspace=0.16, hspace=0.11)
        plt.savefig(f'./SST_trends.pdf',dpi=300)
        plt.savefig(f'./SST_trends.png',dpi=300)
        plt.show()
        return()

def do_trends():
    ### set up data format
    obs_dset = 'ESA' # HadISST, HighResMIP, ESA
    if obs_dset == 'HadISST':
        dataset = 'HadISST_SST_monthly_data_1980_2015'
    elif obs_dset == 'ESA':
        dataset = 'ESA_CCT_SST_monthly_data_1980_2015'
    elif obs_dset == 'HighResMIP':
        dataset = 'HighResMIP_SST_monthly_data_1980_2015'

    if obs_dset == 'HadISST':
        ds_out = xr.Dataset({'lat': (['lat'], np.arange(-30.5,31, 1)),
        'lon': (['lon'], np.arange(120.5,290,1)),})
        encoding={'sst':{'dtype':'float32'},'lon':{'dtype':'float32'},'lat':{'dtype':'float32'}}
    elif obs_dset == 'ESA':
        ds_out = xr.Dataset({'latitude': (['latitude'], np.arange(-30.5,31, 1)),
        'longitude': (['longitude'], np.arange(120.5,290,1)),})
        encoding={'analysed_sst':{'dtype':'float32'},'longitude':{'dtype':'float32'},'latitude':{'dtype':'float32'}}
    elif obs_dset == 'HighResMIP':
        ds_out = xr.Dataset({'latitude': (['latitude'], np.arange(-30.5,31, 1)),
        'longitude': (['longitude'], np.arange(120.5,290,1)),})
        encoding={'sst':{'dtype':'float32'},'longitude':{'dtype':'float32'},'latitude':{'dtype':'float32'}}

    ### These must all be monthly datasets!!
    y_start = 1980
    y_end = 2015
    #obs
    ds = xr.open_dataset(dataset)
    obs_start = np.nanmax([y_start,ds.time[0].dt.year.values.tolist()])
    obs_end = np.min([y_end,ds.time[-1].dt.year.values.tolist()])
    ds = ds.sel(time=slice(str(obs_start)+'-01',str(obs_end)+'-12')).coarsen(time=12,boundary='trim').mean()
    regridder = xe.Regridder(ds, ds_out, 'bilinear', periodic=False)
    dsr = regridder(ds)
    if obs_dset == 'HadISST':
        ds_stats = linregress_xr(dsr.sst)  # ts
    elif obs_dset == 'ESA':
        ds_stats = linregress_xr(dsr.analysed_sst)
    elif obs_dset == 'HighResMIP':
        ds_stats = linregress_xr(dsr.sst)
    ds_stats.to_netcdf(obs_dset+'_trend_'+str(obs_start)+'-'+str(obs_end)+'.nc')
    plot_TP(ds_stats,obs_dset+'' +' '+str(obs_start)+'-'+str(obs_end),'(c)', vmax=1., vmin=-1.)

if __name__ == '__main__':

    plot_TP()
