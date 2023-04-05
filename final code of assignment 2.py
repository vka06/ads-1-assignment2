# -*- coding: utf-8 -*-
"""
Created on Wed Mar  29 16:08:01 2023

@author: karan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def read_df(filepath):
  
    df = pd.read_csv(filepath, skiprows=(4))

    return df


def transposing_dfs(df):

    df_tr = pd.DataFrame.transpose(df)
    df_tr.columns = df['Country Name']
    df_tr = df_tr.iloc[1:, :]
    df.index = df['Country Name']

    return df, df_tr



def bar_plotting(data, y_label, title):

    fig = plt.figure()
    width = 0.8/len(data.columns)
    offset = width/2
    ax = plt.subplot()
    for index, year in enumerate(data.columns):
        ax.bar([ln+offset+width*index for ln in range(len(data.index))],
               data[year], width=width, label=year)
    ax.set_xticks([j+0.4 for j in range(len(data.index))])
    ax.set_xticklabels(data.index, rotation=90)

    ax.set_xlabel('Country Name')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(title='Years', bbox_to_anchor=(1, 1))
    plt.savefig(title+'.png', bbox_inches='tight', dpi=400)
    plt.show()

    return fig


def country(my_dfs, labels, country_name):

    country = pd.DataFrame()
    for i in range(len(my_dfs)):
        country[labels[i]] = my_dfs[i].loc['1990':'2019', country_name]

    return country


countries = ['Australia', 'Bangladesh','Brazil', 'Canada', 'China','France',
             'India', 'Japan', 'Mexico','United Kingdom','United States']
print('Countries for which analysis is proceeded:')
print(countries, '\n')



crop_land = read_df('Permanent cropland (% of land area).csv')
crop_land, crop_land_tr = transposing_dfs(crop_land)

crop = crop_land_tr.loc['1990':'2020', countries]


plt.figure()
for i in crop.columns:
    plt.plot(crop.index, crop[i]/1000000, label=i, linestyle='--')
plt.legend(title='Countries', bbox_to_anchor=(1, 1))
plt.xlabel('Year')
plt.ylabel('values in % ')
plt.title('Permanent cropland (% of land area)')
plt.xticks(crop.index[::3])
plt.show()


agri_land = read_df('Agricultural land (% of land area).csv')
agri_land, agri_land_tr = transposing_dfs(agri_land)

agri = agri_land_tr.loc['1990':'2020', countries]


plt.figure()
for i in agri.columns:
    plt.plot(agri.index, agri[i]/1000000, label=i, linestyle='--')
plt.legend(title='Countries', bbox_to_anchor=(1, 1))
plt.xlabel('Year')
plt.ylabel('values in % ')
plt.title('Agricultural land (% of land area)')
plt.xticks(agri.index[::3])
plt.show()



agri_GDP = read_df('Agriculture value added (% of GDP).csv')
agri_GDP, agri_GDP_tr = transposing_dfs(agri_GDP)

agriGDP = agri_GDP_tr.loc['1990':'2020', countries]



plt.figure()
for i in agriGDP.columns:
    plt.plot(agriGDP.index, agriGDP[i]/1000000, label=i, linestyle='--')
plt.legend(title='Countries', bbox_to_anchor=(1, 1))
plt.xlabel('Year')
plt.ylabel('values in % ')
plt.title('Agriculture,forestry,&fishing,value added(%ofGDP)')
plt.xticks(agriGDP.index[::3])
plt.show()



rural_pop = read_df('Rural population.csv')
rural_pop, rural_pop_tr = transposing_dfs(rural_pop)

years = ['1995','2000', '2005', '2010', '2015', '2019']
rural = rural_pop.loc[countries, years]
print('Rural population data description of years of few countries:')
print(rural.describe())

for i in rural.columns:
    urban[i] = rural[i]/1000000



bar_plotting(rural, 'Rural Population in  Millions', 'RURAL POPULATION')

#Reading Arable land  data and creating its transpose
forest_area = read_df('Forest area (sq. km).csv')
forest_area, forest_area_tr = transposing_dfs(forest_area)

forest = forest_area.loc[countries, years]
print('Forest area (sq. km) data description of years of few countries:')
print(forest.describe(), '\n')

bar_plotting(forest, 'Forest Area in M (sq.km)', 'Forest area (sq. km)')


def correlation_heatmap(country_data, country, color):
    

    for i in country_data.columns:
        country_data[i] = country_data[i].astype(dtype=np.int64)
    corr = country_data.corr().to_numpy()

    fig = plt.subplots(figsize=(8, 8))
    plt.imshow(corr, cmap=color, interpolation='nearest')
    plt.colorbar(orientation='vertical', fraction=0.05)

    plt.xticks(range(len(country_data.columns)),
               country_data.columns, rotation=45, fontsize=15)
    plt.yticks(range(len(country_data.columns)),
               country_data.columns, rotation=0, fontsize=15)

    for i in range(len(country_data.columns)):
        for j in range(len(country_data.columns)):
            plt.text(i, j, corr[i, j].round(2),
                     ha="center", va="center", color='black')
    plt.title(country)
    plt.savefig(country+'.png', bbox_inches='tight', dpi=300)
    plt.show()

    return fig



indicators = ['Agriculture GDP','Forest Area', 'Urban Population', 
              'Agriculture Land', 'Permanent cropland']

dataframes = [agri_GDP,forest_area, urban_pop, agri_land, crop_land]
dataframes_tr = [forest_area_tr, urban_pop_tr, agri_land_tr, crop_land_tr]



india = country(dataframes_tr, indicators, 'India')
correlation_heatmap(india, 'INDIA', 'PiYG')

china = country(dataframes_tr, indicators, 'China')
correlation_heatmap(china, 'CHINA', 'rainbow')