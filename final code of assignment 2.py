# -*- coding: utf-8 -*-
"""
Created on Wed Mar  29 16:08:01 2023

@author: karan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

#read_df function to read csv files
def read_df(filepath):
    
    '''
    Creates DataFrame of given filepath

    Parameters
    ----------
    filepath : STR
        File path of our csv file.

    Returns
    -------
    df : DataFrame
        Dataframe created with given csv file.

    '''

  
    df = pd.read_csv(filepath, skiprows = (4))
    
    df.head()

    return df

#transposing_dfs to create transpose of given data
def transposing_dfs(df):
    
    '''
   Creates transpose dataframe of given dataframe

   Parameters
   ----------
   df : DataFrame
       Dataframe for which transpose to be found.

   Returns
   -------
   df : DataFrame
       Given dataframe.
   df_tr : TYPE
       Transpose of given dataframe.

   '''

    df_tr = pd.DataFrame.transpose(df)
    df_tr.columns = df['Country Name']
    df_tr = df_tr.iloc[1:, :]
    df.index = df['Country Name']

    return df, df_tr


#bar_plotting to plot data over multiple columns as bars
def bar_plotting(data, y_label, title):

    '''
    Plots a barplot of data over multiple columns

    Parameters
    ----------
    data : DataFrame
        Dataframe for which barplot to be plotted.
    y_label : STR
        Plot y-axis label as string.
    title : STR
        Plot title as string.

    Returns
    -------
    fig : Figure
        Plot saved as fig.
        
    '''

    fig = plt.figure()
    width = 0.8/len(data.columns)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'black']
    offset = width/2
    ax = plt.subplot()
    for index, year in enumerate(data.columns):
        ax.bar([ln+offset+width*index for ln in range(len(data.index))],
               data[year], width=width, label=year, color=colors[index])
    ax.set_xticks([j+0.4 for j in range(len(data.index))])
    ax.set_xticklabels(data.index, rotation=90)

    ax.set_xlabel('Country Name')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(title='Years', bbox_to_anchor=(1, 1))
    plt.savefig(title+'.png', bbox_inches='tight', dpi=400)
    plt.show()

    return fig

#country to create dataframe of a specific country
def country(my_dfs, labels, country_name):
    
    '''
   Creates new dataframe where all data from my_dataframes of my country 
   will be in it from index 1990 to 2019

   Parameters
   ----------
   my_dfs : List
       List of dataframes.
   labels : List
       List of string values used as column names of my new dataframe.
   country_name : STR
       Country fow which data to be extracted.

   Returns
   -------
   country : DataFrame
       Dataframe with my country data from all given dataframes from year 
       1990 to 2019.

    '''
    
    country = pd.DataFrame()
    for i in range(len(my_dfs)):
        country[labels[i]] = my_dfs[i].loc['1990':'2019', country_name]

    return country

# choosed some countries am intrested
countries = ['Australia', 'Bangladesh','Brazil', 'Canada', 'China','France',
             'India', 'Japan', 'Mexico','United Kingdom','United States']
print('Countries for which analysis is proceeded:')
print(countries, '\n')


#Reading permanent cropland data and creating its transpose
crop_land = read_df('Permanent cropland (% of land area).csv')
crop_land, crop_land_tr = transposing_dfs(crop_land)

#Slicing data to limit data to my countries
crop = crop_land_tr.loc['1990':'2020', countries]

#Plotting permanent cropland area variation of my countries
plt.figure()

# divided by a million to get data in millions
for i in crop.columns:
    plt.plot(crop.index, crop[i]/1000000, label=i, linestyle='--')
plt.legend(title='Countries', bbox_to_anchor=(1, 1))
plt.xlabel('Year')
plt.ylabel('values in % ')
plt.title('Permanent cropland (% of land area)')
plt.xticks(crop.index[::3])
plt.show()

#Reading agriculture land data and creating its transpose
agri_land = read_df('Agricultural land (% of land area).csv')
agri_land, agri_land_tr = transposing_dfs(agri_land)

#Slicing data to limit data to my countries
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


#Reading agri value added to % GDP land and creating its transpose
agri_GDP = read_df('Agriculture value added (% of GDP).csv')
agri_GDP, agri_GDP_tr = transposing_dfs(agri_GDP)

#Slicing data to limit data to my countries
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


years = ['1995','2000', '2005', '2010', '2015', '2019']

#Reading Rural Population data and creating its transpose
rural_pop = read_df('Rural population.csv')
rural_pop, rural_pop_tr = transposing_dfs(rural_pop)

rural = rural_pop.loc[countries, years]

print('Rural population data description of years of few countries:')
print(rural.describe())


#Values are plotted in Millions
for i in rural.columns:
    rural[i] = rural[i]/1000000


#Plotting Rural Population data variation of my countries in given years
bar_plotting(rural, 'Rural Population in  Millions', 'RURAL POPULATION')

#Reading Forest Area data and creating its transpose
forest_area = read_df('Forest area (sq. km).csv')
forest_area, forest_area_tr = transposing_dfs(forest_area)

forest = forest_area.loc[countries, years]
print('Forest area (sq. km) data description of years of few countries:')
print(forest.describe(), '\n')


#Plotting Forest Area data variation of my countries in given years
bar_plotting(forest, 'Forest Area in M (sq.km)', 'Forest area (sq. km)')

#correlation_heatmap to produce correlation heatmap
def correlation_heatmap(country_data, country, color):
    
    '''
  Plots a heatmap of given data correlation of its columns

  Parameters
  ----------
  country_data : DataFrame
       Dataframe from which heatmap is produced.
  country : STR
      Country name as string.
  color : STR
      cmap value as sring.

  Returns
  -------
  fig : Figure
      Plot saved as figure.

  '''
    

    for i in country_data.columns:
        country_data[i] = country_data[i].astype(dtype=np.int64)
    corr = country_data.corr().to_numpy()

    fig = plt.subplots(figsize=(8, 8))
    plt.imshow(corr, cmap=color, interpolation='nearest')
    plt.colorbar(orientation='vertical', fraction=0.05)
    
    #To show ticks and label them with appropriate names of columns
    plt.xticks(range(len(country_data.columns)),
               country_data.columns, rotation=45, fontsize=15)
    plt.yticks(range(len(country_data.columns)),
               country_data.columns, rotation=0, fontsize=15)
    
    #To create text annotations and display correlation coefficient in plot
    for i in range(len(country_data.columns)):
        for j in range(len(country_data.columns)):
            plt.text(i, j, corr[i, j].round(2),
                     ha="center", va="center", color='black')
    plt.title(country)
    plt.savefig(country+'.png', bbox_inches='tight', dpi=300)
    plt.show()

    return fig


# indicators names lists and its dataframes to create country
#specific data for further analysis
indicators = ['Agriculture GDP','Forest Area', 'Rural Population', 
              'Agriculture Land', 'Permanent cropland']

dataframes = [agri_GDP,forest_area, rural_pop, agri_land, crop_land]
dataframes_tr = [forest_area_tr, rural_pop_tr, agri_land_tr, crop_land_tr]


#Creating India dataframe and plotting its heatmap
india = country(dataframes_tr, indicators, 'India')
correlation_heatmap(india, 'INDIA', 'PiYG')


#Creating China dataframe and plotting its heatmap
china = country(dataframes_tr, indicators, 'China')
correlation_heatmap(china, 'CHINA', 'rainbow')



#used datatframe method groupby()

forgroupby = read_df('Fertilizer consumption.csv')

new_df = forgroupby[['Country Name', 'Indicator Name', '2010', '2011', '2012', 
                     '2013', '2014', '2015']]

#by country name, calculate the mean and mediab of each year's values
grouped_by_countries_mean = new_df.groupby('Country Name').mean()
grouped_by_countries_median = new_df.groupby('Country Name').median()

# resulting DataFrame
print(grouped_by_countries_mean)
print(grouped_by_countries_median)



#for statistical functions

#finding skewness
skewness=[]
kurtness=[]
for i in agri_GDP.columns:
    skewness.append(agri_GDP[i].skew())
    kurtness.append(agri_GDP[i].kurtosis())
    
agri_GDP_sk_kurt = pd.DataFrame()
agri_GDP_sk_kurt.index = agri_GDP.columns
agri_GDP_sk_kurt['Skewness'] = skwness
agri_GDP_sk_kurt['Skewness'] = kurtness

print(agri_GDP_sk_kurt)
