#!/usr/bin/env python
# coding: utf-8


# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc
import streamlit as st


import warnings
warnings.simplefilter('ignore')

dataset_nhl = pd.read_csv("/Users/polinagalisnikova/Downloads/nhlplayoffs.csv")

st.title("NHL Stanley Cup Playoffs (1918 - 2022)")
st.markdown("For my project I decided to choose the **NHL Stanley Cup Playoffs (1918 - 2022)** dataset. It contains information regarding the games, their results and generally statistics in certain NHL seasons. I am going to give an overview and analyze data further.")

# # Dataset Cleanup

# In[2]:


dataset_nhl.info()


# In[3]:


print(f'Shape of the DF is {dataset_nhl.shape[0]} rows by {dataset_nhl.shape[1]} columns.')


# As it can be seen above, all types of data in the table are correct, the number of null cells equals zero, so no changes are needed. The size of DataFrame is 1009 rows by 13 columns. Nevertheless, looking through the dataset I discovered that some teams' names have different versions in different years. This happened because of their possible renaming over the time. In order to avoid mistakes in statistics, I decided to change the ones that I noticed to the present variant.

# In[4]:


dataset_nhl.loc[dataset_nhl['team'].str.contains('Toronto'), 'team'] = 'Toronto Maple Leafs'
dataset_nhl.loc[dataset_nhl['team'].str.contains('Chicago'), 'team'] = 'Chicago Blackhawks'
dataset_nhl.loc[dataset_nhl['team'].str.contains('Detroit'), 'team'] = 'Detroit Red Wings'
dataset_nhl.loc[dataset_nhl['team'].str.contains('Ahaheim'), 'team'] = 'Anaheim Ducks'


# I also decided to modify columns' names to make the table look better.

# In[5]:


dataset_nhl.rename(columns = {'rank':'Rank', 'team':'Team', 'year':'Year', 'games':'Games', 'wins':'Wins', 'losses':'Losses',
                             'ties':'Ties', 'shootout_wins':'Shootout wins', 'shootout_losses':'Shootout losses',
                             'win_loss_percentage':'Win-loss percentage', 'goals_scored':'Goals scored', 'goals_against':'Goals against',
                             'goal_differential':'Goal differential'}, inplace=True)


# # Data transformation

# In the process of analysis I came to the conclusion that some additional information may be useful for the forthcoming description. Owing to this fact, I added 3 more columns to this DataFrame. 
# 1. The first one is **'Playoff appearances'** - the number of teams' appearances in playoffs over the years. 
# 2. The second one is **'Stanley Cups'** - the number of Stanley Cups that the team have won, if any. Basically, it is the amount of times when the team's rank equals 1. 
# 3. The last column is **'Average goals per game'** - it is average number of goals that each team scores per 1 game.

# In[6]:


teams_appearances = {}
for team in dataset_nhl['Team'].unique():
    if team not in teams_appearances:
        teams_appearances[team] = dataset_nhl[dataset_nhl['Team']== team].shape[0]
dataset_nhl['Playoff appearances'] = dataset_nhl['Team'].map(teams_appearances)

stanley_cups = {}
for team in dataset_nhl['Team'].unique():
    stanley_cups[team] = dataset_nhl[(dataset_nhl['Team'] == team) & (dataset_nhl['Rank'] == 1)].shape[0]
dataset_nhl['Stanley Cups'] = dataset_nhl['Team'].map(stanley_cups)
dataset_nhl.head(5)

goals_per_game = {}
for team in dataset_nhl['Team'].unique():
    goals_per_game[team] = (dataset_nhl[dataset_nhl['Team']==team]['Goals scored'].sum()) / (dataset_nhl[dataset_nhl['Team']==team]['Games'].sum())
dataset_nhl['Average goals per game'] = dataset_nhl['Team'].map(goals_per_game)


# # Overview 

# **First of all,** let's look at the DataFrame and understand the things that it consists of. 
# 1. The first column is **'Rank'** - it shows the place where the particular team ended the season on. It also may be useful in order to learn in what part of playoff team lost. For example, if Edmonton Oilers' rank is 4, we understand, that they lost in Conference Final. 
# 2. The next column is **'Team'** - it is basically just name of a team, with the help of which its origin may be learned. 
# 3. The third column is **'Year'**. It contains the year of a certain playoff. 
# 4. The column **'Games'** represents the sum of the next two columns - **'Wins'** and **'Losses'**. 
# 5. **'Ties'** shows whether the regulation time of a game ended with equal score and how many times it happened. 
# 6. There are also two columns **'Shootout wins'** and **'Shootout losses'**, where we can see the results of games, where one of the teams didn't score at all. 
# 7. **'Win-loss percentage'** column describes the ratio of games' results. 
# 8. The columns **'Goals scored'**, **'Goals against'** and **'Goals differential'** shows the number of goals scored by team, goals missed by team and their difference. 
# 9. Columns **'Playoff apearances'**, **'Stanley Cups'** and **'Average goals per game'** were already described in Data Transformation above. 

# In[7]:


dataset_nhl.sample(10)


# Some descriptive statistics is presented in a DataFrame below. There are several columns that may be interesting for us, for example, 'Games' - the average number of games in playoff is about 9. Other numerical fields such as 'Wins', 'Goals scored', etc. provide useful information as well.  

# In[8]:


dataset_nhl.describe()


# Let's look at some basic statistics of this dataset. I would like to output a hystogram of 10 teams in NHL, whose appearances in playoff were the most frequent. 

# In[9]:


teams_appearances_plot = {}
for team in dataset_nhl['Team'].unique():
    if team not in teams_appearances_plot:
        teams_appearances_plot[team] = dataset_nhl[dataset_nhl['Team']== team].shape[0]
        
app = pd.DataFrame(data = [teams_appearances_plot], index = ['Playoff appearances'])
app = app.transpose()
app.sort_values(by='Playoff appearances', ascending = False, inplace = True)

sns.set_style('darkgrid')
current_palette = sns.color_palette('husl', 15)
plt.figure(figsize = (19,6))
sns.barplot(x = app.index[:10], y = app['Playoff appearances'][:10], data = app, palette = current_palette);
plt.title('NHL Teams Playoff Appearances', fontsize = 20);


# We can see that Montreal Canadiens appeared in playoffs more than 90 times. At the same time appearances of Detroit Red Wings and Chicago Blackhawks practically the same. In the top-10 Washington Capitals have the lowest rate. 

# Then, as my favourite team in NHL is New York Rangers, I wanted to know the precise number of its appearances in playoffs.

# In[10]:


print("New York Rangers appeared in playoffs {} times in the period from 1918 to 2022.".format(
sum(dataset_nhl['Team'] == 'New York Rangers')))


# Next, let's see which teams were the most successful over the years and who won the largest number of cups. 

# In[11]:


leaders = {}
for team in dataset_nhl['Team'].unique():
    if team not in leaders:
        leaders[team] = dataset_nhl[(dataset_nhl['Team'] == team)
                                    & (dataset_nhl['Rank'] == 1)].shape[0]
cups = pd.DataFrame(data = [leaders], index = ['Stanley Cups'])
cups = cups.transpose()
cups.sort_values(by='Stanley Cups', ascending = False, inplace = True)

sns.set_style('darkgrid')
colour = ['#0000CD' if (x == 'Montreal Canadiens') else '#6495ED' for x in cups]
plt.figure(figsize = (19,6))
plt.title('Leaders in NHL by the number of Stanley Cups', fontsize = 20);
sns.barplot(x = cups.index[:10], y = cups['Stanley Cups'][:10], data = cups, palette = colour);


# Wow! Montreal Canadiens are undisputed leaders among all of the NHL teams. 

# Tie - is a game ending with each team having the same score. Let's see how many ties were in NHL playoffs over the years. 

# In[12]:


print('Over the years there were only {} ties in NHL playoffs.'.format(dataset_nhl['Ties'].sum()))


# Let's analyze some statictics for the previous NHL season - season 2021/22. I would like to see the diagram that represents goals scored by participants in that season's playoff.

# In[13]:


goals_sc = {}
for year in dataset_nhl['Year'].unique():
    results_2022 = dataset_nhl[dataset_nhl['Year'] == 2022]

for team in results_2022['Team'].unique():
    for goal in results_2022['Goals scored'].unique():
        goals_sc[team] = results_2022[(results_2022['Team'] == team)]['Goals scored'].sum()

goals = pd.DataFrame(data = [goals_sc], index = ['Goals scored'])
goals = goals.transpose()

current_palette = sns.color_palette('deep')
sns.set_style('darkgrid')
radius = 2.1
plt.title('Goals scored by NHL teams in 2022', fontsize = 20, x = 0.6, y = 1.4)
plt.pie(x=goals['Goals scored'], explode=[0.05]*16, pctdistance=0.5, radius = radius, autopct='%.2f', 
        colors = current_palette, textprops={'color':"w", 'fontsize':10})
plt.legend(labels=goals.index, loc='upper left', 
           bbox_to_anchor=(1.7, 1.4), ncol=1, fontsize = 14, title = 'Team');


# As it can be seen, Colorado Avelanche not only won the Stanley Cup, but also scored the largest number of goals. 

# # More Detailed Overview

# We know that the league has been developing over the years and number of its participants has been increasing. Let's see the trend for games played in playoffs per year in certain periods. I defined playoffs from 1918 to 1948 as the first period - past, and from 1988 to 2022 as the second period - present.

# In[14]:


first_period = dataset_nhl[dataset_nhl['Year'] < 1948]
games_f = pd.DataFrame(first_period.groupby(['Year'])['Games'].sum() // 2)
second_period = dataset_nhl[dataset_nhl['Year'] >= 1988]
games_t = pd.DataFrame(second_period.groupby(['Year'])['Games'].sum() // 2)

f, ax = plt.subplots(figsize=(31, 13))
sns.barplot(x=games_t.index, y='Games', data=games_t,
            label="Total", color="#B0C4DE");
sns.barplot(x=games_f.index, y='Games', data=games_f,
            label="Total", color="#4682B4");

sns.set_style('darkgrid')
plt.xticks([])
plt.title('Compairson between games played in NHL playoffs', fontsize = 25);
plt.ylabel('Games', fontsize = 20);
plt.xlabel('Period', fontsize = 20);


# In the barchart above the first period represented in dark blue and the second one is light blue. As we can see, there is a significant difference in the number of games played in them. If in the past the number of games did not exceed 40 per year, nowadays this number is slightly below 100. There is also no particular trend - the data flactuates in both periods, so it is impossible to predict something for the fortcoming playoffs. 

# Then, to get more detailed overview, I wanted to see a compairson of goals for and goals against for the first 10 teams in playoff 2022. 

# In[15]:


for year in dataset_nhl['Year'].unique():
    results_2022 = dataset_nhl[dataset_nhl['Year'] == 2022]
    
goals_df = results_2022.sort_values('Goals scored', ascending=False)[:10].reset_index().melt(id_vars=['Team'], value_vars = ['Goals scored', 'Goals against'])
plt.figure(figsize = (19,6))
current_palette = sns.color_palette("hls")
sns.barplot(data=goals_df, x='Team', y='value', hue='variable', palette = current_palette);
sns.set_style('darkgrid')
plt.ylabel('Scored / against', fontsize = 16)
plt.xlabel('Team', fontsize = 16)
plt.legend(title='Type of goals')
plt.title('Goals scored / against in 2022', fontsize = 20);


# It is curious, that although the number of goals for is smaller than goals against for Carolina Hurricanes, the team's rank is higer that Pittsburgh Penguins, who has the opposite situation. 

# Next, I would like to see how the number of teams' wins is changing in 10 year period. For this reason I decided to build this particular line graph.

# In[22]:


canada = dataset_nhl[dataset_nhl['Team'].str.contains('Toronto') | dataset_nhl['Team'].str.contains('Calgary')
                     | dataset_nhl['Team'].str.contains('Montreal') | dataset_nhl['Team'].str.contains('Vancouver')
                    | dataset_nhl['Team'].str.contains('Edmonton') | dataset_nhl['Team'].str.contains('Ottawa')
                    | dataset_nhl['Team'].str.contains('Winnipeg')]
usa_eastern = dataset_nhl[dataset_nhl['Team'].str.contains('Carolina') | dataset_nhl['Team'].str.contains('Columbus')
                     | dataset_nhl['Team'].str.contains('New') | dataset_nhl['Team'].str.contains('Philadelphia')
                    | dataset_nhl['Team'].str.contains('Pittsburgh') | dataset_nhl['Team'].str.contains('Washington')
                    | dataset_nhl['Team'].str.contains('Buffalo') | dataset_nhl['Team'].str.contains('Detroit')
                 | dataset_nhl['Team'].str.contains('Florida') | dataset_nhl['Team'].str.contains('Tampa')
                 | dataset_nhl['Team'].str.contains('Anaheim')]
usa_western = dataset_nhl[dataset_nhl['Team'].str.contains('Arizona')
                 | dataset_nhl['Team'].str.contains('Chicago') | dataset_nhl['Team'].str.contains('Colorado') 
                | dataset_nhl['Team'].str.contains('Dallas') | dataset_nhl['Team'].str.contains('Minnesota') 
                | dataset_nhl['Team'].str.contains('Nashville') | dataset_nhl['Team'].str.contains('St.')
                | dataset_nhl['Team'].str.contains('Los') | dataset_nhl['Team'].str.contains('San Jose')
                 | dataset_nhl['Team'].str.contains('Seattle') | dataset_nhl['Team'].str.contains('Vegas')]

ten_years_canada = canada[canada['Year'] > 2012]
ten_years_usa_east = usa_eastern[usa_eastern['Year'] > 2012]
ten_years_usa_west = usa_western[usa_western['Year'] > 2012]

sns.relplot(x='Year',
            y ='Wins',
            hue='Team',
            kind='line',
            data=ten_years_canada, height=8.27, aspect=18.7/8.27);
sns.set_style('darkgrid')
plt.title("The NHL Canadian teams' wins in ten year period from 2012 to 2022", fontsize=25);
plt.xlabel('Year', fontsize=16);
plt.ylabel('Wins', fontsize=16);

sns.relplot(x='Year',
            y ='Wins',
            hue='Team',
            kind='line',
            data=ten_years_usa_east, height=8.27, aspect=18.7/8.27);
sns.set_style('darkgrid')
plt.title("The NHL Eastern Conference teams' wins in ten year period from 2012 to 2022", fontsize=25);
plt.xlabel('Year', fontsize=16);
plt.ylabel('Wins', fontsize=16);

sns.relplot(x='Year',
            y ='Wins',
            hue='Team',
            kind='line',
            data=ten_years_usa_west, height=8.27, aspect=18.7/8.27);
sns.set_style('darkgrid')
plt.title("The NHL Western Conference teams' wins in ten year period from 2012 to 2022", fontsize=25);
plt.xlabel('Year', fontsize=16);
plt.ylabel('Wins', fontsize=16);


# As we can see from the data, the highest number of wins have the teams in NHL Eastern Conference. It's slightly above 17.5. Let's look at each line graph individually. 
# 1. **The NHL Canadian teams:** there are 7 out of 8 Canadian teams, which made it to the playoff in the period in question. We can see that the highest point have Montreal Canadiens - their wins number above 12. There are also a lot of ragged lines - it means that a team did not participate in playoff in some year. 
# 2. **The NHL Eastern Conference American teams:** the highest rate here have Tampa Bay Lightning - in 2020 they won the Stanley Cup. The lowest results have Florida Panthers. We can also see a plateau here - for Pittsburgh Penguins the number of wins was the same in 2016 and 2017. 
# 3. **The NHL Western Conference American teams:** several teams here have a leadership by the number of wins - St. Louis Blues, Chicago Blackhawks and Los Angeles Kings. At the same time we can see that the results of LA Kings dropped significantly and they have not been in playoff since 2020. The only team in this category that consistently participate in playoff is Minnesota Wils, although their results are pretty low. 

# Now I want to compare results of New York Rangers and Washington Capitals by the following parameters: Goals for and against, Wins and losses, Shootout wins and shootout losses.  

# In[17]:


for team in dataset_nhl['Team'].unique():
    compair = dataset_nhl[(dataset_nhl['Team'] == 'New York Rangers') | (dataset_nhl['Team'] == 'Washington Capitals')]

current_palette = sns.color_palette("hls")
sns.set_style('darkgrid')
fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharey=True)
sns.boxplot(ax=axes[0], x='Team', y='Goals scored', data=compair, palette = current_palette, width=0.5);
sns.boxplot(ax=axes[1], x='Team', y='Goals against', data=compair, palette = current_palette, width=0.5);

fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
sns.boxplot(ax=axes[0,0], x='Team', y='Wins', data=compair, palette = current_palette, width=0.5);
sns.boxplot(ax=axes[1,0], x='Team', y='Losses', data=compair, palette = current_palette, width=0.5);
sns.boxplot(ax=axes[0,1], x='Team', y='Shootout wins', data=compair, palette = current_palette, width=0.5);
sns.boxplot(ax=axes[1,1], x='Team', y='Shootout losses', data=compair, palette = current_palette, width=0.5);


# 1. **Goals:** overall, the mean number of goals scored is higher for Washington Capitals, but New York Rangers' upper bound is higher. In goals against the mean value is again higher for Washington Capitals. They also have some outliers between 40 and 50 and slightly above 60. 
# 2. **Wins and losses:** Wins mean value is practically the same for both teams. Outliers are also on the same level, but interquartile range is bigger for New York Rangers. Standart deviation is bigger for New York Rangers in losses boxplot.
# 3. **Shootout results**: as for the shootout results, their number differs insignificantly in wins, but there are more losses for Washington Capitals. 

# # Hypothesis

# For the next stage of my analysis I am going to make some statements and then check their validity. 

# #### 1. The number of ties in NHL playoffs from 1918 to 2022 makes up less that 1 per cent of all games in this period. 

# It is known that ties do not happen frequently in hockey games, especially in playoffs. 

# In[18]:


ties_percent = round(dataset_nhl['Ties'].sum() / dataset_nhl['Games'].sum() * 100, 2)
print(f'A number of ties in NHL playoffs from 1918 to 2022 makes up {ties_percent} per cent of all games in this period.')


# We can see that our hypothesis was true. A number of ties in NHL playoffs indeed does not exceed 1 per cent over the whole dataset period. 

# #### 2. If a team scores the lowest number of goals in playoff, it has the lowest rank. 

# Let's look at playoff results in 2022. I am going to sort the dataset by teams' goals scored and then print the lowest possible rank in that playoff. 

# In[19]:


for year in dataset_nhl['Year'].unique():
    results_2022 = dataset_nhl[dataset_nhl['Year'] == 2022]
    
results_2022.sort_values(by='Goals scored', ascending=True, inplace=True)
print('Team with the lowest number of goals scored in 2022: {}'.format(results_2022['Team'].iloc[0]))
print('Rank of this team: {}'.format(results_2022['Rank'].iloc[0]))
print('The lowest possible rank in 2022 playoff: {}'.format(max(results_2022['Rank'])))

for year in dataset_nhl['Year'].unique():
    results_2022 = dataset_nhl[dataset_nhl['Year'] == 2021]
    
results_2022.sort_values(by='Goals scored', ascending=True, inplace=True)
print('Team with the lowest number of goals scored in 2021: {}'.format(results_2022['Team'].iloc[0]))
print('Rank of this team: {}'.format(results_2022['Rank'].iloc[0]))
print('The lowest possible rank in 2021 playoff: {}'.format(max(results_2022['Rank'])))


# As it is seen, the team that scored the least number of goals in 2022 scored the Nashville Predators with the rank 16. The last possible rank is 16, so for 2022 playoff my hypothesis is true. In 2021 playoff the team that scored less than others is St. Louis Blues with the rank 15, so in this case the hypothesis was incorrect. 

# #### 3. if a team appeared in NHL playoffs more frequently than others, it has more Stanley Cups than any other team. 

# In[20]:


teams_appearances_plot = {}
for team in dataset_nhl['Team'].unique():
    if team not in teams_appearances_plot:
        teams_appearances_plot[team] = dataset_nhl[dataset_nhl['Team']== team].shape[0]
        
app = pd.DataFrame(data = [teams_appearances_plot], index = ['Playoff appearances'])
app = app.transpose()
app.sort_values(by='Playoff appearances', ascending = False, inplace = True)

leaders = {}
for team in dataset_nhl['Team'].unique():
    if team not in leaders:
        leaders[team] = dataset_nhl[(dataset_nhl['Team'] == team)
                                    & (dataset_nhl['Rank'] == 1)].shape[0]
cups = pd.DataFrame(data = [leaders], index = ['Stanley Cups'])
cups = cups.transpose()
cups.sort_values(by='Stanley Cups', ascending = False, inplace = True)

sns.set_style('darkgrid')
plt.figure(figsize = (19,6))
sns.barplot(x = app.index[:10], y = app['Playoff appearances'][:10], data = app, color = '#DB7093');
sns.barplot(x = cups.index[:10], y = cups['Stanley Cups'][:10], data = cups, color = '#4682B4');
plt.title('Compairson between playoff appearances and number of Stanley Cups', fontsize = 20);
plt.ylabel(None);
plt.xlabel('Team');


# On this barchart playoff appearances are represented in pink and Stanley Cups are shown in blue. 
# It cannot be said for sure from this statictics if the number of playoff apearances directly affects the number of wins. It still depends on many factors. For example, the difference between Pittsburgh Penguins and New York Rangers appearances is huge, but their number of Stanley Cups in equal. Therefore, my hypothesis cannot be defined as correct or incorrect. 

# #### 4. If a team is from Canada, there is a higher possibility that it will win the Stanley Cup.

# In[21]:


canada = dataset_nhl[dataset_nhl['Team'].str.contains('Toronto') | dataset_nhl['Team'].str.contains('Calgary')
                     | dataset_nhl['Team'].str.contains('Montreal') | dataset_nhl['Team'].str.contains('Vancouver')
                    | dataset_nhl['Team'].str.contains('Edmonton') | dataset_nhl['Team'].str.contains('Ottawa')
                    | dataset_nhl['Team'].str.contains('Winnipeg')]
canada_cups = {}
for team in canada['Team'].unique():
    canada_cups[team] = canada[(canada['Team'] == team) & (canada['Rank'] == 1)].shape[0]
print(f'Total number of cups won by Canadian teams: {sum(canada_cups.values())}')
print(f'Number of teams from Canada: {len(canada_cups)}')
print('Total number of seasons: 104')
print('Ratio of Stanley Cups for Canadian teams is {}%'.format(
    round(sum(canada_cups.values()) / 104 * 100, 2)))


# The number of cups won by Canadian teams is slightly less than 50 per cent. However, if we take into account that there are only 8 of them, this is an impressive result and it's ratio is definitely higher than for American teams.
