import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.request import urlopen
from bs4 import BeautifulSoup
import os
import difflib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

os.chdir('/Users/jcrusor/OneDrive/Data Competitions/MM2017/')

#os.getcwd()


#load data 
regseascmpt = pd.read_csv('RegularSeasonCompactResults.csv')
regseasdtl = pd.read_csv('RegularSeasonDetailedResults.csv')
seasons = pd.read_csv('Seasons.csv')
teams = pd.read_csv('Teams.csv')
tourneyrsltcmpt = pd.read_csv('TourneyCompactResults.csv')
tourneyrsltdtl = pd.read_csv('TourneyDetailedResults.csv')
tourneyseeds = pd.read_csv('TourneySeeds.csv')
tourneyslots = pd.read_csv('TourneySlots.csv')
samplesubmission = pd.read_csv('samplesubmission.csv')
teamnamemapping = pd.read_csv('TeamNameMapping.csv')
#teamnamemapping.columns.values[0]='ESPN_Name'

#####Import ESPN RPI Data

rpi_data = pd.DataFrame()

url1 = 'http://www.espn.com/mens-college-basketball/bpi/_/season/'
season = ['2008','2009','2010','2011','2012','2013','2014','2015','2016','2017']
url2 = '/page/'
maxpgno = ['14','14','14','14','14','14','15','15','15','15']
url3 = '/view/overview'
cnt = 0

for seas in season:
    for i in range(1,int(maxpgno[cnt])+1):
        #print(seas+str(i))
        url = url1+seas+url2+str(i)+url3
        
        soup = BeautifulSoup(urlopen(url).read(),"lxml")
        
        for row in soup('table', {'class': 'bpi__table'})[0].tbody('tr'):
            tds = row('td')
            
            soup2 = BeautifulSoup(str(tds),"lxml")
            
            rk = tds[0].text
            team = soup2.findAll('span')[1].next
            conf = tds[2].text
            wl = tds[3].text
            bpi_rk = tds[4].text
            sos_rk = tds[5].text
            sor_rk = tds[6].text 
            rpi_rk = tds[7].text
            season = seas
            
            df = pd.DataFrame({'rank':[rk],'team':[team], 'conference':[conf], 'winloss':[wl], 'bpi_rk':[bpi_rk], 'sos_rk':[sos_rk], 'sor_rk':[sor_rk], 'rpi_rk':[rpi_rk],'season':[season]})
            
            rpi_data = rpi_data.append(df)
            
    cnt+=1
 
rpi_data = rpi_data[['rank','team','conference','winloss','bpi_rk','sos_rk','sor_rk','rpi_rk','season']].reset_index()
rpi_data.drop(rpi_data.columns[[0]],axis=1,inplace=True)

##Create dictionary to rename columns in ESPN data. This will allow joining to the original data from kaggle.
#teamnamedict = dict(zip(teamnamemapping['ESPN_Name'],teamnamemapping['Team_Name']))
#
#rpi_data['team'].replace(to_replace=teamnamedict,inplace=True)


rpi_data = rpi_data.merge(teamnamemapping, how='left', left_on='team', right_on='\ufeffESPN_Name')
rpi_data['team'] = rpi_data['Team_Name']
rpi_data.drop(['\ufeffESPN_Name','Team_Name'], axis=1,inplace=True)

teams['Team_Id']=teams['Team_Id'].astype(str)

rpi_data = rpi_data.merge(teams[['Team_Id','Team_Name']], how='left', left_on='team', right_on='Team_Name')

rpi_data = rpi_data[['Team_Id','rank', 'team', 'conference', 'winloss', 'bpi_rk', 'sos_rk', 'sor_rk', 'rpi_rk', 'season']]

#rpi_data[rpi_data['rpi_rk']=='--'] = '351'

rpi_data['rpi_rk'] = rpi_data['rpi_rk'].str.replace('--','351')

rpi_data[['rank','bpi_rk', 'sos_rk', 'sor_rk', 'rpi_rk']] = rpi_data[['rank','bpi_rk', 'sos_rk', 'sor_rk', 'rpi_rk']].astype(int)

rpi_data = rpi_data[rpi_data['Team_Id'].notnull()]
                    
#rpi_data2 = rpi_data
#
#rpi_data = rpi_data2
#
#rpi_data[['Team_Id']] = rpi_data[['Team_Id']].astype(str)
#rpi_data[['Team_Id']] = rpi_data.Team_Id.apply(lambda x: x[0:4])
#rpi_data[['season']] = rpi_data[['season']].astype(str)

#####

# #samplesubmission[samplesubmission['id'].str.contains("2016")]

#tourneyrsltcmpt['Season'].plot(kind='hist')
#plt.show()

# tourneyrsltcmpt['Wscore'].plot(kind='hist')
# plt.show()

####################

##Create base structure for regular season wins using teams data set
regseasdtl['Wteam']=regseasdtl['Wteam'].astype(str)
regseasdtl['Lteam']=regseasdtl['Lteam'].astype(str)
regseasdtl['Season']=regseasdtl['Season'].astype(str)

regseaswin = teams

##Join regseaswin with regseasdtl to get data for times teams won and associated details
regseaswin = regseaswin.merge(regseasdtl, how='left', left_on=['Team_Id'], right_on=['Wteam'])

regseaswin['win'] = 1

##Create base structure for regular season losses using teams data set
regseasloss = teams

##Join regseasloss with regseasdtl to get data for times teams lost and associated details
regseasloss = regseasloss.merge(regseasdtl, how='left', left_on='Team_Id', right_on='Lteam')

regseasloss['win'] = 0

##Concatenate wins and loss data
regseasgmstats = pd.concat([regseaswin, regseasloss]).sort(['Team_Id', 'Season', 'Daynum'], ascending=[1,1,1])


regseasgmstats['Opponent']=regseasgmstats.apply(lambda x: x['Wteam'] if x['Wteam'] != x['Team_Id'] else x['Lteam'],axis=1)

##Create aggregate table for team summary metrics by year

teamdtlsummaryw = regseasgmstats[['Team_Id','Team_Name','Season',
                                'Wscore','Wfgm','Wfga','Wfgm3','Wftm','Wfta','Wor','Wdr',
                                'Wast','Wto','Wstl','Wblk','Wpf',
                                'win']][regseasgmstats['Team_Id']==regseasgmstats['Wteam']]
##Rename columns
teamdtlsummaryw.columns = ['Team_Id','Team_Name','Season',
                                'score','fgm','fga','fgm3','ftm','fta','or','dr',
                                'ast','to','stl','blk','pf',
                                'win']


teamdtlsummaryl = regseasgmstats[['Team_Id','Team_Name','Season',
                                'Lscore','Lfgm','Lfga','Lfgm3','Lftm','Lfta','Lor','Ldr',
                                'Last','Lto','Lstl','Lblk','Lpf',
                                'win']][regseasgmstats['Team_Id']==regseasgmstats['Lteam']]
##Rename columns
teamdtlsummaryl.columns = ['Team_Id','Team_Name','Season',
                                'score','fgm','fga','fgm3','ftm','fta','or','dr',
                                'ast','to','stl','blk','pf',
                                'win']

teamdtlsummary = pd.concat([teamdtlsummaryw,teamdtlsummaryl]).sort(['Team_Id', 'Season'], ascending=[1,1])

##Aggregate data
teamdtlsummaryagg = teamdtlsummary.groupby(['Team_Id','Team_Name','Season'])\
                    .agg({'score':{'avg_score':np.mean,'median_score':np.median, 'std_score':np.std},\
                          'fgm':{'avg_fgm':np.mean,'median_fgm':np.median, 'std_fgm':np.std},\
                          'fga':{'avg_fga':np.mean,'median_fga':np.median, 'std_fga':np.std},\
                          'fgm3':{'avg_fgm3':np.mean,'median_fgm3':np.median, 'std_fgm3':np.std},\
                          'ftm':{'avg_ftm':np.mean,'median_ftm':np.median, 'std_ftm':np.std},\
                          'fta':{'avg_fta':np.mean,'median_fta':np.median, 'std_fta':np.std},\
                          'or':{'avg_or':np.mean,'median_or':np.median, 'std_or':np.std},\
                          'dr':{'avg_dr':np.mean,'median_dr':np.median, 'std_dr':np.std},\
                          'ast':{'avg_ast':np.mean,'median_ast':np.median, 'std_ast':np.std},\
                          'to':{'avg_to':np.mean,'median_to':np.median, 'std_to':np.std},\
                          'stl':{'avg_stl':np.mean,'median_stl':np.median, 'std_stl':np.std},\
                          'blk':{'avg_blk':np.mean,'median_blk':np.median, 'std_blk':np.std},\
                          'pf':{'avg_pf':np.mean,'median_pf':np.median, 'std_pf':np.std},\
                          'win':{'win':np.sum}
                    })\
                    .reset_index()

teamdtlsummaryagg.columns = teamdtlsummaryagg.columns.droplevel(0)

teamdtlsummaryagg.columns.values[0] = 'Team_Id'
teamdtlsummaryagg.columns.values[1] = 'Team_Name'
teamdtlsummaryagg.columns.values[2] = 'Season'
##teamdtlsummaryagg.columns.values[20] = 'win'

##Create table for modeling
regseasgmstatsagg1 = regseasgmstats[['Team_Id','Team_Name','Season','Opponent','win']]\
                    .merge(teamdtlsummaryagg[['Team_Id','Season','avg_score','median_score','std_score',\
                    'avg_fgm','median_fgm','std_fgm','avg_fga','median_fga', 'std_fga',\
                    'avg_fgm3','median_fgm3','std_fgm3','avg_ftm','median_ftm', 'std_ftm',\
                    'avg_fta','median_fta','std_fta','avg_or','median_or', 'std_or',\
                    'avg_dr','median_dr','std_dr','avg_ast','median_ast', 'std_ast',\
                    'avg_to','median_to','std_to','avg_stl','median_stl','std_stl',\
                    'avg_blk','median_blk', 'std_blk','avg_pf','median_pf','std_pf','win']], how='left', left_on=['Team_Id','Season'], right_on=['Team_Id','Season'])

regseasgmstatsagg1.rename(columns={'win_x':'win'},inplace=True)
regseasgmstatsagg1.rename(columns={'win_y':'total_wins'},inplace=True)


regseasgmstatsagg2 = regseasgmstatsagg1.merge(teamdtlsummaryagg[['Team_Id','Season','avg_score','median_score','std_score',\
                    'avg_fgm','median_fgm','std_fgm','avg_fga','median_fga', 'std_fga',\
                    'avg_fgm3','median_fgm3','std_fgm3','avg_ftm','median_ftm', 'std_ftm',\
                    'avg_fta','median_fta','std_fta','avg_or','median_or', 'std_or',\
                    'avg_dr','median_dr','std_dr','avg_ast','median_ast', 'std_ast',\
                    'avg_to','median_to','std_to','avg_stl','median_stl','std_stl',\
                    'avg_blk','median_blk', 'std_blk','avg_pf','median_pf','std_pf','win']], how='left', left_on=['Opponent','Season'], right_on=['Team_Id','Season'])

regseasgmstatsagg2.drop('Team_Id_y',axis=1,inplace=True)

old_names = ['Team_Id_x', 'Team_Name', 'Season', 'Opponent', 'win_x',
       'avg_score_x', 'median_score_x', 'std_score_x', 'avg_fgm_x',
       'median_fgm_x', 'std_fgm_x', 'avg_fga_x', 'median_fga_x',
       'std_fga_x', 'avg_fgm3_x', 'median_fgm3_x', 'std_fgm3_x',
       'avg_ftm_x', 'median_ftm_x', 'std_ftm_x', 'avg_fta_x',
       'median_fta_x', 'std_fta_x', 'avg_or_x', 'median_or_x', 'std_or_x',
       'avg_dr_x', 'median_dr_x', 'std_dr_x', 'avg_ast_x', 'median_ast_x',
       'std_ast_x', 'avg_to_x', 'median_to_x', 'std_to_x', 'avg_stl_x',
       'median_stl_x', 'std_stl_x', 'avg_blk_x', 'median_blk_x',
       'std_blk_x', 'avg_pf_x', 'median_pf_x', 'std_pf_x', 'total_wins',
       'Team_Id_y', 'avg_score_y', 'median_score_y', 'std_score_y',
       'avg_fgm_y', 'median_fgm_y', 'std_fgm_y', 'avg_fga_y',
       'median_fga_y', 'std_fga_y', 'avg_fgm3_y', 'median_fgm3_y',
       'std_fgm3_y', 'avg_ftm_y', 'median_ftm_y', 'std_ftm_y', 'avg_fta_y',
       'median_fta_y', 'std_fta_y', 'avg_or_y', 'median_or_y', 'std_or_y',
       'avg_dr_y', 'median_dr_y', 'std_dr_y', 'avg_ast_y', 'median_ast_y',
       'std_ast_y', 'avg_to_y', 'median_to_y', 'std_to_y', 'avg_stl_y',
       'median_stl_y', 'std_stl_y', 'avg_blk_y', 'median_blk_y',
       'std_blk_y', 'avg_pf_y', 'median_pf_y', 'std_pf_y', 'win_y'] 
       
new_names = ['Team_Id', 'Team_Name', 'Season', 'Opponent', 'win',
       'avg_score', 'median_score', 'std_score', 'avg_fgm',
       'median_fgm', 'std_fgm', 'avg_fga', 'median_fga',
       'std_fga', 'avg_fgm3', 'median_fgm3', 'std_fgm3',
       'avg_ftm', 'median_ftm', 'std_ftm', 'avg_fta',
       'median_fta', 'std_fta', 'avg_or', 'median_or', 'std_or',
       'avg_dr', 'median_dr', 'std_dr', 'avg_ast', 'median_ast',
       'std_ast', 'avg_to', 'median_to', 'std_to', 'avg_stl',
       'median_stl', 'std_stl', 'avg_blk', 'median_blk',
       'std_blk', 'avg_pf', 'median_pf', 'std_pf', 'total_wins', 'Opponent', 'opponent_avg_score',
       'opponent_median_score', 'opponent_std_score', 'opponent_avg_fgm', 'opponent_median_fgm', 'opponent_std_fgm',
       'opponent_avg_fga', 'opponent_median_fga', 'opponent_std_fga', 'opponent_avg_fgm3', 'opponent_median_fgm3',
       'opponent_std_fgm3', 'opponent_avg_ftm', 'opponent_median_ftm', 'opponent_std_ftm', 'opponent_avg_fta',
       'opponent_median_fta', 'opponent_std_fta', 'opponent_avg_or', 'opponent_median_or', 'opponent_std_or', 'opponent_avg_dr',
       'opponent_median_dr', 'opponent_std_dr', 'opponent_avg_ast', 'opponent_median_ast', 'opponent_std_ast', 'opponent_avg_to',
       'opponent_median_to', 'opponent_std_to', 'opponent_avg_stl', 'opponent_median_stl', 'opponent_std_stl',
       'opponent_avg_blk', 'opponent_median_blk', 'opponent_std_blk', 'opponent_avg_pf', 'opponent_median_pf', 'opponent_std_pf',
       'opponent_total_wins']

regseasgmstatsagg2.rename(columns=dict(zip(old_names,new_names)),inplace=True)

#regseasgmstatsagg = regseasgmstatsagg1.merge(regseasgmstatsagg2, how='left',left_on=['Team_Id','Season','Opponent'], right_on=['Team_Id','Season','Opponent'] )
#regseasgmstatsagg.rename(columns={'win_x':'win'},inplace=True)
#regseasgmstatsagg.drop('win_y',axis=1,inplace=True)

regseasgmstatsagg = regseasgmstatsagg2
regseasgmstatsagg = regseasgmstatsagg[np.isfinite(regseasgmstatsagg['avg_score'])]
#regseasgmstatsagg['Team_Id'] = regseasgmstatsagg['Team_Id'].astype(str)
regseasgmstatsagg['Season'] = regseasgmstatsagg['Season'].astype(int).astype(str)
regseasgmstatsagg['Opponent'] = regseasgmstatsagg['Opponent'].astype(int).astype(str)

regseasgmstatsagg = regseasgmstatsagg.merge(rpi_data, how='left', left_on=['Team_Id','Season'], right_on=['Team_Id','season'])
#len(regseasgmstatsagg[regseasgmstatsagg['rank'].isnull()])



regseasgmstatsagg = regseasgmstatsagg.merge(rpi_data, how='left', left_on=['Opponent','Season'], right_on=['Team_Id','season'])
#len(regseasgmstatsagg[regseasgmstatsagg['rank_x'].isnull()])

regseasgmstatsagg = regseasgmstatsagg[regseasgmstatsagg['rank_x'].notnull()]

old_names = ['Team_Id_x', 'Team_Name', 'Season', 'Opponent', 'win',
       'avg_score', 'median_score', 'std_score', 'avg_fgm',
       'median_fgm', 'std_fgm', 'avg_fga', 'median_fga',
       'std_fga', 'avg_fgm3', 'median_fgm3', 'std_fgm3',
       'avg_ftm', 'median_ftm', 'std_ftm', 'avg_fta',
       'median_fta', 'std_fta', 'avg_or', 'median_or', 'std_or',
       'avg_dr', 'median_dr', 'std_dr', 'avg_ast', 'median_ast',
       'std_ast', 'avg_to', 'median_to', 'std_to', 'avg_stl',
       'median_stl', 'std_stl', 'avg_blk', 'median_blk',
       'std_blk', 'avg_pf', 'median_pf', 'std_pf', 'total_wins', 'Opponent', 'opponent_avg_score',
       'opponent_median_score', 'opponent_std_score', 'opponent_avg_fgm', 'opponent_median_fgm', 'opponent_std_fgm',
       'opponent_avg_fga', 'opponent_median_fga', 'opponent_std_fga', 'opponent_avg_fgm3', 'opponent_median_fgm3',
       'opponent_std_fgm3', 'opponent_avg_ftm', 'opponent_median_ftm', 'opponent_std_ftm', 'opponent_avg_fta',
       'opponent_median_fta', 'opponent_std_fta', 'opponent_avg_or', 'opponent_median_or', 'opponent_std_or', 'opponent_avg_dr',
       'opponent_median_dr', 'opponent_std_dr', 'opponent_avg_ast', 'opponent_median_ast', 'opponent_std_ast', 'opponent_avg_to',
       'opponent_median_to', 'opponent_std_to', 'opponent_avg_stl', 'opponent_median_stl', 'opponent_std_stl',
       'opponent_avg_blk', 'opponent_median_blk', 'opponent_std_blk', 'opponent_avg_pf', 'opponent_median_pf', 'opponent_std_pf',
       'opponent_total_wins', 'rank_x','team_x','conference_x','winloss_x','bpi_rk_x','sos_rk_x','sor_rk_x',
       'rpi_rk_x','season_x','Team_Id_y','rank_y','team_y','conference_y','winloss_y','bpi_rk_y','sos_rk_y',
       'sor_rk_y','rpi_rk_y','season_y']

new_names = ['Team_Id', 'Team_Name', 'Season', 'Opponent', 'win',
       'avg_score', 'median_score', 'std_score', 'avg_fgm',
       'median_fgm', 'std_fgm', 'avg_fga', 'median_fga',
       'std_fga', 'avg_fgm3', 'median_fgm3', 'std_fgm3',
       'avg_ftm', 'median_ftm', 'std_ftm', 'avg_fta',
       'median_fta', 'std_fta', 'avg_or', 'median_or', 'std_or',
       'avg_dr', 'median_dr', 'std_dr', 'avg_ast', 'median_ast',
       'std_ast', 'avg_to', 'median_to', 'std_to', 'avg_stl',
       'median_stl', 'std_stl', 'avg_blk', 'median_blk',
       'std_blk', 'avg_pf', 'median_pf', 'std_pf', 'total_wins', 'Opponent', 'opponent_avg_score',
       'opponent_median_score', 'opponent_std_score', 'opponent_avg_fgm', 'opponent_median_fgm', 'opponent_std_fgm',
       'opponent_avg_fga', 'opponent_median_fga', 'opponent_std_fga', 'opponent_avg_fgm3', 'opponent_median_fgm3',
       'opponent_std_fgm3', 'opponent_avg_ftm', 'opponent_median_ftm', 'opponent_std_ftm', 'opponent_avg_fta',
       'opponent_median_fta', 'opponent_std_fta', 'opponent_avg_or', 'opponent_median_or', 'opponent_std_or', 'opponent_avg_dr',
       'opponent_median_dr', 'opponent_std_dr', 'opponent_avg_ast', 'opponent_median_ast', 'opponent_std_ast', 'opponent_avg_to',
       'opponent_median_to', 'opponent_std_to', 'opponent_avg_stl', 'opponent_median_stl', 'opponent_std_stl',
       'opponent_avg_blk', 'opponent_median_blk', 'opponent_std_blk', 'opponent_avg_pf', 'opponent_median_pf', 'opponent_std_pf',
       'opponent_total_wins', 'rank','team','conference','winloss','bpi_rk','sos_rk','sor_rk',
       'rpi_rk','season','opponent_Team_Id','opponent_rank','opponent_team','opponent_conference','opponent_winloss','opponent_bpi_rk','opponent_sos_rk',
       'opponent_sor_rk','opponent_rpi_rk','opponent_season']

regseasgmstatsagg.rename(columns=dict(zip(old_names,new_names)),inplace=True)

regseasgmstatsagg.drop(['season','opponent_season','team'],axis=1,inplace=True)


#==============================================================================
# Get the max ranking values to fill null values in regseasgmstatsagg
#==============================================================================
#rpi_data_max = rpi_data.groupby(['season']).agg({'rank':np.max,'bpi_rk':np.max,\
#                                                'sos_rk':np.max,'sor_rk':np.max,'rpi_rk':np.max}).reset_index()
#
#regseasgmstatsagg = regseasgmstatsagg.merge(rpi_data_max, how='left', left_on='Season', right_on='season')                
#                   
#regseasgmstatsagg['rank_x'][regseasgmstatsagg['rank_x'].isnull()] = regseasgmstatsagg['rank_y']          
#regseasgmstatsagg['bpi_rk_x'][regseasgmstatsagg['bpi_rk_x'].isnull()] = regseasgmstatsagg['bpi_rk_y']
#regseasgmstatsagg['sos_rk_x'][regseasgmstatsagg['sos_rk_x'].isnull()] = regseasgmstatsagg['sos_rk_y']
#regseasgmstatsagg['sor_rk_x'][regseasgmstatsagg['sor_rk_x'].isnull()] = regseasgmstatsagg['sor_rk_y']
#regseasgmstatsagg['rpi_rk_x'][regseasgmstatsagg['rpi_rk_x'].isnull()] = regseasgmstatsagg['rpi_rk_y']
#
#regseasgmstatsagg.drop(['season_y', 'bpi_rk_y', 'sor_rk_y', 'sos_rk_y', 'rank_y', 'rpi_rk_y'], axis=1, inplace=True)
#                
#regseasgmstatsagg.rename(columns={'rank_x':'rank','bpi_rk_x':'bpi_rk', 'sos_rk_x':'sos_rk', 'sor_rk_x':'sor_rk', 'rpi_rk_x':'rpi_rk', 'season_x':'season'},inplace=True)          
#     
#datclean = regseasgmstatsagg[np.isfinite(regseasgmstatsagg['avg_score'])]



datclean = regseasgmstatsagg[regseasgmstatsagg['opponent_bpi_rk'].notnull()]

datclean['scorediff'] = datclean['avg_score'] - datclean['opponent_avg_score']

#==============================================================================
# Model Prep
#==============================================================================
x = datclean[['scorediff','std_score', 'opponent_std_score','rank','bpi_rk', 'sos_rk',\
              'sor_rk', 'rpi_rk','opponent_rank','opponent_bpi_rk', 'opponent_sos_rk',\
              'opponent_sor_rk', 'opponent_rpi_rk']]
y = datclean[['win']].values

x2 = datclean[['scorediff','std_score', 'opponent_std_score']]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=1)

#==============================================================================
# Logistical Regression
#==============================================================================
from sklearn.linear_model import LogisticRegression        
     
logreg = LogisticRegression(max_iter=1000, solver='sag')

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_test)

print(logreg.score(x_test,y_test))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#==============================================================================
# LinearSVC
#==============================================================================
from sklearn.svm import LinearSVC


x = datclean[['scorediff','std_score', 'opponent_std_score','rank','bpi_rk', 'sos_rk',\
              'sor_rk', 'rpi_rk','opponent_rank','opponent_bpi_rk', 'opponent_sos_rk',\
              'opponent_sor_rk', 'opponent_rpi_rk']]
y = datclean[['win']].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=1)

linsvc = LinearSVC()

linsvc.fit(x_train, y_train)

y_pred = linsvc.predict(x_test)

print(linsvc.score(x_test,y_test))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#==============================================================================
# SVR
#==============================================================================
from sklearn.svm import SVR

x = datclean[['scorediff','std_score', 'opponent_std_score','rank','bpi_rk', 'sos_rk',\
              'sor_rk', 'rpi_rk','opponent_rank','opponent_bpi_rk', 'opponent_sos_rk',\
              'opponent_sor_rk', 'opponent_rpi_rk']]
y = datclean[['win']].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=1)

svr = SVR()

svr.fit(x_train, y_train)

y_pred = svr.predict(x_test)

print(svr.score(x_test,y_test))

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#==============================================================================
# Logistic Regression 1
#==============================================================================
from sklearn.cross_validation import KFold, cross_val_score

x = datclean[['scorediff','std_score', 'opponent_std_score','rank','bpi_rk', 'sos_rk',\
              'sor_rk', 'rpi_rk','opponent_rank','opponent_bpi_rk', 'opponent_sos_rk',\
              'opponent_sor_rk', 'opponent_rpi_rk']]
y = datclean[['win']].values


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=1)

kf = KFold(len(y), n_folds=10, shuffle=True, random_state=174)
clf = LogisticRegression(C=10,penalty='l1')
scores = cross_val_score(clf, x2, y, cv=kf, scoring='roc_auc')
print ('LogReg score:', np.mean(scores))

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
y_pred_prob = clf.predict_proba(x_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#==============================================================================
# Logistic Regression 2
#==============================================================================
from sklearn.cross_validation import KFold, cross_val_score

x2 = datclean[['scorediff','std_score', 'opponent_std_score']]
y = datclean[['win']].values

x_train, x_test, y_train, y_test = train_test_split(x2,y,test_size=0.4,random_state=1)

kf = KFold(len(y), n_folds=10, shuffle=True, random_state=174)
clf = LogisticRegression(C=0.01,penalty='l1')
scores = cross_val_score(clf, x2, y, cv=kf, scoring='roc_auc')
print ('LogReg score:', np.mean(scores))

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
y_pred_prob = clf.predict_proba(x_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#==============================================================================
# SGD1
#==============================================================================
from sklearn.linear_model import SGDClassifier

x = datclean[['scorediff','std_score', 'opponent_std_score','rank','bpi_rk', 'sos_rk',\
              'sor_rk', 'rpi_rk','opponent_rank','opponent_bpi_rk', 'opponent_sos_rk',\
              'opponent_sor_rk', 'opponent_rpi_rk']]
y = datclean[['win']].values


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=1)

kf = KFold(len(y), n_folds=5, shuffle=True, random_state=174)
sgd = SGDClassifier(loss='log',alpha=10)
scores = cross_val_score(sgd, x, y, cv=kf, scoring='roc_auc')
print ('LogReg score:', np.mean(scores))

sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_test)
y_pred_prob = sgd.predict_proba(x_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#==============================================================================
# KN Classifier
#==============================================================================
from sklearn.neighbors import KNeighborsClassifier

x = datclean[['scorediff','std_score', 'opponent_std_score','rank','bpi_rk', 'sos_rk',\
              'sor_rk', 'rpi_rk','opponent_rank','opponent_bpi_rk', 'opponent_sos_rk',\
              'opponent_sor_rk', 'opponent_rpi_rk']]
y = datclean[['win']].values


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=1)

kf = KFold(len(y), n_folds=5, shuffle=True, random_state=174)
knc = KNeighborsClassifier(n_neighbors=50, algorithm='ball_tree')
scores = cross_val_score(knc, x, y, cv=kf, scoring='roc_auc')
print ('LogReg score:', np.mean(scores))

knc.fit(x_train, y_train)
y_pred = knc.predict(x_test)
y_pred_prob = knc.predict_proba(x_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#==============================================================================
# Decision Tree Classifier
#==============================================================================
from sklearn.ensemble import GradientBoostingClassifier

x = datclean[['scorediff','std_score', 'opponent_std_score','rank','bpi_rk', 'sos_rk',\
              'sor_rk', 'rpi_rk','opponent_rank','opponent_bpi_rk', 'opponent_sos_rk',\
              'opponent_sor_rk', 'opponent_rpi_rk']]
y = datclean[['win']].values


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=1)

gbc = KFold(len(y), n_folds=5, shuffle=True, random_state=174)
gbc = GradientBoostingClassifier(loss='exponential',max_features='sqrt',random_state=1)
scores = cross_val_score(gbc, x, y, cv=kf, scoring='roc_auc')
print ('LogReg score:', np.mean(scores))

dc.fit(x_train, y_train)
y_pred = dc.predict(x_test)
y_pred_prob = dc.predict_proba(x_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#==============================================================================
# Decision Tree Classifier
#==============================================================================
from sklearn.neural_network import MLPClassifier

x = datclean[['scorediff','std_score', 'opponent_std_score','rank','bpi_rk', 'sos_rk',\
              'sor_rk', 'rpi_rk','opponent_rank','opponent_bpi_rk', 'opponent_sos_rk',\
              'opponent_sor_rk', 'opponent_rpi_rk']]
y = datclean[['win']].values


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.4,random_state=1)

mlp = KFold(len(y), n_folds=5, shuffle=True, random_state=174)
mlp = MLPClassifier(alpha=0.01,activation='logistic',random_state=1)
scores = cross_val_score(mlp, x, y, cv=kf, scoring='roc_auc')
print ('LogReg score:', np.mean(scores))

dc.fit(x_train, y_train)
y_pred = dc.predict(x_test)
y_pred_prob = dc.predict_proba(x_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#==============================================================================
# Submission
#==============================================================================
#submission = samplesubmission
#
#submission['season'] = submission['Id'].str.split(pat='_',expand=True)[0]
#submission['team'] = submission['Id'].str.split(pat='_',expand=True)[1]
#submission['opponent'] = submission['Id'].str.split(pat='_',expand=True)[2]

submission = []


for x in range(1101,1465):
    for y in range(1101,1465):
        rec=[str(x),str(y)]
        submission.append(rec)

submission = pd.DataFrame(data=submission, columns=['team','opponent'])
submission = submission[submission['team']!=submission['opponent']]
submission['season'] = '2017'
submission['Pred'] = 0.5
submission['Id'] = submission['season']+'_'+submission['team']+'_'+submission['opponent']

submission = submission[['Id','Pred','season','team','opponent']]

#datclean_2017 = datclean[datclean['Season']=='2017'].groupby('Team_Id','Season')

datclean_2017_team = datclean[['Team_Id','Season','scorediff','std_score','rank','bpi_rk', 'sos_rk',\
              'sor_rk', 'rpi_rk']].groupby(['Team_Id','Season'],axis=0).max().reset_index()

datclean_2017_opp = datclean_2017_team[['Team_Id','Season','std_score','rank','bpi_rk', 'sos_rk',\
              'sor_rk', 'rpi_rk']]

oldcols = ['Team_Id','Season','std_score','rank','bpi_rk', 'sos_rk',\
              'sor_rk', 'rpi_rk']

newcols = ['opponent_Team_Id','Season', 'opponent_std_score','opponent_rank','opponent_bpi_rk', 'opponent_sos_rk',\
              'opponent_sor_rk', 'opponent_rpi_rk']

datclean_2017_opp.rename(columns=dict(zip(oldcols,newcols)),inplace=True)

#datclean_2017_opp = datclean_2017[['opponent_Team_Id','Season', 'opponent_std_score','opponent_rank','opponent_bpi_rk', 'opponent_sos_rk',\
#              'opponent_sor_rk', 'opponent_rpi_rk']].groupby(['opponent_Team_Id','Season'],axis=0).max().reset_index()

sub_all = submission.merge(datclean_2017_team,how='left',left_on=['team','season'],right_on=['Team_Id','Season'])

sub_all = sub_all.merge(datclean_2017_opp,how='left',left_on=['opponent','season'],right_on=['opponent_Team_Id','Season'])

###Impute values for missing data
sub_all = sub_all.fillna(sub_all.mean())



cols = ['scorediff','std_score', 'opponent_std_score','rank','bpi_rk', 'sos_rk',
        'sor_rk', 'rpi_rk','opponent_rank','opponent_bpi_rk', 'opponent_sos_rk',
        'opponent_sor_rk', 'opponent_rpi_rk']

y_pred2 = clf.predict(sub_all[cols])
y_pred_prob2 = clf.predict_proba(sub_all[cols])[:, 1]

sub_all['pred'] = y_pred2 #(sub_all[cols].apply(lambda x: clf.predict(x)))
sub_all['pred_prob'] = y_pred_prob2 

sub_out = sub_all[['Id','pred_prob']]

oldcols = ['Id', 'pred_prob']

newcols = ['Id', 'pred']


sub_out.rename(columns=dict(zip(oldcols,newcols)),inplace=True)

sub_out.to_csv('MM2017Submission_logreg.csv',index=False)



#==============================================================================
# Bracket Prediction
#==============================================================================

east = ['1437','1291','1458','1439','1438','1423','1196','1190','1374','1425','1124','1308','1376','1266','1181','1407']

west = ['1211','1355','1321','1435','1323','1343','1452','1137','1268','1462','1199','1195','1388','1433','1112','1315']

mwest = ['1242','1413','1274','1277','1235','1305','1345','1436','1166','1348','1332','1233','1276','1329','1257','1240']

south = ['1314','1411','1116','1371','1278','1292','1139','1457','1153','1243','1417','1245','1173','1455','1246','1297']

sd = ['1','16','8','9','5','12','4','13','6','11','3','14','7','10','2','15']

eastbracket = pd.DataFrame({'seed':sd,'Team_Id':east})
westbracket = pd.DataFrame({'seed':sd,'Team_Id':west})
mwestbracket = pd.DataFrame({'seed':sd,'Team_Id':mwest})
southbracket = pd.DataFrame({'seed':sd,'Team_Id':south})


###East


rd1a = [0,2,4,6,8,10,12,14]
rd1b = [1,3,5,7,9,11,13,15]

eastrd1a = eastbracket.iloc[rd1a].reset_index()
eastrd1a.drop(['index'],axis=1,inplace=True)
eastrd1b = eastbracket.iloc[rd1b].reset_index()
eastrd1b.drop(['index'],axis=1,inplace=True)
eastrd1 = pd.merge(eastrd1a, eastrd1b, how='inner',left_index=True,right_index=True)
eastrd1 = eastrd1.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
eastrd1 = eastrd1.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
eastrd1.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
eastrd1['winner'] = eastrd1[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

eastbracketrd2 = eastbracket[eastbracket['Team_Id'].isin(eastrd1['winner'])]

rd2a = [0,2,4,6]
rd2b = [1,3,5,7]

eastrd2a = eastbracketrd2.iloc[rd2a].reset_index()
eastrd2a.drop(['index'],axis=1,inplace=True)
eastrd2b = eastbracketrd2.iloc[rd2b].reset_index()
eastrd2b.drop(['index'],axis=1,inplace=True)
eastrd2 = pd.merge(eastrd2a, eastrd2b, how='inner',left_index=True,right_index=True)
eastrd2 = eastrd2.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
eastrd2 = eastrd2.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
eastrd2.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
eastrd2['winner'] = eastrd2[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

eastbracketrd3 = eastbracket[eastbracket['Team_Id'].isin(eastrd2['winner'])]

rd3a = [0,2]
rd3b = [1,3]

eastrd3a = eastbracketrd3.iloc[rd3a].reset_index()
eastrd3a.drop(['index'],axis=1,inplace=True)
eastrd3b = eastbracketrd3.iloc[rd3b].reset_index()
eastrd3b.drop(['index'],axis=1,inplace=True)
eastrd3 = pd.merge(eastrd3a, eastrd3b, how='inner',left_index=True,right_index=True)
eastrd3 = eastrd3.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
eastrd3 = eastrd3.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
eastrd3.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
eastrd3['winner'] = eastrd3[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

eastbracketrd4 = eastbracket[eastbracket['Team_Id'].isin(eastrd3['winner'])]

rd4a = [0]
rd4b = [1]

eastrd4a = eastbracketrd4.iloc[rd4a].reset_index()
eastrd4a.drop(['index'],axis=1,inplace=True)
eastrd4b = eastbracketrd4.iloc[rd4b].reset_index()
eastrd4b.drop(['index'],axis=1,inplace=True)
eastrd4 = pd.merge(eastrd4a, eastrd4b, how='inner',left_index=True,right_index=True)
eastrd4 = eastrd4.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
eastrd4 = eastrd4.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
eastrd4.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
eastrd4['winner'] = eastrd4[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

###West

rd1a = [0,2,4,6,8,10,12,14]
rd1b = [1,3,5,7,9,11,13,15]

westrd1a = westbracket.iloc[rd1a].reset_index()
westrd1a.drop(['index'],axis=1,inplace=True)
westrd1b = westbracket.iloc[rd1b].reset_index()
westrd1b.drop(['index'],axis=1,inplace=True)
westrd1 = pd.merge(westrd1a, westrd1b, how='inner',left_index=True,right_index=True)
westrd1 = westrd1.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
westrd1 = westrd1.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
westrd1.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
westrd1['winner'] = westrd1[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

westbracketrd2 = westbracket[westbracket['Team_Id'].isin(westrd1['winner'])]

rd2a = [0,2,4,6]
rd2b = [1,3,5,7]

westrd2a = westbracketrd2.iloc[rd2a].reset_index()
westrd2a.drop(['index'],axis=1,inplace=True)
westrd2b = westbracketrd2.iloc[rd2b].reset_index()
westrd2b.drop(['index'],axis=1,inplace=True)
westrd2 = pd.merge(westrd2a, westrd2b, how='inner',left_index=True,right_index=True)
westrd2 = westrd2.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
westrd2 = westrd2.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
westrd2.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
westrd2['winner'] = westrd2[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

westbracketrd3 = westbracket[westbracket['Team_Id'].isin(westrd2['winner'])]

rd3a = [0,2]
rd3b = [1,3]

westrd3a = westbracketrd3.iloc[rd3a].reset_index()
westrd3a.drop(['index'],axis=1,inplace=True)
westrd3b = westbracketrd3.iloc[rd3b].reset_index()
westrd3b.drop(['index'],axis=1,inplace=True)
westrd3 = pd.merge(westrd3a, westrd3b, how='inner',left_index=True,right_index=True)
westrd3 = westrd3.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
westrd3 = westrd3.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
westrd3.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
westrd3['winner'] = westrd3[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

westbracketrd4 = westbracket[westbracket['Team_Id'].isin(westrd3['winner'])]

rd4a = [0]
rd4b = [1]

westrd4a = westbracketrd4.iloc[rd4a].reset_index()
westrd4a.drop(['index'],axis=1,inplace=True)
westrd4b = westbracketrd4.iloc[rd4b].reset_index()
westrd4b.drop(['index'],axis=1,inplace=True)
westrd4 = pd.merge(westrd4a, westrd4b, how='inner',left_index=True,right_index=True)
westrd4 = westrd4.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
westrd4 = westrd4.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
westrd4.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
westrd4['winner'] = westrd4[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

###Mwest

rd1a = [0,2,4,6,8,10,12,14]
rd1b = [1,3,5,7,9,11,13,15]

mwestrd1a = mwestbracket.iloc[rd1a].reset_index()
mwestrd1a.drop(['index'],axis=1,inplace=True)
mwestrd1b = mwestbracket.iloc[rd1b].reset_index()
mwestrd1b.drop(['index'],axis=1,inplace=True)
mwestrd1 = pd.merge(mwestrd1a, mwestrd1b, how='inner',left_index=True,right_index=True)
mwestrd1 = mwestrd1.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
mwestrd1 = mwestrd1.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
mwestrd1.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
mwestrd1['winner'] = mwestrd1[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

mwestbracketrd2 = mwestbracket[mwestbracket['Team_Id'].isin(mwestrd1['winner'])]

rd2a = [0,2,4,6]
rd2b = [1,3,5,7]

mwestrd2a = mwestbracketrd2.iloc[rd2a].reset_index()
mwestrd2a.drop(['index'],axis=1,inplace=True)
mwestrd2b = mwestbracketrd2.iloc[rd2b].reset_index()
mwestrd2b.drop(['index'],axis=1,inplace=True)
mwestrd2 = pd.merge(mwestrd2a, mwestrd2b, how='inner',left_index=True,right_index=True)
mwestrd2 = mwestrd2.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
mwestrd2 = mwestrd2.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
mwestrd2.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
mwestrd2['winner'] = mwestrd2[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

mwestbracketrd3 = mwestbracket[mwestbracket['Team_Id'].isin(mwestrd2['winner'])]

rd3a = [0,2]
rd3b = [1,3]

mwestrd3a = mwestbracketrd3.iloc[rd3a].reset_index()
mwestrd3a.drop(['index'],axis=1,inplace=True)
mwestrd3b = mwestbracketrd3.iloc[rd3b].reset_index()
mwestrd3b.drop(['index'],axis=1,inplace=True)
mwestrd3 = pd.merge(mwestrd3a, mwestrd3b, how='inner',left_index=True,right_index=True)
mwestrd3 = mwestrd3.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
mwestrd3 = mwestrd3.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
mwestrd3.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
mwestrd3['winner'] = mwestrd3[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

mwestbracketrd4 = mwestbracket[mwestbracket['Team_Id'].isin(mwestrd3['winner'])]

rd4a = [0]
rd4b = [1]

mwestrd4a = mwestbracketrd4.iloc[rd4a].reset_index()
mwestrd4a.drop(['index'],axis=1,inplace=True)
mwestrd4b = mwestbracketrd4.iloc[rd4b].reset_index()
mwestrd4b.drop(['index'],axis=1,inplace=True)
mwestrd4 = pd.merge(mwestrd4a, mwestrd4b, how='inner',left_index=True,right_index=True)
mwestrd4 = mwestrd4.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
mwestrd4 = mwestrd4.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
mwestrd4.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
mwestrd4['winner'] = mwestrd4[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)


###South

rd1a = [0,2,4,6,8,10,12,14]
rd1b = [1,3,5,7,9,11,13,15]

southrd1a = southbracket.iloc[rd1a].reset_index()
southrd1a.drop(['index'],axis=1,inplace=True)
southrd1b = southbracket.iloc[rd1b].reset_index()
southrd1b.drop(['index'],axis=1,inplace=True)
southrd1 = pd.merge(southrd1a, southrd1b, how='inner',left_index=True,right_index=True)
southrd1 = southrd1.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
southrd1 = southrd1.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
southrd1.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
southrd1['winner'] = southrd1[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

southbracketrd2 = southbracket[southbracket['Team_Id'].isin(southrd1['winner'])]

rd2a = [0,2,4,6]
rd2b = [1,3,5,7]

southrd2a = southbracketrd2.iloc[rd2a].reset_index()
southrd2a.drop(['index'],axis=1,inplace=True)
southrd2b = southbracketrd2.iloc[rd2b].reset_index()
southrd2b.drop(['index'],axis=1,inplace=True)
southrd2 = pd.merge(southrd2a, southrd2b, how='inner',left_index=True,right_index=True)
southrd2 = southrd2.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
southrd2 = southrd2.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
southrd2.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
southrd2['winner'] = southrd2[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

southbracketrd3 = southbracket[southbracket['Team_Id'].isin(southrd2['winner'])]

rd3a = [0,2]
rd3b = [1,3]

southrd3a = southbracketrd3.iloc[rd3a].reset_index()
southrd3a.drop(['index'],axis=1,inplace=True)
southrd3b = southbracketrd3.iloc[rd3b].reset_index()
southrd3b.drop(['index'],axis=1,inplace=True)
southrd3 = pd.merge(southrd3a, southrd3b, how='inner',left_index=True,right_index=True)
southrd3 = southrd3.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
southrd3 = southrd3.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
southrd3.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
southrd3['winner'] = southrd3[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

southbracketrd4 = southbracket[southbracket['Team_Id'].isin(southrd3['winner'])]

rd4a = [0]
rd4b = [1]

southrd4a = southbracketrd4.iloc[rd4a].reset_index()
southrd4a.drop(['index'],axis=1,inplace=True)
southrd4b = southbracketrd4.iloc[rd4b].reset_index()
southrd4b.drop(['index'],axis=1,inplace=True)
southrd4 = pd.merge(southrd4a, southrd4b, how='inner',left_index=True,right_index=True)
southrd4 = southrd4.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
southrd4 = southrd4.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
southrd4.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
southrd4['winner'] = southrd4[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

###Fin4
eastwinner = pd.DataFrame(eastbracket[eastbracket['Team_Id'].isin(eastrd4['winner'])])
westwinner = pd.DataFrame(westbracket[westbracket['Team_Id'].isin(westrd4['winner'])])
mwestwinner = pd.DataFrame(mwestbracket[mwestbracket['Team_Id'].isin(mwestrd4['winner'])])
southwinner = pd.DataFrame(southbracket[southbracket['Team_Id'].isin(southrd4['winner'])])

fin4a = eastwinner
fin4a = fin4a.append(westwinner).reset_index()
fin4a.drop(['index'],axis=1,inplace=True)

rd4a = [0]
rd4b = [1]

fin4a1 = fin4a.iloc[rd4a].reset_index()
fin4a1.drop(['index'],axis=1,inplace=True)
fin4b1 = fin4a.iloc[rd4b].reset_index()
fin4b1.drop(['index'],axis=1,inplace=True)
fin41 = pd.merge(fin4a1, fin4b1, how='inner',left_index=True,right_index=True)
fin41 = fin41.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
fin41 = fin41.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
fin41.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
fin41['winner'] = fin41[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)



fin4b = mwestwinner
fin4b = fin4b.append(southwinner).reset_index()
fin4b.drop(['index'],axis=1,inplace=True)


fin4a2 = fin4b.iloc[rd4a].reset_index()
fin4a2.drop(['index'],axis=1,inplace=True)
fin4b2 = fin4b.iloc[rd4b].reset_index()
fin4b2.drop(['index'],axis=1,inplace=True)
fin42 = pd.merge(fin4a2, fin4b2, how='inner',left_index=True,right_index=True)
fin42 = fin42.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
fin42 = fin42.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
fin42.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
fin42['winner'] = fin42[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)

###Chmp
chmp1 = pd.DataFrame(fin4a[fin4a['Team_Id'].isin(fin41['winner'])])
chmp2 = pd.DataFrame(fin4b[fin4b['Team_Id'].isin(fin42['winner'])])

chmprd = chmp1
chmprd = chmprd.append(chmp2).reset_index()
chmprd.drop(['index'],axis=1,inplace=True)

rd4a = [0]
rd4b = [1]

chmpa = chmprd.iloc[rd4a].reset_index()
chmpa.drop(['index'],axis=1,inplace=True)
chmpb = chmprd.iloc[rd4b].reset_index()
chmpb.drop(['index'],axis=1,inplace=True)
chmp = pd.merge(chmpa, chmpb, how='inner',left_index=True,right_index=True)
chmp = chmp.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_x','Team_Id_y'],right_on=['team','opponent'])
chmp = chmp.merge(sub_all[['team','opponent','pred_prob']], how='left',left_on=['Team_Id_y','Team_Id_x'],right_on=['team','opponent'])
chmp.drop(['team_x','opponent_x','team_y','opponent_y'],axis=1,inplace=True)
chmp['winner'] = chmp[['Team_Id_x','Team_Id_y','pred_prob_x','pred_prob_y']].apply(lambda x: x['Team_Id_x'] if x['pred_prob_x'] > x['pred_prob_y'] else x['Team_Id_y'], axis=1)


#######Plots

##Compare avg. score amount for wins to total avg score amount for wins
purwinpts = regseasgmstats[(regseasgmstats['Team_Id']==1345) & (regseasgmstats['win']==1)].groupby(['Season'])['Wscore'].mean()

allwinpts = regseasgmstats[(regseasgmstats['win']==1)].groupby(['Season'])['Wscore'].mean()


purwinpts.plot()
allwinpts.plot()
plt.legend(['Purdue win avg.', 'Total win avg.'])
plt.show()

##Compare avg. score amount for losses to total avg score amount for losses
purwinpts = regseasgmstats[(regseasgmstats['Team_Id']==1345) & (regseasgmstats['win']==0)].groupby(['Season'])['Lscore'].mean()

allwinpts = regseasgmstats[(regseasgmstats['win']==0)].groupby(['Season'])['Lscore'].mean()


purwinpts.plot()
allwinpts.plot()
plt.legend(['Purdue loss avg.', 'Total loss avg.'])
plt.show()


##Display winscore variation by year using boxplots

allwinptstm = regseasgmstats[regseasgmstats['win']==1].groupby(['Team_Id','Season'])['Wscore'].mean().reset_index()

allwinptstm.boxplot(column='Wscore',by='Season',fontsize=8,figsize=(8,8))
