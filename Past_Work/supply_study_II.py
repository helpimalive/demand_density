from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

###
# Importance of Supply In Rent Growth
####

def feature_importance():
df = pd.read_csv(r'C:\Users\mlarriva\ShareFile\Personal Folders\Documents\Data Provider Exports\Costar Exports\2019 Q3\MultiFamQ32019.csv')
df = df[df['as_of']=='2019 Q3']
df = df[df['period']<='2019 Q3']
# df = df.loc[df['slice'].isin(['1 Star','2 Star','3 Star'])]
df = df.loc[df['slice'].isin(['4 Star','5 Star'])]
df = df.groupby(['geography_name','period']).mean().reset_index()
df.sort_values(by = ['geography_name','period'],inplace=True)
df = pd.pivot_table(df,index =['geography_name','period'])
roll_period = 16
df = df.pct_change(roll_period)
df['effective_rent'] = df['effective_rent'].shift(-roll_period)
df = df.reset_index()
df = df.loc[df['period']<='2015 Q3']

subset_vars = [
       'absorption__pct_',
       'average_sale_price', 'average_units_sold', 
       # 'cap_rate',
       'cbsa_code', 'demand', 'existing_buildings', 'households',
       'industrial_employment',
       'median_cap_rate', 'median_household_income',
       'median_price_per_bldg_sf', 'median_price_per_unit', 'net_absorption',
       'net_completions', 'occupancy', 'office_employment',
       'population', 
       'sold_units', 'stock', 'total_employment_',
       'total_sales_volume', 'transaction_cap_rate',
       'transaction_priceperarea', 'under_construction_buildings',
       'under_construction_stock', 'vacancy']


# df = df[subset_vars]
df.replace([np.inf, -np.inf], np.nan,inplace=True)
df.dropna(inplace=True)

X = df[subset_vars]
Y = df['effective_rent']
names = subset_vars
rf = RandomForestRegressor()
rf.fit(X, Y)
print ("Features sorted by their score:")
var_vals = (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), 
             reverse=True))
print(var_vals)

    var_vals = var_vals[0:10]
    y_vals = [x[0] for x in var_vals]
    x_vals = [x[1] for x in var_vals]
    for x in range(0,len(x_vals),2):
        x_vals[x] = '\n '+x_vals[x]
    fig, ax = plt.subplots()
    # ax.yaxis.set_major_formatter(formatter)
    plt.bar(x_vals, y_vals)
    # plt.xticks(x, ('Bill', 'Fred', 'Mary', 'Sue'))
    plt.show()


feature_importance()
# df = pd.read_csv(r'C:\Users\mlarriva\ShareFile\Personal Folders\Documents\Data Provider Exports\Costar Exports\2019 Q3\MultiFamQ32019.csv')
# df = df[df['as_of']=='2019 Q3']
# df = df[df['period']<='2019 Q3']
# df = df.groupby(['geography_name','period']).mean().reset_index()
# df.sort_values(by = ['geography_name','period'],inplace=True)
# df = pd.pivot_table(df,index =['geography_name','period'])
# df['supply_ratio'] = df['population'].div(df['existing_buildings'])
# roll_period = 8
# df['effective_rent'] = df['effective_rent'].pct_change(roll_period)
# df['effective_rent'] = df['effective_rent'].shift(-roll_period)
# df = df.reset_index()
# df = df.loc[df['period']<='2015 Q3']
# df = df.loc[df['population']>5e5]

df = pd.read_csv(r'C:\Users\mlarriva\ShareFile\Personal Folders\Documents\Data Provider Exports\Costar Exports\2019 Q3\MultiFamQ32019.csv')
df = df.loc[df['as_of']=='2019 Q3']
df = df.loc[(df['period']<='2019 Q3') & (df['period']>='2002 Q1')]
df = df.loc[df['slice'].isin(['4 Star','5 Star'])]
df = df.loc[df['population']>=df['population'].mean()]
df = df.groupby(['geography_name','period']).mean().reset_index().copy()
df.sort_values(by = ['geography_name','period'],inplace=True)
df = pd.pivot_table(df,index =['geography_name','period']).copy()

roll_period = 6
df = df.pct_change(roll_period,limit=1).copy()

df['effective_rent'] = df['effective_rent'].shift(-roll_period)
df = df.reset_index()
df = df.loc[df['period']<='2015 Q3']
df = df.loc[df['period']>='2004 Q3']
# df.dropna(inplace=True)
df['existing_buildings'] = df['existing_buildings']
df = df.loc[(df['existing_buildings']>-.1) & (df['existing_buildings']<1)].copy()

plt.scatter(df['existing_buildings'],(1+df['effective_rent'])**(1/(roll_period/4))-1)
plt.title('3-Star Rent Growth vs Building Change')
plt.xlabel('{:0} year change in existing_buildings'.format(int(roll_period/4)))
plt.ylabel('annual rent growth forward {:0} years'.format(int(roll_period/4)))
plt.show()
