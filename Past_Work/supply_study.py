import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR

lag = 4
lookback = 3

df = pd.read_csv(r'C:\Users\mlarriva\ShareFile\Personal Folders\Documents\Data Provider Exports\Costar Exports\2019 Q3\MultiFamQ32019.csv')
fcp = pd.read_csv(r'C:\Users\mlarriva\ShareFile\Personal Folders\Documents\Data Provider Exports\Costar Exports\FCP_markets.csv')

df = df.merge(fcp,how='left',left_on='geography_name',right_on='FCP_markets')
df = df[df['FCP_markets'].notna()]
df = df[df['period']<'2019 Q3']
master_df = df.copy()
msas = master_df['FCP_markets'].unique()
holder = pd.DataFrame(columns = ['market','MAE','fcast','actual'])
ha = 'right'
va = 'bottom'
for msa in msas:
    print(msa)
    df = master_df.copy()
    df = df[df['FCP_markets']==msa]
    df = df[df['slice']=='3 Star']
    df = df.pivot(index = 'period', columns = 'FCP_markets', values = ['households','stock','effective_rent_growth_12_month'])
    df['households'] = df.pct_change(lag)['households']
    df['stock'] = df.pct_change(lag)['stock']
    df['effective_rent_growth_12_month'] = (df['effective_rent_growth_12_month']).rolling(window=lag).apply(np.mean,raw=True)
    df['effective_rent_growth_12_month'] = df['effective_rent_growth_12_month'].shift(-lag).values
    delta = df['households'] - df['stock']
    delta = delta.reset_index().melt(id_vars = 'period')
    df = df.reset_index()
    df = df.melt(id_vars=['period'])
    df.columns = ['period','metric','FCP_markets','value']
    df = df[df['metric']=='effective_rent_growth_12_month'].dropna().reset_index(drop=True)
    df = df.merge(delta,how='left',left_on = ['period','FCP_markets'],right_on =['period','FCP_markets'])
    df.rename(columns={'value_x':'fwd_rent_growth','value_y':'demand_ch'},inplace=True)
    df = df.loc[np.arange(0,len(df),lag)]
    pers = df['period']
    df.drop(['period','metric','FCP_markets'],axis=1,inplace=True)
    df.dropna(inplace=True)

    
    result = adfuller(df['fwd_rent_growth'])
    rent_adf_pval = result[1]
    if rent_adf_pval<0.05:
        explain = 'reject existence of unit root'
    else:
        explain = 'cannot reject existence of unit root'
    print('fwd_rent_growth autocorr = ',rent_adf_pval,' ',explain)
    result = adfuller(df['demand_ch'])
    rent_adf_pval = result[1]
    if rent_adf_pval<0.05:
        explain = 'reject existence of unit root'
    else:
        explain = 'cannot reject existence of unit root'
    print('demand_ch autocorr = ',rent_adf_pval,' ',explain)

    df_model = df.dropna().copy()
    df_model_train = df_model.loc[df_model.index[0:len(df_model)-lookback]]
    df_model_test = df_model.loc[df_model.index[-lookback:]]

    model = VAR(df_model_train)
    results = model.fit()

    resid_autocorr = results.test_whiteness().pvalue
    print('resid autocorr = ',resid_autocorr)
    normality = results.test_normality().summary().data[1][2]
    print('norm = ',normality)
    grainger_cause = results.test_causality('fwd_rent_growth','demand_ch').summary().data[1][2]
    print('grainger cause = ',grainger_cause)

    lag_order = results.k_ar
    test_vals = pd.DataFrame(results.forecast(df_model_train.values[-lag_order:],lookback))
    test_vals.columns = ['fwd_rent_growth','demand_ch']
    test_vals.index = df_model_test.index
    pred_5q_cum_rent = (1+test_vals).prod()['fwd_rent_growth']**(1/lookback)
    mae = (np.mean(np.abs(((1+df_model_test['fwd_rent_growth']).prod()-(1+test_vals['fwd_rent_growth']).prod()))))
    print('MAE = ',mae)
    print(df_model_test['fwd_rent_growth'].values)
    print(test_vals['fwd_rent_growth'].values)
    xvals = pers[-lookback:].apply(lambda x: x+' - '+str(float(x[0:4])+lag/4)[0:4] +' '+ str(x[5:]))

    plt.scatter(xvals,np.abs(df_model_test['fwd_rent_growth']-test_vals['fwd_rent_growth']))
    if ha =='right':
        ha = 'left'
    elif ha =='left':
        ha = 'right'
    
    if va == 'bottom':
        va = 'top'
    elif va == 'top':
        va = 'bottom'
    
    offset = 0
    if msa in ['Atlanta - GA']:
        offset = 7
    if msa in ['Boston - MA']:
        offset = -7
        
    plt.annotate(msa,
                 (xvals.values[-1],
                 np.abs(df_model_test['fwd_rent_growth']-test_vals['fwd_rent_growth']).values[-1]
                 ),  
                 horizontalalignment=ha,
                 size = 9,
                 xytext=(offset, offset), textcoords='offset pixels',
                )

    fcast_cum = (1+test_vals['fwd_rent_growth']).prod()-1
    actuals = (1+df_model_test['fwd_rent_growth']).prod()-1
    holder = holder.append(pd.DataFrame([[msa,mae,fcast_cum,actuals]],columns = holder.columns),ignore_index=True)

holder = holder[['market','fcast','actual','MAE']]
plt.ylabel('<--better      Mean Absolute Error      worse-->')
plt.xlabel('Forecast Period \n All forecasts made Q4 2015')
plt.title('Accuracy of Forecasted Rent Growth \n Using Supply and Demand Changes \n {}-year Mean Abs Err = {:,.2%}'.format(lookback,holder['MAE'].mean()))
plt.show()    

# plt.ylabel('Predicted Cum. 5YR Rent Growth')
# plt.xlabel('Pop Growth * Workforce Demand Quotient')
# plt.title('FCP is in {} of the {} highest predicted growth markets'.format(i,len(X)))
# plt.legend(['FCP Markets','Other Markets'])

    


