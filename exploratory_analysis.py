# Exploratory Analysis
import pandas as pd
import numpy as np
import config
import matplotlib.pyplot as plt
import seaborn as sns
import data_preprocessing

input = pd.read_csv(config.PATH_SCENARIO, skiprows=6)
output = pd.read_csv(config.PATH_OUTPUT)

# Adjust discount functions
input = input[input['Zeit'] != config.PROJECTION_TIME + 1]
discount_vector = data_preprocessing.generate_discount_vector(input)
input['Diskontfunktion'] = discount_vector
input['Diskontfunktion'] = input['Diskontfunktion'].astype('float')
print('Diskont head: ', input['Diskontfunktion'].head(5))

net_profits = output[output['Variable'] == 'net profit']
net_profits_df = net_profits.copy()
net_profits_df = net_profits_df.iloc[:,:config.PROJECTION_TIME+3]
gross_surplus = output[output['Variable'] == 'gross surplus']
net_profits = net_profits.iloc[:, 3:config.PROJECTION_TIME+3]
gross_surplus = gross_surplus.iloc[:, 3:config.PROJECTION_TIME+3]
inflow_rfb = output[output['Variable'] == 'inflow RfB']
inflow_rfb = inflow_rfb.iloc[:, 3:config.PROJECTION_TIME+3]

net_profits_std = net_profits.std(axis=0)
net_profits_std = np.array(net_profits_std)
gross_surplus_std = gross_surplus.std()
gross_surplus_std = np.array(gross_surplus_std)
inflow_rfb_std = np.array(inflow_rfb.std())

t = np.arange(start=0, stop=60)
s = np.arange(start=0, stop=10001)
plt.figure(0)
plt.plot(t, net_profits_std, '.', label = 'Net Profit')
# plt.plot(t, gross_surplus_std, 'k.', label = 'Gross Surplus')
plt.legend()
plt.title('Standardabweichung über Zeitschritte')

plt.figure(1)
plt.plot(t, gross_surplus_std, '.', label = 'Gross Surplus')
plt.legend()
plt.title('Standardabweichung über Zeitschritte')

plt.figure(2)
plt.plot(t, inflow_rfb_std, '.', label = 'Inflow RfB')
plt.legend()
plt.title('Standardabweichung über Zeitschritte')

# Vergleiche Verteilung zu festem Zeitschritt mit gesamter Verteilung
net_profits_30 = net_profits.iloc[:,0]
net_profits_scenario = net_profits.iloc[0,:]

# plt.figure(1)
# plt.hist(net_profits_30, bins='auto')
# plt.title('Verteilung Net Profit für festen Zeitschritt')

# plt.figure(2)
# plt.hist(net_profits.iloc[:,::10], bins='auto')
# plt.title('Verteilung Net Profit für jedes 10. Jahr')

# plt.figure(3)
# plt.hist(net_profits_scenario, bins='auto')
# plt.title('Verteilung Net Profit für festes Szenario')

# plt.figure(4)
# plt.hist(net_profits.iloc[::2500,:], bins='auto')
# plt.title('Verteilung Net Profit für jedes 1000. Szenario')

# plt.figure(5)
# plt.plot(t, net_profits_scenario, '--')
# plt.xlabel('Zeitschritt')
# plt.ylabel('Net Profit')
# plt.title('Net Profit über Projektion für Szenario 0 (CE-Pfad)')

# Plot confidence interval
net_profits_df.drop(columns=['Variable'], inplace=True)
net_profits_df = pd.melt(net_profits_df, id_vars=['Simulation'], var_name='Zeit')
net_profits_df['Zeit'] = net_profits_df['Zeit'].astype('int32')
net_profits_df.sort_values(by=['Simulation','Zeit'], inplace=True)

# plt.figure(6)
# sns.lineplot(data=net_profits_df,x='Zeit', y='value', ci=98)

# Discount net profits
discount = input['Diskontfunktion'].to_numpy()
net_profits_array = net_profits_df['value'].to_numpy()
discounted_np = discount * net_profits_array
# net_profits_df['value'] = net_profits_df['value'].astype(float)
# net_profits_df['value'] = net_profits_df['value'] * input['Diskontfunktion']
net_profits_df['value'] = discounted_np

# plt.figure(7)
# sns.lineplot(data=net_profits_df,x='Zeit', y='value', ci=98)
# plt.ylabel('Net Profit')
# plt.savefig('./plots/net_profit.pdf')

plt.figure(15)
sns.lineplot(data=output, x=gross_surplus)


# Same with some inputs
print(input.head(3))
# plt.figure(8)
# sns.lineplot(data=input, x='Zeit', y='Diskontfunktion', label='Diskontfunktion')
# sns.lineplot(data=input, x='Zeit', y='Aktien', label='Aktien')
# # sns.lineplot(data=input, x='Zeit', y='Dividenden', label='Dividenden')
# sns.lineplot(data=input, x='Zeit', y='Immobilien', label='Immobilien')
# # sns.lineplot(data=input, x='Zeit', y='Mieten', label='Mieten')
# # sns.lineplot(data=input, x='Zeit', y='10j Spotrate fuer ZZR', label='10j Spotrate')
# plt.ylabel('')
# plt.legend()
# plt.savefig('./plots/szenarien.pdf')

# plt.figure(10)
# # sns.lineplot(data=input, x='Zeit', y='Diskontfunktion', label='Diskontfunktion')
# # sns.lineplot(data=input, x='Zeit', y='Aktien', label='Aktien')
# sns.lineplot(data=input, x='Zeit', y='Dividenden', label='Dividenden')
# # sns.lineplot(data=input, x='Zeit', y='Immobilien', label='Immobilien')
# sns.lineplot(data=input, x='Zeit', y='Mieten', label='Mieten')
# sns.lineplot(data=input, x='Zeit', y='10j Spotrate fuer ZZR', label='10j Spotrate')
# plt.ylabel('')
# plt.legend()

# plt.figure(9)
# sns.lineplot(data=input, x='Zeit', y='1', label='Spot_1')
# sns.lineplot(data=input, x='Zeit', y='3', label='Spot_3')
# sns.lineplot(data=input, x='Zeit', y='5', label='Spot_5')
# sns.lineplot(data=input, x='Zeit', y='10', label='Spot_10')
# sns.lineplot(data=input, x='Zeit', y='15', label='Spot_15')
# sns.lineplot(data=input, x='Zeit', y='20', label='Spot_20')
# sns.lineplot(data=input, x='Zeit', y='30', label='Spot_30')
# plt.legend()
# plt.savefig('./plots/spot_rates.pdf')



plt.show()