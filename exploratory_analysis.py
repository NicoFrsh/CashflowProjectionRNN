# Exploratory Analysis
from cProfile import label
import pandas as pd
import numpy as np
import config
import matplotlib.pyplot as plt
import seaborn as sns
import data_preprocessing

# Set plot font
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
            'font.size' : 20,
            'font.family' : 'lmodern'}
plt.rcParams.update(params)

input = pd.read_csv(config.PATH_SCENARIO, skiprows=6)
output = pd.read_csv(config.PATH_OUTPUT)

# Adjust discount functions
input = input[input['Zeit'] != config.PROJECTION_TIME + 1]
discount_vector = data_preprocessing.generate_discount_vector(input)
# input['Diskontfunktion'] = discount_vector
# input['Diskontfunktion'] = input['Diskontfunktion'].astype('float')
# print('Diskont head: ', input['Diskontfunktion'].head(5))

net_profits = output[output['Variable'] == 'net profit']
net_profits_df = net_profits.copy()
net_profits_df = net_profits_df.iloc[:,:config.PROJECTION_TIME+3]
gross_surplus = output[output['Variable'] == 'gross surplus']
net_profits = net_profits.iloc[:, 3:config.PROJECTION_TIME+3]
print('net_profits:')
print(net_profits.head(5))
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

# plt.figure(3)
# # plt.boxplot(net_profits, )
# sns.boxplot(data=net_profits, y=)
# plt.title("Net Profit - Boxplot")

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

# Boxplot before discounting
plt.figure(99)
plt.boxplot(discounted_np, whis=[5,95], sym='')
plt.title("Boxplot Net Profits")

# Compute Mean, Max, Min, Std etc
# net_profits_mean = np.mean(net_profits_array)
# net_profits_median = np.median(net_profits_array)
# net_profits_min = np.min(net_profits_array)
# net_profits_max = np.max(net_profits_array)
# net_profits_std = np.std(net_profits_array)

net_profits_mean = np.mean(discounted_np)
net_profits_median = np.median(discounted_np)
net_profits_min = np.min(discounted_np)
net_profits_max = np.max(discounted_np)
net_profits_std = np.std(discounted_np)

print('Exploratory Analysis Net Profits:')
print("Min: \t\t", net_profits_min / 10e6, " Mio.")
print("Max: \t\t", net_profits_max)
print("Mean: \t\t", net_profits_mean)
print("Median: \t\t", net_profits_median)
print("Std: \t\t", net_profits_std)
print("Zero count: \t\t", np.count_nonzero(discounted_np==0))
# plt.figure(7)
# sns.lineplot(data=net_profits_df,x='Zeit', y='value', ci=95)
# plt.ylabel('Net Profit')
# plt.xticks([0,10,20,30,40,50,60])
# plt.xlabel('t')
# plt.tight_layout()
# plt.savefig('./plots/net_profit.pdf')

# plt.figure(15)
# sns.lineplot(data=output, x=gross_surplus)


# Same with some inputs
# print(input.head(3))
# plt.figure(8)
# sns.lineplot(data=input, x='Zeit', y='Diskontfunktion', label='Diskontfunktion')
# sns.lineplot(data=input, x='Zeit', y='Aktien', label='Aktien')
# sns.lineplot(data=input, x='Zeit', y='Dividenden', label='Dividenden')
# sns.lineplot(data=input, x='Zeit', y='Immobilien', label='Immobilien')
# sns.lineplot(data=input, x='Zeit', y='Mieten', label='Mieten')
# # sns.lineplot(data=input, x='Zeit', y='10j Spotrate fuer ZZR', label='10j Spotrate')
# plt.ylabel('')
# plt.xlabel('t')
# plt.xticks([0,10,20,30,40,50,60])
# plt.legend()
# plt.tight_layout()
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

# plt.figure(9, figsize=[8,4.8])
# sns.lineplot(data=input, x='Zeit', y='1', label='Spot 1')
# sns.lineplot(data=input, x='Zeit', y='3', label='Spot 3')
# sns.lineplot(data=input, x='Zeit', y='5', label='Spot 5')
# sns.lineplot(data=input, x='Zeit', y='10', label='Spot 10')
# sns.lineplot(data=input, x='Zeit', y='15', label='Spot 15')
# sns.lineplot(data=input, x='Zeit', y='20', label='Spot 20')
# sns.lineplot(data=input, x='Zeit', y='30', label='Spot 30')
# sns.lineplot(data=input, x='Zeit', y='10j Spotrate fuer ZZR', label='10j Spotrate')
# plt.xlabel('t')
# plt.xticks([0,10,20,30,40,50,60])
# plt.ylabel('')
# # plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 12})
# plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.tight_layout()
# plt.savefig('./plots/spot_rates.pdf')



plt.show()