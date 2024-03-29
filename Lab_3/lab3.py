"""Author: Cole BarbesDate Created: 03/29/24Purpose: To analyze a dataset that has an attribute to whether there is fire or no fire \    and tell which attributes correlate most to the fires or no fires"""import statsimport numpy as npimport pandas as pdfrom matplotlib import pyplot as pltdata1 = pd.read_csv('Bejaia_Region.csv')data2 = pd.read_csv('Sidi-Bel_Abbes_Region.csv')# Clean names in dataframes, to remove spacestest = list(data1.columns)data1.columns = [i.replace(' ', '') for i in test]test = list(data2.columns)data2.columns = [i.replace(' ', '') for i in test]# Clean spaces out of the classtest = list(data1['Classes'])data1['Classes'] = [i.replace(' ', '') for i in test]Names = ["Temperature", "RH", "Ws", "Rain"]mean_list1 = []mean_list2 = []for name in Names:    nofire_mean = stats.mean(data1[data1["Classes"] == "notfire"][name].values)    fire_mean = stats.mean(data1[data1["Classes"] == "fire"][name].values)    mean_list1.append(nofire_mean)    mean_list2.append(fire_mean)    output1 = "The mean of the "+name+" Row of the Begaia_Region dataset for when there was no fire is "+str(round(nofire_mean, ndigits=2))    output2 = "The mean of the "+name+" Row of the Begaia_Region dataset for when there was fire is "+str(round(fire_mean, ndigits=2))    print(output1)    print(output2)mean_list = mean_list1+mean_list2print(mean_list)plt.bar(range(len(Names*2)), mean_list, width=0.5)plt.title('More moisture in the air lowers chance of fire')plt.xlabel('No Fire                                                     Fire')plt.ylabel('Mean Value')plt.xticks(ticks=range(len(Names*2)), labels=Names*2)# calculate the median of each feature in the datasetMedian_Names = ['FFMC', 'DMC', 'DC', 'ISI']print()for name in Median_Names:    median = stats.median(data2[name].to_numpy())    output1 = "The median of the "+name+" of the  Sidi-Bel Abbes Region Dataset "+str(median)    print(output1)# end median calculation# calculate the 25%, 60%, 75% quantiles for each attribute of the dataset Begaia_Region datasetvals = [0.25, 0.60, 0.75]print()print("Below is the calculation of the 25%, 60%, 75% quantile ranges of the Begaia Region Dataset:")print("Data is calulated from a subset of features "+str(Names))for name in Names:    for i in vals:        print("The "+str(i*100)+"% quantile for "+name)        print(stats.quantile(data1[name].to_numpy(), i))# end calculation# calculate the standard deviation of the following attributes in the "Sidi-Bel Abbes Region Dataset"sd_names = ["Temperature", "Rain", "BUI", "FWI"]for name in sd_names:    print("The standard deviation of the "+name+" of the Sidi-Bel Abbes Region Dataset is "+str(round(stats.std(data2[name].to_numpy()), ndigits=2)))    # end the sd calculation  of "Bejaia Region Dataset"corr_names = ["Temperature", "Ws", "Rain","FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]corr_target = "RH"print("\nThe correlation coefficients of RH and various other rows")for name in corr_names:    print("The Correlation coefficient of RH and "+name+" is "+str(stats.correlation(data1[corr_target].to_numpy(), data1[name].to_numpy())))#