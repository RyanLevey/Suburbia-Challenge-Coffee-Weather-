
# coding: utf-8

# # Suburbia Challenge
# ## 1st Step: Impory Python Libraries

# In[1]:


import pandas as pd
import sklearn as sk
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().magic('matplotlib inline')


# ## 2nd Step: Import & Read Data

# In[2]:


stock = pd.read_csv('/Users/ryanc/Desktop/Data Science Challenge/Suburbia/stock.csv', delimiter =',', encoding = 'utf-8-sig')
weather_1 = pd.read_csv('/Users/ryanc/Desktop/Data Science Challenge/Suburbia/weather_1.csv', delimiter = ',', encoding = 'utf-8-sig')
retail = pd.read_csv('/Users/ryanc/Desktop/Data Science Challenge/Suburbia/retail_loader.csv', delimiter=',', encoding = 'utf-8-sig')
houseAverage1 = pd.read_csv('/Users/ryanc/Desktop/Data Science Challenge/Suburbia/houseAverage.csv', delimiter = ',', encoding = 'utf-8-sig')
houseAverage2 = pd.read_csv('/Users/ryanc/Desktop/Data Science Challenge/Suburbia/houseAverageM2.csv', delimiter = ',', encoding = 'utf-8-sig')


# In[3]:


weather_1.head()


# In[4]:


retail.head()


# ## Step 3: Begin Data Manipulation

# In[5]:


#For both weather and retail data, set the index to the date in order to match the two data sets. 

weather_1['date'] = pd.to_datetime(weather_1['date'])
weather_1 = weather_1.sort_values(by='date')
weather_1U = weather_1[weather_1['city'] == 'Utrecht, Netherlands']
weather_1U.set_index('date', inplace=True)
weather_1U.head()


# In[6]:


#Set location to Utrecht

retail['date'] = pd.to_datetime(retail['date'])
retail = retail.sort_values(by='date')
retail.set_index('date', inplace=True)
retail = retail[retail['location'] == 'Utrecht']
#retail_ECoffee = retail[retail['name'] == 'Expensive coffee' ]
#retail_ACoffee = retail[retail['name'] == 'Average coffee' ]
#retail_ECoffeeU = retail_ECoffee[retail_ECoffee['location'] == 'Utrecht' ]
retail.head()


# In[7]:


# Put Average and Expensive coffee together in effort to understandthe total amount of coffee being consumed.

retail_ECoffee = retail[retail['name'] == 'Expensive coffee' ]
retail_ACoffee = retail[retail['name'] == 'Average coffee' ]
###### Edit the data frame
retail_ECoffee = retail_ECoffee.drop('_id__$oid',1)
retail_ECoffee = retail_ECoffee.drop('name',1)
retail_ECoffee = retail_ECoffee.drop('location',1)
retail_ACoffee = retail_ACoffee.drop('_id__$oid',1)
retail_ACoffee = retail_ACoffee.drop('name',1)
retail_ACoffee = retail_ACoffee.drop('location',1)

retail_ECoffee = retail_ECoffee.rename(columns={'total_transactions':'E_total_transactions,',
                                               'appearances': 'E_appearances',
                                               'total_transaction_value': 'E_total_transaction_value',
                                               'total_product_value':'E_product_value'})

retail_ACoffee = retail_ACoffee.rename(columns={'total_transactions':'A_total_transactions,',
                                               'appearances': 'A_appearances',
                                               'total_transaction_value': 'A_total_transaction_value',
                                               'total_product_value':'A_product_value'})

rTotalCoffee = retail_ECoffee.join(retail_ACoffee, how='outer')
#retail['total_coffee_transactions'] = retail_ECoffee['total_transactions'] + retail_ACoffee['total_transactions']
#retail['total_Coffee_appearances']
#retail['total_coffee_Trans.Value']
#retail['total_coffee_product_value']
#rfinal = retail_ECoffee.join(retail_ACoffee, how='outer')

rTotalCoffee.head()


# In[8]:


rTotalCoffee['t_total_transaction'] = rTotalCoffee['E_total_transactions,'] + rTotalCoffee['A_total_transactions,']
rTotalCoffee['t_appearances'] = rTotalCoffee['E_appearances'] + rTotalCoffee['A_appearances']
rTotalCoffee['t_total_transaction_value'] = rTotalCoffee['E_total_transaction_value'] + rTotalCoffee['A_total_transaction_value']
rTotalCoffee['t_total_product_value'] = rTotalCoffee['E_product_value'] + rTotalCoffee['E_product_value']

rTotalCoffee.head()


# In[9]:


#Drop Expensive and Average coffee from data frame

rFinalCoffee = rTotalCoffee.drop(rTotalCoffee.columns[[0, 1, 2, 3, 4, 5, 6, 7]], axis=1)

rFinalCoffee.head()


# In[10]:


weather_1U_retail = weather_1U.loc['2017-09-10':'2018-03-01',:]
weather_1U_retail.head()


# In[11]:


weather_1U_retail.shape


# In[12]:


rFinalCoffee.shape


# ## Step 4: Put the Weather and Coffee data together

# In[13]:


final = weather_1U_retail.join(rFinalCoffee, how='outer')

#Drop days that include holidays

final.dropna(how='any').shape
final.head()


# ## Step 5: Make Dummy Variables to provide weather classification (based upon weather conditions)

# In[14]:


#Create Dummy Variables for in order to provide classification

final['coldclear'] = (final['feelsLikeC'] <= 9) & (final['cloudcover'] <= 49)
final['coldcloudy'] = (final['feelsLikeC'] <= 9) & (final['cloudcover'] >= 50)
final['mildclear'] = (final['feelsLikeC'] >= 10) & (final['cloudcover'] <= 49)
final['mildcloudy'] = (final['feelsLikeC'] >= 10) & (final['cloudcover'] >= 50)

#Convert to boolean values to int

final.coldclear = final.coldclear.astype(int)
final.coldcloudy = final.coldcloudy.astype(int)
final.mildclear = final.mildclear.astype(int)
final.mildcloudy = final.mildcloudy.astype(int)

final.tail()


# ## Step 6: Alter dummy variables to use different boolean values (ex. use 2 for true instead of 1) in effort to develop a final classification column for the weather

# In[26]:



final.coldcloudy[final.coldcloudy>0]+=1
final.mildclear[final.mildclear>0]+=2
final.mildcloudy[final.mildcloudy>0]+=3

final['weather'] = final['coldclear'] + final['coldcloudy'] + final['mildclear'] + final['mildcloudy']

final.head()


# # Step 7: Assing each number in the column 'weather' to a string variable. After begin analysis

# In[25]:


final.weather[final.weather==1] = 'cold & clear'
final.weather[final.weather==2]= 'cold & cloudy'
final.weather[final.weather==3]= 'mild & clear'
final.weather[final.weather==4]= 'mild & cloudy'

final = final.drop('coldclear',1)
final = final.drop('coldcloudy',1)
final = final.drop('mildclear',1)
final = final.drop('mildcloudy',1)



final.head()


# # *Attempt at K mean clustering

# In[21]:


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[23]:


cluster = KMeans(n_clusters = 5)
cols = final.columns[1:]
cols


# In[ ]:


cluster = KMeans(n_clusters = 5)


# In[24]:


final['cluster'] = cluster.fit_predict(final[final.columns[2:]])


# # Step 8: Begin Analysis

# In[45]:


sb.lmplot(data=final, x='feelsLikeC', y='t_total_product_value', hue = 'weather', fit_reg=False, size = 7)
sb.set(font_scale=1.5)
plt.title('Cluster of Weather Conditions and Value of Coffee Purchases (Product Value)')


# In[18]:


#For Cold and Clear
sb.pairplot(final, x_vars=['feelsLikeC', 'minTemp'], y_vars='t_appearances', hue='weather', size=7, 
            aspect=0.7)


# In[19]:


sb.pairplot(final, x_vars=['feelsLikeC', 'minTemp', 'maxTemp'], y_vars='t_appearances', hue='weather', size=7, 
            aspect=0.7)


# In[20]:


sb.pairplot(final, x_vars=['feelsLikeC', 'minTemp', 'maxTemp'], y_vars='t_appearances', hue='weather', size=7, 
            aspect=0.7)


# In[21]:


sb.pairplot(final, x_vars=['feelsLikeC', 'minTemp', 'maxTemp'], y_vars='t_appearances', hue='weather', size=7, 
            aspect=0.7)


# In[22]:


sb.pairplot(final, x_vars=['feelsLikeC', 'minTemp', 'maxTemp'], y_vars='t_total_product_value', hue='weather', size=7, 
            aspect=0.7)


# In[23]:


sb.pairplot(final, x_vars=['feelsLikeC', 'minTemp', 'maxTemp'], y_vars='t_total_product_value', hue='weather', size=7, 
            aspect=0.7)


# # Step 9: Conduct linear analysis, begin with data correlations

# In[24]:



print(final.corr())


# In[42]:


sb.pairplot(final, x_vars=['feelsLikeC'], y_vars='t_total_product_value', size=7, 
            aspect=0.7, kind='reg' )
plt.title('Coffee Purchased(Product Value) & Temperature(Feels Like C)')
             


# In[44]:


sb.pairplot(final, x_vars=['cloudcover'], y_vars='t_total_product_value', size=7, 
            aspect=0.7, kind='reg' )
plt.title('Coffee Purchased(Product Value) & Cloud Coverage')


# In[50]:


#Create a list of feature names
feature_cols = ['feelsLikeC', 'minTemp', 'maxTemp']

#Use the list to select a subset of the original df

x = final1[feature_cols]

x.head()


# In[51]:


y = final1['t_appearances']

y = final1.t_appearances

y.head()


# In[52]:


#Split x and y into training and test sets
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=1)


# In[53]:


print (x_train.shape)
print (y_train.shape)
print (x_test.shape)
print (y_test.shape)


# In[54]:


#Import model
from sklearn.linear_model import LinearRegression

#Instantiate
linreg = LinearRegression()

#Fit the model to the training data (learn the coefficiencs)

linreg.fit(x_train, y_train)


# In[55]:


print(linreg.intercept_)
print(linreg.coef_)


# In[56]:


#Pair the feature name with the coefficients
print(zip(feature_cols, linreg.coef_))


# In[57]:


#Make predictions on the testing set
y_pred = linreg.predict(x_test)

from sklearn import metrics
print (np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[54]:


final.plot(y='tempDif', use_index=True)


# In[88]:


final.plot(y='total_product_value', use_index=True)


# In[31]:


sb.regplot(x='feelsLikeC', y='total_product_value', data = final)


# In[32]:


sb.regplot(x='feelsLikeC', y='total_transaction_value', data = final)


# In[35]:


sb.regplot(x='maxTemp', y='total_product_value', data = final)
sb.regplot(x='minTemp', y='total_product_value', data = final)


# In[36]:


sb.regplot(x='maxTemp', y='total_transaction_value', data = final)
sb.regplot(x='minTemp', y='total_transaction_value', data = final)


# In[30]:


sb.regplot(x='cloudcover', y='total_product_value', data = final)


# In[39]:


sb.pairplot(final, x_vars = ['minTemp','maxTemp', 'cloudcover','feelsLikeC'], y_vars = 'total_product_value')


# In[48]:


#Weather Analysis

sb.pairplot(final, x_vars = ['minTemp', 'maxTemp'], y_vars = 'feelsLikeC')


# In[46]:


final.plot(y='appearances', use_index=True)


# In[54]:


#Ammend number of appearances to weather4U_retail
#weather4U_retail[retail_ECoffeeU['appearances']]
#weather4U_retail.head()


x = weather4U_retail['date']
y1 = retail_ECoffeeU['appearances']
y2 = weather4U_retail['cloudcover']

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')

ax1.set_xlabel('Date')
ax1.set_ylabel('Appearances', color='g')
ax2.set_ylabel('Cloudcover', color='b')

plt.show()


# In[126]:


x = weather4U_retail['date']
y1 = retail_ACoffeeU['appearances']
y2 = weather4U_retail['cloudcover']

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')

ax1.set_xlabel('Date')
ax1.set_ylabel('Appearances', color='g')
ax2.set_ylabel('Cloudcover', color='b')

plt.show()


# In[57]:


x = weather4U_retail['date']
y1 = retail_ECoffeeU['appearances']
y2 = weather4U_retail['precipMM']

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')

ax1.set_xlabel('Date')
ax1.set_ylabel('Appearances', color='g')
ax2.set_ylabel('Precipitation', color='b')

plt.show()


# In[115]:


x = retail_ECoffeeU['appearances']
y = weather4U_retail['cloudcover']

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

plt.scatter(x, y)
plt.show()


# In[ ]:


plot(result['weeks'], polyval(p1,result['weeks']),'r-')


# In[80]:


x = retail_ACoffeeU['appearances']
y = weather4U_retail['cloudcover']

plt.scatter(x, y)
plt.show()


# ## Due to the fact that the Expensive Coffee data set has more observations, we can see that it is being sold more often

# In[6]:


retail_ECoffee.shape


# In[119]:


retail_ACoffee.tail()


# In[25]:


retail_ACoffee.head()


# # Next Step: Plot Total Transactions & Product Value

# In[26]:


#For Expensive coffee

sb.pairplot(retail_ECoffee, x_vars = ['total_transactions','total_transaction_value', 'appearances'], y_vars = 'total_product_value')


# In[27]:


sb.pairplot(retail_ECoffee, x_vars = 'total_transactions', y_vars = 'appearances')


# In[12]:


#For Average Coffee

sb.pairplot(retail_ACoffee, x_vars = ['total_transactions','total_transaction_value', 'appearances'], y_vars = 'total_product_value')


# In[13]:


sb.pairplot(retail_ACoffee, x_vars = 'total_transactions', y_vars = 'appearances')

