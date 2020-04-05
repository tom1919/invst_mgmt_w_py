#Configuration File

#Do not change, these are ipython notebook demonstration

#Path for data
dataPath = 'Data/Data_Oct2018_v2.csv'

#Define the factor names
factorName = ['World Equities','US Treasuries','Bond Risk Premium','Inflation Protection','Currency Protection']

#Names of assets
assetName = ['US Equities','Real Estate','Commodities','Corp Bonds']

#Name of date column
dateName = 'Date'

#User Analysis Section.  Change the variables in this section to run user specific analysis

#isDemo is a boolean variable, set to True if the user wants to run custom analysis
isDemo = False

#dataPathUser: Path to User Defined Data
dataPathUser = 'Data/Data_Oct2018_v2.csv'

#factorNameUser: List, defines the factors
factorNameUser= ['World Equities','US Treasuries','Bond Risk Premium','Inflation Protection','Currency Protection']

#assetNameUser: List, defines the asset to be used
assetNameUser = 'Commodities'

#dateName: string, date column
dateNameUser = 'Date'

#Start and End Dates for the Analysis
startDateUser = '2000-01-01'
endDateUser = '2018-09-01'


#Best Subset Regression Related
maxVarsUser = 3

#Elastic Net Related
numL1RatioUser = 10
numLambdasUser = 20
