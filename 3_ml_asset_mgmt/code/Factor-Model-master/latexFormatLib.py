import numpy as np #for numerical array data
import pandas as pd #for tabular data


def write_table(data, nameOfFile, tableName='Name Here', roundNumber = 6):
    '''write_table writes a table into latex readable form
    INPUTS:
        data: pandas df, contains the data
        name: string, name of file
    OUTPUTS:
        file: file where the latex document is'''
    
    with open(nameOfFile, 'w') as file:
        #Write the beginning part of the file
        file.write(r'\begin{table}[ht]')
        file.write('\n')
        file.write('\caption{'+tableName+'}\n')
        file.write('\centering % used for centering table \n')
        file.write(r'\begin{tabular}{')
        for i in range(len(list(data.columns))):
            file.write('c ')
        file.write('c}\n')
        file.write(r'\hline\hline %inserts double horizontal lines')
        file.write('\n')
        for colname in list(data.columns):
            file.write('& ' + str(colname)+' ')
        file.write(r'\\ [0.5ex]')
        file.write('\n')
        #Put in another horizontal line before the data
        file.write(r'\hline % inserts single horizontal line')
        file.write('\n')
        #Now, write the main chunk
        for index in list(data.index):
            file.write(str(index) + ' ')
            for column in list(data.columns):
                if(column == list(data.columns)[-1]):
                    file.write('& '+str(round(data.loc[index, column], roundNumber)))
                else:
                    file.write('& '+str(round(data.loc[index, column], roundNumber))+' ')
            file.write(r'\\')
            if(index == list(data.index)[-1]):
                file.write('[1ex]')
            file.write('\n')

        file.write(r'\hline %inserts single line')
        file.write('\n')
        file.write(r'\end{tabular}')
        file.write('\n')
        file.write(r'\label{table:nonlin} % is used to refer this table in the text')
        file.write('\n')
        file.write(r'\end{table}')



def write_table_fixed(data, nameOfFile, tableName='Name Here', roundNumber = 6):
    '''write_table writes a table into latex readable form
    INPUTS:
        data: pandas df, contains the data
        name: string, name of file
    OUTPUTS:
        file: file where the latex document is'''
    
    with open(nameOfFile, 'w') as file:
        #Write the beginning part of the file
        file.write(r'\begin{table}[ht]')
        file.write('\n')
        file.write('\caption{'+tableName+'}\n')
        file.write('\centering % used for centering table \n')
        file.write(r'\resizebox{\textwidth}{!}{\begin{tabular}{')
        for i in range(len(list(data.columns))):
            file.write('c ')
        file.write('c}\n')
        file.write(r'\hline\hline %inserts double horizontal lines')
        file.write('\n')
        for colname in list(data.columns):
            file.write('& ' + str(colname)+' ')
        file.write(r'\\ [0.5ex]')
        file.write('\n')
        #Put in another horizontal line before the data
        file.write(r'\hline % inserts single horizontal line')
        file.write('\n')
        #Now, write the main chunk
        for index in list(data.index):
            file.write(str(index) + ' ')
            for column in list(data.columns):
                if(column == list(data.columns)[-1]):
                    file.write('& '+str(round(data.loc[index, column], roundNumber)))
                else:
                    file.write('& '+str(round(data.loc[index, column], roundNumber))+' ')
            file.write(r'\\')
            if(index == list(data.index)[-1]):
                file.write('[1ex]')
            file.write('\n')

        file.write(r'\hline %inserts single line')
        file.write('\n')
        file.write(r'\end{tabular}}')
        file.write('\n')
        file.write(r'\label{table:nonlin} % is used to refer this table in the text')
        file.write('\n')
        file.write(r'\end{table}')
