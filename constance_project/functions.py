import seaborn as sns

def mustache(df, column):
    return sns.boxplot(data=df, x=column).set_title(f'Boîte à moustaches pour les données de la colonne {column}')

def IQR(df, column):
    Q1 = list(df[[column]].quantile(0.25))[0]
    Q3 = list(df[[column]].quantile(0.75))[0]
    IQR = Q3 - Q1
    mini = Q1 - (1.5*IQR)
    maxi = Q3 + (1.5*IQR)
    print(df[[column]].describe().loc[['min', 'max']])
    print(f'\n==> Everything that is less than {round(mini, 2) if round(mini, 2) > 0 else 0} and more than {round(maxi, 2)} is an outlier !!!')
    return mini, maxi

def drop_outliers(df, column, mini, maxi):
    df = df[df[column] < maxi]
    df = df[df[column] > mini]
    df.sample(3)
    print('New min and max values and number of rows :')
    print(df[column].describe().loc[['min', 'max', 'count']])
    return df
