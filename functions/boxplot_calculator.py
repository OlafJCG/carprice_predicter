def iqr_calc (df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    return (Q3 + (1.5*IQR)), (Q1 - (1.5*IQR))