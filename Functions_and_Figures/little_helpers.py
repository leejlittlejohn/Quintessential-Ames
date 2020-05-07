chopping_block = ['PID'] # these are features that will not be in the first iteration of the model and they require no further cleaning.
power_transforms_to_do = {}
features_to_dummy = []

def chop(listofcolumns):
    '''appends columns to chopping block, prints chopping block'''
    chopping_block.extend(listofcolumns)
    print(chopping_block)

def linearity_plotter(x, y, df):
    '''checks numeric features visually for linearity, plots raw, squared, square root, and log of x'''
    import matplotlib.pyplot as plt
    
    fig, [ax1, ax2, ax3, ax4] = plt.subplots(1, 4, sharey=True, figsize=(22, 5))
    ax1.scatter(df[x], df[y])
    ax1.set_title('Not Transformed')
    
    ax2.scatter(df[x]**2, df[y])
    ax2.set_title('Squared')
    
    ax3.scatter(np.sqrt(df[x]), df[y])
    ax3.set_title('Square root')
    
    ax4.scatter(np.log(df[x]), df[y]) # this will frequently throw a zero division warning
    ax4.set_title('Log')
    print(f'Should you update power_transforms_to_do? Dict form {x}: transform')
    
def corr_val(x, y, df):
    '''returns only pearson coef'''
    from scipy.stats import pearsonr
    return pearsonr(df[x], df[y])[0]
        
def cat_compare(x, y, df):
    '''prints value_counts and barplot for x v. y'''
    print(df[x].value_counts())
    sns.barplot(x, y, data=df)
    
def lm_tester(df_subset):
    X = df_subset
    y = df_train['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    lm = LinearRegression()
    lm.fit(X_train, y_train)

    preds = lm.predict(X_test)

    resids = y_test - preds

    print(f'Training R2: {cross_val_score(lm, X_train, y_train).mean()}')
    print(f'Testing R2: {r2_score(y_test, preds)}')
    print(f'RMSE: {np.sqrt(mean_squared_error(y_test, preds))}')
    print(f'Intercept: {lm.intercept_}')

    plt.title('Distribution of Errors')
    plt.hist(resids, bins=20);
    
def submission_gen_lm_tester(df, filename):
    X = df
    y = df_train['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    lm = LinearRegression()
    lm.fit(X_train, y_train)

    X_test_sub = df_test_dum[df.columns.values]
    sub_preds = lm.predict(X_test_sub)

    submission = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': sub_preds,})
    submission.to_csv(f'{filename}.csv', index=False)
    
def lasso_tester(df):
    X = df
    y = df_train['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipe = Pipeline([
        ('var_thresh', VarianceThreshold(threshold = 0.05)), # variance threshold 1st
        ('ss', StandardScaler()), # standard scaler 2nd
        ('kbest', SelectKBest(f_regression, k = 'all')), # k best 3rd
        ('lasso', Lasso()) # fit a lasso model
    ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    resids = y_test - preds
    
    print(f'Training R2: {cross_val_score(pipe, X_train, y_train).mean()}')
    print(f'Testing R2: {pipe.score(X_test, y_test)}')
    print(np.sqrt(mean_squared_error(y_test, preds)))
    
    plt.title('Distribution of Errors')
    plt.hist(resids, bins=20);
    
def ridge_tester(df):
    X = df
    y = df_train['SalePrice']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    pipe = Pipeline([
        ('var_thresh', VarianceThreshold(threshold = 0.05)), # variance threshold 1st
        ('ss', StandardScaler()), # standard scaler 2nd
        ('kbest', SelectKBest(f_regression, k = 'all')), # k best 3rd
        ('ridge', Ridge()) # fit a Ridge model
    ])

    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    resids = y_test - preds
    
    print(f'Training R2: {cross_val_score(pipe, X_train, y_train).mean()}')
    print(f'Testing R2: {pipe.score(X_test, y_test)}')
    print(np.sqrt(mean_squared_error(y_test, preds)))
    
    plt.title('Distribution of Errors')
    plt.hist(resids, bins=20);