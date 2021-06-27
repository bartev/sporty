def lower_case_col_names(df):
    """return the data frame with column names converted to lower case
    :param df:
    :returns:
    :rtype:
    """
    return df.rename(columns=lambda c: (str(c)).lower())
