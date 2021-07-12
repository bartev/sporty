def lower_case_col_names(df):
    """return the data frame with column names converted to lower case
    :param df:
    :returns:
    :rtype:
    """
    return df.rename(columns=lambda c: (str(c)).lower())


def drop_suffix(self, suffix):
    "drop the suffix from matching data frame columns"

    self.columns = self.columns.str.replace(fr'{suffix}$', '')
    return self
