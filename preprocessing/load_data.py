import pandas as pd


def load_gop_data(filename):
    """
    loads the GOP data
    Sets the index to the ' id' column
    keeps two colums, 'text' and 'sentiment'
    :param filename:
    :return: pandas.DataFrame
    """
    data_gop = pd.read_csv(filename)
    df_gop = pd.DataFrame(index=data_gop[' id'])
    df_gop = data_gop[['sentiment', 'text']].copy()
    return df_gop

