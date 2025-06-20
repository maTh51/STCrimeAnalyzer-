import pandas as pd

class DatabaseConnection:
    
    def __init__(self, credentials_filepath: str = None):
        if credentials_filepath:
            with open(credentials_filepath, 'r') as f:
                credentials = f.read().split('\n')
                self.user = credentials[0]
                self.pwd  = credentials[1]

    def get_data_all(self, online=False):
        if online:
            # Code for connecting in some Database
            # ...

            return
        else:
            df = pd.read_csv('/home/yan/IC-CrimeAnalytics-modified/API/experimental/dados_BH.csv')
            return df