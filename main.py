import yaml
from database import DatabaseConnection
from util import pre_process
from count_model import StaticCountModel

with open('./experimental/config.yaml', 'r') as file:
    config = yaml.safe_load(file)
dbconfig = config['database']
mdconfig = config['models']
evconfig = config['evaluation']

# df = DatabaseConnection().get_data_all(dbconfig['columns'], dbconfig['filters'])

# points = pre_process(df, dbconfig['filters']['nome_municipio'], dbconfig['columns'])

print(mdconfig.items())