#%%
import os
import sys
import yaml
import warnings
import pandas as pd
import time 
import psutil
import gpustat

from common.utils import pre_process, train_test_split
from common.grid import create_grid
from common.evaluation import EvaluationModel

from models.sthsl.class_auto_test import STHSLModel
from models.stkde_model import STKDEModel
from models.starima.starima import STARIMA
from models.regressions.regressions import REGRESSIONS

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["EXPORT_CUDA"] = "<1>"

# %%
if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    # %%
    # python3 experimental_script.py <config_file> <save_file>
    path_config = sys.argv[1] if len(sys.argv) > 1 else './scripts/config-starima.yaml'
    path_save = sys.argv[2] if len(sys.argv) > 1 else './scripts/results/grid_size'

    # Load config file
    with open(path_config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    for m, params in config['models'].items():
        path_save += f'_{m}'
    path_save += '.csv'

    results = list()


    # ---------------- Variable parameters ----------------------

    grid_size_lst = [1000]
    data_volume = 10
    steps = 1
    temporal_granularity = ['1D']

    start_date = '2023-10-01'
    end_date = '2024-03-01'
    train_end_date = '2024-01-31'
    test_end_date = '2024-02-29'

    config['database']['filters']['start_date'] = start_date
    config['database']['filters']['end_date'] = end_date
    config['evaluation']['train_end_date'] = train_end_date
    config['evaluation']['test_end_date'] = test_end_date

    train_start_date = ['2024-06-01','2024-04-01','2024-01-01','2023-07-01','2022-07-01']
    train_days_lst = [30, 90, 180, 360, 720]
    test_lst = list(zip(train_start_date, train_days_lst))
    
    config['models']['STKDEModel']['slide_window'] = True


    # -----------------------------------------------------------------

    for _ in range(steps):

        tmp = pd.DataFrame()

        for grid_size in grid_size_lst:

            # Variables that need to iterate over loop
            config['evaluation']['grid_size'] = grid_size
            config['evaluation']['temporal_granularity'] = temporal_granularity
            train_dt, train_days = test_lst[0]
            config['database']['filters']['start_date'] = train_dt

            # Load config file
            with open(path_config) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            #Load database
            print('Load Database...')
            df = DatabaseConnection().get_data_all(
                column=config['database']['columns'], 
                filter=config['database']['filters'])

            #Pre-process
            print('Pre process Database...')
            points = pre_process(data=df, 
                                neighborhood=config['database']['filters']['nome_municipio'], 
                                columns=config['database']['columns'])
            
            train_points, test_points = train_test_split(points=points,
                                                        train_end_date=config['evaluation']['train_end_date'],
                                                        test_end_date=config['evaluation']['test_end_date'])

            print(f'Train with {grid_size} grid size...')


            #Create Grid
            print('Create Grid...')
            grid = create_grid(grid_size=config['evaluation']['grid_size'], 
                            municipalities=config['database']['filters']['nome_municipio'])

            train_points = train_points.sample(int(train_points.shape[0]*data_volume/10))

            #Define start status
            time_start = time.time()
            gpu_stats_start = gpustat.GPUStatCollection.new_query()
            memory_usage_start = psutil.virtual_memory()


            #Instance and train models
            print('Train Models...')
            models = []
            for m, params in config['models'].items():
                print(f'Train {m}...')
                if params and 'slide_window' in params.keys():
                    del params['slide_window']
                    model = globals()[m](
                        points=points, 
                        grid = grid, 
                        last_train_date = config['evaluation']['train_end_date'], 
                        temporal_granularity=config['evaluation']['temporal_granularity'], 
                        data_volume=data_volume/10, **params,
                        **params
                    )
                else:
                    model = globals()[m](
                        points=train_points, 
                        grid = grid, 
                        last_train_date = config['evaluation']['train_end_date'],
                        temporal_granularity=config['evaluation']['temporal_granularity'], 
                        **params
                    )
                model.train()
                models.append(model)
            # %%
            #Evaluation
            print('Evaluation Models...')
            eval = EvaluationModel(models = models, 
                                points = test_points, 
                                grid = grid, 
                                start_date = config['evaluation']['train_end_date'],
                                end_date = config['evaluation']['test_end_date'],
                                temporal_granularity = config['evaluation']['temporal_granularity'])
            res = eval.simulate(hit_rate_percentage=config['evaluation']['hit_rate_percentage'])

            res['grid_size'] = grid_size
            res['data_volume'] = data_volume
            res['temporal_window'] = train_dt

            #Define end status
            time_end = time.time()
            # gpu_stats_end = gpustat.GPUStatCollection.new_query()
            # memory_usage_end = psutil.virtual_memory()

            
            res['time'] = time_end - time_start
            # res['memory_usage(%)'] = memory_usage_end.percent - memory_usage_start.percent
            # res['memory_usage(GB)'] = res['memory_usage(%)'] * 0.62
            # for gpu_start, gpu_end in zip(gpu_stats_start.gpus, gpu_stats_end):
            #     res[f'gpu{gpu_start.index}(%)'] =  gpu_end.utilization - gpu_start.utilization
            #     res[f'gpu{gpu_start.index}(GB)'] =  res[f'gpu{gpu_start.index}(%)']*0.24

            tmp = pd.concat([tmp, res])
        
        tmp = tmp.groupby(['grid_size', 'data_volume', 'temporal_granularity', 'temporal_window', 'model']).mean().reset_index()
        results.append(tmp)

    csv = pd.DataFrame()

    for res in results:
        csv = pd.concat([csv, res])

    csv.to_csv(path_save, index=False)