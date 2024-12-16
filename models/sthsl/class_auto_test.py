import torch
import numpy as np
import time
from experimental.sthsl.engine import trainer
from experimental.sthsl.Params import args
from experimental.sthsl.utils import seed_torch, makePrint

from experimental.sthsl.model import *
from experimental.sthsl.DataHandler import DataHandler
from experimental.sthsl.engine import test
from experimental.sthsl.train import makePrint

import math
import GPUtil

import sys
sys.path.append('./experimental/sthsl/')
# import grid
import pandas as pd
# import geopandas as gpd
import pickle
# from tqdm import tqdm
from shapely.geometry import Point
# from shapely.geometry import Polygon
# import matplotlib.pyplot as plt
# from datetime import timedelta
# import evaluation
#TODO: tirar os imports inuteis

PATH = './experimental/sthsl'
class STHSLModel:
    def __init__(self, points, grid, last_train_date, last_test_date, temporal_granularity, batch = 8):
        self.grid_obj = grid
        self.x = max(self.grid_obj['ycell']) + 1
        self.y = max(self.grid_obj['xcell']) + 1
        self.df = points

        self.batch = batch
        self.last_train_day = last_train_date
        self.last_test_day = last_test_date
        a = pd.to_datetime(last_train_date).normalize()
        b = pd.to_datetime(last_test_date).normalize()
        self.days_on_test = (b - a).days

        self.num_periods = int(temporal_granularity[:-1]) if temporal_granularity[-1] != 'D' else 24
        self.num_periods = 24 // self.num_periods

        self.day_granularity = int(temporal_granularity[:-1]) if temporal_granularity[-1] == 'D' else 1
        self.temporal_granularity = temporal_granularity

        self.run_prep()
#===================================================================================================
# ======================================== DATA PREPARATION ========================================
    def create_arrays(self):
        self.df['data_hora_fato'] = pd.to_datetime(self.df['data_hora_fato'])
        # Get the minimum and maximum dates
        min_date = self.df['data_hora_fato'].min().normalize()  # normalize to get date without time
        max_date = self.df['data_hora_fato'].max().normalize() + pd.Timedelta(days=1)  # +1 to include the end of the last day
        # Calculate the number of days
        num_days = math.ceil((max_date - min_date).days / self.day_granularity)
        # num_days = (max_date - min_date).days
        print(min_date)
        print(max_date)
        print(max_date - min_date)
        print("Num days:", num_days)
        # Number of periods
        N = num_days * self.num_periods
        # Create an array of zeros
        array = np.zeros((self.x, self.y, N), dtype=int)
        # Define periods within a day

        if self.num_periods == 1:
            periods = [(pd.Timestamp("00:00:00").time(), pd.Timestamp("23:59:59").time())]
        else:
            period_duration = 24 // self.num_periods
            periods = []
            for i in range(self.num_periods):
                start_time = pd.Timestamp(i * period_duration, unit='h').time()
                end_time = pd.Timestamp((i + 1) * period_duration - 1, unit='h', microsecond=999999).time()
                periods.append((start_time, end_time))

        # periods = [
        #    (pd.Timestamp("00:00:00").time(), pd.Timestamp("07:59:59").time()),
        #    (pd.Timestamp("08:00:00").time(), pd.Timestamp("15:59:59").time()),
        #    (pd.Timestamp("16:00:00").time(), pd.Timestamp("23:59:59").time())
        # ]
        if self.grid_obj.crs.to_epsg() != 4326:
            gdf = self.grid_obj.to_crs(epsg=4326)


        for index, row in self.df.iterrows():
            point = Point(row['numero_longitude'], row['numero_latitude'])
            # Find the polygon containing this point
            matching_row = gdf.contains(point)
            true_indices = list(matching_row[matching_row == True].index)
            if true_indices:
                local = true_indices[0]
                a = local // self.y
                b = local % self.y

                # Extract date and time
                date = row['data_hora_fato'].normalize()  # normalized date (yyyy-mm-dd 00:00:00)
                time = row['data_hora_fato'].time()  # time (hh:mm:ss)

                # Calculate the day index
                day_index = (date - min_date).days // self.day_granularity
                # day_index = (date - min_date).days

                # Determine the period index
                period_in_day = 0
                for period_index, (start, end) in enumerate(periods):
                   if start <= time <= end:
                       period_in_day = period_index
                       break

                # Calculate the overall period index
                period_index = day_index * self.num_periods + period_in_day
                array[a, b, period_index] += 1
                # array[a, b, day_index] += 1

        # To guarantee that model run with less data
        # if self.temporal_granularity[-1] == 'D' and self.day_granularity > 1:
        if self.temporal_granularity[-1] == 'D':
            res1 = array
            res2 = np.ones((self.x, self.y, math.ceil(self.days_on_test/self.day_granularity) * self.num_periods), dtype = int)
            res3 = np.ones((self.x, self.y, math.ceil(self.days_on_test/self.day_granularity) * self.num_periods), dtype = int)

            return np.concatenate((res1, res1),axis=2), res2, res3 

        return array, np.ones((self.x, self.y, self.days_on_test * self.num_periods), dtype = int), np.ones((self.x, self.y, self.days_on_test * self.num_periods), dtype = int)
        # return array, np.ones((self.x, self.y, math.ceil(self.days_on_test/7) * self.num_periods), dtype = int), np.ones((self.x, self.y, math.ceil(self.days_on_test/7) * self.num_periods), dtype = int)

    def save_arrays(self, train, validation, test, print_info = False):

        full_train = np.repeat(train[:, :, :, np.newaxis], repeats=4, axis=-1)
        full_val = np.repeat(validation[:, :, :, np.newaxis], repeats=4, axis=-1)
        full_test = np.repeat(test[:, :, :, np.newaxis], repeats=4, axis=-1)

        with open(f'{PATH}/data/BH_crime/auto_test/trn.pkl', 'wb') as fs:
                pickle.dump(full_train, fs)
        with open(f'{PATH}/data/BH_crime/auto_test/val.pkl', 'wb') as fs:
                pickle.dump(full_val, fs)
        with open(f'{PATH}/data/BH_crime/auto_test/tst.pkl', 'wb') as fs:
                pickle.dump(full_test, fs)

        if print_info:
            print("treino unique:", np.unique(full_train))
            print("val unique:", np.unique(full_val))
            print("teste unique:", np.unique(full_test))
            print("Treino shape:", full_train.shape)
            print("Validation shape:", full_val.shape)
            print("Teste shape:", full_test.shape)

    def run_prep(self):
        train, validation, test = self.create_arrays()
        self.save_arrays(train, validation, test, True)
#===================================================================================================
#============================================== TRAIN ==============================================
    def run_train(self):
        seed_torch()
        device = torch.device(args.device)
        engine = trainer(device, self.batch, True)
        print("start training...", flush=True)
        train_time = []
        bestRes = None
        eval_bestRes = dict()
        eval_bestRes['RMSE'], eval_bestRes['MAE'], eval_bestRes['MAPE'] = 1e6, 1e6, 1e6
        update = False

        for i in range(1, args.epoch+1):
            t1 = time.time()
            metrics, metrics1 = engine.train()
            print(f'Epoch {i:2d} Training Time {time.time() - t1:.3f}s')
            ret = 'Epoch %d/%d, %s %.4f,  %s %.4f' % (i, args.epoch, 'Train Loss = ', metrics, 'preLoss = ', metrics1)
            print(ret)

            test = (i % args.tstEpoch == 0)
            if test:
                res_eval = engine.eval(True, True)
                val_metrics = res_eval['RMSE'] + res_eval['MAE']
                val_best_metrics = eval_bestRes['RMSE'] + eval_bestRes['MAE']
                if (val_metrics) < (val_best_metrics):
                    print('%s %.4f, %s %.4f' % ('Val metrics decrease from', val_best_metrics, 'to', val_metrics))
                    eval_bestRes['RMSE'] = res_eval['RMSE']
                    eval_bestRes['MAE'] = res_eval['MAE']
                    update = True
                reses = engine.eval(False, True)
                # torch.save(engine.model.state_dict(), f"{PATH}/save/BH/sthsl_model.pth")
                # self.latest_save = f"{PATH}/save/BH/sthsl_model.pth"
                self.model = engine.model
                if update:
                    # print(makePrint('Test', i, reses))
                    bestRes = reses
                    update = False
            print()
            t2 = time.time()
            train_time.append(t2-t1)
        # print(makePrint('Best', args.epoch, bestRes))
    
    # def set_gpu_with_most_free_memory(self):
    #     # Obtém a lista de todas as GPUs disponíveis
    #     gpus = GPUtil.getGPUs()

    #     if not gpus:
    #         print("Nenhuma GPU disponível.")
    #         return

    #     # Encontra a GPU com mais memória livre
    #     gpus = GPUtil.getGPUs()
    #     least_used_gpu = min(gpus, key=lambda gpu: gpu.memoryUsed)

    #     print(f"Selecionando GPU com mais memória livre: GPU {least_used_gpu}")

    #     # Configura a GPU escolhida para ser usada pelo PyTorch
    #     if torch.cuda.is_available():
    #         torch.cuda.set_device(least_used_gpu)
    #         os.environ["CUDA_VISIBLE_DEVICES"] = str(least_used_gpu.id)
    #         print(f"PyTorch está usando a GPU {least_used_gpu}")
    #     else:
    #         print("CUDA não está disponível no PyTorch.")

    def train(self):
        t1 = time.time()
        # self.set_gpu_with_most_free_memory()
        self.run_train()
        t2 = time.time()
        print("Total time spent: {:.4f}".format(t2 - t1))
#===================================================================================================
#============================================== TEST ===============================================
    def set_latest_save(self, file_name = f'{PATH}/save/BH/sthsl_model.pth'):
        self.latest_save = file_name

    def test(self, num_batches):
        self.set_latest_save()
        device = torch.device(args.device)
        handler = DataHandler(batch = num_batches, autotest = True)
        # model = STHSL()
        model = self.model
        model.to(device)
        # model.load_state_dict(torch.load(self.latest_save))
        model.eval()
        print('model load successfully')

        with torch.no_grad():
            reses, out  = test(model, handler)
            self.model_outputs = out

        print(makePrint('Best', args.epoch, reses))

    def predict(self, date, period = 0):

        period = pd.to_datetime(date).hour//(24//self.num_periods)
        # print("DATA: ", date)
        
        self.test(1+period)
        # a = pd.to_datetime(self.last_test_day).normalize()
        # b = pd.to_datetime(date).normalize()
        # day = self.days_on_test - (a - b).days - 1
        
        a = pd.to_datetime(self.last_train_day).normalize()
        b = pd.to_datetime(date).normalize()
        # day = (b-a).days - 1 
        day = ((b-a).days - 1)//self.day_granularity 
        
        result = np.mean(self.model_outputs[day][period], axis = 1).reshape(self.x, self.y)
        return result

# grid_size = 50000
# csv_path = 'data/BH_crime/regiao_metropolitana_furtos_and_violents.csv'
# target_crime = 'FURTO'

# grid_obj = grid.create_grid(grid_size)
# csv_data = pd.read_csv(csv_path, sep=';')
# filtered_data = csv_data[csv_data['type'] == target_crime]

# sth = SthslTesting(grid_obj, filtered_data, '05/30/2024', '06/07/2024', 4)
# sth.run_prep()
# sth.run_train()
# sth.set_latest_save("save/BH/epoch8.pth")
# res = sth.predict('06/07/2024')
