import numpy as np
import tensorflow as tf
from model import build_model, build_linear_model, build_naive_model
from training import test_model
from typing import List, Tuple, Optional, Dict, Any


class GLDNETModel:
    def __init__(self, adjacency_matrix: np.ndarray, metrics: List[str],
    loss_rho: float = 0.01, num_gated_relu_layers: int = 3,**kwargs: Dict[str, Any]
    ) -> None:
        """
        Inicializa o modelo GLDNet com as configurações fornecidas.
        """
        
        self.adjacency_matrix = adjacency_matrix
        self.metrics = metrics
        self.loss_rho = loss_rho
        self.num_gated_relu_layers = num_gated_relu_layers
        self.kwargs = kwargs
        self.epochs = kwargs.get('epochs', 15)
        self.training_data = kwargs.get('training_data', None)
        self.validation_data = kwargs.get('validation_data', None)
        self.data = kwargs.get('data', None)
        self.batch_size = kwargs.get('batch_size', 32)
        self.lookback_steps = kwargs.get('lookback_steps', 7)
        self.num_nodes = kwargs.get('num_nodes', 766)
        self.num_features = kwargs.get('num_features', 3)
        self.total_num_timesteps = kwargs.get('total_num_timesteps', 3225)
        self.model = self.model()
    
    def model(self) -> model:
        """
        Constrói o modelo GLDNet com a arquitetura definida.
        """
        model = build_model(self.adjacency_matrix, self.metrics, 
                        self.loss_rho, self.num_gated_relu_layers, **self.kwargs)
        
        return model
    
    def train(self) -> Dict[str, float]:
        """
        Treina o modelo com os dados de entrada.
        """
        training_history = self.model.fit(self.training_data, 
                    self.epochs, self.validation_data, verbose=0)

        return training_history

    def predict(self) -> np.ndarray:
        """
        Realiza predições no conjunto de teste.
        """
        results, predictions = test_model(self.model, self.data, self.batch_size, self.metrics)
        return results, predictions
