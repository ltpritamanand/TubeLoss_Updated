import torch
import torch.nn as nn
import lightning as L
from src.loss import *
from src.metrics import *
from tqdm import tqdm

def objective_function(coverage, width, target_coverage):
    if type(coverage) != torch.Tensor:
        coverage = torch.tensor(coverage)
    
    if torch.abs((coverage - target_coverage)) > target_coverage*2.5/100:
        obj= torch.abs((coverage - target_coverage))*100
    else:
       obj = width
    return obj

class TubeLossModel(L.pytorch.LightningModule):
    """ 
    Neural network model for conformal prediction using Tube Loss.
    
    The Tube Loss implements a piecewise loss function that directly optimizes
    for both coverage and efficiency of prediction intervals.
    """
    def __init__(self,
                 coverage=0.9,
                 x_shape=1,
                 hidden_size=64,
                 dropout=0.1,
                 lr=0.0005,
                 penalty=0,
                 r=0.5,
                 delta=0.03):
        """ 
        Initialize the Tube Loss model.

        Parameters
        ----------
        coverage : float, default=0.9
            Target coverage level (e.g., 0.9 for 90% prediction intervals)
        x_shape : int, default=1
            Input feature dimension
        hidden_size : int, default=64
            Hidden layer dimension
        dropout : float, default=0.1
            Dropout rate for regularization
        lr : float, default=0.0005
            Learning rate for optimization
        penalty : float, default=0
            Additional penalty term (legacy parameter)
        r : float, default=0.5
            Tube movement factor
        delta : float, default=0.03
            Recalibration coefficient on interval width
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.in_shape = x_shape
        self.out_shape = 2  # Two outputs: lower and upper bounds
        self.dropout = dropout
        self.lr = lr
        self.coverage = coverage
        self.quantiles = [(1-coverage)/2, 1-(1-coverage)/2]
        
        # Initialize Tube Loss function
        self.loss_fn = dict_loss["TUBE"](coverage, penalty=penalty, r=r, delta=delta)
        self.build_model()
        self.init_weights()

    def build_model(self):
        """ Construct the network
        """
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.out_shape),
        )

    def init_weights(self):
        """ Initialize the network parameters using orthogonal initialization
        """
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """ Run forward pass
        """
        return self.base_model(x)
    
    
    def step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        loss = self.loss_fn(y_hat, y)
        
        metrics = compute_metrics(y_hat, y[:,0])

        return loss, metrics
    
    def training_step(self, batch, batch_idx):
        """ Training step
        """
        loss, metrics = self.step(batch, batch_idx)
        
        self.log("train_loss", loss)
        self.log("train_coverage", metrics["coverage"])
        self.log("train_width", metrics["interval_width"])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """ Validation step
        """

        loss, metrics = self.step(batch, batch_idx)
        
        self.log("val_loss", loss,sync_dist=True)
        self.log("val_coverage", metrics["coverage"],sync_dist=True)
        self.log("val_width", metrics["interval_width"],sync_dist=True)
        self.log("val_objective", objective_function(metrics["coverage"], metrics["interval_width"], self.coverage),sync_dist=True)
        
        return loss
    
    def configure_optimizers(self):
        """ Configure optimizer
        """
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=self.lr)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        """ Test step
        """
        loss, metrics = self.step(batch, batch_idx)
        
        self.log("test_loss", loss,sync_dist=True)        
        self.log("test_coverage", metrics["coverage"],sync_dist=True)
        self.log("test_width", metrics["interval_width"],sync_dist=True)
                
        return loss
    
    def predict_step(self, batch, batch_idx):
        """ Prediction step
        """
        x, y = batch
        y_hat = self(x)
        return y_hat
    



