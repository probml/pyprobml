
# Based on
# https://cloud.coiled.io/examples/notebooks
#https://cloud.coiled.io/examples/jobs/hyperband-optimization

##### Setup
import coiled

cluster = coiled.Cluster(
    n_workers=2, #10
    software="examples/hyperband-optimization",
)



# Connect Dask to the cluster
import dask.distributed

client = dask.distributed.Client(cluster)
client


# Send module with HiddenLayerNet to workers on cluster
client.upload_file("torch_model.py")

### #Data

import dask.dataframe as dd

features = ["passenger_count", "trip_distance", "fare_amount"]
categorical_features = ["RatecodeID", "payment_type"]
output = ["tpep_pickup_datetime", "tpep_dropoff_datetime"]

#"s3://nyc-tlc/trip data/yellow_tripdata_2019-*.csv", 

df = dd.read_csv(
    "s3://nyc-tlc/trip data/yellow_tripdata_2019-01.csv", 
    parse_dates=output,
    usecols=features + categorical_features + output,
    dtype={
        "passenger_count": "UInt8",
        "RatecodeID": "category",
        "payment_type": "category",
    },
    blocksize="16 MiB",
    storage_options={"anon": True},
)  #.head(n=1000)

print(df.columns)
print(len(df)) # 7,667,792

#storage_options={'key': settings.AWS_ACCESS_KEY_ID,
 #                'secret': settings.AWS_SECRET_ACCESS_KEY})

df = df.repartition(partition_size="10 MiB").persist()

# one hot encode the categorical columns
df = df.categorize(categorical_features)
df = dd.get_dummies(df, columns=categorical_features)

# persist so only download once
df = df.persist()

data = df[[c for c in df.columns if c not in output]]
data = data.fillna(0)


durations = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60  # minutes

from dask_ml.model_selection import train_test_split
import dask

X = data.to_dask_array(lengths=True).astype("float32")
y = durations.to_dask_array(lengths=True).astype("float32")
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, shuffle=True)

# persist the data so it's not re-computed
X_train, X_test, y_train, y_test = dask.persist(X_train, X_test, y_train, y_test)


#### Model

# Import our HiddenLayerNet pytorch model from a local torch_model.py module
from torch_model import HiddenLayerNet
import torch
import torch.optim as optim
import torch.nn as nn
from skorch import NeuralNetRegressor

niceties = {
    "callbacks": False,
    "warm_start": True,
    "train_split": None,
    "max_epochs": 1,
}

class NonNanLossRegressor(NeuralNetRegressor):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        if torch.abs(y_true - y_pred).abs().mean() > 1e6:
            return torch.tensor([0.0], requires_grad=True)
        return super().get_loss(y_pred, y_true, X=X, training=training)

nfeatures = 14 # X_train.shape[1]
model = NonNanLossRegressor(
    module=HiddenLayerNet,
    module__n_features=nfeatures,
    optimizer=optim.SGD,
    criterion=nn.MSELoss,
    lr=0.0001,
    **niceties,
)


#### Hyper-param search space

from scipy.stats import loguniform, uniform

params = {
    "module__activation": ["relu", "elu", "softsign", "leaky_relu", "rrelu"],
    "batch_size": [32, 64, 128, 256],
    "optimizer__lr": loguniform(1e-4, 1e-3),
    "optimizer__weight_decay": loguniform(1e-6, 1e-3),
    "optimizer__momentum": uniform(0, 1),
    "optimizer__nesterov": [True],
}


params = {
    "module__activation": ["relu", "elu", ],
    "batch_size": [32, 64],
    "optimizer__lr": loguniform(1e-4, 1e-3),
    "optimizer__weight_decay": loguniform(1e-6, 1e-3),
    "optimizer__momentum": uniform(0, 1),
    "optimizer__nesterov": [True],
}


from dask_ml.model_selection import HyperbandSearchCV
search = HyperbandSearchCV(model, params, random_state=2, verbose=True, max_iter=9)

y_train2 = y_train.reshape(-1, 1).persist()
search.fit(X_train, y_train2)


print(search.best_score_)

print(search.best_params_)

print(search.best_estimator_)


from dask_ml.wrappers import ParallelPostFit
deployed_model = ParallelPostFit(search.best_estimator_)
deployed_model.score(X_test, y_test)
 

