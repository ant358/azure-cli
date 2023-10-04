import mlflow

# logging
mlflow.autolog()

reg_rate = 0.1
mlflow.log_param("Regularization rate", reg_rate)

# find metrics
experiments = mlflow.search_experiments(max_results=2)
for exp in experiments:
    print(exp.name)

# include archived experiments
from mlflow.entities import ViewType

experiments = mlflow.search_experiments(view_type=ViewType.ALL)
for exp in experiments:
    print(exp.name)

# get experiment by name
exp = mlflow.get_experiment_by_name(experiment_name)
print(exp)

# get experiment by id
mlflow.search_runs(exp.experiment_id)
