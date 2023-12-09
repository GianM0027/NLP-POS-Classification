"""



 /$$$$$$$         /$$$$$$$$                            /$$
| $$__  $$       |__  $$__/                           | $$
| $$  \ $$  /$$$$$$ | $$  /$$$$$$   /$$$$$$   /$$$$$$$| $$$$$$$
| $$  | $$ /$$__  $$| $$ /$$__  $$ /$$__  $$ /$$_____/| $$__  $$
| $$  | $$| $$  \__/| $$| $$  \ $$| $$  \__/| $$      | $$  \ $$
| $$  | $$| $$      | $$| $$  | $$| $$      | $$      | $$  | $$
| $$$$$$$/| $$      | $$|  $$$$$$/| $$      |  $$$$$$$| $$  | $$
|_______/ |__/      |__/ \______/ |__/       \_______/|__/  |__/



"""


from .wrappers import Criterion, OptimizerWrapper
from .metrics import Metric
from .callbacks import EarlyStopper
from .utilities import get_data_loader

import torch

import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm
import itertools
import time

from sre_constants import ANY
from typing import Optional

FIT_PARAMETER_TYPES = torch.utils.data.DataLoader | torch.utils.data.DataLoader | Criterion | list[Metric] | \
                      OptimizerWrapper | int | Optional[EarlyStopper] | bool


def grid_search_train_validation(train_data: tuple[torch.Tensor, torch.Tensor],
                                 val_data: tuple[torch.Tensor, torch.Tensor],
                                 shuffle: bool,
                                 model_hyperparameters_to_test: list[dict[str, ANY]],
                                 training_hyperparameters_to_test: list[dict[str, FIT_PARAMETER_TYPES]],
                                 hyperparameters_key_to_save: list[str],
                                 device: str,
                                 path_to_grid_search_results: str,
                                 n_tests_per_run: Optional[int] = None,
                                 seeds: Optional[list[int]] = None
                                 ) -> pd.DataFrame:
    """
    Perform a grid search train validation over a combination of model and training hyperparameters for a
    deep learning model.

    :param train_data: A tuple containing training data (input and labels).
    :param val_data: A tuple containing validation data (input and labels).
    :param shuffle: A boolean flag indicating whether to shuffle the data during training.
    :param model_hyperparameters_to_test: A list of dictionaries, each specifying model hyperparameters to test.
    :param training_hyperparameters_to_test: A list of dictionaries, each specifying training hyperparameters to test.
    :param hyperparameters_key_to_save: A list of hyperparameter keys to save in the result dataframe.
    :param device: The device (e.g., 'cpu' or 'cuda:0') to run the model training on.
    :param path_to_grid_search_results: path to the file storing the grid search results.
    :param n_tests_per_run: The number of tests to run for each combination of hyperparameters.
    :param seeds: List of the seeds for reproducibility of the results of the grid search.

    :return: A pandas DataFrame containing the results of the grid search, including hyperparameters and performance metrics.

    :Example: path_to_grid_search_results = 'results/grid_search.pkl'

        n_tests_per_run = 3

        optimizers = [OptimizerWrapper(torch.optim.Adam, identifier=f'lr={10**i}', optimizer_partial_params={'lr':10 ** i} for i in range(-2, -5, -1)]
        batch_sizes = [2 ** i for i in range(5, 7)]

        criteria_and_early_stoppers = []
        for i, w in enumerate(weights_strategies):
            criterion = Criterion(f'weighted_cross_entropy_{i}', loss_function=torch.nn.CrossEntropyLoss(reduction='none', weight=w), reduction_function=torch.mean)
            early_stopper = EarlyStopper(monitor='F1_macro', patience=4, delta=0, mode='max', restore_weights=True)
            criteria_and_early_stoppers.append((criterion, early_stopper))

        metrics = [[F1_Score('F1_macro', N_CLASSES, mode='macro', classes_to_exclude=CLASS_INDEXES_TO_EXCLUDE_1)]]


        model_hyperparameters_to_test = [{'model_class': BaselineModel,
                                          'vocabulary': TRAIN_VOCABULARY,
                                          'embedding_dim': EMBEDDING_DIMENSION,
                                          'glove_model_version': GLOVE_MODEL_VERSION,
                                          'high': HIGH,
                                          'low': LOW,
                                          'padding_index': VOCABULARY_LENGTH,
                                          'hidden_size': i,
                                          'output_size': N_CLASSES,
                                          'num_layers': 1,
                                         } for i in range(100, 500, 100)]


        training_hyperparameters_to_test = [{'num_epochs': 200,
                                             'optimizer': p[0],
                                             'batch_size': p[1],
                                             'criterion': p[2][0],
                                             'metrics': p[3],
                                             'early_stopper': p[2][1]
                                             } for p in itertools.product(optimizers, batch_sizes, criteria_and_early_stoppers, metrics)]

        hyperparameters_key_to_save = ['hidden_size', 'num_layers', 'optimizer', 'criterion', 'batch_size']

        grid_search_train_validation(train_data=(torch.tensor(train_sentences_mapped), torch.tensor(train_pos_mapped)),
                                     val_data=(torch.tensor(val_sentences_mapped), torch.tensor(val_pos_mapped)),
                                     shuffle=True,
                                     model_hyperparameters_to_test=model_hyperparameters_to_test,
                                     training_hyperparameters_to_test=training_hyperparameters_to_test,
                                     hyperparameters_key_to_save=hyperparameters_key_to_save,
                                     n_tests_per_run=n_tests_per_run,
                                     device=device,
                                     path_to_grid_search_results=path_to_grid_search_results)

        grid_search_results = joblib.load(path_to_grid_search_results)

    """

    iterator = tqdm(
        iterable=enumerate(itertools.product(training_hyperparameters_to_test, model_hyperparameters_to_test)),
        total=len(training_hyperparameters_to_test) * len(model_hyperparameters_to_test))

    return collect_results(train_data=train_data,
                           val_data=val_data,
                           iterator=iterator,
                           shuffle=shuffle,
                           device=device,
                           hyperparameters_key_to_save=hyperparameters_key_to_save,
                           path_to_results=path_to_grid_search_results,
                           n_tests_per_run=n_tests_per_run,
                           seeds=seeds)


def randomized_search_train_validation(train_data: tuple[torch.Tensor, torch.Tensor],
                                       val_data: tuple[torch.Tensor, torch.Tensor],
                                       shuffle: bool,
                                       model_hyperparameters_to_sample: dict[str, ANY],
                                       training_hyperparameters_to_sample: dict[str, ANY],
                                       hyperparameters_key_to_save: list[str],
                                       n_run: int,
                                       device: str,
                                       path_to_randomized_search_results: str,
                                       n_tests_per_run: Optional[int] = None,
                                       seeds: Optional[list[int]] = None) -> pd.DataFrame:
    """
   Perform randomized hyperparameter search for a deep learning model.

   :param train_data: Tuple containing training input data and labels.
   :param val_data: Tuple containing validation input data and labels.
   :param shuffle: Boolean indicating whether to shuffle the data during training.
   :param model_hyperparameters_to_sample: Dictionary of model hyperparameter names and functions to sample values.
   :param training_hyperparameters_to_sample: Dictionary of training hyperparameter names and functions to sample values.
   :param hyperparameters_key_to_save: List of hyperparameter names to save in the resulting DataFrame.
   :param n_run: Number of runs for the randomized search.
   :param device: Device on which to perform the training (e.g., 'cpu' or 'cuda:0').
   :param path_to_randomized_search_results: path to the file storing the randomized search results.
   :param n_tests_per_run: The number of tests to run for each combination of hyperparameters.
   :param seeds: List of the seeds for reproducibility of the results of the grid search.

   :return: DataFrame containing the results of the randomized search.

   :Example: .... #todo bisogna inserire gli esempi per descrive i modi in cui si puiò utilizzare
    """
    model_hyperparameters_to_test = []
    training_hyperparameters_to_test = []

    i = 0

    while i < n_run:
        model_d = {}
        training_d = {}
        for key, sampler in model_hyperparameters_to_sample.items():
            model_d[key] = sampler()
        for key, sampler in training_hyperparameters_to_sample.items():
            training_d[key] = sampler()

        model_hyperparameters_already_present = np.any([model_d == mc for mc in model_hyperparameters_to_test])
        training_hyperparameters_already_present = np.any([training_d == tc for tc in training_hyperparameters_to_test])

        if not (model_hyperparameters_already_present and training_hyperparameters_already_present):
            model_hyperparameters_to_test.append(model_d)
            training_hyperparameters_to_test.append(training_d)
            i += 1

    iterator = tqdm(iterable=enumerate(zip(training_hyperparameters_to_test, model_hyperparameters_to_test)),
                    total=len(training_hyperparameters_to_test))

    return collect_results(train_data=train_data,
                           val_data=val_data,
                           iterator=iterator,
                           shuffle=shuffle,
                           device=device,
                           hyperparameters_key_to_save=hyperparameters_key_to_save,
                           path_to_results=path_to_randomized_search_results,
                           n_tests_per_run=n_tests_per_run,
                           seeds=seeds)


def collect_results(train_data: tuple[torch.Tensor, torch.Tensor],
                    val_data: tuple[torch.Tensor, torch.Tensor],
                    iterator: ANY,
                    shuffle: bool,
                    device: str,
                    hyperparameters_key_to_save: list[str],
                    path_to_results: str,
                    n_tests_per_run: Optional[int] = None,
                    seeds: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Collects and aggregates training and validation metrics for a given machine learning model across multiple runs.

    Parameters:
    :param train_data: A tuple containing training input data and labels.
    :param val_data: A tuple containing validation input data and labels.
    :param iterator: An iterator providing hyperparameters for both training and the model architecture.
    :param shuffle: A boolean indicating whether to shuffle the training data during training.
    :param device: Device on which to perform the training (e.g., 'cpu' or 'cuda').
    :param hyperparameters_key_to_save: List of keys specifying which hyperparameters to save in the result DataFrame.
    :param path_to_results: path to the file storing the grid or randomized search results.
    :param n_tests_per_run: The number of tests to run for each combination of hyperparameters.
    :param seeds: List of the seeds for reproducibility of the results of the grid search.

    Returns:
    pd.DataFrame: A DataFrame containing aggregated training and validation metrics for each run,
                  with hyperparameters specified in 'hyperparameters_key_to_save'.

    Notes:
    - The function must be used inside another function that generate the combination of the hyperparameter that
      you want to test
    - The function assumes the existence of a 'fit' method in the model class, taking training and validation data loaders.
    - The metrics are assumed to be specified in the 'metrics' field of the training hyperparameters.

    Example:

    # Define data and iterator
    train_data = (train_input, train_labels)
    val_data = (val_input, val_labels)
    iterator = your_iterator_function()

    # Define hyperparameters to save
    hyperparameters_key_to_save = ['learning_rate', 'batch_size', 'optimizer', 'criterion']

    # Collect results
    results_df = collect_results(train_data, val_data, iterator, shuffle=True, n_tests_per_run=3,
                                 device='cuda', hyperparameters_key_to_save=hyperparameters_key_to_save)

    """

    if n_tests_per_run is None and seeds is None:
        raise ValueError('Inconsistent value. One of n_tests_per_run or seeds must be assigned.')
    elif n_tests_per_run is not None and seeds is not None:
        raise ValueError('Ambiguous values assignment for n_tests_per_run and seeds only one must be assigned.')
    elif n_tests_per_run is not None:
        test_iterator = range(n_tests_per_run)
    else:
        test_iterator = seeds

    train_input, train_labels = train_data
    val_input, val_labels = val_data
    dataframe_dict = {key: [] for key in hyperparameters_key_to_save}
    dataframe_dict['mean_time'] = []
    dataframe_dict['std_time'] = []

    for n_run, (training_hyperparameters, model_hyperparameters) in iterator:

        early_stopper_exist = 'early_stopper' in training_hyperparameters and training_hyperparameters[
            'early_stopper'] is not None

        train_data_loader = get_data_loader(data=train_input, label=train_labels,
                                            batch_size=training_hyperparameters['batch_size'], shuffle=shuffle)
        val_data_loader = get_data_loader(data=val_input, label=val_labels,
                                          batch_size=training_hyperparameters['batch_size'], shuffle=False)

        new_training_hyperparameters = training_hyperparameters.copy()
        new_training_hyperparameters.pop('batch_size')

        train_results = {}
        val_results = {}

        for metric in training_hyperparameters['metrics']:
            train_results[metric.name] = []
            val_results[metric.name] = []

        fitting_times = []

        for test_iteration in test_iterator:
            if seeds is not None:
                torch.manual_seed(test_iteration)
                np.random.seed(test_iteration)

            new_model_hyperparameters = model_hyperparameters.copy()
            new_model_hyperparameters.pop('model_class')
            net = model_hyperparameters['model_class'](**new_model_hyperparameters).to(device)

            start_time = time.time()

            result = net.fit(train_loader=train_data_loader,
                             val_loader=val_data_loader,
                             verbose=False,
                             **new_training_hyperparameters)

            end_time = time.time()
            fitting_times.append(end_time - start_time)

            for metric in training_hyperparameters['metrics']:
                history_idx = -1
                if early_stopper_exist:
                    history_idx -= training_hyperparameters['early_stopper'].patience
                train_results[metric.name].append(result['train'][metric.name][history_idx])
                val_results[metric.name].append(result['val'][metric.name][history_idx])

        merged_hyperparameters = {**model_hyperparameters, **training_hyperparameters}

        for key in hyperparameters_key_to_save:
            dataframe_dict[key].append(merged_hyperparameters[key])

        dataframe_dict['mean_time'].append(np.round(np.mean(fitting_times), 3))
        dataframe_dict['std_time'].append(np.round(np.std(fitting_times), 3))

        for metric in training_hyperparameters['metrics']:
            train_mean = np.mean(train_results[metric.name])
            train_std = np.std(train_results[metric.name])

            val_mean = np.mean(val_results[metric.name])
            val_std = np.std(val_results[metric.name])

            if metric.name + '_train_mean' not in dataframe_dict:
                dataframe_dict[metric.name + '_train_mean'] = []
            dataframe_dict[metric.name + '_train_mean'].append(train_mean)

            if metric.name + '_train_std' not in dataframe_dict:
                dataframe_dict[metric.name + '_train_std'] = []
            dataframe_dict[metric.name + '_train_std'].append(train_std)

            if metric.name + '_val_mean' not in dataframe_dict:
                dataframe_dict[metric.name + '_val_mean'] = []
            dataframe_dict[metric.name + '_val_mean'].append(val_mean)

            if metric.name + '_val_std' not in dataframe_dict:
                dataframe_dict[metric.name + '_val_std'] = []
            dataframe_dict[metric.name + '_val_std'].append(val_std)

    df = pd.DataFrame(data=dataframe_dict)
    joblib.dump(df, path_to_results)

    return df


