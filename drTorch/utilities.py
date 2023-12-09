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


import torch
from matplotlib import pyplot as plt


def get_data_loader(batch_size: int,
                    shuffle: bool,
                    data: torch.Tensor,
                    label: torch.Tensor) -> torch.utils.data.DataLoader:
    """
    Create a PyTorch DataLoader for a given dataset and labels.

    This function takes your data and corresponding labels, and wraps them into a PyTorch DataLoader, which is useful
    for batching and shuffling the data during training. The DataLoader can be used in machine learning models,
    especially for tasks like training neural networks.

    :param batch_size: The number of data samples to include in each batch.
    :param shuffle: If True, the data will be shuffled at the beginning of each epoch. Use True for training and False
                    for evaluation and testing.
    :param data: A torch.Tensor representing the input data for your model. Pay attention to the specific data type.
    :param label: A torch.Tensor representing the label data for your model. Pay attention to the specific data type.

    :return: A PyTorch DataLoader object that can be iterated over to access batches of data and labels.

    Example:
    ```
    data = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])                            here the data type is int64
    data1 = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float64)      here the data type is float64

    label = torch.tensor([0, 1, 0])
    batch_size = 2
    shuffle = True

    data_loader = get_data_loader(data, label, batch_size, shuffle)
    ```
    """
    torch_dataSet = torch.utils.data.TensorDataset(data, label)
    return torch.utils.data.DataLoader(torch_dataSet, batch_size=batch_size, shuffle=shuffle)


def custom_collate_with_labels(batch:list[dict[str, torch.tensor]]):
    """
    Custom collate function for creating batches with features and labels.


    :param batch: List of samples, where each sample is a dictionary containing
                  features and labels.
    :return:A tuple containing a dictionary of stacked tensors for features
            and a tensor for labels.


    Example:
    ```python
    data = [{'feature1': torch.randn(3), 'feature2': torch.randn(3), 'label': torch.tensor(1)},
            {'feature1': torch.randn(3), 'feature2': torch.randn(3), 'label': torch.tensor(0)}]

    custom_collate_with_labels(data)
    ```
    """
    keys = list(batch[0].keys())

    # Create a batch dictionary with stacked tensors for features
    batch_dict = {key: torch.stack([item[key].clone().detach() for item in batch]) for key in keys}

    # Add the labels as a tensor
    labels = torch.stack([item['label'].clone().detach() for item in batch])
    del batch_dict['label']

    return batch_dict, labels


def create_custom_dataset(data: dict[str, torch.tensor], labels_series:torch.tensor):
    """
    Custom dataset generator function for pairing features and labels.


    :param data: List of dictionaries, each containing features.
    :param labels_series: Tensor containing labels for each corresponding sample in the 'data'.
    :yields: Dict[str, torch.Tensor]: A dictionary representing a sample with features and labels.


    Example:
    ```python
    data = [{'feature1': torch.randn(3), 'feature2': torch.randn(3)},
            {'feature1': torch.randn(3), 'feature2': torch.randn(3)}]
    labels = torch.tensor([[0, 1], [34567, 4]])

    create_custom_dataset(data, labels)
    ```
    """
    for sample, label in zip(data, labels_series):
        sample['label'] = label
        yield sample


def get_data_loader_test(batch_size: int,
                         shuffle: bool,
                         data: dict[str, torch.tensor],
                         labels: torch.Tensor) -> torch.utils.data.DataLoader:
    """
    Utility function to obtain a DataLoader for testing.


    :param batch_size: Batch size for the DataLoader.
    :param shuffle: Whether to shuffle the data in each epoch.
    :param data: List of dictionaries, each containing features.
    :param labels: Tensor containing labels for each corresponding sample in the 'data'.
    :return: torch.utils.data.DataLoader: DataLoader configured for testing.


    Example:
    ```python
    data = [{'feature1': torch.randn(3), 'feature2': torch.randn(3)},
            {'feature1': torch.randn(3), 'feature2': torch.randn(3)}]
    labels = torch.tensor([[0, 1], [34567, 4]])

    get_data_loader_test(32, True, data, labels)
    ```
    """
    return torch.utils.data.DataLoader(list(create_custom_dataset(data, labels)),
                                       batch_size=batch_size,
                                       collate_fn=custom_collate_with_labels,
                                       shuffle=shuffle)


def plot_history(history: dict[str, list[float]]) -> None:
    """
    Plot training and validation history for each metric.

    Parameters:
    :param history: A dictionary containing training and validation metrics history.

    Notes:
    - Assumes a dictionary structure with 'train' and 'val' keys, each containing metrics as subkeys.
    - This function is specially designed to work with the fitting function defined inside the modules.py file

    Example:
    # Assuming `history` contains your training and validation history
    plot_history(your_history_variable)
    """
    metrics = list(history['train'].keys())  # Assuming all metrics are present in the 'train' field
    num_metrics = len(metrics)

    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 5 * num_metrics))

    for i, metric in enumerate(metrics):
        axes[i].plot(history['train'][metric], label='Training')
        axes[i].plot(history['val'][metric], label='Validation')
        axes[i].set_title(f'{metric} History')
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric)
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()
