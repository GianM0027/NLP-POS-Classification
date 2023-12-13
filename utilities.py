import numpy as np
import pandas as pd


from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm

from wordcloud import WordCloud

import os
from sre_constants import ANY


def merge_data(work_directory_path: str) -> pd.DataFrame:
    """
      Merges data from multiple files located in the work directory.

      :param work_directory_path: A string representing the path to the working directory
      :return: A DataFrame containing data loaded from the files
      """
    files_to_load = ['wsj_{:04d}.dp'.format(i) for i in range(1, 200)]
    tmp_data = []

    for file_path in tqdm(files_to_load):
        current_file_path = os.path.join(work_directory_path, file_path)

        with open(current_file_path, 'r') as file:
            for line in file:
                parts = line.split()
                if len(parts) == 3:
                    word = parts[0]
                    pos_tag = parts[1]
                    info = int(parts[2])
                    source_file = file_path

                    tmp_data.append([word, pos_tag, info, source_file])
    return pd.DataFrame(tmp_data, columns=['WORD', 'POS', 'INFO', 'SOURCE_FILE'])


def split_dataset(df: pd.DataFrame,
                  train_bounds: tuple[int, int],
                  validation_bounds: tuple[int, int],
                  test_bounds: tuple[int, int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits a DataFrame into three subsets based on specific intervals.

    :param df: The source DataFrame to be split.
    :param train_bounds: A tuple of two integers representing the start and end of the training set interval.
    :param validation_bounds: A tuple of two integers representing the start and end of the validation set interval.
    :param test_bounds: A tuple of two integers representing the start and end of the test set interval.
    :return: A tuple containing three DataFrames: the training set, the validation set, and the test set.
    """

    train_files = ['wsj_{:04d}.dp'.format(i) for i in range(train_bounds[0], train_bounds[1] + 1)]
    validation_files = ['wsj_{:04d}.dp'.format(i) for i in range(validation_bounds[0], validation_bounds[1] + 1)]
    test_file = ['wsj_{:04d}.dp'.format(i) for i in range(test_bounds[0], test_bounds[1] + 1)]
    train_df = df[df['SOURCE_FILE'].isin(train_files)]
    validation_df = df[df['SOURCE_FILE'].isin(validation_files)]
    test_df = df[df['SOURCE_FILE'].isin(test_file)]

    return train_df, validation_df, test_df


def create_wordcloud(df: pd.DataFrame, my_class_index: str = 'WORD', f_sizes: tuple[int, int] = (10, 5)) -> None:
    """
    Generates and displays a word cloud based on the specified DataFrame and column.

    :param df: The input DataFrame containing text data.
    :param my_class_index: The column name in the DataFrame that contains the text data.
      Defaults to 'WORD'.
    :param f_sizes: A tuple representing the size of the generated plot.

    :return: This function displays the generated word cloud using Matplotlib.

    Example:
    ```python
    import pandas as pd
    from my_wordcloud_module import create_wordcloud

    # Assuming 'df' is a DataFrame with a column named 'WORD' containing text data
    create_wordcloud(df, my_class_index='WORD', f_sizes=(12, 6))
    ```

    Note:
    - Ensure that the 'wordcloud' and 'matplotlib.pyplot' libraries are installed.
    - You can install them using the following:
      ```
      pip install wordcloud
      pip install matplotlib
      ```
    """
    text = " ".join(df[my_class_index])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    plt.figure(figsize=f_sizes)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def plot_classes_distribution(datasets: dict[str, pd.DataFrame], my_class: str, title: str) -> None:
    """
    Plots the distribution of classes within the provided datasets using a bar chart.

    :param datasets: A dictionary containing the datasets to be visualized, where keys are dataset names and values
                      are Pandas DataFrames. The expected keys are:
                        'full_dataset',
                        'train_dataset',
                        'val_dataset', and
                        'test_dataset'.
    :param my_class: The column name representing the classes in the DataFrames.
    :param title: The title of the plot.
    :return: None

    Example:
        plot_classes_distribution({'full_dataset': df}, 'POS', 'Test')
        plot_classes_distribution({'train_dataset': train_df, 'val_dataset': val_df, 'test_dataset': test_df},
                                    'POS',
                                    'Test'
                                  )

    """
    if len(datasets.keys()) == 1:
        plt.figure(figsize=(15, 5))
        plt.bar(datasets['full_dataset'][my_class].value_counts().index,
                datasets['full_dataset'][my_class].value_counts())
        plt.title(title)
        plt.xlabel('Classes')
        plt.ylabel('Occurrences')
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()

    else:
        plt.figure(figsize=(18, 5))

        train_class_counts = datasets['train_dataset'][my_class].value_counts(sort=False).rename('Train').div(
            len(datasets['train_dataset'])).mul(100)
        val_class_counts = datasets['val_dataset'][my_class].value_counts(sort=False).rename('Val').div(
            len(datasets['val_dataset'])).mul(100)
        test_class_counts = datasets['test_dataset'][my_class].value_counts(sort=False).rename('Test').div(
            len(datasets['test_dataset'])).mul(100)

        comparison_df = pd.concat([train_class_counts, val_class_counts, test_class_counts], axis=1, sort=False)
        comparison_df.reset_index(inplace=True)
        comparison_df.rename(columns={'index': 'POS'}, inplace=True)

        bar_width = 0.3
        classes = comparison_df[my_class]

        class_positions = list(range(len(classes)))

        plt.bar([pos - bar_width for pos in class_positions], comparison_df['Train'], width=bar_width, label='Train',
                color='r')
        plt.bar(class_positions, comparison_df['Val'], width=bar_width, label='Val', color='g')
        plt.bar([pos + bar_width for pos in class_positions], comparison_df['Test'], width=bar_width, label='Test',
                color='b')

        plt.xticks(class_positions, classes, rotation=90, ha='right')

        plt.title(title)
        plt.xlabel('Classes')
        plt.ylabel('Occurrence rate (%)')
        plt.legend()
        plt.tight_layout()

    plt.show()


def build_sentences(df: pd.pandas.DataFrame) -> tuple[list, list]:
    """
    Split a pandas DataFrame into sentences based on periods ('.'). The input DataFrame should contain columns
    'SOURCE_FILE', 'WORD', and 'POS' to identify the source file, words, and part-of-speech tags. The function returns
    two lists: one containing lists of words in each sentence and another containing lists of corresponding
    part-of-speech tags. Sentences are split at periods, and padding with '<pad>' is applied to ensure uniform length.

    :param df: A pandas DataFrame containing the data to be processed.

    :return: A tuple containing two lists - sentence_matrix and pos_matrix.
        - sentence_matrix: A list of lists, each inner list represents a sentence as a sequence of words.
        - pos_matrix: A list of lists, each inner list represents the part-of-speech tags for words in corresponding
                      sentences.
    """
    sentence_matrix = []
    pos_matrix = []

    max_length = 0

    for source_file in df.loc[:, 'SOURCE_FILE'].unique():
        splitted_sentence = []
        splitted_pos = []

        words = df[df['SOURCE_FILE'] == source_file]['WORD'].tolist()
        pos = df[df['SOURCE_FILE'] == source_file]['POS'].tolist()

        for pos_idx, word in enumerate(words):
            if word == '.':

                splitted_sentence.append(word)
                splitted_pos.append(pos[pos_idx])

                sentence_matrix.append(splitted_sentence)
                pos_matrix.append(splitted_pos)

                if len(splitted_sentence) > max_length:
                    max_length = len(splitted_sentence)

                splitted_sentence = []
                splitted_pos = []

            else:
                splitted_sentence.append(word)
                splitted_pos.append(pos[pos_idx])

    for sentence, pos_list in zip(sentence_matrix, pos_matrix):
        sentence += ['<pad>'] * (max_length - len(sentence))
        pos_list += ['<pad>'] * (max_length - len(pos_list))

    return sentence_matrix, pos_matrix


def mapping(matrix: list[list[str]], my_map: dict, oov_index: int) -> list[list[ANY]]:
    """
    Apply a dictionary-based mapping to a matrix (a list of lists).

    This function takes a matrix represented as a list of lists and a mapping dictionary.
    It applies the mapping dictionary to each element in the matrix and returns a new matrix
    with the mapped values.

    :param oov_index: An integer that represent the index of the out of vocabulary token in the embedding matrix
    :param matrix: A list of lists where each element will be mapped according to the my_map dictionary.
    :param my_map: A function that defines the mapping of elements in the matrix.

    :return: A new matrix with elements mapped according to the my_map dictionary.

    Example:
    matrix = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
    my_map = {1: 'A', 2: 'B', 3: 'C'}
    mapped_matrix = mapping(matrix, my_map)
    print(mapped_matrix)
    [['A', 'B', 'C'], ['B', 'C', 'A'], ['C', 'A', 'B']]
    """
    return [[my_map.get(element, oov_index) for element in row] for row in matrix]


def create_classes_weights(list_of_label_index: list[int], list_of_index_to_exclude) -> np.array:
    """
    Calculate class weights for a list of classes.

    This function takes a 2D list of classes (e.g., part-of-speech tags)
    and calculates class weights to use during the training process
    based on the formula n_samples / (n_classes * np.bincount(y)).


    list_of_label_index A list of indexes that represent the classes,
    list_of_index_to_exclude: List of elements to be dropped from consideration when calculating weights.

    :return: An array of class weights corresponding to the input classes. The weights are inversely
             proportional to the class occurrences in the input data. The weights for the 'list_of_index_to_exclude' classes
             to drop are set to 0.
    """

    occurrences = np.bincount(list_of_label_index)
    tmp_occurrences = occurrences.copy()
    tmp_occurrences[list_of_index_to_exclude] = 0
    class_weights = np.sum(tmp_occurrences) / ((len(occurrences) - len(list_of_index_to_exclude)) * occurrences)
    class_weights[list_of_index_to_exclude] = 0
    return class_weights


def plot_confusion_matrix(cm: np.ndarray,
                          labels: list[str | int],
                          figsize: tuple[int, int] = (8, 6),
                          font_scale: float = 1.2,
                          cmap: str = "coolwarm",
                          title: str = "Confusion Matrix") -> None:
    """
    Plot a confusion matrix using seaborn.

    Parameters:
    :param cm: Confusion matrix.
    :param labels: List of class labels.
    :param figsize: Figure size. Default is (8, 6).
    :param font_scale: Font scale for text. Default is 1.2.
    :param cmap: Color map for the heatmap. Default is "coolwarm".
    :param title: Title for the plot.
    :return:
    - None: Displays the plot.

    """
    plt.figure(figsize=figsize)
    sns.set(font_scale=font_scale)  # Adjust the font size if needed

    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, xticklabels=labels, yticklabels=labels)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.title(title)

    plt.show()


def plot_dimensionality_reduction(weights_before: np.array,
                                  weights_after: np.array,
                                  title: str,
                                  colors: np.ndarray,
                                  transformation: str = "LDA") -> None:
    """
    Plots the dimensionality reduction of feature weights before and after a certain transformation.

    :param weights_before: Feature weights before the transformation.
    :param weights_after: Feature weights after the transformation.
    :param title: Title for the plot.
    :param colors: Array of unique colors for each class.
    :param transformation: The transformation technique, either "PCA" or "LDA". Default is "LDA".

    :return:
    - None: Displays the plot.

    Example:
    ```python
    # Usage example with your data
    plot_dimensionality_reduction(your_weights_before, your_weights_after, title='Your Plot Title', colors=np.array(unique_colors))
    ```

    """
    column_to_color = weights_before[:, 1]
    label_encoder = LabelEncoder()
    numeric_column_to_color = label_encoder.fit_transform(column_to_color)

    colored_column = colors[numeric_column_to_color]

    if transformation == "PCA":
        pca_before = PCA(n_components=2)
        components_before = pca_before.fit_transform(weights_before[:, 2:])

        pca_after = PCA(n_components=2)
        components_after = pca_after.fit_transform(weights_after[:, 2:])
    elif transformation == "LDA":
        lda_before = LinearDiscriminantAnalysis(n_components=2)
        components_before = lda_before.fit_transform(weights_before[:, 2:], numeric_column_to_color)

        lda_after = LinearDiscriminantAnalysis(n_components=2)
        components_after = lda_after.fit_transform(weights_after[:, 2:], numeric_column_to_color)
    else:
        raise ValueError("Inconsistent transformation technique. The function supports only PCA and LDA transformation")

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(15, 5))

    ax1.scatter(components_before[:, 0], components_before[:, 1], c=colored_column)
    ax1.set_title('Weights Before')
    ax1.grid()

    ax2.scatter(components_after[:, 0], components_after[:, 1], c=colored_column)
    ax2.set_title('Weights After')
    ax2.grid()

    custom_legend = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], markersize=10)
                     for i in np.unique(numeric_column_to_color)]
    legend_labels = label_encoder.classes_

    ax1.legend(custom_legend,
               legend_labels,
               title="Classes",
               loc="lower center",
               ncol=len(legend_labels) // 3,
               bbox_to_anchor=(1.1, -0.6))

    plt.suptitle(title)

    plt.show()
