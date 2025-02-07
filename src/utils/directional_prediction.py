
# src/ml/models/shared/directional_prediction.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# -------------------------------------------------------------------------------------------
# Predict directions
# -------------------------------------------------------------------------------------------

def calculate_directions(data):
    directions = np.diff(data.squeeze())
    directional_data = np.zeros((directions.size, 2))
    for i, diff in enumerate(directions):
        if diff > 0:
            directional_data[i] = [0, 1]  # Up
        else:
            directional_data[i] = [1, 0]  # Down
    return directional_data

def calculate_movement_metrics(true_labels, predicted_labels, model_name, set_type, data_type):
    if set_type == "Train":
        # Calculate only accuracy for the training set
        accuracy = accuracy_score(true_labels, predicted_labels) * 100
        print(f'{model_name} ({data_type}) | Train Accuracy: {accuracy:.2f}%')

        return {'Train Accuracy': accuracy}

    if set_type == "Val":
        # Calculate only accuracy for the validation set
        accuracy = accuracy_score(true_labels, predicted_labels) * 100
        print(f'{model_name} ({data_type}) | Validation Accuracy: {accuracy:.2f}%')

        return {'Validation Accuracy': accuracy}

    elif set_type == "Test":
        # Calculate metrics for the test set
        accuracy = accuracy_score(true_labels, predicted_labels) * 100
        recall = recall_score(true_labels, predicted_labels, pos_label=1) * 100
        precision_rise = precision_score(true_labels, predicted_labels, pos_label=1) * 100
        precision_fall = precision_score(true_labels, predicted_labels, pos_label=0) * 100
        f1 = f1_score(true_labels, predicted_labels, pos_label=1) * 100

        print(f'{model_name} ({data_type}) | Test Accuracy: {accuracy:.2f}%')
        print(f'{model_name} ({data_type}) | Recall: {recall:.2f}%')
        print(f'{model_name} ({data_type}) | Precision (Rise): {precision_rise:.2f}%')
        print(f'{model_name} ({data_type}) | Precision (Fall): {precision_fall:.2f}%')
        print(f'{model_name} ({data_type}) | F1 Score: {f1:.2f}%')

        # Confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(7, 4)) 
        # sns.heatmap(cm, annot=True, fmt="d", cmap=plt.cm.Blues, xticklabels=["Down", "Up"], yticklabels=["Down", "Up"], cbar=False)
        ax = sns.heatmap(cm, annot=True, fmt="d", cmap="GnBu", xticklabels=["Down", "Up"], yticklabels=["Down", "Up"],
                            cbar=False, square=True, annot_kws={'fontsize':10}, linecolor='black')
        # sns.heatmap(cm, annot=True, fmt="d", cmap="GnBu", xticklabels=["Down", "Up"], yticklabels=["Down", "Up"], cbar=False,
        #     linewidths=.5, linecolor='black', square=True)

        # Add percentages
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j + 0.5, i + 0.55, f'\n({cm_norm[i, j]:.2%})',
                         horizontalalignment='center',
                         verticalalignment='center',
                         color='white' if cm[i, j] > cm.max() / 1.5 else 'black',  # Use contrasting text color
                         fontsize=14, fontweight='bold')
                
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(False)



        plt.xlabel('Predicted', fontsize=10, fontweight='bold')
        plt.ylabel('Actual', fontsize=10, fontweight='bold')
        plt.title(f'ETH Confusion Matrix in 1 Hour Time Frame ' ,fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'C:/Users/arvin/Documents/Coding Project/V4/Algotrading_RL/src/cnn/BTC_1h/figs/ETH1H_ConfusionMatrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        return {
            'Test Accuracy': accuracy,
            'Recall': recall,
            'Precision (Rise)': precision_rise,
            'Precision (Fall)': precision_fall,
            'F1 Score': f1
        }

def evaluate_directional_movement(actual_values_train, backtest_train, actual_values_val, backtest_val, actual_values_test, backtest_test, model_name, data_type, using_darts=True):
    if using_darts:
        train_y = actual_values_train.values()
        train_predictions = backtest_train.values()
        val_y = actual_values_val.values()
        val_predictions = backtest_val.values()
        test_y = actual_values_test.values()
        test_predictions = backtest_test.values()
    else:
        train_y = actual_values_train
        train_predictions = backtest_train
        val_y = actual_values_val
        val_predictions = backtest_val
        test_y = actual_values_test
        test_predictions = backtest_test

    # Calculate directions for training set
    true_directions_train = calculate_directions(train_y)
    predicted_directions_train = calculate_directions(train_predictions)

    # Convert to class labels for training set
    true_labels_train = np.argmax(true_directions_train, axis=1)
    predicted_labels_train = np.argmax(predicted_directions_train, axis=1)

    # Calculate directions for validation set
    true_directions_val = calculate_directions(val_y)
    predicted_directions_val = calculate_directions(val_predictions)

    # Convert to class labels for validation set
    true_labels_val = np.argmax(true_directions_val, axis=1)
    predicted_labels_val = np.argmax(predicted_directions_val, axis=1)

    # Calculate directions for test set
    true_directions_test = calculate_directions(test_y)
    predicted_directions_test = calculate_directions(test_predictions)

    # Convert to class labels for test set
    true_labels_test = np.argmax(true_directions_test, axis=1)
    predicted_labels_test = np.argmax(predicted_directions_test, axis=1)

    # Calculate metrics for training set
    metrics_train = calculate_movement_metrics(true_labels_train, predicted_labels_train, model_name, "Train", data_type)

    # Calculate metrics for validation set
    metrics_val = calculate_movement_metrics(true_labels_val, predicted_labels_val, model_name, "Val", data_type)

    # Calculate metrics for test set
    metrics_test = calculate_movement_metrics(true_labels_test, predicted_labels_test, model_name, "Test", data_type)

    # Combine accuracy into the test metrics dictionary
    metrics_test['Validation Accuracy'] = metrics_val['Validation Accuracy']
    metrics_test['Train Accuracy'] = metrics_train['Train Accuracy']

    return true_labels_test, predicted_labels_test, metrics_test
