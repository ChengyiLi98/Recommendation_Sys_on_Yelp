from sklearn.ensemble import StackingRegressor
from pyspark import SparkContext
import json
import numpy as np
from xgboost import XGBRegressor
import time
import sys
import os
from collections import Counter
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import BayesianRidge

def get_RMSE_res(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate RMSE
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)

    # Calculate absolute errors
    abs_errors = np.abs(y_pred - y_true)

    # Define error ranges
    ranges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, float('inf'))  # Covers >=4
    ]

    # Count errors in each range
    error_distribution = Counter()
    for lower, upper in ranges:
        count = np.sum((abs_errors >= lower) & (abs_errors < upper))
        range_label = f">={lower} and <{upper}" if upper != float('inf') else f">={lower}"
        error_distribution[range_label] = count

    return rmse, dict(error_distribution)


def load_user_data(user_file, sc):
    lines_user = sc.textFile(user_file)

    # Transform to extract and structure user features
    user_features = lines_user.map(json.loads).map(lambda user: (user['user_id'],
                                                                 (
                                                                     float(user.get('average_stars', 3.0)),
                                                                     float(user.get('review_count', 1.0)),
                                                                     float(user.get('fans', 0.0)),
                                                                     float(user.get('useful', 0.0)),
                                                                     float(user.get('funny', 0.0)),
                                                                     float(user.get('cool', 0.0))
                                                                 )
                                                                 )).collectAsMap()  # Collect as a dict for easy lookup

    return user_features


def load_business_data(business_file, sc):
    lines_business = sc.textFile(business_file)
    business_features = lines_business.map(json.loads).map(lambda business: (business['business_id'],
                                                                             (
                                                                                 float(business.get('stars', 3.0)),
                                                                                 float(
                                                                                     business.get('review_count', 1.0)),
                                                                                 business.get("GoodForKids", True)
                                                                             )
                                                                             )).collectAsMap()
    return business_features


def load_tip_data(tip_file, sc):
    """
    Load tip data from a JSON file and calculate user-business interaction counts.
    Returns: dict: A dictionary with keys as (user_id, business_id) and values as the number of interactions.
    """
    # Load the tip JSON data into an RDD
    lines = sc.textFile(tip_file)

    # Parse JSON data and map to (user_id, business_id) pairs
    user_business_interactions = lines.map(json.loads).map(lambda x: (x['user_id'], x['business_id'])) \
        .map(lambda pair: ((pair[0], pair[1]), 1))  # Each interaction is 1

    # Aggregate counts
    interaction_counts = user_business_interactions.reduceByKey(lambda a, b: a + b).collectAsMap()

    return interaction_counts


def load_checkin_data(checkin_file, sc):
    """
    Load check-in data from a JSON file and calculate the total number of check-ins per business.
    Returns: dict: A dict with keys as business_id and values as the total number of check-ins.
    """
    # Load the check-in JSON data into an RDD
    lines = sc.textFile(checkin_file)

    # Parse JSON data and map to (business_id, total_checkin_count)
    checkin_counts = lines.map(json.loads).map(
        lambda x: (x['business_id'], sum(map(int, x.get('time', {}).values())))).collectAsMap()

    return checkin_counts


def load_photo_data(photo_file, sc):
    """
    Load photo data from a JSON file and calculate the number of photos per business.
    Returns: dict: A dictionary with keys as business_id and values as the number of photos.
    """
    # Load the photo JSON data into an RDD
    lines = sc.textFile(photo_file)

    # Parse JSON data and map to (business_id, 1)
    business_photo_counts = lines.map(json.loads).map(lambda x: (x['business_id'], 1)).reduceByKey(
        lambda a, b: a + b).collectAsMap()

    return business_photo_counts


def process_all_data(sc, file_path, user_dict, business_dict, tip_data, photo_counts_dict, checkin_counts_dict,
                     mode='train'):
    """
    Processes the user, business, and tip data, and returns features and labels accordingly.

    Args:
        sc: SparkContext instance.
        file_path (str): Path to the main data file.
        user_dict (dict): User data dictionary.
        business_dict (dict): Business data dictionary.
        tip_data (dict): User-business interaction counts from tip.json.
        photo_counts_dict (dict): Dictionary representing the number of photos associated with a business ID.
        checkin_counts_dict (dict): Dictionary of the number of check-ins at a business, mapped by business ID.
        mode (str): Either 'train' or 'validation'.
            - 'train': The function extracts features and labels for supervised learning from the given data file.
            - 'validation': The function extracts features and user-business pairs for validation purposes, without assuming a target value.

    Returns:
        tuple: Depending on the mode:
            - If 'train': (X_data, Y_data)
            - If 'validation': (u_b_list, X_data, Y_data)
    """
    # Broadcast dictionaries

    user_dict_bc = sc.broadcast(user_dict)
    business_dict_bc = sc.broadcast(business_dict)
    tip_data_bc = sc.broadcast(tip_data)
    photo_data_bc = sc.broadcast(photo_counts_dict)
    checkin_data_bc = sc.broadcast(checkin_counts_dict)

    # Load and preprocess the data
    data = sc.textFile(file_path)
    data.count()
    header = data.first()
    data_rdd = data.filter(lambda r: r != header).cache()

    # Function to process each line
    def process_line(line):
        row = line.strip().split(",")

        if mode == 'train':
            u, b, rate = row[0], row[1], float(row[2])
            label = rate
        else:
            u, b = row[0], row[1]
            label = float(row[2]) if len(row) > 2 else None

        # Get features from broadcast dictionaries
        u_features = user_dict_bc.value.get(u, (None, None, None, None, None, None))
        b_features = business_dict_bc.value.get(b, (None, None))

        # Get tip interactions feature (user-business interaction count)
        tip_interaction_cnt = tip_data_bc.value.get((u, b), 0)  # Defaults to 0 if no interaction
        # Get photo interactions feature (user-business interaction count)
        photo_interaction_cnt = photo_data_bc.value.get((u, b), 0)
        # Get check-in count feature
        checkin_count = checkin_data_bc.value.get(b, 0)
        # Combine features
        features = [
            u_features[3],  # useful
            u_features[4],  # funny
            u_features[5],  # cool
            u_features[0],  # user_avg_star
            u_features[1],  # user_review_cnt
            u_features[2],  # user_fans
            b_features[0],  # bus_avg_star
            b_features[1],  # bus_review_cnt
            # b_features[2],  # kid
            tip_interaction_cnt,  # tip interaction feature
            photo_interaction_cnt,  # photo interaction feature
            checkin_count
        ]

        if mode == 'train':
            return (features, label)
        else:
            return ((u, b), features, label)  # Include user-business pair, features, and optional label

    # Processing logic depending on mode
    if mode == 'train':
        processed_data = data_rdd.map(process_line).cache()
        X_data = np.array(processed_data.map(lambda x: x[0]).collect())  # Collect features
        Y_data = np.array(processed_data.map(lambda x: x[1]).collect())  # Collect labels
        return X_data, Y_data
    else:
        processed_data = data_rdd.map(process_line).cache()
        u_b_list = processed_data.map(lambda x: x[0]).collect()  # Collect user-business pairs
        X_data = np.array(processed_data.map(lambda x: x[1]).collect())  # Collect features

        # Check if any entry has a non-None label
        contains_labels = any(entry[2] is not None for entry in processed_data.collect())

        # If labels are present, extract them; otherwise, set Y_data to None
        if contains_labels:
            Y_data = np.array([entry[2] for entry in processed_data.collect()])
        else:  # No labels are present
            Y_data = None

        return u_b_list, X_data, Y_data
def train_stacking_model(X_train, Y_train, X_val):
    """
    Train a Stacking Regressor using XGBoost, CatBoost, and RandomForest as base models,
    and XGBoost as the meta-model.

    Args:
        X_train (array): Training features.
        Y_train (array): Training labels.
        X_val (array): Validation features.

    Returns:
        np.array: Predictions on the validation set.
    """
    # Define base models
    base_models = [('bayesian_ridge', BayesianRidge()),
                   ('lightgbm',
                    LGBMRegressor(max_depth=10, num_leaves= 16, learning_rate=0.05, n_estimators=500, random_state=66, n_jobs=-1)),

        ('catboost', CatBoostRegressor(
        iterations=1000,       # Number of trees
        depth=10,             # Maximum tree depth
        learning_rate=0.03,
        l2_leaf_reg= 5,        # L2 regularization
        loss_function='RMSE',
        verbose= 0
    )),
        ('random_forest', RandomForestRegressor(n_estimators=500, max_depth=5, random_state=66, n_jobs= -1, verbose= 0))
    ]

    # Define meta-model
    meta_model = XGBRegressor(
        max_depth=10,
        learning_rate=0.03,
        min_child_weight=2,
        n_estimators=300,
        subsample=0.75,
        colsample_bytree=0.75,
        gamma=0.5,
        reg_lambda=5,
        reg_alpha=0.5,
        random_state=66,
    )

    # Create Stacking Regressor
    stack_regressor = StackingRegressor(estimators=base_models, final_estimator=meta_model)

    # Train the Stacking Regressor
    stack_regressor.fit(X_train, Y_train)

    # Predict on validation set
    y_val_pred = stack_regressor.predict(X_val)

    return y_val_pred


def hybrid_model_main_with_stacking(folder_path, val_path):
    """
    Main function to train models and make predictions using Stacking Regressor.

    Args:
        folder_path (str): Path to training data folder.
        val_path (str): Path to validation data.

    Returns:
        tuple: Ground truth labels, predictions from Stacking Regressor, and user-business pairs.
    """
    user_dict = load_user_data(os.path.join(folder_path, 'user.json'), sc)
    business_dict = load_business_data(os.path.join(folder_path, 'business.json'), sc)
    tip_dict = load_tip_data(os.path.join(folder_path, 'tip.json'), sc)
    photo_counts_dict = load_photo_data(os.path.join(folder_path, 'photo.json'), sc)
    checkin_counts_dict = load_checkin_data(os.path.join(folder_path, 'checkin.json'), sc)

    X_train, Y_train = process_all_data(sc, os.path.join(folder_path, 'yelp_train.csv'), user_dict, business_dict,
                                        tip_dict, photo_counts_dict, checkin_counts_dict, mode='train')

    user_bus_list, X_val, y_true = process_all_data(sc, val_path, user_dict, business_dict, tip_dict, photo_counts_dict,
                                                    checkin_counts_dict, mode='validation')

    # Train Stacking Regressor and make predictions
    y_final_pred = train_stacking_model(X_train, Y_train, X_val)

    return y_true, y_final_pred, user_bus_list


# Update main function call
if __name__ == '__main__':
    folder_path = sys.argv[1]
    test_f = sys.argv[2]
    out_f = sys.argv[3]
    sc = SparkContext("local[*]", "Competition")
    sc.setLogLevel("ERROR")

    initial_t = time.time()

    y_true, y_final_pred, user_bus_list = hybrid_model_main_with_stacking(folder_path, test_f)

    y_final_pred = np.array(y_final_pred)

    rmse, error_distribution = get_RMSE_res(y_true, y_final_pred)

    print("RMSE:", rmse)
    print("Error Distribution:", error_distribution)

    # Output predictions to file
    with open(out_f, "w") as f:
        f.write("user_id,business_id,prediction\n")
        for (user_id, business_id), prediction in zip(user_bus_list, y_final_pred):
            f.write(f"{user_id},{business_id},{prediction}\n")

    end_t = time.time()
    print('Duration:', end_t - initial_t)
