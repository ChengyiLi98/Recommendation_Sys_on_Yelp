'''
Method  Description:
I found out that in the hybrid recommendation system I built, item-based cf is not contributing to lower RMSE,
so I only kept xgboost and fine tune it.

To further improve performance, Random Forest was introduced and fine-tuned as an additional model. Despite rigorous
hyperparameter tuning, it consistently underperformed compared to the XGBoost model. Therefore, the final
system exclusively uses XGBoost with carefully tuned hyperparameters for better generalization and lower RMSE.

The fine-tuning process for XGBoost included optimizing parameters such as max_depth, learning_rate, and
n_estimators by using grid search.

Error Distribution:
>=0 and <1: 102175
>=1 and <2: 32880
>=2 and <3: 6167
>=3 and <4: 821
>=4: 1

RMSE:
0.97966

Execution Time:
147s
'''

from pyspark import SparkContext
import json
import numpy as np
from xgboost import XGBRegressor
import time
import sys
import os
from sklearn.ensemble import RandomForestRegressor
from collections import Counter


# os.environ['PYSPARK_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ['PYSPARK_DRIVER_PYTHON'] = '/usr/local/bin/python3.6'
# os.environ["PYSPARK_PYTHON"] = "C:/Program Files/Python36/python.exe"
# os.environ["PYSPARK_DRIVER_PYTHON"] = "C:/Program Files/Python36/python.exe"


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

    Args:
        tip_file (str): The path to the tip JSON file.
        sc: SparkContext instance.

    Returns:
        dict: A dictionary with keys as (user_id, business_id) and values as the number of interactions.
    """
    # Load the tip JSON data into an RDD
    lines = sc.textFile(tip_file)

    # Parse JSON data and map to (user_id, business_id) pairs
    user_business_interactions = lines.map(json.loads).map(lambda x: (x['user_id'], x['business_id'])) \
        .map(lambda pair: ((pair[0], pair[1]), 1))  # Each interaction is 1

    # Aggregate counts
    interaction_counts = user_business_interactions.reduceByKey(lambda a, b: a + b).collectAsMap()

    return interaction_counts


def load_photo_data(photo_file, sc):
    """
    Load photo data from a JSON file and calculate the number of photos per business.

    Args:
        photo_file (str): Path to the photo JSON file.
        sc: SparkContext instance.

    Returns:
        dict: A dictionary with keys as business_id and values as the number of photos.
    """
    # Load the photo JSON data into an RDD
    lines = sc.textFile(photo_file)

    # Parse JSON data and map to (business_id, 1)
    business_photo_counts = lines.map(json.loads) \
        .map(lambda x: (x['business_id'], 1)) \
        .reduceByKey(lambda a, b: a + b) \
        .collectAsMap()

    return business_photo_counts


def process_all_data(sc, file_path, user_dict, business_dict, tip_data, photo_counts_dict, mode='train'):
    """
    Processes the user, business, and tip data, and returns features and labels accordingly.

    Args:
        sc: SparkContext instance.
        file_path (str): Path to the main data file.
        user_dict (dict): User data dictionary.
        business_dict (dict): Business data dictionary.
        tip_data (dict): User-business interaction counts from tip.json.
        mode (str): Either 'train' or 'validation'.

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
            tip_interaction_cnt,  # tip interaction feature
            photo_interaction_cnt  # photo interaction feature
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


def train_xgb_model(X_train, Y_train):
    # XGBoost can handle NaN values by using a default direction in its decision trees.
    # It will skip or impute missing values during training and prediction.
    custom_xgb = XGBRegressor(
        max_depth=10,
        learning_rate=0.02,
        min_child_weight=2,
        n_estimators=500,
        subsample=0.75,
        colsample_bytree=0.75,
        gamma=0.5,
        reg_lambda=5,
        reg_alpha=0.5,
        random_state=66
    )
    custom_xgb.fit(X_train, Y_train)
    return custom_xgb


def train_rf_model(X_train, Y_train):
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=66,
        n_jobs=-1
    )
    rf_model.fit(X_train, Y_train)
    return rf_model


def model_main_with_rf(folder_path, val_path):
    user_dict = load_user_data(os.path.join(folder_path, 'user.json'), sc)
    business_dict = load_business_data(os.path.join(folder_path, 'business.json'), sc)
    tip_dict = load_tip_data(os.path.join(folder_path, 'tip.json'), sc)
    photo_counts_dict = load_photo_data(os.path.join(folder_path, 'photo.json'), sc)
    X_train, Y_train = process_all_data(sc, os.path.join(folder_path, 'yelp_train.csv'), user_dict, business_dict,
                                        tip_dict, photo_counts_dict, mode='train')

    user_bus_list, X_val, y_true = process_all_data(sc, val_path, user_dict, business_dict, tip_dict, photo_counts_dict,
                                                    mode='validation')

    # Train XGBoost model
    xgb_model = train_xgb_model(X_train, Y_train)
    y_xgb_pred = xgb_model.predict(X_val)

    # Train Random Forest model
    rf_model = train_rf_model(X_train, Y_train)
    y_rf_pred = rf_model.predict(X_val)

    return y_true, y_xgb_pred, y_rf_pred, user_bus_list


if __name__ == '__main__':
    folder_path = sys.argv[1]
    test_f = sys.argv[2]
    out_f = sys.argv[3]
    sc = SparkContext("local[*]", "Competition")
    sc.setLogLevel("ERROR")

    initial_t = time.time()

    y_true, y_xgb_pred, y_rf_pred, user_bus_list = model_main_with_rf(folder_path, test_f)

    alpha = 0

    y_xgb_pred = np.array(y_xgb_pred)
    y_rf_pred = np.array(y_rf_pred)

    # Hybrid prediction with weights for each model

    hybrid_pred = alpha * y_rf_pred + (1 - alpha) * y_xgb_pred
    rmse, error_distribution = get_RMSE_res(y_true, hybrid_pred)
    print("RMSE:", rmse)
    print("Error Distribution:", error_distribution)

    with open(out_f, "w") as f:
        f.write("user_id,business_id,prediction\n")
        for (user_id, business_id), prediction in zip(user_bus_list, hybrid_pred):
            f.write(f"{user_id},{business_id},{prediction}\n")

    end_t = time.time()
    print('Duration:', end_t - initial_t)