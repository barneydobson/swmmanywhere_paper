import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def fill_nans_with_ann(df_o, df_p):
    """
    Fill NaN values in each column of df_o using an ANN regression model trained on df_p.

    Args:
        df_o (pd.DataFrame): DataFrame with NaN values to be filled.
        df_p (pd.DataFrame): DataFrame used for training the ANN regression model.

    Returns:
        pd.DataFrame: A copy of df_o with NaN values filled using ANN regression.
    """
    df_filled = df_o.copy()

    for col in df_filled.columns:
        # Check if the column has NaN values
        if df_filled[col].isnull().any() and not df_filled[col].isnull().all():
            # Split the data into features and target
            X = df_p.values
            y = df_filled[col].values

            # Handle NaN values in the target column
            is_nan = pd.isna(y)
            X_train = X[~is_nan]
            y_train = y[~is_nan]

            # Standardize the data
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)

            # Train the ANN regression model
            ann = MLPRegressor(hidden_layer_sizes=(100,), random_state=42)
            ann.fit(X_train, y_train)

            # Predict the missing values
            X_test = scaler.transform(X[is_nan])
            y_pred = ann.predict(X_test)

            # Fill the missing values in the original DataFrame
            df_filled.loc[is_nan, col] = y_pred

    return df_filled