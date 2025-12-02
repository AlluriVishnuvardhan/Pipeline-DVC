import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# avoid duplicate handlers when re-running in same interpreter/session
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
else:
    # ensure handlers include ours (idempotent)
    found_console = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    found_file = any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    if not found_console:
        logger.addHandler(console_handler)
    if not found_file:
        logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file and validate contents."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s: %s', params_path, params)
        if params is None:
            raise ValueError("params.yaml is empty or invalid YAML (safe_load returned None).")
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file (URL or local path)."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s; shape=%s', data_url, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        # be defensive: only drop columns that exist
        cols_to_drop = [c for c in ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'] if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)
            logger.debug('Dropped columns: %s', cols_to_drop)

        # rename only if columns exist
        rename_map = {}
        if 'v1' in df.columns:
            rename_map['v1'] = 'target'
        if 'v2' in df.columns:
            rename_map['v2'] = 'text'
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
            logger.debug('Renamed columns: %s', rename_map)
        else:
            logger.warning('Expected columns v1/v2 not found; current columns: %s', df.columns.tolist())

        logger.debug('Data preprocessing completed; new shape=%s', df.shape)
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, output_data_dir: str) -> None:
    """Save the train and test datasets into local folder structure."""
    try:
        raw_data_path = os.path.join(output_data_dir, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_fp = os.path.join(raw_data_path, "train.csv")
        test_fp = os.path.join(raw_data_path, "test.csv")
        train_data.to_csv(train_fp, index=False)
        test_data.to_csv(test_fp, index=False)
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')

        # validate nested structure gracefully and provide a default
        test_size = None
        if isinstance(params, dict) and 'data_ingestion' in params:
            test_size = params['data_ingestion'].get('test_size')
        if test_size is None:
            logger.warning("test_size not found in params.yaml under 'data_ingestion'. Using default 0.2")
            test_size = 0.2
        else:
            try:
                test_size = float(test_size)
            except Exception:
                logger.warning("test_size in params.yaml not convertible to float. Using default 0.2")
                test_size = 0.2

        # Source URL for dataset
        data_url = 'https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv'

        df = load_data(data_url=data_url)
        final_df = preprocess_data(df)

        # If dataset is empty or too small, raise
        if final_df.empty:
            raise ValueError("Loaded dataframe is empty after preprocessing.")

        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)

        # local output folder
        save_data(train_data, test_data, output_data_dir='./data')

        logger.info("Data ingestion finished successfully. Train shape: %s; Test shape: %s",
                    train_data.shape, test_data.shape)

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
