import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import numpy as np
from typing import List, Dict, Union, Optional
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import timedelta, datetime
import pandas as pd
import argparse
from joblib import Parallel, delayed
import logging
from zoneinfo import ZoneInfo
import matplotlib.dates as mdates
from scipy import stats
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import r2_score
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global configuration
CONFIG = {
    "window_size_hours": 2,
    "window_steps": 24,  # 5-min steps in 2 hours
    "insulin_lifetime_hours": 4.0,  # Default insulin lifetime
    "min_carbs": 0,
    "max_carbs": 150,
    "min_bg": 40,
    "max_bg": 400,
    "min_insulin": 0,
    "max_insulin": 30,
    "min_icr": 5,
    "max_icr": 20,
    "min_isf": 10,
    "max_isf": 100,
    "timezone": "UTC",
    # Nuevos parámetros para work, sleep y activity
    "max_work_intensity": 10,
    "max_sleep_quality": 10,
    "max_activity_intensity": 10
}

def calculate_iob(timestamp: datetime, basal_df: pl.DataFrame, bolus_df: pl.DataFrame, 
                 insulin_lifetime_hours: float = CONFIG["insulin_lifetime_hours"]) -> float:
    """
    Calculate Insulin on Board (IOB) at a given timestamp.
    
    Args:
        timestamp: Time point to calculate IOB for
        basal_df: DataFrame with basal insulin data
        bolus_df: DataFrame with bolus insulin data
        insulin_lifetime_hours: Insulin lifetime in hours
        
    Returns:
        Calculated IOB value
    """
    # Calculate time window for insulin consideration
    start_time = timestamp - timedelta(hours=insulin_lifetime_hours)
    
    # Get relevant insulin doses
    if basal_df is not None:
        basal_doses = basal_df.filter(
            (pl.col("Timestamp") >= start_time) & 
            (pl.col("Timestamp") <= timestamp)
        )
    else:
        basal_doses = pl.DataFrame()
    
    bolus_doses = bolus_df.filter(
        (pl.col("Timestamp") >= start_time) & 
        (pl.col("Timestamp") <= timestamp)
    )
    
    # Calculate remaining insulin using exponential decay
    total_iob = 0.0
    
    # Process basal doses
    if not basal_doses.is_empty():
        for row in basal_doses.iter_rows(named=True):
            time_diff = (timestamp - row["Timestamp"]).total_seconds() / 3600
            remaining_fraction = np.exp(-time_diff / insulin_lifetime_hours)
            total_iob += row.get("value", 0.0) * remaining_fraction
    
    # Process bolus doses
    if not bolus_doses.is_empty():
        for row in bolus_doses.iter_rows(named=True):
            time_diff = (timestamp - row["Timestamp"]).total_seconds() / 3600
            remaining_fraction = np.exp(-time_diff / insulin_lifetime_hours)
            total_iob += row.get("value", 0.0) * remaining_fraction
    
    return total_iob

def generate_cgm_window(df: pl.DataFrame, timestamp: datetime, 
                       window_hours: int = CONFIG["window_size_hours"]) -> Optional[np.ndarray]:
    """
    Generate a fixed-size window of CGM values before a timestamp.
    
    Args:
        df: DataFrame with CGM data
        timestamp: End time for the window
        window_hours: Window size in hours
        
    Returns:
        Numpy array with CGM values or None if window is incomplete
    """
    start_time = timestamp - timedelta(hours=window_hours)
    window = df.filter(
        (pl.col("Timestamp") >= start_time) & 
        (pl.col("Timestamp") <= timestamp)
    ).sort("Timestamp")
    
    # Check if we have enough data points
    if window.height < CONFIG["window_steps"]:
        logging.warning(f"Insufficient data points for window at {timestamp}. Found {window.height}, need {CONFIG['window_steps']}")
        return None
        
    # Get the last window_steps points and ensure float64 type
    values = window.tail(CONFIG["window_steps"]).get_column("value").cast(pl.Float64).to_numpy()
    
    # Check for missing values
    if np.any(np.isnan(values)):
        logging.warning(f"Found NaN values in window at {timestamp}")
        return None
    
    # Ensure the array is float64
    values = values.astype(np.float64)
    
    # Log window statistics
    logging.debug(f"Generated window at {timestamp}: mean={np.mean(values):.2f}, std={np.std(values):.2f}, shape={values.shape}")
        
    return values

def clip_and_normalize_time(dt: datetime) -> float:
    """
    Convert time to normalized float between 0 and 1.
    
    Args:
        dt: Datetime object
        
    Returns:
        Normalized time value
    """
    total_minutes = dt.hour * 60 + dt.minute
    return total_minutes / (24 * 60)

def load_data(data_dir: str) -> Dict[str, pl.DataFrame]:
    """
    Load data from XML files in the specified directory.
    
    Args:
        data_dir: Directory containing XML files
        
    Returns:
        Dictionary mapping data types to DataFrames
    """
    logging.info(f"Loading data from {data_dir}")
    
    # Dictionary to store DataFrames for each data type
    data_dict = {}
    
    # Process each XML file
    for xml_file in glob.glob(os.path.join(data_dir, "*.xml")):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Get subject ID from filename
            subject_id = os.path.basename(xml_file).split('.')[0]
            
            for data_type_elem in root:
                data_type = data_type_elem.tag
                if data_type == 'patient':
                    continue
                
                records = []
                for event in data_type_elem:
                    record_dict = dict(event.attrib)
                    record_dict['SubjectID'] = subject_id
                    records.append(record_dict)
                
                if records:
                    df = pl.DataFrame(records)
                    
                    # Process timestamps for all data types
                    timestamp_cols = [col for col in df.columns if any(ts in col.lower() for ts in ['ts', 'time', 'date', 'begin', 'end'])]
                    for col in timestamp_cols:
                        try:
                            # First parse to datetime
                            df = df.with_columns(pl.col(col).str.strptime(pl.Datetime('ns')).alias("Timestamp"))
                            
                            # Handle timezone if specified
                            if CONFIG["timezone"] != "UTC":
                                df = df.with_columns(
                                    pl.col("Timestamp")
                                    .dt.replace_time_zone(CONFIG["timezone"])
                                    .dt.cast_time_zone("UTC")
                                )
                            
                            # Ensure timestamp is in ns units
                            df = df.with_columns(pl.col('Timestamp').cast(pl.Datetime('ns')))
                            break
                        except Exception as e:
                            logging.warning(f"Failed to parse timestamp column {col}: {e}")
                            continue
                    
                    # Convert numeric columns
                    for col in df.columns:
                        if col not in ['Timestamp', 'SubjectID']:
                            try:
                                df = df.with_columns(pl.col(col).cast(pl.Float64))
                            except:
                                pass  # Keep as string if conversion fails
                    
                    # Add to data dictionary
                    if data_type not in data_dict:
                        data_dict[data_type] = df
                    else:
                        data_dict[data_type] = pl.concat([data_dict[data_type], df])
                    
                    logging.info(f"Processed {data_type} data: {df.shape} rows")
        
        except Exception as e:
            logging.error(f"Error processing {xml_file}: {e}")
            continue
    
    # Log final shapes
    for data_type, df in data_dict.items():
        logging.info(f"Final {data_type} shape: {df.shape}")
    
    return data_dict

def preprocess_cgm(cgm_df: pl.DataFrame) -> pl.DataFrame:
    """
    Preprocess CGM data with resampling and interpolation.
    
    Args:
        cgm_df: CGM DataFrame with Timestamp and value columns
        
    Returns:
        Preprocessed CGM DataFrame with uniform 5-minute intervals
    """
    logging.info("Starting CGM preprocessing")
    
    # Ensure timestamp is datetime and truncate to 5-min intervals
    cgm_df = cgm_df.with_columns([
        pl.col('Timestamp').dt.truncate("5m"),
        pl.col('value').cast(pl.Float64)
    ])
    
    # Sort by timestamp and subject
    cgm_df = cgm_df.sort(['SubjectID', 'Timestamp'])
    
    # Process each subject separately
    dfs = []
    for subject_id in cgm_df['SubjectID'].unique():
        logging.info(f"Processing CGM data for subject {subject_id}")
        
        # Filter data for this subject
        subject_df = cgm_df.filter(pl.col('SubjectID') == subject_id)
        
        # Get time range for this subject
        start_time = subject_df['Timestamp'].min()
        end_time = subject_df['Timestamp'].max()
        
        # Create uniform time index with 5-minute intervals
        time_range = pd.date_range(start=start_time, end=end_time, freq='5min')
        time_df = pl.DataFrame({
            'Timestamp': time_range,
            'SubjectID': subject_id
        })
        
        # Ensure timestamp types match
        subject_df = subject_df.with_columns(pl.col('Timestamp').cast(pl.Datetime('ns')))
        time_df = time_df.with_columns(pl.col('Timestamp').cast(pl.Datetime('ns')))
        
        # Join with original data
        subject_df = time_df.join(
            subject_df,
            on=['Timestamp', 'SubjectID'],
            how='left'
        )
        
        # Handle missing values:
        # 1. Linear interpolation for gaps up to 30 minutes (6 points)
        # 2. Forward fill for gaps up to 2 hours (24 points)
        # 3. Backward fill remaining gaps
        subject_df = subject_df.with_columns([
            pl.col('value')
            .cast(pl.Float64)
            .interpolate()
            .forward_fill(limit=24)
            .backward_fill()
        ])
        
        # Calculate quality metrics
        total_points = subject_df.height
        missing_points = subject_df.filter(pl.col('value').is_null()).height
        interpolated_ratio = missing_points / total_points if total_points > 0 else 1.0
        
        if interpolated_ratio > 0.3:  # More than 30% interpolated
            logging.warning(f"Subject {subject_id} has {interpolated_ratio:.1%} interpolated values")
        
        dfs.append(subject_df)
    
    # Combine all subjects
    result_df = pl.concat(dfs)
    
    # Final sort
    result_df = result_df.sort(['SubjectID', 'Timestamp'])
    
    logging.info(f"Completed CGM preprocessing. Final shape: {result_df.shape}")
    return result_df

def join_signals(data: Dict[str, pl.DataFrame]) -> pl.DataFrame:
    """
    Join all signals using timestamp and subject ID as keys.
    
    Args:
        data: Dictionary of DataFrames
        
    Returns:
        Joined DataFrame with standardized column names
    """
    logging.info("Joining signals")
    
    if 'glucose_level' not in data:
        raise ValueError("glucose_level data is required")
        
    df = data['glucose_level']
    
    # Ensure timestamp is in the correct format (nanoseconds)
    try:
        df = df.with_columns(pl.col('Timestamp').cast(pl.Datetime('ns')))
        logging.info(f"Glucose level timestamp type: {df['Timestamp'].dtype}")
    except Exception as e:
        logging.error(f"Error converting glucose_level timestamps: {e}")
        raise
    
    # Define column mappings for each data type
    column_mappings = {
        'bolus': {
            'dose': 'bolus',
            'bwz_carb_input': 'carb_input'
        },
        'meal': {
            'carbs': 'meal_carbs'
        },
        'basal': {
            'dose': 'basal_rate'
        },
        'temp_basal': {
            'dose': 'temp_basal_rate'
        },
        'work': {
            'intensity': 'work_intensity'
        },
        'sleep': {
            'quality': 'sleep_quality'
        },
        'activity': {
            'intensity': 'activity_intensity'
        }
    }
    
    # Join each signal type
    for signal_type, signal_df in data.items():
        if signal_type == 'glucose_level':
            continue
            
        logging.info(f"Joining {signal_type} data")
        
        try:
            # Ensure timestamp type matches (convert to nanoseconds)
            if 'Timestamp' in signal_df.columns:
                signal_df = signal_df.with_columns(pl.col('Timestamp').cast(pl.Datetime('ns')))
                logging.info(f"{signal_type} timestamp type before join: {signal_df['Timestamp'].dtype}")
            elif 'tbegin' in signal_df.columns:
                signal_df = signal_df.with_columns(pl.col('tbegin').str.strptime(pl.Datetime('ns')).alias('Timestamp'))
            elif 'tend' in signal_df.columns:
                signal_df = signal_df.with_columns(pl.col('tend').str.strptime(pl.Datetime('ns')).alias('Timestamp'))
            
            # Rename columns if mapping exists
            if signal_type in column_mappings:
                for old_name, new_name in column_mappings[signal_type].items():
                    if old_name in signal_df.columns:
                        signal_df = signal_df.rename({old_name: new_name})
            
            # Select columns to join
            join_keys = {'Timestamp', 'SubjectID'}
            value_cols = [col for col in signal_df.columns 
                         if col not in join_keys and col not in df.columns]
            
            if value_cols:
                signal_df = signal_df.select(['Timestamp', 'SubjectID'] + value_cols)
                
                # Verify timestamp types match before joining
                if df['Timestamp'].dtype != signal_df['Timestamp'].dtype:
                    logging.warning(f"Timestamp type mismatch before join - df: {df['Timestamp'].dtype}, signal_df: {signal_df['Timestamp'].dtype}")
                    signal_df = signal_df.with_columns(pl.col('Timestamp').cast(pl.Datetime('ns')))
                
                # Join with main dataframe
                df = df.join(
                    signal_df,
                    on=['Timestamp', 'SubjectID'],
                    how='left'
                )
                
                logging.info(f"Added columns from {signal_type}: {value_cols}")
                
        except Exception as e:
            logging.error(f"Error joining {signal_type} data: {e}")
            logging.error(f"Signal DataFrame columns: {signal_df.columns}")
            logging.error(f"Signal DataFrame dtypes: {signal_df.dtypes}")
            continue
    
    # Fill nulls appropriately
    numeric_cols = [col for col in df.columns 
                   if col not in ['Timestamp', 'SubjectID'] and 
                   df[col].dtype in (pl.Float32, pl.Float64)]
    
    df = df.with_columns([
        pl.col(col).fill_null(0.0) for col in numeric_cols
    ])
    
    logging.info(f"Completed joining signals. Final shape: {df.shape}")
    return df

def feature_engineering(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create lag features and rolling statistics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    # Ensure required columns exist
    for col in ['bolus', 'carbs', 'value']:
        if col not in df.columns:
            df = df.with_columns(pl.lit(0.0).alias(col))
    # Create lags for glucose and carbs
    for lag in [5, 10, 15, 30]:
        df = df.with_columns([
            pl.col('value').shift(lag).alias(f'glucose_lag_{lag}m'),
            pl.col('carbs').shift(lag).alias(f'carbs_lag_{lag}m')
        ])
    # Calculate rolling statistics
    df = df.with_columns([
        pl.col('value').rolling_mean(window_size=6).alias('glucose_rolling_mean_30m'),
        pl.col('value').rolling_std(window_size=6).alias('glucose_rolling_std_30m')
    ])
    # Create target variable (next bolus)
    df = df.with_columns(
        pl.col('bolus').shift(-1).alias('target_bolus')
    )
    return df

def normalize_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize continuous features using Min-Max scaling.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Normalized DataFrame
    """
    continuous_cols = [
        'value', 'glucose_lag_5m', 'glucose_lag_10m', 
        'glucose_lag_15m', 'glucose_lag_30m', 'glucose_rolling_mean_30m',
        'glucose_rolling_std_30m', 'basis_heart_rate', 'basis_gsr',
        'basis_skin_temperature', 'basis_air_temperature', 'basis_steps', 'acceleration'
    ]
    
    for col in continuous_cols:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:  # Avoid division by zero
                df = df.with_columns(
                    ((pl.col(col) - min_val) / (max_val - min_val)).alias(f'{col}_normalized')
                )
    
    return df

def plot_glucose_bolus(cgm_df: pl.DataFrame, df_final: pl.DataFrame, save_path: str = None):
    """Plot glucose time series and overlay nonzero bolus events as vertical stems."""
    plt.figure(figsize=(15, 6))
    times = cgm_df['Timestamp']
    glucose = cgm_df['value']
    plt.plot(times, glucose, label='Glucose')
    # Overlay bolus events
    bolus_events = df_final.filter(pl.col('bolus') > 0)
    if bolus_events.height > 0:
        plt.stem(bolus_events['Timestamp'], bolus_events['bolus'], linefmt="C1-", markerfmt="C1o", basefmt=" ", label="Bolus")
    plt.title('Glucose and Bolus Over Time')
    plt.xlabel('Time')
    plt.ylabel('Glucose (mg/dL)')
    plt.legend()
    # Zoom to 24h slice if possible
    if len(times) > 0:
        start = times[0]
        end = start + pd.Timedelta(hours=24)
        plt.xlim(start, end)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        plt.gcf().autofmt_xdate()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_bolus_histogram(df_final: pl.DataFrame, save_path: str = None):
    """Plot histogram of nonzero bolus doses from final features."""
    bolus = df_final.filter(pl.col('bolus') > 0)['bolus'].to_numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(bolus, bins=30, edgecolor="black")
    plt.title('Distribution of Bolus Doses')
    plt.xlabel('Bolus Dose (U)')
    plt.ylabel('Frequency')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_carbs_bolus_scatter(js: pl.DataFrame, save_path: str = None):
    """Scatter plot of meal_carbs vs bolus, colored by subject, with regression line and grid."""
    try:
        js = js.sort(['SubjectID', 'Timestamp'])
        js = js.with_columns([
            pl.col('meal_carbs').forward_fill().over('SubjectID')
        ])
        
        # Filter for valid data points with more strict criteria
        df_plot = js.filter(
            (pl.col('bolus') > 0) & 
            (pl.col('meal_carbs').is_not_null()) &
            (pl.col('meal_carbs') > 0) &
            (pl.col('meal_carbs').is_finite()) &
            (pl.col('bolus').is_finite())
        )
        
        if df_plot.height < 2:
            logging.warning("Insufficient valid data points for carbs-bolus scatter plot.")
            return
            
        carbs = df_plot['meal_carbs'].to_numpy()
        bolus = df_plot['bolus'].to_numpy()
        
        # Add small jitter to carbs for better visualization
        carbs_jitter = carbs + np.random.normal(0, 0.5, size=carbs.shape)
        subject_ids = df_plot['SubjectID'].to_numpy()
        unique_subjects, subject_numeric = np.unique(subject_ids, return_inverse=True)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(carbs_jitter, bolus, c=subject_numeric, cmap='tab20', alpha=0.6)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ticks=np.arange(len(unique_subjects)))
        cbar.ax.set_yticklabels(unique_subjects)
        cbar.set_label('Subject ID')
        
        plt.xlabel('Carbohydrates (g)')
        plt.ylabel('Bolus Dose (U)')
        plt.title('Carbohydrate Intake vs Bolus Dose')
        
        # Add regression line if we have enough valid points
        if len(carbs) > 1:
            try:
                # Use Theil-Sen estimator for robust regression
                reg = TheilSenRegressor(random_state=42)
                reg.fit(carbs.reshape(-1, 1), bolus)
                
                if np.isfinite(reg.coef_[0]) and np.isfinite(reg.intercept_):
                    x_line = np.linspace(min(carbs), max(carbs), 100)
                    y_line = reg.predict(x_line.reshape(-1, 1))
                    
                    # Calculate R² score
                    y_pred = reg.predict(carbs.reshape(-1, 1))
                    r2 = r2_score(bolus, y_pred)
                    
                    plt.plot(x_line, y_line, 'r--', 
                            label=f'Robust fit (R²={r2:.2f})')
                    plt.legend()
            except Exception as e:
                logging.warning(f"Could not fit regression line: {e}")
        
        plt.grid(True, linestyle='--', alpha=0.5)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logging.error(f"Error in plot_carbs_bolus_scatter: {e}")
        if save_path and plt.get_fignums():
            plt.close()

def plot_glucose_boxplot(cgm_df: pl.DataFrame, df_final: pl.DataFrame, save_path: str = None):
    """Boxplot of glucose 30 min before and after bolus events."""
    # For each bolus event, get glucose 30 min before and after
    before = []
    after = []
    for row in df_final.filter(pl.col('bolus') > 0).iter_rows(named=True):
        ts = row['Timestamp']
        # 30 min before
        before_vals = cgm_df.filter((pl.col('Timestamp') >= ts - pd.Timedelta(minutes=30)) & (pl.col('Timestamp') < ts))['value'].to_numpy()
        if len(before_vals) > 0:
            before.append(before_vals)
        # 30 min after
        after_vals = cgm_df.filter((pl.col('Timestamp') > ts) & (pl.col('Timestamp') <= ts + pd.Timedelta(minutes=30)))['value'].to_numpy()
        if len(after_vals) > 0:
            after.append(after_vals)
    # Flatten arrays
    before_flat = np.concatenate(before) if before else np.array([])
    after_flat = np.concatenate(after) if after else np.array([])
    plt.figure(figsize=(10, 6))
    plt.boxplot([before_flat, after_flat], tick_labels=["Before", "After"])
    plt.title('Glucose Levels 30 min Before and After Bolus Events')
    plt.xlabel('Event Window')
    plt.ylabel('Glucose (mg/dL)')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_correlation_heatmap(df_final: pl.DataFrame, save_path: str = None):
    """Correlation heatmap of final numeric features, removing NaN values."""
    # Select numeric columns, excluding CGM columns
    numeric_cols = [col for col, dtype in zip(df_final.columns, df_final.dtypes) 
                   if dtype in (pl.Float32, pl.Float64) and not col.startswith('cgm_')]
    
    if not numeric_cols:
        logging.warning("No numeric columns found for correlation heatmap.")
        return
        
    df_pd = df_final.select(numeric_cols).to_pandas()
    
    # Calculate correlation matrix with error handling
    try:
        corr = df_pd.corr()
        # Remove rows and columns that are all NaN
        corr = corr.dropna(axis=0, how='all').dropna(axis=1, how='all')
        
        if corr.empty:
            logging.warning("No valid correlations found for heatmap.")
            return
            
        plt.figure(figsize=(12, 8))
        
        # Use a diverging colormap centered at 0
        im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, fraction=0.046, pad=0.04)
        
        plt.title('Feature Correlation Heatmap')
        plt.xticks(ticks=np.arange(len(corr.columns)), labels=corr.columns, rotation=45, ha='right')
        plt.yticks(ticks=np.arange(len(corr.index)), labels=corr.index)
        
        # Annotate each cell with correlation value
        for i in range(len(corr.index)):
            for j in range(len(corr.columns)):
                value = corr.iloc[i, j]
                if np.isfinite(value):
                    plt.text(j, i, f"{value:.2f}", ha="center", va="center", 
                            color="black", fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
    except Exception as e:
        logging.error(f"Error creating correlation heatmap: {e}")

def export_data(df: pl.DataFrame, output_path: str):
    """
    Export processed data to Parquet format.
    
    Args:
        df: Processed DataFrame
        output_path: Path to save the data
    """
    df.write_parquet(output_path)

def generate_windows(df: pl.DataFrame, window_size: int = CONFIG["window_steps"]) -> pl.DataFrame:
    """
    Generate sliding windows of CGM data for each subject.
    
    Args:
        df: DataFrame with preprocessed CGM data
        window_size: Number of steps in each window
        
    Returns:
        DataFrame with CGM windows and associated features
    """
    logging.info("Generating CGM windows")
    
    if 'bolus' not in df.columns:
        raise ValueError("Column 'bolus' not found in DataFrame")
    
    windows = []
    total_bolus_events = 0
    for subject_id in df['SubjectID'].unique():
        logging.info(f"Processing windows for subject {subject_id}")
        subject_df = df.filter(pl.col('SubjectID') == subject_id)
        bolus_events = subject_df.filter(pl.col('bolus') > 0)
        total_bolus_events += bolus_events.height
        logging.info(f"Found {bolus_events.height} bolus events for subject {subject_id}")
        for row in bolus_events.iter_rows(named=True):
            window = generate_cgm_window(subject_df, row['Timestamp'])
            if window is not None:
                windows.append({
                    'SubjectID': subject_id,
                    'Timestamp': row['Timestamp'],
                    'cgm_window': window.tolist(),
                    'bolus': row['bolus'],
                    'carb_input': row.get('carb_input', 0.0),
                    'meal_carbs': row.get('meal_carbs', 0.0),
                    'basal_rate': row.get('basal_rate', 0.0),
                    'temp_basal_rate': row.get('temp_basal_rate', 0.0)
                })
    logging.info(f"Total bolus events considered for windows: {total_bolus_events}")
    if not windows:
        raise ValueError("No valid windows generated")
    result_df = pl.DataFrame(windows)
    # Diagnostics for nonzero events
    for col in ['bolus', 'carb_input', 'meal_carbs']:
        if col in result_df.columns:
            nonzero = result_df.filter(pl.col(col) > 0).height
            logging.info(f"Nonzero {col} events after generate_windows: {nonzero}")
    logging.info(f"Generated {len(windows)} windows")
    return result_df

def extract_features(df: pl.DataFrame, meal_df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """
    Extract features from the joined DataFrame.
    
    Args:
        df: Joined DataFrame
        meal_df: Optional DataFrame with meal data
        
    Returns:
        DataFrame with extracted features
    """
    logging.info("Extracting features")
    
    # Calculate hour of day using native Polars expressions
    df = df.with_columns([
        ((pl.col('Timestamp').dt.hour() * 60 + pl.col('Timestamp').dt.minute()) / (24 * 60)).alias('hour_of_day')
    ])
    
    # Get last CGM value as bg_input - handle both list and array types
    if 'cgm_window' in df.columns:
        df = df.with_columns([
            pl.when(pl.col('cgm_window').list.len() > 0)
            .then(pl.col('cgm_window').list.get(-1))
            .otherwise(0.0)
            .alias('bg_input')
        ])
    
    # Add meal features if available
    if meal_df is not None:
        # Ensure timestamps are in the same format (nanoseconds)
        meal_df = meal_df.with_columns(pl.col('Timestamp').cast(pl.Datetime('ns')))
        df = df.with_columns(pl.col('Timestamp').cast(pl.Datetime('ns')))
        
        # Create a list to store the matched meals
        matched_meals = []
        
        # Process each bolus event
        for row in df.iter_rows(named=True):
            bolus_time = row['Timestamp']
            
            # Look for meals within 1 hour after the bolus
            start_time = bolus_time
            end_time = bolus_time + timedelta(hours=1)
            
            # Find meals within the time window
            meals_in_window = meal_df.filter(
                (pl.col('Timestamp') >= start_time) & 
                (pl.col('Timestamp') <= end_time)
            )
            
            if meals_in_window.height > 0:
                # Get the closest meal
                meals_in_window = meals_in_window.with_columns([
                    (pl.col('Timestamp') - bolus_time).alias('time_diff')
                ]).sort('time_diff')
                
                # Use .to_dicts()[0] to get a dict
                closest_meal = meals_in_window.to_dicts()[0]
                matched_meals.append({
                    'Timestamp': bolus_time,
                    'meal_carbs': closest_meal.get('carbs', 0.0),
                    'meal_time_diff': closest_meal.get('time_diff', None).total_seconds() / 60 if closest_meal.get('time_diff', None) is not None else None,  # in minutes
                    'meals_in_window': meals_in_window.height  # Number of meals found in window
                })
            else:
                matched_meals.append({
                    'Timestamp': bolus_time,
                    'meal_carbs': 0.0,
                    'meal_time_diff': None,
                    'meals_in_window': 0
                })
        
        # Create DataFrame from matched meals
        meals_df = pl.DataFrame(matched_meals)
        
        # Ensure Timestamp is pl.Datetime('ns') in both DataFrames before join
        meals_df = meals_df.with_columns(pl.col('Timestamp').cast(pl.Datetime('ns')))
        df = df.with_columns(pl.col('Timestamp').cast(pl.Datetime('ns')))
        
        # Join with main DataFrame
        df = df.join(
            meals_df,
            on='Timestamp',
            how='left'
        )
        
        # Fill missing values
        df = df.with_columns([
            pl.col('meal_carbs').fill_null(0.0),
            pl.col('meal_time_diff').fill_null(0.0),
            pl.col('meals_in_window').fill_null(0)
        ])
        
        # Add features about meal timing
        df = df.with_columns([
            # Whether there was a meal in the window
            pl.when(pl.col('meals_in_window') > 0)
            .then(1.0)
            .otherwise(0.0)
            .alias('has_meal'),
            
            # Normalized time difference (in hours)
            (pl.col('meal_time_diff') / 60.0).alias('meal_time_diff_hours')
        ])
        
    else:
        df = df.with_columns([
            pl.lit(0.0).alias('meal_carbs'),
            pl.lit(0.0).alias('meal_time_diff'),
            pl.lit(0.0).alias('meal_time_diff_hours'),
            pl.lit(0.0).alias('has_meal'),
            pl.lit(0).alias('meals_in_window')
        ])
    
    # Normalize optional features
    optional_features = ['work_intensity', 'sleep_quality', 'activity_intensity']
    for feature in optional_features:
        if feature in df.columns:
            # Normalize only non-null values
            df = df.with_columns([
                pl.when(pl.col(feature).is_not_null())
                .then(pl.col(feature) / CONFIG[f'max_{feature}'])  # Normalize to [0,1]
                .otherwise(None)
                .alias(feature)
            ])
    
    # Fill any remaining NaN or infinite values with 0 (except optional features)
    numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) 
                   if dtype in (pl.Float32, pl.Float64) and col not in optional_features]
    df = df.with_columns([
        pl.when(pl.col(col).is_finite())
        .then(pl.col(col))
        .otherwise(0.0)
        for col in numeric_cols
    ])
    
    logging.info(f"Completed feature transformations. Shape: {df.shape}")
    return df

def transform_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Apply final transformations to features.
    
    Args:
        df: DataFrame with extracted features
        
    Returns:
        DataFrame with transformed features
    """
    logging.info("Applying feature transformations")
    
    def log1p_array(x):
        """Apply log1p transformation to array data."""
        try:
            if isinstance(x, (list, np.ndarray)):
                x = np.array(x, dtype=np.float64)
                if np.any(np.isnan(x)) or np.any(np.isinf(x)):
                    logging.warning("Found NaN or inf values in array, replacing with zeros")
                    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                return np.log1p(x).tolist()
            elif isinstance(x, (int, float)):
                # Handle single float/int values
                return np.log1p(float(x))
            else:
                logging.error(f"Unexpected type in log1p_array: {type(x)}, value: {x}")
                return 0.0
        except Exception as e:
            logging.error(f"Error in log1p_array: {str(e)}, type: {type(x)}, value: {x}")
            return 0.0
    
    # Process CGM windows if present
    if 'cgm_window' in df.columns:
        logging.info("Processing CGM windows")
        try:
            # Convert CGM windows to individual columns
            window_size = CONFIG["window_steps"]
            cgm_cols = [f'cgm_{i}' for i in range(window_size)]
            
            # Extract CGM values into separate columns
            df = df.with_columns([
                pl.col('cgm_window').list.get(i).alias(f'cgm_{i}')
                for i in range(window_size)
            ])
            
            # Apply log1p transformation to each CGM column with explicit return dtype
            for col in cgm_cols:
                df = df.with_columns([
                    pl.col(col).map_elements(log1p_array, return_dtype=pl.Float64).alias(f'{col}_log')
                ])
            
            # Calculate glucose trend and variability
            if all(col in df.columns for col in cgm_cols):
                df = df.with_columns([
                    # Trend: difference between last and first value
                    (pl.col('cgm_23') - pl.col('cgm_0')).alias('glucose_trend'),
                    # Variability: standard deviation
                    pl.concat_list([pl.col(col) for col in cgm_cols]).list.std().alias('glucose_variability')
                ])
            
            # Drop original CGM window column
            df = df.drop('cgm_window')
            logging.info("Successfully processed CGM windows")
            
        except Exception as e:
            logging.error(f"Error processing CGM windows: {str(e)}")
            # Keep original data if processing fails
            pass
    
    # Normalize optional features if they exist
    optional_features = ['work_intensity', 'sleep_quality', 'activity_intensity']
    for feature in optional_features:
        if feature in df.columns:
            df = df.with_columns([
                pl.when(pl.col(feature).is_not_null())
                .then(pl.col(feature) / CONFIG[f'max_{feature}'])
                .otherwise(None)
                .alias(feature)
            ])
    
    # Fill any remaining NaN or infinite values with 0 (except optional features)
    numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) 
                   if dtype in (pl.Float32, pl.Float64) and col not in optional_features]
    df = df.with_columns([
        pl.when(pl.col(col).is_finite())
        .then(pl.col(col))
        .otherwise(0.0)
        for col in numeric_cols
    ])
    
    logging.info(f"Completed feature transformations. Shape: {df.shape}")
    return df

def main():
    """Main function to process OhioT1DM dataset."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process OhioT1DM dataset')
    parser.add_argument('--data-dirs', nargs='+', 
                      default=['data/OhioT1DM/2018/train', 'data/OhioT1DM/2020/train',
                              'data/OhioT1DM/2018/test', 'data/OhioT1DM/2020/test'],
                      help='List of data directories to process')
    parser.add_argument('--output-dir', default='new_ohio/processed_data',
                      help='Output directory for processed data')
    parser.add_argument('--plots-dir', default='new_ohio/processed_data/plots',
                      help='Output directory for plots')
    parser.add_argument('--timezone', default='UTC',
                      help='Timezone to use for timestamps')
    parser.add_argument('--n-jobs', type=int, default=-1,
                      help='Number of parallel jobs (-1 for all cores)')
    args = parser.parse_args()
    
    # Update config with command line arguments
    CONFIG['timezone'] = args.timezone
    
    # Create output directories
    Path(args.output_dir).mkdir(exist_ok=True)
    Path(args.plots_dir).mkdir(exist_ok=True, parents=True)
    
    # Process each data directory
    for data_dir in args.data_dirs:
        logging.info(f"\nProcessing directory: {data_dir}")
        logging.info("=" * 50)
        
        try:
            # Load data
            data = load_data(data_dir)
            
            if not data:
                logging.error(f"No data loaded from {data_dir}")
                continue
            
            # Process CGM data
            if 'glucose_level' in data:
                # Preprocess CGM
                data['glucose_level'] = preprocess_cgm(data['glucose_level'])
                
                # Join all signals
                df = join_signals(data)
                
                # Generate windows around bolus events
                df_windows = generate_windows(df)
                
                # Extract features
                df_features = extract_features(df_windows, data.get('meal'))
                
                # Transform features
                df_final = transform_features(df_features)
                
                # Determine output path based on whether it's train or test data
                is_test = 'test' in data_dir
                output_subdir = 'test' if is_test else 'train'
                output_path = f"{args.output_dir}/{output_subdir}/processed_{Path(data_dir).name}.parquet"
                
                # Create subdirectory if it doesn't exist
                Path(f"{args.output_dir}/{output_subdir}").mkdir(exist_ok=True)
                
                # Export processed data
                export_data(df_final, output_path)
                logging.info(f"Exported processed data to {output_path}")
                
                # Generate plots
                logging.info("Generating plots...")
                
                # Time series plot
                plot_glucose_bolus(data['glucose_level'], df_final, f"{args.plots_dir}/glucose_bolus_{Path(data_dir).name}.png")
                # Distribution plots
                plot_bolus_histogram(df_final, f"{args.plots_dir}/bolus_hist_{Path(data_dir).name}.png")
                plot_carbs_bolus_scatter(df, f"{args.plots_dir}/carbs_bolus_scatter_{Path(data_dir).name}.png")
                plot_glucose_boxplot(data['glucose_level'], df_final, f"{args.plots_dir}/glucose_boxplot_{Path(data_dir).name}.png")
                # Correlation heatmap of final features
                plot_correlation_heatmap(df_final, f"{args.plots_dir}/correlation_heatmap_{Path(data_dir).name}.png")
                
                logging.info("Completed plot generation")
                
            else:
                logging.error(f"No glucose level data found in {data_dir}")
                
        except Exception as e:
            logging.error(f"Error processing {data_dir}: {e}")
            continue

if __name__ == "__main__":
    main()
