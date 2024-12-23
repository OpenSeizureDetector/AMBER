from config import Config as config
from OsdbDataProcessor.osdb_data_label_generator import OsdbDataLabelGenerator
from OsdbDataProcessor.osdb_data_reshaper import OsdbDataReshaper
from OsdbDataProcessor.osdb_interpolator import OsdbInterpolator
from OsdbDataProcessor.osdb_data_loader import OsdbDataLoader

if __name__ == "__main__":
    
    # Example usage:
    file_path = 'Data/osdb_3min_allSeizures.json'  # Replace with your JSON file path
    processor = OsdbDataLabelGenerator(file_path)
    df_result = processor.process_data()
    data_reshaper = OsdbDataReshaper(df_result)
    reshaped_df = data_reshaper.reshape_data()
    # Initialize Interpolator and interpolate the 'hr' column
    interpolator = OsdbInterpolator(reshaped_df, column_to_interpolate="hr")
    interpolator.interpolate_column(new_column_name="interpolated_hr", interval=config.N_TIME_STEPS, time_step=config.time_step_length)
    # Retrieve the updated DataFrame
    interpolated_df = interpolator.get_dataframe()    
    print(interpolated_df)
    