from osdb_data_loader import OsdbDataLoader
from config1 import Config
from osdb_data_reshaper import OsdbDataReshaper
from data_interpolator import DataInterpolator
from data_annotator import DataAnnotator
from osdb_label_generator import OsdbLabelGenerator

def main():
    file_path = "Data/osdb_3min_allSeizures.json"
    #labels = 'Scripts/ipd_dataset.csv'  # Provide your actual file path
    #osdb_dl = OsdbDataLoader(file_path=osdb_dataset, time_steps=Config.N_TIME_STEPS)
    # Access the DataFrame
    #df_sensordata = osdb_dl.df_sensordata   
      
    processor = OsdbLabelGenerator(file_path)
    # Process the data and get the resulting DataFrame
    df_sensordata = processor.process_data()
    data_reshaper = OsdbDataReshaper(df_sensordata)
    df_reshaped = data_reshaper.reshape_data()
    
    # Initialize Interpolator and interpolate the 'hr' column
    interpolator = DataInterpolator(df_reshaped, column_to_interpolate="hr")
    interpolator.interpolate_column(new_column_name="interpolated_hr", interval=125, time_step=5)
        
    # Retrieve the updated DataFrame
    updated_df = interpolator.get_dataframe()
    print(updated_df[995:1005])

if __name__ == "__main__":
    main()