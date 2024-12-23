## 📁 **OSDB Data Processor Component** 

The **OSDB Data Processor** component is designed to process raw JSON data from the **OpenSeizure Database (OSDB)**. The component includes several modular classes located in the `OsdbDataProcessor` folder, which form a data processing pipeline. The pipeline loads raw **JSON** data, reshapes the data for timeseries analysis, runs an interpolator over the heart rate attribute and outputs a dataframe that can be integrated with the **AMBER Model**. 

### 📂 **OsdbDataProcessor Overview**

- **`Config`**: Defines the configuration parameters required throughout the data pipeline (e.g., batch size, time steps, and sampling rate).
- **`OsdbDataLoader`**: Connects to the **OSDB** and reads raw **JSON** data to generate an unlabelled dataframe of sensor data (acceleration (vector magntiude 1D and fast fourier transform **FFT**), heart rate and sp02)
- **`OsdbLabelGenerator`**:  Connects to the **OSDB** and reads raw **JSON** data to generate a labelled dataframe using the **seizureTimes** attribute
- **`OsdbDataReshaper`**: Reshapes the generated dataframe into a time series format for input into the **AMBER** model.
- **`Interpolator`**: Interpolates duplcaite **hr** values, applying cubic spline interpolation to ensure smooth and accurate data continuity between the timestep t and t+1


---

### **`How to Run the Data Processor (main.py)`** 🔍
#### 1. **Import OSDB Data Processing Classes**:
``` python
from config import Config
from OsdbDataProcessor.osdb_data_label_generator import OsdbDataLabelGenerator
from OsdbDataProcessor.osdb_data_reshaper import OsdbDataReshaper
from OsdbDataProcessor.osdb_interpolator import OsdbInterpolator
from OsdbDataProcessor.osdb_data_loader import OsdbDataLoader
```
#### 2. **Set path to the OSDB JSON file**:
``` python
file_path = 'Data/osdb_3min_allSeizures.json'  # Replace with your JSON file path
```

#### 2. **Load OSDB cata label generator and pass file path**:
``` python
osdb_processor = OsdbDataLabelGenerator(file_path)
df_result = processor.process_data()
```

#### 3. **Reshape the Dataframe for Timeseries Analysis**:
``` python
data_reshaper = OsdbDataReshaper(df_result)
reshaped_df = data_reshaper.reshape_data()
```

#### 4. **Initialise the Interpolator and interpolate the 'hr' column**:
``` python
interpolator = OsdbInterpolator(reshaped_df, column_to_interpolate="hr")
interpolator.interpolate_column(new_column_name="interpolated_hr", interval=config.N_TIME_STEPS, time_step=config.time_step_length)
# Retrieve the updated DataFrame
dataset_df = dataset_df.get_dataframe()    
print(dataset_df.sample())#print sample row
```
---

2. **Open Root from Integrated Terminal**:
```bash
path/AMBER/AMBER main python.py
```

---

