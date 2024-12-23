class Config:

    #Globals
     #Globals
    batch_size = 128
    num_classes = 2  # classes, seizure/no seizure
    epochs = 10     # Epoch iterations
    time_step_length = 5
    row_hidden = 128  # hidden neurons in conv layers
    col_hidden = 128   # hidden neurons in the Bi-LSTM layers
    RANDOM_SEED = 333    
    N_TIME_STEPS = 125   # 50 records in each sequence
    N_FEATURES = 3     # mag,hr,roi_Ratio,output
    step = 100           # window overlap = 50 -10 = 40  (80% overlap)
    N_CLASSES = 2      # class labels
  