import onnx
import onnxruntime as rt
from processing import Predict
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob


file_path = "onnx_timing_model.onnx" # path to ONNX model file

onnx_model = onnx.load(file_path) #  load saved ONNX model

providers = ['CPUExecutionProvider']
model = rt.InferenceSession(file_path, providers=providers)
output_names = ['lstm_model'] # output layer name

SEQUENCE_LENGTH = 30
STRIDE = 1

filenames = glob.glob('timing_videos/*.avi') # path to videos

final_predictions = []

for file in tqdm(filenames):

    file_path = file
    
    predict = Predict(file_path, SEQUENCE_LENGTH, STRIDE) # Data management class object
    
    frames = predict.get_frames()
    
    image_sequence = predict.get_image_sequence(frames)
    
    # Input chunked to sequence length by window and stride
    chunked_sequence = predict.get_chunked_sequence(image_sequence)
    
    # create empty np array for predictions
    pred=np.arange(int(len(image_sequence)),dtype=float)
    pred=np.full_like(pred,np.nan,dtype=float)
    
    # run sliding window predictions with stride
    start=0
    end = SEQUENCE_LENGTH

    # Generate prediction for each chunked sequence
    for i in range(len(chunked_sequence)):
      tempArr=np.arange(int(len(image_sequence)),dtype=float)
      tempArr=np.full_like(tempArr,np.nan,dtype=float)
      prediction = model.run(output_names, {"InputLayer": np.expand_dims(chunked_sequence[i], axis=0)}) # Run prediction
      prediction = prediction[0][0]
      tempArr[start:end]=prediction
      pred=np.vstack([pred,tempArr])
      start+=STRIDE
      end+=STRIDE

    # Calculate the mean of all predictions
    mean = np.nanmean(pred,axis=0)
    
    # remove padded frames from predictions
    predictions = np.resize(mean, mean.size-predict.num_padded_frames)
    
    # Get predictions for ED and ES phases
    ED_predictions, ES_predictions = predict.get_predictions(predictions)
    
    final_predictions.append([file, "ED", ED_predictions])
    final_predictions.append([file, "ES", ES_predictions])

# Convert final predictions to dataframe    
df = pd.DataFrame(final_predictions)
# Save to csv - or change to other file type
df.to_csv("predictions.csv", index=False)

