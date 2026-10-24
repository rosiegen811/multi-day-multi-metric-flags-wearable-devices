# FILES
- 'Stress_Level_v1.csv' and 'Stress_Level_v2.csv' contain self-reported stress levels during the stress protocol for each stage of the study,  respectively.
- 'subject-info.csv' contains demographic data such as age, weight, and height for all participants.
- 'data_constraints.txt' contains detailed information of issues such as incorrect wristband placements, incomplete protocols, and connection problems. 
- 'Wearable_Dataset.ipynb'  is a Jupyter Notebook that provides visualizations of all recorded signals and includes sample code for managing the data.
- The main folder 'Wearable_Dataset' contains three folders for each type of activity: 'STRESS', 'AEROBIC' and 'ANAEROBIC'.
- Each of these folders includes participant subfolders named as 'S01','S02','f01',etc.
- Each of the folders corresponding to each participant contains Empatica E4 files: 'ACC.csv', 'BVP.csv', 'EDA.csv', 'HR.csv', 'IBI.csv', 'tags.csv' and 'TEMP.csv'.

# RELEVANT INFO
- Before downloading, we recommend reviewing the Jupyter Notebook ('Wearable_Dataset.ipynb') to determine whether the dataset fits your needs. This file is provided to read, open, visualize and start working with the data. To execute the notebook, ensure that basic Python libraries such as pandas,os, numpy, time, and matplotlib are installed.
- Participants from the first stage are labeled as 'Sxx' while those from the second stage are labeled as 'fxx'.
- Session dates and event marks are expressed in UTC and have been adjusted for de-identification. The time shifts have been applied consistently across all records to maintain signal alignment. 
- Empatica signal files: the first row is the initial time of the session expressed in UTC (Empatica provides time in Unix timestamp format, but files are already converted to UTC). The second row is the sample rate expressed in Hz.
- 'ACC.csv': data from x, y, and z axis are stored in first, second, and third column, respectively.
- 'IBI.csv': the first column is the time (respect to the initial time) of the detected inter-beat interval expressed in seconds (s). The second column is the duration in seconds (s) of the detected inter-beat interval (i.e., the distance in seconds from the previous beat).


