import threading
import numpy as np
import pandas as pd

emo_lock = threading.Condition(threading.Lock())
behav_lock = threading.Condition(threading.Lock())

emo_smooth_lock = threading.Condition(threading.Lock())
behav_smooth_lock = threading.Condition(threading.Lock())

emo_assign_lock = threading.Condition(threading.Lock())
behav_assign_lock = threading.Condition(threading.Lock())

# Feature array
emotion_feature_list = []
behavior_feature_list = []

# Smoothing weight array
emotion_smooth_list = []
behavior_smooth_list = []

# Assign Hits weight array
emotion_assign_list = []
behavior_assign_list = []

final_list = []

INPUT_SIZE = 3800

# Colab
emotion_feature = np.load("/content/drive/My Drive/Dataset/FEATURES/Train_features/emotion.npy")
behaviour_feature = np.load("/content/drive/My Drive/Dataset/FEATURES/Train_features/behavior.npy")
emotion_label = np.load("/content/drive/My Drive/Dataset/FEATURES/Train_features/emotion_label.npy")
behaviour_label = np.load("/content/drive/My Drive/Dataset/FEATURES/Train_features/behavior_label.npy")
threat_label = np.load("/content/drive/My Drive/Dataset/FEATURES/Train_features/threat_label.npy")

test_behaviour_feature = np.load("/content/drive/My Drive/Dataset/FEATURES/Test_features/test_behaviour.npy")
test_behaviour_label = np.load("/content/drive/My Drive/Dataset/FEATURES/Test_features/test_behaviour_labels.npy")
test_emotion_feature = np.load("/content/drive/My Drive/Dataset/FEATURES/Test_features/test_emotion.npy")
test_emotion_label = np.load("/content/drive/My Drive/Dataset/FEATURES/Test_features/test_emotion_labels.npy")
test_threat_label = np.load("/content/drive/My Drive/Dataset/FEATURES/Test_features/test_threat_labels.npy")
# PC
# emotion_feature = np.load("data/behavior.npy")
# behaviour_feature = np.load("data/behavior.npy")
# emotion_label = np.load("data/emotion_label.npy")
# behaviour_label = np.load("data/behavior_label.npy")
# threat_label = np.load("data/threat_label.npy")



# Random samples
# emotion_feature = np.random.rand(INPUT_SIZE,10)
# behaviour_feature = np.random.rand(INPUT_SIZE,10)
# emotion_label = np.random.randint(2, size=INPUT_SIZE)
# behaviour_label = np.random.randint(2, size=INPUT_SIZE)
# threat_label = np.random.randint(2, size=INPUT_SIZE)

# Zoo data
# data = pd.read_csv("data/zoo-mini.csv")
# label = data.iloc[:,-1].values
# data = data.iloc[:,1:-1].values
#
# emotion_feature = data[:,:8]
# behaviour_feature = data[:,8:]
# emotion_label = label
# behaviour_label = label
# threat_label = label
#
# INPUT_SIZE = data.shape[0]
