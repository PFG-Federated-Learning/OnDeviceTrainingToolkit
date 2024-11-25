BATCH_SIZE = 64
FEATURE_SHAPE = [28,28,1] # A list that defines the shape of the input of your model
OUTPUT_SHAPE = [10] # A list that defines the shape of the output of your model
TFLITE_MODEL_FILE = "model_mnist.tflite" # File where TFLite model will be saved
SAVED_MODEL_DIR = "saved_model" # Directory where the keras model will be saved
FEATS_BIN_FILE = "mnist_feats.bin"
LABELS_BIN_FILE = "mnist_labels.bin"