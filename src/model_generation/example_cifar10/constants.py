BATCH_SIZE = 32
FEATURE_SHAPE = [32, 32, 3]  # A list that defines the shape of the input of your model
OUTPUT_SHAPE = [10]  # A list that defines the shape of the output of your model
TFLITE_MODEL_FILE = "model_cifar10.tflite"  # File where TFLite model will be saved
SAVED_MODEL_DIR = "saved_model_cifar10"  # Directory where the keras model will be saved
FEATS_BIN_FILE = "cifar10_feats.bin"
LABELS_BIN_FILE = "cifar10_labels.bin"
RUN_TFLITE_TRAINING = True  # Either to test the training function in the TFLite model
