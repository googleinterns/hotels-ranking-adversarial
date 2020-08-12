"""Constants to be used across files"""

# Store the paths to files containing training and test instances.
_TRAIN_DATA_PATH = "train.tfrecords"
_TEST_DATA_PATH = "test.tfrecords"

# Store the vocabulary path for query and document tokens.
_VOCAB_PATH = "vocab.txt"

# The maximum number of documents per query in the dataset.
# Document lists are padded or truncated to this size.
_LIST_SIZE = 50

# The document relevance label.
_LABEL_FEATURE = "relevance"

# Padding labels are set negative so that the corresponding examples can be
# ignored in loss and metrics.
_PADDING_LABEL = -1

# Learning rate for optimizer.
_LEARNING_RATE = 0.05

# Parameters to the scoring function.
_BATCH_SIZE = 1
_HIDDEN_LAYER_DIMS = ["64", "32", "16"]
_DROPOUT_RATE = 0.8
_GROUP_SIZE = 1  # Pointwise scoring.

# Location of model directory and number of training steps.
_MODEL_DIR = "/tmp/ranking_model_dir"
_NUM_TRAIN_STEPS = 16000

_EMBEDDING_DIMENSION = 20
_FULL_EMBEDDING = _EMBEDDING_DIMENSION * 2 #embeddings to encode both question and answer