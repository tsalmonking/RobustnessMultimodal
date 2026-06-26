# Models
WEIGHTS_PATH = "model/clip-vit-large-patch14_None_8_8_0.4_True10_best.pt"
NAME_LLM = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
NAME_IMG_EMBED = "openai/clip-vit-large-patch14"
FF_WEIGHTS_PATH = "model/clip-vit-base-patch32_None_8_8_0.4_True10_best.pt"
FF_NAME_IMG_EMBED = "openai/clip-vit-base-patch32"

# Model parameters
BATCH_SIZE = 4
N_TOKENS = 512
THRESHOLD = 0.5

# Attack parameters
SOURCE_LABEL = 0 # Fake
TARGET_LABEL = 1 # Real
## Image attack parameters
PGD_ITERS = 25
EPSILON = 3 / 255
ALPHA_FACTOR = 2.0
## Textual attack parameters
K_BERT_ATTACK = 48 # Number of candidates to consider for each word in the attack
THRESHOLD_PRED_SCORE = 0
MAX_WORDS_TO_ATTACK = 512
MAX_CANDIDATES_PER_WORD = 32 # Maximum number of candidates to consider for each word in the attack
MAX_WORDS_FOR_IMPORTANCE = 512
