from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Flatten, Dense

# Rebuild the original SID220 architecture
model = Sequential([
    InputLayer(shape=(21, 3)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(26, activation='linear')
])

# Load pre-trained weights from your .h5 file
model.load_weights("asl-now-weights.h5")

# Compile model for prediction
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"]
)

model.save("sid220_rebuilt_model.keras")
