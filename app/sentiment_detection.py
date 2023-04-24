import numpy as np
import ktrain

class Emotion:
    def __init__(self):
        self.predictor = None

    def load_model(self):
        # Load predictor
        self.predictor = ktrain.load_predictor('./text classification/saved_model')

    def prediction(self, text):
        return self.predictor.predict(text)


if __name__ == '__main__':
    emotion = Emotion()

    emotion.load_model()

    inputArray = []

    while True:
        query = input("> ")

        currentEmotion = emotion.prediction(query)

        inputArray.append(currentEmotion)

        # emotion capture of 3 last inputs
        if len(inputArray) == 3:
            unique, counts = np.unique(inputArray, return_counts=True)

            # create a dict with the emotions and their occurrences
            emotion_dict = dict(zip(unique, counts))
            print(emotion_dict)

            # get the dominant emotion
            dominantEmotion = max(emotion_dict, key=emotion_dict.get)
            print(f"dominant emotion: {dominantEmotion}")

            inputArray.clear()

        if query in 'q':
            break