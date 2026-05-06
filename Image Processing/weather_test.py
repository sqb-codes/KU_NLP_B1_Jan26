import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("weather_model.h5")
print("Model Loaded Successfully...")

class_names = ['dew', 'fogsmog', 'frost', 'glaze', 'hail', 'lightning', 'rain', 'rainbow', 'rime', 'sandstorm', 'snow']

cap = cv2.VideoCapture("weather_vid.mp4")

while True:
    flag, frame = cap.read()
    if not flag:
        break
    # resize to 128,128 because images are trained on this size only
    img = cv2.resize(frame, (128, 128))
    # Convert frame into numpy array
    img_array = np.array(img)
    # Normalization - normalize pixel values(same as training)
    img_array = img_array / 255.0
    # Expand dimensions - (1,128,128,3)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_label = class_names[predicted_index]
    confidence_score = np.max(predictions[0])
    cv2.putText(frame, f"{predicted_label} : {confidence_score:.2f}",
                (20,40), cv2.FONT_HERSHEY_COMPLEX, 2,
                (0,255,0), 2)
    cv2.imshow("Weather detection :",frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
