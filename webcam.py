import cv2
import os

def take_photo(filename='photo.jpg'):
    # Specify the save path
    save_directory = r'C:\Users\Fuzlan\OneDrive\Documents\Aifusion\krish\Gen-AI-With-Deep-Seek-R1\photos'
    # Ensure the directory exists
    os.makedirs(save_directory, exist_ok=True)
    # Full file path
    full_path = os.path.join(save_directory, filename)

    # Open the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open camera")
        return None

    print("Press 'c' to capture the photo, or 'q' to quit.")
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Display the live webcam feed
        cv2.imshow('Webcam', frame)

        # Wait for user input
        key = cv2.waitKey(1)
        if key == ord('c'):  # 'c' to capture
            cv2.imwrite(full_path, frame)
            print(f"Photo saved as {full_path}")
            break
        elif key == ord('q'):  # 'q' to quit
            print("Exiting without saving.")
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to take a photo
try:
    filename = 'photo.jpg'
    take_photo(filename=filename)
    # Display the saved image
    saved_image = cv2.imread(os.path.join(r'C:\Users\Fuzlan\OneDrive\Documents\Aifusion\krish\Gen-AI-With-Deep-Seek-R1\photos', filename))
    if saved_image is not None:
        cv2.imshow('Saved Photo', saved_image)
        print(f"Displaying saved photo: {filename}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
except Exception as e:
    print(f"An error occurred: {e}")
