import cv2
import os
import numpy as np
import face_recognition
from numba import njit

# Paths
video_path = 'videoooooooooo.mp4'
known_faces_path = 'known_faces/'
alerts_path = 'alerts/'  # Folder to save alert images
os.makedirs(alerts_path, exist_ok=True)

# Parameters
FACE_MATCH_THRESHOLD = 0.5  # Lower values = more strict matching
FRAME_SKIP = 2  # Process 1 frame, then skip this many frames (0 = process all frames)

# Numba-accelerated face distance
@njit
def compute_face_distances(known_encodings, face_encoding):
    distances = np.empty(len(known_encodings))
    for i in range(len(known_encodings)):
        diff = known_encodings[i] - face_encoding
        distances[i] = np.sqrt(np.dot(diff, diff))
    return distances

# Load known faces
def load_known_faces(directory):
    known_encodings = []
    known_names = []

    for person in os.listdir(directory):
        person_dir = os.path.join(directory, person)
        if os.path.isdir(person_dir):
            for img_file in os.listdir(person_dir):
                if img_file.lower().endswith(('jpg', 'jpeg', 'png')):
                    path = os.path.join(person_dir, img_file)
                    image = face_recognition.load_image_file(path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(person)
    return known_encodings, known_names

def main():
    print("Loading known faces...")
    known_encodings, known_names = load_known_faces(known_faces_path)
    print(f"Loaded {len(known_encodings)} known faces.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    # Get video properties for the output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    output_path = 'output_video.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    skip_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Frame skipping logic
        if skip_counter > 0:
            # Skip processing but still write frame to output
            out.write(frame)
            skip_counter -= 1
            frame_idx += 1
            continue
        else:
            # Reset skip counter
            skip_counter = FRAME_SKIP

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process this frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for idx, ((top, right, bottom, left), face_encoding) in enumerate(zip(face_locations, face_encodings)):
            distances = compute_face_distances(np.array(known_encodings), face_encoding)
            best_match_index = np.argmin(distances)
            name = "Unknown"
            if distances[best_match_index] < FACE_MATCH_THRESHOLD:
                name = known_names[best_match_index]

            if name == "Unknown":
                continue

            # Draw rectangle and name
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Add background rectangle for text
            text = name
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Create background rectangle
            text_x = left
            text_y = top - 10
            cv2.rectangle(frame,
                        (text_x, text_y - text_height - 5),
                        (text_x + text_width + 10, text_y + 5),
                        (0, 0, 0),
                        -1)  # -1 means filled rectangle

            # Draw text on top of background
            cv2.putText(frame, name, (text_x + 5, text_y), font, font_scale, (255, 255, 255), thickness)

            # --- Alert logic ---
            # Crop the face from the frame (with boundary checks)
            cropped_face = frame[max(0, top):min(bottom, height), max(0, left):min(right, width)]

            # Create filename for alert image
            alert_img_filename = os.path.join(alerts_path, f"{name}_frame{frame_idx}_face{idx}.jpg")

            # Save cropped face image
            cv2.imwrite(alert_img_filename, cropped_face)

            # Print alert message
            print(f"Alert: Wanted person '{name}' found! Saved alert image: {alert_img_filename}")

        # Write the frame to the output video
        out.write(frame)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    # Release everything
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Output video saved to {output_path}")

if __name__ == "__main__":
    main()