# File: preprocess_align_face.py
# --- USES MEDIAPIPE AND DYNAMICALLY DETERMINES SIZE WITH TIMM ---

import cv2
import os
import argparse
from tqdm import tqdm
import mediapipe as mp
import numpy as np
import timm
import torch
model_names = "mnasnet_small.lamb_in1k"


def get_model_input_size(model_name: str) -> int:
    try:
        model = timm.create_model(model_name, pretrained=False)
        data_cfg = timm.data.resolve_model_data_config(model)
        input_size = data_cfg['input_size']
        image_size = input_size[-1]
        print(f"✅ Determined input size for '{model_name}': {image_size}x{image_size}")
        del model
        return image_size
    except Exception as e:
        print(f"❌ Error: Could not determine input size for model '{model_name}'.")
        print(f"Please check if the model name is correct. Timm error: {e}")
        exit()


def align_and_crop_face(frame, face_mesh, desired_face_size=224):
    """
    Detects faces and landmarks using MediaPipe, aligns the face, 
    and returns the cropped, aligned face.
    """
    try:
        frame.flags.writeable = False
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame.flags.writeable = True

        if not results.multi_face_landmarks:
            return None, "no_face"

        face_landmarks = results.multi_face_landmarks[0]
        img_h, img_w, _ = frame.shape
        landmarks = np.array([(lm.x * img_w, lm.y * img_h) for lm in face_landmarks.landmark])

        # Calculate alignment angle based on eye landmarks
        left_eye_center = landmarks[133]
        right_eye_center = landmarks[362]

        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Align the face
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)
        M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
        aligned_frame = cv2.warpAffine(frame, M, (img_w, img_h), flags=cv2.INTER_CUBIC)
        # Crop the face from the aligned image
        ones = np.ones(shape=(len(landmarks), 1))
        landmarks_ones = np.hstack([landmarks, ones])
        transformed_landmarks = M.dot(landmarks_ones.T).T
        x_min, y_min = np.min(transformed_landmarks, axis=0)
        x_max, y_max = np.max(transformed_landmarks, axis=0)
        pad = 20
        x_min = max(0, int(x_min - pad))
        y_min = max(0, int(y_min - pad))
        x_max = min(img_w, int(x_max + pad))
        y_max = min(img_h, int(y_max + pad))

        face_roi = aligned_frame[y_min:y_max, x_min:x_max]

        if face_roi.size == 0:
            return None, "crop_failed"

        # Resize to the final desired size
        resized_face = cv2.resize(face_roi, (desired_face_size, desired_face_size), interpolation=cv2.INTER_AREA)

        return resized_face, "success"

    except Exception:
        return None, "processing_error"


def run_preprocessing(source_dir, output_dir, face_size):

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    print("✅ MediaPipe Face Mesh model loaded successfully.")

    processed_count, skipped_no_face, skipped_fail = 0, 0, 0
    class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for class_name in tqdm(class_names, desc="Processing Classes"):
        source_class_path = os.path.join(source_dir, class_name)
        output_class_path = os.path.join(output_dir, class_name)
        os.makedirs(output_class_path, exist_ok=True)

        for img_name in tqdm(os.listdir(source_class_path), desc=f"  Processing {class_name}", leave=False):
            source_img_path = os.path.join(source_class_path, img_name)
            output_img_path = os.path.join(output_class_path, img_name)

            try:
                frame = cv2.imread(source_img_path)
                if frame is None:
                    skipped_fail += 1
                    continue

                aligned_face, status = align_and_crop_face(
                    frame, face_mesh, desired_face_size=face_size)

                if status == "success":
                    cv2.imwrite(output_img_path, aligned_face)
                    processed_count += 1
                elif status == "no_face":
                    skipped_no_face += 1
                else:
                    skipped_fail += 1
            except Exception as e:
                print(f"Critical error on {img_name}: {e}")
                skipped_fail += 1

    face_mesh.close()

    print("\n" + "="*40)
    print("Preprocessing Complete!")
    print(f"✅ Images Processed (Aligned & Cropped): {processed_count}")
    print(f"⚠️ Skipped (No face detected): {skipped_no_face}")
    print(f"⚠️ Skipped (Processing/Crop failed): {skipped_fail}")
    print("="*40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocessing: Detect, Align, and Crop Faces. Automatically determines size from model name.")
    parser.add_argument("--input", type=str, required=True, help="Path to the source data directory.")
    parser.add_argument("--output", type=str, required=True, help="Path to the directory to save results.")

    args = parser.parse_args()

    # --- Dynamically get the input size ---
    image_size = get_model_input_size(model_names)

    source_folder_name = os.path.basename(os.path.normpath(args.input))
    output_destination = os.path.join(args.output, f"{source_folder_name}_face_aligned_{image_size}px")

    run_preprocessing(args.input, output_destination, image_size)
