import streamlit as st
import cv2
import requests
import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from PIL import Image
import time

# Load Azure credentials
with open("config.json") as f:
    config = json.load(f)

subscription_key = config["subscription_key"]
endpoint = config["endpoint"]

detect_url = endpoint + "/face/v1.0/detect"
verify_url = endpoint + "/face/v1.0/verify"

headers = {
    'Ocp-Apim-Subscription-Key': subscription_key,
    'Content-Type': 'application/octet-stream'
}
params = {
    'returnFaceId': 'true'
}

# -------------------- Upload known face --------------------
st.title("🔐 Azure Face Attendance System")

with st.expander("📤 Upload Known Face"):
    uploaded_file = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])
    name = st.text_input("Name for the uploaded face")
    if uploaded_file and name:
        if not os.path.exists("known_faces"):
            os.makedirs("known_faces")
        path = os.path.join("known_faces", f"{name}.jpg")
        with open(path, "wb") as f:
            f.write(uploaded_file.read())
        st.success(f"✅ Saved as {name}.jpg in known_faces")

# -------------------- Clear Attendance --------------------
if st.button("🗑 Clear Attendance Log"):
    if os.path.exists("attendance.csv"):
        os.remove("attendance.csv")
    st.success("Attendance log cleared.")

# -------------------- Display Attendance --------------------
st.subheader("📋 Attendance Log")
if os.path.exists("attendance.csv"):
    df = pd.read_csv("attendance.csv")
    st.dataframe(df)
else:
    st.info("No attendance recorded yet.")

# -------------------- Start Attendance --------------------
if st.button("🎬 Start Attendance"):
    cam = cv2.VideoCapture(0)
    FRAME_WINDOW = st.image([])

    ret, frame = cam.read()
    cam.release()

    if not ret:
        st.error("❌ Could not access webcam.")
    else:
        # Save captured image
        cv2.imwrite("live.jpg", frame)
        with open("live.jpg", "rb") as f:
            live_img = f.read()

        # Detect face in live image
        response = requests.post(detect_url, headers=headers, params=params, data=live_img)
        detected = response.json()

        if not detected or "faceId" not in detected[0]:
            st.error("❌ No face detected in live image.")
        else:
            live_face_id = detected[0]['faceId']
            matched_name = None

            # Loop through known faces
            for filename in os.listdir("known_faces"):
                with open(os.path.join("known_faces", filename), "rb") as f:
                    known_img = f.read()

                resp = requests.post(detect_url, headers=headers, params=params, data=known_img)
                known = resp.json()

                if known and "faceId" in known[0]:
                    known_face_id = known[0]['faceId']
                    verify_data = {
                        "faceId1": live_face_id,
                        "faceId2": known_face_id
                    }
                    res = requests.post(verify_url, headers=headers, json=verify_data).json()

                    if res.get("isIdentical") and res.get("confidence", 0) > 0.6:
                        matched_name = os.path.splitext(filename)[0]
                        break

            # Mark attendance
            if matched_name:
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open("attendance.csv", "a") as f:
                    f.write(f"{matched_name},{now}\n")
                st.success(f"✅ {matched_name} marked present at {now}")
            else:
                st.warning("❌ Face not recognized.")

