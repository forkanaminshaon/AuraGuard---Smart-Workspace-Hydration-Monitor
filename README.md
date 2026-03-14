# AuraGuard - Smart Workspace & Hydration Monitor

AuraGuard is a **real-time computer vision application** designed to reduce digital distractions and promote healthy work habits. Using a webcam and a lightweight YOLO object detection model, AuraGuard monitors desk activity and provides instant visual feedback when the user becomes distracted by their phone or forgets to stay hydrated.

The system runs **entirely locally**, ensuring privacy while delivering real-time insights through a visual **Heads-Up Display (HUD)**.

---

# Features

✅ **Real-Time Object Detection**  
Detects important objects in the workspace using **YOLOv8**.

📱 **Phone Distraction Detection**  
Tracks how long a phone is visible and alerts the user if distraction continues.

💧 **Hydration Monitoring**  
Reminds users to drink water if a bottle or cup has not been detected for a specified period.

🖥 **Live Heads-Up Display (HUD)**  
Displays system status, timers, and alerts directly on the webcam feed.

🔒 **Privacy First**  
Runs completely **offline** — no cloud processing.

⚡ **Lightweight & Real-Time**  
Optimized using **YOLOv8 Nano** for fast inference on standard laptops.

---

# 🧠 System Architecture

```
Webcam Input
      │
      ▼
OpenCV Video Capture
      │
      ▼
YOLO Object Detection (Ultralytics)
      │
      ▼
Object Filtering Layer
(person, phone, bottle, cup)
      │
      ▼
Temporal Logic Engine
(phone timer & hydration timer)
      │
      ▼
HUD Rendering (OpenCV Overlay)
      │
      ▼
Real-Time Display
```

---

# 🛠 Tech Stack

| Technology | Purpose |
|------------|--------|
| Python | Core programming language |
| OpenCV | Video capture and visualization |
| YOLOv8 (Ultralytics) | Real-time object detection |
| NumPy | Frame processing |
| Google Colab / Local Python | Execution environment |

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/forkanaminshaon/AuraGuard---Smart-Workspace-Hydration-Monitor.git
cd AuraGuard---Smart-Workspace-Hydration-Monitor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶ Running the Project

Run the main script:

```bash
python auraguard.py
```

Press **Q** to exit the application.

---

# Configuration

You can modify the thresholds inside the script:

```python
PHONE_DISTRACTION_THRESHOLD = 5
HYDRATION_REMINDER_THRESHOLD = 30
```

These values control how long the system waits before showing alerts.

---

# 🎥 Demo

The system detects:

- 📱 Phone usage  
- 💧 Hydration activity  
- 👤 User presence  

Example alerts:

```
⚠ PUT DOWN YOUR PHONE!
💧 TIME TO DRINK WATER!
```

Demo video available in the repository:

```
auraguard_demo.mp4
```

---

# 📂 Project Structure

```
AuraGuard---Smart-Workspace-Hydration-Monitor/
│
├── auraguard.py
├── requirements.txt
├── architecture.txt
├── auraguard_demo.mp4
├── report.pdf
└── README.md
```

---

# 📊 Future Improvements

Possible enhancements:

- Posture detection using **MediaPipe**
- Eye strain detection
- Productivity analytics dashboard
- Session statistics logging
- Weekly behavior analysis using **Data Science**

---

# 📚 Learning Objectives

This project demonstrates:

- Deploying **pre-trained YOLO models**
- Filtering object detection results
- Implementing **temporal behavior tracking**
- Designing real-time **computer vision HUD interfaces**

---

# Author

**Forkan Amin Shaon**

🎓 B.Sc. in Applied Mathematics 
📊 M.Sc. in Applied Statistics & Data Science 

Interested in **Artificial Intelligence, Machine Learning, Data Science, and Computer Vision.**

---

# ⭐ Support

If you found this project interesting, consider **starring ⭐ the repository**.
