# Face Recognition using OpenCV and JSON

This project implements a simple **face recognition system** in Python using OpenCV. The system loads student images from a folder, detects faces, stores them in a JSON-based database, and performs real-time recognition using a webcam.

---

## Features

- Load student images and metadata from a JSON file.
- Detect faces using Haar Cascade Classifier.
- Store face images in memory for comparison.
- Real-time face recognition via webcam.
- Display recognized names and matching scores.
- Handles unknown faces gracefully.

---

## Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- NumPy

Install dependencies:

```bash
pip install opencv-python numpy
```
---
##Project Structure
<img width="600" height="107" alt="image" src="https://github.com/user-attachments/assets/caee4acb-0a61-47e5-b147-6cd70da6e527" />

## Example students.json format:
<img width="734" height="295" alt="image" src="https://github.com/user-attachments/assets/9ca71494-b79a-4af5-bf93-a08dbfcb594d" />

---
## How to Run

1. Make sure you have the student images in the `photo/` folder.
2. Update `students.json` with student IDs, names, and filenames.
3. Run the script:

```bash
python main.py
```
4. A window will open showing the webcam feed.
5. Detected faces will be recognized and labeled.
6. Press `q` to quit.
