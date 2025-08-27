# 📊 Video Analytics for Customer Time Spent at Billing Counters

This project is a solution for the internship task from **NeubAltics Tech Pvt. Ltd.**  
It analyzes video footage of a retail store to measure **customer wait times** and **service times** at billing counters, providing valuable insights for **queue management** and **staff optimization**.

---

## ✨ Features

- **🎯 Object Detection & Tracking**  
  Uses the **YOLOv8** model to detect and track individual customers in real time.

- **🖼 ROI-Based Monitoring**  
  Defines specific zones for both **queues** and **counters** to track customer movement.

- **📈 Behavioral Analysis**  
  Calculates key metrics for each customer:  
  - **Wait Time** → Time spent waiting in the queue before reaching the counter  
  - **Service Time** → Time spent being served at the counter  

- **📂 Data Export**  
  Generates a `customer_journey_report.csv` with detailed analytics for each customer.

- **📷 Visual Snapshots**  
  Saves a snapshot of the counter area the moment a customer enters, providing a visual log.

- **🌐 Streamlit Web App**  
  Interactive web-based interface (`index.py`) for uploading and analyzing videos.

---

## ⚙️ Setup and Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Prakash2503/video-analytics
   cd video-analytics
