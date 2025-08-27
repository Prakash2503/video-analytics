
 Video Analytics for Customer Time Spent at Billing
 Counters
 
 This project is a solution for the internship task from NeubAltics Tech Pvt. Ltd. It analyzes video
 footage of a retail store to measure customer wait times and service times at billing counters,
 providing valuable insights for queue management and staff optimization.
 Features: Object Detection & Tracking: Uses the YOLOv8 model to detect and track individual
 customers in real-time. ROI-Based Monitoring: Defines specific zones for both queues and
 counters to track customer movement. Behavioral Analysis: Calculates key metrics for each
 customer: Wait Time: The time spent waiting in a queue before reaching the counter. Service
 Time: The time spent being served at the counter. Data Export: Generates a
 customer_journey_report.csv with detailed analytics for each customer. Visual Snapshots: Saves
 a snapshot of the counter area the moment a customer enters, providing a visual log. Streamlit
 Web App: Interactive web-based interface for running inference using index.py.
 Setup and Installation: Clone the repository:
 git clone https://github.com/your-username/your-repository-name.git
 cd your-repository-name
 Create and activate a virtual environment (recommended):
 python -m venv venv
 source venv/bin/activate # On Windows, use venv\Scripts\activate
 Install the required libraries:
 pip install -r requirements.txt
 How to Run: Place the input video file (e.g., input.mp4) in the main project directory. Run the main
 script from your terminal:
 python main.py The script will process the video, display the live analysis with bounding boxes, and
 print a final report in the console. A customer_journey_report.csv file and a counter_snapshots
 folder will be created in the project directory.
 Alternatively, run the Streamlit web application:
 streamlit run index.py Upload your video through the Streamlit interface to analyze it interactively.
 This project was created as part of the internship task at NeubAItics Tech Pvt. Ltd
