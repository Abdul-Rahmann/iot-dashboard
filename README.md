iot_dashboard/
├── app.py                # Main entry point for the Dash dashboard
├── assets/               # For static assets like CSS/JS (used in Dash layouts)
├── components/           # Modular Dash components for better structure
├── data/                 # For raw and processed IoT datasets
│   ├── models/           # Trained ML/NN models go here
├── ml/                   # Machine/Deep Learning models and logic
│   ├── anomaly_detection_autoencoder.py
│   ├── predictive_maintenance_rnn.py (future step)
├── simulation/           # IoT data simulation or mocking scripts
├── utils/                # Helper/utility functions shared across the project
├── environment.yml       # Conda environment with pinned dependencies
└── README.md             # Project description and setup guide