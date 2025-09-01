# 🧬 PocketHunter Suite - Streamlit Edition

A modern, web-based interface for **PocketHunter**, an advanced molecular dynamics pocket detection and analysis tool. This Streamlit application provides a beautiful, user-friendly interface for running molecular simulations and analysis with real-time monitoring and interactive visualizations.

## ✨ Key Features

### 🎨 User-Centric Design

  - **Intuitive UI:** Enjoy a clean, modern interface with gradient backgrounds, responsive layouts, and smooth animations.
  - **Real-time Monitoring:** Track the progress of your tasks with live updates and progress bars.
  - **Interactive Visualizations:** Analyze results with dynamic charts from **Plotly** and 3D molecular visualizations.

### 🔬 Scientific Workflow

  - **Modular Pipeline:** Run complex scientific workflows in a step-by-step, reproducible manner.
  - **Unique Job IDs:** Every step generates a unique ID, ensuring full reproducibility and easy tracking.
  - **Advanced Parameter Control:** Fine-tune scientific parameters for pocket detection, clustering, and docking to suit your specific research needs.
  - **Integrated Docking:** Seamlessly perform molecular docking with **SMINA** on identified pockets.

-----

## 🚀 Quick Start

### Prerequisites

  - Python 3.8+
  - A running Redis server
  - PocketHunter CLI tools
  - SMINA docking software (for molecular docking)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:bogrum/PocketHunter-Suite.git
    cd streamlit_pockethunter_app
    ```
2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Install docking dependencies (optional):**
    ```bash
    ./install_docking_deps.sh
    ```
4.  **Start your Redis server:**
    ```bash
    redis-server
    ```
5.  **Start the Celery worker:**
    ```bash
    celery -A celery_app worker --loglevel=info
    ```
6.  **Run the Streamlit application:**
    ```bash
    streamlit run main.py
    ```

-----

## 📋 Modular Workflow Guide

The application guides you through a four-step scientific pipeline. Each step can be executed independently using a previous step's Job ID or by uploading new files.

1.  **Extract Frames:** Convert molecular dynamics trajectories (`.xtc`) to individual PDB files.
2.  **Detect Pockets:** Identify potential binding sites using the PDB files.
3.  **Cluster Pockets:** Group similar pockets to find representative binding sites.
4.  **Molecular Docking:** Dock ligands to the cluster representatives to analyze binding affinities.

A dedicated **Task Monitor** allows you to view the status, filter, and manage all of your jobs in one place.

-----

## 🏗️ Architecture

The application is built on a scalable and robust architecture.

  * **Frontend:** The user interface is built with **Streamlit**, enhanced with custom CSS for a modern look. It uses **Plotly** for rich, interactive data visualizations.
  * **Backend:** Heavy computational tasks are handled asynchronously by **Celery**. **Redis** serves as the message broker, facilitating communication and real-time updates between the frontend and the backend.
  * **Data Flow:** Files are uploaded to an organized directory structure. Celery workers process these files, and results are stored in a dedicated `results/` directory, ready for download.

-----

## 🎯 Molecular Docking

The molecular docking feature is a core component of the pipeline. It is integrated using **SMINA**, a fast and accurate docking tool.

### Docking Parameters

You can configure key docking parameters directly from the sidebar:

  - **Number of Poses:** Controls the number of possible binding poses (1-50).
  - **Exhaustiveness:** Determines the accuracy and computational cost of the docking run (1-20, higher is more accurate).
  - **pH:** Specifies the protonation state of the molecules (4.0-10.0, default 7.4).


-----

## 🐛 Troubleshooting

  * **Redis Connection:** Ensure your Redis server is actively running before starting the Celery worker and the Streamlit app.
  * **Celery Workers:** Check the worker's terminal logs for any errors if a task is not running correctly.
  * **File Permissions:** Verify that the `uploads` and `results` directories have the correct read/write permissions.

-----

## 🤝 Contributing

We welcome contributions! Please open an issue or submit a pull request on GitHub.

-----

Made with ❤️ for the scientific community.