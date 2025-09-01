# 🧬 PocketHunter Suite - Streamlit Edition

A modern, web-based interface for PocketHunter - an advanced molecular dynamics pocket detection and analysis tool. This Streamlit application provides a beautiful, user-friendly interface for running pocket detection pipelines with real-time monitoring and visualization capabilities.

## ✨ Features

### 🎨 Modern Design
- **Beautiful UI**: Gradient backgrounds, modern cards, and smooth animations
- **Responsive Layout**: Works perfectly on desktop and mobile devices
- **Real-time Updates**: Live progress tracking and status monitoring
- **Interactive Visualizations**: Plotly charts and 3D molecular visualizations

### 🔬 Scientific Capabilities
- **Modular Pipeline**: Execute individual pipeline steps with reproducible workflows
- **Job ID Tracking**: Each step generates a unique job ID for tracking and reproducibility
- **Advanced Parameters**: Fine-tune detection, clustering, and docking parameters
- **Multiple Input Methods**: Upload files or use previous step results
- **Molecular Docking**: Dock ligands to identified pockets using SMINA

### 📊 Monitoring & Analysis
- **Task Monitor**: Real-time monitoring of all running tasks
- **Progress Tracking**: Live progress bars and status updates
- **Results Visualization**: Interactive charts and data tables
- **Export Capabilities**: Download results in various formats

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Redis server (for Celery task queue)
- PocketHunter CLI tools
- SMINA docking software (for molecular docking)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd streamlit_pockethunter_app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install docking dependencies (optional)**
   ```bash
   ./install_docking_deps.sh
   ```

4. **Start Redis server**
   ```bash
   redis-server
   ```

5. **Start Celery worker**
   ```bash
   celery -A celery_app worker --loglevel=info
   ```

6. **Run the Streamlit app**
   ```bash
   streamlit run main.py
   ```

## 📋 Usage Guide

### Modular Workflow
1. **Step 1: Extract Frames**: Convert trajectory to PDB files
   - Upload trajectory (.xtc) and topology (.pdb/.gro) files
   - Adjust extraction parameters
   - Run extraction and get a unique Job ID
   
2. **Step 2: Detect Pockets**: Identify potential binding sites
   - Use Job ID from Step 1 or upload PDB files directly
   - Configure detection parameters
   - Run detection and get a unique Job ID
   
3. **Step 3: Cluster Pockets**: Group similar pockets and find representatives
   - Use Job ID from Step 2 or upload pockets.csv directly
   - Configure clustering parameters
   - Run clustering and get final results
   
4. **Step 4: Molecular Docking**: Dock ligands to representative pockets
   - Use cluster representatives from Step 3
   - Upload ligand library in PDBQT format
   - Configure docking parameters (poses, exhaustiveness, pH)
   - Run docking and analyze binding affinities
   
5. **Task Monitor**: Track progress across all steps

### Task Monitor
- View all running and completed tasks
- Filter by status, type, or task state
- Download results and manage job data
- Real-time auto-refresh capabilities

## 🎯 Key Features

### File Upload & Job ID Management
- **Drag & Drop**: Easy file upload interface
- **Multiple Formats**: Support for .xtc, .pdb, .gro, .zip files
- **Validation**: Automatic file format checking
- **Progress Tracking**: Upload progress indicators
- **Job ID Caching**: Automatic job ID transfer between steps
- **Reproducibility**: Each step generates unique job IDs for tracking

### Parameter Tuning
- **Frame Extraction**: Adjust stride and thread count
- **Pocket Detection**: Configure detection sensitivity
- **Clustering**: Choose between DBSCAN and hierarchical methods
- **Docking**: Configure poses, exhaustiveness, and pH parameters
- **Advanced Options**: Fine-tune clustering and docking parameters

### Results Analysis
- **Interactive Tables**: Sortable and filterable data tables
- **Visualizations**: Probability distributions, scatter plots
- **Statistics**: Summary metrics and cluster analysis
- **Export Options**: Download CSV, ZIP, or individual files

## 🏗️ Architecture

### Frontend
- **Streamlit**: Modern web framework for data apps
- **Custom CSS**: Beautiful gradients and animations
- **Plotly**: Interactive charts and visualizations
- **Responsive Design**: Mobile-friendly interface

### Backend
- **Celery**: Asynchronous task processing
- **Redis**: Message broker and result backend
- **PocketHunter**: Core scientific algorithms
- **File Management**: Organized upload and result storage

### Data Flow
1. **Upload**: Files stored in organized directory structure
2. **Processing**: Celery tasks handle heavy computations
3. **Monitoring**: Real-time status updates via Redis
4. **Results**: Organized output with download capabilities

## 🔬 Molecular Docking

The PocketHunter Suite now includes molecular docking capabilities as Step 4 of the pipeline. This feature allows you to dock ligands to the representative pockets identified in the clustering step.

### Docking Features
- **SMINA Integration**: Uses SMINA (modified AutoDock Vina) for docking
- **Flexible Input**: Support for PDBQT files and ZIP archives
- **Configurable Parameters**: Adjustable poses, exhaustiveness, and pH
- **Results Analysis**: Interactive visualization of docking scores and poses
- **Export Options**: Download results in various formats

### Docking Workflow
1. **Prepare Ligands**: Convert your ligands to PDBQT format
2. **Select Representatives**: Choose cluster representatives from Step 3
3. **Configure Parameters**: Set docking parameters in the sidebar
4. **Run Docking**: Execute docking calculations
5. **Analyze Results**: View scores, poses, and download results

### Docking Parameters
- **Number of Poses**: Controls conformational sampling (1-50)
- **Exhaustiveness**: Docking accuracy (1-20, higher = more accurate)
- **pH**: Protonation state (4.0-10.0, default 7.4)

For detailed docking documentation, see [DOCKING_INTEGRATION.md](DOCKING_INTEGRATION.md).

## 🔧 Configuration
```bash
# Redis configuration
REDIS_URL=redis://localhost:6379/0

# Celery configuration
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0

# File paths
UPLOAD_DIR=./uploads
RESULTS_DIR=./results
```

### Customization
- **Themes**: Modify CSS in main.py for custom styling
- **Parameters**: Adjust default values in individual app files
- **Visualizations**: Customize Plotly charts and layouts
- **File Formats**: Add support for additional file types

## 📊 Performance

### Optimization Features
- **Asynchronous Processing**: Non-blocking task execution
- **Progress Tracking**: Real-time updates without page refresh
- **Memory Management**: Efficient file handling and cleanup
- **Caching**: Redis-based result caching

### Scalability
- **Multiple Workers**: Scale Celery workers for high throughput
- **Queue Management**: Prioritize and manage task queues
- **Resource Monitoring**: Track CPU and memory usage
- **Error Handling**: Robust error recovery and reporting

## 🐛 Troubleshooting

### Common Issues
1. **Redis Connection**: Ensure Redis server is running
2. **Celery Workers**: Check worker status and logs
3. **File Permissions**: Verify upload and results directory access
4. **Memory Issues**: Monitor system resources during large jobs

### Debug Mode
```bash
# Enable debug logging
export STREAMLIT_LOG_LEVEL=debug
streamlit run main.py --logger.level=debug
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests and documentation
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions
- Include error handling

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PocketHunter Team**: Core scientific algorithms
- **Streamlit Community**: Web framework and components
- **Plotly**: Interactive visualizations
- **Celery**: Asynchronous task processing

## 📞 Support

For questions, issues, or feature requests:
- Create an issue on GitHub
- Check the documentation
- Review troubleshooting guide
- Contact the development team

---

**Made with ❤️ for the scientific community** 