# NextSight

NextSight is a smart vision system for real-time hand tracking and object detection, designed for interactive demonstrations and quality control applications.

## Features

- **Hand Tracking**: Real-time hand detection and finger tracking using MediaPipe
- **Object Detection**: Intelligent detection and classification of objects (jars with/without lids)
- **Real-time Performance**: Optimized for Acer Nitro V with RTX 4050 GPU
- **Interactive Demo**: Perfect for exhibitions and demonstrations

## Hardware Requirements

- **Primary**: Acer Nitro V with NVIDIA RTX 4050
- **Camera**: Any USB camera or built-in webcam
- **RAM**: 8GB+ recommended
- **Storage**: 2GB+ free space

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/stackTD/NextSight.git
cd NextSight
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install the package:
```bash
pip install -e .
```

### Running the Application

```bash
python src/main.py
```

## Project Structure

```
NextSight/
├── config/          # Configuration files
├── src/             # Source code
│   ├── core/        # Core computer vision modules
│   ├── models/      # ML models
│   ├── utils/       # Utility functions
│   └── ui/          # User interface
├── data/            # Data storage
├── scripts/         # Training and utility scripts
├── tests/           # Test suite
├── docs/            # Documentation
└── assets/          # Media assets
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
black src/ tests/
flake8 src/ tests/
```

## Author

Created by **stackTD** for smart exhibition demonstrations.

## License

This project is licensed under the MIT License.