# VoicedTrainer
A training system with voice

## Installation

### Prerequisites
- Python 3.6+
- Git

### Setup
1. Clone the repository
   ```
   git clone https://github.com/stho32/VoicedTrainer.git
   cd VoicedTrainer
   ```

2. Set up the virtual environment
   ```
   python -m venv venv
   ```

3. Activate the virtual environment
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On Unix or MacOS:
     ```
     source venv/bin/activate
     ```

4. Install the package in development mode
   ```
   pip install -e .
   ```

## Usage
After installation, you can run the application using:
```
voiced-trainer
```

Or directly with Python:
```
python -m voiced_trainer.main
```

## Development

### Project Structure
- `voiced_trainer/` - Main package directory
- `tests/` - Test files
- `setup.py` - Package setup file
- `requirements.txt` - Project dependencies
