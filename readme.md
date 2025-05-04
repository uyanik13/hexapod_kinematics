# Hexapod Robot Leg Kinematics

This is a Python implementation of the kinematics for a hexapod robot.

## Installation & Setup

1. Create a virtual environment:

Windows:
```bash
python -m venv .venv
```

2. Activate the virtual environment:

Windows (PowerShell):
```bash
.\.venv\Scripts\activate
```

Windows (Command Prompt):
```bash
.venv\Scripts\activate.bat
```

Linux/Mac:
```bash
source .venv/bin/activate
```

3. Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the script:

```bash
python test_kinematic.py
```

## Troubleshooting

If you encounter installation issues:

1. Make sure you're in the correct directory:
```bash
cd path/to/hexapod-kinematics
```

2. Try removing and recreating the virtual environment:
```bash
python -m venv venv
```

3. If pip install fails, try:
```bash
python -m pip install -r requirements.txt
```

## Contributing

Feel free to open issues or submit pull requests for improvements.

## License

MIT License
