# Quantum Tunneling Simulation with QuTiP and Manim

A quantum mechanics simulation demonstrating wave packet tunneling through a potential barrier using QuTiP for physics calculations and Manim for visualization.

## Features

- **Quantum Wave Packet Evolution**: Accurate time evolution using the Crank-Nicolson method
- **Absorbing Boundary Conditions**: Complex potential absorbers prevent wave reflections
- **Real-time Physics Tracking**: Monitor transmission, reflection, and barrier probabilities
- **Professional Animation**: High-quality visualization with Manim
- **Uncertainty Principle Compliance**: Proper quantum mechanical normalization

## Physics Implementation

- **Framework**: QuTiP (Quantum Toolbox in Python)
- **Time Evolution**: Unitary evolution with Hamiltonian operator
- **Boundary Conditions**: Exponential absorbing potentials at domain edges
- **Wave Packet**: Gaussian initial state with configurable momentum and position
- **Barrier**: Adjustable height and width potential barrier

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- qutip
- numpy
- scipy
- matplotlib
- manim
- rich

## Usage

### Run Simulation
```bash
python qutip_wave_simulation.py
```

### Generate Animation
```bash
manim quantum_animation.py QuantumTunneling
```

### Auto-commit Changes
```bash
node auto-commit.js
```

## File Structure

- `qutip_wave_simulation.py` - Main quantum simulation engine
- `quantum_animation.py` - Manim animation renderer
- `auto-commit.js` - Automated git commit script
- `Data/` - Simulation and animation output data
- `requirements.txt` - Python dependencies

## Video Demo

[View the quantum tunneling animation here](media/videos/quantum_animation/1080p60/QuantumTunneling.mp4)

