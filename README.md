# OpenWebEllipsometer

Streamlit-based application for fitting thin-film optical parameters from analyzer intensity measurements.

## Installation

Clone this repository and install the required Python packages. Below are two example setups using Conda or plain pip.

### Using Conda

```bash
conda create -n openwebellipsometer python=3.10
conda activate openwebellipsometer
conda install -c conda-forge numpy pandas scipy streamlit
```

### Using Pip

```bash
python3 -m venv venv
source venv/bin/activate  # on Windows use "venv\\Scripts\\activate"
pip install numpy pandas scipy streamlit
```

Once the dependencies are installed, run the application from the repository root with:

```bash
streamlit run app.py
```

## Data format

The measurement table must contain the following columns:

- `wavelength_nm` – wavelength in nanometers
- `incidence_deg` – angle of incidence relative to the normal
- `analyzer_deg` – analyzer rotation angle (0 means polarization orthogonal to the incoming one)
- `intensity` – measured signal level

Rows can be added manually via the interface or you can upload a CSV file with these columns.

## Usage

1. Run the application with `streamlit run app.py`.
2. Enter or upload the measurement table.
3. Specify the initial optical parameters and mark which ones should be optimised.
4. Press "Start optimisation". The fitted parameters and the model's root mean squared deviation will be displayed.
