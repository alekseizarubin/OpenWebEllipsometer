# OpenWebEllipsometer

Streamlit-based application for fitting thin-film optical parameters from analyzer intensity measurements.
It now supports setting the polariser angle, intensity offsets and calibration of systematic angle errors.

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

The application also allows setting the angle between the incidence plane and
the polarisation direction of the incoming light. This angle can be optimised
along with a constant offset for the analyzer and incidence angles during
calibration.

Rows can be added manually via the interface or you can upload a TSV file with these columns. The file should use a dot (`.`) as the decimal separator. The application assumes intensities are normalised, but you may also fit a scaling factor when providing raw detector counts.
If needed the helper ``normalise_measurements`` can rescale uploaded data to the 0–1 range.

## Usage

1. Run the application with `streamlit run app.py`.
2. Enter or upload the measurement table.
3. Specify the initial optical parameters and mark which ones should be optimised.
4. Press "Start optimisation". The fitted parameters and the model's root mean squared deviation will be displayed.

### Parameter bounds and intensity scale

Each optimised value can have custom lower and upper bounds. These bounds are
configured in the sidebar next to the corresponding optimisation checkbox. The
application also provides an **intensity scale** parameter. When fitting raw
detector counts that are not normalised to the incident power, enable optimisation
of this scale factor so the model can match the measurement units. An additional
offset parameter accounts for detector background.

### Calibration

Use ``calibrate_system`` with a sample of known optical properties to determine
systematic offsets for the polariser, incidence and analyzer angles. The
resulting parameters can then be used for subsequent measurements.
