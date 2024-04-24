# SAE Circuits

TO-DO



## Setup and Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management. Follow these instructions to set up your environment.

### Prerequisites

- Python 3.12
- Poetry

### Installing Poetry

If you don't already have Poetry installed, you can install it by running:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

This command will install Poetry and add it to your PATH. For more detailed instructions, visit the [Poetry installation guide](https://python-poetry.org/docs/#installation).

### Setting Up the Project

Once Poetry is installed, clone the repository and install the dependencies:

```bash
git clone https://github.com/charlieoneill11/sae-circuits.git
cd sae-circuits
poetry install
```

This will install all necessary dependencies as specified in the `pyproject.toml` file.

### Running the Project

To run the main script, use Poetry to handle the environment:

```bash
poetry run python main.py --config config.yaml
```