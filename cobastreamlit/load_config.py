from typing import Any, Dict, Tuple

import io
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import toml
from PIL import Image


def get_project_root() -> str:

	return str(Path(__file__).parent)

# Load TOML config file

@st.cache(allow_output_mutation=True, ttl=300)
def load_config(
		config_readme_filename: str
) -> Dict[Any, Any]:

	config_readme = toml.load(Path(get_project_root()) / f"config/{config_readme_filename}")
	return dict(config_readme)

readme = load_config("config_readme.toml")		