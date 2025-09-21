# app.py -- Streamlit UI for capsa_service_network.py with permanent background logo
import os
import io
import sys
import time
import traceback
import importlib
import importlib.util
import base64
from typing import Optional

import streamlit as st
import pandas as pd

# === CONFIG: set desired background style ===
# Options: "watermark" (faint centered image) or "cover" (full background)
BACKGROUND_STYLE = "watermark"

st.set_page_config(page_title="CapSA Runner", layout="wide")

MODULE_NAME = "capsa_service_network"
MODULE_FILE = MODULE_NAME + ".py"

# ---------- Permanent background logo helper ----------
def _file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def set_permanent_background(style: str = "watermark", filename: str = "logo_bg.png"):
    """
    Loads an image file (filename located next to this script) and injects CSS:
      - watermark: faint centered image behind the UI
      - cover: full-screen cover background
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(base_dir, filename)
    if not os.path.exists(logo_path):
        # nothing to do if file missing
        return

    try:
        b64 = _file_to_base64(logo_path)
        ext = os.path.splitext(logo_path)[1].lower().replace(".", "")
        mime = "image/png" if ext not in ("jpg", "jpeg") else "image/jpeg"

        if style == "watermark":
            css = f"""
            <style>
            .stApp {{
              background-color: transparent;
            }}
            .stApp::before {{
              content: "";
              position: fixed;
              left: 50%;
              top: 50%;
              transform: translate(-50%, -50%);
              width: 60vw;
              max-width: 1200px;
              max-height: 700px;
              background-image: url("data:{mime};base64,{b64}");
              background-repeat: no-repeat;
              background-position: center;
              background-size: contain;
              opacity: 0.12;  /* adjust faintness here */
              z-index: 0;
              pointer-events: none;
            }}
            .stApp > .main {{
              position: relative;
              z-index: 1;
            }}
            .stApp .main {{
              background-color: rgba(255,255,255,0.88);
              padding: 1rem;
              border-radius: 8px;
            }}
            </style>
            """
        else:  # cover
            css = f"""
            <style>
            .stApp {{
              background-image: url("data:{mime};base64,{b64}");
              background-size: cover;
              background-position: center;
              background-repeat: no-repeat;
              background-attachment: fixed;
            }}
            .stApp .main {{
              background-color: rgba(255,255,255,0.78);
              padding: 1rem;
              border-radius: 8px;
            }}
            </style>
            """

        st.markdown(css, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Couldn't set permanent background: {e}")

# Apply permanent background immediately
set_permanent_background(style=BACKGROUND_STYLE, filename="logo_bg.png")


# ---------- UI content ----------
st.title("CapSA service model runner")

st.markdown(
    """
    This app will load `capsa_service_network.py` from the project folder and run its `main()` function.
    Make sure `capsa_service_network.py` is present in the folder and that your input Excel is named `network_inputs.xlsx`.
    """
)

# Project folder input
default_proj = os.getcwd()
PROJECT_DIR = st.text_input("Project folder (where capsa_service_network.py lives)", value=default_proj)

# show files in folder
st.subheader("Project folder & files")
if not os.path.exists(PROJECT_DIR):
    st.error(f"Project folder does not exist: {PROJECT_DIR}")
    st.stop()

try:
    files = os.listdir(PROJECT_DIR)
    st.write(files)
except Exception as e:
    st.error(f"Could not list directory {PROJECT_DIR}: {e}")
    st.stop()

# Model parameter inputs
st.subheader("CapSA Network Design_QWIK Supply Chain")
iters = st.number_input("Iterations (iters)", value=100, min_value=1, step=1)
pop_size = st.number_input("Population size (pop_size)", value=60, min_value=1, step=1)
primary_radius = st.number_input("Primary service radius (km)", value=30.0, step=1.0, format="%.1f")
secondary_radius = st.number_input("Secondary service radius (km)", value=50.0, step=1.0, format="%.1f")
d_max = st.number_input("Max assign distance (D_MAX)", value=150.0, step=1.0, format="%.1f")
viable_kg = st.number_input("Min hub viability (KG)", value=20000.0, step=1000.0, format="%.1f")
replace_radius = st.number_input("Merge / replace radius (km)", value=5.0, step=0.5, format="%.1f")
run_button = st.button("Run CapSA model")

# ---------- helper: robust module loader ----------
def load_capsa_module(project_dir):
    project_dir = os.path.abspath(project_dir)
    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)

    try:
        if MODULE_NAME in sys.modules:
            mod = importlib.reload(sys.modules[MODULE_NAME])
        else:
            mod = importlib.import_module(MODULE_NAME)
        return mod
    except Exception as e_import:
        module_path = os.path.join(project_dir, MODULE_FILE)
        if not os.path.exists(module_path):
            raise FileNotFoundError(f"{MODULE_FILE} not found in {project_dir}. Original import error: {e_import}")
        try:
            spec = importlib.util.spec_from_file_location(MODULE_NAME, module_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sys.modules[MODULE_NAME] = mod
            return mod
        except Exception as e_file:
            raise RuntimeError(f"Failed to load {MODULE_FILE} from {project_dir}. Import error: {e_import}\nFile load error: {e_file}")

# ---------- UI: upload Excel (optional) ----------
st.subheader("Input Excel (optional)")
uploaded = st.file_uploader("Upload network_inputs.xlsx (Pins,DCs,Hubs). If provided it will replace the file in the project folder.", type=["xlsx","xls"], accept_multiple_files=False)
use_uploaded_excel = False
if uploaded is not None:
    try:
        safe_path = os.path.join(PROJECT_DIR, "network_inputs.xlsx")
        with open(safe_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Saved uploaded file to: {safe_path}")
        use_uploaded_excel = True
    except Exception as e:
        st.error(f"Failed to save uploaded file to project dir: {e}")
        st.stop()

# ---------- Run the model ----------
if run_button:
    st.info("Loading module and running model... (this may take time)")
    try:
        cs = load_capsa_module(PROJECT_DIR)
    except Exception as e:
        st.error("Failed to load module: " + str(e))
        st.text(traceback.format_exc())
        st.stop()

    if hasattr(cs, "CFG") and isinstance(cs.CFG, dict):
        try:
            cs.CFG["iters"] = int(iters)
            cs.CFG["pop_size"] = int(pop_size)
            cs.CFG["R30"] = float(primary_radius)
            cs.CFG["R50"] = float(secondary_radius)
            cs.CFG["D_MAX"] = float(d_max)
            cs.CFG["VIABLE_KG"] = float(viable_kg)
            cs.CFG["REPLACE_RADIUS"] = float(replace_radius)
            cs.CFG["out_dir"] = os.path.join(PROJECT_DIR, cs.CFG.get("out_dir", "out"))
            if use_uploaded_excel:
                cs.CFG["excel_path"] = os.path.join(PROJECT_DIR, "network_inputs.xlsx")
            st.success("Module CFG updated with UI values.")
        except Exception as e:
            st.warning("Couldn't set some CFG entries: " + str(e))

    log_buf = io.StringIO()
    start_time = time.time()
    try:
        with st.spinner("Running capsa_service_network.main() ... (this can take several minutes)"):
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = log_buf
            sys.stderr = log_buf
            try:
                if hasattr(cs, "main"):
                    cs.main()
                elif hasattr(cs, "run_capsa"):
                    cs.run_capsa()
                elif hasattr(cs, "run"):
                    cs.run()
                else:
                    raise AttributeError("Module has no callable `main()`/`run_capsa()`/`run()`")
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    except Exception as e:
        st.error("Model execution failed: " + str(e))
        st.text("=== Captured logs ===")
        st.text(log_buf.getvalue())
        st.text(traceback.format_exc())
        st.stop()
    finally:
        elapsed = time.time() - start_time

    st.success(f"Model run finished in {elapsed:.1f}s.")
    with st.expander("Show run logs"):
        st.text(log_buf.getvalue())

    # show output files in out dir
    out_dir = None
    try:
        out_dir = cs.CFG.get("out_dir") if hasattr(cs, "CFG") else os.path.join(PROJECT_DIR, "out")
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(PROJECT_DIR, out_dir)
    except Exception:
        out_dir = os.path.join(PROJECT_DIR, "out")

    st.write("Model output directory:", out_dir)
    if os.path.exists(out_dir):
        outs = sorted(os.listdir(out_dir))
        st.write(outs)
        for fn in outs:
            path = os.path.join(out_dir, fn)
            if fn.lower().endswith(".csv"):
                try:
                    df = pd.read_csv(path)
                    st.subheader(fn)
                    st.dataframe(df.head(50))
                except Exception:
                    st.write(f"Can't preview {fn}")
        for fn in outs:
            if fn.lower().endswith((".xlsx", ".csv")):
                path = os.path.join(out_dir, fn)
                with open(path, "rb") as f:
                    data = f.read()
                st.download_button(label=f"Download {fn}", data=data, file_name=fn)
    else:
        st.warning("Output directory does not exist (module may not have written files).")

st.markdown("---")
st.caption("Make sure streamlit is launched from the project folder (or that your hosting provider uses this repo).")
