"""Run locally: python -m streamlit run dashboard.py
Deploy: Streamlit Community Cloud (main file = dashboard.py)"""
import time, os
import joblib, numpy as np, pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Classroom Air Quality", page_icon="🌬️", layout="wide")
COLORS = {"Good": "#2ecc71", "Moderate": "#f1c40f", "Poor": "#e74c3c"}
FEATURES = ["gas", "temperature", "humidity"]


@st.cache_resource
def load_model():
    """Load the saved model; if it cannot be loaded (e.g. library version change), retrain from the CSV."""
    try:
        if os.path.exists("air_quality_model.joblib"):
            return joblib.load("air_quality_model.joblib")
    except Exception:
        pass
    if os.path.exists("air_quality_data.csv"):
        from sklearn.tree import DecisionTreeClassifier
        df = pd.read_csv("air_quality_data.csv").dropna()
        return DecisionTreeClassifier(max_depth=3, random_state=42).fit(df[FEATURES], df["label"])
    return None


@st.cache_resource
def get_serial(port):
    import serial
    return serial.Serial(port, 115200, timeout=2)


@st.cache_data(ttl=15)
def fetch_cloud(channel, key, n=100):
    """Read the latest n records from a ThingSpeak channel (field1=temp, field2=humidity, field3=gas)."""
    import requests
    params = {"results": n}
    if key:
        params["api_key"] = key
    r = requests.get(f"https://api.thingspeak.com/channels/{channel}/feeds.json", params=params, timeout=10)
    r.raise_for_status()
    df = pd.DataFrame(r.json().get("feeds", []))
    if df.empty:
        return df
    df = df.rename(columns={"created_at": "time", "field1": "temperature", "field2": "humidity", "field3": "gas"})
    df["time"] = pd.to_datetime(df["time"])
    for c in ["temperature", "humidity", "gas"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["temperature", "humidity", "gas"])[["time", "temperature", "humidity", "gas"]]


model = load_model()
st.title(" Air-Quality Monitor (ML)")
if model is None:
    st.error("Model not found. Run the notebook first to create air_quality_model.joblib")
    st.stop()


def predict(t, h, g):
    X = pd.DataFrame([[g, t, h]], columns=FEATURES)
    return model.predict(X)[0]


def show_dashboard(d):
    last = d.iloc[-1]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Temperature", f"{last.temperature:.1f} °C")
    k2.metric("Humidity", f"{last.humidity:.1f} %")
    k3.metric("Gas (ADC)", int(last.gas))
    k4.markdown(f"<h2 style='color:{COLORS[last.prediction]}'>{last.prediction}</h2>", unsafe_allow_html=True)
    if last.prediction == "Poor":
        st.warning("Poor air quality: open windows / ventilate the classroom.")

    gauge = go.Figure(go.Indicator(mode="gauge+number", value=float(last.gas),
        title={"text": "MQ135 raw ADC"},
        gauge={"axis": {"range": [0, 1023]},
               "steps": [{"range": [0, 350], "color": "#d5f5e3"},
                         {"range": [350, 650], "color": "#fcf3cf"},
                         {"range": [650, 1023], "color": "#fadbd8"}]}))
    c1, c2 = st.columns([1, 2])
    c1.plotly_chart(gauge, use_container_width=True)
    c2.plotly_chart(px.line(d, x="time", y=["temperature", "humidity"], title="Temperature & Humidity"),
                    use_container_width=True)
    st.plotly_chart(px.scatter(d, x="time", y="gas", color="prediction",
                               color_discrete_map=COLORS, title="Gas reading & predicted class"),
                    use_container_width=True)
    st.download_button("Download data", d.to_csv(index=False), "session_data.csv")


mode = st.sidebar.radio("Data source", ["Cloud (ThingSpeak)", "Simulation", "Live (USB serial)",
                                        "Upload CSV", "Manual input"])

# ---------- Manual ----------
if mode == "Manual input":
    c1, c2, c3 = st.columns(3)
    t = c1.slider("Temperature (°C)", 10.0, 45.0, 25.0)
    h = c2.slider("Humidity (%)", 10.0, 100.0, 55.0)
    g = c3.slider("MQ135 ADC (0-1023)", 0, 1023, 300)
    r = predict(t, h, g)
    st.markdown(f"<h1 style='color:{COLORS[r]}'>Air Quality: {r}</h1>", unsafe_allow_html=True)

# ---------- Upload ----------
elif mode == "Upload CSV":
    f = st.file_uploader("CSV with columns: temperature, humidity, gas", type="csv")
    if f:
        df = pd.read_csv(f)
        df["prediction"] = model.predict(df[FEATURES])
        st.dataframe(df.tail(50), use_container_width=True)
        st.plotly_chart(px.pie(df, names="prediction", color="prediction",
                               color_discrete_map=COLORS, title="Prediction share"), use_container_width=True)
        st.plotly_chart(px.line(df.reset_index(), x="index", y="gas", color="prediction",
                                color_discrete_map=COLORS, title="Gas reading"), use_container_width=True)

# ---------- Cloud (ThingSpeak) ----------
elif mode.startswith("Cloud"):
    st_autorefresh(interval=15000, key="refresh_cloud")
    try:
        conf = dict(st.secrets["thingspeak"])
    except Exception:
        conf = {}
    channel = conf.get("channel_id") or st.sidebar.text_input("ThingSpeak channel ID")
    key = conf.get("read_key", "") or st.sidebar.text_input("Read API key (empty if channel is public)", type="password")
    if not channel:
        st.info("Enter the ThingSpeak channel ID (or set it in the app secrets) to see live data.")
    else:
        try:
            d = fetch_cloud(str(channel), key)
        except Exception as e:
            st.error(f"Could not read ThingSpeak: {e}")
            st.stop()
        if d.empty:
            st.warning("No data in this channel yet. Check that the NodeMCU is powered and connected to WiFi.")
        else:
            d = d.copy()
            d["prediction"] = model.predict(d[FEATURES])
            last = d.iloc[-1]
            age = (pd.Timestamp.now(tz="UTC") - last.time).total_seconds() / 60
            st.caption(f"Last reading {age:.0f} min ago")
            show_dashboard(d)

# ---------- Live USB / Simulation ----------
else:
    st_autorefresh(interval=2000, key="refresh")
    if "hist" not in st.session_state:
        st.session_state.hist = pd.DataFrame(columns=["time", "temperature", "humidity", "gas", "prediction"])
    ser = None
    if mode.startswith("Live"):
        port = st.sidebar.text_input("Serial port", "COM10")
        try:
            ser = get_serial(port)
        except Exception as e:
            st.error(f"Cannot open {port}: {e}  (USB serial works only on the PC the NodeMCU is plugged into)")
            st.stop()

    t = h = g = None
    if ser:
        line = ser.readline().decode(errors="ignore").strip().split(",")
        if len(line) == 3:
            try:
                t, h, g = float(line[0]), float(line[1]), int(line[2])
            except ValueError:
                pass
    else:
        rng = np.random.default_rng(int(time.time()))
        t, h, g = rng.normal(28, 3), rng.normal(60, 12), int(rng.normal(450, 200))
        g = max(0, min(1023, g))

    if t is not None:
        r = predict(t, h, g)
        row = pd.DataFrame([[pd.Timestamp.now(), t, h, g, r]], columns=st.session_state.hist.columns)
        st.session_state.hist = pd.concat([st.session_state.hist, row]).tail(300)

    if len(st.session_state.hist):
        show_dashboard(st.session_state.hist)
