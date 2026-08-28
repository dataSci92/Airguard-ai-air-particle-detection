import io

import streamlit as st
from PIL import Image, ImageDraw

from analytics import summarize_detections
from detector import AirGuardDetector


st.set_page_config(
    page_title="AirGuard AI",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 AirGuard AI")
st.caption("AI-assisted visual pollution monitoring and air-quality analytics")


# Sidebar
with st.sidebar:
    st.header("Configuration")

    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
    )

    st.divider()

    st.info(
        "For real PM2.5/PM10 measurement, connect calibrated "
        "air-quality sensors. The camera model is intended for "
        "visible pollution indicators."
    )


# Load detector
detector = AirGuardDetector()


# Dashboard metrics
col1, col2, col3 = st.columns(3)

col1.metric(
    "Model Status",
    "Ready" if detector.ready else "Model Not Installed"
)

col2.metric(
    "Detection Mode",
    "YOLO"
)

col3.metric(
    "Sensor Mode",
    "Optional"
)


# Upload image
uploaded = st.file_uploader(
    "Upload a camera frame",
    type=["jpg", "jpeg", "png", "webp"],
)


if uploaded:

    image = Image.open(
        io.BytesIO(uploaded.read())
    ).convert("RGB")

    st.subheader("AI Detection")

    if st.button(
        "Run AirGuard Detection",
        type="primary"
    ):

        if not detector.ready:

            st.warning(
                "No trained model was found. "
                "Add your YOLO model at models/best.pt "
                "and run the application again."
            )

            st.image(
                image,
                caption="Uploaded camera frame",
                use_container_width=True,
            )

        else:

            with st.spinner("Running AI detection..."):

                detections = detector.predict(
                    image,
                    confidence
                )

            # Draw detections
            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)

            for detection in detections:

                x1, y1, x2, y2 = detection["bbox"]

                label = (
                    f'{detection["class"]} '
                    f'{detection["confidence"]:.0%}'
                )

                draw.rectangle(
                    (x1, y1, x2, y2),
                    outline="lime",
                    width=3,
                )

                draw.text(
                    (x1 + 4, y1 + 4),
                    label,
                    fill="lime",
                )


            # Analytics
            summary = summarize_detections(
                detections
            )


            left, right = st.columns(
                [1.5, 1]
            )


            with left:

                st.image(
                    annotated,
                    caption="AI-annotated frame",
                    use_container_width=True,
                )


            with right:

                st.metric(
                    "Detections",
                    summary["total_detections"]
                )

                st.metric(
                    "Average Confidence",
                    f'{summary["average_confidence"]:.1%}'
                )

                st.metric(
                    "Visual Pollution Indicator",
                    summary[
                        "visual_pollution_indicator"
                    ]
                )

                st.write(
                    "Detected Classes"
                )

                st.json(
                    summary["classes"]
                )

else:

    st.info(
        "Upload a camera frame to start an analysis."
    )


# Sensor section
st.divider()

st.subheader(
    "Optional Sensor Readings"
)

pm1, pm2, pm10 = st.columns(3)

pm1.metric(
    "PM1.0",
    "—",
    help="Connect a calibrated sensor to show live values."
)

pm2.metric(
    "PM2.5",
    "—"
)

pm10.metric(
    "PM10",
    "—"
)


st.caption(
    "AirGuard AI is a portfolio prototype. "
    "Visual detections are not a substitute for "
    "calibrated environmental measurements."
)
