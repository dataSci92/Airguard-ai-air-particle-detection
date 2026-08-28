from collections import Counter


def summarize_detections(
    detections: list[dict]
) -> dict:

    counts = Counter(
        detection["class"]
        for detection in detections
    )


    total = len(detections)


    if total == 0:

        indicator = "Low"

    elif total < 5:

        indicator = "Moderate"

    else:

        indicator = "High"


    average_confidence = (

        sum(
            detection["confidence"]
            for detection in detections
        ) / total

        if total

        else 0.0
    )


    return {

        "total_detections":
            total,

        "classes":
            dict(counts),

        "average_confidence":
            average_confidence,

        "visual_pollution_indicator":
            indicator

    }
