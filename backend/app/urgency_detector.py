def detect_urgency(review):

    review = str(review).lower()

    urgent_keywords = [
        "fraud",
        "scam",
        "refund",
        "legal",
        "police",
        "court",
        "fake",
        "broken",
        "damaged",
        "late",
        "delay",
        "delayed",
        "deducted",
        "charged",
        "payment",
        "stolen",
        "cancel",
        "account blocked",
        "account frozen"
    ]
     
    matched_keywords = []

    for keyword in urgent_keywords:
        if keyword in review:
            matched_keywords.append(keyword)

    return {
        "urgent": len(matched_keywords) > 0,
        "matched_keywords": matched_keywords
    }