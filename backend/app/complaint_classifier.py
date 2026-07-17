def classify_complaint(review):

    if review is None:
        return "Other"

    review = str(review).lower()

    if "delivery" in review or "late" in review:
        return "Delivery"

    elif "refund" in review or "money" in review:
        return "Refund"

    elif "payment" in review or "charged" in review:
        return "Payment"

    elif "broken" in review or "quality" in review:
        return "Product Quality"

    elif "support" in review or "customer service" in review:
        return "Customer Support"

    return "Other"