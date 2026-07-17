def route_review(category, urgent):

    routes = {

        "Delivery": "Delivery Team",

        "Refund": "Finance Team",

        "Payment": "Payment Team",

        "Product Quality": "Quality Assurance Team",

        "Customer Support": "Customer Support Team",

        "Other": "General Support Team"

    }

    assigned_team = routes.get(category, "General Support Team")

    priority = "High" if urgent else "Normal"

    return {
        "assigned_team": assigned_team,
        "priority": priority
    }