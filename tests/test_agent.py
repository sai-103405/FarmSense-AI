from src.agent import route_question


def test_crop_question_route():
    route = route_question(
        "What crop should I grow?"
    )

    assert route == "crop"


def test_soil_question_route():
    route = route_question(
        "What crop is suitable for this soil?"
    )

    assert route == "crop"


def test_knowledge_question_route():
    route = route_question(
        "What are the symptoms of tomato early blight?"
    )

    assert route == "knowledge"


def test_disease_question_route():
    route = route_question(
        "What disease is on this leaf?"
    )

    assert route == "knowledge"


def test_image_question_route():
    route = route_question(
        "Analyze this leaf",
        image_path="test_leaf.jpg"
    )

    assert route == "disease"
