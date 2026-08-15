from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
from threading import Timer
import webbrowser
import re

app = Flask(__name__)

# -------------------------------------------------------------------
# FOOD DATASET
# -------------------------------------------------------------------

data = {
    "Food": [
        "Pizza", "Pasta", "Lasagna", "Risotto", "Burger",
        "French Fries", "Chicken Wings", "Sushi", "Sushi Roll", "Ramen",
        "Tempura", "Miso Soup", "Salad", "Greek Salad", "Caesar Salad",
        "Tacos", "Burrito", "Quesadilla", "Nachos", "Guacamole",
        "Pancakes", "Waffles", "Omelette", "French Toast", "Idli",
        "Dosa", "Masala Dosa", "Samosa", "Biryani", "Chicken Biryani",
        "Fried Rice", "Noodles", "Butter Chicken", "Chicken Curry",
        "Paneer Tikka", "Palak Paneer", "Chole", "Rajma Rice",
        "Falafel", "Hummus", "Shawarma", "Grilled Chicken",
        "Fruit Salad", "Oatmeal", "Porridge", "Smoothie", "Ice Cream",
        "Brownie", "Cheesecake", "Chocolate Cake"
    ],
    "Ingredients": [
        "dough cheese tomato pepperoni herbs",
        "pasta tomato sauce garlic cheese herbs",
        "pasta tomato sauce cheese beef herbs",
        "rice parmesan cheese butter mushroom herbs",
        "beef lettuce tomato cheese bun onion",
        "potato salt oil crispy",
        "chicken wings garlic pepper chili sauce",
        "rice fish seaweed wasabi soy sauce",
        "rice fish avocado seaweed soy sauce sesame",
        "noodles chicken broth egg soy sauce mushroom",
        "shrimp flour oil soy sauce tempura",
        "miso tofu seaweed soy sauce broth",
        "lettuce tomato cucumber carrot dressing healthy",
        "lettuce tomato cucumber olives feta cheese dressing",
        "lettuce parmesan cheese croutons dressing",
        "corn tortilla meat lettuce cheese salsa chili",
        "tortilla rice beans meat cheese salsa",
        "tortilla cheese pepper onion beans",
        "corn tortilla cheese jalapeno salsa chips",
        "avocado tomato onion lime chili healthy",
        "flour eggs milk sugar syrup breakfast",
        "flour eggs milk butter sugar breakfast",
        "eggs onion tomato pepper cheese breakfast",
        "bread eggs milk cinnamon sugar breakfast",
        "rice lentil fermented batter healthy breakfast",
        "rice lentil fermented batter potato chutney breakfast",
        "rice lentil potato spices batter breakfast",
        "potato peas flour onion spices snack",
        "rice chicken spices onion saffron yogurt",
        "rice chicken spices onion saffron yogurt",
        "rice vegetables soy sauce egg garlic",
        "noodles vegetables soy sauce garlic chili",
        "chicken tomato butter cream spices",
        "chicken onion tomato spices garlic curry",
        "paneer yogurt chili spices grilled",
        "paneer spinach cream spices healthy",
        "chickpea onion tomato spices healthy",
        "kidney beans rice tomato onion spices",
        "chickpea tahini parsley garlic lemon healthy",
        "chickpea tahini garlic lemon olive oil healthy",
        "chicken pita garlic yogurt onion spices",
        "chicken herbs pepper garlic grilled healthy",
        "apple banana orange grapes berries healthy",
        "oats milk banana honey healthy breakfast",
        "oats milk sugar cinnamon healthy breakfast",
        "banana berries milk yogurt healthy",
        "milk cream sugar vanilla dessert",
        "chocolate flour butter sugar cocoa dessert",
        "cream cheese sugar biscuit vanilla dessert",
        "flour cocoa chocolate eggs sugar dessert"
    ],
    "Category": [
        "Italian", "Italian", "Italian", "Italian",
        "American", "American", "American",
        "Japanese", "Japanese", "Japanese", "Japanese", "Japanese",
        "Healthy", "Healthy", "Healthy",
        "Mexican", "Mexican", "Mexican", "Mexican", "Mexican",
        "Breakfast", "Breakfast", "Breakfast", "Breakfast",
        "Indian", "Indian", "Indian", "Indian", "Indian", "Indian",
        "Asian", "Asian", "Indian", "Indian", "Indian", "Indian",
        "Indian", "Indian", "Middle Eastern", "Middle Eastern",
        "Middle Eastern", "Healthy",
        "Healthy", "Healthy", "Healthy", "Healthy",
        "Dessert", "Dessert", "Dessert", "Dessert"
    ]
}

df = pd.DataFrame(data)

# -------------------------------------------------------------------
# TEXT NORMALIZATION / ALIASES
# -------------------------------------------------------------------
def normalize_text(text):
    """Normalize any user-entered text safely."""
    text = "" if text is None else str(text)
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

# Common misspellings / alternate spellings.
ALIASES = {
    "shushi": "sushi",
    "sushie": "sushi",
    "sush": "sushi",
    "sushi roll": "sushi roll",
    "frenchfries": "french fries",
    "masala dosa": "masala dosa",
    "biriyani": "biryani",
    "biriani": "biryani",
    "chiken biryani": "chicken biryani",
    "chiken": "chicken",
    "omlette": "omelette",
    "omlet": "omelette",
    "pancake": "pancakes",
    "waffle": "waffles",
    "burger": "burger",
    "burgar": "burger",
    "piza": "pizza",
    "pizaa": "pizza",
    "pizaaa": "pizza",
    "pastaa": "pasta",
    "noodle": "noodles",
    "fried rice": "fried rice",
    "icecream": "ice cream",
    "ice cream": "ice cream",
}

# Dataset lookup using normalized names.
food_lookup = {
    normalize_text(food): food
    for food in df["Food"]
}

# Include aliases in lookup.
for alias, target in ALIASES.items():
    target_key = normalize_text(target)
    if target_key in food_lookup:
        food_lookup[normalize_text(alias)] = food_lookup[target_key]

# -------------------------------------------------------------------
# TF-IDF MODEL
# -------------------------------------------------------------------
df["SearchText"] = (
    df["Food"].astype(str) + " " +
    df["Ingredients"].astype(str)
)

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(df["SearchText"])


def find_exact_or_fuzzy_food(user_text):
    """
    Returns:
        (matched_food_name, confidence)
    or:
        (None, 0)
    """
    cleaned = normalize_text(user_text)

    if not cleaned:
        return None, 0.0

    # Exact / alias match.
    if cleaned in food_lookup:
        return food_lookup[cleaned], 1.0

    # Fuzzy match against food names.
    best_food = None
    best_score = 0.0

    for normalized_name, real_name in food_lookup.items():
        score = SequenceMatcher(None, cleaned, normalized_name).ratio()

        # Also compare individual words for inputs such as
        # "chicken biriyani please".
        if cleaned in normalized_name or normalized_name in cleaned:
            score = max(score, 0.90)

        if score > best_score:
            best_score = score
            best_food = real_name

    # Only accept a fuzzy match when it is reasonably close.
    if best_score >= 0.68:
        return best_food, best_score

    return None, 0.0


def recommend_food(input_text, category_filter="All", limit=5):
    """
    Recommendation engine.

    It safely handles:
      - exact food names
      - upper/lower case
      - common spelling mistakes
      - partial food names
      - ingredient/keyword text such as "chicken spicy"
      - category filtering
      - unknown input without crashing
    """
    user_text = normalize_text(input_text)

    if not user_text:
        return [], None, "Please enter a food name or food-related text."

    # Keep only valid category values.
    valid_categories = set(df["Category"]) | {"All"}
    if category_filter not in valid_categories:
        category_filter = "All"

    matched_food, confidence = find_exact_or_fuzzy_food(user_text)

    # Candidate rows based on the selected category.
    candidates = df.copy()

    if category_filter != "All":
        candidates = candidates[candidates["Category"] == category_filter].copy()

    if candidates.empty:
        return [], matched_food, f"No foods are available in the '{category_filter}' category."

    candidate_indices = candidates.index.tolist()

    # ---------------------------------------------------------------
    # CASE 1: Input matches a known food
    # ---------------------------------------------------------------
    if matched_food is not None:
        source_index = df.index[df["Food"] == matched_food][0]

        # Similarity from the selected food to all foods.
        similarities = cosine_similarity(
            tfidf_matrix[source_index],
            tfidf_matrix[candidate_indices]
        )[0]

        scored = list(zip(candidate_indices, similarities))
        scored.sort(key=lambda x: x[1], reverse=True)

        recommendations = []

        for index, score in scored:
            # Do not recommend the exact same food.
            if df.at[index, "Food"] == matched_food:
                continue

            recommendations.append({
                "food": df.at[index, "Food"],
                "category": df.at[index, "Category"],
                "score": round(float(score), 3)
            })

            if len(recommendations) == limit:
                break

        return recommendations, matched_food, None

    # ---------------------------------------------------------------
    # CASE 2: Input is not a known food.
    # Use the typed text as a search query against food names +
    # ingredients. This lets inputs such as "chicken", "spicy rice",
    # "cheesy pasta", etc. work without crashing.
    # ---------------------------------------------------------------
    query_vector = vectorizer.transform([user_text])
    similarities = cosine_similarity(query_vector, tfidf_matrix[candidate_indices])[0]

    scored = list(zip(candidate_indices, similarities))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Require at least a small semantic/textual connection.
    meaningful = [(index, score) for index, score in scored if score >= 0.05]

    if not meaningful:
        return [], None, (
            f"Couldn't find '{input_text}' in the food dataset. "
            "Please try a different food name"

        )

    recommendations = []
    for index, score in meaningful[:limit]:
        recommendations.append({
            "food": df.at[index, "Food"],
            "category": df.at[index, "Category"],
            "score": round(float(score), 3)
        })

    return recommendations, None, None


@app.route("/", methods=["GET", "POST"])
def index():
    recommended_foods = []
    input_food = ""
    category_filter = "All"
    matched_food = None
    message = None
    message_type = ""

    if request.method == "POST":
        input_food = request.form.get("food", "").strip()
        category_filter = request.form.get("category", "All")

        (
            recommended_foods,
            matched_food,
            message
        ) = recommend_food(input_food, category_filter)

        if message:
            message_type = "error"
        elif matched_food:
            message = f"Recommendations based on: {matched_food}"
            message_type = "success"
        else:
            message = f"Recommendations based on: {input_food}"
            message_type = "success"

    categories = ["All"] + sorted(df["Category"].unique().tolist())

    return render_template(
        "index.html",
        recommended_foods=recommended_foods,
        input_food=input_food,
        category_filter=category_filter,
        categories=categories,
        message=message,
        message_type=message_type
    )


def open_browser():
    """Open the application in the user's normal external browser."""
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == "__main__":
    # Automatically opens Chrome/Edge/default browser.
    Timer(1.0, open_browser).start()

    # use_reloader=False prevents Flask from opening the browser twice.
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )
