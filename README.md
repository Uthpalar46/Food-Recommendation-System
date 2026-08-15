## Overview

A web-based food recommendation system built with Flask that recommends similar foods based on food ingredients, and selected categories. It uses TF-IDF and cosine similarity to generate relevant recommendations.

## Features

* Ingredient-based food recommendations
* TF-IDF and cosine similarity
* Category filtering
* Fuzzy matching for spelling mistakes
* Partial and keyword-based search
* Handles unknown inputs safely
* Responsive web interface

## Tech Stack

* *Backend:* Python, Flask
* *Machine Learning:* Scikit-learn, TF-IDF, Cosine Similarity
* *Data Processing:* Pandas
* *Frontend:* HTML, CSS, JavaScript

## How It Works
1. User enters a food name or food-related keywords.
2. Input is normalized and checked for exact or fuzzy matches.
3. TF-IDF converts food and ingredient information into numerical vectors.
4. Cosine similarity identifies the most similar foods.
5. The top recommendations are displayed based on the selected category.

## Project Structure
```Food_Recommendation_System/
├── backend.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Installation

Install the required packages:
* pip install -r requirements.txt
  
Run the application:
* python backend.py

## Author

Developed as an academic Machine Learning project using Python, Flask, and Scikit-learn.
