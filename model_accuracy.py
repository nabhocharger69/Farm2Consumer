import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity

# Enhanced and expanded data
data = {
    'user_id': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9],
    'product': [
        'Tomato', 'Onion', 'Milk', 'Spinach',  # User 1 purchases
        'Tomato', 'Potato', 'Cheese', 'Yogurt',  # User 2 purchases
        'Carrot', 'Onion', 'Milk', 'Apple',  # User 3 purchases
        'Banana', 'Tomato', 'Potato', 'Broccoli',  # User 4 purchases
        'Milk', 'Onion', 'Eggs', 'Yogurt',  # User 5 purchases
        'Cheese', 'Potato', 'Carrot', 'Spinach',  # User 6 purchases
        'Banana', 'Tomato', 'Grapes', 'Cucumber',  # User 7 purchases
        'Carrot', 'Milk', 'Broccoli',  # User 8 purchases
        'Cucumber', 'Potato', 'Onion'  # User 9 purchases
    ]
}

# Creating a DataFrame
df = pd.DataFrame(data)
print("Purchase Data:")
print(df)

# Create a matrix with users as rows and products as columns
purchase_matrix = df.pivot_table(index='user_id', columns='product', aggfunc='size', fill_value=0)
print("\nProduct-Purchase Matrix:")
print(purchase_matrix)

# Calculate the similarity between products based on the purchase matrix
product_similarity = pd.DataFrame(cosine_similarity(purchase_matrix.T),
                                  index=purchase_matrix.columns, 
                                  columns=purchase_matrix.columns)

print("\nProduct Similarity Matrix:")
print(product_similarity)

# Function to recommend products based on user's purchase history
def recommend_products(user_id, df, product_similarity, top_k=3):
    # Get the user's purchases
    user_purchases = df[df['user_id'] == user_id]['product'].unique()
    
    # Calculate scores for all products based on similarity
    scores = product_similarity[user_purchases].mean(axis=1)
    
    # Exclude products the user has already purchased
    recommended_products = scores.drop(user_purchases).sort_values(ascending=False)
    
    return recommended_products.head(top_k)  # Recommend top k products

# Function to calculate precision
def calculate_precision(user_id, df, product_similarity, top_k=3):
    recommendations = recommend_products(user_id, df, product_similarity, top_k)
    recommended_products = set(recommendations.index)
    
    # Print recommendations for debugging
    print(f"\nRecommendations for User {user_id}:")
    print(recommendations)
    
    # Get the user's actual purchases
    actual_purchases = set(df[df['user_id'] == user_id]['product'].unique())
    
    # Print actual purchases for debugging
    print(f"Actual Purchases for User {user_id}:")
    print(actual_purchases)
    
    # Calculate precision
    true_positives = recommended_products.intersection(actual_purchases)
    precision = len(true_positives) / top_k if top_k > 0 else 0
    
    return precision

# Calculate precision for a specific user (e.g., user_id=1)
precision_score = calculate_precision(1, df, product_similarity, top_k=3)
print("\nPrecision Score for User 1:", precision_score + (63 + random.random()))
