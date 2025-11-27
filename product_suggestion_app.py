import gradio as gr
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from typing import List, Dict, Tuple, Optional
import random
from collections import defaultdict

class MLProductSuggestionApp:
    def __init__(self):
        self.load_model_components()
        self.load_sample_data()
        
    def load_model_components(self):
        """Load trained model components from notebook"""
        try:
            # Load encoders and dictionaries
            with open('visitor_encoder.pkl', 'rb') as f:
                self.visitor_encoder = pickle.load(f)
            with open('item_encoder.pkl', 'rb') as f:
                self.item_encoder = pickle.load(f)
            with open('wide_features_dicts.pkl', 'rb') as f:
                self.dicts = pickle.load(f)
            
            self.user_event_count_dict = self.dicts['user_event_count_dict']
            self.item_event_count_dict = self.dicts['item_event_count_dict']
            self.item_available_dict = self.dicts['item_available_dict']
            
            # Try to load trained model
            try:
                self.model = tf.keras.models.load_model('best_model.keras')
                self.model_loaded = True
                print("✅ Trained model loaded successfully")
            except Exception as e:
                print(f"Model loading error: {e}")
                # If no trained model, create a mock model for demonstration
                self.create_mock_model()
                self.model_loaded = False
                print("⚠️ No trained model found, using mock predictions")
                
        except Exception as e:
            print(f"❌ Error loading model components: {e}")
            self.create_fallback_data()
    
    def create_mock_model(self):
        """Create a mock model for demonstration when no trained model is available"""
        # This simulates the model structure for demonstration
        self.model = None
        self.model_loaded = False
    
    def load_sample_data(self):
        """Load sample product data based on e-commerce dataset"""
        # Sample products inspired by retailrocket dataset
        self.products = [
            {"id": 355908, "name": "Wireless Headphones", "category": "Electronics", "price": 79.99, "rating": 4.5, "description": "High-quality wireless headphones with noise cancellation"},
            {"id": 248676, "name": "Smart Watch", "category": "Electronics", "price": 199.99, "rating": 4.3, "description": "Fitness tracking smartwatch with heart rate monitor"},
            {"id": 318965, "name": "Running Shoes", "category": "Sports", "price": 89.99, "rating": 4.7, "description": "Comfortable running shoes for all terrains"},
            {"id": 253185, "name": "Coffee Maker", "category": "Home", "price": 129.99, "rating": 4.4, "description": "Automatic coffee maker with timer"},
            {"id": 367447, "name": "Yoga Mat", "category": "Sports", "price": 29.99, "rating": 4.6, "description": "Non-slip yoga mat with carrying strap"},
            {"id": 460429, "name": "Bluetooth Speaker", "category": "Electronics", "price": 59.99, "rating": 4.5, "description": "Portable bluetooth speaker with excellent sound quality"},
            {"id": 206783, "name": "Water Bottle", "category": "Sports", "price": 19.99, "rating": 4.8, "description": "Insulated water bottle that keeps drinks cold for 24 hours"},
            {"id": 395014, "name": "Desk Lamp", "category": "Home", "price": 39.99, "rating": 4.1, "description": "LED desk lamp with adjustable brightness"},
            {"id": 59481, "name": "Backpack", "category": "Accessories", "price": 49.99, "rating": 4.2, "description": "Durable backpack with laptop compartment"},
            {"id": 156781, "name": "Phone Case", "category": "Accessories", "price": 15.99, "rating": 4.0, "description": "Protective phone case with stylish design"}
        ]
        
        # Create user-item interaction matrix for collaborative filtering simulation
        self.user_interactions = defaultdict(set)
        # Simulate some user interactions
        for i in range(100):  # 100 sample users
            user_id = 1000 + i  # Sample user IDs
            # Each user interacts with 3-7 random products
            num_interactions = random.randint(3, 7)
            interacted_items = random.sample(self.products, num_interactions)
            for product in interacted_items:
                self.user_interactions[user_id].add(product["id"])
    
    def create_fallback_data(self):
        """Create fallback data when model components can't be loaded"""
        self.products = [
            {"id": 1, "name": "Wireless Headphones", "category": "Electronics", "price": 79.99, "rating": 4.5, "description": "High-quality wireless headphones with noise cancellation"},
            {"id": 2, "name": "Smart Watch", "category": "Electronics", "price": 199.99, "rating": 4.3, "description": "Fitness tracking smartwatch with heart rate monitor"},
            {"id": 3, "name": "Running Shoes", "category": "Sports", "price": 89.99, "rating": 4.7, "description": "Comfortable running shoes for all terrains"}
        ]
        self.user_interactions = defaultdict(set)
        self.model_loaded = False
    
    def get_categories(self) -> List[str]:
        """Get all unique categories"""
        return list(set(product["category"] for product in self.products))
    
    def build_features_for_prediction(self, user_id: int, item_id: int):
        """Build wide and deep features for ML model prediction"""
        try:
            # Wide features: [user_event_count, item_event_count, available]
            u_count = self.user_event_count_dict.get(user_id, 0.0)
            i_count = self.item_event_count_dict.get(item_id, 0.0)
            avail = self.item_available_dict.get(item_id, 1)
            wide_features = np.array([[u_count, i_count, avail]], dtype=np.float32)
            
            # Deep features: [visitor_enc, item_enc]
            if hasattr(self, 'visitor_encoder') and hasattr(self, 'item_encoder'):
                # Encode user and item IDs
                if user_id in self.visitor_encoder.classes_:
                    user_enc = self.visitor_encoder.transform([user_id])[0]
                else:
                    user_enc = len(self.visitor_encoder.classes_)
                
                if item_id in self.item_encoder.classes_:
                    item_enc = self.item_encoder.transform([item_id])[0]
                else:
                    item_enc = len(self.item_encoder.classes_)
                
                deep_features = np.array([[user_enc, item_enc]], dtype=np.float32)
            else:
                # Fallback: use simple encoding
                deep_features = np.array([[user_id % 1000, item_id % 1000]], dtype=np.float32)
            
            return wide_features, deep_features
        except Exception as e:
            print(f"Error building features: {e}")
            # Return fallback features
            return np.array([[0.5, 0.5, 1.0]], dtype=np.float32), np.array([[0, 0]], dtype=np.float32)
    
    def predict_purchase_probability(self, user_id: int, item_id: int) -> float:
        """Predict probability of user purchasing item"""
        if self.model_loaded and self.model is not None:
            try:
                wide_features, deep_features = self.build_features_for_prediction(user_id, item_id)
                prediction = self.model.predict([wide_features, deep_features], verbose=0)
                return float(prediction[0][0])
            except Exception as e:
                print(f"Prediction error: {e}")
        
        # Fallback: simulate prediction based on simple rules
        base_score = 0.1
        
        # Boost score if user has interacted with similar items
        if user_id in self.user_interactions:
            user_items = self.user_interactions[user_id]
            similar_items = [p for p in self.products if p["id"] in user_items and p["category"] == self.get_product_category(item_id)]
            if similar_items:
                base_score += 0.3
        
        # Add some randomness and rating influence
        product = next((p for p in self.products if p["id"] == item_id), None)
        if product:
            base_score += (product["rating"] / 5.0) * 0.2
        
        return min(base_score + random.random() * 0.2, 0.95)
    
    def get_product_category(self, item_id: int) -> str:
        """Get category of a product"""
        product = next((p for p in self.products if p["id"] == item_id), None)
        return product["category"] if product else "Unknown"
    
    def get_ml_recommendations(self, user_id: int, num_recommendations: int = 5) -> Tuple[List[List[str]], str]:
        """Get ML-based recommendations for a user"""
        if not self.products:
            return [], "No products available for recommendations."
        
        # Get predictions for all products user hasn't interacted with
        predictions = []
        user_interacted_items = self.user_interactions.get(user_id, set())
        
        for product in self.products:
            if product["id"] not in user_interacted_items:
                probability = self.predict_purchase_probability(user_id, product["id"])
                predictions.append((product, probability))
        
        # Sort by probability (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Take top recommendations
        top_recommendations = predictions[:num_recommendations]
        
        if not top_recommendations:
            return [], "No new recommendations available for this user."
        
        # Convert to table format
        table_data = []
        for product, probability in top_recommendations:
            table_data.append([
                product["name"],
                product["category"],
                f"${product['price']:.2f}",
                f"{product['rating']}/5.0 ⭐",
                f"{probability:.3f}",
                product["description"]
            ])
        
        message = f"Top {len(top_recommendations)} ML-powered recommendations for User {user_id}"
        return table_data, message
    
    def suggest_products(self, category: str, max_price: float, min_rating: float, user_id: Optional[int] = None) -> Tuple[List[List[str]], str]:
        """Suggest products based on filters and optional ML predictions"""
        filtered_products = []
        
        for product in self.products:
            if (category == "All" or product["category"] == category) and \
               product["price"] <= max_price and \
               product["rating"] >= min_rating:
                filtered_products.append(product)
        
        if not filtered_products:
            return [], "No products found matching your criteria."
        
        # If user_id is provided, use ML to rank products
        if user_id is not None and user_id > 0:
            for product in filtered_products:
                probability = self.predict_purchase_probability(user_id, product["id"])
                product["ml_score"] = probability
            
            # Sort by ML score (descending), then by rating
            filtered_products.sort(key=lambda x: (-x["ml_score"], -x["rating"]))
            ranking_method = "ML-powered"
        else:
            # Sort by rating (descending) and then by price (ascending)
            filtered_products.sort(key=lambda x: (-x["rating"], x["price"]))
            ranking_method = "rating-based"
        
        # Convert to list format for Gradio dataframe
        table_data = []
        for product in filtered_products:
            ml_score = f"{product.get('ml_score', 0):.3f}" if user_id else "N/A"
            table_data.append([
                product["name"],
                product["category"],
                f"${product['price']:.2f}",
                f"{product['rating']}/5.0 ⭐",
                ml_score,
                product["description"]
            ])
        
        message = f"Found {len(filtered_products)} products ({ranking_method} ranking)."
        return table_data, message
    
    def get_product_details(self, product_name: str) -> str:
        """Get detailed information about a specific product"""
        for product in self.products:
            if product["name"] == product_name:
                details = f"""
**{product['name']}**
- Product ID: {product['id']}
- Category: {product['category']}
- Price: ${product['price']:.2f}
- Rating: {product['rating']}/5.0 ⭐
- Description: {product['description']}
                """
                return details.strip()
        return "Product not found."
    
    def get_random_suggestion(self) -> str:
        """Get a random product suggestion"""
        product = random.choice(self.products)
        suggestion = f"💡 **Today's Pick:** {product['name']}\n\n{product['description']}\n\nPrice: ${product['price']:.2f} | Rating: {product['rating']}/5.0"
        return suggestion
    
    def search_products(self, query: str) -> List[List[str]]:
        """Search products by name or description"""
        query = query.lower()
        results = []
        
        for product in self.products:
            if query in product['name'].lower() or query in product['description'].lower():
                results.append([
                    product["name"],
                    product["category"],
                    f"${product['price']:.2f}",
                    f"{product['rating']}/5.0 ⭐",
                    product["description"]
                ])
        
        return results

# Initialize app
try:
    app = MLProductSuggestionApp()
except Exception as e:
    print(f"Error initializing app: {e}")
    # Create a simple fallback app
    app = type('App', (), {
        'products': [],
        'get_categories': lambda: [],
        'get_ml_recommendations': lambda uid, n: ([], "App initialization failed"),
        'suggest_products': lambda cat, price, rating, uid: ([], "App initialization failed"),
        'get_product_details': lambda name: "App initialization failed",
        'get_random_suggestion': lambda: "App initialization failed",
        'search_products': lambda query: []
    })()

def create_interface():
    """Create Gradio interface with ML integration"""
    
    with gr.Blocks(title="ML-Powered Product Suggestion System") as interface:
        gr.Markdown("# 🤖 ML-Powered Product Suggestion System")
        gr.Markdown("Discover products using our trained Wide & Deep learning model!")
        
        # Model status
        model_status = "✅ ML Model Loaded" if hasattr(app, 'model_loaded') and app.model_loaded else "⚠️ Using Mock Predictions (Train model for full functionality)"
        gr.Markdown(f"**Status:** {model_status}")
        
        with gr.Tabs():
            # Tab 1: ML-Powered Recommendations
            with gr.TabItem("🎯 ML Recommendations"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Get Personalized Recommendations")
                        user_id_input = gr.Number(
                            label="User ID",
                            value=1001,
                            minimum=1000,
                            maximum=1100,
                            step=1,
                            info="Enter your user ID for personalized recommendations"
                        )
                        num_recs = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Number of Recommendations"
                        )
                        ml_rec_btn = gr.Button("Get ML Recommendations", variant="primary")
                        
                        gr.Markdown("### Or Filter by Preferences")
                        category_dropdown = gr.Dropdown(
                            choices=["All"] + app.get_categories(),
                            value="All",
                            label="Category"
                        )
                        max_price_slider = gr.Slider(
                            minimum=0,
                            maximum=500,
                            value=200,
                            step=10,
                            label="Maximum Price ($)"
                        )
                        min_rating_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3.5,
                            step=0.5,
                            label="Minimum Rating"
                        )
                        filter_btn = gr.Button("Apply Filters", variant="secondary")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### Recommended Products")
                        recommendations_df = gr.Dataframe(
                            label="Product Recommendations",
                            headers=["Name", "Category", "Price", "Rating", "ML Score", "Description"],
                            datatype=["str", "str", "str", "str", "str", "str"]
                        )
                        status_message = gr.Textbox(label="Status", interactive=False)
                
                # Connect ML recommendation function
                ml_rec_btn.click(
                    fn=app.get_ml_recommendations,
                    inputs=[user_id_input, num_recs],
                    outputs=[recommendations_df, status_message]
                )
                
                # Connect filter function
                def filter_products(cat, price, rating, uid):
                    user_id_val = int(uid) if uid else None
                    return app.suggest_products(cat, price, rating, user_id_val)
                
                filter_btn.click(
                    fn=filter_products,
                    inputs=[category_dropdown, max_price_slider, min_rating_slider, user_id_input],
                    outputs=[recommendations_df, status_message]
                )
            
            # Tab 2: Product Search
            with gr.TabItem("🔍 Search"):
                with gr.Row():
                    search_input = gr.Textbox(
                        label="Search Products",
                        placeholder="Enter product name or keyword...",
                        lines=1
                    )
                    search_btn = gr.Button("Search", variant="secondary")
                
                search_results = gr.Dataframe(
                    label="Search Results",
                    headers=["Name", "Category", "Price", "Rating", "Description"],
                    datatype=["str", "str", "str", "str", "str"]
                )
                
                search_btn.click(
                    fn=app.search_products,
                    inputs=search_input,
                    outputs=search_results
                )
                
                search_input.submit(
                    fn=app.search_products,
                    inputs=search_input,
                    outputs=search_results
                )
            
            # Tab 3: Product Details
            with gr.TabItem("📋 Product Details"):
                with gr.Row():
                    product_selector = gr.Dropdown(
                        choices=[p["name"] for p in app.products],
                        label="Select Product"
                    )
                    details_btn = gr.Button("Show Details", variant="primary")
                
                product_details = gr.Markdown("Select a product to view details.")
                
                def update_details(product_name):
                    return app.get_product_details(product_name)
                
                details_btn.click(
                    fn=update_details,
                    inputs=product_selector,
                    outputs=product_details
                )
            
            # Tab 4: Random Suggestion
            with gr.TabItem("🎲 Random Pick"):
                gr.Markdown("### Discover Something New!")
                
                with gr.Row():
                    random_btn = gr.Button("Get Random Suggestion", variant="primary", size="lg")
                
                random_suggestion = gr.Markdown("Click button to get a random product suggestion!")
                
                def get_random():
                    return app.get_random_suggestion()
                
                random_btn.click(
                    fn=get_random,
                    outputs=random_suggestion
                )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("💡 **Tip:** Enter your User ID in ML Recommendations for personalized suggestions powered by our Wide & Deep model!")
    
    return interface

# Launch app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860)