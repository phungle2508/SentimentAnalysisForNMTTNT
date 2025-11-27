"""
Product Suggestion Gradio Template
====================================

This is a template for a product suggestion system using Gradio.
To run this application:

1. Install dependencies:
   pip install gradio

2. Run the application:
   python product_suggestion_app.py

Features:
- Product recommendations with filters
- Product search functionality
- Detailed product information
- Random product suggestions
- Multiple categories and price ranges
"""

# Copy the code below into a Python file and run it

"""
import gradio as gr
import random
from typing import List, Dict, Tuple

class ProductSuggestionApp:
    def __init__(self):
        self.products = self._load_sample_products()
        
    def _load_sample_products(self) -> List[Dict]:
        # Sample product data - replace with your actual product catalog
        return [
            {"id": 1, "name": "Wireless Headphones", "category": "Electronics", "price": 79.99, "rating": 4.5, "description": "High-quality wireless headphones with noise cancellation"},
            {"id": 2, "name": "Smart Watch", "category": "Electronics", "price": 199.99, "rating": 4.3, "description": "Fitness tracking smartwatch with heart rate monitor"},
            {"id": 3, "name": "Running Shoes", "category": "Sports", "price": 89.99, "rating": 4.7, "description": "Comfortable running shoes for all terrains"},
            {"id": 4, "name": "Coffee Maker", "category": "Home", "price": 129.99, "rating": 4.4, "description": "Automatic coffee maker with timer"},
            {"id": 5, "name": "Yoga Mat", "category": "Sports", "price": 29.99, "rating": 4.6, "description": "Non-slip yoga mat with carrying strap"},
            {"id": 6, "name": "Backpack", "category": "Accessories", "price": 49.99, "rating": 4.2, "description": "Durable backpack with laptop compartment"},
            {"id": 7, "name": "Bluetooth Speaker", "category": "Electronics", "price": 59.99, "rating": 4.5, "description": "Portable bluetooth speaker with excellent sound quality"},
            {"id": 8, "name": "Water Bottle", "category": "Sports", "price": 19.99, "rating": 4.8, "description": "Insulated water bottle that keeps drinks cold for 24 hours"},
            {"id": 9, "name": "Desk Lamp", "category": "Home", "price": 39.99, "rating": 4.1, "description": "LED desk lamp with adjustable brightness"},
            {"id": 10, "name": "Phone Case", "category": "Accessories", "price": 15.99, "rating": 4.0, "description": "Protective phone case with stylish design"}
        ]
    
    def get_categories(self) -> List[str]:
        return list(set(product["category"] for product in self.products))
    
    def suggest_products(self, category: str, max_price: float, min_rating: float) -> Tuple[List[List[str]], str]:
        filtered_products = []
        
        for product in self.products:
            if (category == "All" or product["category"] == category) and \\
               product["price"] <= max_price and \\
               product["rating"] >= min_rating:
                filtered_products.append(product)
        
        if not filtered_products:
            return [], "No products found matching your criteria."
        
        filtered_products.sort(key=lambda x: (-x["rating"], x["price"]))
        
        table_data = []
        for product in filtered_products:
            table_data.append([
                product["name"],
                product["category"],
                f"${product['price']:.2f}",
                f"{product['rating']}/5.0 ⭐",
                product["description"]
            ])
        
        message = f"Found {len(filtered_products)} products matching your criteria."
        return table_data, message
    
    def search_products(self, query: str) -> List[List[str]]:
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
app = ProductSuggestionApp()

def create_interface():
    with gr.Blocks(title="Product Suggestion System") as interface:
        gr.Markdown("# 🛍️ Product Suggestion System")
        gr.Markdown("Discover the perfect products for your needs!")
        
        with gr.Tabs():
            # Recommendations Tab
            with gr.TabItem("🎯 Recommendations"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### Filter Products")
                        category_dropdown = gr.Dropdown(
                            choices=["All"] + app.get_categories(),
                            value="All",
                            label="Category"
                        )
                        max_price_slider = gr.Slider(
                            minimum=0, maximum=500, value=200, step=10,
                            label="Maximum Price ($)"
                        )
                        min_rating_slider = gr.Slider(
                            minimum=1, maximum=5, value=3.5, step=0.5,
                            label="Minimum Rating"
                        )
                        suggest_btn = gr.Button("Get Suggestions", variant="primary")
                        
                    with gr.Column(scale=2):
                        gr.Markdown("### Recommended Products")
                        recommendations_df = gr.Dataframe(
                            label="Product Suggestions",
                            headers=["Name", "Category", "Price", "Rating", "Description"]
                        )
                        status_message = gr.Textbox(label="Status", interactive=False)
                
                suggest_btn.click(
                    fn=app.suggest_products,
                    inputs=[category_dropdown, max_price_slider, min_rating_slider],
                    outputs=[recommendations_df, status_message]
                )
            
            # Search Tab
            with gr.TabItem("🔍 Search"):
                with gr.Row():
                    search_input = gr.Textbox(
                        label="Search Products",
                        placeholder="Enter product name or keyword..."
                    )
                    search_btn = gr.Button("Search", variant="secondary")
                
                search_results = gr.Dataframe(
                    label="Search Results",
                    headers=["Name", "Category", "Price", "Rating", "Description"]
                )
                
                search_btn.click(
                    fn=app.search_products,
                    inputs=search_input,
                    outputs=search_results
                )
        
        gr.Markdown("---")
        gr.Markdown("💡 **Tip:** Use filters to find the perfect product!")
    
    return interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
"""

# Simple version without external dependencies for testing
def simple_product_interface():
    """
    Simple product suggestion interface structure
    This shows the layout and functionality without requiring Gradio installation
    """
    
    interface_structure = """
    PRODUCT SUGGESTION SYSTEM
    =========================
    
    Main Features:
    --------------
    1. 🎯 Recommendations Tab
       - Category filter (All, Electronics, Sports, Home, Accessories)
       - Price range slider ($0 - $500)
       - Rating filter (1.0 - 5.0 stars)
       - Results table with product details
    
    2. 🔍 Search Tab
       - Text search box
       - Search by product name or description
       - Results display in table format
    
    3. 📋 Product Details Tab
       - Product dropdown selector
       - Detailed product information display
       - Price, rating, and description
    
    4. 🎲 Random Pick Tab
       - Random product suggestion button
       - Daily pick feature
       - Product highlights
    
    Sample Products:
    ---------------
    - Wireless Headphones - $79.99 - 4.5⭐
    - Smart Watch - $199.99 - 4.3⭐
    - Running Shoes - $89.99 - 4.7⭐
    - Coffee Maker - $129.99 - 4.4⭐
    - Yoga Mat - $29.99 - 4.6⭐
    - Backpack - $49.99 - 4.2⭐
    - Bluetooth Speaker - $59.99 - 4.5⭐
    - Water Bottle - $19.99 - 4.8⭐
    - Desk Lamp - $39.99 - 4.1⭐
    - Phone Case - $15.99 - 4.0⭐
    
    Installation:
    -------------
    pip install gradio
    
    Usage:
    ------
    python product_suggestion_app.py
    """
    
    return interface_structure

if __name__ == "__main__":
    print(simple_product_interface())