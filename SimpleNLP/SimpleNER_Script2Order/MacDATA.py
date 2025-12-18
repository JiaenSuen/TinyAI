
MENU = {
    "Big Mac": 5.99,
    "Quarter Pounder with Cheese": 6.29,
    "Double Quarter Pounder with Cheese": 7.99,
    "Cheeseburger": 2.49,
    "Double Cheeseburger": 3.79,
    "Hamburger": 2.19,
    "McDouble": 3.29,
    "McChicken": 2.99,
    "Spicy McChicken": 3.19,
    "McCrispy": 5.49,
    "Spicy McCrispy": 5.69,
    "Deluxe McCrispy": 6.29,
    "Spicy Deluxe McCrispy": 6.49,
    "Filet-O-Fish": 5.29,
    "4 Piece Chicken McNuggets": 3.29,
    "6 Piece Chicken McNuggets": 4.29,
    "10 Piece Chicken McNuggets": 6.49,
    "20 Piece Chicken McNuggets": 10.99,
    "World Famous Fries Small": 2.79,
    "World Famous Fries Medium": 3.79,
    "World Famous Fries Large": 4.49,
    "Apple Slices": 1.49,
    "Side Salad": 3.29,
    "Soft Drink Small": 1.99,
    "Soft Drink Medium": 2.49,
    "Soft Drink Large": 2.99,
    "Bottled Water": 1.99,
    "Orange Juice": 2.89,
    "Apple Juice": 1.99,
    "Premium Roast Coffee Small": 1.49,
    "Premium Roast Coffee Medium": 1.99,
    "Premium Roast Coffee Large": 2.29,
    "Iced Coffee Small": 2.49,
    "Iced Coffee Medium": 2.99,
    "Iced Coffee Large": 3.49,
    "Vanilla Shake Small": 3.29,
    "Vanilla Shake Medium": 3.99,
    "Vanilla Shake Large": 4.59,
    "Chocolate Shake Small": 3.29,
    "Chocolate Shake Medium": 3.99,
    "Chocolate Shake Large": 4.59,
    "Strawberry Shake Small": 3.29,
    "Strawberry Shake Medium": 3.99,
    "Strawberry Shake Large": 4.59,
    "McFlurry with OREO": 4.49,
    "McFlurry with M&M's": 4.49,
    "Hot Fudge Sundae": 3.29,
    "Caramel Sundae": 3.29,
    "Apple Pie": 1.79,
    "Cookie": 1.29,
    "Happy Meal Hamburger": 4.99,
    "Happy Meal Cheeseburger": 5.49,
    "Happy Meal 4 Piece Nuggets": 5.99,
}




 
test_data_raw = [
    ("I'd like 2 Big Mac.",
     [("2", "QUANTITY"), ("Big Mac", "PRODUCT")]),

    ("Can I get 1 McChicken and 3 World Famous Fries Medium?",
     [("1", "QUANTITY"), ("McChicken", "PRODUCT"),
      ("3", "QUANTITY"), ("World Famous Fries Medium", "PRODUCT")]),

    ("Hello, I want a Cheeseburger, please.",
     [("Cheeseburger", "PRODUCT")]),

    ("Give me 4 10 Piece Chicken McNuggets.",
     [("4", "QUANTITY"), ("10 Piece Chicken McNuggets", "PRODUCT")]),

    ("One Big Mac and 2 Soft Drink Large.",
     [("Big Mac", "PRODUCT"),
      ("2", "QUANTITY"), ("Soft Drink Large", "PRODUCT")]),

    ("I'll have 3 Filet-O-Fish.",
     [("3", "QUANTITY"), ("Filet-O-Fish", "PRODUCT")]),

    ("Order: 5 Apple Pie.",
     [("5", "QUANTITY"), ("Apple Pie", "PRODUCT")]),

    ("Hi, 2 Double Cheeseburger.",
     [("2", "QUANTITY"), ("Double Cheeseburger", "PRODUCT")]),

    ("I need 1 Happy Meal Hamburger.",
     [("1", "QUANTITY"), ("Happy Meal Hamburger", "PRODUCT")]),

    ("Just 3 McCrispy.",
     [("3", "QUANTITY"), ("McCrispy", "PRODUCT")]),

    ("Hey, can you add 2 Quarter Pounder with Cheese.",
     [("2", "QUANTITY"), ("Quarter Pounder with Cheese", "PRODUCT")]),

    ("I’d love two Big Mac.",
     [("Big Mac", "PRODUCT")]),

    ("Please prepare 4 Vanilla Shake Small.",
     [("4", "QUANTITY"), ("Vanilla Shake Small", "PRODUCT")]),

    ("Quick order: a McFlurry with OREO.",
     [("McFlurry with OREO", "PRODUCT")]),

    ("For lunch, 1 Cookie.",
     [("1", "QUANTITY"), ("Cookie", "PRODUCT")]),
]