"""Fixed business data for Bella Charm London.

Prices depend only on the top-level category, not the chosen variant, so
there's no per-item catalog to search -- just these fixed lookups. See
customer-conversation-scenarios.md for where these numbers come from.
"""

CATEGORY_PRICES = {
    "Bracelet": 15,
    "Charm": 5,
    "Watch": 25,
    "Accessories": 5,
}

CATEGORY_VARIANTS = {
    "Watch": ["Gold", "Silver", "Apple Watch"],
    "Bracelet": [
        "Gold", "Silver", "Copper", "Rose Gold", "Pink",
        "Black", "Purple", "Brown", "Red",
    ],
    "Charm": ["Flag", "Zodiac", "Heart", "Animal", "Birth Month", "Letters"],
    # Currently just the bracelet-connector "Key" -- a single-item category
    # for now, but kept as a proper category (not a special case) so more
    # accessories can be added later without any FSM changes.
    "Accessories": ["Key"],
}

# Plain "category.lower() + 's'" doesn't work for every category (Watch ->
# "watchs", Accessories -> "accessoriess"), so spell each one out.
CATEGORY_PLURALS = {
    "Bracelet": "bracelets",
    "Charm": "charms",
    "Watch": "watches",
    "Accessories": "accessories",
}

# For "{variant} {category}" phrasing (e.g. "Gold Bracelet", "Heart Charm")
# -- Bracelet/Charm/Watch are already singular, but the category itself is
# called "Accessories" (plural), which would otherwise read as "Key
# Accessories" instead of "Key Accessory".
CATEGORY_SINGULAR = {
    "Bracelet": "Bracelet",
    "Charm": "Charm",
    "Watch": "Watch",
    "Accessories": "Accessory",
}

DELIVERY_FEES = {
    "UK": 3,
    "International": 10,
}

STORE_HOURS = "Open daily, 9:00 - 18:00"
STORE_ADDRESS = "Inverness St, London NW1 7HJ, United Kingdom"

CURRENCY_SYMBOL = "£"
