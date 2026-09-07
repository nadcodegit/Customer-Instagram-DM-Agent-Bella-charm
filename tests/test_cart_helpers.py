from bella_charm_agent.state import cart_line, cart_total


def test_cart_line_single_quantity_has_no_multiplier_suffix():
    item = {"category": "Bracelet", "variant": "Gold", "price": 15, "quantity": 1, "tag": None}
    assert cart_line(item) == "- Gold Bracelet (£15)"


def test_cart_line_multi_quantity_shows_suffix_and_multiplies_price():
    item = {"category": "Charm", "variant": "Heart", "price": 5, "quantity": 2, "tag": None}
    assert cart_line(item) == "- Heart Charm x2 (£10)"


def test_cart_total_sums_price_times_quantity_across_items():
    cart = [
        {"category": "Charm", "variant": "Heart", "price": 5, "quantity": 2, "tag": None},
        {"category": "Bracelet", "variant": "Gold", "price": 15, "quantity": 1, "tag": None},
    ]
    assert cart_total(cart) == 25
