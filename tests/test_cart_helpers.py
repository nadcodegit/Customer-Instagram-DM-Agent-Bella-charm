from bella_charm_agent.state import cart_line, cart_total


def test_cart_line_single_quantity_has_no_multiplier_suffix():
    item = {"category": "Bracelet", "variant": "Gold", "price": 15, "quantity": 1, "tag": None}
    assert cart_line(item) == "- Gold Bracelet (£15)"


def test_cart_line_multi_quantity_shows_suffix_and_multiplies_price():
    item = {"category": "Charm", "variant": "Heart", "price": 5, "quantity": 2, "tag": None}
    assert cart_line(item) == "- Heart Charm x2 (£10)"


def test_cart_line_shows_a_design_detail_when_present():
    # tag carries a customer-named design/color/pattern for Charm items
    # (see AddToCartAnswer.detail in graph.py) -- it has to show up here,
    # since this line is what the owner reads to actually fulfill the
    # order, not just an internal field.
    item = {"category": "Charm", "variant": "Animal", "price": 5, "quantity": 1, "tag": "a dog"}
    assert cart_line(item) == "- Animal Charm (a dog) (£5)"


def test_cart_line_detail_and_multiplier_suffix_combine():
    item = {"category": "Charm", "variant": "Claddagh", "price": 5, "quantity": 2, "tag": "gold"}
    assert cart_line(item) == "- Claddagh Charm (gold) x2 (£10)"


def test_cart_total_sums_price_times_quantity_across_items():
    cart = [
        {"category": "Charm", "variant": "Heart", "price": 5, "quantity": 2, "tag": None},
        {"category": "Bracelet", "variant": "Gold", "price": 15, "quantity": 1, "tag": None},
    ]
    assert cart_total(cart) == 25
