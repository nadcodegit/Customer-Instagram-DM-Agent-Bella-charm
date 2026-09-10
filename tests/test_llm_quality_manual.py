"""Occasional check of real LLM answer quality on the tricky cases found
during manual testing. NOT run by a plain `pytest` -- see the `live_llm`
marker in pyproject.toml. Run explicitly with:

    uv run pytest -m live_llm -v

Hits the real Groq API, so it needs GROQ_API_KEY set (skipped otherwise),
and it's slower and costs API calls -- that's why it's opt-in rather than
part of the normal test run.

These assertions are softer than test_llm_nodes.py's: a real model's
answer can shift a little between runs, or after Groq updates a model. A
failure here means "go take a look," not necessarily "something broke."
Good moments to run this: after changing a prompt in graph.py, after
Groq deprecates/changes the configured model, or every so often just to
make sure quality hasn't quietly drifted.
"""

import os

import pytest

from bella_charm_agent import graph
from bella_charm_agent.constants import CATEGORY_VARIANTS

pytestmark = [
    pytest.mark.live_llm,
    pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="requires a real GROQ_API_KEY"),
]


def test_variant_matcher_understands_an_indirect_description(make_state):
    state = make_state("cat charm please")
    charm_variants = CATEGORY_VARIANTS["Charm"]
    match = graph._match_step_answer(
        state,
        charm_variants,
        f"Which Charm would you like: {', '.join(charm_variants)}?",
    )
    assert match.matched_option == "Animal"


def test_payment_reconciliation_matcher_understands_wanting_to_browse_more(make_state):
    state = make_state("I want to see watches")
    match = graph._match_step_answer(
        state,
        ["Not paid yet - add more items", "Already paid"],
        "Before we continue -- have you already sent payment for your "
        "previous order, or would you like to add more items to it first?",
    )
    assert match.matched_option == "Not paid yet - add more items"


def test_payment_reconciliation_matcher_understands_a_plain_no(make_state):
    state = make_state("no")
    match = graph._match_step_answer(
        state,
        ["Not paid yet - add more items", "Already paid"],
        "Before we continue -- have you already sent payment for your "
        "previous order, or would you like to add more items to it first?",
    )
    assert match.matched_option == "Not paid yet - add more items"


def test_new_request_classifier_recognizes_a_price_question(make_state):
    result = graph.handle_new_request(make_state("how much is a charm?"))
    assert result["current_step"] == "awaiting_browse_offer"
    assert "£5" in result["draft_reply"]


def test_new_request_classifier_recognizes_store_hours(make_state):
    result = graph.handle_new_request(make_state("are you open right now?"))
    assert "9:00" in result["draft_reply"]


def test_new_request_classifier_recognizes_out_of_scope_requests(make_state):
    # "other" sends its fixed holding reply automatically now -- the
    # signal that it needs the owner's personal attention is
    # owner_followups, not needs_human/a review pause.
    result = graph.handle_new_request(make_state("can you make me a custom engraved necklace?"))
    assert result["needs_human"] is False
    assert result["owner_followups"] == ["can you make me a custom engraved necklace?"]


def test_add_to_cart_extracts_a_mentioned_quantity(make_state):
    state = make_state(
        "yes, 2 hearts please", pending_selection={"category": "Charm", "variant": "Heart"}
    )
    result = graph._handle_add_to_cart(state)
    assert result["cart"][0]["quantity"] == 2


def test_delivery_address_flags_a_destination_mismatch(make_state):
    state = make_state("Paris, France", delivery_destination="UK")
    result = graph._handle_delivery_address(state)
    # Re-prompted for clarification, not advanced to final confirmation.
    assert "current_step" not in result


def test_category_extractor_picks_up_a_named_variant_in_the_same_message(make_state):
    # Regression check: without an explicit list of valid variant strings
    # in the prompt, the model tended to leave `variant` empty even when
    # the customer clearly named one (e.g. "gold watch" only advanced to
    # "which watch?" instead of skipping straight to add-to-cart-confirm).
    result = graph._handle_category(make_state("gold watch"))
    assert result["current_step"] == "awaiting_add_to_cart_confirm"


def test_more_items_extractor_picks_up_a_named_category_and_variant(make_state):
    state = make_state("yes, heart charm please", cart=[])
    result = graph._handle_more_items(state)
    assert result["current_step"] == "awaiting_add_to_cart_confirm"
    assert result["pending_selection"] == {"category": "Charm", "variant": "Heart"}


def test_more_items_extractor_picks_up_a_named_category_and_quantity(make_state):
    # Regression check for a real manual-testing session: "add 2 charms to
    # my cart please" at the more-items step got misread as unrelated to
    # the yes/no question and fell through to the off-topic/price-inquiry
    # path, silently dropping the "2". Same fix as the named-variant case
    # above, but for quantity.
    state = make_state("add 2 charms to my cart please", cart=[])
    result = graph._handle_more_items(state)
    assert result["current_step"] == "awaiting_variant"
    assert result["pending_selection"] == {"category": "Charm", "quantity": 2}


def test_browse_offer_reuses_a_category_already_known_from_the_opening_message(make_state):
    # Regression check: a customer who opened with "how much is a charm?"
    # (category captured then) and later just says "yes" shouldn't be
    # asked "Charm, Bracelet, or Watch?" again.
    state = make_state("yes", pending_selection={"category": "Charm"})
    result = graph._handle_browse_offer(state)
    assert result["current_step"] == "awaiting_variant"


def test_add_to_cart_decline_with_a_named_alternative_skips_back_to_confirm(make_state):
    # "no, I want a zodiac one instead" -- declining Animal, but naming a
    # different variant in the same breath. Shouldn't have to restart the
    # browse-more loop and re-pick Charm from scratch.
    state = make_state(
        "no, I want a zodiac one instead", pending_selection={"category": "Charm", "variant": "Animal"}
    )
    result = graph._handle_add_to_cart(state)
    assert result["current_step"] == "awaiting_add_to_cart_confirm"
    assert result["pending_selection"]["variant"] == "Zodiac"


def test_variant_matcher_recognizes_the_newer_charm_subcategories(make_state):
    # Flag/Heart/Animal were already covered above; Zodiac, Birth Month,
    # and Letters were added later based on the real product line and
    # hadn't been checked against the live model yet.
    charm_variants = CATEGORY_VARIANTS["Charm"]
    question = f"Which Charm would you like: {', '.join(charm_variants)}?"
    cases = {
        "do you have a letter charm, like the letter M?": "Letters",
        "I want a birthday month one, I'm a July baby": "Birth Month",
        "got any star sign charms? I'm a leo": "Zodiac",
    }
    for message, expected in cases.items():
        state = make_state(message)
        match = graph._match_step_answer(state, charm_variants, question)
        assert match.matched_option == expected, f"{message!r} -> {match.matched_option!r}"


def test_new_request_classifier_recognizes_a_payment_proof_question(make_state):
    result = graph.handle_new_request(make_state("Thank you, want me to send the receipt here?"))
    assert result["draft_reply"] == graph.PAYMENT_PROOF_REPLY


def test_delivery_destination_matcher_does_not_force_a_match_for_in_person_pickup(make_state):
    # Regression: "I'll come and take it from you" isn't UK delivery or
    # International delivery -- it's neither, and used to get force-matched
    # to "UK" instead of correctly falling through to escalation.
    state = make_state("I will come and take it from you")
    match = graph._match_step_answer(
        state, ["UK", "International"], "Is this delivery within the UK, or international?"
    )
    assert match.matched_option is None
