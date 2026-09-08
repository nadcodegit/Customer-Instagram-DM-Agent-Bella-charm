"""Tests for the graph.py functions that call the LLM.

Each test replaces `graph._llm` (or `graph._match_step_answer` directly)
with a fixed canned response via the `fake_llm` fixture, so these test the
Python logic around the call -- routing, state updates, off-topic
handling -- not whether a real model correctly understands the message.
That's a separate, non-deterministic question (see the manual/diagnostic
testing done earlier in the project).
"""

from bella_charm_agent import graph


def test_handle_new_request_price_inquiry_with_named_category(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph, "_llm", fake_llm(graph.NewRequestClassification(intent="price_inquiry", category="Charm"))
    )
    result = graph.handle_new_request(make_state("how much is a charm"))
    assert result["current_step"] == "awaiting_browse_offer"
    assert "£5" in result["draft_reply"]
    # Remembered so a later "yes" doesn't have to re-ask the category.
    assert result["pending_selection"] == {"category": "Charm"}


def test_handle_new_request_price_inquiry_without_category_lists_all_three(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph, "_llm", fake_llm(graph.NewRequestClassification(intent="price_inquiry", category=None))
    )
    result = graph.handle_new_request(make_state("how much are your things"))
    for price in ("£15", "£5", "£25"):
        assert price in result["draft_reply"]


def test_handle_new_request_store_hours(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.NewRequestClassification(intent="store_hours")))
    result = graph.handle_new_request(make_state("are you open today?"))
    assert "9:00" in result["draft_reply"]
    assert result["needs_human"] is False


def test_handle_new_request_other_sends_holding_reply_and_logs_for_the_owner(make_state, monkeypatch, fake_llm):
    # "other" no longer gates on review -- its reply is fixed/pre-approved
    # same as everything else, so it sends automatically. The owner still
    # gets to see it, just via owner_followups instead of a live pause.
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.NewRequestClassification(intent="other")))
    result = graph.handle_new_request(make_state("can you make me a custom design?"))
    assert result["needs_human"] is False
    assert result["draft_reply"] == graph.OWNER_HANDOFF_REPLY
    assert result["owner_followups"] == ["can you make me a custom design?"]


def test_handle_new_request_greeting_gets_a_friendly_reply_not_a_handoff(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.NewRequestClassification(intent="greeting")))
    result = graph.handle_new_request(make_state("hello"))
    assert result["draft_reply"] == graph.GREETING_REPLY
    assert result["needs_human"] is False


def test_handle_new_request_payment_proof_gets_the_fixed_reply(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.NewRequestClassification(intent="payment_proof")))
    result = graph.handle_new_request(make_state("can I send you the payment screenshot here?"))
    assert result["draft_reply"] == graph.PAYMENT_PROOF_REPLY
    assert result["needs_human"] is False


def test_handle_step_answer_applies_the_matched_transition(make_state, monkeypatch):
    monkeypatch.setattr(
        graph,
        "_match_step_answer",
        lambda state, options, question: graph.StepAnswerMatch(matched_option="UK"),
    )
    state = make_state("UK please", current_step="awaiting_delivery_destination")
    result = graph.handle_step_answer(state)
    assert result["delivery_destination"] == "UK"
    assert result["current_step"] == "awaiting_delivery_address"


def test_handle_step_answer_off_topic_and_out_of_scope_flags_owner_without_advancing(
    make_state, monkeypatch, fake_llm
):
    monkeypatch.setattr(
        graph,
        "_match_step_answer",
        lambda state, options, question: graph.StepAnswerMatch(matched_option=None),
    )
    # Off-topic messages get reclassified (see _handle_off_topic_message) to
    # check whether they're actually a valid Scenario 1/2 request before
    # escalating -- here they genuinely aren't, so it escalates.
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.NewRequestClassification(intent="other")))
    state = make_state("do you insure shipments?", current_step="awaiting_delivery_destination")
    result = graph.handle_step_answer(state)
    assert result["owner_followups"] == ["do you insure shipments?"]
    # Still waiting on the same question -- current_step isn't advanced.
    assert "current_step" not in result


def test_handle_step_answer_off_topic_but_actually_a_price_question_gets_answered_inline(
    make_state, monkeypatch, fake_llm
):
    monkeypatch.setattr(
        graph,
        "_match_step_answer",
        lambda state, options, question: graph.StepAnswerMatch(matched_option=None),
    )
    # The customer didn't answer "want to see options?" -- they asked about
    # a different product instead. That's still answerable, so it should
    # get answered, then the original question re-asked, without escalating
    # to the owner or losing the in-progress flow.
    monkeypatch.setattr(
        graph, "_llm", fake_llm(graph.NewRequestClassification(intent="price_inquiry", category="Bracelet"))
    )
    state = make_state("how much are your bracelets?", current_step="awaiting_delivery_destination")
    result = graph.handle_step_answer(state)
    assert "£15" in result["draft_reply"]
    assert "current_step" not in result
    assert "owner_followups" not in result


def test_handle_step_answer_off_topic_greeting_gets_a_friendly_reply_and_resumes(
    make_state, monkeypatch, fake_llm
):
    monkeypatch.setattr(
        graph,
        "_match_step_answer",
        lambda state, options, question: graph.StepAnswerMatch(matched_option=None),
    )
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.NewRequestClassification(intent="greeting")))
    state = make_state("hey!", current_step="awaiting_delivery_destination")
    result = graph.handle_step_answer(state)
    assert graph.GREETING_REPLY in result["draft_reply"]
    assert "current_step" not in result
    assert "owner_followups" not in result


def test_handle_add_to_cart_extracts_quantity_and_adds_it_once(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph, "_llm", fake_llm(graph.AddToCartAnswer(matched=True, add_to_cart=True, quantity=2))
    )
    state = make_state(
        "yes, 2 please",
        pending_selection={"category": "Charm", "variant": "Heart"},
    )
    result = graph._handle_add_to_cart(state)
    assert len(result["cart"]) == 1
    assert result["cart"][0]["quantity"] == 2
    assert "x2" in result["draft_reply"]
    assert result["current_step"] == "awaiting_more_items"


def test_handle_add_to_cart_declined_leaves_cart_empty(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.AddToCartAnswer(matched=True, add_to_cart=False)))
    state = make_state("no thanks", pending_selection={"category": "Charm", "variant": "Heart"})
    result = graph._handle_add_to_cart(state)
    assert result["cart"] == []


def test_handle_add_to_cart_declined_with_named_variant_jumps_to_new_confirm(
    make_state, monkeypatch, fake_llm
):
    # "no, I want a zodiac charm instead" -- declining Animal, but naming a
    # different variant in the same breath. Shouldn't have to say "yes/no"
    # to browse more and re-pick a category from scratch.
    monkeypatch.setattr(
        graph,
        "_llm",
        fake_llm(
            graph.AddToCartAnswer(matched=True, add_to_cart=False, alternative_variant="Zodiac")
        ),
    )
    state = make_state(
        "no, I want a zodiac charm instead", pending_selection={"category": "Charm", "variant": "Animal"}
    )
    result = graph._handle_add_to_cart(state)
    assert result["current_step"] == "awaiting_add_to_cart_confirm"
    assert result["pending_selection"] == {"category": "Charm", "variant": "Zodiac"}
    # The declined item was never added.
    assert "cart" not in result


def test_handle_add_to_cart_declined_with_named_category_and_variant(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph,
        "_llm",
        fake_llm(
            graph.AddToCartAnswer(
                matched=True, add_to_cart=False, alternative_category="Watch", alternative_variant="Gold"
            )
        ),
    )
    state = make_state(
        "no, I'd rather have a gold watch", pending_selection={"category": "Charm", "variant": "Animal"}
    )
    result = graph._handle_add_to_cart(state)
    assert result["current_step"] == "awaiting_add_to_cart_confirm"
    assert result["pending_selection"] == {"category": "Watch", "variant": "Gold"}


def test_handle_add_to_cart_declined_with_invalid_alternative_variant_falls_back_to_variant_step(
    make_state, monkeypatch, fake_llm
):
    monkeypatch.setattr(
        graph,
        "_llm",
        # "Rainbow" isn't a real Charm variant -- guards against advancing
        # on a hallucinated or mismatched combination.
        fake_llm(
            graph.AddToCartAnswer(matched=True, add_to_cart=False, alternative_variant="Rainbow")
        ),
    )
    state = make_state(
        "no, something else", pending_selection={"category": "Charm", "variant": "Animal"}
    )
    result = graph._handle_add_to_cart(state)
    assert result["current_step"] == "awaiting_variant"


def test_handle_add_to_cart_off_topic_flags_owner_and_reasks(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph,
        "_llm",
        fake_llm(
            {
                graph.AddToCartAnswer: graph.AddToCartAnswer(matched=False),
                graph.NewRequestClassification: graph.NewRequestClassification(intent="other"),
            }
        ),
    )
    state = make_state(
        "what's your return policy?",
        pending_selection={"category": "Charm", "variant": "Heart"},
    )
    result = graph._handle_add_to_cart(state)
    assert result["owner_followups"] == ["what's your return policy?"]
    assert "cart" not in result


def test_handle_delivery_address_consistent_moves_to_final_confirm(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph,
        "_llm",
        fake_llm(
            graph.AddressExtraction(
                contains_address=True, address="42 Baker St, London", consistent_with_destination=True
            )
        ),
    )
    state = make_state("42 Baker St, London", delivery_destination="UK")
    result = graph._handle_delivery_address(state)
    assert result["current_step"] == "awaiting_final_confirm"
    assert result["delivery_address"] == "42 Baker St, London"


def test_handle_delivery_address_inconsistent_with_destination_asks_to_clarify(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph,
        "_llm",
        fake_llm(
            graph.AddressExtraction(
                contains_address=True, address="Paris, France", consistent_with_destination=False
            )
        ),
    )
    state = make_state("Paris, France", delivery_destination="UK")
    result = graph._handle_delivery_address(state)
    assert "current_step" not in result
    assert "Paris, France" in result["draft_reply"]


def test_handle_step_answer_dispatches_add_to_cart_step_to_its_own_handler(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph, "_llm", fake_llm(graph.AddToCartAnswer(matched=True, add_to_cart=True, quantity=1))
    )
    state = make_state(
        "yes",
        current_step="awaiting_add_to_cart_confirm",
        pending_selection={"category": "Bracelet", "variant": "Gold"},
    )
    result = graph.handle_step_answer(state)
    assert result["current_step"] == "awaiting_more_items"
    assert len(result["cart"]) == 1


def test_handle_step_answer_dispatches_delivery_address_step_to_its_own_handler(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph,
        "_llm",
        fake_llm(
            graph.AddressExtraction(contains_address=True, address="1 High St", consistent_with_destination=True)
        ),
    )
    state = make_state("1 High St", current_step="awaiting_delivery_address", delivery_destination="UK")
    result = graph.handle_step_answer(state)
    assert result["current_step"] == "awaiting_final_confirm"


def test_handle_more_items_no_moves_to_delivery(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph, "_llm", fake_llm(graph.MoreItemsAnswer(matched=True, wants_more=False))
    )
    state = make_state(
        "no that's all",
        cart=[{"category": "Bracelet", "variant": "Copper", "price": 15, "quantity": 1, "tag": None}],
    )
    result = graph._handle_more_items(state)
    assert result["current_step"] == "awaiting_delivery_destination"


def test_handle_more_items_yes_without_category_asks_category(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph, "_llm", fake_llm(graph.MoreItemsAnswer(matched=True, wants_more=True, category=None))
    )
    result = graph._handle_more_items(make_state("yeah sure"))
    assert result["current_step"] == "awaiting_category"


def test_handle_more_items_yes_with_named_category_skips_straight_to_variant(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph,
        "_llm",
        fake_llm(graph.MoreItemsAnswer(matched=True, wants_more=True, category="Charm")),
    )
    # The customer volunteered a category while answering -- shouldn't have
    # to repeat it when asked "which category" a moment later.
    result = graph._handle_more_items(make_state("do you also have heart charm?"))
    assert result["current_step"] == "awaiting_variant"
    assert result["pending_selection"] == {"category": "Charm"}


def test_handle_more_items_yes_with_category_and_variant_skips_to_add_to_cart_confirm(
    make_state, monkeypatch, fake_llm
):
    monkeypatch.setattr(
        graph,
        "_llm",
        fake_llm(graph.MoreItemsAnswer(matched=True, wants_more=True, category="Charm", variant="Heart")),
    )
    # Named both the category AND the specific variant in one message --
    # should skip past "which category" AND "which variant" entirely.
    result = graph._handle_more_items(make_state("yes, heart charm please"))
    assert result["current_step"] == "awaiting_add_to_cart_confirm"
    assert result["pending_selection"] == {"category": "Charm", "variant": "Heart"}


def test_handle_more_items_with_invalid_variant_for_category_falls_back_to_variant_step(
    make_state, monkeypatch, fake_llm
):
    monkeypatch.setattr(
        graph,
        "_llm",
        # "Gold" isn't a real Charm variant -- guards against a hallucinated
        # or mismatched combination advancing further than it should.
        fake_llm(graph.MoreItemsAnswer(matched=True, wants_more=True, category="Charm", variant="Gold")),
    )
    result = graph._handle_more_items(make_state("yes, gold charm"))
    assert result["current_step"] == "awaiting_variant"


def test_handle_more_items_off_topic_reclassifies(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph,
        "_llm",
        fake_llm(
            {
                graph.MoreItemsAnswer: graph.MoreItemsAnswer(matched=False),
                graph.NewRequestClassification: graph.NewRequestClassification(intent="other"),
            }
        ),
    )
    result = graph._handle_more_items(make_state("actually my last order never arrived"))
    assert result["owner_followups"] == ["actually my last order never arrived"]


def test_handle_step_answer_dispatches_more_items_step_to_its_own_handler(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph,
        "_llm",
        fake_llm(graph.MoreItemsAnswer(matched=True, wants_more=True, category="Watch")),
    )
    state = make_state("show me watches too", current_step="awaiting_more_items")
    result = graph.handle_step_answer(state)
    assert result["current_step"] == "awaiting_variant"


def test_handle_category_names_category_only_asks_variant(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph, "_llm", fake_llm(graph.CategoryAnswer(matched=True, category="Bracelet"))
    )
    result = graph._handle_category(make_state("bracelet"))
    assert result["current_step"] == "awaiting_variant"
    assert result["pending_selection"] == {"category": "Bracelet"}


def test_handle_category_names_category_and_variant_skips_to_add_to_cart_confirm(
    make_state, monkeypatch, fake_llm
):
    monkeypatch.setattr(
        graph,
        "_llm",
        fake_llm(graph.CategoryAnswer(matched=True, category="Watch", variant="Gold")),
    )
    result = graph._handle_category(make_state("gold watch"))
    assert result["current_step"] == "awaiting_add_to_cart_confirm"
    assert result["pending_selection"] == {"category": "Watch", "variant": "Gold"}


def test_handle_category_with_invalid_variant_falls_back_to_variant_step(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph,
        "_llm",
        # "Heart" isn't a real Watch variant.
        fake_llm(graph.CategoryAnswer(matched=True, category="Watch", variant="Heart")),
    )
    result = graph._handle_category(make_state("watch, heart one"))
    assert result["current_step"] == "awaiting_variant"


def test_handle_category_off_topic_reclassifies(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph,
        "_llm",
        fake_llm(
            {
                graph.CategoryAnswer: graph.CategoryAnswer(matched=False),
                graph.NewRequestClassification: graph.NewRequestClassification(intent="store_hours"),
            }
        ),
    )
    result = graph._handle_category(make_state("what time do you close?"))
    assert "9:00" in result["draft_reply"]
    assert "current_step" not in result


def test_handle_step_answer_dispatches_category_step_to_its_own_handler(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph, "_llm", fake_llm(graph.CategoryAnswer(matched=True, category="Charm", variant="Zodiac"))
    )
    state = make_state("zodiac charm", current_step="awaiting_category")
    result = graph.handle_step_answer(state)
    assert result["current_step"] == "awaiting_add_to_cart_confirm"


def test_handle_browse_offer_yes_with_category_already_known_skips_category_question(
    make_state, monkeypatch, fake_llm
):
    # Category came from the original message ("how much is a charm?"),
    # stashed in pending_selection by handle_new_request. A plain "yes"
    # here shouldn't re-ask "Charm, Bracelet, or Watch?".
    monkeypatch.setattr(
        graph, "_llm", fake_llm(graph.BrowseOfferAnswer(matched=True, wants_to_browse=True))
    )
    state = make_state("yes", pending_selection={"category": "Charm"})
    result = graph._handle_browse_offer(state)
    assert result["current_step"] == "awaiting_variant"
    assert result["pending_selection"] == {"category": "Charm"}


def test_handle_browse_offer_yes_naming_category_and_variant_skips_to_add_to_cart_confirm(
    make_state, monkeypatch, fake_llm
):
    # No category known yet, but the customer names both in this reply.
    monkeypatch.setattr(
        graph,
        "_llm",
        fake_llm(graph.BrowseOfferAnswer(matched=True, wants_to_browse=True, category="Watch", variant="Gold")),
    )
    result = graph._handle_browse_offer(make_state("yes, gold watch"))
    assert result["current_step"] == "awaiting_add_to_cart_confirm"
    assert result["pending_selection"] == {"category": "Watch", "variant": "Gold"}


def test_handle_browse_offer_yes_with_nothing_known_asks_category(make_state, monkeypatch, fake_llm):
    # A general price question ("how much are your things?") leaves no
    # category in pending_selection -- falls back to the full question.
    monkeypatch.setattr(
        graph, "_llm", fake_llm(graph.BrowseOfferAnswer(matched=True, wants_to_browse=True))
    )
    result = graph._handle_browse_offer(make_state("yes"))
    assert result["current_step"] == "awaiting_category"


def test_handle_browse_offer_no_ends_the_turn(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph, "_llm", fake_llm(graph.BrowseOfferAnswer(matched=True, wants_to_browse=False))
    )
    result = graph._handle_browse_offer(make_state("no thanks", pending_selection={"category": "Charm"}))
    assert result["current_step"] == "start"


def test_handle_step_answer_dispatches_browse_offer_step_to_its_own_handler(make_state, monkeypatch, fake_llm):
    monkeypatch.setattr(
        graph, "_llm", fake_llm(graph.BrowseOfferAnswer(matched=True, wants_to_browse=True))
    )
    state = make_state("yes", current_step="awaiting_browse_offer", pending_selection={"category": "Watch"})
    result = graph.handle_step_answer(state)
    assert result["current_step"] == "awaiting_variant"


def test_handle_post_order_start_answers_an_answerable_message_before_asking_reconciliation(
    make_state, monkeypatch, fake_llm
):
    # Regression: this used to ignore whatever the customer said and
    # always ask the reconciliation question outright.
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.NewRequestClassification(intent="greeting")))
    result = graph.handle_post_order_start(make_state("thank you!"))
    assert graph.GREETING_REPLY in result["draft_reply"]
    assert result["current_step"] == "awaiting_payment_reconciliation"


def test_handle_post_order_start_answers_a_payment_proof_question(make_state, monkeypatch, fake_llm):
    # The exact reported case: "want me to send the receipt here?" right
    # after an order was confirmed.
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.NewRequestClassification(intent="payment_proof")))
    result = graph.handle_post_order_start(make_state("Thank you, want me to send the receipt here?"))
    assert graph.PAYMENT_PROOF_REPLY in result["draft_reply"]
    assert result["current_step"] == "awaiting_payment_reconciliation"


def test_handle_post_order_start_escalates_an_unanswerable_message_then_still_asks_reconciliation(
    make_state, monkeypatch, fake_llm
):
    monkeypatch.setattr(graph, "_llm", fake_llm(graph.NewRequestClassification(intent="other")))
    result = graph.handle_post_order_start(make_state("want me to send the receipt here?"))
    assert result["owner_followups"] == ["want me to send the receipt here?"]
    assert result["current_step"] == "awaiting_payment_reconciliation"
