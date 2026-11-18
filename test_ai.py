from consumable_engine import ConsumablePersonalizationEngine

engine = ConsumablePersonalizationEngine(
    "consumable_predictor.pkl"
)

result = engine.resolve_expiration(
    category="생필품",
    subcategory="욕실용품",
    total_amount=750,
    unit="ml"
)

print(result)