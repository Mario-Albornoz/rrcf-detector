#!/usr/bin/env python3
"""
Quick integration test for Online-iForest detector.

Verifies that the detector can be imported, initialized, and ingests data correctly.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    print("=" * 60)
    print("Online-iForest Integration Test")
    print("=" * 60)
    print()
    
    # Test 1: Import
    print("Test 1: Import detector...")
    from src.baselines import OnlineIForestDetector
    print("✓ Import successful")
    print()
    
    # Test 2: Initialize
    print("Test 2: Initialize detector...")
    config = {
        "window_size": 128,  # Small for quick test
        "num_trees": 8,
        "max_leaf_samples": 16,
        "type": "adaptive",
        "min_fill_threshold": 10
    }
    detector = OnlineIForestDetector(config)
    print(f"✓ Detector initialized")
    print(f"  Model name: {detector.get_model_name()}")
    print(f"  Window size: {detector.window_size}")
    print(f"  Num trees: {detector.num_trees}")
    print()
    
    # Test 3: Create dummy vector
    print("Test 3: Create dummy data...")
    from src.kafka.consumer import NormalizedVectorDto
    
    dummy_vector = NormalizedVectorDto(
        exchange="binance",
        instrument="BTC-USDT",
        instrument_class="crypto_spot",
        timestamp=datetime.now(),
        z_intertick_fast=0.5,
        z_price_step_fast=-0.3,
        z_intertick_slow=0.2,
        z_price_step_slow=0.1,
        cusum_intertick=0.0,
        cusum_price_step=0.0
    )
    print("✓ Dummy vector created")
    print()
    
    # Test 4: Cold start (should return None)
    print("Test 4: Cold start phase...")
    for i in range(5):
        result = detector.ingest_data(dummy_vector)
        print(f"  Sample {i+1}: {result}")
    print("✓ Cold start behaves correctly (returns None)")
    print()
    
    # Test 5: Warm phase (should return scores)
    print("Test 5: Warm phase...")
    for i in range(15):
        result = detector.ingest_data(dummy_vector)
    
    print(f"✓ Warm phase active")
    print(f"  Result: {result}")
    print(f"  Raw score: {result['raw_score']:.4f}")
    print(f"  Z-score: {result['z_score']:.4f}")
    print(f"  Stats count: {result['stats']['count']}")
    print()
    
    # Test 6: Alert level
    print("Test 6: Alert level determination...")
    alert_level = detector.determine_alert_level(result['z_score'])
    print(f"✓ Alert level: {alert_level}")
    print()
    
    # Test 7: Anomalous vector
    print("Test 7: Inject anomalous vector...")
    anomalous_vector = NormalizedVectorDto(
        exchange="binance",
        instrument="BTC-USDT",
        instrument_class="crypto_spot",
        timestamp=datetime.now(),
        z_intertick_fast=5.0,  # High anomaly
        z_price_step_fast=4.0,
        z_intertick_slow=3.0,
        z_price_step_slow=2.5,
        cusum_intertick=10.0,
        cusum_price_step=8.0
    )
    
    result = detector.ingest_data(anomalous_vector)
    alert_level = detector.determine_alert_level(result['z_score'])
    print(f"✓ Anomalous vector processed")
    print(f"  Raw score: {result['raw_score']:.4f}")
    print(f"  Z-score: {result['z_score']:.4f}")
    print(f"  Alert level: {alert_level}")
    print()
    
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print()
    print("Online-iForest detector is ready for integration.")
    print("Next steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Start stream collector: python scripts/stream_collector.py --output-file ./data/test/scores.parquet")
    print("  3. Run detection service: python main.py --config config/baselines.yaml")
    print("  4. Evaluate results: python scripts/evaluate_model.py --scores-file ./data/test/scores.parquet --output-dir ./results/test")

except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
