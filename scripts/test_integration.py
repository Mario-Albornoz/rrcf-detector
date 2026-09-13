#!/usr/bin/env python3
"""
Integration test for multi-model pipeline.

Tests the complete flow without requiring Kafka:
1. All 5 detectors can be imported and instantiated
2. All detectors can process dummy vectors
3. All detectors return correct output format
4. Multi-model runner can be initialized
5. Stream collector and evaluator can be imported

Run before full simulation to catch configuration/dependency issues.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test 1: Verify all imports work."""
    print("=" * 60)
    print("Test 1: Imports")
    print("=" * 60)
    
    try:
        # Baseline detectors
        from src.baselines import (
            BaseDetector,
            ZScoreDetector,
            IsolationForestDetector,
            HalfSpaceTreesDetector,
            OnlineIForestDetector,
            RRCFDetectorAdapter,
        )
        print("✓ All detector classes imported")
        
        # Worker and config
        from src.detection.generic_worker import GenericWorker
        from src.config import ServiceConfig
        from src.kafka.consumer import NormalizedVectorDto
        print("✓ Worker and config classes imported")
        
        # Multi-model runner (don't instantiate yet, just import)
        import scripts.run_multi_model
        print("✓ Multi-model runner imported")
        
        # Evaluation
        import scripts.evaluate_model
        print("✓ Evaluation script imported")
        
        print("\n✅ All imports successful!\n")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_detector_instantiation():
    """Test 2: Verify all detectors can be instantiated."""
    print("=" * 60)
    print("Test 2: Detector Instantiation")
    print("=" * 60)
    
    from src.baselines import (
        ZScoreDetector,
        IsolationForestDetector,
        HalfSpaceTreesDetector,
        OnlineIForestDetector,
        RRCFDetectorAdapter,
    )
    
    configs = {
        "rrcf": {
            "class": RRCFDetectorAdapter,
            "config": {"window_size": 100, "min_fill_threshold": 10}
        },
        "zscore": {
            "class": ZScoreDetector,
            "config": {"training_samples": 100}
        },
        "isoforest": {
            "class": IsolationForestDetector,
            "config": {"training_samples": 100, "n_estimators": 10, "contamination": 0.1}
        },
        "halfspace": {
            "class": HalfSpaceTreesDetector,
            "config": {"window_size": 100, "n_trees": 5, "height": 4, "min_fill_threshold": 10}
        },
        "onlineiforest": {
            "class": OnlineIForestDetector,
            "config": {"window_size": 128, "num_trees": 8, "max_leaf_samples": 16, "type": "adaptive", "min_fill_threshold": 10}
        },
    }
    
    detectors = {}
    
    for model_name, model_info in configs.items():
        try:
            detector = model_info["class"](model_info["config"])
            detectors[model_name] = detector
            print(f"✓ {model_name}: {detector.get_model_name()}")
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
    print(f"\n✅ All {len(detectors)} detectors instantiated!\n")
    return True, detectors


def test_data_ingestion(detectors):
    """Test 3: Verify all detectors can process dummy data."""
    print("=" * 60)
    print("Test 3: Data Ingestion")
    print("=" * 60)
    
    from src.kafka.consumer import NormalizedVectorDto
    
    # Create dummy vector
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
    
    results = {}
    
    # Warm up detectors (feed enough samples to get past cold start)
    print("Warming up detectors (feeding 50 samples)...")
    for i in range(50):
        for model_name, detector in detectors.items():
            detector.ingest_data(dummy_vector)
    print("✓ Warmup complete\n")
    
    # Now test scoring
    print("Testing scoring:")
    for model_name, detector in detectors.items():
        try:
            result = detector.ingest_data(dummy_vector)
            
            if result is None:
                print(f"⚠ {model_name}: Still in cold start (increase warmup)")
                continue
            
            # Verify output format
            required_keys = ["raw_score", "z_score", "stats"]
            missing_keys = [k for k in required_keys if k not in result]
            
            if missing_keys:
                print(f"✗ {model_name}: Missing keys: {missing_keys}")
                return False, None
            
            results[model_name] = result
            print(f"✓ {model_name}:")
            print(f"    raw_score: {result['raw_score']:.4f}")
            print(f"    z_score: {result['z_score']:.4f}")
            print(f"    stats_count: {result['stats']['count']}")
            
        except Exception as e:
            print(f"✗ {model_name} ingestion failed: {e}")
            import traceback
            traceback.print_exc()
            return False, None
    
    print(f"\n✅ All {len(results)} detectors processed data!\n")
    return True, results


def test_output_format(detectors, results):
    """Test 4: Verify output matches expected format for Kafka publishing."""
    print("=" * 60)
    print("Test 4: Output Format Validation")
    print("=" * 60)
    
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
    
    # Simulate what GenericWorker does
    for model_name, detector in detectors.items():
        result = results.get(model_name)
        if result is None:
            print(f"⚠ {model_name}: Skipping (no result)")
            continue
        
        # Build output message (same as GenericWorker)
        alert_level = detector.determine_alert_level(result["z_score"])
        
        score_output = {
            "exchange": dummy_vector.exchange,
            "instrument": dummy_vector.instrument,
            "instrument_class": dummy_vector.instrument_class,
            "timestamp": dummy_vector.timestamp.isoformat(),
            "model": detector.get_model_name(),  # ← KEY FIELD!
            "raw_score": result["raw_score"],
            "z_score": result["z_score"],
            "alert_level": alert_level,
            "stats_mean": result["stats"]["mean"],
            "stats_std": result["stats"]["std"],
            "stats_count": result["stats"]["count"],
        }
        
        # Verify all required fields present
        required_fields = [
            "exchange", "instrument", "instrument_class", "timestamp",
            "model", "raw_score", "z_score", "alert_level",
            "stats_mean", "stats_std", "stats_count"
        ]
        
        missing = [f for f in required_fields if f not in score_output]
        if missing:
            print(f"✗ {model_name}: Missing fields: {missing}")
            return False
        
        print(f"✓ {model_name}: Output format valid")
        print(f"    model tag: '{score_output['model']}'")
        print(f"    alert_level: '{score_output['alert_level']}'")
    
    print(f"\n✅ All outputs match expected format!\n")
    return True


def test_model_name_uniqueness(detectors):
    """Test 5: Verify all model names are unique."""
    print("=" * 60)
    print("Test 5: Model Name Uniqueness")
    print("=" * 60)
    
    model_names = [detector.get_model_name() for detector in detectors.values()]
    
    print(f"Model names: {model_names}")
    
    if len(model_names) != len(set(model_names)):
        print("✗ Duplicate model names detected!")
        return False
    
    expected_names = {"rrcf", "zscore", "isoforest", "halfspace", "onlineiforest"}
    actual_names = set(model_names)
    
    if expected_names != actual_names:
        print(f"✗ Model names don't match expected!")
        print(f"  Expected: {expected_names}")
        print(f"  Actual: {actual_names}")
        print(f"  Missing: {expected_names - actual_names}")
        print(f"  Extra: {actual_names - expected_names}")
        return False
    
    print(f"✓ All 5 model names unique and correct")
    print(f"\n✅ Model naming validated!\n")
    return True


def test_dependencies():
    """Test 6: Verify required packages are installed."""
    print("=" * 60)
    print("Test 6: Dependencies")
    print("=" * 60)
    
    required_packages = [
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("pyarrow", "pyarrow"),
        ("confluent_kafka", "confluent-kafka"),
        ("sklearn", "scikit-learn"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("river", "river"),
        ("rrcf", "rrcf"),
    ]
    
    missing = []
    
    for package_name, pip_name in required_packages:
        try:
            __import__(package_name)
            print(f"✓ {pip_name}")
        except ImportError:
            print(f"✗ {pip_name} not installed")
            missing.append(pip_name)
    
    if missing:
        print(f"\n✗ Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        return False
    
    print(f"\n✅ All dependencies installed!\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("Multi-Model Pipeline Integration Test")
    print("=" * 60)
    print()
    
    # Track results
    tests_passed = 0
    tests_total = 6
    
    # Test 1: Imports
    if test_imports():
        tests_passed += 1
    else:
        print("⚠ Stopping tests (import failure)")
        sys.exit(1)
    
    # Test 2: Instantiation
    success, detectors = test_detector_instantiation()
    if success:
        tests_passed += 1
    else:
        print("⚠ Stopping tests (instantiation failure)")
        sys.exit(1)
    
    # Test 3: Data ingestion
    success, results = test_data_ingestion(detectors)
    if success:
        tests_passed += 1
    else:
        print("⚠ Continuing with remaining tests...")
    
    # Test 4: Output format
    if test_output_format(detectors, results):
        tests_passed += 1
    
    # Test 5: Model name uniqueness
    if test_model_name_uniqueness(detectors):
        tests_passed += 1
    
    # Test 6: Dependencies
    if test_dependencies():
        tests_passed += 1
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Tests passed: {tests_passed}/{tests_total}")
    print()
    
    if tests_passed == tests_total:
        print("🎉 All tests passed!")
        print()
        print("Your pipeline is ready to run:")
        print("  1. python scripts/stream_collector.py --output-file ./data/test/scores.parquet")
        print("  2. python scripts/run_multi_model.py --config config/baselines.yaml")
        print("  3. python simulator.py --duration 60")
        print("  4. python scripts/evaluate_model.py --scores-file ./data/test/scores.parquet --output-dir ./results/test")
        print()
        return 0
    else:
        print(f"⚠ {tests_total - tests_passed} test(s) failed")
        print("Fix issues before running full simulation.")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
