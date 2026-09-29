.PHONY: help install test clean run-multi run-collector run-all stop check-kafka create-topics

CONFIG ?= config/baselines.yaml

help:
	@echo "RRCF Detector - Multi-Model Anomaly Detection"
	@echo ""
	@echo "Usage:"
	@echo "  make install          Install dependencies"
	@echo "  make test            Run integration tests"
	@echo "  make check-kafka     Verify Kafka is running"
	@echo "  make create-topics   Create required Kafka topics"
	@echo ""
	@echo "  make run-collector   Start stream collector (Terminal 1)"
	@echo "  make run-multi       Start multi-model service (Terminal 2)"
	@echo "  make run-all         Start both in background"
	@echo "  make stop            Stop background services"
	@echo ""
	@echo "  make clean           Clean generated data files"
	@echo ""
	@echo "Configuration:"
	@echo "  CONFIG=$(CONFIG)"
	@echo "  Override: make run-multi CONFIG=config/production.yaml"

install:
	pip install -r requirements.txt
	@echo ""
	@echo "✓ Dependencies installed"

test:
	python3 scripts/test_integration.py

check-kafka:
	@echo "Checking Kafka connection..."
	@kafka-topics.sh --list --bootstrap-server localhost:9092 >/dev/null 2>&1 && \
		echo "✓ Kafka is running" || \
		(echo "✗ Kafka is not running on localhost:9092" && exit 1)

create-topics: check-kafka
	@echo "Creating Kafka topics..."
	@kafka-topics.sh --create --topic normalized-vectors \
		--bootstrap-server localhost:9092 \
		--partitions 4 --replication-factor 1 \
		--if-not-exists 2>/dev/null && \
		echo "✓ Created normalized-vectors" || \
		echo "  normalized-vectors already exists"
	@kafka-topics.sh --create --topic anomaly-scores \
		--bootstrap-server localhost:9092 \
		--partitions 4 --replication-factor 1 \
		--if-not-exists 2>/dev/null && \
		echo "✓ Created anomaly-scores" || \
		echo "  anomaly-scores already exists"

run-collector:
	python3 scripts/stream_collector.py --config $(CONFIG)

run-multi:
	python3 scripts/run_multi_model.py --config $(CONFIG)

run-all:
	@echo "Starting services in background..."
	python3 scripts/stream_collector.py --config $(CONFIG) > logs/collector.log 2>&1 & echo $$! > .collector.pid
	@sleep 2
	python3 scripts/run_multi_model.py --config $(CONFIG) > logs/multi_model.log 2>&1 & echo $$! > .multi_model.pid
	@echo "✓ Services started"
	@echo "  Stream collector PID: $$(cat .collector.pid)"
	@echo "  Multi-model PID: $$(cat .multi_model.pid)"
	@echo ""
	@echo "View logs:"
	@echo "  tail -f logs/collector.log"
	@echo "  tail -f logs/multi_model.log"
	@echo ""
	@echo "Stop services:"
	@echo "  make stop"

stop:
	@if [ -f .collector.pid ]; then \
		echo "Stopping stream collector (PID $$(cat .collector.pid))..."; \
		kill $$(cat .collector.pid) 2>/dev/null || true; \
		rm .collector.pid; \
	fi
	@if [ -f .multi_model.pid ]; then \
		echo "Stopping multi-model service (PID $$(cat .multi_model.pid))..."; \
		kill $$(cat .multi_model.pid) 2>/dev/null || true; \
		rm .multi_model.pid; \
	fi
	@echo "✓ Services stopped"

clean:
	rm -f .collector.pid .multi_model.pid
	rm -rf data/*
	rm -rf results/*
	@echo "✓ Cleaned generated files"

dirs:
	mkdir -p data logs results
	@echo "✓ Created directories"
