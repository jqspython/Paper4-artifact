.PHONY: reproduce-synthetic monitor-matrix regenerate-manuscript

reproduce-synthetic:
	python -B -X utf8 -m paper4_artifact_cli

monitor-matrix:
	python -B -X utf8 -m paper4_monitor_matrix --output results/engineering-additions/e-b-monitor-summary.csv

regenerate-manuscript:
	test -n "$(MANUSCRIPT)"
	python -B -X utf8 -m paper4_public_reproduce --input-dir results --output-dir work/reproduced --manuscript "$(MANUSCRIPT)"
