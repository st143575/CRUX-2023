cd ..

uv run crux score task1 ./tmp/example_eval_data_task1_pred.tab ./tmp/example_eval_data_task1_true.tab

uv run crux score task1 ./tmp/example_eval_data_task1_pred.tab ./tmp/example_eval_data_task1_true.tab --json ./tmp/example_eval_data_task1_report.json

uv run crux score task1 ./tmp/example_eval_data_task1_pred.tab ./tmp/example_eval_data_task1_true.tab --docs ./tmp/example_eval_data_task1_root_uids.txt --lenient -v