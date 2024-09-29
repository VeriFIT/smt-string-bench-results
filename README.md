# SMT String Bench Results

## Running the experimental evaluation

We assume you have [VeriFIT/smt-bench](https://github.com/VeriFIT/smt-bench) set up and running on an evaluation server
where you run the experimental evaluation (according to its instructions).

1. Set up the Python virtual environment:
    ```sh
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

2. Get new tasks from the server with experimental results (change host, port, etc. if running from a different server):
    ```sh
    ./get_tasks_and_generate_csv.sh
    ```

3. Process the results (choose one):
  - Run the Jupyter evaluation notebook `eval.ipynb`:
    - Set the correct tools and benchmarks to evaluate (set the version of `NOODLER`).
    - Run the first 4 cells to load the benchmarks and other cells based on your need.
  - Only prepare the results for manual processing:
      ```sh
      ./pyco_proc.py [options] <requested_tasks_file_with_results.tasks>
      ```
      Store the processed results and evaluate them manually.

4. Exit the Python virtual environment:
    ```sh
    deactivate
    ```
