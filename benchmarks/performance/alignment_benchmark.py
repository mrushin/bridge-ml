from bridge_ml import OntologyMatcher
import time

def run_benchmark(num_iterations: int = 100):
    matcher = OntologyMatcher()

    # Add your benchmark code here
    start_time = time.time()
    # Run your benchmarks
    end_time = time.time()

    print(f"Average time per iteration: {(end_time - start_time) / num_iterations:.3f}s")

if __name__ == '__main__':
    run_benchmark()
