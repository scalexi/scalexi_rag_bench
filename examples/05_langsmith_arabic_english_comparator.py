from langsmith import Client
import numpy as np
import os
from datetime import datetime
from uuid import UUID
from typing import Dict, Any, Optional

def get_experiment_stats(project_name: str, experiment_name: str, max_runs: int = 1000) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print("❌ LANGCHAIN_API_KEY is not set")
        return None

    client = Client(api_key=api_key)
    
    try:
        print(f"Fetching project with ID: {project_name}")
        project = client.read_project(project_id=project_name, include_stats=True)
        print(f"Project name: {project.name}")

        print(f"Calling list_runs with project_name={project.name}")
        # Try to parse experiment_name as a UUID to determine if it's a dataset_id
        is_dataset_id = True
        try:
            UUID(experiment_name)
        except ValueError:
            is_dataset_id = False

        if is_dataset_id:
            print(f"Treating experiment_name={experiment_name} as dataset_id")
            runs = list(client.list_runs(
                project_name=project.name,
                execution_order=1,
                limit=max_runs
            ))
        else:
            print(f"Treating experiment_name={experiment_name} as experiment metadata")
            runs = list(client.list_runs(
                project_name=project.name,
                execution_order=1,
                limit=max_runs
            ))

        if not runs:
            print(f"No runs found in project '{project_name}'.")
            return None

        print(f"Fetched {len(runs)} runs")

        # Filter runs by experiment_name (dataset_id or metadata.experiment)
        valid_runs = []
        for i, run in enumerate(runs):
            try:
                metadata_experiment = run.extra.get("metadata", {}).get("experiment")
                dataset_id = None
                if "example" in run.inputs:
                    if hasattr(run.inputs["example"], "dataset_id"):
                        dataset_id = str(run.inputs["example"].dataset_id)
                    elif isinstance(run.inputs["example"], dict):
                        dataset_id = run.inputs["example"].get("dataset_id")
                    elif isinstance(run.inputs["example"], str) and "dataset_id=" in run.inputs["example"]:
                        for part in run.inputs["example"].split(","):
                            if "dataset_id=" in part:
                                dataset_id = part.split("=")[1].strip("')")
                                break

                if is_dataset_id and dataset_id == experiment_name:
                    valid_runs.append(run)
                elif not is_dataset_id and metadata_experiment == experiment_name:
                    valid_runs.append(run)
            except Exception as e:
                print(f"Run {i}: Error checking experiment metadata: {e}")

        if not valid_runs:
            print(f"No valid runs found for experiment '{experiment_name}' after filtering.")
            return None

        print(f"Processing {len(valid_runs)} valid runs for experiment '{experiment_name}'")

        # Group runs by evaluator type
        evaluator_runs = {}
        for run in valid_runs:
            evaluator_name = run.name.lower() if hasattr(run, 'name') and run.name else "unknown"
            if evaluator_name not in evaluator_runs:
                evaluator_runs[evaluator_name] = []
            evaluator_runs[evaluator_name].append(run)
        
        print(f"\nFound evaluator types: {', '.join(evaluator_runs.keys())}")

        latencies = []
        prompt_tokens = []
        completion_tokens = []
        total_tokens = []
        error_count = 0
        
        # Process runs for standard metrics
        for i, run in enumerate(valid_runs):
            try:
                if run.error:
                    error_count += 1
                
                if run.start_time and run.end_time:
                    try:
                        if isinstance(run.start_time, str) and isinstance(run.end_time, str):
                            start = datetime.fromisoformat(run.start_time.replace('Z', '+00:00'))
                            end = datetime.fromisoformat(run.end_time.replace('Z', '+00:00'))
                        else:
                            start = run.start_time
                            end = run.end_time
                        latency = (end - start).total_seconds()
                        if latency >= 0:
                            latencies.append(latency)
                    except Exception as e:
                        print(f"Run {i}: Error calculating latency: {e}")
                
                try:
                    if run.prompt_tokens is not None:
                        prompt_tokens.append(int(run.prompt_tokens))
                    if run.completion_tokens is not None:
                        completion_tokens.append(int(run.completion_tokens))
                    if run.total_tokens is not None:
                        total_tokens.append(int(run.total_tokens))
                except (ValueError, TypeError) as e:
                    print(f"Run {i}: Error processing tokens: {e}")

            except Exception as e:
                print(f"Run {i}: General error: {e}")

        # Create stats dictionary with standard metrics
        stats = {
            "Total Runs": len(valid_runs),
            "Error Rate (%)": (error_count / len(valid_runs)) * 100 if valid_runs else 0,
        }
        
        if latencies:
            stats.update({
                "Latency (P50)": np.percentile(latencies, 50),
                "Latency (P99)": np.percentile(latencies, 99),
            })
        
        if prompt_tokens:
            stats.update({
                "Avg Prompt Tokens": np.mean(prompt_tokens),
                "Total Prompt Tokens": np.sum(prompt_tokens),
            })
        
        if completion_tokens:
            stats.update({
                "Avg Completion Tokens": np.mean(completion_tokens),
                "Total Completion Tokens": np.sum(completion_tokens),
            })
        
        if total_tokens:
            stats.update({
                "Avg Total Tokens": np.mean(total_tokens),
                "Total Tokens": np.sum(total_tokens),
            })

        # Map typical evaluator names to metric names
        evaluator_to_metric = {
            "correctness_evaluator": "Correctness",
            "groundedness_evaluator": "Groundedness",
            "relevance_evaluator": "Relevance",
            "coherence_evaluator": "Coherence",
            "helpfulness_evaluator": "Helpfulness",
            "completeness_evaluator": "Completeness",
            "conciseness_evaluator": "Conciseness",
            "repetitions_evaluator": "Repetitions",
            "correctness": "Correctness",
            "groundedness": "Groundedness",
            "relevance": "Relevance",
            "coherence": "Coherence",
            "helpfulness": "Helpfulness",
            "completeness": "Completeness",
            "conciseness": "Conciseness",
            "repetitions": "Repetitions",
            "mrr_evaluator": "MRR",
            "mrr": "MRR",
            "ndcg_evaluator": "NDCG",
            "ndcg": "NDCG",
            "precision": "Precision@k",
            "precision_evaluator": "Precision@k",
            "precision_at_k": "Precision@k",
            "precision_at_k_evaluator": "Precision@k",
            "recall": "Recall@k",
            "recall_evaluator": "Recall@k", 
            "recall_at_k": "Recall@k",
            "recall_at_k_evaluator": "Recall@k"
        }

        # Add metrics from evaluator runs
        for evaluator_name, runs in evaluator_runs.items():
            scores = []
            for run in runs:
                if hasattr(run, 'outputs') and isinstance(run.outputs, dict) and 'score' in run.outputs:
                    try:
                        score = float(run.outputs['score'])
                        scores.append(score)
                    except (ValueError, TypeError):
                        pass
            
            if scores:
                # Map evaluator name to a standardized metric name if possible
                metric_name = evaluator_to_metric.get(evaluator_name.lower(), evaluator_name.capitalize())
                stats[metric_name] = np.mean(scores)
                print(f"Added metric {metric_name} with average score {stats[metric_name]:.4f} from {len(scores)} evaluations")

        return stats

    except Exception as e:
        print(f"❌ Error fetching experiment stats: {e}")
        return None

def compare_experiments(
    project_id_1: str,
    experiment_id_1: str,
    project_id_2: str,
    experiment_id_2: str,
    max_runs: int = 1000
) -> None:
    """
    Compare statistics between two experiments from different projects.
    
    Args:
        project_id_1: ID of the first project
        experiment_id_1: ID of the first experiment
        project_id_2: ID of the second project
        experiment_id_2: ID of the second experiment
        max_runs: Maximum number of runs to analyze
    """
    print("\n🔄 Fetching stats for first experiment...")
    stats_1 = get_experiment_stats(project_id_1, experiment_id_1, max_runs)
    
    print("\n🔄 Fetching stats for second experiment...")
    stats_2 = get_experiment_stats(project_id_2, experiment_id_2, max_runs)
    
    if not stats_1 or not stats_2:
        print("❌ Could not compare experiments due to missing stats")
        return
    
    print("\n📊 Comparison Results:")
    print("=" * 60)
    print(f"{'Metric':<25} {'Experiment 1':>15} {'Experiment 2':>15} {'Difference':>15}")
    print("-" * 60)
    
    # Get all unique metrics
    all_metrics = set(list(stats_1.keys()) + list(stats_2.keys()))
    
    for metric in sorted(all_metrics):
        val1 = stats_1.get(metric, "N/A")
        val2 = stats_2.get(metric, "N/A")
        
        # Calculate difference if both values are numeric
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = val2 - val1
            diff_str = f"{diff:>+.2f}" if isinstance(diff, float) else f"{diff:>+d}"
        else:
            diff_str = "N/A"
        
        # Format values
        val1_str = f"{val1:.2f}" if isinstance(val1, float) else str(val1)
        val2_str = f"{val2:.2f}" if isinstance(val2, float) else str(val2)
        
        print(f"{metric:<25} {val1_str:>15} {val2_str:>15} {diff_str:>15}")

if __name__ == "__main__":
    # Example usage
    project_id_1 = "d596020e-6a95-46db-86db-2f9b885b3547"
    experiment_id_1 = "8e2b4edd-7c69-4774-bde9-17b4a9ad6d2d"
    project_id_2 = "d596020e-6a95-46db-86db-2f9b885b3547"
    experiment_id_2 = "42ae5806-27ed-43d2-a1ef-0cd2b3e3da62"
    
    compare_experiments(
        project_id_1,
        experiment_id_1,
        project_id_2,
        experiment_id_2
    )
