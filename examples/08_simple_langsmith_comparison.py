from langsmith import Client
import numpy as np
import os
from typing import Dict, Any, Optional
from datetime import datetime

def get_experiment_stats(project_id: str, experiment_id: str) -> Optional[Dict[str, Any]]:
    """
    Get statistics for a specific experiment in a LangSmith project.
    
    Args:
        project_id: The ID of the project
        experiment_id: The ID of the experiment (dataset ID)
    
    Returns:
        Dictionary of statistics or None if error
    """
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print("❌ LANGCHAIN_API_KEY is not set")
        return None

    client = Client(api_key=api_key)
    
    try:
        print(f"Fetching runs for project '{project_id}' and experiment '{experiment_id}'")
        
        # First get the project by ID to find its name
        try:
            project = client.read_project(project_id=project_id)
            project_name = project.name
            print(f"Found project: {project_name}")
        except Exception as e:
            # If we can't get the project by ID, try using the ID as the name
            print(f"Could not find project by ID, trying as name: {e}")
            project_name = project_id
        
        # Get runs for this experiment using project name
        runs = list(client.list_runs(
            project_name=project_name,
            execution_order=1,
            limit=1000
        ))
        
        if not runs:
            print(f"No runs found in project '{project_name}'")
            return None
        
        print(f"Found {len(runs)} runs, filtering by experiment ID")
        
        # Filter runs by experiment/dataset ID
        valid_runs = []
        for run in runs:
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
            
            if dataset_id == experiment_id:
                valid_runs.append(run)
        
        if not valid_runs:
            print(f"No runs found for experiment '{experiment_id}'")
            return None
        
        print(f"Processing {len(valid_runs)} valid runs")
        
        # Group runs by evaluator type
        evaluator_runs = {}
        for run in valid_runs:
            evaluator_name = run.name.lower() if hasattr(run, 'name') and run.name else "unknown"
            if evaluator_name not in evaluator_runs:
                evaluator_runs[evaluator_name] = []
            evaluator_runs[evaluator_name].append(run)
        
        # Collect basic statistics
        stats = {"Total Runs": len(valid_runs)}
        
        # Calculate latency stats
        latencies = []
        for run in valid_runs:
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
                except Exception:
                    pass
        
        if latencies:
            stats["Latency (P50)"] = np.percentile(latencies, 50)
            stats["Latency (P99)"] = np.percentile(latencies, 99)
        
        # Calculate token stats
        tokens = {"prompt": [], "completion": [], "total": []}
        for run in valid_runs:
            if run.prompt_tokens is not None:
                tokens["prompt"].append(int(run.prompt_tokens))
            if run.completion_tokens is not None:
                tokens["completion"].append(int(run.completion_tokens))
            if run.total_tokens is not None:
                tokens["total"].append(int(run.total_tokens))
        
        if tokens["prompt"]:
            stats["Avg Prompt Tokens"] = np.mean(tokens["prompt"])
        if tokens["completion"]:
            stats["Avg Completion Tokens"] = np.mean(tokens["completion"])
        if tokens["total"]:
            stats["Avg Total Tokens"] = np.mean(tokens["total"])
        
        # Evaluator name mapping
        metric_names = {
            "correctness": "Correctness",
            "groundedness": "Groundedness",
            "relevance": "Relevance",
            "coherence": "Coherence",
            "helpfulness": "Helpfulness",
            "conciseness": "Conciseness",
            "mrr": "MRR",
            "ndcg": "NDCG",
            "precision_at_k": "Precision@k",
            "recall_at_k": "Recall@k"
        }
        
        # Add metrics from evaluator runs
        for evaluator_name, runs in evaluator_runs.items():
            # Clean up evaluator name to match our mapping
            clean_name = evaluator_name.replace("_evaluator", "")
            
            # Get scores from all runs for this evaluator
            scores = []
            for run in runs:
                if hasattr(run, 'outputs') and isinstance(run.outputs, dict) and 'score' in run.outputs:
                    try:
                        score = float(run.outputs['score'])
                        scores.append(score)
                    except (ValueError, TypeError):
                        pass
            
            if scores:
                # Use mapped name if available, otherwise capitalize the evaluator name
                metric_name = metric_names.get(clean_name, clean_name.capitalize())
                stats[metric_name] = np.mean(scores)
        
        return stats

    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def compare_experiments(experiment1, experiment2):
    """
    Compare two experiments and display results.
    
    Args:
        experiment1: Tuple of (project_id, experiment_id, name)
        experiment2: Tuple of (project_id, experiment_id, name)
    """
    project1, exp1, name1 = experiment1
    project2, exp2, name2 = experiment2
    
    print(f"\n🔄 Fetching stats for {name1}...")
    stats1 = get_experiment_stats(project1, exp1)
    
    print(f"\n🔄 Fetching stats for {name2}...")
    stats2 = get_experiment_stats(project2, exp2)
    
    if not stats1 or not stats2:
        print("❌ Could not compare experiments due to missing stats")
        return
    
    print("\n📊 Comparison Results:")
    print("=" * 70)
    print(f"{'Metric':<25} {name1:>20} {name2:>20} {'Difference':>15}")
    print("-" * 70)
    
    # Get all unique metrics
    all_metrics = set(list(stats1.keys()) + list(stats2.keys()))
    
    for metric in sorted(all_metrics):
        val1 = stats1.get(metric, "N/A")
        val2 = stats2.get(metric, "N/A")
        
        # Calculate difference if both values are numeric
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = val2 - val1
            diff_str = f"{diff:>+.2f}" if isinstance(diff, float) else f"{diff:>+d}"
        else:
            diff_str = "N/A"
        
        # Format values
        val1_str = f"{val1:.2f}" if isinstance(val1, float) else str(val1)
        val2_str = f"{val2:.2f}" if isinstance(val2, float) else str(val2)
        
        print(f"{metric:<25} {val1_str:>20} {val2_str:>20} {diff_str:>15}")

if __name__ == "__main__":
    # Define experiments to compare
    arabic_experiment = (
        "d596020e-6a95-46db-86db-2f9b885b3547",  # project_id
        "8e2b4edd-7c69-4774-bde9-17b4a9ad6d2d",  # experiment_id
        "Arabic"                                  # name for display
    )
    
    english_experiment = (
        "d596020e-6a95-46db-86db-2f9b885b3547",  # project_id
        "42ae5806-27ed-43d2-a1ef-0cd2b3e3da62",  # experiment_id
        "English"                                 # name for display
    )
    
    # Run comparison
    compare_experiments(arabic_experiment, english_experiment) 

   