from langsmith import Client
import numpy as np
import os
from datetime import datetime
from uuid import UUID

def get_experiment_stats(project_name: str, experiment_name: str, max_runs: int = 1000):
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print("❌ LANGCHAIN_API_KEY is not set")
        return

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
            return

        print(f"Fetched {len(runs)} runs")

        # Filter runs by experiment_name (dataset_id or metadata.experiment)
        valid_runs = []
        for i, run in enumerate(runs):
            try:
                metadata_experiment = run.extra.get("metadata", {}).get("experiment")
                dataset_id = None
                # Handle different ways dataset_id might be stored
                if "example" in run.inputs:
                    if hasattr(run.inputs["example"], "dataset_id"):
                        dataset_id = str(run.inputs["example"].dataset_id)
                    elif isinstance(run.inputs["example"], dict):
                        dataset_id = run.inputs["example"].get("dataset_id")
                    elif isinstance(run.inputs["example"], str) and "dataset_id=" in run.inputs["example"]:
                        # Parse dataset_id from string representation
                        for part in run.inputs["example"].split(","):
                            if "dataset_id=" in part:
                                dataset_id = part.split("=")[1].strip("')")
                                break

                print(f"Run {i}: metadata.experiment={metadata_experiment}, dataset_id={dataset_id}")

                # Match experiment
                if is_dataset_id and dataset_id == experiment_name:
                    valid_runs.append(run)
                elif not is_dataset_id and metadata_experiment == experiment_name:
                    valid_runs.append(run)
                else:
                    print(f"Run {i}: Skipped (does not match experiment '{experiment_name}')")
            except Exception as e:
                print(f"Run {i}: Error checking experiment metadata: {e}")

        if not valid_runs:
            print(f"No valid runs found for experiment '{experiment_name}' after filtering.")
            return

        print(f"Processing {len(valid_runs)} valid runs for experiment '{experiment_name}'")

        latencies = []
        prompt_tokens = []
        completion_tokens = []
        total_tokens = []
        error_count = 0

        for i, run in enumerate(valid_runs):
            try:
                if run.error:
                    error_count += 1
                
                # Process latency
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
                        else:
                            print(f"Run {i}: Skipping negative latency")
                    except Exception as e:
                        print(f"Run {i}: Error calculating latency: {e}")
                
                # Process tokens
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

        # Calculate stats
        stats = {
            "Total Runs": len(valid_runs),
            "Error Rate (%)": (error_count / len(valid_runs)) * 100 if valid_runs else 0,
        }
        
        if latencies:
            try:
                stats.update({
                    "Latency (P50)": np.percentile(latencies, 50),
                    "Latency (P99)": np.percentile(latencies, 99),
                })
            except Exception as e:
                print(f"Error computing latency stats: {e}")
        
        if prompt_tokens:
            try:
                stats.update({
                    "Avg Prompt Tokens": np.mean(prompt_tokens),
                    "Total Prompt Tokens": np.sum(prompt_tokens),
                })
            except Exception as e:
                print(f"Error computing prompt token stats: {e}")
        
        if completion_tokens:
            try:
                stats.update({
                    "Avg Completion Tokens": np.mean(completion_tokens),
                    "Total Completion Tokens": np.sum(completion_tokens),
                })
            except Exception as e:
                print(f"Error computing completion token stats: {e}")
        
        if total_tokens:
            try:
                stats.update({
                    "Avg Total Tokens": np.mean(total_tokens),
                    "Total Tokens": np.sum(total_tokens),
                })
            except Exception as e:
                print(f"Error computing total token stats: {e}")

        print(f"📊 Stats for Experiment '{experiment_name}' in Project '{project_name}':")
        for key, value in stats.items():
            print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

        return stats

    except Exception as e:
        print(f"❌ Error fetching experiment stats: {e}")
        return

project_id = "d596020e-6a95-46db-86db-2f9b885b3547"
experiment_name = "e51988b5-8994-4c39-9f75-4c3d071b13b7"
get_experiment_stats(project_id, experiment_name)