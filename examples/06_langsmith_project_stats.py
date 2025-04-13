from langsmith import Client
import os
def get_project_stats(project_id: str):
    # Initialize LangSmith client
    # Get API key from environment variable
    api_key = os.getenv("LANGCHAIN_API_KEY")
    client = Client(api_key=api_key)   
    
    try:
        # Fetch project details
        project = client.read_project(project_id=project_id, include_stats=True)
        
        # Extract relevant stats
        stats = {
            "Name": project.name,
            "Run Count": project.run_count,
            "Latency (P50)": project.latency_p50,
            "Latency (P99)": project.latency_p99,
            "Total Tokens": project.total_tokens,
            "Prompt Tokens": project.prompt_tokens,
            "Completion Tokens": project.completion_tokens,
            "Total Cost": project.total_cost,
            "Error Rate": project.error_rate,
            "Feedback Stats": project.feedback_stats
        }
        
        # Print stats
        print("Experiment Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error fetching project stats: {e}")

# Replace with your project ID
project_id = "d596020e-6a95-46db-86db-2f9b885b3547"
print("LangSmith API Key: ", os.getenv("LANGCHAIN_API_KEY"))

# Call the function
get_project_stats(project_id)