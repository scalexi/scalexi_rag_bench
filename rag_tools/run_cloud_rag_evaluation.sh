#!/bin/bash

# Cloud RAG Evaluation Pipeline Script
# This script runs the RAG evaluation pipeline with OpenAI GPT models:
# 1. Creates a vector store from the dataset
# 2. Runs evaluation on the OpenAI-based RAG system
# 3. Generates reports with metrics

# Define color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
PURPLE='\033[0;35m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Define emojis
ROCKET="🚀"
DATABASE="🗃️"
SEARCH="🔍"
CHECK="✅"
ERROR="❌"
CONFIG="🔧"
DONE="✨"
FILE="📄"
CLOUD="☁️"

# Function to print headers
print_header() {
  local text="$1"
  local emoji="$2"
  local border=$(printf '=%.0s' $(seq 1 $((${#text} + 15))))
  echo -e "\n${BLUE}${border}${NC}"
  echo -e "${BLUE}    ${emoji} ${text}${NC}"
  echo -e "${BLUE}${border}${NC}\n"
}

# Function to print steps
print_step() {
  local text="$1"
  local emoji="$2"
  echo -e "\n${GREEN}${emoji} ${text}${NC}"
}

# Function to print errors
print_error() {
  local text="$1"
  echo -e "${RED}${ERROR} ${text}${NC}"
}

# Function to display file content
display_config_file() {
  local file="$1"
  if [ -f "$file" ]; then
    echo -e "\n${CYAN}${FILE} Configuration file: ${BOLD}$file${NC}"
    echo -e "${YELLOW}----------------------------------------${NC}"
    cat "$file" | sed 's/^/  /'
    echo -e "${YELLOW}----------------------------------------${NC}"
  else
    print_error "Configuration file not found: $file"
  fi
}

# Function to display usage
show_usage() {
  echo -e "${YELLOW}Usage: $0 [OPTIONS]${NC}"
  echo -e "${CYAN}Options:${NC}"
  echo -e "  ${GREEN}-f, --force-rebuild${NC}  Force rebuilding the vector store even if it exists"
  echo -e "  ${GREEN}-s, --skip-setup${NC}     Skip vector store setup and use existing configuration"
  echo -e "  ${GREEN}-h, --help${NC}           Show this help message"
}

# Parse command line arguments
FORCE_REBUILD=false
SKIP_SETUP=false

while [[ $# -gt 0 ]]; do
  case $1 in
    -f|--force-rebuild)
      FORCE_REBUILD=true
      shift
      ;;
    -s|--skip-setup)
      SKIP_SETUP=true
      shift
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      print_error "Unknown option: $1"
      show_usage
      exit 1
      ;;
  esac
done

# Make sure we're in the project root directory
cd "$(dirname "$0")/.."
ROOT_DIR="$(pwd)"

# Set fixed config path for cloud evaluation
CONFIG_PATH="${ROOT_DIR}/config/llm_cloud_rag.yaml"

# Define updated config path
UPDATED_CONFIG_PATH="${CONFIG_PATH%.yaml}_with_vectorstore.yaml"

print_header "Cloud RAG Evaluation Pipeline Starting" "${CLOUD}"
print_step "Configuration file: ${BOLD}${CONFIG_PATH}${NC}" "${CONFIG}"

# Display configuration file content
display_config_file "$CONFIG_PATH"

if [ "$SKIP_SETUP" = false ]; then
  # Create the vector store
  print_header "Stage 1: Vector Store Setup" "${DATABASE}"
  
  # Build command with options
  SETUP_CMD="${ROOT_DIR}/rag_tools/rag_evaluation_pipeline.py --config ${CONFIG_PATH} --skip-evaluation"
  
  if [ "$FORCE_REBUILD" = true ]; then
    SETUP_CMD="${SETUP_CMD} --force-rebuild"
    print_step "Force rebuild: ${BOLD}Yes${NC}" "${CONFIG}"
  fi
  
  python ${SETUP_CMD}
  
  # Check if the setup was successful
  if [ $? -ne 0 ]; then
    print_error "Vector store setup failed. Check logs for details."
    exit 1
  fi
  
  # Get the updated config path if it exists
  if [ -f "$UPDATED_CONFIG_PATH" ]; then
    print_step "Using vector store configuration: ${BOLD}${UPDATED_CONFIG_PATH}${NC}" "${CONFIG}"
    
    # Show the differences between the original and updated config
    echo -e "\n${CYAN}${CONFIG} Configuration changes:${NC}"
    echo -e "${YELLOW}----------------------------------------${NC}"
    diff -y --suppress-common-lines "$CONFIG_PATH" "$UPDATED_CONFIG_PATH" | sed 's/^/  /'
    echo -e "${YELLOW}----------------------------------------${NC}"
    
    CONFIG_PATH="$UPDATED_CONFIG_PATH"
  fi
fi

# Run the evaluation
print_header "Stage 2: Running Cloud RAG Evaluation" "${SEARCH}"
python "${ROOT_DIR}/rag_tools/rag_evaluation_pipeline.py" --config "$CONFIG_PATH" --skip-setup

# Check if successful
if [ $? -eq 0 ]; then
  print_header "Cloud RAG Evaluation Complete" "${DONE}"
else
  print_error "Cloud RAG evaluation failed. Check logs for details."
  exit 1
fi 