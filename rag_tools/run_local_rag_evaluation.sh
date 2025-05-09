#!/bin/bash

# Local RAG Evaluation Pipeline Script
# This script runs the RAG evaluation pipeline with Ollama local models:
# 1. Creates a vector store from the dataset
# 2. Runs evaluation on the local Ollama-based RAG system
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
LOCAL="🖥️"

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

# Function to check if Ollama is running
check_ollama() {
  echo -e "${YELLOW}Checking if Ollama is running...${NC}"
  if curl -s http://localhost:11434/api/version >/dev/null; then
    echo -e "${GREEN}${CHECK} Ollama is running${NC}"
    return 0
  else
    print_error "Ollama is not running. Please start Ollama with 'ollama serve' before continuing."
    return 1
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

# Set fixed config path for local evaluation
CONFIG_PATH="${ROOT_DIR}/config/ollama_local_rag.yaml"

# Define updated config path
UPDATED_CONFIG_PATH="${CONFIG_PATH%.yaml}_with_vectorstore.yaml"

print_header "Local RAG Evaluation Pipeline Starting" "${LOCAL}"
print_step "Configuration file: ${BOLD}${CONFIG_PATH}${NC}" "${CONFIG}"

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
  print_error "Configuration file not found: $CONFIG_PATH"
  print_step "Checking for alternative configuration file..." "${SEARCH}"
  
  # Try to find a suitable config file
  ALT_CONFIG=$(find ${ROOT_DIR}/config -name "*ollama*" -o -name "*local*" | head -n 1)
  
  if [ -n "$ALT_CONFIG" ]; then
    print_step "Found alternative configuration: ${BOLD}${ALT_CONFIG}${NC}" "${CHECK}"
    CONFIG_PATH="$ALT_CONFIG"
    UPDATED_CONFIG_PATH="${CONFIG_PATH%.yaml}_with_vectorstore.yaml"
  else
    print_error "No suitable configuration file found. Please create one first."
    exit 1
  fi
fi

# Display configuration file content
display_config_file "$CONFIG_PATH"

# Check if Ollama is running
check_ollama || exit 1

# Build command with options
CMD="${ROOT_DIR}/rag_tools/rag_evaluation_pipeline.py --config ${CONFIG_PATH}"

if [ "$FORCE_REBUILD" = true ]; then
  CMD="${CMD} --force-rebuild"
  print_step "Force rebuild: ${BOLD}Yes${NC}" "warning"
fi

if [ "$SKIP_SETUP" = true ]; then
  CMD="${CMD} --skip-setup"
  print_step "Skip setup: ${BOLD}Yes${NC}" "info"
fi

# Create the vector store
print_header "Stage 1: Vector Store Setup" "${DATABASE}"
python ${CMD}

# Check if the pipeline was successful
if [ $? -ne 0 ]; then
  print_error "RAG pipeline failed. Check logs for details."
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

# Run the evaluation (skip setup since we just did it)
print_header "Stage 2: Running Local RAG Evaluation" "${SEARCH}"
python "${ROOT_DIR}/rag_tools/rag_evaluation_pipeline.py" --config "$CONFIG_PATH" --skip-setup

# Check if successful
if [ $? -eq 0 ]; then
  print_header "Local RAG Evaluation Complete" "${DONE}"
else
  print_error "Local RAG evaluation failed. Check logs for details."
  exit 1
fi 