#!/bin/bash

# RAG Runner Helper Script
# This script helps users choose the right RAG evaluation script

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
CLOUD="☁️"
LOCAL="🖥️"
INFO="ℹ️"
ARROW="➡️"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Show header
echo -e "${BLUE}${BOLD}=======================================${NC}"
echo -e "${BLUE}${BOLD}    ${ROCKET} RAG Evaluation Helper${NC}"
echo -e "${BLUE}${BOLD}=======================================${NC}\n"

# Show available options
echo -e "${CYAN}${BOLD}Choose your RAG Evaluation Pipeline:${NC}\n"

echo -e "${CLOUD} ${GREEN}${BOLD}Cloud RAG Evaluation${NC} (OpenAI, GPT models)"
echo -e "   ${ARROW} ${CYAN}./rag_tools/run_cloud_rag_evaluation.sh${NC} [OPTIONS]\n"

echo -e "${LOCAL} ${GREEN}${BOLD}Local RAG Evaluation${NC} (Ollama, local models)"
echo -e "   ${ARROW} ${CYAN}./rag_tools/run_local_rag_evaluation.sh${NC} [OPTIONS]\n"

echo -e "${YELLOW}${BOLD}Common Options:${NC}"
echo -e "  ${CYAN}-f, --force-rebuild${NC}  Force rebuilding the vector store even if it exists"
echo -e "  ${CYAN}-s, --skip-setup${NC}     Skip vector store setup and use existing configuration"
echo -e "  ${CYAN}-h, --help${NC}           Show help message\n"

echo -e "${PURPLE}${INFO} Examples:${NC}"
echo -e "${CYAN}  # Run cloud evaluation with default config:${NC}"
echo -e "  ./rag_tools/run_cloud_rag_evaluation.sh\n"

echo -e "${CYAN}  # Run local evaluation with force rebuild:${NC}"
echo -e "  ./rag_tools/run_local_rag_evaluation.sh --force-rebuild\n"

echo -e "${CYAN}  # Get detailed help for a specific script:${NC}"
echo -e "  ./rag_tools/run_cloud_rag_evaluation.sh --help\n"

# Check if any arguments were provided
if [ $# -gt 0 ]; then
    # If the first argument is 'cloud', run the cloud script
    if [ "$1" == "cloud" ]; then
        shift
        echo -e "${BLUE}${BOLD}Running Cloud RAG Evaluation...${NC}\n"
        "$SCRIPT_DIR/run_cloud_rag_evaluation.sh" "$@"
    # If the first argument is 'local', run the local script
    elif [ "$1" == "local" ]; then
        shift
        echo -e "${BLUE}${BOLD}Running Local RAG Evaluation...${NC}\n"
        "$SCRIPT_DIR/run_local_rag_evaluation.sh" "$@"
    else
        echo -e "${RED}Unknown option: $1${NC}"
        echo -e "Use ${CYAN}cloud${NC} or ${CYAN}local${NC} as first argument to run directly"
    fi
fi 