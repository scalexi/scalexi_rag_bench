#!/bin/bash

# Configuration variables
VERSION="1.1.0"
BRANCH="main"
MESSAGE="added vectorstore from text data source ${VERSION}"
REMOTE="origin"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        print_message $RED "Error: $1 is not installed"
        exit 1
    fi
}

# Check required commands
check_command "git"

# Function to check git status
check_git_status() {
    if ! git rev-parse --is-inside-work-tree &> /dev/null; then
        print_message $RED "Error: Not a git repository"
        exit 1
    fi
}

# Function to update repository
update_repo() {
    print_message $YELLOW "Starting repository update process..."
    
    # Fetch latest changes
    print_message $YELLOW "Fetching latest changes..."
    if ! git fetch $REMOTE; then
        print_message $RED "Error: Failed to fetch from remote"
        exit 1
    fi

    # Switch to specified branch
    print_message $YELLOW "Switching to branch: $BRANCH"
    if ! git checkout $BRANCH; then
        print_message $RED "Error: Failed to switch to branch $BRANCH"
        exit 1
    fi

    # Pull latest changes
    print_message $YELLOW "Pulling latest changes..."
    if ! git pull $REMOTE $BRANCH; then
        print_message $RED "Error: Failed to pull latest changes"
        exit 1
    fi

    # Add all changes
    print_message $YELLOW "Adding changes..."
    if ! git add .; then
        print_message $RED "Error: Failed to add changes"
        exit 1
    fi

    # Commit changes
    print_message $YELLOW "Committing changes..."
    if ! git commit -m "$MESSAGE"; then
        print_message $RED "Error: Failed to commit changes"
        exit 1
    fi

    # Push changes
    print_message $YELLOW "Pushing changes..."
    if ! git push $REMOTE $BRANCH; then
        print_message $RED "Error: Failed to push changes"
        exit 1
    fi

    print_message $GREEN "Repository successfully updated!"
    print_message $GREEN "Version: $VERSION"
    print_message $GREEN "Branch: $BRANCH"
    print_message $GREEN "Commit message: $MESSAGE"
}

# Main execution
check_git_status
update_repo 