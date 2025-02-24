#!/bin/bash

# ANSI color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored text
print_color() {
    local color=$1
    local text=$2
    echo -e "${color}${text}${NC}"
}

# Function to display the main menu
show_main_menu() {
    clear
    print_color "$BLUE" "Welcome to the LoRA Test Runner 🚀\n"
    echo "Please select a test scenario:"
    echo "1) Basic single-step tests"
    echo "2) Multi-step conversation tests"
    echo "3) Interactive light control"
    echo "4) Interactive weather check"
    echo "5) Error handling tests"
    echo "6) Custom test (specify parameters)"
    echo "7) Exit"
    echo
}

# Function to get yes/no input
get_yes_no() {
    local prompt=$1
    local default=$2
    local answer
    
    while true; do
        read -p "$prompt" answer
        case $answer in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            "" ) 
                if [ "$default" = "Y" ]; then
                    return 0
                else
                    return 1
                fi
                ;;
            * ) echo "Please answer yes or no.";;
        esac
    done
}

# Function to configure custom test
configure_custom_test() {
    local params=""
    print_color "$YELLOW" "\nCustom Test Configuration:"
    
    if get_yes_no "Include multi-step tests? (y/N): " "N"; then
        params="$params --multi-step-test"
    fi
    
    if get_yes_no "Run in interactive mode? (y/N): " "N"; then
        params="$params --interactive"
    fi
    
    if get_yes_no "Test lights scenario? (y/N): " "N"; then
        params="$params --lights"
    fi
    
    if get_yes_no "Test weather scenario? (y/N): " "N"; then
        params="$params --weather"
    fi
    
    if get_yes_no "Include error cases? (y/N): " "N"; then
        params="$params --failure"
    fi
    
    echo -e "\nConfigured command: python scripts/test_simple.py $params\n"
    
    if get_yes_no "Proceed with this configuration? (Y/n): " "Y"; then
        run_test "$params"
    fi
}

# Function to run the test with given parameters
run_test() {
    local params=$1
    
    # Ensure we're in the right virtual environment
    if [[ "$VIRTUAL_ENV" != *"$(pwd)/venv"* ]]; then
        print_color "$YELLOW" "\nActivating virtual environment..."
        source venv/bin/activate
    fi
    
    # Add current directory to Python path
    print_color "$YELLOW" "Setting up Python path..."
    export PYTHONPATH="$PYTHONPATH:$(pwd)"
    
    print_color "$GREEN" "\nRunning test...\n"
    python scripts/test_simple.py $params
    
    local exit_code=$?
    echo
    if [ $exit_code -eq 0 ]; then
        print_color "$GREEN" "Test completed successfully! 🎉"
    else
        print_color "$RED" "Test failed with exit code $exit_code"
    fi
    
    echo
    if get_yes_no "Would you like to run another test? (y/N): " "N"; then
        return 0
    else
        return 1
    fi
}

# Main loop
while true; do
    show_main_menu
    read -p "Enter your choice (1-7): " choice
    
    case $choice in
        1)
            run_test "" || break
            ;;
        2)
            run_test "--multi-step-test" || break
            ;;
        3)
            if get_yes_no "Include error scenarios? (y/N): " "N"; then
                run_test "--lights --failure" || break
            else
                run_test "--lights" || break
            fi
            ;;
        4)
            if get_yes_no "Include error scenarios? (y/N): " "N"; then
                run_test "--weather --failure" || break
            else
                run_test "--weather" || break
            fi
            ;;
        5)
            run_test "--lights --weather --failure" || break
            ;;
        6)
            configure_custom_test || break
            ;;
        7)
            print_color "$GREEN" "\nThank you for using the LoRA Test Runner! 👋\n"
            exit 0
            ;;
        *)
            print_color "$RED" "\nInvalid option. Please try again.\n"
            sleep 1
            ;;
    esac
done 