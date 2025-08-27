#!/bin/bash

# Enhanced Log File Request Arrival Rate Analysis Script - Quick Start Script

echo "=========================================="
echo "Enhanced Log File Request Arrival Rate Analysis Script"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python3 not found, please install Python3 first"
    exit 1
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import pandas, numpy, matplotlib, seaborn, rich" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip3 install -r requirements_analysis.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi

# Check if log file exists
LOG_FILE="/mnt/shared/data/replay-logs-origin.log"
if [ ! -f "$LOG_FILE" ]; then
    echo "Warning: Default log file does not exist: $LOG_FILE"
    echo "Please use -i parameter to specify the correct log file path"
    echo ""
fi

# Display usage instructions
echo "Usage:"
echo "  python3 analyze_arrival_rate.py [options]"
echo ""
echo "Common options:"
echo "  -i <file_path>   Specify log file path"
echo "  --sample-range <start> <end>  Set sampling range (e.g., 0.0 0.2 for first 20% of requests)"
echo "  -o <output_dir>  Specify output directory"
echo "  --no-plot        Do not generate charts, only output statistics"
echo "  -v               Enable detailed logging"
echo "  -h               Show help information"
echo ""

# Ask user if they want to run analysis
read -p "Do you want to run the enhanced analysis now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting enhanced analysis..."
    echo ""
    
    # Run the enhanced analysis script
    python3 analyze_arrival_rate.py "$@"
    
    echo ""
    echo "Enhanced analysis completed! Results saved in analysis_results/ directory"
    echo "Includes anomaly detection, advanced visualization and other enhanced features"
else
    echo "Analysis cancelled. You can run it manually later:"
    echo "  python3 analyze_arrival_rate.py"
fi 