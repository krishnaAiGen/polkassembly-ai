#!/bin/bash

# Polkadot AI Chatbot Stop Script
# This script stops Redis server and the FastAPI application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PID_DIR="./pids"

echo -e "${BLUE}🛑 Stopping Polkadot AI Chatbot System${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Function to stop a process
stop_process() {
    local pid_file=$1
    local process_name=$2
    local force_stop=${3:-false}
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${BLUE}🛑 Stopping $process_name (PID: $pid)...${NC}"
            
            if [ "$force_stop" = true ]; then
                kill -9 "$pid" 2>/dev/null || true
                echo -e "${GREEN}✅ $process_name forcefully stopped${NC}"
            else
                kill "$pid" 2>/dev/null || true
                
                # Wait for graceful shutdown
                local count=0
                while ps -p "$pid" > /dev/null 2>&1 && [ $count -lt 10 ]; do
                    sleep 1
                    count=$((count + 1))
                done
                
                if ps -p "$pid" > /dev/null 2>&1; then
                    echo -e "${YELLOW}⚠️  $process_name not stopping gracefully, force killing...${NC}"
                    kill -9 "$pid" 2>/dev/null || true
                    echo -e "${GREEN}✅ $process_name forcefully stopped${NC}"
                else
                    echo -e "${GREEN}✅ $process_name stopped gracefully${NC}"
                fi
            fi
        else
            echo -e "${YELLOW}⚠️  $process_name process not running (PID: $pid)${NC}"
        fi
        
        # Remove PID file
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}⚠️  $process_name PID file not found${NC}"
    fi
}

# Function to stop all processes
stop_all() {
    echo -e "${BLUE}🛑 Stopping all services...${NC}"
    
    # Stop API Server
    stop_process "$PID_DIR/api_server.pid" "API Server"
    
    # Stop Redis Server
    stop_process "$PID_DIR/redis.pid" "Redis Server"
    
    echo ""
    echo -e "${GREEN}✅ All services stopped${NC}"
}

# Function to show status
show_status() {
    echo ""
    echo -e "${BLUE}📊 System Status:${NC}"
    echo -e "${BLUE}================${NC}"
    
    # Check Redis
    local redis_pid_file="$PID_DIR/redis.pid"
    if [ -f "$redis_pid_file" ]; then
        local redis_pid=$(cat "$redis_pid_file")
        if ps -p "$redis_pid" > /dev/null 2>&1; then
            echo -e "${RED}❌ Redis: Still running (PID: $redis_pid)${NC}"
        else
            echo -e "${GREEN}✅ Redis: Stopped${NC}"
        fi
    else
        echo -e "${GREEN}✅ Redis: Not running${NC}"
    fi
    
    # Check API Server
    local api_pid_file="$PID_DIR/api_server.pid"
    if [ -f "$api_pid_file" ]; then
        local api_pid=$(cat "$api_pid_file")
        if ps -p "$api_pid" > /dev/null 2>&1; then
            echo -e "${RED}❌ API Server: Still running (PID: $api_pid)${NC}"
        else
            echo -e "${GREEN}✅ API Server: Stopped${NC}"
        fi
    else
        echo -e "${GREEN}✅ API Server: Not running${NC}"
    fi
}

# Function to force stop all processes
force_stop() {
    echo -e "${RED}💥 Force stopping all processes...${NC}"
    
    # Force stop API Server
    stop_process "$PID_DIR/api_server.pid" "API Server" true
    
    # Force stop Redis Server
    stop_process "$PID_DIR/redis.pid" "Redis Server" true
    
    echo ""
    echo -e "${GREEN}✅ All services force stopped${NC}"
}

# Function to clean up logs and PID files
cleanup() {
    echo -e "${BLUE}🧹 Cleaning up...${NC}"
    
    # Remove PID files
    rm -f "$PID_DIR"/*.pid 2>/dev/null || true
    
    # Optionally remove log files (uncomment if needed)
    # rm -f "./logs"/*.log 2>/dev/null || true
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
}

# Parse command line arguments
case "${1:-}" in
    "force"|"-f"|"--force")
        force_stop
        ;;
    "cleanup"|"-c"|"--cleanup")
        stop_all
        cleanup
        ;;
    "status"|"-s"|"--status")
        show_status
        exit 0
        ;;
    "help"|"-h"|"--help"|"")
        echo -e "${BLUE}Usage: $0 [OPTION]${NC}"
        echo ""
        echo "Options:"
        echo "  (no option)    Stop all services gracefully"
        echo "  force, -f      Force stop all services"
        echo "  cleanup, -c    Stop services and clean up PID files"
        echo "  status, -s     Show current status"
        echo "  help, -h       Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0              # Stop gracefully"
        echo "  $0 force        # Force stop"
        echo "  $0 cleanup      # Stop and cleanup"
        echo "  $0 status       # Show status"
        exit 0
        ;;
    *)
        echo -e "${RED}❌ Unknown option: $1${NC}"
        echo "Use '$0 help' for usage information"
        exit 1
        ;;
esac

# Show final status
show_status

echo ""
echo -e "${GREEN}🎉 Polkadot AI Chatbot system stopped successfully!${NC}"
echo ""
echo -e "${YELLOW}💡 To restart: ./start_pa_ai.sh${NC}" 