#!/bin/bash

# Polkadot AI Chatbot Startup Script
# This script starts Redis server and the FastAPI application with Gunicorn

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
API_HOST=${API_HOST:-"0.0.0.0"}
API_PORT=${API_PORT:-8000}
WORKERS=${WORKERS:-3}
REDIS_HOST=${REDIS_HOST:-"localhost"}
REDIS_PORT=${REDIS_PORT:-6379}
LOG_DIR="./logs"
PID_DIR="./pids"

# Create directories if they don't exist
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

echo -e "${BLUE}🚀 Starting Polkadot AI Chatbot System${NC}"
echo -e "${BLUE}=====================================${NC}"
echo ""

# Function to check if a process is running
check_process() {
    local pid_file=$1
    local process_name=$2
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $process_name is already running (PID: $pid)${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️  $process_name PID file exists but process not running, cleaning up...${NC}"
            rm -f "$pid_file"
        fi
    fi
    return 1
}

# Function to start Redis server
start_redis() {
    local redis_pid_file="$PID_DIR/redis.pid"
    local redis_log_file="$LOG_DIR/redis.log"
    
    echo -e "${BLUE}📦 Starting Redis server...${NC}"
    
    if check_process "$redis_pid_file" "Redis"; then
        return 0
    fi
    
    # Check if Redis is installed
    if ! command -v redis-server &> /dev/null; then
        echo -e "${RED}❌ Redis server not found. Please install Redis first.${NC}"
        echo "Installation commands:"
        echo "  macOS: brew install redis"
        echo "  Ubuntu/Debian: sudo apt-get install redis-server"
        echo "  CentOS/RHEL: sudo yum install redis"
        exit 1
    fi
    
    # Start Redis server
    echo "Starting Redis server on $REDIS_HOST:$REDIS_PORT..."
    nohup redis-server --port "$REDIS_PORT" > "$redis_log_file" 2>&1 &
    local redis_pid=$!
    echo "$redis_pid" > "$redis_pid_file"
    
    # Wait a moment for Redis to start
    sleep 2
    
    # Test Redis connection
    if redis-cli -p "$REDIS_PORT" ping > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Redis server started successfully (PID: $redis_pid)${NC}"
        echo -e "${BLUE}   Logs: $redis_log_file${NC}"
    else
        echo -e "${RED}❌ Failed to start Redis server${NC}"
        echo -e "${YELLOW}   Check logs: $redis_log_file${NC}"
        exit 1
    fi
}

# Function to start API server
start_api_server() {
    local api_pid_file="$PID_DIR/api_server.pid"
    local api_log_file="$LOG_DIR/api_server.log"
    local api_error_log_file="$LOG_DIR/api_server_error.log"
    
    echo -e "${BLUE}🤖 Starting API server with Gunicorn...${NC}"
    
    if check_process "$api_pid_file" "API Server"; then
        return 0
    fi
    
    # Check if required packages are installed
    if ! python -c "import gunicorn" 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Gunicorn not found. Installing...${NC}"
        pip install gunicorn
    fi
    
    # Check if the application module exists
    if [ ! -f "src/rag/api_server.py" ]; then
        echo -e "${RED}❌ API server module not found at src/rag/api_server.py${NC}"
        exit 1
    fi
    
    # Start API server with Gunicorn
    echo "Starting API server on $API_HOST:$API_PORT with $WORKERS workers..."
    
    nohup gunicorn \
        --bind "$API_HOST:$API_PORT" \
        --workers "$WORKERS" \
        --worker-class uvicorn.workers.UvicornWorker \
        --access-logfile "$LOG_DIR/access.log" \
        --error-logfile "$api_error_log_file" \
        --log-level info \
        --timeout 120 \
        --keep-alive 5 \
        --max-requests 1000 \
        --max-requests-jitter 100 \
        --preload \
        src.rag.api_server:app > "$api_log_file" 2>&1 &
    
    local api_pid=$!
    echo "$api_pid" > "$api_pid_file"
    
    # Wait a moment for the server to start
    sleep 3
    
    # Test API server
    if curl -s "http://$API_HOST:$API_PORT/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ API server started successfully (PID: $api_pid)${NC}"
        echo -e "${BLUE}   URL: http://$API_HOST:$API_PORT${NC}"
        echo -e "${BLUE}   Docs: http://$API_HOST:$API_PORT/docs${NC}"
        echo -e "${BLUE}   Logs: $api_log_file${NC}"
        echo -e "${BLUE}   Error Logs: $api_error_log_file${NC}"
    else
        echo -e "${YELLOW}⚠️  API server may still be starting up...${NC}"
        echo -e "${BLUE}   Check logs: $api_log_file${NC}"
        echo -e "${BLUE}   Check error logs: $api_error_log_file${NC}"
    fi
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
            echo -e "${GREEN}✅ Redis: Running (PID: $redis_pid)${NC}"
        else
            echo -e "${RED}❌ Redis: Not running${NC}"
        fi
    else
        echo -e "${RED}❌ Redis: Not started${NC}"
    fi
    
    # Check API Server
    local api_pid_file="$PID_DIR/api_server.pid"
    if [ -f "$api_pid_file" ]; then
        local api_pid=$(cat "$api_pid_file")
        if ps -p "$api_pid" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ API Server: Running (PID: $api_pid)${NC}"
        else
            echo -e "${RED}❌ API Server: Not running${NC}"
        fi
    else
        echo -e "${RED}❌ API Server: Not started${NC}"
    fi
    
    # Show URLs
    echo ""
    echo -e "${BLUE}🌐 Access URLs:${NC}"
    echo -e "${BLUE}===============${NC}"
    echo -e "${GREEN}API Base URL:${NC} http://$API_HOST:$API_PORT"
    echo -e "${GREEN}API Docs:${NC}     http://$API_HOST:$API_PORT/docs"
    echo -e "${GREEN}Health Check:${NC} http://$API_HOST:$API_PORT/health"
    echo ""
    echo -e "${BLUE}📁 Log Files:${NC}"
    echo -e "${BLUE}=============${NC}"
    echo -e "${GREEN}Redis Log:${NC}    $LOG_DIR/redis.log"
    echo -e "${GREEN}API Log:${NC}      $LOG_DIR/api_server.log"
    echo -e "${GREEN}API Error Log:${NC} $LOG_DIR/api_server_error.log"
    echo -e "${GREEN}Access Log:${NC}   $LOG_DIR/access.log"
}

# Main execution
main() {
    echo -e "${BLUE}🔧 Configuration:${NC}"
    echo "  API Host: $API_HOST"
    echo "  API Port: $API_PORT"
    echo "  Workers: $WORKERS"
    echo "  Redis Host: $REDIS_HOST"
    echo "  Redis Port: $REDIS_PORT"
    echo ""
    
    # Start Redis
    start_redis
    
    # Start API Server
    start_api_server
    
    # Show status
    show_status
    
    echo ""
    echo -e "${GREEN}🎉 Polkadot AI Chatbot system started successfully!${NC}"
    echo ""
    echo -e "${YELLOW}💡 Useful commands:${NC}"
    echo "  Stop system:    ./stop_pa_ai.sh"
    echo "  View logs:      tail -f $LOG_DIR/api_server.log"
    echo "  Health check:   curl http://$API_HOST:$API_PORT/health"
    echo "  API docs:       open http://$API_HOST:$API_PORT/docs"
    echo ""
}

# Run main function
main "$@" 