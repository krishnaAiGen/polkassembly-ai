#!/bin/bash

# Polkadot AI Chatbot Startup Script
# This script starts Redis server and the FastAPI application with Gunicorn

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
API_HOST=${API_HOST:-"0.0.0.0"}
API_PORT=${API_PORT:-8000}
WORKERS=${WORKERS:-4}
REDIS_HOST=${REDIS_HOST:-"localhost"}
REDIS_PORT=${REDIS_PORT:-6379}
LOG_DIR="./logs"
PID_DIR="./pids"

# Create directories if they don't exist
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

echo -e "${PURPLE}🚀 Starting Polkadot AI Chatbot System${NC}"
echo -e "${PURPLE}=====================================${NC}"
echo ""

# Function to log with timestamp
log_with_timestamp() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

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

# Function to check dependencies
check_dependencies() {
    echo -e "${BLUE}🔍 Checking dependencies...${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}❌ Python 3 not found. Please install Python 3.8+${NC}"
        exit 1
    fi
    
    # Check run_server.py
    if [ ! -f "run_server.py" ]; then
        echo -e "${RED}❌ run_server.py not found in current directory${NC}"
        exit 1
    fi
    
    # Check requirements
    local missing_packages=()
    
    if ! python3 -c "import gunicorn" 2>/dev/null; then
        missing_packages+=("gunicorn")
    fi
    
    if ! python3 -c "import uvicorn" 2>/dev/null; then
        missing_packages+=("uvicorn[standard]")
    fi
    
    if [ ${#missing_packages[@]} -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Installing missing packages: ${missing_packages[*]}${NC}"
        pip3 install "${missing_packages[@]}"
    fi
    
    echo -e "${GREEN}✅ Dependencies checked${NC}"
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
        echo ""
        echo -e "${CYAN}Installation commands:${NC}"
        echo "  macOS:         brew install redis"
        echo "  Ubuntu/Debian: sudo apt-get install redis-server"
        echo "  CentOS/RHEL:   sudo yum install redis"
        echo "  Docker:        docker run -d -p 6379:6379 redis:alpine"
        exit 1
    fi
    
    # Start Redis server with nohup
    echo "Starting Redis server on $REDIS_HOST:$REDIS_PORT..."
    log_with_timestamp "Starting Redis server..." >> "$redis_log_file"
    
    nohup redis-server \
        --port "$REDIS_PORT" \
        --bind "$REDIS_HOST" \
        --save 900 1 \
        --save 300 10 \
        --save 60 10000 \
        --rdbcompression yes \
        --rdbchecksum yes \
        --logfile "$redis_log_file" \
        --loglevel notice \
        --timeout 300 \
        --tcp-keepalive 300 >> "$redis_log_file" 2>&1 &
    
    local redis_pid=$!
    echo "$redis_pid" > "$redis_pid_file"
    
    # Wait for Redis to start
    echo "Waiting for Redis to initialize..."
    sleep 3
    
    # Test Redis connection with timeout
    local redis_ready=false
    local attempts=0
    while [ $attempts -lt 10 ]; do
        if timeout 3 redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping > /dev/null 2>&1; then
            redis_ready=true
            break
        fi
        attempts=$((attempts + 1))
        echo "Waiting for Redis... (attempt $attempts/10)"
        sleep 1
    done
    
    if [ "$redis_ready" = true ]; then
        log_with_timestamp "Redis server started successfully (PID: $redis_pid)" >> "$redis_log_file"
        echo -e "${GREEN}✅ Redis server started successfully (PID: $redis_pid)${NC}"
        echo -e "${BLUE}   Host: $REDIS_HOST:$REDIS_PORT${NC}"
        echo -e "${BLUE}   Logs: $redis_log_file${NC}"
    else
        echo -e "${RED}❌ Failed to start Redis server or connection timeout${NC}"
        echo -e "${YELLOW}   Check logs: $redis_log_file${NC}"
        exit 1
    fi
}

# Function to start API server
start_api_server() {
    local api_pid_file="$PID_DIR/api_server.pid"
    local api_log_file="$LOG_DIR/api_server.log"
    local api_error_log_file="$LOG_DIR/api_server_error.log"
    local gunicorn_access_log="$LOG_DIR/gunicorn_access.log"
    
    echo -e "${BLUE}🤖 Starting API server with Gunicorn...${NC}"
    
    if check_process "$api_pid_file" "API Server"; then
        return 0
    fi
    
    # Log startup
    log_with_timestamp "Starting Polkadot AI API server with Gunicorn..." >> "$api_log_file"
    
    # Start API server with Gunicorn using run_server.py
    echo "Starting API server on $API_HOST:$API_PORT with $WORKERS workers..."
    
    # Use gunicorn with the run_server module
    nohup gunicorn \
        --bind "$API_HOST:$API_PORT" \
        --workers "$WORKERS" \
        --worker-class uvicorn.workers.UvicornWorker \
        --worker-connections 1000 \
        --max-requests 1000 \
        --max-requests-jitter 50 \
        --timeout 120 \
        --keep-alive 5 \
        --graceful-timeout 30 \
        --access-logfile "$gunicorn_access_log" \
        --error-logfile "$api_error_log_file" \
        --capture-output \
        --log-level info \
        --log-config-json <(cat <<EOF
{
    "version": 1,
    "disable_existing_loggers": false,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["default"]
    }
}
EOF
        ) \
        --preload \
        --enable-stdio-inheritance \
        src.rag.api_server:app >> "$api_log_file" 2>&1 &
    
    local api_pid=$!
    echo "$api_pid" > "$api_pid_file"
    
    # Log PID
    log_with_timestamp "API server started with PID: $api_pid" >> "$api_log_file"
    
    # Wait for server to initialize
    echo "Waiting for API server to initialize..."
    sleep 5
    
    # Test API server with multiple attempts
    local api_ready=false
    local attempts=0
    while [ $attempts -lt 15 ]; do
        if curl -s --max-time 5 "http://$API_HOST:$API_PORT/health" > /dev/null 2>&1; then
            api_ready=true
            break
        fi
        attempts=$((attempts + 1))
        echo "Waiting for API server... (attempt $attempts/15)"
        sleep 2
    done
    
    if [ "$api_ready" = true ]; then
        log_with_timestamp "API server health check passed" >> "$api_log_file"
        echo -e "${GREEN}✅ API server started successfully (PID: $api_pid)${NC}"
        echo -e "${BLUE}   URL: http://$API_HOST:$API_PORT${NC}"
        echo -e "${BLUE}   Docs: http://$API_HOST:$API_PORT/docs${NC}"
        echo -e "${BLUE}   Health: http://$API_HOST:$API_PORT/health${NC}"
        echo -e "${BLUE}   Main Log: $api_log_file${NC}"
        echo -e "${BLUE}   Error Log: $api_error_log_file${NC}"
        echo -e "${BLUE}   Access Log: $gunicorn_access_log${NC}"
    else
        echo -e "${YELLOW}⚠️  API server may still be starting up...${NC}"
        echo -e "${BLUE}   Check logs: $api_log_file${NC}"
        echo -e "${BLUE}   Check error logs: $api_error_log_file${NC}"
        echo -e "${YELLOW}   You can test manually: curl http://$API_HOST:$API_PORT/health${NC}"
    fi
}

# Function to show comprehensive status
show_status() {
    echo ""
    echo -e "${PURPLE}📊 System Status Report:${NC}"
    echo -e "${PURPLE}========================${NC}"
    
    # System info
    echo -e "${CYAN}🖥️  System Information:${NC}"
    echo "   Timestamp: $(date)"
    echo "   User: $(whoami)"
    echo "   Working Directory: $(pwd)"
    echo ""
    
    # Check Redis
    echo -e "${CYAN}📦 Redis Server:${NC}"
    local redis_pid_file="$PID_DIR/redis.pid"
    if [ -f "$redis_pid_file" ]; then
        local redis_pid=$(cat "$redis_pid_file")
        if ps -p "$redis_pid" > /dev/null 2>&1; then
            echo -e "   Status: ${GREEN}✅ Running (PID: $redis_pid)${NC}"
            if timeout 2 redis-cli -p "$REDIS_PORT" ping > /dev/null 2>&1; then
                echo -e "   Connection: ${GREEN}✅ Responsive${NC}"
                local redis_info=$(redis-cli -p "$REDIS_PORT" info server | grep "redis_version" | cut -d: -f2 | tr -d '\r')
                echo "   Version: $redis_info"
            else
                echo -e "   Connection: ${RED}❌ Not responding${NC}"
            fi
        else
            echo -e "   Status: ${RED}❌ Not running${NC}"
        fi
    else
        echo -e "   Status: ${RED}❌ Not started${NC}"
    fi
    echo ""
    
    # Check API Server
    echo -e "${CYAN}🤖 API Server:${NC}"
    local api_pid_file="$PID_DIR/api_server.pid"
    if [ -f "$api_pid_file" ]; then
        local api_pid=$(cat "$api_pid_file")
        if ps -p "$api_pid" > /dev/null 2>&1; then
            echo -e "   Status: ${GREEN}✅ Running (PID: $api_pid)${NC}"
            if curl -s --max-time 3 "http://$API_HOST:$API_PORT/health" > /dev/null 2>&1; then
                echo -e "   Health Check: ${GREEN}✅ Healthy${NC}"
            else
                echo -e "   Health Check: ${YELLOW}⚠️  Not responding${NC}"
            fi
        else
            echo -e "   Status: ${RED}❌ Not running${NC}"
        fi
    else
        echo -e "   Status: ${RED}❌ Not started${NC}"
    fi
    
    # Show URLs
    echo ""
    echo -e "${PURPLE}🌐 Access URLs:${NC}"
    echo -e "${PURPLE}===============${NC}"
    echo -e "${GREEN}API Base URL:${NC} http://$API_HOST:$API_PORT"
    echo -e "${GREEN}API Documentation:${NC} http://$API_HOST:$API_PORT/docs"
    echo -e "${GREEN}Health Check:${NC} http://$API_HOST:$API_PORT/health"
    echo -e "${GREEN}OpenAPI Schema:${NC} http://$API_HOST:$API_PORT/openapi.json"
    
    # Show log files
    echo ""
    echo -e "${PURPLE}📁 Log Files:${NC}"
    echo -e "${PURPLE}=============${NC}"
    if [ -f "$LOG_DIR/redis.log" ]; then
        local redis_log_size=$(du -h "$LOG_DIR/redis.log" | cut -f1)
        echo -e "${GREEN}Redis Log:${NC} $LOG_DIR/redis.log (${redis_log_size})"
    fi
    if [ -f "$LOG_DIR/api_server.log" ]; then
        local api_log_size=$(du -h "$LOG_DIR/api_server.log" | cut -f1)
        echo -e "${GREEN}API Server Log:${NC} $LOG_DIR/api_server.log (${api_log_size})"
    fi
    if [ -f "$LOG_DIR/api_server_error.log" ]; then
        local error_log_size=$(du -h "$LOG_DIR/api_server_error.log" | cut -f1)
        echo -e "${GREEN}Error Log:${NC} $LOG_DIR/api_server_error.log (${error_log_size})"
    fi
    if [ -f "$LOG_DIR/gunicorn_access.log" ]; then
        local access_log_size=$(du -h "$LOG_DIR/gunicorn_access.log" | cut -f1)
        echo -e "${GREEN}Access Log:${NC} $LOG_DIR/gunicorn_access.log (${access_log_size})"
    fi
}

# Main execution
main() {
    echo -e "${BLUE}🔧 Configuration:${NC}"
    echo "  API Host: $API_HOST"
    echo "  API Port: $API_PORT" 
    echo "  Workers: $WORKERS"
    echo "  Redis Host: $REDIS_HOST"
    echo "  Redis Port: $REDIS_PORT"
    echo "  Log Directory: $LOG_DIR"
    echo "  PID Directory: $PID_DIR"
    echo ""
    
    # Check dependencies first
    check_dependencies
    echo ""
    
    # Start Redis
    start_redis
    echo ""
    
    # Start API Server
    start_api_server
    
    # Show comprehensive status
    show_status
    
    echo ""
    echo -e "${GREEN}🎉 Polkadot AI Chatbot system started successfully!${NC}"
    echo ""
    echo -e "${YELLOW}💡 Useful commands:${NC}"
    echo "  Stop system:      ./stop_pa_ai.sh"
    echo "  View API logs:    tail -f $LOG_DIR/api_server.log"
    echo "  View Redis logs:  tail -f $LOG_DIR/redis.log" 
    echo "  View access logs: tail -f $LOG_DIR/gunicorn_access.log"
    echo "  Health check:     curl http://$API_HOST:$API_PORT/health"
    echo "  Status check:     ./stop_pa_ai.sh status"
    echo "  Force stop:       ./stop_pa_ai.sh force"
    echo ""
    echo -e "${CYAN}📖 Documentation:${NC} http://$API_HOST:$API_PORT/docs"
    echo ""
}

# Run main function
main "$@" 