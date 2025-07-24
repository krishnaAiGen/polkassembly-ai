#!/bin/bash

# Polkadot AI Chatbot Stop Script
# This script stops Redis server and the FastAPI application

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
PID_DIR="./pids"
LOG_DIR="./logs"

echo -e "${PURPLE}🛑 Stopping Polkadot AI Chatbot System${NC}"
echo -e "${PURPLE}=====================================${NC}"
echo ""

# Function to log with timestamp
log_with_timestamp() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Function to check if a process is running
is_process_running() {
    local pid=$1
    ps -p "$pid" > /dev/null 2>&1
}

# Function to stop a process
stop_process() {
    local pid_file=$1
    local process_name=$2
    local force_stop=${3:-false}
    local log_file=${4:-""}
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if is_process_running "$pid"; then
            echo -e "${BLUE}🛑 Stopping $process_name (PID: $pid)...${NC}"
            
            # Log the shutdown attempt
            if [ -n "$log_file" ] && [ -f "$log_file" ]; then
                log_with_timestamp "Stopping $process_name (PID: $pid)" >> "$log_file"
            fi
            
            if [ "$force_stop" = true ]; then
                echo "   Force stopping process..."
                kill -9 "$pid" 2>/dev/null || true
                echo -e "${GREEN}✅ $process_name forcefully stopped${NC}"
            else
                echo "   Sending graceful shutdown signal (SIGTERM)..."
                kill "$pid" 2>/dev/null || true
                
                # Wait for graceful shutdown
                local count=0
                local max_wait=15
                while is_process_running "$pid" && [ $count -lt $max_wait ]; do
                    echo "   Waiting for graceful shutdown... ($((count+1))/$max_wait)"
                    sleep 1
                    count=$((count + 1))
                done
                
                if is_process_running "$pid"; then
                    echo -e "${YELLOW}⚠️  $process_name not stopping gracefully, force killing...${NC}"
                    kill -9 "$pid" 2>/dev/null || true
                    sleep 1
                    if is_process_running "$pid"; then
                        echo -e "${RED}❌ Failed to stop $process_name (PID: $pid)${NC}"
                    else
                        echo -e "${GREEN}✅ $process_name forcefully stopped${NC}"
                    fi
                else
                    echo -e "${GREEN}✅ $process_name stopped gracefully${NC}"
                fi
            fi
            
            # Log successful shutdown
            if [ -n "$log_file" ] && [ -f "$log_file" ]; then
                log_with_timestamp "$process_name stopped successfully" >> "$log_file"
            fi
        else
            echo -e "${YELLOW}⚠️  $process_name process not running (stale PID: $pid)${NC}"
        fi
        
        # Remove PID file
        rm -f "$pid_file"
        echo "   Removed PID file: $pid_file"
    else
        echo -e "${YELLOW}⚠️  $process_name PID file not found${NC}"
        echo "   Expected PID file: $pid_file"
    fi
}

# Function to kill processes by name (fallback method)
kill_processes_by_name() {
    local process_pattern=$1
    local process_name=$2
    
    echo -e "${BLUE}🔍 Looking for $process_name processes...${NC}"
    
    local pids=$(pgrep -f "$process_pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Found $process_name processes with PIDs: $pids"
        for pid in $pids; do
            if is_process_running "$pid"; then
                echo "   Killing $process_name process (PID: $pid)..."
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
        echo -e "${GREEN}✅ All $process_name processes terminated${NC}"
    else
        echo -e "${GREEN}✅ No $process_name processes found${NC}"
    fi
}

# Function to stop all processes
stop_all() {
    local force_mode=${1:-false}
    
    echo -e "${BLUE}🛑 Stopping all services...${NC}"
    
    if [ "$force_mode" = true ]; then
        echo -e "${RED}💥 Using force mode (SIGKILL)${NC}"
    else
        echo -e "${GREEN}🤝 Using graceful mode (SIGTERM → SIGKILL if needed)${NC}"
    fi
    echo ""
    
    # Stop API Server first (it might depend on Redis)
    echo -e "${CYAN}🤖 Stopping API Server...${NC}"
    stop_process "$PID_DIR/api_server.pid" "API Server" "$force_mode" "$LOG_DIR/api_server.log"
    
    echo ""
    
    # Stop Redis Server
    echo -e "${CYAN}📦 Stopping Redis Server...${NC}"
    stop_process "$PID_DIR/redis.pid" "Redis Server" "$force_mode" "$LOG_DIR/redis.log"
    
    echo ""
    
    # Fallback: kill any remaining processes
    if [ "$force_mode" = true ]; then
        echo -e "${YELLOW}🧹 Cleanup: Searching for remaining processes...${NC}"
        kill_processes_by_name "gunicorn.*polkassembly" "Gunicorn"
        kill_processes_by_name "redis-server" "Redis"
        echo ""
    fi
    
    echo -e "${GREEN}✅ All services stopped${NC}"
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
    
    local any_running=false
    
    # Check Redis
    echo -e "${CYAN}📦 Redis Server:${NC}"
    local redis_pid_file="$PID_DIR/redis.pid"
    if [ -f "$redis_pid_file" ]; then
        local redis_pid=$(cat "$redis_pid_file")
        if is_process_running "$redis_pid"; then
            echo -e "   Status: ${RED}❌ Still running (PID: $redis_pid)${NC}"
            any_running=true
            
            # Check if Redis is actually responding
            if timeout 2 redis-cli ping > /dev/null 2>&1; then
                echo -e "   Connection: ${RED}❌ Still responding to commands${NC}"
            else
                echo -e "   Connection: ${YELLOW}⚠️  Not responding (may be shutting down)${NC}"
            fi
        else
            echo -e "   Status: ${GREEN}✅ Stopped${NC}"
        fi
    else
        echo -e "   Status: ${GREEN}✅ Not running${NC}"
        
        # Check for any rogue redis processes
        local redis_pids=$(pgrep -f "redis-server" 2>/dev/null || true)
        if [ -n "$redis_pids" ]; then
            echo -e "   Warning: ${YELLOW}⚠️  Found orphaned Redis processes: $redis_pids${NC}"
            any_running=true
        fi
    fi
    echo ""
    
    # Check API Server
    echo -e "${CYAN}🤖 API Server:${NC}"
    local api_pid_file="$PID_DIR/api_server.pid"
    if [ -f "$api_pid_file" ]; then
        local api_pid=$(cat "$api_pid_file")
        if is_process_running "$api_pid"; then
            echo -e "   Status: ${RED}❌ Still running (PID: $api_pid)${NC}"
            any_running=true
            
            # Check if API is still responding
            if curl -s --max-time 2 "http://localhost:8000/health" > /dev/null 2>&1; then
                echo -e "   Health Check: ${RED}❌ Still responding to requests${NC}"
            else
                echo -e "   Health Check: ${YELLOW}⚠️  Not responding (may be shutting down)${NC}"
            fi
        else
            echo -e "   Status: ${GREEN}✅ Stopped${NC}"
        fi
    else
        echo -e "   Status: ${GREEN}✅ Not running${NC}"
        
        # Check for any rogue gunicorn processes
        local gunicorn_pids=$(pgrep -f "gunicorn.*polkassembly\|uvicorn" 2>/dev/null || true)
        if [ -n "$gunicorn_pids" ]; then
            echo -e "   Warning: ${YELLOW}⚠️  Found orphaned API processes: $gunicorn_pids${NC}"
            any_running=true
        fi
    fi
    echo ""
    
    # Show process summary
    echo -e "${PURPLE}🔍 Process Summary:${NC}"
    echo -e "${PURPLE}==================${NC}"
    
    if [ "$any_running" = true ]; then
        echo -e "${RED}❌ Some processes are still running${NC}"
        echo -e "${YELLOW}   Try: $0 force     # Force stop all processes${NC}"
        echo -e "${YELLOW}   Or:  $0 cleanup   # Stop and cleanup${NC}"
    else
        echo -e "${GREEN}✅ All processes stopped successfully${NC}"
    fi
    
    # Show port usage
    echo ""
    echo -e "${PURPLE}🌐 Port Usage:${NC}"
    echo -e "${PURPLE}==============${NC}"
    
    # Check port 8000 (API)
    local port_8000=$(lsof -ti:8000 2>/dev/null || true)
    if [ -n "$port_8000" ]; then
        echo -e "${YELLOW}⚠️  Port 8000 still in use by PID(s): $port_8000${NC}"
    else
        echo -e "${GREEN}✅ Port 8000 (API) is free${NC}"
    fi
    
    # Check port 6379 (Redis)
    local port_6379=$(lsof -ti:6379 2>/dev/null || true)
    if [ -n "$port_6379" ]; then
        echo -e "${YELLOW}⚠️  Port 6379 still in use by PID(s): $port_6379${NC}"
    else
        echo -e "${GREEN}✅ Port 6379 (Redis) is free${NC}"
    fi
    
    # Show log information
    echo ""
    echo -e "${PURPLE}📁 Log Files Status:${NC}"
    echo -e "${PURPLE}===================${NC}"
    
    if [ -f "$LOG_DIR/api_server.log" ]; then
        local api_log_size=$(du -h "$LOG_DIR/api_server.log" | cut -f1)
        local api_log_modified=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LOG_DIR/api_server.log" 2>/dev/null || echo "unknown")
        echo -e "${GREEN}API Server Log:${NC} $LOG_DIR/api_server.log (${api_log_size}, modified: $api_log_modified)"
    else
        echo -e "${YELLOW}API Server Log:${NC} Not found"
    fi
    
    if [ -f "$LOG_DIR/redis.log" ]; then
        local redis_log_size=$(du -h "$LOG_DIR/redis.log" | cut -f1)
        local redis_log_modified=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$LOG_DIR/redis.log" 2>/dev/null || echo "unknown")
        echo -e "${GREEN}Redis Log:${NC} $LOG_DIR/redis.log (${redis_log_size}, modified: $redis_log_modified)"
    else
        echo -e "${YELLOW}Redis Log:${NC} Not found"
    fi
    
    if [ -f "$LOG_DIR/gunicorn_access.log" ]; then
        local access_log_size=$(du -h "$LOG_DIR/gunicorn_access.log" | cut -f1)
        echo -e "${GREEN}Access Log:${NC} $LOG_DIR/gunicorn_access.log (${access_log_size})"
    else
        echo -e "${YELLOW}Access Log:${NC} Not found"
    fi
}

# Function to force stop all processes
force_stop() {
    echo -e "${RED}💥 Force stopping all processes...${NC}"
    echo ""
    stop_all true
}

# Function to clean up logs and PID files
cleanup() {
    echo -e "${BLUE}🧹 Cleaning up files...${NC}"
    
    # Remove PID files
    local pid_files_removed=0
    if [ -d "$PID_DIR" ]; then
        for pid_file in "$PID_DIR"/*.pid; do
            if [ -f "$pid_file" ]; then
                rm -f "$pid_file"
                echo "   Removed: $pid_file"
                pid_files_removed=$((pid_files_removed + 1))
            fi
        done
    fi
    
    echo -e "${GREEN}✅ Cleanup completed${NC}"
    echo "   PID files removed: $pid_files_removed"
    
    # Optionally remove log files (user must explicitly confirm)
    if [ "$1" = "--with-logs" ]; then
        echo ""
        echo -e "${YELLOW}🗑️  Removing log files...${NC}"
        rm -f "$LOG_DIR"/*.log 2>/dev/null || true
        echo -e "${GREEN}✅ Log files removed${NC}"
    else
        echo "   Log files preserved (use --with-logs to remove)"
    fi
}

# Function to show help
show_help() {
    echo -e "${BLUE}Usage: $0 [OPTION]${NC}"
    echo ""
    echo -e "${CYAN}Options:${NC}"
    echo "  (no option)           Stop all services gracefully"
    echo "  force, -f, --force    Force stop all services (SIGKILL)"
    echo "  cleanup, -c, --cleanup  Stop services and clean up PID files"
    echo "  status, -s, --status  Show current system status"
    echo "  help, -h, --help      Show this help message"
    echo ""
    echo -e "${CYAN}Examples:${NC}"
    echo "  $0                    # Graceful stop"
    echo "  $0 force              # Force stop everything"
    echo "  $0 cleanup            # Stop and cleanup PID files"
    echo "  $0 cleanup --with-logs # Stop, cleanup PID files and remove logs"
    echo "  $0 status             # Show detailed status"
    echo ""
    echo -e "${CYAN}Logs:${NC}"
    echo "  View API logs:        tail -f $LOG_DIR/api_server.log"
    echo "  View Redis logs:      tail -f $LOG_DIR/redis.log"
    echo "  View access logs:     tail -f $LOG_DIR/gunicorn_access.log"
    echo ""
}

# Parse command line arguments
case "${1:-}" in
    "force"|"-f"|"--force")
        force_stop
        ;;
    "cleanup"|"-c"|"--cleanup")
        stop_all false
        cleanup "$2"
        ;;
    "status"|"-s"|"--status")
        show_status
        exit 0
        ;;
    "help"|"-h"|"--help")
        show_help
        exit 0
        ;;
    "")
        # Default: graceful stop
        stop_all false
        ;;
    *)
        echo -e "${RED}❌ Unknown option: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

# Show final status
show_status

echo ""
echo -e "${GREEN}🎉 Polkadot AI Chatbot system stopped successfully!${NC}"
echo ""
echo -e "${YELLOW}💡 To restart the system:${NC} ./start_pa_ai.sh"
echo -e "${YELLOW}💡 To view logs:${NC} tail -f $LOG_DIR/api_server.log"
echo "" 