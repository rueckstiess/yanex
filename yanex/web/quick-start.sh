#!/bin/bash

# Quick start script for yanex web UI
echo "🚀 Yanex Web UI Quick Start"
echo "=========================="

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: Please run this script from the yanex/web directory"
    echo "   cd yanex/web && ./quick-start.sh"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed"
    echo "   Please install Node.js from https://nodejs.org/"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ Error: npm is not installed"
    echo "   Please install npm (comes with Node.js)"
    exit 1
fi

echo "✅ Node.js and npm are available"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ Error: Failed to install dependencies"
        exit 1
    fi
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Build the frontend
echo "🔨 Building frontend..."
npm run build
if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to build frontend"
    exit 1
fi
echo "✅ Frontend built successfully"

echo ""
echo "🎉 Setup complete! You can now run:"
echo "   yanex ui"
echo ""
echo "   Or with custom options:"
echo "   yanex ui --host 0.0.0.0 --port 3000"
echo "   yanex ui --reload  # for development"
echo ""


