#!/bin/bash
set -e

echo "Building Yanex Web UI..."

cd yanex/web

# Check for Node.js
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

# Check for npm
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed"
    echo "npm should come with Node.js installation"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Build the Next.js app (static export)
echo "Building Next.js static export..."
npm run build

# Verify build
if [ ! -d "out" ]; then
    echo "Error: Build failed - out directory not created"
    exit 1
fi

echo "✅ Web UI built successfully!"
echo "Build artifacts are in yanex/web/out/"
