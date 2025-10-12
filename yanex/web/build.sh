#!/bin/bash

# Build script for yanex web UI

set -e

echo "Building yanex web UI..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed. Please install Node.js to build the web UI."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed. Please install npm to build the web UI."
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Build the Next.js app
echo "Building Next.js app..."
npm run build

# Create static directory if it doesn't exist
mkdir -p static

# Copy built files to static directory
echo "Copying built files to static directory..."
cp -r .next/static/* static/ 2>/dev/null || true
cp -r .next/standalone/* static/ 2>/dev/null || true
cp -r public/* static/ 2>/dev/null || true

# Copy the main HTML file
if [ -f ".next/server/app/page.html" ]; then
    cp .next/server/app/page.html static/index.html
elif [ -f ".next/server/pages/index.html" ]; then
    cp .next/server/pages/index.html static/index.html
fi

echo "Build complete! Static files are in the static/ directory."
