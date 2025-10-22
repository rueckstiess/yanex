#!/bin/bash

# Setup script for yanex web UI

set -e

echo "Setting up yanex web UI..."

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed."
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed."
    echo "Please install npm (usually comes with Node.js)"
    exit 1
fi

echo "Node.js version: $(node --version)"
echo "npm version: $(npm --version)"

# Install dependencies
echo "Installing dependencies..."
npm install

echo "Setup complete!"
echo ""
echo "To build the web UI:"
echo "  npm run build"
echo ""
echo "To start the development server:"
echo "  npm run dev"
echo ""
echo "To build and copy to static directory:"
echo "  ./build.sh"


