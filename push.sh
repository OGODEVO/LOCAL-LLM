#!/bin/bash

# Read GitHub repository URL from .env file
GITHUB_REPO_URL=$(grep GITHUB_REPO_URL .env | cut -d '=' -f2)

if [ -z "$GITHUB_REPO_URL" ]; then
  echo "Error: GITHUB_REPO_URL not found in .env file."
  exit 1
fi

git add .
git commit -m "Automated push from local RAG chatbot."
git push $GITHUB_REPO_URL main
