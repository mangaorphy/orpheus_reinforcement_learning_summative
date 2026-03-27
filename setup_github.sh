#!/bin/bash
# =============================================================
#  GitHub Repository Setup — Savanna RL Summative
#  Run this once from inside your savanna_rl folder
# =============================================================

echo ""
echo "=================================================="
echo "  GITHUB REPO SETUP — savanna_rl_summative"
echo "=================================================="

# 1. Initialize git
git init
echo "  Git initialized"

# 2. Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.pyc
*.pyo
.Python
*.egg-info/
.eggs/

# Virtual environments
.venv/
venv/
env/

# Large model files (download separately)
models/**/*.zip
models/**/*.pt
*.zip

# Logs (can be regenerated)
logs/

# Results (keep JSON, ignore plots)
results/*.png

# Jupyter checkpoints
.ipynb_checkpoints/

# macOS
.DS_Store

# IDE
.vscode/
.idea/
EOF
echo "  .gitignore created"

# 3. Stage all files
git add .
git commit -m "Initial commit: Savanna Acoustic Threat Detection RL System

- Custom Gymnasium environment (20x20 savanna grid)
- Pygame renderer with acoustic heatmap, drone trail, mini-map
- DQN training script with 10 hyperparameter configs
- PPO, A2C, REINFORCE training scripts (10 configs each)
- Kaggle training notebook for GPU-accelerated training
- Memory-managed train_all.py pipeline
- main.py entry point for running best agent
- Full requirements.txt
"
echo "  Initial commit done"

echo ""
echo "=================================================="
echo "  NOW: Create repo on GitHub"
echo "=================================================="
echo ""
echo "  1. Go to: https://github.com/new"
echo "  2. Repository name: your_name_rl_summative"
echo "     (replace 'your_name' with your actual name)"
echo "  3. Set to PUBLIC"
echo "  4. Do NOT add README, .gitignore, or license"
echo "     (we already have these)"
echo "  5. Click 'Create repository'"
echo "  6. Copy the repo URL shown (ends in .git)"
echo ""
echo "  Then run these commands (replace URL with yours):"
echo ""
echo "    git remote add origin https://github.com/YOUR_NAME/your_name_rl_summative.git"
echo "    git branch -M main"
echo "    git push -u origin main"
echo ""
echo "  Done! Your code is on GitHub."
