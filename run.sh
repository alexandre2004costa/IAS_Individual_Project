#!/bin/bash
# =============================================================================
# Interactive Evaluation Pipeline
# =============================================================================
# 
# Runs evaluation step-by-step with user confirmation
# =============================================================================

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}======================================================================"
echo "Interactive Evaluation Pipeline"
echo "======================================================================${NC}"
echo ""


# Step 1: Generate datasets
echo -e "${YELLOW}Step 1: Generate BiasOnDemand datasets${NC}"
echo "This will create ~40+ datasets"
python generate.py
echo -e "${GREEN}✓ Datasets generated${NC}"

# Step 2: Fairness evaluation
echo ""
echo -e "${YELLOW}Step 2: Evaluate fairness metrics${NC}"
echo "This will compute 8 fairness metrics for all datasets"
python fairness_train.py
echo -e "${GREEN}✓ Fairness evaluation complete${NC}"

# Step 3: Explainability evaluation
echo ""
echo -e "${YELLOW}Step 3: Evaluate explainability (simplified DoX)${NC}"
echo "This will compute DoX-inspired metrics for all datasets"
python exokanation_train.py
echo -e "${GREEN}✓ Explainability evaluation complete${NC}"

# Step 3: Robustness evaluation
echo ""
echo -e "${YELLOW}Step 3: Evaluate Robustness (simplified DoX)${NC}"
echo "This will compute 3 metrics for all datasets"
python robustness_train.py
echo -e "${GREEN}✓ Robustness evaluation complete${NC}"


# Done
echo ""
echo -e "${GREEN}======================================================================"
echo "✓ PIPELINE COMPLETE!"
echo "======================================================================${NC}"
echo ""
echo "Results are in:"
echo "  • datasets/"
echo "  • results/"
