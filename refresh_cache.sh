#!/bin/bash
#
# Auto-refresh FX cache from Yahoo Finance
# Run this hourly or as needed to keep data fresh
#

cd /Users/denielnankov/Documents/Verdad_Technical_Case_Study
source venv_fx/bin/activate
python populate_fx_cache.py >> cache_refresh.log 2>&1

echo "Cache refreshed at $(date)" >> cache_refresh.log
