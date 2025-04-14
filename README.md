# Baseball Pitch Impact on Betting Odds Analysis

This project analyzes how different types of baseball pitches and pitch sequences affect live betting odds. It combines Statcast pitch data with live betting odds to identify patterns and relationships between pitch events and odds movements.

## Features

- Fetches and analyzes Statcast pitch-by-pitch data
- Collects live betting odds from The Odds API
- Analyzes pitch sequences and their impact on game state
- Visualizes relationships between pitch events and odds movements
- Provides comprehensive statistical analysis of pitch impact

## Requirements

- Python 3.9+
- Required Python packages (see requirements.txt)
- The Odds API key (for live odds analysis)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/baseball-odds-analysis.git
cd baseball-odds-analysis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your The Odds API key:
```
ODDS_API_KEY=your_api_key_here
```

## Usage

Run the main analysis script:
```bash
python main.py
```

The script will:
1. Fetch recent pitch data from Statcast
2. Analyze pitch impact and sequences
3. Collect live odds data (if API key is provided)
4. Generate visualizations in the `data` directory

## Project Structure

- `main.py`: Main analysis script
- `data/`: Directory for output files and visualizations
- `requirements.txt`: Python package dependencies
- `.env`: Configuration file for API keys

## Output

The analysis generates several visualizations and data files:
- Pitch type distribution
- Pitch count impact
- Pitch sequence analysis
- Odds movement analysis (if API key is provided)
- Raw data files in CSV format

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 